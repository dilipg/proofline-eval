from datetime import datetime, timedelta, timezone

import pytest

import ingest_semantica as ing
import proofline_eval as pe

T0 = datetime(2025, 1, 1, tzinfo=timezone.utc)


def test_entity_key_folds_spelling_and_class_tails():
    assert ing.entity_key("Kavrel-7") == ing.entity_key("Kavrel 7") == "kavrel7"
    assert ing.entity_key("Orvane-QA benchmark") == ing.entity_key("Orvane-QA")
    assert ing.entity_key("KA-7") != ing.entity_key("Kavrel-7")


@pytest.mark.parametrize("s,v", [("0.74 vex-score", "0.74"), (".7", "0.70"), ("0.74", "0.74"),
                                 ("Kavrel-7", None), ("7", None), ("this task", None)])
def test_parse_value(s, v):
    assert ing.parse_value(s) == v


def test_snake_and_onto_kind():
    assert ing.snake("evaluatedOn") == "evaluated_on" and ing.snake("used for") == "used_for"
    assert ing.onto_kind("Samplingmethod") == "SamplingMethod" and ing.onto_kind("THING") == "THING"


@pytest.mark.parametrize("key,role", [
    ("intro_method", "introduces"), ("intro_dataset", "introduces"), ("intro_material", "introduces"),
    ("uses", "uses"), ("studies", "studies"), ("reports", "mentions"), ("extends", "mentions"),
    ("evaluated_on", "mentions"), ("measured_by", "mentions")])
def test_role_cues_match_every_planted_phrasing(key, role):
    for tpl in pe._SAY[key]:
        s = tpl.format(e="Kavrel-7", k="sampling", d="Orvane-QA", x="vex-score", v="0.74", f="Dorvin-2")
        assert ing.infer_role(s) == role, s


def test_normalize_card_builds_reports_from_values():
    x = ing.CardExtraction("c1", [("Kavrel-7", "SamplingMethod"), ("Orvane-QA", "Benchmark"),
                                  ("vex-score", "Metric")],
                           [("Kavrel-7", "evaluated_on", "Orvane-QA benchmark"),
                            ("Kavrel-7", "measured_by", "0.74 vex-score"),
                            ("Kavrel-7", "used_for", "this task")])
    text = "we introduce Kavrel-7 , a sampling method . Kavrel-7 reaches 0.74 vex-score on Orvane-QA ."
    ments, facts = ing.normalize_card(x, text)
    assert ("kavrel7", "Kavrel-7", "SamplingMethod", "introduces") in ments
    assert ("orvaneqa", "Orvane-QA", "Benchmark", "mentions") in ments
    assert ("kavrel7", "evaluated_on", "orvaneqa", None) in facts
    assert ("kavrel7", "reports", "orvaneqa", "0.74") in facts
    assert not any(f[1] == "used_for" for f in facts)       # object is not a known entity


def test_normalize_card_drops_a_value_with_two_datasets():
    x = ing.CardExtraction("c1", [("Kavrel-7", "SamplingMethod"), ("Orvane-QA", "Benchmark"),
                                  ("Dorane-QA", "Benchmark")],
                           [("Kavrel-7", "evaluated_on", "Orvane-QA"), ("Kavrel-7", "evaluated_on", "Dorane-QA"),
                            ("Kavrel-7", "measured_by", "0.74")])
    _m, facts = ing.normalize_card(x, "Kavrel-7 Orvane-QA Dorane-QA .")
    assert not any(f[1] == "reports" for f in facts)


def test_resolve_merges_groups_within_one_class_only():
    per_card = {"c1": [("kavrel7", "Kavrel-7", "SamplingMethod", "introduces"),
                       ("orvaneqa", "Orvane-QA", "Benchmark", "mentions")],
                "c2": [("ka7", "KA-7", "Method", "uses")]}
    calls = []

    def groups(cands):
        ids = sorted(c["id"] for c in cands)
        calls.append(ids)
        return [["kavrel7", "ka7"]] if {"kavrel7", "ka7"} <= set(ids) else []

    ents, key2id = ing.resolve_entities(per_card, groups)
    assert key2id["kavrel7"] == key2id["ka7"] == "x_kavrel7" != key2id["orvaneqa"]
    assert calls == [["ka7", "kavrel7"]]               # a one-candidate class is never asked
    assert {i for i, _n, _k in ents} == {"x_kavrel7", "x_orvaneqa"}


def test_resolve_offers_only_same_number_names_to_the_detector():
    # Measured on Semantica 0.7.0: distinct names score HIGHER than true aliases
    # (Brakar-Corpus~Brakar-Text 0.80, Dorula-78~Dorith-85 0.78, alias Kavrel-7~KA-7 0.70),
    # so a detector that would merge everything must only ever see same-number names.
    per_card = {"c1": [("dorula78", "Dorula-78", "Method", "uses"), ("dorith85", "Dorith-85", "Method", "uses"),
                       ("do78", "DO-78", "Method", "uses"), ("brakarcorpus", "Brakar-Corpus", "Corpus", "uses"),
                       ("brakartext", "Brakar-Text", "Corpus", "uses")]}
    offered = []

    def merge_everything(cands):
        offered.append(sorted(c["id"] for c in cands))
        return [[c["id"] for c in cands]]

    _e, key2id = ing.resolve_entities(per_card, merge_everything)
    assert offered == [["do78", "dorula78"]]
    assert key2id["dorula78"] == key2id["do78"]
    assert len({key2id["dorith85"], key2id["dorula78"], key2id["brakarcorpus"], key2id["brakartext"]}) == 4


def test_validity_windows_follow_supersession():
    rows = [dict(id="a", committed_at=T0, supersedes_id=None),
            dict(id="b", committed_at=T0 + timedelta(days=4), supersedes_id="a")]
    w = ing.validity_windows(rows)
    assert w == {"a": (T0, T0 + timedelta(days=4)), "b": (T0 + timedelta(days=4), None)}


def test_assemble_attaches_windows():
    w = {"c1": (T0, None)}
    results = {"c1": ([("kavrel7", "Kavrel-7", "SamplingMethod", "uses")],
                      [("kavrel7", "extends", "kavrel7", None)])}
    ments, facts = ing.assemble(results, {"kavrel7": "x_kavrel7"}, w)
    assert ments == [("c1", "x_kavrel7", "Kavrel-7", "uses")]
    assert facts == [("c1", "x_kavrel7", "extends", "x_kavrel7", None, T0, None)]


def test_pick_cards_priority_limit_and_determinism():
    rows = [dict(id=f"c{i}") for i in range(10)]
    a = ing.pick_cards(rows, 4, {"c7", "c9"}, seed=1)
    assert [r["id"] for r in a[:2]] == ["c7", "c9"] and len(a) == 4
    assert a == ing.pick_cards(rows, 4, {"c7", "c9"}, seed=1)
    assert ing.pick_cards(rows, 0, set(), seed=1) == rows


def test_error_gate():
    ing.error_gate(5, 100)
    with pytest.raises(SystemExit):
        ing.error_gate(6, 100)


def test_cache_roundtrip(tmp_path):
    c = ing.ExtractCache(tmp_path / "x.sqlite")
    k = c.key("semantica:regex", "some text")
    assert c.get(k) is None
    c.put(k, {"e": [["Kavrel-7", "Method"]], "t": []})
    assert ing.ExtractCache(tmp_path / "x.sqlite").get(k) == {"e": [["Kavrel-7", "Method"]], "t": []}
    assert k != c.key("semantica:llm:anthropic:claude-opus-5", "some text")


def test_cache_key_changes_with_the_adapter_version(monkeypatch):
    # a fix to the adapter must never be answered from output cached before the fix
    before = ing.ExtractCache.key("semantica:regex", "t")
    monkeypatch.setattr(ing, "ADAPTER_VERSION", ing.ADAPTER_VERSION + "-next")
    assert ing.ExtractCache.key("semantica:regex", "t") != before


def test_write_back_replaces_same_source_and_keeps_others(seeded):
    with seeded.conn.cursor() as cur:
        cur.execute("SELECT id, committed_at FROM cards ORDER BY id LIMIT 1")
        r = cur.fetchone()
    args = ([r["id"]], [("x_a", "A-1", "SamplingMethod")], [(r["id"], "x_a", "A-1", "uses")],
            [(r["id"], "x_a", "extends", "x_a", None, r["committed_at"], None)], [("Samplingmethod", None)])

    def count(src):
        with seeded.conn.cursor() as cur:
            return [cur.execute(f"SELECT count(*) n FROM {t} WHERE source = %s", (src,)).fetchone()["n"]
                    for t in ("entities", "mentions", "facts", "onto_classes", "extracted_cards")]
    try:
        ing.write_back(seeded, "test:a", *args)
        ing.write_back(seeded, "test:a", *args)
        ing.write_back(seeded, "test:b", *args)
        assert count("test:a") == [1, 1, 1, 1, 1] and count("test:b") == [1, 1, 1, 1, 1]
    finally:
        with seeded.conn.cursor() as cur:
            for t in ("entities", "mentions", "facts", "onto_classes", "extracted_cards"):
                cur.execute(f"DELETE FROM {t} WHERE source LIKE 'test:%%'")
