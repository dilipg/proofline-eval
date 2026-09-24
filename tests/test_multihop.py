import re
from collections import Counter, defaultdict

import pytest

import proofline_eval as pe


@pytest.fixture(scope="module")
def quick():
    cfg = pe.Config(profile="quick")
    return pe.build_corpus(cfg, cfg.p["cards"], cfg.p["prooflines"])


def mh(corpus):
    return corpus.queries.get("multihop", [])


def test_every_family_reaches_its_target(quick):
    fam = Counter(q.family for q in mh(quick) if q.answerable)
    for f in pe.MH_FAMILIES:
        assert fam[f] >= 150, (f, fam[f])       # before the disconnection filter
    assert sum(1 for q in mh(quick) if q.family == "twin") > 0


def test_questions_never_name_the_bridge(quick):
    suffixes = {"qa", "bench", "corpus", "text", "score", "oxide", "nitride", "alloy", "polymer"}
    toks = {t for _i, n, _k, _p in quick.entities for t in re.findall(r"[a-z]+", n.lower())
            if len(t) > 3 and t not in suffixes}
    for q in mh(quick):
        words = set(re.findall(r"[a-z]+", q.text.lower()))
        assert not words & toks, (q.id, words & toks)


def test_gold_is_valid_at_as_of_and_slots_are_disjoint(quick):
    by_id = {c.id: c for c in quick.cards}
    succ = {c.supersedes_id: c for c in quick.cards if c.supersedes_id}
    for q in mh(quick):
        flat = [c for s in q.slots for c in s]
        assert len(flat) == len(set(flat)), q.id
        assert set(q.rel) == set(flat)
        for cid in flat:
            c = by_id[cid]
            assert c.committed_at <= q.as_of, q.id
            if q.intent == "current":
                nxt = succ.get(cid)
                assert nxt is None or nxt.committed_at > q.as_of, q.id


def test_twins_exclude_every_report_of_their_method(quick):
    by_q = {q.id: q for q in mh(quick)}
    groups = defaultdict(list)
    for q in mh(quick):
        if q.pair_id:
            groups[q.pair_id].append(q)
    twins = [q for q in mh(quick) if q.family == "twin"]
    assert twins
    for t in twins:
        orig = next(x for x in groups[t.pair_id] if x.answerable)
        assert t.text == orig.text and t.as_of < orig.as_of
        gold = orig.slots[-1]
        assert all(pe_c.committed_at > t.as_of for pe_c in quick.cards if pe_c.id in gold)
        assert by_q[t.id].rel == {} and not t.answerable


def test_chrono_pairs_have_different_answers(quick):
    groups = defaultdict(list)
    for q in mh(quick):
        if q.family == "chrono_asof":
            groups[q.pair_id].append(q)
    assert groups
    for members in groups.values():
        assert len(members) == 2
        a, b = sorted(members, key=lambda q: q.as_of)
        assert a.text == b.text and a.answer["value"] != b.answer["value"]


def test_history_is_history_intent_with_ordered_versions(quick):
    by_id = {c.id: c for c in quick.cards}
    hs = [q for q in mh(quick) if q.family == "history"]
    assert hs and all(q.intent == "history" for q in hs)
    for q in hs:
        versions = [next(iter(s)) for s in q.slots[1:]]
        assert [by_id[v].version for v in versions] == sorted(by_id[v].version for v in versions)
        assert len(set(q.answer["values"])) > 1


def test_aggregates_are_multi_document(quick):
    for q in mh(quick):
        if q.family == "aggregate_set":
            assert len(q.answer["set"]) == len(q.slots) - 1 >= 2
        if q.family == "aggregate_count":
            assert q.answer["count"] == len(q.slots) - 1 >= 2


def test_entities_off_builds_no_multihop():
    cfg = pe.Config(profile="smoke", entities=False)
    c = pe.build_corpus(cfg, cfg.p["cards"], cfg.p["prooflines"])
    assert "multihop" not in c.queries
