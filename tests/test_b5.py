import pytest

import proofline_eval as pe


class _NoTrace:
    enabled = False

    def span(self, *a, **k):
        from contextlib import nullcontext
        return nullcontext(None)

    def event(self, *a, **k):
        pass

    def score(self, *a, **k):
        pass


@pytest.fixture(scope="module")
def ingested(seeded):
    emb = pe.HashEmbedder(256)
    with seeded.conn.cursor() as cur:
        cur.execute("SELECT id, txt FROM cards ORDER BY id")
        rows = cur.fetchall()
    seeded.set_embeddings([r["id"] for r in rows], emb.encode([r["txt"] for r in rows]))
    pe._INDEX_CACHE.clear()
    pe.filter_disconnected(seeded, pe.Config(profile="smoke"), emb)
    return seeded, emb


def test_b5_without_extraction_skips_ontology_scorers(ingested):
    store, emb = ingested
    cfg = pe.Config(profile="smoke", extractor="none:missing")
    out = pe.branch5_multihop(store, cfg, emb, _NoTrace())
    assert out["oracle"]["chain_recall"] == 1.0 and out["oracle"]["order_tau"] == 1.0
    assert set(out["skipped"]) == {"onto_walk", "onto_llm_walk"}
    assert {"hybrid_rrf_fresh", "two_step", "dag_walk"} <= set(out["scorers"])
    assert out["comparisons"]


def test_b5_walks_an_extraction(ingested):
    store, emb = ingested
    with store.conn.cursor() as cur:        # a perfect extraction, copied from the plant
        cur.execute("DELETE FROM entities WHERE source='test:truth'")
        cur.execute("INSERT INTO entities SELECT id, name, kind, 'test:truth' FROM true_entities")
        cur.execute("INSERT INTO mentions SELECT card_id, entity_id, surface, role, 'test:truth' "
                    "FROM true_mentions")
        cur.execute("INSERT INTO extracted_cards SELECT id, 'test:truth' FROM cards")
    try:
        pe._INDEX_CACHE.clear()
        out = pe.branch5_multihop(store, pe.Config(profile="smoke", extractor="test:truth"), emb, _NoTrace())
        assert not out["skipped"] and "onto_walk" in out["scorers"]
        assert out["scorers"]["onto_walk"]["n"] > 0
        # the capped walker sample spans families (sorted ids would give one family only)
        small = pe.branch5_multihop(store, pe.Config(profile="smoke", extractor="test:truth",
                                                     llm_queries=14), emb, _NoTrace())
        assert len(small["walker_sample"]) >= 5, small["walker_sample"]
    finally:
        with store.conn.cursor() as cur:
            for t in ("entities", "mentions", "extracted_cards"):
                cur.execute(f"DELETE FROM {t} WHERE source='test:truth'")
        pe._INDEX_CACHE.clear()


def _truth_extraction(store, fraction=1.0):
    with store.conn.cursor() as cur:
        for t in ("entities", "mentions", "extracted_cards"):
            cur.execute(f"DELETE FROM {t} WHERE source='test:truth'")
        cur.execute("INSERT INTO entities SELECT id, name, kind, 'test:truth' FROM true_entities")
        cur.execute("INSERT INTO mentions SELECT card_id, entity_id, surface, role, 'test:truth' FROM true_mentions")
        cur.execute("INSERT INTO extracted_cards SELECT id, 'test:truth' FROM cards ORDER BY id "
                    "LIMIT (SELECT ceil(count(*) * %s) FROM cards)", (fraction,))
    pe._INDEX_CACHE.clear()


def _clear_truth_extraction(store):
    with store.conn.cursor() as cur:
        for t in ("entities", "mentions", "extracted_cards"):
            cur.execute(f"DELETE FROM {t} WHERE source='test:truth'")
    pe._INDEX_CACHE.clear()


def test_b5_reports_extraction_coverage(ingested):
    store, emb = ingested
    try:
        _truth_extraction(store, 1.0)
        full = pe.branch5_multihop(store, pe.Config(profile="smoke", extractor="test:truth"), emb, _NoTrace())
        assert full["extraction"]["coverage"] == 1.0 and not full["extraction"]["partial"]
        _truth_extraction(store, 0.5)
        half = pe.branch5_multihop(store, pe.Config(profile="smoke", extractor="test:truth"), emb, _NoTrace())
        assert half["extraction"]["partial"] and half["extraction"]["coverage"] < 0.6
    finally:
        _clear_truth_extraction(store)


def test_b5_refuses_circular_scorers(ingested, monkeypatch):
    store, emb = ingested
    monkeypatch.setattr(pe, "ONTOLOGY_DERIVED_LABELS", {"multihop"})
    try:
        _truth_extraction(store, 1.0)
        out = pe.branch5_multihop(store, pe.Config(profile="smoke", extractor="test:truth"), emb, _NoTrace())
        assert out["skipped"]["onto_walk"].startswith("REFUSED")
        assert "onto_walk" not in out["scorers"]
    finally:
        _clear_truth_extraction(store)


def test_b5_does_not_report_a_walker_that_failed(ingested, monkeypatch, tmp_path):
    from types import SimpleNamespace
    store, emb = ingested

    class _Down:
        def create(self, **kw):
            raise RuntimeError("no credentials")

    monkeypatch.setattr(pe, "make_walker", lambda spec: pe.AnthropicWalker(
        "m", client=SimpleNamespace(messages=_Down()), cache_path=tmp_path / "llm.sqlite"))
    try:
        _truth_extraction(store, 1.0)
        out = pe.branch5_multihop(store, pe.Config(profile="smoke", extractor="test:truth",
                                                   walker="anthropic:m", llm_queries=6), emb, _NoTrace())
        assert out["skipped"]["onto_llm_walk"].startswith("NOT MEASURED")
        assert "onto_llm_walk" not in out["scorers"]
        assert not any("onto_llm_walk" in (c["a"], c["b"]) for c in out["comparisons"])
    finally:
        _clear_truth_extraction(store)


def test_b5_explains_why_every_scorer_answers_the_twins(ingested, capsys):
    store, emb = ingested
    out = pe.branch5_multihop(store, pe.Config(profile="smoke", extractor="none:missing"), emb, _NoTrace())
    if any(s["hard_checks"]["answered_a_no_answer_query"]["fails"] for s in out["scorers"].values()):
        assert "second-hop signal" in capsys.readouterr().out
