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
