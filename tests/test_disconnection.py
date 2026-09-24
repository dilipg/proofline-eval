from collections import defaultdict

import numpy as np
import pytest

import proofline_eval as pe


@pytest.fixture(scope="module")
def filtered(seeded):
    emb = pe.HashEmbedder(256)
    with seeded.conn.cursor() as cur:
        cur.execute("SELECT id, txt FROM cards ORDER BY id")
        rows = cur.fetchall()
    seeded.set_embeddings([r["id"] for r in rows], emb.encode([r["txt"] for r in rows]))
    pe._INDEX_CACHE.clear()
    counts = pe.filter_disconnected(seeded, pe.Config(profile="smoke"), emb)
    return seeded, emb, counts


def test_kept_queries_really_need_the_hop(filtered):
    store, emb, counts = filtered
    idx = pe.build_index(store, None)
    kept = [q for q in pe.load_queries(store, "multihop") if q.family in pe.LAST_HOP_FAMILIES]
    assert sum(c["kept"] for f, c in counts.items() if f in pe.LAST_HOP_FAMILIES) == len(kept)
    for q in kept:
        m = idx.snapshot_mask(q.as_of).copy()
        for cid in q.slots[0]:
            m[idx.pos[cid]] = False
        lex = np.where(m, idx.bm25(q.text), -np.inf)
        den = np.where(m, idx.cosine(emb.encode_queries([q.text])[0]), -np.inf)
        top = {idx.ids[i] for i in pe.topn(lex, 10) if np.isfinite(lex[i]) and lex[i] > 0}
        top |= {idx.ids[i] for i in pe.topn(den, 10) if np.isfinite(den[i])}
        assert not top & q.slots[-1], q.id


def test_pairs_are_dropped_together(filtered):
    store, _emb, _counts = filtered
    groups = defaultdict(set)
    for q in pe.load_queries(store, "multihop", include_dropped=True):
        if q.pair_id:
            groups[q.pair_id].add(q.id)
    kept = {q.id for q in pe.load_queries(store, "multihop")}
    for members in groups.values():
        assert members <= kept or not (members & kept)


def test_counts_are_reported_per_family(filtered):
    _s, _e, counts = filtered
    assert set(pe.LAST_HOP_FAMILIES) <= set(counts)
    assert "pairs_dropped" in counts
