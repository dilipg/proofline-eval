from collections import Counter, defaultdict, deque

import pytest

import proofline_eval as pe


@pytest.fixture(scope="module")
def quick():
    cfg = pe.Config(profile="quick")
    return pe.build_corpus(cfg, cfg.p["cards"], cfg.p["prooflines"])


def reach(corpus, a, t, hops=pe.DAG_HOPS):
    """Independent 0-1 BFS: undirected parent/citation edges cost one hop, supersession
    links none (a version is the same paper); both ends must exist at t."""
    by = {c.id: c for c in corpus.cards}
    adj = defaultdict(set)
    for c in corpus.cards:
        for d in c.parent_ids + c.citation_ids:
            if d in by and by[d].committed_at <= c.committed_at:
                adj[c.id].add((d, 1)); adj[d].add((c.id, 1))
        if c.supersedes_id:
            adj[c.id].add((c.supersedes_id, 0)); adj[c.supersedes_id].add((c.id, 0))
    best, dq = {a: 0}, deque([a])
    while dq:
        u = dq.popleft()
        for w, cost in adj[u]:
            d = best[u] + cost
            if d <= hops and by[w].committed_at <= t and d < best.get(w, hops + 1):
                best[w] = d
                dq.appendleft(w) if cost == 0 else dq.append(w)
    return best


def mh(corpus):
    return [q for q in corpus.queries.get("multihop", []) if q.answerable]


def test_every_question_has_exactly_one_stratum(quick):
    for q in mh(quick):
        assert sum(s in q.slices for s in ("dag_reachable", "entity_only")) == 1, q.id
        assert "linked" not in q.slices and "unlinked" not in q.slices


def test_strata_are_measured_not_planted(quick):
    for q in mh(quick)[::7]:
        a = next(iter(q.slots[0]))
        seen = reach(quick, a, q.as_of)
        ok = all(any(c in seen for c in s) for s in q.slots[1:])
        assert ("dag_reachable" in q.slices) == ok, q.id


def test_the_dag_is_not_starved(quick):
    qs = mh(quick)
    share = sum("dag_reachable" in q.slices for q in qs) / len(qs)
    assert 0.35 <= share <= 0.65, share
    per = Counter((q.family, "dag_reachable" in q.slices) for q in qs)
    for f in pe.MH_FAMILIES:
        assert per[(f, True)] >= 20 and per[(f, False)] >= 20, (f, per[(f, True)], per[(f, False)])


def test_multi_card_slot_is_reachable_through_any_card(quick):
    qs = [q for q in mh(quick) if q.family == "aggregate_set" and "dag_reachable" in q.slices
          and any(len(s) > 1 for s in q.slots[1:])]
    assert qs, "no multi-card aggregate slot in the DAG stratum"
    q = qs[0]
    seen = reach(quick, next(iter(q.slots[0])), q.as_of)
    assert all(any(c in seen for c in s) for s in q.slots[1:])


def test_the_sql_walk_agrees_with_the_stratum(seeded, smoke_corpus):
    qs = [q for q in mh(smoke_corpus) if "dag_reachable" in q.slices][:10]
    assert qs
    for q in qs:
        a = next(iter(q.slots[0]))
        depth, _l = pe.dag_neighbourhood(seeded.conn, [a], q.as_of)
        assert all(any(c in depth for c in s) for s in q.slots[1:]), q.id
