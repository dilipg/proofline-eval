from types import SimpleNamespace

import pytest

import proofline_eval as pe
from util import EMB, day, flood, index_from, load_dag, query

# A (day 5) cites B (day 2); B cites C (day 1); F (day 9) cites A: a citation from the
# future relative to a question asked on day 6.
SPEC = [("A", 5, "quanibraion modrievment lattilency calibrated", None),
        ("B", 2, "thermoulaity caltroposis isostabity reached", None),
        ("C", 1, "gratilure retulaure ossiness", None),
        ("F", 9, "futurilent postdatious", None)]
LINKS = [("A", "B", 5), ("B", "C", 2), ("F", "A", 9)]


def test_dag_sql_reaches_within_hops_and_only_as_of(dag_db):
    load_dag(dag_db, SPEC, LINKS)
    depth, links = pe.dag_neighbourhood(dag_db, ["A"], day(6), hops=3)
    assert depth == {"A": 0, "B": 1, "C": 2}
    assert {frozenset(l) for l in links} == {frozenset(("A", "B")), frozenset(("B", "C"))}


def test_dag_sql_time_blind_uses_the_future_citation(dag_db):
    load_dag(dag_db, SPEC, LINKS)
    depth, _l = pe.dag_neighbourhood(dag_db, ["A"], day(6), hops=3, time_blind=True)
    assert depth["F"] == 1


def test_dag_sql_supersession_only_as_of(dag_db):
    spec = SPEC + [("B2", 8, "thermoulaity caltroposis isostabity revised", "B")]
    load_dag(dag_db, spec, LINKS)
    assert "B2" not in pe.dag_neighbourhood(dag_db, ["A"], day(6))[0]
    assert pe.dag_neighbourhood(dag_db, ["A"], day(8))[0]["B2"] == 1   # A - B, and B2 is B


def test_dag_sql_stops_at_the_hop_cap_on_a_cycle(dag_db):
    spec = [(c, i, c.lower(), None) for i, c in enumerate("PQRSTU", start=1)]
    chain = [("Q", "P", 2), ("R", "Q", 3), ("S", "R", 4), ("T", "S", 5), ("U", "T", 6), ("U", "P", 6)]
    load_dag(dag_db, spec, chain)
    depth, _l = pe.dag_neighbourhood(dag_db, ["P"], day(7), hops=2)
    assert depth == {"P": 0, "Q": 1, "U": 1, "R": 2, "T": 2}


def _walk(dag_db, spec, links, cls=pe.DagWalk):
    load_dag(dag_db, spec, links)
    sc = cls()
    sc.prepare(index_from(spec), EMB, SimpleNamespace(conn=dag_db))
    return sc


Q = "quanibraion modrievment lattilency"


def test_dag_walk_reaches_a_card_that_shares_no_words(dag_db):
    sc = _walk(dag_db, flood(SPEC), LINKS)
    out = sc.run(query(Q, 6), 5)
    assert out.ids[0] == "A" and "B" in out.ids and "F" not in out.ids
    control = pe.TwoStep()
    control.prepare(index_from(flood(SPEC)), EMB)
    assert "B" not in control.run(query(Q, 6), 5).ids        # the walk is why B is there
    assert pe.DagWalk.uses_graph


def test_dag_walk_reaches_a_co_citing_card_in_two_hops(dag_db):
    # A and G both cite X; G shares no word with the question
    spec = flood([("A", 5, "quanibraion modrievment lattilency", None), ("X", 1, "originatious", None),
                  ("G", 4, "thermoulaity caltroposis", None)])
    sc = _walk(dag_db, spec, [("A", "X", 5), ("G", "X", 4)])
    assert "G" in sc.run(query(Q, 6), 5).ids


def test_dag_walk_chain_is_traversal_order_not_date_order(dag_db):
    sc = _walk(dag_db, SPEC, LINKS)
    out = sc.run(query(Q, 6), 5)
    walked = [c for c in out.chain if c in {"A", "B", "C"}]
    assert len(walked) >= 2 and walked == ["A", "B", "C"][:len(walked)]   # depth order, not day order


def test_dag_walk_with_no_edges_is_the_first_stage(dag_db):
    sc = _walk(dag_db, flood(SPEC), [])
    out = sc.run(query(Q, 6), 5)
    fs = pe.first_stage(sc.idx, EMB, query(Q, 6))
    assert out.ids[0] == "A" and set(out.ids) <= {sc.idx.ids[i] for i in fs}


def test_dag_walk_without_a_store_says_so():
    sc = pe.DagWalk()
    sc.prepare(index_from(SPEC), EMB)
    with pytest.raises(SystemExit, match="Postgres"):
        sc.run(query("quanibraion", 6), 3)


def test_dag_sql_version_chain_costs_no_hop(dag_db):
    # P cites Q; Q has versions Q2, Q3: all of Q's versions sit one hop from P
    spec = [("P", 5, "p", None), ("Q", 1, "q", None), ("Q2", 2, "q2", "Q"), ("Q3", 3, "q3", "Q2")]
    load_dag(dag_db, spec, [("P", "Q", 5)])
    assert pe.dag_neighbourhood(dag_db, ["P"], day(6), hops=1)[0] == {"P": 0, "Q": 1, "Q2": 1, "Q3": 1}
