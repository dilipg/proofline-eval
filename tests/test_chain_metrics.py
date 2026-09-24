import math

import proofline_eval as pe
from util import day

SLOTS = [{"a"}, {"b1", "b2"}, {"c"}]
TS = {"a": 3.0, "b1": 1.0, "b2": 1.5, "c": 2.0, "x": 0.0}


def test_slot_and_chain_recall():
    m = pe.slot_metrics(["a", "x", "b2"], SLOTS, 10, "bridge3")
    assert m["slot_recall"] == 2 / 3 and m["chain_recall"] == 0.0 and m["last_hop_recall"] == 0.0
    m = pe.slot_metrics(["c", "b1", "a"], SLOTS, 10, "bridge3")
    assert m["chain_recall"] == 1.0 and m["last_hop_recall"] == 1.0
    assert math.isnan(pe.slot_metrics(["a"], SLOTS, 10, "timeline")["last_hop_recall"])
    assert pe.slot_metrics(["c", "b1", "a"], SLOTS, 2, "bridge3")["chain_recall"] == 0.0


def test_chain_order_is_time_order():
    assert pe.chain_order(["b1", "c", "a"], SLOTS, TS) == (1.0, 1.0)      # 1.0 < 2.0 < 3.0
    tau, exact = pe.chain_order(["a", "c", "b1"], SLOTS, TS)
    assert tau == -1.0 and exact == 0.0
    tau, exact = pe.chain_order(["b1", "a"], SLOTS, TS)                   # c missing
    assert tau == 1.0 and exact == 0.0


def test_score_runs_uses_the_chain_not_the_ranking():
    q = pe.Query(id="q", text="t", provenance="multihop", as_of=day(9), source_card="a",
                 rel={c: 2.0 for c in ("a", "b1", "b2", "c")}, slices=[], answerable=True,
                 family="bridge3", slots=SLOTS)
    r = pe.RunResult("s", "multihop", 0, {"q": ["a", "c", "b1"]}, {"q": 1.0},
                     chains={"q": ["b1", "c", "a"]})
    pe.score_runs([r], [q], 10, ts=TS)
    pq = r.per_query["q"]
    assert pq["chain_recall"] == 1.0 and pq["order_tau"] == 1.0 and pq["order_exact"] == 1.0


def test_history_queries_are_exempt_from_the_supersession_check():
    rows = {"v1": {"supersedes_id": None, "committed_at": day(1)},
            "v2": {"supersedes_id": "v1", "committed_at": day(2)}}
    mk = lambda intent: pe.Query(id="q", text="t", provenance="multihop", as_of=day(5),
                                 source_card=None, rel={}, slices=[], answerable=True,
                                 intent=intent)
    run = pe.RunResult("s", "multihop", 0, {"q": ["v1", "v2"]}, {"q": 1.0})
    assert pe.hard_checks(run, rows, [mk("current")], 10)["superseded_above_successor"]["fails"] == 1
    assert pe.hard_checks(run, rows, [mk("history")], 10)["superseded_above_successor"]["fails"] == 0
