import proofline_eval as pe
from util import EMB, day, index_from, query

# A (the question's subject) cites B; B holds the answer but shares no words with A.
SPEC = [("A", 5, "quanibraion modrievment lattilency calibrated", None),
        ("B", 2, "thermoulaity caltroposis isostabity reached", None),
        ("B2", 8, "thermoulaity caltroposis isostabity reached again", "B"),
        ("N", 3, "gratilure retulaure noise", None)]


def _idx():
    idx = index_from(SPEC)
    idx.attach_graph([("A", "B", day(5))])
    return idx


def test_pool_is_intent_aware():
    idx = _idx()
    cur = pe.pool(idx, query("x", 9))
    hist = pe.pool(idx, query("x", 9, intent="history"))
    assert not cur[idx.pos["B"]] and hist[idx.pos["B"]] and cur[idx.pos["B2"]]


def test_dag_walk_reaches_the_cited_card():
    sc = pe.DagWalk()
    sc.prepare(_idx(), EMB)
    out = sc.run(query("quanibraion modrievment lattilency", 6), 5)
    assert out.ids[0] == "A" and "B" in out.ids
    assert out.chain == sorted(out.chain, key=lambda c: {"A": 5, "B": 2, "B2": 8, "N": 3}[c])
    assert pe.DagWalk.uses_graph


def test_dag_walk_follows_citations_backwards_too():
    # from the cited card B, the walk must reach its citer A through cited_by
    idx = _idx()
    assert idx.cited_by["B"] == [("A", day(5).timestamp())]
    sc = pe.DagWalk()
    sc.prepare(idx, EMB)
    out = sc.run(query("thermoulaity caltroposis isostabity", 6), 5)
    assert out.ids[0] == "B" and "A" in out.ids


def test_two_step_uses_feedback_terms_without_the_graph():
    sc = pe.TwoStep()
    sc.prepare(_idx(), EMB)
    out = sc.run(query("quanibraion modrievment", 6), 5)
    assert out.ids and out.ids[0] == "A" and not pe.TwoStep.uses_graph
