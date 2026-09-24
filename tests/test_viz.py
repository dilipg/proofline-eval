from datetime import datetime, timedelta, timezone

import ingest_semantica as ing

T0 = datetime(2025, 1, 1, tzinfo=timezone.utc)


def card(cid, day, sup=None, pl="pl"):
    return dict(id=cid, proofline_id=pl, title=f"title {cid}", committed_at=T0 + timedelta(days=day),
                supersedes_id=sup, is_current=True)


CARDS = {c["id"]: c for c in [card("a", 0), card("b", 5), card("c", 10, sup="a")]}
EDGES = [("b", "a", "citation", T0 + timedelta(days=5))]
ENTS = {"x_k": ("Kavrel-7", "SamplingMethod"), "x_o": ("Orvane-QA", "Benchmark")}
MENTS = [("a", "x_k", "Kavrel-7", "introduces"), ("b", "x_k", "Kavrel-7", "uses"),
         ("a", "x_o", "Orvane-QA", "mentions")]
FACTS = [("a", "x_k", "reports", "x_o", "0.74", T0, T0 + timedelta(days=10))]


def test_bfs_sample_respects_limit_and_allowed():
    adj = {"a": {"b"}, "b": {"a", "c"}, "c": {"b"}}
    assert ing.bfs_sample(adj, ["a"], 2, lambda n: True) == ["a", "b"]
    assert ing.bfs_sample(adj, ["a"], 10, lambda n: n != "c") == ["a", "b"]
    assert ing.bfs_sample(adj, ["z"], 10, lambda n: n != "z") == []


def test_dag_graph_is_snapshot_correct():
    g7 = ing.dag_graph(CARDS, EDGES, T0 + timedelta(days=7), None, 10)
    assert {n["id"] for n in g7["entities"]} == {"a", "b"}
    assert {n["id"]: n["type"] for n in g7["entities"]}["a"] == "current"
    g12 = ing.dag_graph(CARDS, EDGES, T0 + timedelta(days=12), "a", 10)
    assert {n["id"]: n["type"] for n in g12["entities"]}["a"] == "superseded"


def test_ontology_graph_drops_expired_facts():
    g9 = ing.ontology_graph(CARDS, ENTS, MENTS, FACTS, T0 + timedelta(days=9), "a", 10)
    assert any(r["type"] == "reports 0.74" for r in g9["relationships"])
    g11 = ing.ontology_graph(CARDS, ENTS, MENTS, FACTS, T0 + timedelta(days=11), "a", 10)
    assert not any(r["type"].startswith("reports") for r in g11["relationships"])


def test_seed_can_be_a_proofline():
    live = set(CARDS)
    assert ing.resolve_seed(CARDS, "pl", live, lambda c: 0) == "a"
    assert ing.resolve_seed(CARDS, "nope", live, lambda c: 0) is None


def test_graph_before_anything_existed_is_empty():
    before = T0 - timedelta(days=1)
    assert ing.dag_graph(CARDS, EDGES, before, None, 10)["entities"] == []
    assert ing.ontology_graph(CARDS, ENTS, MENTS, FACTS, before, None, 10)["entities"] == []


def test_parse_asofs_makes_utc():
    d = ing.parse_asofs("2025-06-01, 2026-01-01T12:00:00+00:00")
    assert [x.tzinfo is not None for x in d] == [True, True] and d[0].year == 2025
    assert ing.parse_asofs(None) == []
