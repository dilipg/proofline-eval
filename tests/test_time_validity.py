from datetime import datetime, timedelta, timezone

import proofline_eval as pe

T0 = datetime(2025, 1, 1, tzinfo=timezone.utc)


def mk_card(cid, day, parents=(), cites=(), support=()):
    return pe.Card(id=cid, proofline_id="pl", topic=0, kc_type="finding", kc_stage="results",
                   title=cid, one_line=cid, body=cid, committed_at=T0 + timedelta(days=day),
                   version=1, supersedes_id=None, is_current=True, parent_ids=list(parents),
                   citation_ids=list(cites), true_support=list(support), terms=[],
                   inherited_terms=[], findability=0.0)


def mk_corpus(cards):
    return pe.Corpus(cards=cards, by_id={c.id: c for c in cards}, queries={},
                     held_out_prooflines=set(), topics={}, stats={})


def test_forward_edges_are_dropped_and_counted():
    a = mk_card("a", 0)
    b = mk_card("b", 5, parents=["a", "c"], support=["c"])
    c = mk_card("c", 9, cites=["a"])
    rows, dropped = pe.time_valid_edges(mk_corpus([a, b, c]))
    assert {(s, d, k) for s, d, k, _ in rows} == {("b", "a", "parent"), ("c", "a", "citation")}
    assert dropped == {"parent": 1, "true_support": 1}
    when = {"b": b.committed_at, "c": c.committed_at}
    assert all(vf == when[s] for s, _d, _k, vf in rows)


def test_same_instant_and_duplicates():
    a = mk_card("a", 3)
    b = mk_card("b", 3, parents=["a", "a"])
    rows, dropped = pe.time_valid_edges(mk_corpus([a, b]))
    assert [(s, d, k) for s, d, k, _ in rows] == [("b", "a", "parent")]
    assert dropped == {}


def test_version_cited_at_never_points_forward():
    chain = [(T0 + timedelta(days=10), "p1v1"), (T0 + timedelta(days=20), "p1v2")]
    assert pe.version_cited_at(chain, T0 + timedelta(days=5)) is None
    assert pe.version_cited_at(chain, T0 + timedelta(days=15)) == "p1v1"
    assert pe.version_cited_at(chain, T0 + timedelta(days=25)) == "p1v2"


def _index(days):
    rows = {cid: dict(id=cid, proofline_id="pl", txt=cid, embedding=None, is_current=True,
                      supersedes_id=None, committed_at=T0 + timedelta(days=d))
            for cid, d in days.items()}
    return pe.Index(rows)


def test_indeg_counts_only_citations_that_existed():
    idx = _index({"old": 0, "x": 3, "y": 8})
    idx.attach_graph([("x", "old", T0 + timedelta(days=3)), ("y", "old", T0 + timedelta(days=8))])
    assert idx.indeg_at("old", T0 + timedelta(days=1)) == 0
    assert idx.indeg_at("old", T0 + timedelta(days=3)) == 1
    assert idx.indeg_at("old", T0 + timedelta(days=30)) == 2
    assert idx.indeg_at("x", T0 + timedelta(days=30)) == 0


def test_indeg_ignores_citers_outside_the_corpus():
    idx = _index({"old": 0, "x": 3})
    idx.attach_graph([("x", "old", T0 + timedelta(days=3)),
                      ("ghost", "old", T0 + timedelta(days=4))])
    assert idx.indeg_at("old", T0 + timedelta(days=30)) == 1
    assert not hasattr(idx, "indeg")
