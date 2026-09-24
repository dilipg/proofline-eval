import proofline_eval as pe
from util import EMB, day, index_from, query

# A uses entity M; B reports M. No citation between them, no shared words.
SPEC = [("A", 5, "quanibraion modrievment lattilency calibrated", None),
        ("B", 2, "thermoulaity caltroposis isostabity reached", None),
        ("N", 3, "gratilure retulaure noise", None)]
ENTS = {"x_m": ("Kavrel-7", "SamplingMethod"), "x_d": ("Orvane-QA", "Benchmark")}
MENTS = [("A", "x_m", "Kavrel-7", "uses"), ("B", "x_m", "Kavrel-7", "mentions"),
         ("B", "x_d", "Orvane-QA", "mentions")]


def _idx(facts=()):
    idx = index_from(SPEC)
    idx.onto = pe.OntologyIndex(idx, ENTS, MENTS, list(facts))
    return idx


def test_onto_walk_crosses_an_entity_bridge_the_text_cannot():
    sc = pe.OntoWalk()
    sc.prepare(_idx(), EMB)
    out = sc.run(query("quanibraion modrievment lattilency", 6), 3)
    assert out.ids[0] == "A" and "B" in out.ids
    assert "N" not in out.ids or out.ids.index("B") < out.ids.index("N")   # the bridge beats noise
    hy = pe.HybridRRF()
    hy.prepare(_idx(), EMB)
    assert "B" not in hy.run(query("quanibraion modrievment lattilency", 6), 1).ids


def test_ppr_respects_as_of_and_fact_validity():
    facts = [("B", "x_m", "evaluated_on", "x_d", None, day(2), day(4))]    # expired at day 4
    idx = _idx(facts)
    seed = {idx.pos["A"]: 1.0}
    cur = idx.onto.ppr(seed, day(6), "current", idx.snapshot_mask(day(6)))
    early = idx.onto.ppr(seed, day(4.5), "current", idx.snapshot_mask(day(4.5)))
    assert cur[idx.pos["B"]] > 0
    assert early[idx.pos["A"]] == 0 and early[idx.pos["B"]] == 0          # A not yet written


def test_circularity_is_checked_for_ontology_labels(monkeypatch):
    monkeypatch.setattr(pe, "ONTOLOGY_DERIVED_LABELS", {"onto_mined"})
    assert pe.check_circularity(pe.OntoWalk(), "onto_mined")
    assert pe.check_circularity(pe.OntoWalk(), "multihop") is None


FLOOD = SPEC[:2] + [(f"N{i:02d}", 3, f"quanibraion filler{i} padding{i}", None) for i in range(12)]


def test_onto_walk_spends_part_of_k_on_the_second_hop():
    idx = index_from(FLOOD)
    idx.onto = pe.OntologyIndex(idx, ENTS, MENTS, [])
    sc = pe.OntoWalk()
    sc.prepare(idx, EMB)
    assert "B" in sc.run(query("quanibraion modrievment lattilency", 6), 10).ids
