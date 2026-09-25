from types import SimpleNamespace

import proofline_eval as pe
from util import EMB, day, flood, index_from, load_dag, query

# The question's card A uses method m. B reports on dataset d. A fact "m reports on d"
# only appears on F, written on day 9: after a question asked on day 6.
SPEC = flood([("A", 5, "quanibraion modrievment lattilency calibrated", None),
              ("B", 2, "thermoulaity caltroposis isostabity reached", None),
              ("F", 9, "futurilent postdatious", None)])
ENTS = {"x_m": ("Kavrel-7", "SamplingMethod"), "x_d": ("Orvane-QA", "Benchmark")}
MENTS = [("A", "x_m", "Kavrel-7", "uses"), ("B", "x_d", "Orvane-QA", "mentions"),
         ("F", "x_m", "Kavrel-7", "mentions"), ("F", "x_d", "Orvane-QA", "mentions")]
FACTS = [("F", "x_m", "reports", "x_d", "0.74", day(9), None)]


def _onto(cls):
    idx = index_from(SPEC)
    idx.onto = pe.OntologyIndex(idx, ENTS, MENTS, FACTS)
    sc = cls()
    sc.prepare(idx, EMB)
    return sc


def test_blind_walks_traverse_the_future_but_never_return_it(dag_db):
    q = query("quanibraion modrievment lattilency", 6)
    aware, blind = _onto(pe.OntoWalk).run(q, 5), _onto(pe.OntoWalkBlind).run(q, 5)
    assert "B" not in aware.ids and "B" in blind.ids          # only the future fact bridges them
    assert "F" not in aware.ids and "F" not in blind.ids      # results stay as of day 6
    load_dag(dag_db, SPEC, [("F", "A", 9), ("F", "B", 9)])
    sc = pe.DagWalkBlind()
    sc.prepare(index_from(SPEC), EMB, SimpleNamespace(conn=dag_db))
    out = sc.run(q, 5)
    assert "B" in out.ids and "F" not in out.ids              # co-cited by a future paper


def test_ablations_are_registered_and_flagged():
    assert pe.SCORERS["dag_walk_blind"].time_blind and pe.SCORERS["onto_walk_blind"].time_blind
    assert not pe.SCORERS["dag_walk"].time_blind and not pe.SCORERS["onto_walk"].time_blind
    assert pe.SCORERS["dag_walk_blind"].uses_graph and pe.SCORERS["onto_walk_blind"].uses_ontology
