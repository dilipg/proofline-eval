import numpy as np

import proofline_eval as pe
from util import EMB, day, index_from, query

SPEC = [("a", 0, "quanibraion modrievment lattilency", None),
        ("b", 10, "quanibraion modrievment lattilency revised", "a"),
        ("z", 1, "caltroposis thermoulaity unrelated", None)]


def test_current_at_is_as_of_not_end_of_record():
    idx = index_from(SPEC)
    assert idx.current_at(day(5)).tolist() == [True, True, True]      # b does not exist yet
    assert idx.current_at(day(12)).tolist() == [False, True, True]
    assert idx.current_mask.tolist() == [False, True, True]           # end of record, unchanged


def test_fresh_keeps_the_version_that_was_current_at_as_of():
    sc = pe.HybridFresh()
    sc.prepare(index_from(SPEC), EMB)
    assert "a" in sc.run(query("quanibraion modrievment", 5), 5).ids
    assert "a" not in sc.run(query("quanibraion modrievment", 12), 5).ids


class _FakeCE:
    def predict(self, pairs, batch_size=64, show_progress_bar=False):
        return np.zeros(len(pairs))


def test_rerank_candidates_are_as_of():
    sc = pe.CrossEncoderRerank()
    pe.Scorer.prepare(sc, index_from(SPEC), EMB)      # skip the model download
    sc.ce = _FakeCE()
    assert "a" in sc.run(query("quanibraion modrievment", 5), 5).ids
