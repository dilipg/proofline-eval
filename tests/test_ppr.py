import numpy as np

import proofline_eval as pe


def test_ppr_walk_keeps_mass_near_the_seed_and_off_unreachable_nodes():
    # 0 - 1 - 2, and 3 alone
    src = np.array([0, 1, 1, 2]); dst = np.array([1, 0, 2, 1])
    p = pe.ppr_walk(4, src, dst, np.array([1.0, 0, 0, 0]))
    assert p[0] > p[1] > p[2] > 0 and p[3] == 0
    assert abs(p.sum() - 1.0) < 1e-9


def test_ppr_walk_with_no_seed_mass_is_all_zero():
    assert not pe.ppr_walk(3, np.array([0]), np.array([1]), np.zeros(3)).any()


def test_ppr_walk_with_no_edges_leaves_the_mass_on_the_seeds():
    p = pe.ppr_walk(3, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.array([2.0, 0, 2.0]))
    assert p[1] == 0 and p[0] > 0 and p[2] > 0
