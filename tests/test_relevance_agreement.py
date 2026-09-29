import numpy as np

from data_pipelines.relevance_agreement import cohen_kappa, krippendorff_alpha_nominal


def test_perfect_agreement_gives_one():
    labels = np.array([1, 0, 1, 1, 0, 0])
    assert krippendorff_alpha_nominal(labels, labels) == 1.0
    assert cohen_kappa(labels, labels) == 1.0


def test_alpha_matches_hand_computation():
    # Coincidence matrix for 8 items with one disagreement: n = 16, o_00 = 6, o_11 = 8, o_01 = o_10 = 1;
    # marginals 7 and 9; alpha = 1 - (n - 1) * 2 / (n^2 - 7^2 - 9^2) = 1 - 30 / 126.
    a = np.array([1, 1, 0, 0, 1, 0, 1, 1])
    b = np.array([1, 0, 0, 0, 1, 0, 1, 1])
    assert abs(krippendorff_alpha_nominal(a, b) - (1 - 30 / 126)) < 1e-12


def test_missing_items_are_skipped():
    a = np.array([1, np.nan, 0, 1])
    b = np.array([1, 0, 0, np.nan])
    assert krippendorff_alpha_nominal(a, b) == 1.0
