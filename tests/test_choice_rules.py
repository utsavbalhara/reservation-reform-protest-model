import numpy as np

from merit_allocation.choice_rules import over_and_above


def test_open_seats_go_to_the_top_whatever_the_category():
    weights = np.ones(10)
    sc = np.array([1, 0, 0, 0, 0, 0, 0, 0, 1, 1], float)
    open_allocation, reserve = over_and_above(weights, {"SC": sc}, open_seats=3, reserve_seats={"SC": 2})
    assert open_allocation.tolist() == [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    # The SC candidate at the top took an open seat, so the two SC seats go to the next SC candidates.
    assert reserve["SC"].tolist() == [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]


def test_within_category_merit_is_respected_and_quotas_fill():
    weights = np.ones(8)
    eligibility = {"A": np.array([0, 1, 0, 1, 0, 1, 0, 1], float), "B": np.array([0, 0, 1, 0, 1, 0, 1, 0], float)}
    open_allocation, reserve = over_and_above(weights, eligibility, open_seats=1, reserve_seats={"A": 2, "B": 1})
    assert reserve["A"].tolist() == [0, 1, 0, 1, 0, 0, 0, 0]
    assert reserve["B"].tolist() == [0, 0, 1, 0, 0, 0, 0, 0]


def test_fractional_eligibility_gives_expected_allocations():
    weights = np.array([10.0, 10.0, 10.0])
    open_allocation, reserve = over_and_above(weights, {"R": np.array([0.5, 0.5, 0.5])}, open_seats=5, reserve_seats={"R": 10})
    assert open_allocation.tolist() == [5, 0, 0]
    # 2.5 eligible remain at the top point, then 5 at the next, then 2.5 of the third point's 5.
    assert np.allclose(reserve["R"], [2.5, 5.0, 2.5])


def test_unfilled_reserve_seats_revert_to_open():
    weights = np.ones(5)
    open_allocation, reserve = over_and_above(weights, {"R": np.array([0, 0, 0, 0, 1], float)}, open_seats=1, reserve_seats={"R": 3})
    assert reserve["R"].sum() == 1
    assert open_allocation.sum() == 3
    assert open_allocation.tolist() == [1, 1, 1, 0, 0]
