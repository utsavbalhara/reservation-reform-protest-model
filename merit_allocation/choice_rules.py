"""The over-and-above choice rule for one institution with homogeneous seats.

This is the rule India uses to implement vertical reservations, as formalized by Sonmez and Yenmez (2022, Econometrica)
and characterized axiomatically by Aygun and Turhan (2023, arXiv:2305.11758):
  Stage 1. Open-category seats go to the highest-merit applicants, whatever their category.
  Stage 2. Among the remaining applicants, each reserve category's seats go to its highest-merit eligible members.
Reserve seats left unfilled can revert to the open category (de-reservation; Aygun and Turhan 2020, 2023).

Applicants are represented as merit-ordered "mass points": each carries a weight (number of candidates it stands for)
and, for each reserve category, the share of its weight eligible for that category. Fractional eligibility lets the
rule be applied to expected counts (for example, the expected share of SC candidates at a given merit level whose
family is below the income line) without simulation noise. Horizontal reservations (women, persons with disabilities)
and applicants' preferences over institutions and programmes are not modelled.
"""
import numpy as np


def reserved_first(weights, eligibility, open_seats, reserve_seats):
    """Reserve seats first, then open seats to the best remaining applicants. Not India's legal rule for a single
    institution, but in centralized multi-programme admission top reserved-category candidates often take reserved
    seats at better programmes, and the aggregate outcome can resemble this order (see merged_pool.calibrate_order)."""
    weights = np.asarray(weights, float)
    reserve_allocation, taken = {}, np.zeros_like(weights)
    for category, seats in reserve_seats.items():
        eligible = weights * np.asarray(eligibility[category], float)
        cumulative = np.cumsum(eligible)
        reserve_allocation[category] = np.clip(seats - (cumulative - eligible), 0, eligible)
        taken += reserve_allocation[category]
    remaining = weights - taken
    cumulative = np.cumsum(remaining)
    open_allocation = np.clip(open_seats - (cumulative - remaining), 0, remaining)
    return open_allocation, reserve_allocation


def over_and_above(weights, eligibility, open_seats, reserve_seats, dereserve_unfilled=True):
    """Allocate seats. weights: candidates per mass point, sorted from highest merit down.
    eligibility: dict reserve_category -> array of eligible shares per mass point (sum over categories <= 1).
    Returns (open_allocation, {category: reserve_allocation}) as arrays of candidates allocated per mass point."""
    weights = np.asarray(weights, float)
    cumulative = np.cumsum(weights)
    open_allocation = np.clip(open_seats - (cumulative - weights), 0, weights)
    remaining = weights - open_allocation
    reserve_allocation = {}
    unfilled = 0.0
    for category, seats in reserve_seats.items():
        eligible = remaining * np.asarray(eligibility[category], float)
        cumulative_eligible = np.cumsum(eligible)
        allocation = np.clip(seats - (cumulative_eligible - eligible), 0, eligible)
        reserve_allocation[category] = allocation
        unfilled += seats - allocation.sum()
    if dereserve_unfilled and unfilled > 1e-9:
        still_remaining = remaining - sum(reserve_allocation.values())
        cumulative_left = np.cumsum(still_remaining)
        extra = np.clip(unfilled - (cumulative_left - still_remaining), 0, still_remaining)
        open_allocation = open_allocation + extra
    return open_allocation, reserve_allocation
