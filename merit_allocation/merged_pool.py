"""Seats by eligibility segment under today's reservations and under a merged income-only pool, for IIT admissions.

Candidates: the fitted latent merit distributions (latent_merit.py), one mass point per candidate.
Status quo: the over-and-above rule with open 40.5%, GEN-EWS 10%, OBC-NCL 27%, SC 15%, ST 7.5% of seats.
Reform: the over-and-above rule with the open share unchanged and one reserve category, family income below the line,
holding the rest of the seats (59.5% in the main case; 49.5% and 69.5% as variants).

Family income is not observed per candidate, so each category's share above the line is drawn from a prior, with an
optional gradient (richer families are more common higher up the merit distribution). Draws are kept only if the
implied share of today's IIT entrants above the line lies in the range the JIC reports support (see INCOME_PRIORS).
"""
from dataclasses import dataclass

import numpy as np
from scipy.special import expit, logit, ndtri

from .choice_rules import over_and_above, reserved_first
from .jee_advanced_data import CATEGORIES, REPORT_TABLES
from .latent_merit import LatentMeritFit

STATUS_QUO_SHARES = {"open": 0.405, "GEN-EWS": 0.10, "OBC-NCL": 0.27, "SC": 0.15, "ST": 0.075}
POOL_SHARES = (0.595, 0.495, 0.695)
SEGMENTS = ("GEN_above", "GEN_below", "GEN-EWS", "OBC-NCL_above", "OBC-NCL_below", "SC_above", "SC_below", "ST_above", "ST_below")

# Priors on the share of each category's JEE (Advanced) candidates from families above the Rs 8 lakh line (gross income,
# all sources). GEN-EWS is below by definition. GEN includes OBC creamy-layer candidates. OBC-NCL can be above on gross
# income because the creamy-layer test excludes salary and farm income. The admitted-share constraint comes from the JIC
# parental-income tables: 37.5% of 2016 admits with income reported had family income above Rs 5 lakh, and 26.5% of 2018
# admits had a father's income above Rs 8 lakh; allowing for income growth since then, 28-45% of today's entrants.
INCOME_PRIORS = {"GEN": (0.70, 0.95), "OBC-NCL": (0.10, 0.40), "SC": (0.10, 0.45), "ST": (0.10, 0.45)}
MERIT_GRADIENT_PRIOR = (0.0, 0.8)
ADMITTED_ABOVE_LINE_SHARE = (0.28, 0.45)


@dataclass
class IncomeOverlay:
    above_share: dict
    merit_gradient: float


def sample_overlay(random_generator) -> IncomeOverlay:
    shares = {category: float(random_generator.uniform(*bounds)) for category, bounds in INCOME_PRIORS.items()}
    shares["GEN-EWS"] = 0.0
    return IncomeOverlay(shares, float(random_generator.uniform(*MERIT_GRADIENT_PRIOR)))


def mass_points(fit: LatentMeritFit, overlay: IncomeOverlay):
    """One point per candidate, sorted by merit. Returns z, category index, probability of being above the line."""
    z_parts, category_parts, above_parts = [], [], []
    for index, category in enumerate(CATEGORIES):
        count = int(round(fit.candidates[index]))
        quantiles = ndtri((np.arange(count) + 0.5) / count)
        z = fit.mean[index] + fit.sd[index] * quantiles
        share = overlay.above_share[category]
        if share <= 0:
            above = np.zeros(count)
        else:
            # Choose the intercept so the category mean equals the target share exactly.
            low, high = -20.0, 20.0
            for _ in range(60):
                middle = 0.5 * (low + high)
                if expit(middle + overlay.merit_gradient * quantiles).mean() > share:
                    high = middle
                else:
                    low = middle
            above = expit(0.5 * (low + high) + overlay.merit_gradient * quantiles)
        z_parts.append(z)
        category_parts.append(np.full(count, index))
        above_parts.append(above)
    z = np.concatenate(z_parts)
    order = np.argsort(-z, kind="stable")
    return z[order], np.concatenate(category_parts)[order], np.concatenate(above_parts)[order]


def _segment_split(category, above, proportional, below_only=None):
    """Seats by segment. 'proportional' seats at a point are split between above- and below-line candidates in
    proportion to their shares; 'below_only' seats went to below-line candidates only."""
    below_only = np.zeros_like(proportional) if below_only is None else below_only
    totals = {}
    for index, name in enumerate(CATEGORIES):
        members = category == index
        if name == "GEN-EWS":
            totals["GEN-EWS"] = float((proportional + below_only)[members].sum())
        else:
            totals[f"{name}_above"] = float((proportional * above)[members].sum())
            totals[f"{name}_below"] = float((proportional * (1 - above) + below_only)[members].sum())
    return totals


def allocate(fit: LatentMeritFit, overlay: IncomeOverlay, seats: float, pool_share: float = 0.595, order: str = "open_first"):
    """order: 'open_first' is the over-and-above rule; 'reserved_first' mimics the aggregate outcome of multi-programme
    matching in which top reserved-category candidates take reserved seats at better programmes."""
    rule = over_and_above if order == "open_first" else reserved_first
    z, category, above = mass_points(fit, overlay)
    weights = np.ones_like(z)
    membership = {name: (category == index).astype(float) for index, name in enumerate(CATEGORIES) if name != "GEN"}
    open_sq, reserve_sq = rule(weights, membership, STATUS_QUO_SHARES["open"] * seats, {name: STATUS_QUO_SHARES[name] * seats for name in membership})
    status_quo_points = open_sq + sum(reserve_sq.values())
    open_reform, reserve_reform = rule(weights, {"income": 1 - above}, (1 - pool_share) * seats, {"income": pool_share * seats})
    return {"status_quo": _segment_split(category, above, status_quo_points),
            "reform": _segment_split(category, above, open_reform, reserve_reform["income"]),
            "candidates": _segment_split(category, above, weights),
            "status_quo_by_category": [float(status_quo_points[category == index].sum()) for index in range(len(CATEGORIES))],
            "status_quo_open_by_category": [float(open_sq[category == index].sum()) for index in range(len(CATEGORIES))],
            "admitted_above_line_share": float((status_quo_points * above).sum() / status_quo_points.sum())}


EXPANSION_PRECEDENT = 1.25  # the 2019 EWS introduction added about 2.14 lakh seats, roughly a quarter of central intake


def allocate_regimes(fit: LatentMeritFit, overlay: IncomeOverlay, seats: float, order: str = "reserved_first") -> dict:
    """Seats by segment under today's rules and under three alternatives, all with the same candidates:
      merged_pool:            one income-based pool of 59.5% (the reform);
      merged_pool_expanded:   the same with total seats expanded by a quarter (lever L2, 2019 EWS precedent);
      caste_income_filter:    caste categories kept, but SC and ST seats open only to below-line families (lever L3);
                              OBC-NCL and EWS keep their current income rules."""
    rule = over_and_above if order == "open_first" else reserved_first
    z, category, above = mass_points(fit, overlay)
    weights = np.ones_like(z)
    membership = {name: (category == index).astype(float) for index, name in enumerate(CATEGORIES) if name != "GEN"}
    shares = {name: STATUS_QUO_SHARES[name] for name in membership}
    open_sq, reserve_sq = rule(weights, membership, STATUS_QUO_SHARES["open"] * seats, {name: shares[name] * seats for name in membership})
    status_quo = _segment_split(category, above, open_sq + sum(reserve_sq.values()))
    regimes = {"status_quo": status_quo}
    for name, total in (("merged_pool", seats), ("merged_pool_expanded", seats * EXPANSION_PRECEDENT)):
        open_part, reserve_part = rule(weights, {"income": 1 - above}, (1 - POOL_SHARES[0]) * total, {"income": POOL_SHARES[0] * total})
        regimes[name] = _segment_split(category, above, open_part, reserve_part["income"])
    filtered = dict(membership)
    below_only = {}
    for name in ("SC", "ST"):
        filtered[name] = membership[name] * (1 - above)
    open_part, reserve_part = rule(weights, filtered, STATUS_QUO_SHARES["open"] * seats, {name: shares[name] * seats for name in filtered})
    proportional = open_part + reserve_part["GEN-EWS"] + reserve_part["OBC-NCL"]
    below_only = reserve_part["SC"] + reserve_part["ST"]
    regimes["caste_income_filter"] = _segment_split(category, above, proportional, below_only)
    regimes["candidates"] = _segment_split(category, above, weights)
    return regimes


def seats_for_year(year: int) -> float:
    allotted = REPORT_TABLES[year]["allotted"]
    return float(sum(allotted)) if allotted is not None else 17760.0
