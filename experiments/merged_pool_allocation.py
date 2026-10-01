"""Workstream A: who gains and who loses IIT seats if caste categories are merged into one income-based pool.

For each validated JEE (Advanced) year, fit the latent merit model, then for many draws of the income priors compute
seats by segment under today's rules and under the merged pool. Keep only draws whose implied share of today's entrants
from above-line families matches the JIC parental-income evidence. Report seat changes per segment, per 1,000
candidates, and the material change per candidate relative to above-line SC candidates, which is the unit the protest
model uses (above-line SC = 1).

Main case: reserved-first aggregate processing, which reproduces the published allotments by category; the
over-and-above (open-first) rule is reported as a bound. Pool shares: 59.5% (all current reservations merged), 49.5%
and 69.5%.
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from merit_allocation.jee_advanced_data import CATEGORIES, REPORT_TABLES, VALIDATED_YEARS
from merit_allocation.latent_merit import LatentMeritFit, fit_year, validate_fit
from merit_allocation.merged_pool import ADMITTED_ABOVE_LINE_SHARE, POOL_SHARES, SEGMENTS, allocate, allocate_regimes, sample_overlay, seats_for_year

RESULTS_FOLDER = Path(__file__).resolve().parent.parent / "results"
ORDERS = ("reserved_first", "open_first")
REFERENCE_SEGMENT = "SC_above"


def evaluate_draw(arguments):
    fit_parameters, seats, draw_seed = arguments
    fit = LatentMeritFit(**fit_parameters)
    random_generator = np.random.default_rng(draw_seed)
    for _ in range(200):
        overlay = sample_overlay(random_generator)
        check = allocate(fit, overlay, seats, POOL_SHARES[0], "reserved_first")
        if ADMITTED_ABOVE_LINE_SHARE[0] <= check["admitted_above_line_share"] <= ADMITTED_ABOVE_LINE_SHARE[1]:
            break
    else:
        return None
    outcome = {"overlay": {"above_share": overlay.above_share, "merit_gradient": overlay.merit_gradient},
               "admitted_above_line_share": check["admitted_above_line_share"], "cases": {}}
    for order in ORDERS:
        for pool_share in POOL_SHARES:
            result = check if (order == "reserved_first" and pool_share == POOL_SHARES[0]) else allocate(fit, overlay, seats, pool_share, order)
            outcome["cases"][f"{order}|{pool_share}"] = {key: result[key] for key in ("status_quo", "reform", "candidates")}
    # The two levers that are themselves allocation rules (L2 seat expansion, L3 caste quotas with an income filter).
    regimes = allocate_regimes(fit, overlay, seats, "reserved_first")
    for regime in ("merged_pool_expanded", "caste_income_filter", "abolition"):
        outcome["cases"][f"lever|{regime}"] = {"status_quo": regimes["status_quo"], "reform": regimes[regime], "candidates": regimes["candidates"]}
    return outcome


def per_candidate_change(case):
    return {segment: (case["reform"][segment] - case["status_quo"][segment]) / case["candidates"][segment] for segment in SEGMENTS}


def summarise(values):
    values = np.array(values, float)
    return {"median": round(float(np.median(values)), 4), "p05": round(float(np.percentile(values, 5)), 4), "p95": round(float(np.percentile(values, 95)), 4)}


def seats_by_category(status_quo):
    """Status-quo seats by reservation category (GEN = open-category seats won by GEN candidates of either income)."""
    return [status_quo["GEN_above"] + status_quo["GEN_below"], status_quo["GEN-EWS"],
            status_quo["OBC-NCL_above"] + status_quo["OBC-NCL_below"], status_quo["SC_above"] + status_quo["SC_below"],
            status_quo["ST_above"] + status_quo["ST_below"]]


def allotment_check(kept):
    """Model status-quo seats by category (median over draws) against the published allotments, for both orders."""
    check = {}
    for year in VALIDATED_YEARS:
        actual = REPORT_TABLES[year]["allotted"]
        if not actual:
            continue
        check[str(year)] = {"categories": list(CATEGORIES), "actual": actual}
        for order in ORDERS:
            model = np.median([seats_by_category(outcome["cases"][f"{order}|{POOL_SHARES[0]}"]["status_quo"])
                               for draw_year, outcome in kept if draw_year == year], axis=0)
            check[str(year)][order] = {"model": np.round(model).tolist(),
                                       "largest_absolute_gap": int(round(np.max(np.abs(model - np.array(actual)))))}
    return check


def main():
    parser = argparse.ArgumentParser(description="Merged-pool seat allocation for IIT admissions.")
    parser.add_argument("--draws", type=int, default=400)
    arguments = parser.parse_args()

    fits, validation = {}, {}
    for year in VALIDATED_YEARS:
        fit = fit_year(year)
        fits[year] = fit
        validation[year] = validate_fit(fit)
        print(year, "fitted", flush=True)
    jobs = [({"year": year, "candidates": fits[year].candidates, "mean": fits[year].mean, "sd": fits[year].sd,
              "log_likelihood": fits[year].log_likelihood, "crl_size": fits[year].crl_size, "imputed_counts": fits[year].imputed_counts},
             seats_for_year(year), 100_000 * year + draw) for year in VALIDATED_YEARS for draw in range(arguments.draws)]
    with ProcessPoolExecutor() as pool:
        outcomes = list(pool.map(evaluate_draw, jobs, chunksize=8))
    kept = [(job[0]["year"], outcome) for job, outcome in zip(jobs, outcomes) if outcome is not None]
    print(f"kept {len(kept)} of {len(jobs)} draws", flush=True)

    summary = {}
    case_keys = [f"{order}|{pool_share}" for order in ORDERS for pool_share in POOL_SHARES] + ["lever|merged_pool_expanded", "lever|caste_income_filter", "lever|abolition"]
    for key in case_keys:
        rows = {segment: {"status_quo": [], "reform": [], "percent_change": [], "change_per_1000_candidates": [], "relative_material_change": []}
                for segment in SEGMENTS}
        for year, outcome in kept:
            case = outcome["cases"][key]
            change = per_candidate_change(case)
            # The unit is the reform's loss for an above-line SC candidate in the same draw: the reform case's own for the
            # reform variants, the main reform case's for the levers (so lever tables share the reform table's unit).
            reference_case = case if not key.startswith("lever|") else outcome["cases"][f"reserved_first|{POOL_SHARES[0]}"]
            reference_loss = -per_candidate_change(reference_case)[REFERENCE_SEGMENT]
            for segment in SEGMENTS:
                status_quo, reform = case["status_quo"][segment], case["reform"][segment]
                rows[segment]["status_quo"].append(status_quo)
                rows[segment]["reform"].append(reform)
                rows[segment]["percent_change"].append((reform / status_quo - 1) * 100 if status_quo > 0 else np.nan)
                rows[segment]["change_per_1000_candidates"].append(change[segment] * 1000)
                # Positive = loss, in units of the above-line SC candidate's loss under the reform (the protest model's unit).
                rows[segment]["relative_material_change"].append(-change[segment] / reference_loss)
        summary[key] = {segment: {measure: summarise(values) for measure, values in measures.items()} for segment, measures in rows.items()}

    fitted = {str(year): {"categories": list(CATEGORIES), "candidates": np.round(fits[year].candidates).tolist(), "mean": np.round(fits[year].mean, 4).tolist(),
                          "sd": np.round(fits[year].sd, 4).tolist(), "log_likelihood": round(fits[year].log_likelihood, 2),
                          "crl_size": fits[year].crl_size, "imputed_counts": fits[year].imputed_counts} for year in VALIDATED_YEARS}
    output = {"years": list(VALIDATED_YEARS), "draws_per_year": arguments.draws, "draws_kept": len(kept),
              "admitted_above_line_share_constraint": ADMITTED_ABOVE_LINE_SHARE, "latent_fits": fitted, "validation": {str(k): v for k, v in validation.items()},
              "allotment_check": allotment_check(kept),
              "summary": summary,
              "draws": [{"year": year, "overlay": outcome["overlay"], "admitted_above_line_share": outcome["admitted_above_line_share"],
                         "main_case": outcome["cases"][f"reserved_first|{POOL_SHARES[0]}"]} for year, outcome in kept]}
    RESULTS_FOLDER.mkdir(exist_ok=True)
    path = RESULTS_FOLDER / "merged_pool_allocation.json"
    path.write_text(json.dumps(output, indent=1))
    main = summary[f"reserved_first|{POOL_SHARES[0]}"]
    for segment in SEGMENTS:
        print(f"{segment:>15}: change {main[segment]['percent_change']['median']:+7.1f}% [{main[segment]['percent_change']['p05']:+.0f}, {main[segment]['percent_change']['p95']:+.0f}]"
              f"  relative material change {main[segment]['relative_material_change']['median']:+.3f} [{main[segment]['relative_material_change']['p05']:+.2f}, {main[segment]['relative_material_change']['p95']:+.2f}]")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
