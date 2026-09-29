"""At what ratio of symbolic to material grievance weight would the lever ranking change?

The ratio w_s / w_m is not identified by the turnout target. This sweep multiplies it by k over a wide grid, holding the
product w_s x w_m fixed (w_s scales by sqrt(k), w_m by 1 / sqrt(k)), and recalibrates the mean threshold at every point
so the baseline always hits the same turnout target. At each point it reports every lever's paired effect and the
symbolic-only validation share. Two things follow: the break-even ratios at which material levers overtake symbolic
ones, and which ratios the 2018-type validation check (symbolic-only turnout between 10 and 30 lakh, with the baseline
at target) would admit.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from protest_simulation import SCENARIO_BY_KEY
from protest_simulation.calibration import CALIBRATION_TARGET_MIDPOINT_LAKH, calibrate_mean_threshold
from protest_simulation.monte_carlo import CAMPAIGN_SEED_OFFSET, paired_effects_from_runs, paired_worlds
from protest_simulation.parallel import run_campaigns
from protest_simulation.specifications import get_specification

RESULTS_FOLDER = Path(__file__).resolve().parent.parent / "results"
RATIO_MULTIPLIERS = (0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)
COMPARED = ("symbolic_only_validation", "grandfathering", "seat_expansion", "hybrid_caste_subquotas", "sub_classification",
            "consensus_commission", "compensation", "credible_guarantees", "managed_transition")
MATERIAL_LEVERS = ("grandfathering", "seat_expansion", "compensation")
SYMBOLIC_LEVERS = ("hybrid_caste_subquotas", "consensus_commission", "credible_guarantees")
VALIDATION_TARGET_SYMBOLIC_ONLY_LAKH = (10.0, 30.0)


def run_scenario(population, specification, base, scenario, run_count):
    worlds = paired_worlds(SCENARIO_BY_KEY[scenario].interventions if scenario != "baseline" else (), run_count,
                           base_parameters=base, world_sampler=specification.world_sampler)
    outcomes = run_campaigns(population, [(world, CAMPAIGN_SEED_OFFSET + index) for index, world in enumerate(worlds)])
    return {"peak_day_protesters": [o.peak_day_protesters for o in outcomes],
            "cumulative_unique_protesters": [o.cumulative_unique_protesters for o in outcomes],
            "total_deaths": [o.total_deaths for o in outcomes]}


def main():
    parser = argparse.ArgumentParser(description="Sweep the symbolic-to-material weight ratio with recalibration.")
    parser.add_argument("--specification", default="stylized")
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--calibration-runs", type=int, default=20)
    parser.add_argument("--agents", type=int, default=120_000)
    arguments = parser.parse_args()

    specification = get_specification(arguments.specification)
    population = specification.build_population(arguments.agents)
    reference = specification.base_parameters
    reference_ratio = reference.symbolic_threat_weight / reference.material_loss_weight
    points = []
    for multiplier in RATIO_MULTIPLIERS:
        base = reference.copy()
        base.symbolic_threat_weight = reference.symbolic_threat_weight * np.sqrt(multiplier)
        base.material_loss_weight = reference.material_loss_weight / np.sqrt(multiplier)
        threshold, calibrated_peak, _ = calibrate_mean_threshold(population, base, run_count=arguments.calibration_runs,
                                                                 world_sampler=specification.world_sampler)
        base.mean_participation_threshold = threshold
        baseline = run_scenario(population, specification, base, "baseline", arguments.runs)
        point = {"ratio_multiplier": multiplier, "symbolic_to_material_weight_ratio": round(reference_ratio * multiplier, 3),
                 "symbolic_threat_weight": round(base.symbolic_threat_weight, 4), "material_loss_weight": round(base.material_loss_weight, 4),
                 "calibrated_mean_threshold": round(threshold, 4), "baseline_median_peak_lakh": round(float(np.median(baseline["peak_day_protesters"])) / 1e5, 1),
                 "effects": {}}
        for scenario in COMPARED:
            runs = run_scenario(population, specification, base, scenario, arguments.runs)
            effects = paired_effects_from_runs(runs, baseline)
            point["effects"][scenario] = {"peak_change_percent": effects["peak"]["change_percent"], "ci95": effects["peak"]["change_percent_ci95"],
                                          "median_peak_lakh": round(float(np.median(runs["peak_day_protesters"])) / 1e5, 1)}
        levers = [key for key in COMPARED if key not in ("symbolic_only_validation", "managed_transition")]
        order = sorted(levers, key=lambda key: point["effects"][key]["peak_change_percent"])
        point["ranking_strongest_first"] = order
        point["strongest_material_lever"] = min(MATERIAL_LEVERS, key=lambda key: point["effects"][key]["peak_change_percent"])
        point["strongest_symbolic_lever"] = min(SYMBOLIC_LEVERS, key=lambda key: point["effects"][key]["peak_change_percent"])
        point["material_lever_is_strongest"] = order[0] in MATERIAL_LEVERS
        symbolic_only = point["effects"]["symbolic_only_validation"]["median_peak_lakh"]
        point["symbolic_only_share_of_baseline"] = round(symbolic_only / point["baseline_median_peak_lakh"], 3)
        point["passes_symbolic_only_validation"] = VALIDATION_TARGET_SYMBOLIC_ONLY_LAKH[0] <= symbolic_only <= VALIDATION_TARGET_SYMBOLIC_ONLY_LAKH[1]
        points.append(point)
        print(multiplier, threshold, point["baseline_median_peak_lakh"], symbolic_only, order[:3], point["passes_symbolic_only_validation"], flush=True)

    output = {"specification": arguments.specification, "run_count": arguments.runs, "calibration_target_lakh": CALIBRATION_TARGET_MIDPOINT_LAKH,
              "reference_ratio": reference_ratio, "validation_target_symbolic_only_lakh": VALIDATION_TARGET_SYMBOLIC_ONLY_LAKH, "points": points}
    admitted = [point["symbolic_to_material_weight_ratio"] for point in points if point["passes_symbolic_only_validation"]]
    output["ratios_admitted_by_validation"] = [min(admitted), max(admitted)] if admitted else None
    flips = [point["symbolic_to_material_weight_ratio"] for point in points if point["material_lever_is_strongest"]]
    output["ratios_where_a_material_lever_is_strongest"] = flips
    RESULTS_FOLDER.mkdir(exist_ok=True)
    path = RESULTS_FOLDER / f"material_symbolic_break_even_{arguments.specification}.json"
    path.write_text(json.dumps(output, indent=1))
    print(f"admitted by validation: {output['ratios_admitted_by_validation']}; material lever strongest at: {flips}")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
