import argparse
import json
from pathlib import Path

import numpy as np

from protest_simulation import BASELINE_ABRUPT_INCOME_ONLY_SWITCH, build_shared_population, simulate_protest_campaign
from protest_simulation.policy_interventions import remove_all_material_loss

RESULTS_FOLDER = Path(__file__).resolve().parent.parent / "results"
CALIBRATION_TARGET_PEAK_LAKH = (20, 100)
VALIDATION_TARGET_SYMBOLIC_ONLY_LAKH = (10, 30)


def median_outcome(population, parameters, run_count):
    outcomes = [simulate_protest_campaign(population, parameters, np.random.default_rng(seed)) for seed in range(run_count)]
    return {
        "median_peak_lakh": round(float(np.median([o.peak_day_protesters for o in outcomes])) / 1e5, 1),
        "median_cumulative_crore": round(float(np.median([o.cumulative_unique_protesters for o in outcomes])) / 1e7, 2),
        "median_deaths": float(np.median([o.total_deaths for o in outcomes])),
    }


def sweep_mean_threshold(population, thresholds, run_count):
    response_curve = []
    for mean_threshold in thresholds:
        baseline = BASELINE_ABRUPT_INCOME_ONLY_SWITCH.copy()
        baseline.mean_participation_threshold = float(mean_threshold)
        symbolic_only = baseline.copy()
        remove_all_material_loss(symbolic_only)
        point = {
            "mean_participation_threshold": round(float(mean_threshold), 2),
            "baseline": median_outcome(population, baseline, run_count),
            "symbolic_only_validation": median_outcome(population, symbolic_only, run_count),
        }
        response_curve.append(point)
        print(point, flush=True)
    return response_curve


def main():
    parser = argparse.ArgumentParser(description="Trace turnout against mean threshold to calibrate the baseline.")
    parser.add_argument("--from-threshold", type=float, default=5.0)
    parser.add_argument("--to-threshold", type=float, default=8.5)
    parser.add_argument("--step", type=float, default=0.25)
    parser.add_argument("--runs", type=int, default=6)
    parser.add_argument("--agents", type=int, default=120_000)
    arguments = parser.parse_args()

    population = build_shared_population(arguments.agents)
    thresholds = np.arange(arguments.from_threshold, arguments.to_threshold + 1e-9, arguments.step)
    curve = sweep_mean_threshold(population, thresholds, arguments.runs)
    RESULTS_FOLDER.mkdir(exist_ok=True)
    output_path = RESULTS_FOLDER / "threshold_response_curve.json"
    output_path.write_text(json.dumps({
        "calibration_target_peak_lakh": CALIBRATION_TARGET_PEAK_LAKH,
        "validation_target_symbolic_only_lakh": VALIDATION_TARGET_SYMBOLIC_ONLY_LAKH,
        "response_curve": curve,
    }, indent=1))
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
