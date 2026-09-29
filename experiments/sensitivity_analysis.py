import argparse
import json
from pathlib import Path

import numpy as np

from protest_simulation import BASELINE_ABRUPT_INCOME_ONLY_SWITCH, SCENARIO_BY_KEY, build_synthetic_india, run_paired_monte_carlo
from protest_simulation.monte_carlo import POPULATION_SEED, build_shared_population

RESULTS_FOLDER = Path(__file__).resolve().parent.parent / "results"
RELATIVE_PERTURBATION = 0.25
MEAN_THRESHOLD_PERTURBATION = 0.25
PERTURBED_PARAMETERS = (
    "loss_aversion",
    "material_loss_weight",
    "symbolic_threat_weight",
    "opposition_party_amplifier",
    "max_neighbourhood_influence",
    "max_national_visibility_influence",
    "participation_threshold_spread",
    "decision_noise",
    "fatigue_per_protest_day",
    "deaths_per_crore_protester_days",
    "symbolic_threat_rise_per_death",
    "mean_participation_threshold",
)
RANKED_SCENARIOS = ("grandfathering", "seat_expansion", "hybrid_caste_subquotas", "sub_classification",
                    "consensus_commission", "compensation", "credible_guarantees", "managed_transition", "hybrid_package")
CONVERGENCE_AGENT_COUNTS = (30_000, 60_000, 120_000, 240_000)


def perturbed(parameter_name: str, direction: int):
    parameters = BASELINE_ABRUPT_INCOME_ONLY_SWITCH.copy()
    if parameter_name == "mean_participation_threshold":
        parameters.mean_participation_threshold += direction * MEAN_THRESHOLD_PERTURBATION
    else:
        setattr(parameters, parameter_name, getattr(parameters, parameter_name) * (1 + direction * RELATIVE_PERTURBATION))
    return parameters


def spearman_correlation(first, second):
    first_ranks = np.argsort(np.argsort(first))
    second_ranks = np.argsort(np.argsort(second))
    return float(np.corrcoef(first_ranks, second_ranks)[0, 1])


def median_change_by_scenario(population, base_parameters, baseline_median, run_count):
    changes = {}
    for key in RANKED_SCENARIOS:
        runs = run_paired_monte_carlo(population, SCENARIO_BY_KEY[key].interventions, run_count, base_parameters=base_parameters)
        changes[key] = round((float(np.median(runs.peak_day_protesters)) / baseline_median - 1) * 100, 1)
    return changes


def sensitivity_and_rank_stability(population, baseline_run_count, ranking_run_count):
    reference_baseline = float(np.median(run_paired_monte_carlo(population, (), ranking_run_count).peak_day_protesters))
    reference_changes = median_change_by_scenario(population, BASELINE_ABRUPT_INCOME_ONLY_SWITCH, reference_baseline, ranking_run_count)
    reference_order = [reference_changes[key] for key in RANKED_SCENARIOS]
    print("reference", reference_changes, flush=True)
    rows = []
    for parameter_name in PERTURBED_PARAMETERS:
        for direction in (-1, 1):
            parameters = perturbed(parameter_name, direction)
            baseline_runs = run_paired_monte_carlo(population, (), baseline_run_count, base_parameters=parameters)
            baseline_median = float(np.median(baseline_runs.peak_day_protesters))
            ranking_baseline = float(np.median(baseline_runs.peak_day_protesters[:ranking_run_count]))
            changes = median_change_by_scenario(population, parameters, ranking_baseline, ranking_run_count)
            single_levers = {key: changes[key] for key in RANKED_SCENARIOS[:7]}
            rows.append({
                "parameter": parameter_name,
                "direction": "lower" if direction < 0 else "higher",
                "baseline_median_peak_lakh": round(baseline_median / 1e5, 1),
                "percent_change_by_scenario": changes,
                "rank_correlation_with_reference": round(spearman_correlation(reference_order, [changes[key] for key in RANKED_SCENARIOS]), 3),
                "strongest_single_lever": min(single_levers, key=single_levers.get),
                "weakest_single_lever": max(single_levers, key=single_levers.get),
            })
            print(rows[-1]["parameter"], rows[-1]["direction"], rows[-1]["baseline_median_peak_lakh"],
                  rows[-1]["rank_correlation_with_reference"], rows[-1]["strongest_single_lever"], rows[-1]["weakest_single_lever"], flush=True)
    return {"reference_baseline_median_peak_lakh": round(reference_baseline / 1e5, 1), "reference_percent_change": reference_changes, "perturbations": rows}


def agent_count_convergence(run_count):
    rows = []
    for agent_count in CONVERGENCE_AGENT_COUNTS:
        population = build_synthetic_india(agent_count, np.random.default_rng(POPULATION_SEED))
        baseline = run_paired_monte_carlo(population, (), run_count).summary()
        hybrid = run_paired_monte_carlo(population, SCENARIO_BY_KEY["hybrid_caste_subquotas"].interventions, run_count).summary()
        rows.append({"agent_count": agent_count, "baseline": baseline, "hybrid_caste_subquotas": hybrid})
        print("convergence", agent_count, baseline["median_peak_lakh"], hybrid["median_peak_lakh"], flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description="One-at-a-time sensitivity, ranking stability and agent-count convergence.")
    parser.add_argument("--baseline-runs", type=int, default=20)
    parser.add_argument("--ranking-runs", type=int, default=10)
    parser.add_argument("--convergence-runs", type=int, default=20)
    arguments = parser.parse_args()

    population = build_shared_population()
    sensitivity = {
        "relative_perturbation": RELATIVE_PERTURBATION,
        "mean_threshold_perturbation": MEAN_THRESHOLD_PERTURBATION,
        **sensitivity_and_rank_stability(population, arguments.baseline_runs, arguments.ranking_runs),
        "agent_count_convergence": agent_count_convergence(arguments.convergence_runs),
    }
    RESULTS_FOLDER.mkdir(exist_ok=True)
    output_path = RESULTS_FOLDER / "sensitivity_analysis.json"
    output_path.write_text(json.dumps(sensitivity, indent=1))
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
