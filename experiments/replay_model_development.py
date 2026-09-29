import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from protest_simulation import BASELINE_ABRUPT_INCOME_ONLY_SWITCH, SCENARIO_BY_KEY, build_shared_population, run_paired_monte_carlo, simulate_protest_campaign
from protest_simulation.policy_interventions import remove_all_material_loss
from protest_simulation.protest_campaign import material_loss_felt_by_each_agent, symbolic_threat_felt_by_each_agent
from protest_simulation.synthetic_population import SOCIAL_GROUP_NAMES

RESULTS_FOLDER = Path(__file__).resolve().parent.parent / "results"

ITERATION_ONE_UNBOUNDED_CONTAGION = replace(
    BASELINE_ABRUPT_INCOME_ONLY_SWITCH,
    material_loss_weight=1.0,
    symbolic_threat_weight=2.2,
    participation_threshold_spread=0.55,
    decision_noise=0.22,
    social_influence_saturates=False,
    max_neighbourhood_influence=1.6,
    max_national_visibility_influence=25.0,
    mean_participation_threshold=3.05,
)
ITERATION_TWO_NARROW_THRESHOLDS = replace(
    ITERATION_ONE_UNBOUNDED_CONTAGION,
    social_influence_saturates=True,
    max_neighbourhood_influence=0.9,
    max_national_visibility_influence=0.6,
    mean_participation_threshold=5.1,
)
ITERATION_TWO_THRESHOLD_UNCERTAINTY = 0.05
ITERATION_THREE_REWEIGHTINGS = (
    {"symbolic_threat_weight": 3.6, "material_loss_weight": 0.5, "mean_participation_threshold": 5.3},
    {"symbolic_threat_weight": 5.0, "material_loss_weight": 0.4, "mean_participation_threshold": 6.4},
    {"symbolic_threat_weight": 6.0, "material_loss_weight": 0.3, "mean_participation_threshold": 7.1},
)
ITERATION_TWO_SCENARIOS = ("baseline", "symbolic_only_validation", "grandfathering", "hybrid_caste_subquotas",
                           "consensus_commission", "compensation", "managed_transition")


def median_peak_lakh(population, parameters, run_count):
    peaks = [simulate_protest_campaign(population, parameters, np.random.default_rng(seed)).peak_day_protesters for seed in range(run_count)]
    return round(float(np.median(peaks)) / 1e5, 1)


def symbolic_only(parameters):
    adjusted = parameters.copy()
    remove_all_material_loss(adjusted)
    return adjusted


def first_bandh_day_participation_without_contagion(population, parameters) -> dict:
    grievance = (
        parameters.material_loss_weight * material_loss_felt_by_each_agent(population, parameters)
        + parameters.symbolic_threat_weight * population.identity_strength * symbolic_threat_felt_by_each_agent(population, parameters)
    )
    threshold = parameters.mean_participation_threshold + parameters.participation_threshold_spread * population.threshold_standard_score
    push = (parameters.organizational_capacity_by_group * parameters.opposition_party_amplifier)[population.social_group]
    chance = 1 / (1 + np.exp(-(grievance + push * parameters.mobilization_on_bandh_days - threshold) / parameters.decision_noise))
    return {name: round(float(chance[population.social_group == index].mean()) * 100, 2) for index, name in enumerate(SOCIAL_GROUP_NAMES)}


def replay_iteration_one(population, run_count):
    sweep = []
    for mean_threshold in (2.8, 3.05, 3.3):
        parameters = replace(ITERATION_ONE_UNBOUNDED_CONTAGION, mean_participation_threshold=mean_threshold)
        sweep.append({"mean_participation_threshold": mean_threshold, "median_peak_lakh": median_peak_lakh(population, parameters, run_count)})
        print("iteration 1", sweep[-1], flush=True)
    return {
        "description": "Linear, unbounded contagion; narrow thresholds; material loss weighted as heavily as symbolic threat.",
        "threshold_sweep": sweep,
        "bandh_day_participation_percent_without_contagion": first_bandh_day_participation_without_contagion(population, ITERATION_ONE_UNBOUNDED_CONTAGION),
    }


def replay_iteration_two(population, run_count, curve_run_count):
    curve = []
    for mean_threshold in np.round(np.arange(4.6, 5.61, 0.1), 2):
        parameters = replace(ITERATION_TWO_NARROW_THRESHOLDS, mean_participation_threshold=float(mean_threshold))
        curve.append({"mean_participation_threshold": float(mean_threshold), "median_peak_lakh": median_peak_lakh(population, parameters, curve_run_count)})
        print("iteration 2 curve", curve[-1], flush=True)
    scenario_results = {}
    for key in ITERATION_TWO_SCENARIOS:
        runs = run_paired_monte_carlo(population, SCENARIO_BY_KEY[key].interventions, run_count,
                                      base_parameters=ITERATION_TWO_NARROW_THRESHOLDS,
                                      mean_threshold_standard_deviation=ITERATION_TWO_THRESHOLD_UNCERTAINTY)
        scenario_results[key] = runs.summary()
        print("iteration 2", key, scenario_results[key], flush=True)
    return {
        "description": "Saturating contagion, but narrow thresholds and heavy material weighting.",
        "threshold_response_curve": curve,
        "scenarios": scenario_results,
    }


def replay_iteration_three(population, run_count):
    attempts = []
    for overrides in ITERATION_THREE_REWEIGHTINGS:
        parameters = replace(ITERATION_TWO_NARROW_THRESHOLDS, **overrides)
        baseline_peak = median_peak_lakh(population, parameters, run_count)
        symbolic_peak = median_peak_lakh(population, symbolic_only(parameters), run_count)
        attempts.append({**overrides, "baseline_median_peak_lakh": baseline_peak, "symbolic_only_median_peak_lakh": symbolic_peak,
                         "symbolic_only_share_of_baseline": round(symbolic_peak / baseline_peak, 3) if baseline_peak else None})
        print("iteration 3", attempts[-1], flush=True)
    return {"description": "Symbolic threat reweighted upward with thresholds still narrow.", "attempts": attempts}


def summarize_final_iteration():
    central = json.loads((RESULTS_FOLDER / "intervention_comparison_central.json").read_text())["scenarios"]
    return {
        "description": "Wide threshold distribution (a long tail of low-threshold activists), final weights.",
        "bandh_day_participation_percent_without_contagion": None,
        "scenarios": {key: central[key]["summary"] for key in ITERATION_TWO_SCENARIOS},
    }


def main():
    parser = argparse.ArgumentParser(description="Replay each model iteration's failure with the current code.")
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--quick-runs", type=int, default=6)
    arguments = parser.parse_args()

    population = build_shared_population()
    final_iteration = summarize_final_iteration()
    final_iteration["bandh_day_participation_percent_without_contagion"] = first_bandh_day_participation_without_contagion(
        population, BASELINE_ABRUPT_INCOME_ONLY_SWITCH)
    history = {
        "iteration_1_unbounded_contagion": replay_iteration_one(population, arguments.quick_runs),
        "iteration_2_narrow_thresholds": replay_iteration_two(population, arguments.runs, arguments.quick_runs),
        "iteration_3_reweighting": replay_iteration_three(population, arguments.quick_runs),
        "iteration_4_final": final_iteration,
    }
    RESULTS_FOLDER.mkdir(exist_ok=True)
    output_path = RESULTS_FOLDER / "model_development_history.json"
    output_path.write_text(json.dumps(history, indent=1))
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
