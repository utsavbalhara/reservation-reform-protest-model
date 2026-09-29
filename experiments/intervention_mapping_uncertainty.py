"""How robust is the lever ranking to the values chosen for each intervention's effect?

Each Monte Carlo run draws one plausible world and one set of intervention effect sizes from the priors in
intervention_priors.py, and applies every lever with that run's effect sizes to that run's world. The outputs are the
probability that each lever is the strongest single lever, the full rank distribution, paired effects with bootstrap
intervals, partial rank correlations showing which effect sizes drive which results, and, for suppression, how often
force backfires as a function of the regime drawn.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from protest_simulation.intervention_priors import (
    LEVER_PRIORS, PACKAGES, SINGLE_LEVERS, SUPPRESSION_PRIORS, interventions_for, mapping_for_run, reference_mapping)
from protest_simulation.monte_carlo import CAMPAIGN_SEED_OFFSET, paired_effects_from_runs, paired_worlds
from protest_simulation.parallel import run_campaigns
from protest_simulation.policy_interventions import remove_all_material_loss
from protest_simulation.specifications import get_specification

RESULTS_FOLDER = Path(__file__).resolve().parent.parent / "results"
SCENARIOS = ("baseline", "symbolic_only_validation") + SINGLE_LEVERS + tuple(PACKAGES) + tuple(SUPPRESSION_PRIORS)


def partial_rank_correlations(inputs: np.ndarray, output: np.ndarray, names) -> dict:
    ranked_inputs = np.column_stack([rankdata(column) for column in inputs.T])
    ranked_output = rankdata(output)
    correlations = {}
    for j, name in enumerate(names):
        others = np.column_stack([np.ones(len(output)), np.delete(ranked_inputs, j, axis=1)])
        residual_x = ranked_inputs[:, j] - others @ np.linalg.lstsq(others, ranked_inputs[:, j], rcond=None)[0]
        residual_y = ranked_output - others @ np.linalg.lstsq(others, ranked_output, rcond=None)[0]
        correlations[name] = round(float(np.corrcoef(residual_x, residual_y)[0, 1]), 3)
    return correlations


def scenario_worlds(specification, scenario, run_count, mappings):
    worlds = paired_worlds((), run_count, base_parameters=specification.base_parameters, world_sampler=specification.world_sampler)
    for world, mapping in zip(worlds, mappings):
        if scenario == "symbolic_only_validation":
            remove_all_material_loss(world)
        elif scenario != "baseline":
            for intervention in interventions_for(scenario, mapping):
                intervention(world)
    return worlds


def main():
    parser = argparse.ArgumentParser(description="Propagate priors over intervention effect sizes.")
    parser.add_argument("--specification", default="stylized")
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--agents", type=int, default=120_000)
    arguments = parser.parse_args()

    specification = get_specification(arguments.specification)
    population = specification.build_population(arguments.agents)
    mappings = [mapping_for_run(index) for index in range(arguments.runs)]
    jobs, owners = [], []
    for scenario in SCENARIOS:
        for index, world in enumerate(scenario_worlds(specification, scenario, arguments.runs, mappings)):
            jobs.append((world, CAMPAIGN_SEED_OFFSET + index))
            owners.append((scenario, index))
    outcomes = run_campaigns(population, jobs)
    runs = {scenario: {"peak_day_protesters": [None] * arguments.runs, "cumulative_unique_protesters": [None] * arguments.runs,
                       "total_deaths": [None] * arguments.runs, "protester_days": [None] * arguments.runs,
                       "conceded": [None] * arguments.runs} for scenario in SCENARIOS}
    for (scenario, index), outcome in zip(owners, outcomes):
        runs[scenario]["peak_day_protesters"][index] = outcome.peak_day_protesters
        runs[scenario]["cumulative_unique_protesters"][index] = outcome.cumulative_unique_protesters
        runs[scenario]["total_deaths"][index] = outcome.total_deaths
        runs[scenario]["protester_days"][index] = outcome.protester_days
        runs[scenario]["conceded"][index] = outcome.conceded_on_day is not None

    baseline_peaks = np.array(runs["baseline"]["peak_day_protesters"])
    log_ratio = {scenario: np.log(np.array(runs[scenario]["peak_day_protesters"]) / baseline_peaks) for scenario in SCENARIOS}
    single = np.column_stack([log_ratio[lever] for lever in SINGLE_LEVERS])
    ranks = np.column_stack([rankdata(row, method="min") for row in single]).T  # 1 = largest reduction
    rank_distribution = {lever: {str(rank): round(float(np.mean(ranks[:, j] == rank)), 3) for rank in range(1, len(SINGLE_LEVERS) + 1)}
                         for j, lever in enumerate(SINGLE_LEVERS)}
    pairwise = {a: {b: round(float(np.mean(log_ratio[a] < log_ratio[b])), 3) for b in SINGLE_LEVERS if b != a} for a in SINGLE_LEVERS}

    mapping_names = [f"{lever}.{name}" for lever, values in {**LEVER_PRIORS, **SUPPRESSION_PRIORS}.items() for name in values]
    mapping_matrix = np.array([[mapping[lever][name] for lever, values in {**LEVER_PRIORS, **SUPPRESSION_PRIORS}.items() for name in values]
                               for mapping in mappings])
    drivers = {}
    for scenario in SINGLE_LEVERS + tuple(SUPPRESSION_PRIORS):
        own = [j for j, name in enumerate(mapping_names) if name.startswith(scenario + ".")]
        drivers[scenario] = partial_rank_correlations(mapping_matrix[:, own], log_ratio[scenario], [mapping_names[j] for j in own]) if own else {}

    suppression = {}
    for scenario in SUPPRESSION_PRIORS:
        cumulative_ratio = np.array(runs[scenario]["cumulative_unique_protesters"]) / np.array(runs["baseline"]["cumulative_unique_protesters"])
        days_ratio = np.array(runs[scenario]["protester_days"]) / np.array(runs["baseline"]["protester_days"])
        response = mapping_matrix[:, mapping_names.index(f"{scenario}.death_response_factor")]
        bins = np.quantile(response, [0, 0.25, 0.5, 0.75, 1.0])
        by_response = []
        for low, high in zip(bins[:-1], bins[1:]):
            chosen = (response >= low) & (response <= high)
            by_response.append({"death_response_factor": [round(float(low), 2), round(float(high), 2)],
                                "share_cumulative_higher": round(float(np.mean(cumulative_ratio[chosen] > 1)), 3),
                                "share_peak_higher": round(float(np.mean(log_ratio[scenario][chosen] > 0)), 3),
                                "median_protester_day_ratio": round(float(np.median(days_ratio[chosen])), 3)})
        suppression[scenario] = {
            "share_of_runs_peak_higher": round(float(np.mean(log_ratio[scenario] > 0)), 3),
            "share_of_runs_cumulative_higher": round(float(np.mean(cumulative_ratio > 1)), 3),
            "share_of_runs_protester_days_higher": round(float(np.mean(days_ratio > 1)), 3),
            "by_death_response_factor_quartile": by_response,
            "drivers_of_cumulative_ratio": partial_rank_correlations(
                mapping_matrix[:, [j for j, name in enumerate(mapping_names) if name.startswith(scenario + ".")]], np.log(cumulative_ratio),
                [name for name in mapping_names if name.startswith(scenario + ".")]),
        }

    output = {
        "specification": arguments.specification,
        "run_count": arguments.runs,
        "agent_count": arguments.agents,
        "reference_mapping": reference_mapping(),
        "probability_strongest_single_lever": {lever: rank_distribution[lever]["1"] for lever in SINGLE_LEVERS},
        "probability_weakest_single_lever": {lever: rank_distribution[lever][str(len(SINGLE_LEVERS))] for lever in SINGLE_LEVERS},
        "rank_distribution": rank_distribution,
        "probability_row_beats_column": pairwise,
        "paired_effects": {scenario: paired_effects_from_runs(runs[scenario], runs["baseline"]) for scenario in SCENARIOS if scenario != "baseline"},
        "drivers_of_effect_partial_rank_correlation": drivers,
        "suppression_backfire": suppression,
        "mappings": mappings,
        "runs": runs,
    }
    RESULTS_FOLDER.mkdir(exist_ok=True)
    path = RESULTS_FOLDER / f"intervention_mapping_uncertainty_{arguments.specification}.json"
    path.write_text(json.dumps(output, indent=1))
    print(json.dumps({key: output[key] for key in ("probability_strongest_single_lever", "probability_weakest_single_lever")}, indent=1))
    print(json.dumps({key: value["peak"]["change_percent"] for key, value in output["paired_effects"].items()}))
    print(json.dumps({key: {k: v for k, v in value.items() if k.startswith("share")} for key, value in suppression.items()}))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
