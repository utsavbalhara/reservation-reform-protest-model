import argparse
import json
from pathlib import Path

import numpy as np

from protest_simulation import SCENARIOS, SOCIAL_GROUP_NAMES, simulate_protest_campaign
from protest_simulation.monte_carlo import CAMPAIGN_SEED_OFFSET, MOBILIZATION_REGIMES, WORLD_DRAW_SEED_OFFSET
from protest_simulation.specifications import get_specification

RESULTS_FOLDER = Path(__file__).resolve().parent.parent / "results"
PEOPLE_PER_LAKH = 1e5


def trajectories_for_scenario(population, scenario, run_count, threshold_shift, specification):
    daily_totals, daily_by_group, daily_deaths, bandh_days = [], [], [], []
    for run_index in range(run_count):
        world = specification.world_sampler(specification.base_parameters, np.random.default_rng(WORLD_DRAW_SEED_OFFSET + run_index))
        world.mean_participation_threshold += threshold_shift
        for intervention in scenario.interventions:
            intervention(world)
        outcome = simulate_protest_campaign(population, world, np.random.default_rng(CAMPAIGN_SEED_OFFSET + run_index))
        daily_totals.append(outcome.daily_protesters / PEOPLE_PER_LAKH)
        daily_by_group.append(outcome.daily_protesters_by_group / PEOPLE_PER_LAKH)
        daily_deaths.append(outcome.daily_deaths)
        bandh_days.append([int(day) for day in getattr(outcome, "bandh_days", world.bandh_call_days)])
    daily_totals = np.array(daily_totals)
    daily_by_group = np.array(daily_by_group)
    return {
        "code": scenario.code,
        "label": scenario.label,
        "category": scenario.category,
        "median_daily_lakh": np.round(np.median(daily_totals, axis=0), 3).tolist(),
        "daily_lakh_10th_percentile": np.round(np.percentile(daily_totals, 10, axis=0), 3).tolist(),
        "daily_lakh_90th_percentile": np.round(np.percentile(daily_totals, 90, axis=0), 3).tolist(),
        "median_daily_lakh_by_group": {
            name: np.round(np.median(daily_by_group[:, :, index], axis=0), 3).tolist()
            for index, name in enumerate(SOCIAL_GROUP_NAMES)
        },
        "mean_cumulative_deaths": np.round(np.cumsum(np.mean(daily_deaths, axis=0)), 2).tolist(),
        "bandh_days_by_run": bandh_days,
    }


def main():
    parser = argparse.ArgumentParser(description="Record day-by-day turnout paths for every scenario.")
    parser.add_argument("--regime", choices=sorted(MOBILIZATION_REGIMES), default="central")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--agents", type=int, default=120_000)
    parser.add_argument("--specification", default="stylized")
    arguments = parser.parse_args()

    specification = get_specification(arguments.specification)
    population = specification.build_population(arguments.agents)
    threshold_shift = MOBILIZATION_REGIMES[arguments.regime]
    trajectories = {
        "specification": arguments.specification,
        "regime": arguments.regime,
        "run_count": arguments.runs,
        "bandh_call_days": list(specification.base_parameters.bandh_call_days),
        "scenarios": {},
    }
    for scenario in SCENARIOS:
        trajectories["scenarios"][scenario.key] = trajectories_for_scenario(population, scenario, arguments.runs, threshold_shift, specification)
        print(f"{scenario.code:>4} {scenario.label}: peak of median path {max(trajectories['scenarios'][scenario.key]['median_daily_lakh']):.1f} lakh", flush=True)
    RESULTS_FOLDER.mkdir(exist_ok=True)
    prefix = "" if arguments.specification == "stylized" else f"{arguments.specification}_"
    output_path = RESULTS_FOLDER / f"daily_trajectories_{prefix}{arguments.regime}.json"
    output_path.write_text(json.dumps(trajectories, ensure_ascii=False))
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
