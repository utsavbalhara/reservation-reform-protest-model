from dataclasses import dataclass

import numpy as np

from .model_parameters import BASELINE_ABRUPT_INCOME_ONLY_SWITCH, ProtestModelParameters
from .parameter_uncertainty import draw_plausible_world
from .protest_campaign import simulate_protest_campaign
from .synthetic_population import build_synthetic_india

POPULATION_SEED = 7
WORLD_DRAW_SEED_OFFSET = 1000
CAMPAIGN_SEED_OFFSET = 5000
DEFAULT_AGENT_COUNT = 120_000

MOBILIZATION_REGIMES = {
    "central": 0.0,
    "high-mobilization": -0.5,
}

PEOPLE_PER_LAKH = 1e5
PEOPLE_PER_CRORE = 1e7


@dataclass
class ScenarioRuns:
    peak_day_protesters: list
    cumulative_unique_protesters: list
    total_deaths: list

    def summary(self) -> dict:
        peaks = np.array(self.peak_day_protesters)
        return {
            "median_peak_lakh": round(float(np.median(peaks)) / PEOPLE_PER_LAKH, 1),
            "peak_lakh_10th_percentile": round(float(np.percentile(peaks, 10)) / PEOPLE_PER_LAKH, 1),
            "peak_lakh_90th_percentile": round(float(np.percentile(peaks, 90)) / PEOPLE_PER_LAKH, 1),
            "median_cumulative_crore": round(float(np.median(self.cumulative_unique_protesters)) / PEOPLE_PER_CRORE, 2),
            "median_deaths": float(np.median(self.total_deaths)),
        }


def build_shared_population(agent_count: int = DEFAULT_AGENT_COUNT):
    return build_synthetic_india(agent_count, np.random.default_rng(POPULATION_SEED))


def run_paired_monte_carlo(
    population,
    interventions: tuple,
    run_count: int,
    regime_threshold_shift: float = 0.0,
    base_parameters: ProtestModelParameters = BASELINE_ABRUPT_INCOME_ONLY_SWITCH,
) -> ScenarioRuns:
    runs = ScenarioRuns([], [], [])
    for run_index in range(run_count):
        world = draw_plausible_world(base_parameters, np.random.default_rng(WORLD_DRAW_SEED_OFFSET + run_index))
        world.mean_participation_threshold += regime_threshold_shift
        for intervention in interventions:
            intervention(world)
        outcome = simulate_protest_campaign(population, world, np.random.default_rng(CAMPAIGN_SEED_OFFSET + run_index))
        runs.peak_day_protesters.append(outcome.peak_day_protesters)
        runs.cumulative_unique_protesters.append(outcome.cumulative_unique_protesters)
        runs.total_deaths.append(outcome.total_deaths)
    return runs
