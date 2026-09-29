from dataclasses import dataclass

import numpy as np

from .model_parameters import BASELINE_ABRUPT_INCOME_ONLY_SWITCH, ProtestModelParameters
from .parameter_uncertainty import MEAN_THRESHOLD_STANDARD_DEVIATION, draw_plausible_world
from .protest_campaign import simulate_protest_campaign
from .synthetic_population import build_synthetic_india

POPULATION_SEED = 7
WORLD_DRAW_SEED_OFFSET = 1000
CAMPAIGN_SEED_OFFSET = 5000
DEFAULT_AGENT_COUNT = 120_000
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20_260_929

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
    bandh_days_called: list = None
    conceded_on_day: list = None
    campaign_days_run: list = None

    def summary(self) -> dict:
        peaks = np.array(self.peak_day_protesters)
        summary = {
            "median_peak_lakh": round(float(np.median(peaks)) / PEOPLE_PER_LAKH, 1),
            "peak_lakh_10th_percentile": round(float(np.percentile(peaks, 10)) / PEOPLE_PER_LAKH, 1),
            "peak_lakh_90th_percentile": round(float(np.percentile(peaks, 90)) / PEOPLE_PER_LAKH, 1),
            "median_cumulative_crore": round(float(np.median(self.cumulative_unique_protesters)) / PEOPLE_PER_CRORE, 2),
            "median_deaths": float(np.median(self.total_deaths)),
        }
        if self.bandh_days_called:
            summary["median_bandhs_called"] = float(np.median([len(days) for days in self.bandh_days_called]))
        if self.conceded_on_day:
            conceded = [day for day in self.conceded_on_day if day is not None]
            summary["share_of_runs_conceded"] = round(len(conceded) / len(self.conceded_on_day), 3)
            summary["median_concession_day"] = float(np.median(conceded)) if conceded else None
        if self.campaign_days_run:
            summary["median_campaign_days"] = float(np.median(self.campaign_days_run))
        return summary

    def paired_effects(self, baseline: "ScenarioRuns") -> dict:
        """Effects relative to a baseline that used the same world and campaign seeds, run by run.

        The headline statistic for each outcome is the median of the per-run ratios (scenario / baseline),
        with a percentile bootstrap 95% interval over runs. Reporting the ratio of scenario medians instead
        can misstate an intervention's effect when the run-to-run spread is large, which it is here.
        """
        return paired_effects_from_runs(
            {"peak_day_protesters": self.peak_day_protesters, "cumulative_unique_protesters": self.cumulative_unique_protesters,
             "total_deaths": self.total_deaths},
            {"peak_day_protesters": baseline.peak_day_protesters, "cumulative_unique_protesters": baseline.cumulative_unique_protesters,
             "total_deaths": baseline.total_deaths},
        )


def paired_effects_from_runs(scenario_runs: dict, baseline_runs: dict,
                             resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED) -> dict:
    effects = {}
    for outcome, short in (("peak_day_protesters", "peak"), ("cumulative_unique_protesters", "cumulative"), ("total_deaths", "deaths")):
        scenario = np.asarray(scenario_runs[outcome], float)
        baseline = np.asarray(baseline_runs[outcome], float)
        if scenario.shape != baseline.shape:
            raise ValueError("paired effects need the same number of runs in scenario and baseline")
        defined = baseline > 0
        ratios = scenario[defined] / baseline[defined]
        rng = np.random.default_rng(seed)
        resampled = rng.integers(0, ratios.size, (resamples, ratios.size))
        bootstrap_medians = np.median(ratios[resampled], axis=1)
        effects[short] = {
            "median_paired_ratio": round(float(np.median(ratios)), 4),
            "change_percent": round((float(np.median(ratios)) - 1) * 100, 1),
            "change_percent_ci95": [round((float(np.percentile(bootstrap_medians, 2.5)) - 1) * 100, 1),
                                    round((float(np.percentile(bootstrap_medians, 97.5)) - 1) * 100, 1)],
            "share_of_runs_higher_than_baseline": round(float(np.mean(scenario[defined] > baseline[defined])), 3),
            "paired_runs": int(defined.sum()),
        }
    return effects


def build_shared_population(agent_count: int = DEFAULT_AGENT_COUNT, **population_options):
    return build_synthetic_india(agent_count, np.random.default_rng(POPULATION_SEED), **population_options)


def paired_worlds(interventions: tuple, run_count: int, regime_threshold_shift: float = 0.0,
                  base_parameters: ProtestModelParameters = BASELINE_ABRUPT_INCOME_ONLY_SWITCH,
                  mean_threshold_standard_deviation: float = MEAN_THRESHOLD_STANDARD_DEVIATION,
                  world_sampler=draw_plausible_world) -> list:
    """Run r of every scenario draws its uncertain parameters from seed 1000 + r, so scenarios are paired."""
    worlds = []
    for run_index in range(run_count):
        world = world_sampler(base_parameters, np.random.default_rng(WORLD_DRAW_SEED_OFFSET + run_index), mean_threshold_standard_deviation)
        world.mean_participation_threshold += regime_threshold_shift
        for intervention in interventions:
            intervention(world)
        worlds.append(world)
    return worlds


def run_paired_monte_carlo(
    population,
    interventions: tuple,
    run_count: int,
    regime_threshold_shift: float = 0.0,
    base_parameters: ProtestModelParameters = BASELINE_ABRUPT_INCOME_ONLY_SWITCH,
    mean_threshold_standard_deviation: float = MEAN_THRESHOLD_STANDARD_DEVIATION,
    workers: int = None,
    world_sampler=draw_plausible_world,
) -> ScenarioRuns:
    from .parallel import run_campaigns

    worlds = paired_worlds(interventions, run_count, regime_threshold_shift, base_parameters, mean_threshold_standard_deviation, world_sampler)
    outcomes = run_campaigns(population, [(world, CAMPAIGN_SEED_OFFSET + run_index) for run_index, world in enumerate(worlds)], workers)
    runs = ScenarioRuns([], [], [], [], [], [])
    for world, outcome in zip(worlds, outcomes):
        runs.peak_day_protesters.append(outcome.peak_day_protesters)
        runs.cumulative_unique_protesters.append(outcome.cumulative_unique_protesters)
        runs.total_deaths.append(outcome.total_deaths)
        runs.bandh_days_called.append(list(getattr(outcome, "bandh_days", world.bandh_call_days)))
        runs.conceded_on_day.append(getattr(outcome, "conceded_on_day", None))
        runs.campaign_days_run.append(getattr(outcome, "campaign_days_run", world.campaign_length_days))
    return runs
