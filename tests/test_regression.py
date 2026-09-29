"""The stylized reference specification must keep producing exactly the numbers it produced before any
structural option was added. The snapshot was taken from the original code at 20,000 agents."""
import hashlib

import numpy as np
import pytest

from protest_simulation import BASELINE_ABRUPT_INCOME_ONLY_SWITCH, SCENARIO_BY_KEY, run_paired_monte_carlo, simulate_protest_campaign

SNAPSHOT_SCENARIOS = ("baseline", "hybrid_caste_subquotas", "heavy_policing", "internet_shutdown", "managed_transition")


def population_fingerprint(population):
    integer_parts = np.concatenate([population.social_group, population.is_most_deprived_tier, population.is_above_income_line,
                                    population.neighbourhood]).astype(np.int64).tobytes()
    return hashlib.sha256(integer_parts + population.identity_strength.tobytes() + population.threshold_standard_score.tobytes()).hexdigest()


def test_population_construction_is_unchanged(small_population, reference_outputs):
    assert population_fingerprint(small_population) == reference_outputs["population_sha"]


@pytest.mark.parametrize("scenario_key", SNAPSHOT_SCENARIOS)
def test_reference_campaigns_are_unchanged(small_population, reference_outputs, scenario_key):
    expected = reference_outputs[scenario_key]
    parameters = SCENARIO_BY_KEY[scenario_key].apply_to(BASELINE_ABRUPT_INCOME_ONLY_SWITCH)
    outcome = simulate_protest_campaign(small_population, parameters, np.random.default_rng(3), record_neighbourhood_turnout=True)
    assert outcome.peak_day_protesters == expected["peak"]
    assert outcome.cumulative_unique_protesters == expected["cumulative"]
    assert outcome.total_deaths == expected["deaths"]
    assert outcome.daily_protesters.tolist() == expected["daily"]
    assert outcome.daily_protesters_by_group.tolist() == expected["daily_by_group"]
    assert outcome.daily_deaths.tolist() == expected["daily_deaths"]
    day_five = hashlib.sha256(np.round(outcome.daily_neighbourhood_turnout[5], 12).tobytes()).hexdigest()
    assert day_five == expected["neighbourhood_day5_sha"]


def test_paired_monte_carlo_is_unchanged(small_population, reference_outputs):
    runs = run_paired_monte_carlo(small_population, SCENARIO_BY_KEY["consensus_commission"].interventions, 4, workers=1)
    expected = reference_outputs["paired_consensus_4runs"]
    assert runs.peak_day_protesters == expected["peaks"]
    assert runs.cumulative_unique_protesters == expected["cumulative"]
    assert runs.total_deaths == expected["deaths"]
