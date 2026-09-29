from dataclasses import replace

import numpy as np
import pytest

from protest_simulation import BASELINE_ABRUPT_INCOME_ONLY_SWITCH, SCENARIO_BY_KEY, simulate_protest_campaign

from .reference_implementation import simulate_reference_campaign

SCENARIOS_TO_CROSS_CHECK = ("baseline", "heavy_policing", "internet_shutdown", "sub_classification", "hybrid_package", "symbolic_only_validation")


@pytest.mark.parametrize("scenario_key", SCENARIOS_TO_CROSS_CHECK)
def test_scalar_reimplementation_matches_vectorised_model(tiny_population, scenario_key):
    # A lower threshold than the calibrated one makes these 2,000-agent campaigns busy enough to exercise deaths
    # and the martyr effect, which the calibrated setting rarely triggers at this size.
    parameters = SCENARIO_BY_KEY[scenario_key].apply_to(replace(BASELINE_ABRUPT_INCOME_ONLY_SWITCH.copy(), mean_participation_threshold=5.2,
                                                                 deaths_per_crore_protester_days=40.0, campaign_length_days=16,
                                                                 bandh_call_days=(2, 7, 12)))
    vectorised = simulate_protest_campaign(tiny_population, parameters, np.random.default_rng(99))
    scalar = simulate_reference_campaign(tiny_population, parameters, np.random.default_rng(99))
    assert scalar["daily_deaths"] == vectorised.daily_deaths.tolist()
    assert np.allclose(scalar["daily_protesters"], vectorised.daily_protesters, rtol=0, atol=1e-6)
    assert np.allclose(scalar["daily_by_group"], vectorised.daily_protesters_by_group, rtol=0, atol=1e-6)
    assert scalar["cumulative"] == pytest.approx(vectorised.cumulative_unique_protesters)
    assert scalar["deaths"] == vectorised.total_deaths


def test_cross_check_exercises_deaths_and_martyr_effect(tiny_population):
    parameters = replace(BASELINE_ABRUPT_INCOME_ONLY_SWITCH.copy(), mean_participation_threshold=5.2, deaths_per_crore_protester_days=40.0,
                         campaign_length_days=16, bandh_call_days=(2, 7, 12))
    outcome = simulate_reference_campaign(tiny_population, parameters, np.random.default_rng(99))
    assert outcome["deaths"] > 0
    assert outcome["peak"] > 0
