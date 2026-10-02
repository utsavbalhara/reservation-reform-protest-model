from dataclasses import fields, replace

import numpy as np
import pytest

from protest_simulation import BASELINE_ABRUPT_INCOME_ONLY_SWITCH, SCENARIO_BY_KEY, SCENARIOS, run_paired_monte_carlo, simulate_protest_campaign
from protest_simulation import policy_interventions as levers
from protest_simulation.monte_carlo import paired_effects_from_runs
from protest_simulation.parameter_uncertainty import draw_plausible_world
from protest_simulation.protest_campaign import material_loss_felt_by_each_agent
from protest_simulation.synthetic_population import GENERAL, OBC, POPULATION_SHARE_BY_GROUP, SC, ST


def run(population, parameters, seed=0):
    return simulate_protest_campaign(population, parameters, np.random.default_rng(seed))


# Population -------------------------------------------------------------------------------------------------------

def test_group_shares_match_targets(small_population):
    shares = np.bincount(small_population.social_group, minlength=4) / small_population.agent_count
    assert np.allclose(shares, POPULATION_SHARE_BY_GROUP, atol=0.01)


def test_above_line_shares_are_ordered_by_group(small_population):
    share = {g: small_population.is_above_income_line[small_population.social_group == g].mean() for g in (SC, ST, OBC, GENERAL)}
    assert share[GENERAL] > share[OBC] > share[SC] > share[ST] > 0
    overall = small_population.is_above_income_line.mean()
    assert 0.17 < overall < 0.23


def test_only_sc_and_st_have_tiers(small_population):
    tiers = small_population.is_most_deprived_tier
    assert not tiers[small_population.social_group >= OBC].any()
    sc_st = small_population.social_group <= ST
    assert 0.45 < tiers[sc_st].mean() < 0.55


def test_reference_neighbourhoods_are_single_group_and_about_100_agents(small_population):
    for neighbourhood in range(small_population.neighbourhood_count):
        members = small_population.neighbourhood == neighbourhood
        assert np.unique(small_population.social_group[members]).size == 1
        assert 50 <= members.sum() <= 200


# Campaign accounting ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("scenario_key", [scenario.key for scenario in SCENARIOS])
def test_campaign_accounting_identities(small_population, scenario_key):
    outcome = run(small_population, SCENARIO_BY_KEY[scenario_key].apply_to(BASELINE_ABRUPT_INCOME_ONLY_SWITCH))
    people = small_population.people_represented_per_agent
    assert np.allclose(outcome.daily_protesters_by_group.sum(axis=1), outcome.daily_protesters)
    assert outcome.peak_day_protesters == outcome.daily_protesters.max()
    assert outcome.cumulative_unique_protesters >= outcome.peak_day_protesters
    assert outcome.cumulative_unique_protesters <= small_population.agent_count * people
    assert outcome.total_deaths == outcome.daily_deaths.sum() and (outcome.daily_deaths >= 0).all()
    assert (outcome.daily_protesters >= 0).all()
    # Counts are whole agents.
    assert np.allclose(outcome.daily_protesters / people, np.round(outcome.daily_protesters / people))


def test_same_seed_same_campaign(small_population):
    first = run(small_population, BASELINE_ABRUPT_INCOME_ONLY_SWITCH, seed=4)
    second = run(small_population, BASELINE_ABRUPT_INCOME_ONLY_SWITCH, seed=4)
    assert first.daily_protesters.tolist() == second.daily_protesters.tolist()
    assert first.total_deaths == second.total_deaths


def test_general_category_barely_protests_against_a_reform_that_favours_it(small_population):
    outcome = run(small_population, BASELINE_ABRUPT_INCOME_ONLY_SWITCH)
    assert outcome.daily_protesters_by_group[:, GENERAL].sum() <= 0.01 * outcome.daily_protesters.sum()


# Monotone responses, in expectation over seeds --------------------------------------------------------------------

def median_peak(population, parameters, seeds=range(6)):
    return float(np.median([run(population, parameters, seed).peak_day_protesters for seed in seeds]))


@pytest.mark.parametrize("field_name, values, direction", [
    ("mean_participation_threshold", (6.0, 6.6, 7.2), -1),
    ("symbolic_threat_weight", (2.6, 3.3, 4.0), +1),
    ("material_loss_weight", (0.2, 0.45, 0.9), +1),
    ("opposition_party_amplifier", (1.0, 1.5, 2.0), +1),
])
def test_turnout_moves_in_the_expected_direction(small_population, field_name, values, direction):
    peaks = [median_peak(small_population, replace(BASELINE_ABRUPT_INCOME_ONLY_SWITCH.copy(), **{field_name: value})) for value in values]
    assert all(direction * (later - earlier) > 0 for earlier, later in zip(peaks, peaks[1:])), peaks


def test_symbolic_only_scenario_has_no_material_loss(small_population):
    parameters = SCENARIO_BY_KEY["symbolic_only_validation"].apply_to(BASELINE_ABRUPT_INCOME_ONLY_SWITCH)
    assert not material_loss_felt_by_each_agent(small_population, parameters).any()


# Interventions and pairing ------------------------------------------------------------------------------------------

EXPECTED_FIELDS_CHANGED = {
    levers.grandfather_current_cohorts_with_ten_year_glide: {"material_loss_sc_st_above_income_line", "material_loss_sc_st_below_income_line",
                                                              "material_loss_obc_below_income_line", "symbolic_threat_by_group"},
    levers.expand_seats_so_no_group_loses: {"material_loss_sc_st_below_income_line", "material_loss_obc_below_income_line"},
    levers.keep_caste_subquotas_with_income_filter: {"symbolic_threat_by_group", "material_loss_sc_st_below_income_line",
                                                     "material_loss_obc_below_income_line"},
    levers.sub_classify_to_favour_most_deprived: {"sub_classification_gain_for_most_deprived_tier", "most_deprived_tier_share_of_symbolic_threat",
                                                   "better_off_tier_symbolic_threat_factor"},
    levers.build_consensus_through_data_first_commission: {"opposition_party_amplifier", "symbolic_threat_by_group"},
    levers.compensate_above_line_losers: {"material_loss_sc_st_above_income_line"},
    levers.guarantee_untouched_protections: {"symbolic_threat_by_group"},
    levers.shut_down_internet: {"internet_shutdown_active"},
    levers.deploy_heavy_policing: {"heavy_policing_turnout_cost", "deaths_per_crore_protester_days", "symbolic_threat_rise_per_death"},
    levers.remove_all_material_loss: {"material_loss_sc_st_above_income_line", "material_loss_sc_st_below_income_line",
                                      "material_loss_obc_below_income_line", "material_loss_general_below_income_line"},
}


def changed_fields(before, after):
    changed = set()
    for field in fields(before):
        old, new = getattr(before, field.name), getattr(after, field.name)
        if isinstance(old, np.ndarray):
            if not np.array_equal(old, new):
                changed.add(field.name)
        elif old != new:
            changed.add(field.name)
    return changed


@pytest.mark.parametrize("intervention", list(EXPECTED_FIELDS_CHANGED))
def test_each_intervention_touches_only_its_own_parameters(intervention):
    before = BASELINE_ABRUPT_INCOME_ONLY_SWITCH.copy()
    after = before.copy()
    intervention(after)
    assert changed_fields(before, after) == EXPECTED_FIELDS_CHANGED[intervention]


def test_applying_a_scenario_never_mutates_the_baseline():
    before = BASELINE_ABRUPT_INCOME_ONLY_SWITCH.copy()
    for scenario in SCENARIOS:
        scenario.apply_to(BASELINE_ABRUPT_INCOME_ONLY_SWITCH)
    assert changed_fields(before, BASELINE_ABRUPT_INCOME_ONLY_SWITCH) == set()


def test_world_draws_depend_only_on_the_seed():
    first = draw_plausible_world(BASELINE_ABRUPT_INCOME_ONLY_SWITCH, np.random.default_rng(1003))
    second = draw_plausible_world(BASELINE_ABRUPT_INCOME_ONLY_SWITCH, np.random.default_rng(1003))
    assert changed_fields(first, second) == set()


def test_parallel_and_serial_runs_agree(small_population):
    serial = run_paired_monte_carlo(small_population, SCENARIO_BY_KEY["heavy_policing"].interventions, 5, workers=1)
    parallel = run_paired_monte_carlo(small_population, SCENARIO_BY_KEY["heavy_policing"].interventions, 5, workers=3)
    assert serial.peak_day_protesters == parallel.peak_day_protesters
    assert serial.total_deaths == parallel.total_deaths


# Statistics ----------------------------------------------------------------------------------------------------------

def test_paired_effects_recover_a_known_ratio():
    baseline = {"peak_day_protesters": [10.0, 20.0, 40.0, 80.0], "cumulative_unique_protesters": [1.0, 2.0, 3.0, 4.0], "total_deaths": [2, 4, 6, 8]}
    halved = {key: [value / 2 for value in values] for key, values in baseline.items()}
    effects = paired_effects_from_runs(halved, baseline)
    for outcome in ("peak", "cumulative", "deaths"):
        assert effects[outcome]["change_percent"] == -50.0
        assert effects[outcome]["change_percent_ci95"] == [-50.0, -50.0]
        assert effects[outcome]["share_of_runs_higher_than_baseline"] == 0.0


def test_paired_effects_skip_runs_with_a_zero_baseline():
    baseline = {"peak_day_protesters": [1.0, 1.0], "cumulative_unique_protesters": [1.0, 1.0], "total_deaths": [0, 5]}
    scenario = {"peak_day_protesters": [1.0, 1.0], "cumulative_unique_protesters": [1.0, 1.0], "total_deaths": [3, 10]}
    assert paired_effects_from_runs(scenario, baseline)["deaths"]["paired_runs"] == 1
