"""Each structural option does what its documentation says."""
from dataclasses import replace

import numpy as np
import pytest

from protest_simulation import BASELINE_ABRUPT_INCOME_ONLY_SWITCH, simulate_protest_campaign
from protest_simulation.protest_campaign import (
    SEGMENT_NAMES,
    eligibility_segment_of_each_agent,
    material_loss_felt_by_each_agent,
    threshold_scores,
)
from protest_simulation.synthetic_population import GENERAL, OBC, SC, build_synthetic_india

BASE = BASELINE_ABRUPT_INCOME_ONLY_SWITCH


def with_options(**options):
    return replace(BASE.copy(), **options)


def run(population, parameters, seed=0):
    return simulate_protest_campaign(population, parameters, np.random.default_rng(seed))


@pytest.mark.parametrize("shape", ["logistic", "activist_mixture"])
def test_threshold_shapes_are_standardised_and_keep_agent_order(small_population, shape):
    scores = threshold_scores(small_population, with_options(threshold_distribution=shape))
    assert abs(scores.mean()) < 0.03 and abs(scores.std() - 1) < 0.03
    order = np.argsort(small_population.threshold_standard_score)
    assert (np.diff(scores[order]) >= 0).all()


def test_activist_mixture_has_a_heavier_low_tail_than_normal(small_population):
    normal = threshold_scores(small_population, BASE)
    mixture = threshold_scores(small_population, with_options(threshold_distribution="activist_mixture"))
    assert np.quantile(mixture, 0.01) < np.quantile(normal, 0.01) - 0.3


def test_endogenous_bandhs_follow_the_calling_rule(small_population):
    never = run(small_population, with_options(bandh_schedule="endogenous", bandh_call_expected_turnout=1e12))
    assert never.bandh_days == []
    for seed in range(5):
        always = run(small_population, with_options(bandh_schedule="endogenous", bandh_call_expected_turnout=0.0, random_streams="split"), seed)
        assert always.bandh_days[0] == BASE.first_bandh_call_day
        gaps = np.diff(always.bandh_days)
        assert ((gaps >= BASE.bandh_refractory_days[0]) & (gaps <= BASE.bandh_refractory_days[1])).all()
        # With no calling threshold, organizations call as often as the refractory gap and the call limit allow.
        room_for_another = always.bandh_days[-1] + BASE.bandh_refractory_days[0] < BASE.campaign_length_days
        assert len(always.bandh_days) == BASE.max_bandh_calls or not room_for_another


def test_fixed_schedule_reports_its_bandh_days(small_population):
    assert run(small_population, BASE).bandh_days == list(BASE.bandh_call_days)


def test_certain_concession_happens_on_day_one_and_cancels_remaining_bandhs(small_population):
    parameters = with_options(concession_rule=True, concession_pressure_midpoint=-100.0, concession_max_daily_hazard=1.0,
                              random_streams="split")
    outcome = run(small_population, parameters)
    assert outcome.conceded_on_day == 1
    assert outcome.bandh_days == []


def test_concession_relieves_grievance_and_can_trigger_backlash(small_population):
    quiet = with_options(concession_rule=True, concession_pressure_midpoint=-100.0, concession_max_daily_hazard=1.0,
                         random_streams="split", concession_grievance_relief=0.9)
    backlash = replace(quiet, counter_mobilization_symbolic_threat=2.5)
    uncontested = with_options(random_streams="split")
    assert run(small_population, quiet).daily_protesters[5:].sum() < run(small_population, uncontested).daily_protesters[5:].sum()
    general_quiet = run(small_population, quiet).daily_protesters_by_group[:, GENERAL].sum()
    general_backlash = run(small_population, backlash).daily_protesters_by_group[:, GENERAL].sum()
    assert general_backlash > general_quiet


def test_split_deaths_have_the_intended_mean_and_police_share(small_population):
    parameters = with_options(death_model="negative_binomial_split", death_dispersion=1e6, random_streams="split",
                              deaths_per_crore_protester_days=400.0, mean_participation_threshold=5.5)
    totals, police, protester_days = 0, 0, 0.0
    for seed in range(12):
        outcome = run(small_population, parameters, seed)
        totals += outcome.total_deaths
        police += outcome.daily_police_deaths.sum()
        protester_days += outcome.protester_days
    expected = 400.0 * protester_days / 1e7
    assert totals == pytest.approx(expected, rel=0.08)
    assert police / totals == pytest.approx(0.5, abs=0.05)


def test_deaths_that_intimidate_reduce_turnout_relative_to_deaths_that_mobilize(small_population):
    common = dict(death_model="negative_binomial_split", random_streams="split", deaths_per_crore_protester_days=200.0,
                  symbolic_threat_rise_per_death=0.02)
    backfire = np.median([run(small_population, with_options(death_response=1.5, **common), s).protester_days for s in range(8)])
    deterrence = np.median([run(small_population, with_options(death_response=-1.0, **common), s).protester_days for s in range(8)])
    assert deterrence < backfire


def test_split_streams_share_agent_draws_across_scenarios(small_population):
    calm = run(small_population, with_options(random_streams="split", deaths_per_crore_protester_days=0.0), seed=5)
    violent = run(small_population, with_options(random_streams="split", deaths_per_crore_protester_days=500.0), seed=5)
    assert calm.daily_protesters[0] == violent.daily_protesters[0]


def dissimilarity_index(population, group):
    in_group = population.social_group == group
    per_neighbourhood_group = np.bincount(population.neighbourhood, weights=in_group, minlength=population.neighbourhood_count)
    per_neighbourhood_rest = np.bincount(population.neighbourhood, weights=~in_group, minlength=population.neighbourhood_count)
    return 0.5 * np.abs(per_neighbourhood_group / in_group.sum() - per_neighbourhood_rest / (~in_group).sum()).sum()


@pytest.mark.parametrize("share", [1.0, 0.6, 0.3])
def test_dissimilarity_index_is_the_same_group_share_plus_the_small_unit_floor(share):
    floor = dissimilarity_index(build_synthetic_india(40_000, np.random.default_rng(7), same_group_neighbourhood_share=0.0), SC)
    assert 0.05 < floor < 0.2
    population = build_synthetic_india(40_000, np.random.default_rng(7), same_group_neighbourhood_share=share)
    assert dissimilarity_index(population, SC) == pytest.approx(share + (1 - share) * floor, abs=0.03)


def test_same_group_neighbour_share_matches_the_formula():
    share = 0.6
    population = build_synthetic_india(40_000, np.random.default_rng(7), same_group_neighbourhood_share=share)
    sc = population.social_group == SC
    sc_per_neighbourhood = np.bincount(population.neighbourhood, weights=sc, minlength=population.neighbourhood_count)
    size = np.bincount(population.neighbourhood, minlength=population.neighbourhood_count)
    same_group_neighbours = (sc_per_neighbourhood / size)[population.neighbourhood[sc]].mean()
    assert same_group_neighbours == pytest.approx(share + (1 - share) * sc.mean(), abs=0.03)


def test_eligibility_segments_cover_every_agent_and_respect_the_flags():
    population = build_synthetic_india(40_000, np.random.default_rng(7), obc_creamy_share_of_above_line=0.4, ews_asset_exclusion_share=0.1)
    segment = eligibility_segment_of_each_agent(population)
    assert (segment >= 0).all()
    obc_above = (population.social_group == OBC) & population.is_above_income_line
    assert (segment[obc_above & population.is_obc_creamy_layer] == SEGMENT_NAMES.index("OBC_creamy")).all()
    share_creamy = population.is_obc_creamy_layer[obc_above].mean()
    assert share_creamy == pytest.approx(0.4, abs=0.03)
    general_below = (population.social_group == GENERAL) & ~population.is_above_income_line
    assert population.is_asset_excluded_from_ews[general_below].mean() == pytest.approx(0.1, abs=0.02)


def test_segment_table_reproduces_the_reference_material_losses(small_population):
    reference = material_loss_felt_by_each_agent(small_population, BASE)
    table = {"SC_above": 1.0, "ST_above": 1.0, "SC_below": 0.35, "ST_below": 0.35, "OBC_below": 0.25, "General_below_ews": -0.15}
    assert np.array_equal(material_loss_felt_by_each_agent(small_population, with_options(material_change_by_segment=table)), reference)


def test_extra_population_options_leave_the_reference_draws_untouched(small_population):
    detailed = build_synthetic_india(small_population.agent_count, np.random.default_rng(7), obc_creamy_share_of_above_line=0.5,
                                     ews_asset_exclusion_share=0.1)
    assert np.array_equal(detailed.social_group, small_population.social_group)
    assert np.array_equal(detailed.neighbourhood, small_population.neighbourhood)
    assert np.array_equal(detailed.threshold_standard_score, small_population.threshold_standard_score)


def test_levers_at_reference_mapping_match_the_scenario_catalogue():
    from protest_simulation import SCENARIO_BY_KEY
    from protest_simulation.intervention_priors import interventions_for, reference_mapping
    from .test_properties import changed_fields

    mapping = reference_mapping()
    for key in ("grandfathering", "seat_expansion", "hybrid_caste_subquotas", "sub_classification", "consensus_commission",
                "compensation", "credible_guarantees", "heavy_policing", "internet_shutdown", "managed_transition", "hybrid_package"):
        catalogue = SCENARIO_BY_KEY[key].apply_to(BASE)
        with_priors = BASE.copy()
        for intervention in interventions_for(key, mapping):
            intervention(with_priors)
        different = changed_fields(catalogue, with_priors) - {"symbolic_threat_rise_per_death", "death_response"}
        # Floating-point: 1/1.5 as a multiplier versus division by 1.5 can differ in the last bit.
        for name in different:
            assert np.allclose(getattr(catalogue, name), getattr(with_priors, name), rtol=1e-12), (key, name)
        # The catalogue raises the martyr effect per death; the priors raise the response to deaths. Same product.
        effective = lambda p: p.symbolic_threat_rise_per_death * p.death_response
        assert effective(catalogue) == pytest.approx(effective(with_priors), rel=1e-12), key


def test_mapping_draws_are_reproducible_and_inside_their_ranges():
    from protest_simulation.intervention_priors import LEVER_PRIORS, SUPPRESSION_PRIORS, mapping_for_run

    assert mapping_for_run(4) == mapping_for_run(4)
    for lever, values in {**LEVER_PRIORS, **SUPPRESSION_PRIORS}.items():
        for name, (low, high, reference) in values.items():
            assert low <= reference <= high
            assert low <= mapping_for_run(4)[lever][name] <= high
