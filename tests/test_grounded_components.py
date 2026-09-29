"""Tests for the data-grounded components: geography, merit allocation, episode simulation and derived data files."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from merit_allocation.choice_rules import over_and_above, reserved_first
from merit_allocation.jee_advanced_data import CATEGORIES, VALIDATED_YEARS, read_rank_table
from merit_allocation.latent_merit import LatentMeritFit
from merit_allocation.merged_pool import IncomeOverlay, SEGMENTS, allocate
from protest_simulation import BASELINE_ABRUPT_INCOME_ONLY_SWITCH, simulate_protest_campaign
from protest_simulation.episode_simulation import EPISODE_SHOCKS, episode_parameters, party_amplifier, turnout_by_state
from protest_simulation.geography import build_synthetic_india_by_district, district_table, state_names
from protest_simulation.synthetic_population import GENERAL, SC, ST

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def district_population():
    return build_synthetic_india_by_district(40_000, np.random.default_rng(7), same_group_neighbourhood_share=0.8)


# Geography ------------------------------------------------------------------------------------------------------------

def test_district_table_shares_are_valid_and_match_census_totals():
    table = district_table()
    shares = table[["share_sc", "share_st", "share_obc", "share_general"]].to_numpy()
    assert (shares >= -1e-12).all() and np.allclose(shares.sum(axis=1), 1)
    assert table.sc.sum() / table.population.sum() == pytest.approx(0.166, abs=0.002)
    assert table.st.sum() / table.population.sum() == pytest.approx(0.086, abs=0.002)
    assert len(table) == 640 and (table.state == "Telangana").sum() == 10


def test_district_population_reproduces_state_composition(district_population):
    names = state_names(district_population)
    agent_state = district_population.state_of_district[district_population.district]
    punjab = agent_state == names.index("Punjab")
    assert (district_population.social_group[punjab] == SC).mean() == pytest.approx(0.319, abs=0.04)
    mizoram = agent_state == names.index("Mizoram")
    assert (district_population.social_group[mizoram] == ST).mean() > 0.85


def test_neighbourhoods_stay_within_a_state(district_population):
    agent_state = district_population.state_of_district[district_population.district]
    for neighbourhood in range(0, district_population.neighbourhood_count, 5):
        assert np.unique(agent_state[district_population.neighbourhood == neighbourhood]).size == 1


def test_district_model_runs_and_state_turnout_adds_up(district_population):
    parameters = episode_parameters(BASELINE_ABRUPT_INCOME_ONLY_SWITCH, EPISODE_SHOCKS["sc_st_bharat_bandh_2018"], 1.2)
    outcome = simulate_protest_campaign(district_population, parameters, np.random.default_rng(1), record_agent_protest_days=True)
    assert turnout_by_state(district_population, outcome).sum() == pytest.approx(outcome.protester_days)
    assert outcome.bandh_days == [1]


# Episode simulation ---------------------------------------------------------------------------------------------------

def test_party_backing_maps_onto_the_amplifier():
    assert party_amplifier(0.0) == pytest.approx(1.0)
    assert party_amplifier(0.6) == pytest.approx(1.5)
    assert party_amplifier(0.2) < party_amplifier(0.6)


def test_episode_parameters_remove_material_loss_and_scale_the_threat():
    parameters = episode_parameters(BASELINE_ABRUPT_INCOME_ONLY_SWITCH, EPISODE_SHOCKS["upper_caste_bandh_2018"], 0.5)
    assert parameters.material_loss_sc_st_above_income_line == 0 and parameters.material_loss_general_below_income_line == 0
    assert parameters.symbolic_threat_by_group[GENERAL] == pytest.approx(0.5)
    assert parameters.symbolic_threat_by_group[SC] < 0


# Merit allocation -----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("year", VALIDATED_YEARS)
def test_extracted_rank_lists_are_internally_consistent(year):
    ranks, categories, list_sizes = read_rank_table(year)
    assert ranks.min() == 1 and np.all(np.diff(ranks) >= 0)
    rows = list(csv.DictReader(open(ROOT / "data" / "derived" / f"jee_advanced_ranks_{year}.csv")))
    for category in CATEGORIES[1:]:
        joined = sorted((int(r["category_rank"]), int(r["crl_rank"])) for r in rows if r["category"] == category and r["crl_rank"])
        crl_in_category_order = [crl for _, crl in joined]
        inversions = sum(a > b for a, b in zip(crl_in_category_order, crl_in_category_order[1:]))
        # A correct join orders both lists by the same marks. 2021 GEN-EWS has two inversions in 4,329 (likely two
        # misread roll numbers); anything above 0.1% would indicate a broken extraction.
        assert inversions <= 0.001 * len(joined), (year, category, inversions)


def test_2023_rank_lists_match_the_sizes_the_report_states():
    _, _, sizes = read_rank_table(2023)
    ranks, _, _ = read_rank_table(2023)
    assert len(ranks) == 26321
    assert sizes == {"GEN-EWS": 5371, "OBC-NCL": 9049, "SC": 5004, "ST": 1645}


def test_reserved_first_differs_from_open_first_only_in_order():
    weights = np.ones(6)
    eligibility = {"R": np.array([1, 0, 1, 0, 1, 0], float)}
    open_a, reserve_a = over_and_above(weights, eligibility, 2, {"R": 2})
    open_b, reserve_b = reserved_first(weights, eligibility, 2, {"R": 2})
    assert (open_a + reserve_a["R"]).sum() == (open_b + reserve_b["R"]).sum() == 4
    assert reserve_b["R"].tolist() == [1, 0, 1, 0, 0, 0]


@pytest.mark.parametrize("order", ["open_first", "reserved_first"])
def test_merged_pool_allocates_every_seat_in_both_regimes(order):
    fit = LatentMeritFit(2023, np.array([39675., 30101., 67285., 29551., 13760.]), np.array([0, -0.725, -1.003, -1.755, -2.337]),
                         np.array([1, 1.03, 1.105, 1.179, 1.345]), 0, 26321, False)
    overlay = IncomeOverlay({"GEN": 0.85, "GEN-EWS": 0.0, "OBC-NCL": 0.25, "SC": 0.25, "ST": 0.25}, 0.4)
    result = allocate(fit, overlay, 17340.0, 0.595, order)
    assert sum(result["status_quo"].values()) == pytest.approx(17340.0, rel=1e-6)
    assert sum(result["reform"].values()) == pytest.approx(17340.0, rel=1e-6)
    assert set(result["reform"]) == set(SEGMENTS)
    # Above-line candidates cannot hold income-pool seats, so their reform seats come from open seats only.
    assert result["reform"]["SC_above"] <= result["status_quo"]["SC_above"]


# Derived data ---------------------------------------------------------------------------------------------------------

def test_episode_targets_are_present_and_ordered_as_reported():
    targets = json.loads((ROOT / "data" / "derived" / "episode_targets.json").read_text())
    rate = {key: value["corrected_core_events_per_1000_india_events"]["median"] for key, value in targets.items()}
    assert rate["sc_st_bharat_bandh_2018"] > rate["sc_st_bharat_bandh_2024"] > rate["ews_quota_2019"]
    assert targets["sc_st_bharat_bandh_2018"]["deaths"]["low"] == 11


def test_relevance_audit_labels_are_binary_or_missing():
    rows = list(csv.DictReader(open(ROOT / "data" / "derived" / "gdelt_relevance_audit.csv")))
    assert len(rows) > 200
    assert {row["relevant"] for row in rows} <= {"0.0", "1.0", ""}


def test_allocation_rule_levers_shift_segment_table_and_packages_apply_them_first():
    from protest_simulation.intervention_priors import apply_lever, interventions_for, reference_mapping
    from protest_simulation.model_parameters import BASELINE_ABRUPT_INCOME_ONLY_SWITCH
    from protest_simulation.policy_interventions import expand_seats_so_no_group_loses

    reform = {"SC_above": 1.0, "SC_below": 0.6, "ST_above": 1.1, "ST_below": 0.8, "OBC_creamy": 0.2, "OBC_above_ncl": 0.7,
              "OBC_below": -0.2, "General_above": 0.2, "General_below_ews": -0.4, "General_below_asset_excluded": -2.0}
    expanded = dict(reform, SC_below=0.3, OBC_below=-0.5)
    parameters = BASELINE_ABRUPT_INCOME_ONLY_SWITCH.copy()
    parameters.material_change_by_segment = dict(reform)
    parameters.reform_material_change_by_segment = dict(reform)
    parameters.lever_material_tables = {"seat_expansion": expanded}

    catalogue = parameters.copy()
    expand_seats_so_no_group_loses(catalogue)
    assert all(abs(catalogue.material_change_by_segment[k] - expanded[k]) < 1e-12 for k in expanded)

    mapping = reference_mapping()
    mapping["seat_expansion"]["below_line_loss_removed"] = 0.5
    halfway = parameters.copy()
    apply_lever(halfway, "seat_expansion", mapping)
    assert abs(halfway.material_change_by_segment["SC_below"] - 0.45) < 1e-12
    assert halfway.material_change_by_segment["SC_above"] == 1.0

    order = [f.__defaults__ for f in interventions_for("managed_transition", reference_mapping())]
    assert order[0][0] == "seat_expansion"
