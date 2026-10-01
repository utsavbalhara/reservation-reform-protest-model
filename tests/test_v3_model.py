import numpy as np
import pytest

from protest_v3.episodes import EPISODE_KEYS, episode_threat, load_targets, schedule
from protest_v3.model import Shock, V3Parameters, expected_events, local_mobilization, simulate
from protest_v3.population import COMMUNITIES, COMMUNITY_NAMES, IDENTITY_GROUP_COUNT, build_population

AGENTS = 30_000


@pytest.fixture(scope="module")
def population():
    return build_population(AGENTS, 0.8)


def test_communities_live_in_their_state_and_parent_group(population):
    base = population.base
    for index, name in enumerate(COMMUNITY_NAMES):
        state, parent, share = COMMUNITIES[name]
        members = population.community == index
        assert members.any()
        assert {population.state_names[s] for s in np.unique(population.agent_state[members])} == {state}
        assert np.all(base.social_group[members] == parent)
        in_state = population.agent_state == population.state_names.index(state)
        assert abs(members.sum() / in_state.sum() - share) < 0.02


def test_local_mobilization_has_mean_one(population):
    parameters = V3Parameters(covariate_effect=np.array([0.8, -0.3, 0.5]))
    assert local_mobilization(population, parameters).mean() == pytest.approx(1.0)


def test_no_threat_no_protest_and_more_threat_more_protest(population):
    parameters = V3Parameters()
    quiet = simulate(population, parameters, Shock(np.zeros(IDENTITY_GROUP_COUNT)), 3, (1,), np.random.default_rng(1))
    assert quiet.daily.sum() < 0.002 * 146e7
    peaks = []
    for magnitude in (0.5, 1.0, 1.5):
        outcome = simulate(population, parameters, Shock(episode_threat("sc_st_bharat_bandh_2018", magnitude)), 3, (1,), np.random.default_rng(1))
        peaks.append(outcome.peak)
    assert peaks[0] < peaks[1] < peaks[2]


def test_simulation_is_deterministic_given_seed(population):
    shock = Shock(episode_threat("jat_2016", 2.0))
    a = simulate(population, V3Parameters(), shock, 5, (1, 2, 3, 4), np.random.default_rng(3))
    b = simulate(population, V3Parameters(), shock, 5, (1, 2, 3, 4), np.random.default_rng(3))
    assert np.array_equal(a.daily_by_state, b.daily_by_state)


def test_community_episode_stays_in_its_state(population):
    outcome = simulate(population, V3Parameters(), Shock(episode_threat("jat_2016", 2.5)), 5, (1, 2, 3, 4), np.random.default_rng(2))
    by_state = outcome.daily_by_state.sum(axis=0)
    assert by_state[population.state_names.index("Haryana")] > 0.9 * by_state.sum()


def test_observation_model_is_increasing_and_counts_delhi_more(population):
    states = population.state_names
    turnout = np.zeros((1, len(states)))
    turnout[0, states.index("Bihar")] = 1e6
    low = expected_events(turnout, states, -1.0, 0.7, 3.0).sum()
    high = expected_events(turnout * 2, states, -1.0, 0.7, 3.0).sum()
    assert high > low
    delhi = np.zeros((1, len(states)))
    delhi[0, states.index("Delhi")] = 1e6
    assert expected_events(delhi, states, -1.0, 0.7, 3.0).sum() == pytest.approx(3.0 * low)


def test_every_episode_has_a_schedule_matching_its_core_days():
    targets = load_targets()
    for key in EPISODE_KEYS:
        plan = schedule(targets[key])
        assert len(plan.action_days) == len(targets[key]["daily_core_events"])
        assert plan.days == plan.action_days[-1] + 1
