"""The v3 protest simulator.

Each agent decides each day whether to take part. Net motivation is
    grievance  = w_m * L(material change of the agent's eligibility segment)
               + w_s * identity strength * (recognition threat felt + martyr effect, if threatened)
    social pull = beta_n * tanh(neighbourhood turnout / n*) + beta_v * tanh(identity-group turnout / v*)
    organization = amplifier * capacity(identity group) * mobilization(day) * local(district)
minus the agent's threshold, which rises by the fatigue increment each day the agent protests. The agent protests with
probability logistic(net / tau), but only if it has a stake (it feels a recognition threat or faces a material loss) and
has heard of the protest. Without the stake rule, the low tail of the threshold distribution and the organizational
push put tens of lakhs of unaffected people on the street in every episode (a Gujjar blockade in Rajasthan mobilized
SC agents in Kerala), which set a floor under every episode's size and spread protest by population.

Awareness: on the announcement day a share of the stakeholders (initial awareness) has heard of the call. Each day an
unaware stakeholder hears of it with probability 1 - exp(-d * (tanh(neighbourhood turnout / n*) + tanh(identity-group
turnout / v*))): people learn of a protest by seeing it nearby or among their own. This lets an agitation build up
over days, as the Jat (2016) and Gujjar (2019) agitations did, instead of peaking on its first day. Recognition threat is set per identity group (SC, ST, OBC, General, and the state
communities), with the most-deprived SC/ST tier feeling a share of its group's threat and the better-off tier a
multiple. local(district) = exp(kappa . standardized district covariates), renormalized to a population mean of 1, so
the covariates move protest between districts without changing its national level.

Deaths are negative binomial. Their mean is the death rate times protester-days, weighted in each state by
(share of the state on the street / 1%)^eta: violence comes with concentrated agitation more than with a thin,
countrywide bandh. Each death adds to the threat felt by the groups on the street (the martyr effect). Concession is an option, off by default: no episode identifies it.

The simulator returns daily turnout by state and identity group, protester-days by district and deaths, which the
observation model turns into expected news-event counts.
"""
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .population import IDENTITY_GROUP_COUNT, V3Population

PEOPLE_PER_CRORE = 1e7
INTENSITY_PATH = Path(__file__).resolve().parent.parent / "data" / "derived" / "gdelt_state_reporting_intensity.csv"


@lru_cache
def _intensity_table() -> dict:
    table = pd.read_csv(INTENSITY_PATH)
    return dict(zip(table.state, table.all_events_intensity))


def reporting_intensity(state_names) -> np.ndarray:
    """GDELT events per head in each state relative to India (any topic); 1 where a state has no figure."""
    table = _intensity_table()
    return np.array([max(table.get(name, 1.0), 0.02) for name in state_names])


def _default_capacity():
    # SC, ST, OBC, General, then the five communities (one shared value, estimated).
    return np.array([0.6, 0.4, 0.4, 0.1] + [0.5] * (IDENTITY_GROUP_COUNT - 4))


@dataclass
class V3Parameters:
    loss_aversion: float = 2.25
    material_weight: float = 0.45
    recognition_weight: float = 3.3
    mean_threshold: float = 7.0
    threshold_spread: float = 1.6
    decision_noise: float = 0.3
    fatigue: float = 0.10
    neighbourhood_influence: float = 0.9
    neighbourhood_saturation: float = 0.10
    national_influence: float = 0.6
    national_saturation: float = 0.02
    party_amplifier: float = 1.0
    capacity: np.ndarray = field(default_factory=_default_capacity)
    mobilization_action_day: float = 1.0
    mobilization_other_day: float = 0.15
    covariate_effect: np.ndarray = field(default_factory=lambda: np.zeros(3))   # urban, literacy, phone
    action_day_death_rate: float = 40.0     # deaths per crore protester-days on action days
    other_day_death_share: float = 0.1
    death_dispersion: float = 2.0
    death_concentration_power: float = 0.0
    martyr_effect_per_death: float = 0.012
    martyr_effect_cap: float = 0.35
    initial_awareness: float = 1.0
    awareness_diffusion: float = 0.0
    concession: bool = False
    concession_max_daily_hazard: float = 0.15
    concession_pressure_midpoint: float = 1.0
    concession_pressure_width: float = 0.25
    concession_relief: float = 0.6

    def copy(self, **changes):
        return replace(self, capacity=self.capacity.copy(), covariate_effect=self.covariate_effect.copy(), **changes)


@dataclass
class Shock:
    """What the agents face: recognition threat by identity group, its tier pattern, and material change by segment."""
    threat: np.ndarray                                  # length IDENTITY_GROUP_COUNT
    deprived_tier_share: float = 0.8
    better_off_tier_factor: float = 1.0
    material_by_segment: dict = None                    # segment name -> change (positive = loss); None = none
    deprived_tier_material_gain: float = 0.0


@dataclass
class Outcome:
    daily_by_state: np.ndarray        # days x states, protesters
    daily_by_identity: np.ndarray     # days x identity groups, protesters
    district_protester_days: np.ndarray
    daily_deaths: np.ndarray
    daily_expected_deaths: np.ndarray
    unique_participants: float
    conceded_on_day: int = None

    @property
    def daily(self):
        return self.daily_by_state.sum(axis=1)

    @property
    def peak(self):
        return float(self.daily.max())


def agent_threat(population: V3Population, shock: Shock) -> np.ndarray:
    base = population.base
    tier = np.where(base.is_most_deprived_tier, shock.deprived_tier_share, np.where(base.is_sc_or_st, shock.better_off_tier_factor, 1.0))
    return np.asarray(shock.threat, float)[population.identity_group] * tier


def agent_material(population: V3Population, shock: Shock, loss_aversion: float) -> np.ndarray:
    from protest_simulation.protest_campaign import SEGMENT_NAMES, eligibility_segment_of_each_agent
    base = population.base
    material = np.zeros(population.agent_count)
    if shock.material_by_segment:
        table = np.array([shock.material_by_segment.get(name, 0.0) for name in SEGMENT_NAMES], float)
        material = table[eligibility_segment_of_each_agent(base)]
    material = material - shock.deprived_tier_material_gain * (base.is_sc_or_st & base.is_most_deprived_tier)
    return np.where(material > 0, loss_aversion * material, material)


def local_mobilization(population: V3Population, parameters: V3Parameters) -> np.ndarray:
    local = np.exp(population.covariates @ parameters.covariate_effect)
    return local / local.mean()


def simulate(population: V3Population, parameters: V3Parameters, shock: Shock, days: int, action_days,
             random_generator: np.random.Generator) -> Outcome:
    base = population.base
    n = population.agent_count
    identity = population.identity_group
    agent_rng, event_rng = (np.random.default_rng(seed) for seed in random_generator.integers(0, 2 ** 63 - 1, 2))
    threat = agent_threat(population, shock)
    threatened = threat > 0
    material = agent_material(population, shock, parameters.loss_aversion)
    stake = threatened | (material > 0)
    aware = stake & (agent_rng.random(n) < parameters.initial_awareness)
    threshold = parameters.mean_threshold + parameters.threshold_spread * base.threshold_standard_score
    push = parameters.party_amplifier * parameters.capacity[identity] * local_mobilization(population, parameters)
    hood = base.neighbourhood
    hood_size = np.maximum(np.bincount(hood, minlength=base.neighbourhood_count), 1)
    identity_size = np.maximum(np.bincount(identity, minlength=IDENTITY_GROUP_COUNT), 1)
    state_count = len(population.state_names)
    people = population.people_per_agent
    action = set(int(day) for day in action_days)
    state_people = np.maximum(np.bincount(population.agent_state, minlength=state_count) * people, 1.0)

    hood_turnout = np.zeros(base.neighbourhood_count)
    identity_turnout = np.zeros(IDENTITY_GROUP_COUNT)
    martyr = np.zeros(IDENTITY_GROUP_COUNT)
    ever = np.zeros(n, bool)
    district_days = np.zeros(len(base.district_table))
    by_state, by_identity, deaths, expected_deaths = [], [], [], []
    conceded, largest, deaths_so_far = None, 0.0, 0
    for day in range(days):
        is_action = day in action and conceded is None
        mobilization = parameters.mobilization_action_day if is_action else parameters.mobilization_other_day
        grievance = (parameters.material_weight * material
                     + parameters.recognition_weight * base.identity_strength * (threat + martyr[identity] * threatened))
        pull = (parameters.neighbourhood_influence * np.tanh(hood_turnout[hood] / parameters.neighbourhood_saturation)
                + parameters.national_influence * np.tanh(identity_turnout[identity] / parameters.national_saturation))
        net = grievance + pull + push * mobilization - threshold
        protests = aware & (agent_rng.random(n) < 1 / (1 + np.exp(-net / parameters.decision_noise)))
        ever |= protests
        threshold = threshold + parameters.fatigue * protests
        hood_turnout = np.bincount(hood, weights=protests, minlength=base.neighbourhood_count) / hood_size
        counts = np.bincount(identity, weights=protests, minlength=IDENTITY_GROUP_COUNT)
        identity_turnout = counts / identity_size
        if parameters.awareness_diffusion > 0:
            exposure = (np.tanh(hood_turnout[hood] / parameters.neighbourhood_saturation)
                        + np.tanh(identity_turnout[identity] / parameters.national_saturation))
            aware |= stake & (agent_rng.random(n) < 1 - np.exp(-parameters.awareness_diffusion * exposure))
        state_today = np.bincount(population.agent_state, weights=protests, minlength=state_count) * people
        district_days += np.bincount(base.district, weights=protests, minlength=len(district_days)) * people
        total = state_today.sum()
        rate = parameters.action_day_death_rate * (1.0 if is_action else parameters.other_day_death_share)
        intensity = (state_today / state_people / 0.01) ** parameters.death_concentration_power
        mean_deaths = rate * float((state_today * intensity).sum()) / PEOPLE_PER_CRORE
        died = int(event_rng.poisson(event_rng.gamma(parameters.death_dispersion, mean_deaths / parameters.death_dispersion))) if mean_deaths > 0 else 0
        if died:
            share = counts / max(counts.sum(), 1)
            martyr = np.minimum(martyr + parameters.martyr_effect_per_death * died * share * IDENTITY_GROUP_COUNT, parameters.martyr_effect_cap)
        deaths_so_far += died
        largest = max(largest, total)
        if parameters.concession and conceded is None:
            pressure = largest / PEOPLE_PER_CRORE + deaths_so_far / 20.0
            hazard = parameters.concession_max_daily_hazard / (1 + np.exp(-(pressure - parameters.concession_pressure_midpoint) / parameters.concession_pressure_width))
            if event_rng.random() < hazard:
                conceded = day + 1
                threat = threat * (1 - parameters.concession_relief)
                material = material * (1 - parameters.concession_relief)
                threatened = threat > 0
        by_state.append(state_today)
        by_identity.append(counts * people)
        deaths.append(died)
        expected_deaths.append(mean_deaths)
    return Outcome(daily_by_state=np.array(by_state), daily_by_identity=np.array(by_identity), district_protester_days=district_days,
                   daily_deaths=np.array(deaths), daily_expected_deaths=np.array(expected_deaths), unique_participants=float(ever.sum() * people), conceded_on_day=conceded)


def expected_events(daily_by_state: np.ndarray, reporting_intensity: np.ndarray, scale_log10: float, exponent: float,
                    reporting_power: float) -> np.ndarray:
    """Observation model: expected news-reported events on each day and state,
    alpha * intensity_s^delta * (protesters_s / 1 lakh)^gamma, where intensity_s is how heavily GDELT reports state s
    per head on any topic (data/derived/gdelt_state_reporting_intensity.csv). Returns a days x states array."""
    weights = (np.maximum(daily_by_state, 0.0) / 1e5) ** exponent
    return 10 ** scale_log10 * weights * np.asarray(reporting_intensity, float) ** reporting_power
