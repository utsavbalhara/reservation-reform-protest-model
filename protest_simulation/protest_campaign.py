from dataclasses import dataclass, field

import numpy as np
from scipy.special import ndtr

from .model_parameters import ProtestModelParameters
from .synthetic_population import GENERAL, OBC, SOCIAL_GROUP_COUNT, SyntheticPopulation

PEOPLE_PER_CRORE = 1e7
ACTIVE_DAY_SHARE_OF_PEAK = 0.10

# Eligibility segments. The reference population only produces the first-listed segment of each group pair
# (every above-line OBC family is creamy layer; every below-line General family passes the EWS asset test).
SEGMENT_NAMES = ("SC_above", "SC_below", "ST_above", "ST_below", "OBC_creamy", "OBC_above_ncl", "OBC_below",
                 "General_above", "General_below_ews", "General_below_asset_excluded")


@dataclass
class CampaignOutcome:
    peak_day_protesters: float
    cumulative_unique_protesters: float
    total_deaths: int
    daily_protesters: np.ndarray
    daily_protesters_by_group: np.ndarray
    daily_deaths: np.ndarray
    daily_neighbourhood_turnout: list = field(default_factory=list)
    bandh_days: list = field(default_factory=list)
    conceded_on_day: int = None
    campaign_days_run: int = None
    daily_police_deaths: np.ndarray = None
    agent_protest_days: np.ndarray = None

    @property
    def protester_days(self) -> float:
        return float(self.daily_protesters.sum())


def eligibility_segment_of_each_agent(population: SyntheticPopulation) -> np.ndarray:
    group, above = population.social_group, population.is_above_income_line
    creamy = population.is_obc_creamy_layer if population.is_obc_creamy_layer is not None else above
    asset_excluded = (population.is_asset_excluded_from_ews if population.is_asset_excluded_from_ews is not None
                      else np.zeros(population.agent_count, bool))
    segment = np.full(population.agent_count, -1)
    segment[(group == 0) & above] = SEGMENT_NAMES.index("SC_above")
    segment[(group == 0) & ~above] = SEGMENT_NAMES.index("SC_below")
    segment[(group == 1) & above] = SEGMENT_NAMES.index("ST_above")
    segment[(group == 1) & ~above] = SEGMENT_NAMES.index("ST_below")
    segment[(group == OBC) & above & creamy] = SEGMENT_NAMES.index("OBC_creamy")
    segment[(group == OBC) & above & ~creamy] = SEGMENT_NAMES.index("OBC_above_ncl")
    segment[(group == OBC) & ~above] = SEGMENT_NAMES.index("OBC_below")
    segment[(group == GENERAL) & above] = SEGMENT_NAMES.index("General_above")
    segment[(group == GENERAL) & ~above & ~asset_excluded] = SEGMENT_NAMES.index("General_below_ews")
    segment[(group == GENERAL) & ~above & asset_excluded] = SEGMENT_NAMES.index("General_below_asset_excluded")
    return segment


def material_loss_felt_by_each_agent(population: SyntheticPopulation, parameters: ProtestModelParameters) -> np.ndarray:
    if parameters.material_change_by_segment is None:
        material_loss = np.zeros(population.agent_count)
        sc_or_st = population.is_sc_or_st
        above_line = population.is_above_income_line
        material_loss[sc_or_st & above_line] = parameters.material_loss_sc_st_above_income_line
        material_loss[sc_or_st & ~above_line] = parameters.material_loss_sc_st_below_income_line
        material_loss[(population.social_group == OBC) & ~above_line] = parameters.material_loss_obc_below_income_line
        material_loss[(population.social_group == GENERAL) & ~above_line] = parameters.material_loss_general_below_income_line
    else:
        by_segment = np.array([parameters.material_change_by_segment.get(name, 0.0) for name in SEGMENT_NAMES], float)
        material_loss = by_segment[eligibility_segment_of_each_agent(population)]
    material_loss[population.is_sc_or_st & population.is_most_deprived_tier] -= parameters.sub_classification_gain_for_most_deprived_tier
    losses_weighted_more_than_gains = np.where(material_loss > 0, parameters.loss_aversion * material_loss, material_loss)
    return losses_weighted_more_than_gains


def symbolic_threat_felt_by_each_agent(population: SyntheticPopulation, parameters: ProtestModelParameters) -> np.ndarray:
    tier_share = np.where(population.is_most_deprived_tier, parameters.most_deprived_tier_share_of_symbolic_threat,
                          np.where(population.is_sc_or_st, parameters.better_off_tier_symbolic_threat_factor, 1.0))
    return parameters.symbolic_threat_by_group[population.social_group] * tier_share


def threshold_scores(population: SyntheticPopulation, parameters: ProtestModelParameters) -> np.ndarray:
    """Map each agent's standard-normal score onto the chosen distribution, keeping agents in the same order."""
    z = population.threshold_standard_score
    if parameters.threshold_distribution == "normal":
        return z
    u = np.clip(ndtr(z), 1e-12, 1 - 1e-12)
    if parameters.threshold_distribution == "logistic":
        return np.log(u / (1 - u)) * np.sqrt(3) / np.pi
    if parameters.threshold_distribution == "activist_mixture":
        share, offset, spread = parameters.activist_core_share, parameters.activist_core_offset, parameters.activist_core_spread
        grid = np.linspace(-14, 8, 20001)
        mixture_cdf = (1 - share) * ndtr(grid) + share * ndtr((grid + offset) / spread)
        x = np.interp(u, mixture_cdf, grid)
        mean = -share * offset
        variance = (1 - share) * 1.0 + share * (spread ** 2 + offset ** 2) - mean ** 2
        return (x - mean) / np.sqrt(variance)
    raise ValueError(f"unknown threshold distribution {parameters.threshold_distribution!r}")


def negative_binomial(random_generator, mean: float, size: float) -> int:
    """Gamma-Poisson draw with the given mean and size parameter: variance = mean + mean^2 / size."""
    if mean <= 0:
        return 0
    return int(random_generator.poisson(random_generator.gamma(size, mean / size)))


def simulate_protest_campaign(
    population: SyntheticPopulation,
    parameters: ProtestModelParameters,
    random_generator: np.random.Generator,
    record_neighbourhood_turnout: bool = False,
    record_agent_protest_days: bool = False,
) -> CampaignOutcome:
    group = population.social_group
    days = parameters.campaign_length_days
    if parameters.random_streams == "split":
        agent_seed, event_seed, organization_seed = random_generator.integers(0, 2 ** 63 - 1, size=3)
        agent_rng, event_rng = np.random.default_rng(agent_seed), np.random.default_rng(event_seed)
        organization_rng = np.random.default_rng(organization_seed)
    else:
        agent_rng = event_rng = organization_rng = random_generator

    felt_material_loss = material_loss_felt_by_each_agent(population, parameters)
    felt_symbolic_threat = symbolic_threat_felt_by_each_agent(population, parameters)
    feels_threatened = felt_symbolic_threat > 0
    participation_threshold = (
        parameters.mean_participation_threshold
        + parameters.participation_threshold_spread * threshold_scores(population, parameters)
    )
    organizational_push_by_group = parameters.organizational_capacity_by_group * parameters.opposition_party_amplifier

    neighbourhood_influence = parameters.max_neighbourhood_influence
    national_influence = parameters.max_national_visibility_influence
    bandh_mobilization = parameters.mobilization_on_bandh_days
    death_rate = parameters.deaths_per_crore_protester_days
    other_death_multiplier = parameters.other_death_multiplier
    if parameters.internet_shutdown_active:
        neighbourhood_influence *= parameters.internet_shutdown_coordination_factor
        national_influence *= parameters.internet_shutdown_coordination_factor
        bandh_mobilization *= parameters.internet_shutdown_bandh_mobilization_factor
        if parameters.death_model == "poisson":
            death_rate *= parameters.internet_shutdown_violence_factor
        else:
            other_death_multiplier *= parameters.internet_shutdown_violence_factor

    endogenous_bandhs = parameters.bandh_schedule == "endogenous"
    if endogenous_bandhs:
        low, high = parameters.bandh_refractory_days
        refractory_gaps = organization_rng.integers(low, high + 1, size=parameters.max_bandh_calls)
    concession_uniforms = organization_rng.random(days) if parameters.concession_rule else None

    neighbourhood_size = np.bincount(population.neighbourhood, minlength=population.neighbourhood_count)
    has_ever_protested = np.zeros(population.agent_count, bool)
    protest_days = np.zeros(population.agent_count, np.int16) if record_agent_protest_days else None
    neighbourhood_turnout = np.zeros(population.neighbourhood_count)
    group_turnout = np.zeros(SOCIAL_GROUP_COUNT)
    martyrdom_rise_by_group = np.zeros(SOCIAL_GROUP_COUNT)
    people_per_agent = population.people_represented_per_agent
    group_members = [group == g for g in range(SOCIAL_GROUP_COUNT)]

    scheduled_bandh_day = None
    bandh_calls_made, next_call_check_day = 0, parameters.first_bandh_call_day - 1
    conceded_on_day, largest_turnout_so_far, deaths_so_far = None, 0.0, 0
    bandh_days = []

    def net_motivation_of_everyone(mobilization_today, is_bandh_day):
        grievance = (
            parameters.material_loss_weight * felt_material_loss
            + parameters.symbolic_threat_weight
            * population.identity_strength
            * (felt_symbolic_threat + martyrdom_rise_by_group[group] * feels_threatened)
        )
        if parameters.social_influence_saturates:
            social_pull = neighbourhood_influence * np.tanh(
                neighbourhood_turnout[population.neighbourhood] / parameters.neighbourhood_turnout_at_saturation
            ) + national_influence * np.tanh(group_turnout[group] / parameters.national_turnout_at_saturation)
        else:
            social_pull = (
                neighbourhood_influence * neighbourhood_turnout[population.neighbourhood]
                + national_influence * group_turnout[group]
            )
        net = grievance + social_pull + organizational_push_by_group[group] * mobilization_today - participation_threshold
        if is_bandh_day:
            net -= parameters.heavy_policing_turnout_cost
        return net

    daily_protesters, daily_protesters_by_group, daily_deaths, daily_police_deaths, daily_neighbourhood_turnout = [], [], [], [], []
    for day in range(days):
        if endogenous_bandhs:
            is_bandh_day = day == scheduled_bandh_day
        else:
            is_bandh_day = day in parameters.bandh_call_days and conceded_on_day is None
        if is_bandh_day:
            bandh_days.append(day)
        mobilization_today = bandh_mobilization if is_bandh_day else parameters.mobilization_on_ordinary_days

        net_motivation = net_motivation_of_everyone(mobilization_today, is_bandh_day)
        chance_of_protesting = 1 / (1 + np.exp(-net_motivation / parameters.decision_noise))
        protests_today = agent_rng.random(population.agent_count) < chance_of_protesting
        has_ever_protested |= protests_today
        if protest_days is not None:
            protest_days += protests_today
        participation_threshold = participation_threshold + parameters.fatigue_per_protest_day * protests_today

        protesters_in_neighbourhood = np.bincount(
            population.neighbourhood, weights=protests_today, minlength=population.neighbourhood_count
        )
        neighbourhood_turnout = protesters_in_neighbourhood / np.maximum(neighbourhood_size, 1)
        group_turnout = np.array([protests_today[group == g].mean() for g in range(SOCIAL_GROUP_COUNT)])
        protesters_today = protests_today.sum() * people_per_agent

        todays_death_rate = death_rate * parameters.bandh_day_death_rate_multiplier if is_bandh_day else death_rate
        if parameters.death_model == "poisson":
            deaths_today = event_rng.poisson(todays_death_rate * protesters_today / PEOPLE_PER_CRORE)
            police_deaths_today, martyr_weighted_deaths = 0, deaths_today
        elif parameters.death_model == "negative_binomial_split":
            base_mean = todays_death_rate * protesters_today / PEOPLE_PER_CRORE
            share = parameters.police_attributed_death_share
            police_deaths_today = negative_binomial(event_rng, base_mean * share * parameters.police_death_multiplier, parameters.death_dispersion)
            other_deaths_today = negative_binomial(event_rng, base_mean * (1 - share) * other_death_multiplier, parameters.death_dispersion)
            deaths_today = police_deaths_today + other_deaths_today
            martyr_weighted_deaths = police_deaths_today + parameters.martyr_weight_of_other_deaths * other_deaths_today
        else:
            raise ValueError(f"unknown death model {parameters.death_model!r}")

        if deaths_today:
            protester_share_by_group = np.array(
                [protests_today[group == g].sum() for g in range(SOCIAL_GROUP_COUNT)], float
            )
            protester_share_by_group /= max(protester_share_by_group.sum(), 1)
            if parameters.death_model == "poisson" and parameters.death_response == 1.0:
                martyrdom_rise_by_group = np.minimum(
                    martyrdom_rise_by_group
                    + parameters.symbolic_threat_rise_per_death * deaths_today * protester_share_by_group * SOCIAL_GROUP_COUNT,
                    parameters.max_martyrdom_symbolic_rise,
                )
            else:
                martyrdom_rise_by_group = np.clip(
                    martyrdom_rise_by_group
                    + parameters.death_response * parameters.symbolic_threat_rise_per_death * martyr_weighted_deaths
                    * protester_share_by_group * SOCIAL_GROUP_COUNT,
                    -parameters.max_martyrdom_symbolic_rise, parameters.max_martyrdom_symbolic_rise,
                )

        deaths_so_far += int(deaths_today)
        largest_turnout_so_far = max(largest_turnout_so_far, protesters_today)
        if parameters.concession_rule and conceded_on_day is None:
            pressure = largest_turnout_so_far / PEOPLE_PER_CRORE + deaths_so_far / parameters.concession_deaths_scale
            hazard = parameters.concession_max_daily_hazard / (
                1 + np.exp(-(pressure - parameters.concession_pressure_midpoint) / parameters.concession_pressure_width))
            if concession_uniforms[day] < hazard:
                conceded_on_day = day + 1
                relief = 1 - parameters.concession_grievance_relief
                felt_material_loss = felt_material_loss * relief
                felt_symbolic_threat = np.where(group == GENERAL, parameters.counter_mobilization_symbolic_threat, felt_symbolic_threat * relief)
                feels_threatened = felt_symbolic_threat > 0
                scheduled_bandh_day = None

        if endogenous_bandhs and conceded_on_day is None and bandh_calls_made < parameters.max_bandh_calls:
            if is_bandh_day:
                bandh_calls_made += 1
                scheduled_bandh_day = None
                if bandh_calls_made < parameters.max_bandh_calls:
                    next_call_check_day = day + int(refractory_gaps[bandh_calls_made - 1]) - 1
            elif scheduled_bandh_day is None and day >= next_call_check_day and bandh_calls_made < parameters.max_bandh_calls:
                expected = 1 / (1 + np.exp(-net_motivation_of_everyone(bandh_mobilization, True) / parameters.decision_noise))
                if expected.sum() * people_per_agent >= parameters.bandh_call_expected_turnout:
                    scheduled_bandh_day = day + 1

        daily_protesters.append(protesters_today)
        daily_protesters_by_group.append(
            [protests_today[group == g].sum() * people_per_agent for g in range(SOCIAL_GROUP_COUNT)]
        )
        daily_deaths.append(deaths_today)
        daily_police_deaths.append(police_deaths_today)
        if record_neighbourhood_turnout:
            daily_neighbourhood_turnout.append(neighbourhood_turnout.copy())

    daily_protesters = np.array(daily_protesters)
    peak = float(daily_protesters.max())
    active_days = np.nonzero(daily_protesters >= ACTIVE_DAY_SHARE_OF_PEAK * peak)[0] if peak > 0 else np.array([-1])
    return CampaignOutcome(
        peak_day_protesters=peak,
        cumulative_unique_protesters=float(has_ever_protested.sum() * people_per_agent),
        total_deaths=int(np.sum(daily_deaths)),
        daily_protesters=daily_protesters,
        daily_protesters_by_group=np.array(daily_protesters_by_group),
        daily_deaths=np.array(daily_deaths),
        daily_neighbourhood_turnout=daily_neighbourhood_turnout,
        bandh_days=bandh_days,
        conceded_on_day=conceded_on_day,
        campaign_days_run=int(active_days[-1]) + 1,
        daily_police_deaths=np.array(daily_police_deaths),
        agent_protest_days=protest_days,
    )
