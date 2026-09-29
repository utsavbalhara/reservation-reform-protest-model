from dataclasses import dataclass, field

import numpy as np

from .model_parameters import ProtestModelParameters
from .synthetic_population import GENERAL, OBC, SOCIAL_GROUP_COUNT, SyntheticPopulation

PEOPLE_PER_CRORE = 1e7


@dataclass
class CampaignOutcome:
    peak_day_protesters: float
    cumulative_unique_protesters: float
    total_deaths: int
    daily_protesters: np.ndarray
    daily_protesters_by_group: np.ndarray
    daily_deaths: np.ndarray
    daily_neighbourhood_turnout: list = field(default_factory=list)


def material_loss_felt_by_each_agent(population: SyntheticPopulation, parameters: ProtestModelParameters) -> np.ndarray:
    material_loss = np.zeros(population.agent_count)
    sc_or_st = population.is_sc_or_st
    above_line = population.is_above_income_line
    material_loss[sc_or_st & above_line] = parameters.material_loss_sc_st_above_income_line
    material_loss[sc_or_st & ~above_line] = parameters.material_loss_sc_st_below_income_line
    material_loss[(population.social_group == OBC) & ~above_line] = parameters.material_loss_obc_below_income_line
    material_loss[(population.social_group == GENERAL) & ~above_line] = parameters.material_loss_general_below_income_line
    material_loss[sc_or_st & population.is_most_deprived_tier] -= parameters.sub_classification_gain_for_most_deprived_tier
    losses_weighted_more_than_gains = np.where(material_loss > 0, parameters.loss_aversion * material_loss, material_loss)
    return losses_weighted_more_than_gains


def symbolic_threat_felt_by_each_agent(population: SyntheticPopulation, parameters: ProtestModelParameters) -> np.ndarray:
    tier_share = np.where(population.is_most_deprived_tier, parameters.most_deprived_tier_share_of_symbolic_threat, 1.0)
    return parameters.symbolic_threat_by_group[population.social_group] * tier_share


def simulate_protest_campaign(
    population: SyntheticPopulation,
    parameters: ProtestModelParameters,
    random_generator: np.random.Generator,
    record_neighbourhood_turnout: bool = False,
) -> CampaignOutcome:
    group = population.social_group
    felt_material_loss = material_loss_felt_by_each_agent(population, parameters)
    felt_symbolic_threat = symbolic_threat_felt_by_each_agent(population, parameters)
    feels_threatened = felt_symbolic_threat > 0
    participation_threshold = (
        parameters.mean_participation_threshold
        + parameters.participation_threshold_spread * population.threshold_standard_score
    )
    organizational_push_by_group = parameters.organizational_capacity_by_group * parameters.opposition_party_amplifier

    neighbourhood_influence = parameters.max_neighbourhood_influence
    national_influence = parameters.max_national_visibility_influence
    bandh_mobilization = parameters.mobilization_on_bandh_days
    death_rate = parameters.deaths_per_crore_protester_days
    if parameters.internet_shutdown_active:
        neighbourhood_influence *= parameters.internet_shutdown_coordination_factor
        national_influence *= parameters.internet_shutdown_coordination_factor
        bandh_mobilization *= parameters.internet_shutdown_bandh_mobilization_factor
        death_rate *= parameters.internet_shutdown_violence_factor

    neighbourhood_size = np.bincount(population.neighbourhood, minlength=population.neighbourhood_count)
    has_ever_protested = np.zeros(population.agent_count, bool)
    neighbourhood_turnout = np.zeros(population.neighbourhood_count)
    group_turnout = np.zeros(SOCIAL_GROUP_COUNT)
    martyrdom_rise_by_group = np.zeros(SOCIAL_GROUP_COUNT)
    people_per_agent = population.people_represented_per_agent

    daily_protesters, daily_protesters_by_group, daily_deaths, daily_neighbourhood_turnout = [], [], [], []
    for day in range(parameters.campaign_length_days):
        is_bandh_day = day in parameters.bandh_call_days
        mobilization_today = bandh_mobilization if is_bandh_day else parameters.mobilization_on_ordinary_days

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
        net_motivation = grievance + social_pull + organizational_push_by_group[group] * mobilization_today - participation_threshold
        if is_bandh_day:
            net_motivation -= parameters.heavy_policing_turnout_cost

        chance_of_protesting = 1 / (1 + np.exp(-net_motivation / parameters.decision_noise))
        protests_today = random_generator.random(population.agent_count) < chance_of_protesting
        has_ever_protested |= protests_today
        participation_threshold = participation_threshold + parameters.fatigue_per_protest_day * protests_today

        protesters_in_neighbourhood = np.bincount(
            population.neighbourhood, weights=protests_today, minlength=population.neighbourhood_count
        )
        neighbourhood_turnout = protesters_in_neighbourhood / np.maximum(neighbourhood_size, 1)
        group_turnout = np.array([protests_today[group == g].mean() for g in range(SOCIAL_GROUP_COUNT)])
        protesters_today = protests_today.sum() * people_per_agent

        deaths_today = random_generator.poisson(death_rate * protesters_today / PEOPLE_PER_CRORE)
        if deaths_today:
            protester_share_by_group = np.array(
                [protests_today[group == g].sum() for g in range(SOCIAL_GROUP_COUNT)], float
            )
            protester_share_by_group /= max(protester_share_by_group.sum(), 1)
            martyrdom_rise_by_group = np.minimum(
                martyrdom_rise_by_group
                + parameters.symbolic_threat_rise_per_death * deaths_today * protester_share_by_group * SOCIAL_GROUP_COUNT,
                parameters.max_martyrdom_symbolic_rise,
            )

        daily_protesters.append(protesters_today)
        daily_protesters_by_group.append(
            [protests_today[group == g].sum() * people_per_agent for g in range(SOCIAL_GROUP_COUNT)]
        )
        daily_deaths.append(deaths_today)
        if record_neighbourhood_turnout:
            daily_neighbourhood_turnout.append(neighbourhood_turnout.copy())

    daily_protesters = np.array(daily_protesters)
    return CampaignOutcome(
        peak_day_protesters=float(daily_protesters.max()),
        cumulative_unique_protesters=float(has_ever_protested.sum() * people_per_agent),
        total_deaths=int(np.sum(daily_deaths)),
        daily_protesters=daily_protesters,
        daily_protesters_by_group=np.array(daily_protesters_by_group),
        daily_deaths=np.array(daily_deaths),
        daily_neighbourhood_turnout=daily_neighbourhood_turnout,
    )
