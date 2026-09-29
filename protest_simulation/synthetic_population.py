from dataclasses import dataclass

import numpy as np

INDIA_POPULATION = 146e7

SC, ST, OBC, GENERAL = 0, 1, 2, 3
SOCIAL_GROUP_NAMES = ("SC", "ST", "OBC", "General")
SOCIAL_GROUP_COUNT = len(SOCIAL_GROUP_NAMES)
POPULATION_SHARE_BY_GROUP = np.array([0.166, 0.086, 0.42, 0.328])

MOST_DEPRIVED_TIER_SHARE_WITHIN_SC_AND_ST = 0.5
ABOVE_INCOME_LINE_SHARE_IN_BETTER_OFF_TIER = {SC: 0.19, ST: 0.16}
ABOVE_INCOME_LINE_SHARE_IN_MOST_DEPRIVED_TIER = {SC: 0.05, ST: 0.04}
ABOVE_INCOME_LINE_SHARE_IN_OTHER_GROUPS = {OBC: 0.17, GENERAL: 0.30}

IDENTITY_STRENGTH_BETA_SHAPE = (2, 3)
PEOPLE_PER_NEIGHBOURHOOD_IN_AGENTS = 100
TIER_MIXING_NOISE_WITHIN_NEIGHBOURHOODS = 0.8


@dataclass
class SyntheticPopulation:
    social_group: np.ndarray
    is_most_deprived_tier: np.ndarray
    is_above_income_line: np.ndarray
    identity_strength: np.ndarray
    threshold_standard_score: np.ndarray
    neighbourhood: np.ndarray
    neighbourhood_count: int
    agent_count: int

    @property
    def people_represented_per_agent(self) -> float:
        return INDIA_POPULATION / self.agent_count

    @property
    def is_sc_or_st(self) -> np.ndarray:
        return self.social_group <= ST


def build_synthetic_india(agent_count: int, random_generator: np.random.Generator) -> SyntheticPopulation:
    social_group = random_generator.choice(SOCIAL_GROUP_COUNT, size=agent_count, p=POPULATION_SHARE_BY_GROUP)

    is_sc_or_st = social_group <= ST
    is_most_deprived_tier = np.zeros(agent_count, bool)
    is_most_deprived_tier[is_sc_or_st] = (
        random_generator.random(is_sc_or_st.sum()) < MOST_DEPRIVED_TIER_SHARE_WITHIN_SC_AND_ST
    )

    is_above_income_line = np.zeros(agent_count, bool)
    income_rank_draw = random_generator.random(agent_count)
    for group in range(SOCIAL_GROUP_COUNT):
        belongs_to_group = social_group == group
        if group in (SC, ST):
            chance_above_line = np.where(
                is_most_deprived_tier,
                ABOVE_INCOME_LINE_SHARE_IN_MOST_DEPRIVED_TIER[group],
                ABOVE_INCOME_LINE_SHARE_IN_BETTER_OFF_TIER[group],
            )
            is_above_income_line[belongs_to_group] = (
                income_rank_draw[belongs_to_group] < chance_above_line[belongs_to_group]
            )
        else:
            is_above_income_line[belongs_to_group] = (
                income_rank_draw[belongs_to_group] < ABOVE_INCOME_LINE_SHARE_IN_OTHER_GROUPS[group]
            )

    identity_strength = random_generator.beta(*IDENTITY_STRENGTH_BETA_SHAPE, agent_count)
    threshold_standard_score = random_generator.standard_normal(agent_count)

    neighbourhood, neighbourhood_count = _assign_same_group_neighbourhoods(
        social_group, is_most_deprived_tier, random_generator
    )

    return SyntheticPopulation(
        social_group=social_group,
        is_most_deprived_tier=is_most_deprived_tier,
        is_above_income_line=is_above_income_line,
        identity_strength=identity_strength,
        threshold_standard_score=threshold_standard_score,
        neighbourhood=neighbourhood,
        neighbourhood_count=neighbourhood_count,
        agent_count=agent_count,
    )


def _assign_same_group_neighbourhoods(social_group, is_most_deprived_tier, random_generator):
    neighbourhood = np.empty(social_group.size, int)
    next_neighbourhood_id = 0
    for group in range(SOCIAL_GROUP_COUNT):
        members = np.where(social_group == group)[0]
        random_generator.shuffle(members)
        if group in (SC, ST):
            tier_sorting_key = is_most_deprived_tier[members] + random_generator.normal(
                0, TIER_MIXING_NOISE_WITHIN_NEIGHBOURHOODS, members.size
            )
            members = members[np.argsort(tier_sorting_key)]
        neighbourhoods_in_group = max(1, members.size // PEOPLE_PER_NEIGHBOURHOOD_IN_AGENTS)
        position_in_group = np.arange(members.size)
        neighbourhood[members] = next_neighbourhood_id + position_in_group * neighbourhoods_in_group // members.size
        next_neighbourhood_id += neighbourhoods_in_group
    return neighbourhood, next_neighbourhood_id
