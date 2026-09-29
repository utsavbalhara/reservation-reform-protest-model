"""A district-level synthetic India.

Agents are allocated to the 640 Census 2011 districts in proportion to population (largest remainder). Within each
district, the SC and ST shares are the district's own (Census 2011 Primary Census Abstract); the rest of the population
is split between OBC and General using the state's NFHS-5 (2019-21) ratio of OBC to OBC-plus-Other household heads.
SC and ST tiers, income-line status, identity strength and threshold scores follow the national model. Neighbourhoods
of about 100 agents are formed within districts; a share h of agents live in single-group neighbourhoods and the rest in
mixed ones (see synthetic_population.build_synthetic_india for what h means).

Sources: data/derived/census2011_district_sc_st.csv (from the Census 2011 PCA as scraped by pigshell/india-census-2011;
Telangana's ten districts separated from Andhra Pradesh) and data/derived/nfhs5_caste_of_household_head_by_state.csv
(NFHS-5 India Report, Table 2.10).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .synthetic_population import (
    ABOVE_INCOME_LINE_SHARE_IN_BETTER_OFF_TIER, ABOVE_INCOME_LINE_SHARE_IN_MOST_DEPRIVED_TIER, ABOVE_INCOME_LINE_SHARE_IN_OTHER_GROUPS,
    GENERAL, IDENTITY_STRENGTH_BETA_SHAPE, INDIA_POPULATION, MOST_DEPRIVED_TIER_SHARE_WITHIN_SC_AND_ST, OBC,
    PEOPLE_PER_NEIGHBOURHOOD_IN_AGENTS, SC, SOCIAL_GROUP_COUNT, ST, TIER_MIXING_NOISE_WITHIN_NEIGHBOURHOODS, SyntheticPopulation)

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DISTRICT_TABLE_PATH = REPOSITORY_ROOT / "data" / "derived" / "census2011_district_sc_st.csv"
NFHS_TABLE_PATH = REPOSITORY_ROOT / "data" / "derived" / "nfhs5_caste_of_household_head_by_state.csv"
NFHS_NAME_FOR_STATE = {"Dadra & Nagar Haveli": "Dadra & Nagar Haveli and Daman & Diu", "Daman & Diu": "Dadra & Nagar Haveli and Daman & Diu"}


def district_table() -> pd.DataFrame:
    districts = pd.read_csv(DISTRICT_TABLE_PATH)
    nfhs = pd.read_csv(NFHS_TABLE_PATH).set_index("state")
    obc_ratio = {}
    for state in districts.state.unique():
        row = nfhs.loc[NFHS_NAME_FOR_STATE.get(state, state)]
        obc_ratio[state] = row["obc"] / (row["obc"] + row["other"])
    districts["share_sc"] = districts.sc / districts.population
    districts["share_st"] = districts.st / districts.population
    rest = 1 - districts.share_sc - districts.share_st
    districts["obc_ratio_of_rest"] = districts.state.map(obc_ratio)
    districts["share_obc"] = rest * districts.obc_ratio_of_rest
    districts["share_general"] = rest - districts.share_obc
    return districts


def _largest_remainder(weights: np.ndarray, total: int) -> np.ndarray:
    exact = weights / weights.sum() * total
    counts = np.floor(exact).astype(int)
    counts[np.argsort(-(exact - counts))[: total - counts.sum()]] += 1
    return counts


def build_synthetic_india_by_district(agent_count: int, random_generator: np.random.Generator,
                                      same_group_neighbourhood_share: float = 1.0,
                                      obc_creamy_share_of_above_line: float = 1.0,
                                      ews_asset_exclusion_share: float = 0.0) -> SyntheticPopulation:
    table = district_table()
    per_district = _largest_remainder(table.population.to_numpy(float), agent_count)
    district = np.repeat(np.arange(len(table)), per_district)
    shares = table[["share_sc", "share_st", "share_obc", "share_general"]].to_numpy()[district]
    uniform = random_generator.random(agent_count)
    social_group = (uniform[:, None] > np.cumsum(shares, axis=1)[:, :-1]).sum(axis=1)

    is_sc_or_st = social_group <= ST
    is_most_deprived_tier = is_sc_or_st & (random_generator.random(agent_count) < MOST_DEPRIVED_TIER_SHARE_WITHIN_SC_AND_ST)
    chance_above = np.zeros(agent_count)
    for group in (SC, ST):
        members = social_group == group
        chance_above[members] = np.where(is_most_deprived_tier[members], ABOVE_INCOME_LINE_SHARE_IN_MOST_DEPRIVED_TIER[group],
                                         ABOVE_INCOME_LINE_SHARE_IN_BETTER_OFF_TIER[group])
    for group in (OBC, GENERAL):
        chance_above[social_group == group] = ABOVE_INCOME_LINE_SHARE_IN_OTHER_GROUPS[group]
    is_above_income_line = random_generator.random(agent_count) < chance_above
    identity_strength = random_generator.beta(*IDENTITY_STRENGTH_BETA_SHAPE, agent_count)
    threshold_standard_score = random_generator.standard_normal(agent_count)

    # Neighbourhoods: within each (state, group) block -- or (state, mixed) for agents in mixed areas -- agents are
    # ordered by district (Census district codes run roughly geographically within a state) and then by tier, and cut
    # into runs of about 100. A neighbourhood therefore lies within one district or spans two adjacent ones.
    lives_in_mixed_area = random_generator.random(agent_count) >= same_group_neighbourhood_share
    group_key = np.where(lives_in_mixed_area, SOCIAL_GROUP_COUNT, social_group)
    tier_noise = is_most_deprived_tier + random_generator.normal(0, TIER_MIXING_NOISE_WITHIN_NEIGHBOURHOODS, agent_count)
    state_codes = pd.factorize(table.state)[0]
    agent_state = state_codes[district]
    order = np.lexsort((random_generator.random(agent_count), tier_noise * is_sc_or_st, district, group_key, agent_state))
    neighbourhood = np.empty(agent_count, int)
    next_id = 0
    block_key = agent_state[order] * (SOCIAL_GROUP_COUNT + 1) + group_key[order]
    boundaries = np.flatnonzero(np.diff(block_key)) + 1
    for block in np.split(np.arange(agent_count), boundaries):
        members = order[block]
        count = max(1, members.size // PEOPLE_PER_NEIGHBOURHOOD_IN_AGENTS)
        neighbourhood[members] = next_id + np.arange(members.size) * count // members.size
        next_id += count

    is_obc_creamy_layer = is_asset_excluded_from_ews = None
    if obc_creamy_share_of_above_line < 1.0:
        is_obc_creamy_layer = (social_group == OBC) & is_above_income_line & (random_generator.random(agent_count) < obc_creamy_share_of_above_line)
    if ews_asset_exclusion_share > 0.0:
        is_asset_excluded_from_ews = (social_group == GENERAL) & ~is_above_income_line & (random_generator.random(agent_count) < ews_asset_exclusion_share)

    states = sorted(table.state.unique())
    state_of_district = np.array([states.index(state) for state in table.state])
    return SyntheticPopulation(
        social_group=social_group, is_most_deprived_tier=is_most_deprived_tier, is_above_income_line=is_above_income_line,
        identity_strength=identity_strength, threshold_standard_score=threshold_standard_score, neighbourhood=neighbourhood,
        neighbourhood_count=next_id, agent_count=agent_count, is_obc_creamy_layer=is_obc_creamy_layer,
        is_asset_excluded_from_ews=is_asset_excluded_from_ews, district=district,
        district_table=table.assign(state_index=state_of_district), state_of_district=state_of_district, population_total=INDIA_POPULATION)


def state_names(population: SyntheticPopulation) -> list:
    return sorted(population.district_table.state.unique())
