"""The v3 synthetic population: the district population of protest_simulation.geography, plus

  - district covariates (Census 2011): urban share, literacy and household phone ownership, standardized across
    districts; they shape how easily a district mobilizes on action days (coefficients estimated in calibration);
  - state communities that mobilized for reservation in the episodes: Jats in Haryana, Patidars in Gujarat, Kapus in
    Andhra Pradesh and Marathas in Maharashtra (General category), and Gujjars in Rajasthan (OBC). Their population
    shares are commonly cited estimates (there has been no caste census since 1931) and are assumptions. Members are
    assigned whole neighbourhoods at a time within their parent group's neighbourhoods in their state, so communities
    are residentially clustered, as jatis are.

Agents carry an identity group: their community if they belong to one, otherwise their social group (SC, ST, OBC,
General). National visibility and organization act on identity groups.
"""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from protest_simulation.geography import build_synthetic_india_by_district
from protest_simulation.synthetic_population import GENERAL, OBC, SOCIAL_GROUP_COUNT

DERIVED = Path(__file__).resolve().parent.parent / "data" / "derived"
POPULATION_SEED = 7
OBC_CREAMY_SHARE = 0.5
EWS_ASSET_EXCLUSION_SHARE = 0.15

# name -> (state, parent group, share of the state's population). Commonly cited estimates; assumptions.
COMMUNITIES = {
    "jat": ("Haryana", GENERAL, 0.25),
    "patidar": ("Gujarat", GENERAL, 0.13),
    "kapu": ("Andhra Pradesh", GENERAL, 0.17),
    "maratha": ("Maharashtra", GENERAL, 0.31),
    "gujjar": ("Rajasthan", OBC, 0.07),
}
COMMUNITY_NAMES = tuple(COMMUNITIES)
IDENTITY_GROUP_NAMES = ("SC", "ST", "OBC", "General") + COMMUNITY_NAMES
IDENTITY_GROUP_COUNT = len(IDENTITY_GROUP_NAMES)
COVARIATES = ("urban_share", "literacy_rate", "phone_share")


@dataclass
class V3Population:
    base: object                 # SyntheticPopulation from protest_simulation.geography
    identity_group: np.ndarray   # social group, or SOCIAL_GROUP_COUNT + community index
    community: np.ndarray        # -1, or community index
    agent_state: np.ndarray      # state index of each agent
    state_names: list
    covariates: np.ndarray       # agents x len(COVARIATES), standardized across districts
    district_covariates: pd.DataFrame
    mixing: float
    is_hindu_general: np.ndarray  # General-category agents assigned Hindu by their state's NFHS-5 Hindu share

    @property
    def agent_count(self):
        return self.base.agent_count

    @property
    def people_per_agent(self):
        return self.base.people_represented_per_agent


def district_covariates(table):
    covariates = pd.read_csv(DERIVED / "census2011_district_covariates.csv").set_index("district_code")
    values = covariates.reindex(table.district_code)[list(COVARIATES)].to_numpy(float)
    return (values - values.mean(axis=0)) / values.std(axis=0), covariates


@lru_cache(maxsize=16)
def build_population(agent_count: int = 120_000, mixing: float = 0.8) -> V3Population:
    random_generator = np.random.default_rng(POPULATION_SEED)
    base = build_synthetic_india_by_district(agent_count, random_generator, same_group_neighbourhood_share=round(mixing, 2),
                                             obc_creamy_share_of_above_line=OBC_CREAMY_SHARE,
                                             ews_asset_exclusion_share=EWS_ASSET_EXCLUSION_SHARE)
    table = base.district_table
    states = sorted(table.state.unique())
    agent_state = base.state_of_district[base.district]
    community = np.full(agent_count, -1)
    community_rng = np.random.default_rng(POPULATION_SEED + 1)
    for index, (name, (state, parent, share)) in enumerate(COMMUNITIES.items()):
        in_state = agent_state == states.index(state)
        target = int(round(share * in_state.sum()))
        candidates = in_state & (base.social_group == parent) & (community < 0)
        hoods = np.unique(base.neighbourhood[candidates])
        community_rng.shuffle(hoods)
        assigned = 0
        for hood in hoods:
            if assigned >= target:
                break
            members = candidates & (base.neighbourhood == hood)
            take = np.flatnonzero(members)[: target - assigned]
            community[take] = index
            assigned += take.size
    identity_group = np.where(community >= 0, SOCIAL_GROUP_COUNT + community, base.social_group)
    standardized, raw = district_covariates(table)
    hindu_share = np.array([hindu_share_of_state(state) for state in states])
    general = identity_group == GENERAL
    is_hindu_general = general & (np.random.default_rng(POPULATION_SEED + 2).random(agent_count) < hindu_share[agent_state])
    return V3Population(base=base, identity_group=identity_group, community=community, agent_state=agent_state,
                        state_names=states, covariates=standardized[base.district], district_covariates=raw, mixing=mixing,
                        is_hindu_general=is_hindu_general)


def hindu_share_of_state(state: str) -> float:
    """Share of household heads who are Hindu (NFHS-5, 2019-21). Religion is assigned independently of caste category
    within a state, a simplification: the NFHS state tables give religion and caste separately, not jointly."""
    from protest_simulation.geography import NFHS_NAME_FOR_STATE, NFHS_TABLE_PATH
    table = pd.read_csv(NFHS_TABLE_PATH).set_index("state")
    name = NFHS_NAME_FOR_STATE.get(state, state)
    return float(table.loc[name, "hindu"]) / 100 if name in table.index else float(table.loc["India", "hindu"]) / 100
