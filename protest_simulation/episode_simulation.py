"""Simulate historical reservation-related shocks with the same model, for calibration and out-of-sample tests.

Each mechanism episode is a short campaign: an announcement day and a bandh day. Its shock is a symbolic threat vector
(which groups feel threatened, and in which direction) scaled by an episode-specific magnitude, which is unknown and
estimated. Party backing, coded from reporting (data_pipelines/episodes.py), sets the opposition-party amplifier:
backing 0 means no party machinery (amplifier 1.0); the reform scenario's amplifier of 1.5 corresponds to backing 0.6,
the level coded for the 2024 bandh, which the INDIA bloc parties supported.

Magnitudes are expressed relative to the reform's symbolic threat to SC agents (1.0). The paper's symbolic-only
validation scenario V is the reform's own symbolic threat, so 'the reform threatens SC/ST recognition about as much as
the 2018 dilution did' corresponds to a 2018 magnitude of 1.
"""
from dataclasses import dataclass

import numpy as np

from .model_parameters import ProtestModelParameters

REFORM_PARTY_BACKING = 0.6


@dataclass(frozen=True)
class EpisodeShock:
    key: str
    direction: tuple            # symbolic threat per group (SC, ST, OBC, General) before scaling
    party_backing: float
    deprived_tier_share: float = 0.8   # share of the threat felt by the most-deprived SC/ST tier


EPISODE_SHOCKS = {
    # 2018 dilution of the Atrocities Act: a threat to SC and ST protections; upper castes broadly welcomed it.
    "sc_st_bharat_bandh_2018": EpisodeShock("sc_st_bharat_bandh_2018", (1.0, 0.9, 0.0, -0.2), 0.2),
    # 2018 restoration: a symbolic loss for upper castes (and some OBC groups), a gain for SC/ST.
    "upper_caste_bandh_2018": EpisodeShock("upper_caste_bandh_2018", (-0.3, -0.3, 0.3, 1.0), 0.1),
    # 2024 sub-classification with a suggested creamy layer: a threat to SC/ST, felt less by the most-deprived sub-castes
    # that stood to gain from sub-classification (tier share is estimated).
    "sc_st_bharat_bandh_2024": EpisodeShock("sc_st_bharat_bandh_2024", (1.0, 0.9, 0.0, 0.0), 0.6, deprived_tier_share=0.4),
    # 2019 EWS quota with seat expansion: a small threat to the logic of caste-based reservation; General gains.
    "ews_quota_2019": EpisodeShock("ews_quota_2019", (1.0, 0.9, 0.45, -0.4), 0.0),
}


def party_amplifier(backing: float, reform_amplifier: float = 1.5) -> float:
    return 1.0 + (reform_amplifier - 1.0) * backing / REFORM_PARTY_BACKING


def episode_parameters(base: ProtestModelParameters, shock: EpisodeShock, magnitude: float,
                       party_backing: float = None, deprived_tier_share: float = None) -> ProtestModelParameters:
    parameters = base.copy()
    parameters.symbolic_threat_by_group = np.array(shock.direction, float) * magnitude
    parameters.most_deprived_tier_share_of_symbolic_threat = shock.deprived_tier_share if deprived_tier_share is None else deprived_tier_share
    for name in ("material_loss_sc_st_above_income_line", "material_loss_sc_st_below_income_line",
                 "material_loss_obc_below_income_line", "material_loss_general_below_income_line"):
        setattr(parameters, name, 0.0)
    parameters.material_change_by_segment = None
    parameters.sub_classification_gain_for_most_deprived_tier = 0.0
    parameters.opposition_party_amplifier = party_amplifier(shock.party_backing if party_backing is None else party_backing,
                                                            base.opposition_party_amplifier)
    parameters.campaign_length_days = 2
    parameters.bandh_schedule = "fixed"
    parameters.bandh_call_days = (1,)
    parameters.concession_rule = False
    return parameters


def turnout_by_state(population, outcome) -> np.ndarray:
    """People-days of protest in each state over the episode (needs record_agent_protest_days=True)."""
    agent_state = population.state_of_district[population.district]
    state_count = int(population.state_of_district.max()) + 1
    return np.bincount(agent_state, weights=outcome.agent_protest_days, minlength=state_count) * population.people_represented_per_agent
