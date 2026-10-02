"""The ten historical episodes as v3 shocks.

National episodes threaten groups by direction (SC, ST, OBC, General), as in the grounded specification; the state
communities share their parent group's direction in those episodes. The six community agitations (Patidar 2015, Jat
2016, Kapu 2016, Maratha 2017 and 2018, Gujjar 2019) threaten only the community: a demand for inclusion, felt as a
grievance by its members. Each episode's magnitude is unknown and estimated. Party backing is coded from reporting
(data_pipelines/episodes.py) and sets the party amplifier.

Each episode is simulated from an announcement day (day 0) through its core days, which are the action days; days
between non-consecutive core days (Maratha 2018) are ordinary days.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .population import COMMUNITIES, COMMUNITY_NAMES, IDENTITY_GROUP_COUNT

TARGETS_PATH = Path(__file__).resolve().parent.parent / "data" / "derived" / "episode_targets_v3.json"
ALIGNMENT_PATH = Path(__file__).resolve().parent.parent / "data" / "derived" / "state_government_alignment.csv"
# Rajasthan, Madhya Pradesh and Chhattisgarh changed to Congress governments in December 2018, before the EWS amendment.
LEFT_NDA_BEFORE_JANUARY_2019 = ("Rajasthan", "Madhya Pradesh", "Chhattisgarh")
REFERENCE_AMPLIFIER = 1.5        # party amplifier at the backing coded for 2024 (0.6)
REFERENCE_BACKING = 0.6

NATIONAL_DIRECTIONS = {
    "sc_st_bharat_bandh_2018": (1.0, 0.9, 0.0, -0.2),
    "upper_caste_bandh_2018": (-0.3, -0.3, 0.3, 1.0),
    "sc_st_bharat_bandh_2024": (1.0, 0.9, 0.0, 0.0),
    "ews_quota_2019": (1.0, 0.9, 0.45, -0.4),
}
COMMUNITY_OF_EPISODE = {"patidar_2015": "patidar", "jat_2016": "jat", "kapu_2016": "kapu",
                        "maratha_march_mumbai_2017": "maratha", "maratha_quota_2018": "maratha", "gujjar_2019": "gujjar"}
EPISODE_KEYS = tuple(NATIONAL_DIRECTIONS) + tuple(COMMUNITY_OF_EPISODE)


def group_vector(direction) -> np.ndarray:
    """Threat by identity group from a (SC, ST, OBC, General) direction; communities take their parent group's value."""
    vector = np.zeros(IDENTITY_GROUP_COUNT)
    vector[:4] = direction
    for index, name in enumerate(COMMUNITY_NAMES):
        vector[4 + index] = direction[COMMUNITIES[name][1]]
    return vector


def episode_threat(key: str, magnitude: float) -> np.ndarray:
    if key in NATIONAL_DIRECTIONS:
        return group_vector(NATIONAL_DIRECTIONS[key]) * magnitude
    vector = np.zeros(IDENTITY_GROUP_COUNT)
    vector[4 + COMMUNITY_NAMES.index(COMMUNITY_OF_EPISODE[key])] = magnitude
    return vector


def aligned_states(column: str, drop=()) -> tuple:
    """States whose government belonged to the Union's ruling coalition (NDA) on a date."""
    import pandas as pd
    table = pd.read_csv(ALIGNMENT_PATH)
    return tuple(sorted(set(table.loc[table[column] == 1, "state"]) - set(drop)))


def episode_aligned_states(key: str) -> tuple:
    """National episodes target the Union government's policy; community agitations target their own state's."""
    if key in ("sc_st_bharat_bandh_2018", "upper_caste_bandh_2018"):
        return aligned_states("nda_aligned_2018_04_02")
    if key == "sc_st_bharat_bandh_2024":
        return aligned_states("nda_aligned_2024_08_21")
    if key == "ews_quota_2019":
        return aligned_states("nda_aligned_2018_04_02", drop=LEFT_NDA_BEFORE_JANUARY_2019)
    return ()


def party_amplifier(backing: float) -> float:
    return 1.0 + (REFERENCE_AMPLIFIER - 1.0) * backing / REFERENCE_BACKING


@dataclass(frozen=True)
class EpisodeSchedule:
    days: int
    action_days: tuple
    core_day_indices: tuple


def load_targets() -> dict:
    return json.loads(TARGETS_PATH.read_text())


def schedule(target: dict) -> EpisodeSchedule:
    action = tuple(1 + offset for offset in target["action_day_offsets"])
    return EpisodeSchedule(days=action[-1] + 1, action_days=action, core_day_indices=action)
