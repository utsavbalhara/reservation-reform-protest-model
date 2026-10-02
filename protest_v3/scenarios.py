"""Reform scenarios for the v3 model, each tied to the closest historical episode where one exists.

Every scenario is run on a retained calibration set, and its protest is reported relative to a replay of the 2 April 2018
bandh on the same set and random numbers. Relative size is what the event data identify; headcounts are not.

Reforms (threat direction over SC, ST, OBC, General; the state communities take their parent group's value):
  income_only            one income test at Rs 8 lakh, quotas merged into one pool. Threat: the 2018 shock times R,
                         R ~ U(0.5, 1.5); no episode anchors R. Seats: allocation model, merged pool.
  income_only_expanded   the same with a quarter more seats (2019 EWS precedent). Seats: allocation model.
  sc_st_creamy_layer     an income filter inside SC and ST quotas (OBC and EWS unchanged). Threat: anchored to the 2024
                         bandh, which protested a suggested creamy layer together with sub-classification, times
                         U(0.5, 1) because it is one of the two. Seats: allocation model, caste quotas closed to
                         above-line SC and ST families.
  sub_classification     a larger share of SC and ST benefits for the most-deprived tier. Threat: anchored to 2024 in
                         the same way, with the deprived tier's share of the threat as estimated for 2024. Seats: no
                         rank-list data by sub-caste; the deprived tier's gain is a judgement, U(0.2, 0.8).
  abolition              no reservation of any kind. Threat: no episode is close; R x U(1, 2) times the 2018 shock, and
                         OBC, who lose their quota too, at 0.9 of the SC threat instead of 0.45. Seats: allocation
                         model, every seat on the open merit list.
References: replays of the 2018 bandh and of the EWS amendment (the null episode).

Modifiers, applicable to any reform; their effects are judgements with stated ranges:
  phased       grandfathering with a ten-year glide: material change x U(0.1, 0.6), threat x U(0.85, 1);
  negotiated   a cross-party commission: no party backing, threat x U(0.7, 1);
  guaranteed   statutory guarantees of what stays untouched: threat x U(0.7, 1).
"""
from dataclasses import dataclass

import numpy as np

from .episodes import episode_threat, group_vector, party_amplifier

REFORMS = ("income_only", "income_only_expanded", "sc_st_creamy_layer", "sub_classification", "abolition")
MODIFIER_SETS = {"none": (), "phased": ("phased",), "negotiated": ("negotiated",), "all_three": ("phased", "negotiated", "guaranteed")}
ALLOCATION_CASE = {"income_only": "reserved_first|0.595", "income_only_expanded": "lever|merged_pool_expanded",
                   "sc_st_creamy_layer": "lever|caste_income_filter", "abolition": "lever|abolition"}
LABELS = {"income_only": "Income-only test, merged pool", "income_only_expanded": "Income-only, a quarter more seats",
          "sc_st_creamy_layer": "SC/ST creamy layer (income filter)", "sub_classification": "SC/ST sub-classification",
          "abolition": "Abolish all reservation", "replay_2018": "2018 bandh (replay)", "replay_ews_2019": "EWS amendment (replay)"}


@dataclass
class ScenarioDraw:
    threat: np.ndarray
    deprived_tier_share: float
    material_by_segment: dict
    deprived_tier_material_gain: float
    amplifier: float


def material_table(case):
    if case is None:
        return None
    from protest_simulation.grounded_specification import ALLOCATION_RESULTS, material_change_by_segment
    import json
    if case not in json.loads(ALLOCATION_RESULTS.read_text())["summary"]:
        raise KeyError(f"allocation case {case} missing; rerun experiments.merged_pool_allocation")
    return dict(material_change_by_segment(case))


def draw(reform: str, modifiers: tuple, sample: dict, rng: np.random.Generator) -> ScenarioDraw:
    m2018, m2024 = sample["magnitude_sc_st_bharat_bandh_2018"], sample["magnitude_sc_st_bharat_bandh_2024"]
    backing = rng.uniform(0.2, 0.6)        # between the 2018 (local) and 2024 (party-backed) levels
    deprived, gain = 0.8, 0.0
    R = rng.uniform(0.5, 1.5)
    if reform == "replay_2018":
        return ScenarioDraw(episode_threat("sc_st_bharat_bandh_2018", m2018), 0.8, None, 0.0, party_amplifier(0.2))
    if reform == "replay_ews_2019":
        return ScenarioDraw(episode_threat("ews_quota_2019", sample["magnitude_ews_quota_2019"]), 0.8, None, 0.0, party_amplifier(0.0))
    if reform in ("income_only", "income_only_expanded"):
        threat = group_vector((1.0, 0.9, 0.45, -0.4)) * R * m2018
    elif reform == "sc_st_creamy_layer":
        threat = group_vector((1.0, 0.9, 0.0, 0.0)) * m2024 * rng.uniform(0.5, 1.0)
    elif reform == "sub_classification":
        threat = group_vector((1.0, 0.9, 0.0, 0.0)) * m2024 * rng.uniform(0.5, 1.0)
        deprived, gain = sample["deprived_tier_share_2024"], rng.uniform(0.2, 0.8)
    elif reform == "abolition":
        threat = group_vector((1.0, 0.9, 0.9, -0.4)) * R * rng.uniform(1.0, 2.0) * m2018
    else:
        raise KeyError(reform)
    material = material_table(ALLOCATION_CASE.get(reform))
    if "phased" in modifiers:
        factor = rng.uniform(0.1, 0.6)
        if material:
            material = {k: v * factor if v > 0 else v for k, v in material.items()}
        threat = threat * rng.uniform(0.85, 1.0)
    if "negotiated" in modifiers:
        backing = 0.0
        threat = threat * rng.uniform(0.7, 1.0)
    if "guaranteed" in modifiers:
        threat = threat * rng.uniform(0.7, 1.0)
    return ScenarioDraw(threat, deprived, material, gain, party_amplifier(backing))
