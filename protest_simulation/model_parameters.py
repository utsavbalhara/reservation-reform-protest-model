from dataclasses import dataclass, field, replace

import numpy as np


def _symbolic_threat_of_abolishing_caste_quotas():
    return np.array([1.0, 0.9, 0.45, -0.4])


def _organizational_capacity_sc_st_obc_general():
    return np.array([0.60, 0.40, 0.40, 0.10])


@dataclass
class ProtestModelParameters:
    loss_aversion: float = 2.25
    material_loss_weight: float = 0.45
    symbolic_threat_weight: float = 3.3
    symbolic_threat_by_group: np.ndarray = field(default_factory=_symbolic_threat_of_abolishing_caste_quotas)
    most_deprived_tier_share_of_symbolic_threat: float = 0.8

    material_loss_sc_st_above_income_line: float = 1.0
    material_loss_sc_st_below_income_line: float = 0.35
    material_loss_obc_below_income_line: float = 0.25
    material_loss_general_below_income_line: float = -0.15
    sub_classification_gain_for_most_deprived_tier: float = 0.0

    organizational_capacity_by_group: np.ndarray = field(default_factory=_organizational_capacity_sc_st_obc_general)
    opposition_party_amplifier: float = 1.5

    social_influence_saturates: bool = True
    max_neighbourhood_influence: float = 0.9
    neighbourhood_turnout_at_saturation: float = 0.10
    max_national_visibility_influence: float = 0.6
    national_turnout_at_saturation: float = 0.02

    mean_participation_threshold: float = 6.6
    participation_threshold_spread: float = 1.4
    decision_noise: float = 0.3
    fatigue_per_protest_day: float = 0.10

    deaths_per_crore_protester_days: float = 4.0
    symbolic_threat_rise_per_death: float = 0.012
    max_martyrdom_symbolic_rise: float = 0.35

    heavy_policing_turnout_cost: float = 0.0
    internet_shutdown_active: bool = False
    internet_shutdown_coordination_factor: float = 0.6
    internet_shutdown_bandh_mobilization_factor: float = 0.8
    internet_shutdown_violence_factor: float = 2.5

    campaign_length_days: int = 40
    bandh_call_days: tuple = (2, 12, 25)
    mobilization_on_bandh_days: float = 1.0
    mobilization_on_ordinary_days: float = 0.15

    def copy(self) -> "ProtestModelParameters":
        return replace(
            self,
            symbolic_threat_by_group=self.symbolic_threat_by_group.copy(),
            organizational_capacity_by_group=self.organizational_capacity_by_group.copy(),
        )


BASELINE_ABRUPT_INCOME_ONLY_SWITCH = ProtestModelParameters()
