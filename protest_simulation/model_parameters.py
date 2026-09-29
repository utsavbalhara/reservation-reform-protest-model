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
    # Symbolic threat felt by the better-off SC/ST tier, relative to its group's (sub-classification can raise it).
    better_off_tier_symbolic_threat_factor: float = 1.0
    # Grounded specification only: the 2024 episode's estimated shock relative to 2018's, for the parameter set drawn
    # in this world, and the reform's shock relative to 2018 (R). Their ratio is the share of the reform's symbolic
    # threat that a 2024-type change (an income filter inside SC/ST quotas) carries; lever L3 uses it.
    episode_2024_threat_relative_to_2018: float = None
    reform_shock_ratio: float = 1.0

    material_loss_sc_st_above_income_line: float = 1.0
    material_loss_sc_st_below_income_line: float = 0.35
    material_loss_obc_below_income_line: float = 0.25
    material_loss_general_below_income_line: float = -0.15
    sub_classification_gain_for_most_deprived_tier: float = 0.0
    # Optional per-segment material change, keyed by segment name (see protest_campaign.MATERIAL_SEGMENTS).
    # When set, it replaces the five material fields above; it is how the merit-allocation model feeds the ABM.
    material_change_by_segment: dict = None
    # Optional material-change tables for levers that are themselves allocation rules ("seat_expansion",
    # "hybrid_caste_subquotas"), keyed by lever and then by segment. Used only when material_change_by_segment is set.
    lever_material_tables: dict = None
    reform_material_change_by_segment: dict = None

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

    # ---------------------------------------------------------------------------------------------------------------
    # Structural options. Every default below reproduces the stylized reference specification exactly.
    # ---------------------------------------------------------------------------------------------------------------
    # Random numbers. "single": one stream, as in the reference. "split": agent decisions, events and organizational
    # decisions use separate streams derived from the campaign seed, so scenarios with different event histories
    # still share every agent's daily draw (tighter pairing).
    random_streams: str = "single"

    # Shape of the threshold distribution, applied to each agent's standard-normal score by quantile mapping, so the
    # ordering of agents is preserved. "normal" (reference), "logistic" (heavier tails), "activist_mixture" (a small
    # low-threshold core). All three have mean 0 and variance 1 before scaling by the spread.
    threshold_distribution: str = "normal"
    activist_core_share: float = 0.03
    activist_core_offset: float = 3.0
    activist_core_spread: float = 0.5

    # Bandh calls. "fixed": on bandh_call_days (reference). "endogenous": organizations call a bandh for the next day
    # when they expect at least bandh_call_expected_turnout people to come out, starting on first_bandh_call_day, then
    # no sooner than a refractory gap after the previous bandh, up to max_bandh_calls, and never after a concession.
    bandh_schedule: str = "fixed"
    bandh_call_expected_turnout: float = 10e5
    first_bandh_call_day: int = 2
    bandh_refractory_days: tuple = (10, 13)
    max_bandh_calls: int = 4

    # Government concession. Pressure = (largest daily turnout so far, in crore) + (deaths so far / deaths scale).
    # The daily chance of conceding is max_daily_hazard x logistic((pressure - midpoint) / width). A concession
    # removes a share of every agent's symbolic threat and material change from the next day, cancels further bandh
    # calls, and gives the General category a symbolic threat of counter_mobilization_symbolic_threat (backlash).
    concession_rule: bool = False
    concession_pressure_midpoint: float = 1.0
    concession_pressure_width: float = 0.25
    concession_deaths_scale: float = 20.0
    concession_max_daily_hazard: float = 0.15
    concession_grievance_relief: float = 0.6
    counter_mobilization_symbolic_threat: float = 0.0

    # Deaths. "poisson" (reference): one Poisson draw with mean kappa x protester-days. "negative_binomial_split":
    # police-attributed and other deaths are drawn separately from negative binomials with the given size parameter.
    # Heavy policing scales the police component, internet shutdowns the other component. Police-attributed deaths
    # count fully toward the martyr effect; other deaths count with weight martyr_weight_of_other_deaths.
    # Deaths per protester-day on bandh days relative to ordinary days (bandh enforcement brings clashes and firing).
    bandh_day_death_rate_multiplier: float = 1.0
    death_model: str = "poisson"
    death_dispersion: float = 1.0
    police_attributed_death_share: float = 0.5
    police_death_multiplier: float = 1.0
    other_death_multiplier: float = 1.0
    martyr_weight_of_other_deaths: float = 1.0
    # Multiplies the martyr effect per death. Negative values mean deaths intimidate instead of mobilizing
    # (a deterrence regime); the accumulated effect is then bounded below by -max_martyrdom_symbolic_rise.
    death_response: float = 1.0

    def copy(self) -> "ProtestModelParameters":
        return replace(
            self,
            symbolic_threat_by_group=self.symbolic_threat_by_group.copy(),
            organizational_capacity_by_group=self.organizational_capacity_by_group.copy(),
            material_change_by_segment=None if self.material_change_by_segment is None else dict(self.material_change_by_segment),
            reform_material_change_by_segment=None if self.reform_material_change_by_segment is None else dict(self.reform_material_change_by_segment),
            lever_material_tables=None if self.lever_material_tables is None else {k: dict(v) for k, v in self.lever_material_tables.items()},
        )

    def uses_reference_structure(self) -> bool:
        return (self.random_streams == "single" and self.threshold_distribution == "normal" and self.bandh_schedule == "fixed"
                and not self.concession_rule and self.death_model == "poisson" and self.death_response == 1.0)


BASELINE_ABRUPT_INCOME_ONLY_SWITCH = ProtestModelParameters()
