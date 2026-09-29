from dataclasses import dataclass
from typing import Callable

from .model_parameters import ProtestModelParameters
from .synthetic_population import OBC

BELOW_LINE_RESERVED_SEGMENTS = ("SC_below", "ST_below", "OBC_below")
ABOVE_LINE_LOSING_SEGMENTS = ("SC_above", "ST_above", "OBC_above_ncl")


def _scale_segment_losses(parameters: ProtestModelParameters, factor: float, segments=None) -> None:
    """With an allocation-derived segment table in use, scale the losses (positive entries) of the given segments."""
    if parameters.material_change_by_segment is None:
        return
    parameters.material_change_by_segment = {
        segment: value * factor if value > 0 and (segments is None or segment in segments) else value
        for segment, value in parameters.material_change_by_segment.items()}


def _use_lever_table(parameters: ProtestModelParameters, lever: str) -> bool:
    """Shift each segment's material change by the lever's allocation-derived change relative to the reform."""
    tables = parameters.lever_material_tables
    if parameters.material_change_by_segment is None or not tables or lever not in tables:
        return False
    reform = parameters.reform_material_change_by_segment or parameters.material_change_by_segment
    parameters.material_change_by_segment = {segment: value + tables[lever][segment] - reform[segment]
                                             for segment, value in parameters.material_change_by_segment.items()}
    return True


def grandfather_current_cohorts_with_ten_year_glide(parameters: ProtestModelParameters) -> None:
    share_of_loss_felt_now = 0.3
    parameters.material_loss_sc_st_above_income_line *= share_of_loss_felt_now
    parameters.material_loss_sc_st_below_income_line *= share_of_loss_felt_now
    parameters.material_loss_obc_below_income_line *= share_of_loss_felt_now
    _scale_segment_losses(parameters, share_of_loss_felt_now)
    parameters.symbolic_threat_by_group = parameters.symbolic_threat_by_group * 0.95


def expand_seats_so_no_group_loses(parameters: ProtestModelParameters) -> None:
    parameters.material_loss_sc_st_below_income_line = 0.0
    parameters.material_loss_obc_below_income_line = 0.0
    # Grounded specification: the material changes of the merged pool with a quarter more seats (2019 EWS precedent).
    if not _use_lever_table(parameters, "seat_expansion"):
        _scale_segment_losses(parameters, 0.0, BELOW_LINE_RESERVED_SEGMENTS)


def keep_caste_subquotas_with_income_filter(parameters: ProtestModelParameters, symbolic_threat_retained: float = 0.4) -> None:
    symbolic_threat = parameters.symbolic_threat_by_group.copy()
    symbolic_threat[: OBC + 1] *= symbolic_threat_retained
    parameters.symbolic_threat_by_group = symbolic_threat
    parameters.material_loss_sc_st_below_income_line = 0.0
    parameters.material_loss_obc_below_income_line = 0.0
    # Grounded specification: caste quotas kept but closed to above-line SC and ST families, whose seats go to
    # below-line SC and ST candidates (a gain for them), from the allocation model.
    if not _use_lever_table(parameters, "hybrid_caste_subquotas"):
        _scale_segment_losses(parameters, 0.0, BELOW_LINE_RESERVED_SEGMENTS)


def sub_classify_to_favour_most_deprived(parameters: ProtestModelParameters) -> None:
    parameters.sub_classification_gain_for_most_deprived_tier = 0.5
    parameters.most_deprived_tier_share_of_symbolic_threat = 0.3


def build_consensus_through_data_first_commission(parameters: ProtestModelParameters) -> None:
    parameters.opposition_party_amplifier /= 1.5
    parameters.symbolic_threat_by_group = parameters.symbolic_threat_by_group * 0.8


def compensate_above_line_losers(parameters: ProtestModelParameters) -> None:
    parameters.material_loss_sc_st_above_income_line *= 0.5
    _scale_segment_losses(parameters, 0.5, ABOVE_LINE_LOSING_SEGMENTS)


def guarantee_untouched_protections(parameters: ProtestModelParameters) -> None:
    parameters.symbolic_threat_by_group = parameters.symbolic_threat_by_group * 0.85


def shut_down_internet(parameters: ProtestModelParameters) -> None:
    parameters.internet_shutdown_active = True


def deploy_heavy_policing(parameters: ProtestModelParameters, death_multiplier: float = 3.0) -> None:
    parameters.heavy_policing_turnout_cost = 0.4
    if parameters.death_model == "poisson":
        parameters.deaths_per_crore_protester_days *= death_multiplier
    else:
        # Only police-attributed deaths rise; scale that component so total deaths rise by the same multiplier.
        share = parameters.police_attributed_death_share
        parameters.police_death_multiplier *= (death_multiplier - (1 - share)) / share
    parameters.symbolic_threat_rise_per_death *= 1.5


def remove_all_material_loss(parameters: ProtestModelParameters) -> None:
    parameters.material_loss_sc_st_above_income_line = 0.0
    parameters.material_loss_sc_st_below_income_line = 0.0
    parameters.material_loss_obc_below_income_line = 0.0
    parameters.material_loss_general_below_income_line = 0.0
    if parameters.material_change_by_segment is not None:
        parameters.material_change_by_segment = {segment: 0.0 for segment in parameters.material_change_by_segment}


@dataclass(frozen=True)
class Scenario:
    key: str
    code: str
    label: str
    category: str
    interventions: tuple

    def apply_to(self, parameters: ProtestModelParameters) -> ProtestModelParameters:
        adjusted = parameters.copy()
        for intervention in self.interventions:
            intervention(adjusted)
        return adjusted


# Allocation-rule levers come first in each package, so that levers scaling losses act on the rule actually adopted
# (the order does not matter for the stylized specification, where these operations commute).
MANAGED_TRANSITION = (
    expand_seats_so_no_group_loses,
    grandfather_current_cohorts_with_ten_year_glide,
    build_consensus_through_data_first_commission,
    compensate_above_line_losers,
    guarantee_untouched_protections,
)

HYBRID_DESIGN = (
    keep_caste_subquotas_with_income_filter,
    sub_classify_to_favour_most_deprived,
    grandfather_current_cohorts_with_ten_year_glide,
    build_consensus_through_data_first_commission,
    compensate_above_line_losers,
    guarantee_untouched_protections,
)

SCENARIOS = (
    Scenario("baseline", "Base", "Abrupt ₹8L income-only switch", "reference", ()),
    Scenario("grandfathering", "L1", "Grandfather current cohorts + 10-year glide", "lever", (grandfather_current_cohorts_with_ten_year_glide,)),
    Scenario("seat_expansion", "L2", "Expand seats so no group loses", "lever", (expand_seats_so_no_group_loses,)),
    Scenario("hybrid_caste_subquotas", "L3", "Keep caste sub-quotas, add income filter", "lever", (keep_caste_subquotas_with_income_filter,)),
    Scenario("sub_classification", "L4", "Sub-classify to favour most-deprived", "lever", (sub_classify_to_favour_most_deprived,)),
    Scenario("consensus_commission", "L5", "Data-first commission + cross-party consensus", "lever", (build_consensus_through_data_first_commission,)),
    Scenario("compensation", "L6", "Compensate above-line losers", "lever", (compensate_above_line_losers,)),
    Scenario("credible_guarantees", "L7", "Guarantee untouched protections", "lever", (guarantee_untouched_protections,)),
    Scenario("internet_shutdown", "B1", "Internet shutdowns", "suppression", (shut_down_internet,)),
    Scenario("heavy_policing", "B2", "Heavy policing and mass arrests", "suppression", (deploy_heavy_policing,)),
    Scenario("managed_transition", "C1", "Managed transition package (L1+L2+L5+L6+L7)", "package", MANAGED_TRANSITION),
    Scenario("hybrid_package", "C2", "Hybrid design package (L3+L4+L1+L5+L6+L7)", "package", HYBRID_DESIGN),
    Scenario("symbolic_only_validation", "V", "Validation: symbolic threat only (2018-type shock)", "reference", (remove_all_material_loss,)),
)

SCENARIO_BY_KEY = {scenario.key: scenario for scenario in SCENARIOS}


def hybrid_with_weaker_symbolic_relief(symbolic_threat_retained: float) -> Callable[[ProtestModelParameters], None]:
    def keep_caste_subquotas_retaining(parameters: ProtestModelParameters) -> None:
        keep_caste_subquotas_with_income_filter(parameters, symbolic_threat_retained)

    return keep_caste_subquotas_retaining
