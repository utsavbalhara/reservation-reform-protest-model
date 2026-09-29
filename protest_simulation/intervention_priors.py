"""Uncertainty over how each intervention enters the model.

The scenario catalogue in policy_interventions.py fixes each lever's effect at one value (for example, credible
guarantees cut symbolic threat by 15%). Those values are judgements, not estimates. Here every such value gets a
uniform prior wide enough to express that ignorance; the reference value sits inside every range. Drawing the
mapping once per Monte Carlo run and applying it to every lever in that run keeps the comparison paired.

Suppression gets the same treatment, with ranges that run from pure deterrence (force keeps people home and deaths
intimidate) to strong backfire (force kills and deaths mobilize), so that backfire has to emerge from the dynamics
instead of being assumed.
"""
import numpy as np

from .policy_interventions import income_filter_threat_retained_from_episodes
from .synthetic_population import OBC

LEVER_PRIORS = {
    "grandfathering": {"material_factor": (0.1, 0.6, 0.3), "symbolic_factor": (0.85, 1.0, 0.95)},
    "seat_expansion": {"below_line_loss_removed": (0.6, 1.0, 1.0)},
    "hybrid_caste_subquotas": {"symbolic_retained": (0.3, 0.9, 0.4)},
    "sub_classification": {"deprived_tier_gain": (0.2, 0.8, 0.5), "deprived_tier_symbolic_share": (0.2, 0.8, 0.3),
                           "better_off_tier_symbolic_factor": (1.0, 1.5, 1.2)},
    "consensus_commission": {"amplifier_factor": (0.5, 0.9, 1 / 1.5), "symbolic_factor": (0.7, 1.0, 0.8)},
    "compensation": {"above_line_loss_factor": (0.25, 0.75, 0.5)},
    "credible_guarantees": {"symbolic_factor": (0.7, 1.0, 0.85)},
}
SUPPRESSION_PRIORS = {
    "heavy_policing": {"turnout_cost": (0.1, 0.8, 0.4), "death_multiplier": (1.0, 5.0, 3.0), "death_response_factor": (-0.5, 2.0, 1.5)},
    "internet_shutdown": {"coordination_factor": (0.4, 0.9, 0.6), "bandh_mobilization_factor": (0.6, 1.0, 0.8),
                          "violence_factor": (1.0, 3.0, 2.5), "death_response_factor": (0.5, 1.5, 1.0)},
}
SINGLE_LEVERS = tuple(LEVER_PRIORS)
BELOW_LINE = ("SC_below", "ST_below", "OBC_below")
# Levers that replace the allocation rule. In a package they are applied first, so that levers scaling losses
# (grandfathering, compensation) act on the losses of the rule actually adopted.
ALLOCATION_RULE_LEVERS = ("hybrid_caste_subquotas", "seat_expansion")
PACKAGES = {
    "managed_transition": ("grandfathering", "seat_expansion", "consensus_commission", "compensation", "credible_guarantees"),
    "hybrid_package": ("hybrid_caste_subquotas", "sub_classification", "grandfathering", "consensus_commission", "compensation",
                       "credible_guarantees"),
}
MAPPING_SEED_OFFSET = 3000


def reference_mapping() -> dict:
    return {lever: {name: bounds[2] for name, bounds in values.items()} for lever, values in {**LEVER_PRIORS, **SUPPRESSION_PRIORS}.items()}


def sample_mapping(random_generator: np.random.Generator) -> dict:
    """One draw of every lever's and suppression measure's effect size, in a fixed order."""
    mapping = {}
    for lever, values in {**LEVER_PRIORS, **SUPPRESSION_PRIORS}.items():
        mapping[lever] = {name: float(random_generator.uniform(low, high)) for name, (low, high, _) in values.items()}
    return mapping


def mapping_for_run(run_index: int) -> dict:
    return sample_mapping(np.random.default_rng(MAPPING_SEED_OFFSET + run_index))


def apply_lever(parameters, lever: str, mapping: dict) -> None:
    values = mapping[lever]
    if lever == "grandfathering":
        for name in ("material_loss_sc_st_above_income_line", "material_loss_sc_st_below_income_line", "material_loss_obc_below_income_line"):
            setattr(parameters, name, getattr(parameters, name) * values["material_factor"])
        _scale_segment_losses(parameters, lambda segment, value: value * values["material_factor"] if value > 0 else value)
        parameters.symbolic_threat_by_group = parameters.symbolic_threat_by_group * values["symbolic_factor"]
    elif lever == "seat_expansion":
        keep = 1 - values["below_line_loss_removed"]
        parameters.material_loss_sc_st_below_income_line *= keep
        parameters.material_loss_obc_below_income_line *= keep
        if not _move_towards_lever_table(parameters, lever, values["below_line_loss_removed"]):
            _scale_segment_losses(parameters, lambda segment, value: value * keep if segment in BELOW_LINE and value > 0 else value)
    elif lever == "hybrid_caste_subquotas":
        symbolic = parameters.symbolic_threat_by_group.copy()
        symbolic[: OBC + 1] *= values["symbolic_retained"]
        # Grounded specification: SC and ST keep the share implied by the 2024 episode for this world's calibration set,
        # instead of the prior; the prior still applies to OBC, which the 2024 episode did not concern.
        from_episodes = income_filter_threat_retained_from_episodes(parameters)
        if from_episodes is not None:
            symbolic[:OBC] = parameters.symbolic_threat_by_group[:OBC] * from_episodes
        parameters.symbolic_threat_by_group = symbolic
        parameters.material_loss_sc_st_below_income_line = 0.0
        parameters.material_loss_obc_below_income_line = 0.0
        if not _move_towards_lever_table(parameters, lever, 1.0):
            _scale_segment_losses(parameters, lambda segment, value: 0.0 if segment in BELOW_LINE else value)
    elif lever == "sub_classification":
        parameters.sub_classification_gain_for_most_deprived_tier = values["deprived_tier_gain"]
        parameters.most_deprived_tier_share_of_symbolic_threat = values["deprived_tier_symbolic_share"]
        parameters.better_off_tier_symbolic_threat_factor = values["better_off_tier_symbolic_factor"]
    elif lever == "consensus_commission":
        parameters.opposition_party_amplifier *= values["amplifier_factor"]
        parameters.symbolic_threat_by_group = parameters.symbolic_threat_by_group * values["symbolic_factor"]
    elif lever == "compensation":
        parameters.material_loss_sc_st_above_income_line *= values["above_line_loss_factor"]
        _scale_segment_losses(parameters, lambda segment, value: value * values["above_line_loss_factor"]
                              if segment in ("SC_above", "ST_above", "OBC_above_ncl") and value > 0 else value)
    elif lever == "credible_guarantees":
        parameters.symbolic_threat_by_group = parameters.symbolic_threat_by_group * values["symbolic_factor"]
    elif lever == "heavy_policing":
        parameters.heavy_policing_turnout_cost = values["turnout_cost"]
        if parameters.death_model == "poisson":
            parameters.deaths_per_crore_protester_days *= values["death_multiplier"]
        else:
            # Only police-attributed deaths rise; scale that component so total deaths rise by the same multiplier.
            share = parameters.police_attributed_death_share
            parameters.police_death_multiplier *= (values["death_multiplier"] - (1 - share)) / share
        parameters.death_response *= values["death_response_factor"]
    elif lever == "internet_shutdown":
        parameters.internet_shutdown_active = True
        parameters.internet_shutdown_coordination_factor = values["coordination_factor"]
        parameters.internet_shutdown_bandh_mobilization_factor = values["bandh_mobilization_factor"]
        parameters.internet_shutdown_violence_factor = values["violence_factor"]
        parameters.death_response *= values["death_response_factor"]
    else:
        raise KeyError(lever)


def _move_towards_lever_table(parameters, lever: str, fraction: float) -> bool:
    """With allocation-derived tables (grounded specification), an allocation-rule lever moves each segment's material
    change from the reform's value towards the lever's own table by `fraction` (1 = the allocation model's full result).
    Applied to the current values as a shift, so levers that scale losses afterwards act on the result."""
    tables = parameters.lever_material_tables
    if parameters.material_change_by_segment is None or not tables or lever not in tables:
        return False
    reform = parameters.reform_material_change_by_segment or parameters.material_change_by_segment
    parameters.material_change_by_segment = {
        segment: value + fraction * (tables[lever][segment] - reform[segment])
        for segment, value in parameters.material_change_by_segment.items()}
    return True


def _scale_segment_losses(parameters, rule) -> None:
    if parameters.material_change_by_segment is not None:
        parameters.material_change_by_segment = {segment: rule(segment, value) for segment, value in parameters.material_change_by_segment.items()}


def interventions_for(scenario_key: str, mapping: dict) -> tuple:
    levers = PACKAGES.get(scenario_key, (scenario_key,))
    levers = tuple(sorted(levers, key=lambda lever: lever not in ALLOCATION_RULE_LEVERS))
    return tuple((lambda parameters, lever=lever: apply_lever(parameters, lever, mapping)) for lever in levers)
