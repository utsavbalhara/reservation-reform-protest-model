"""The grounded specification: the stylized model with its assumed inputs replaced wherever data allow.

What changes relative to the stylized reference model, and where each input comes from:
  - Population: 640 Census 2011 districts; state OBC shares from NFHS-5; neighbourhoods within state and group blocks
    with a share h of same-group neighbourhoods (geography.py). Eligibility follows the actual rules: part of the
    above-line OBC population is non-creamy today because salary and farm income are excluded (prior share creamy 0.3-0.7),
    and part of the below-line General population fails the EWS asset tests (prior 0.05-0.25).
  - Material change per eligibility segment: from the merged-pool allocation model for IIT seats (merit_allocation/,
    results/merged_pool_allocation.json), in units of the loss of an above-line SC candidate (= 1), per candidate. The
    seat-expansion and caste-quota-with-income-filter levers get their own tables from the same model.
  - Dynamics: the mean threshold, threshold spread, neighbourhood mixing, bandh-day death rate and dispersion come from the
    parameter sets that survive history matching to four episodes (results/episode_calibration_nroy_samples.json).
    Each Monte Carlo run draws one surviving set.
  - Reform shock size: the reform's symbolic threat is the 2018 episode's estimated magnitude times R. R = 1 is the
    assumption behind the original validation (the reform threatens caste recognition about as much as the 2018 dilution);
    R is not identified by any data and is varied in sensitivity analysis.
  - Structure: organizations call bandhs when they expect a credible turnout; the government may concede under pressure,
    after which the General category may counter-mobilize; deaths are split into police-attributed and other deaths.
Everything not listed keeps its stylized value and remains an assumption.
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from .geography import build_synthetic_india_by_district
from .model_parameters import BASELINE_ABRUPT_INCOME_ONLY_SWITCH, ProtestModelParameters
from .monte_carlo import WORLD_DRAW_SEED_OFFSET
from .parameter_uncertainty import draw_plausible_world
from .specifications import SPECIFICATIONS, Specification

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
ALLOCATION_RESULTS = REPOSITORY_ROOT / "results" / "merged_pool_allocation.json"
CALIBRATION_SAMPLES = REPOSITORY_ROOT / "results" / "episode_calibration_nroy_samples.json"
REFORM_TO_2018_SHOCK_RATIO = 1.0
ORDINARY_TO_BANDH_DEATH_RATE = 0.1
OBC_CREAMY_SHARE = 0.5
EWS_ASSET_EXCLUSION_SHARE = 0.15
DEFAULT_MIXING = 0.8

# How each allocation segment maps onto the protest model's eligibility segments.
SEGMENT_MAP = {"SC_above": "SC_above", "SC_below": "SC_below", "ST_above": "ST_above", "ST_below": "ST_below",
               "OBC_above_ncl": "OBC-NCL_above", "OBC_below": "OBC-NCL_below", "General_below_ews": "GEN-EWS",
               "General_below_asset_excluded": "GEN_below", "General_above": "GEN_above", "OBC_creamy": "GEN_above"}


@lru_cache(maxsize=None)
def material_change_by_segment(case: str = "reserved_first|0.595") -> dict:
    """Median per-candidate material change for each protest-model segment (above-line SC = 1; positive = loss)."""
    summary = json.loads(ALLOCATION_RESULTS.read_text())["summary"][case]
    return {model_segment: float(summary[allocation_segment]["relative_material_change"]["median"])
            for model_segment, allocation_segment in SEGMENT_MAP.items()}


# Levers that are themselves allocation rules, and the allocation-model case that gives their material changes.
LEVER_CASES = {"seat_expansion": "lever|merged_pool_expanded", "hybrid_caste_subquotas": "lever|caste_income_filter"}


def lever_material_tables() -> dict:
    """Material-change tables for the allocation-rule levers (L2 seat expansion, L3 caste quotas with an income filter),
    relative to the same unit as the reform table (the reform's loss for an above-line SC candidate)."""
    summary = json.loads(ALLOCATION_RESULTS.read_text())["summary"]
    return {lever: material_change_by_segment(case) for lever, case in LEVER_CASES.items() if case in summary}


@lru_cache(maxsize=1)
def calibrated_samples() -> list:
    return json.loads(CALIBRATION_SAMPLES.read_text())


def grounded_base_parameters() -> ProtestModelParameters:
    parameters = BASELINE_ABRUPT_INCOME_ONLY_SWITCH.copy()
    parameters.material_change_by_segment = dict(material_change_by_segment())
    parameters.reform_material_change_by_segment = dict(parameters.material_change_by_segment)
    parameters.lever_material_tables = lever_material_tables() or None
    parameters.random_streams = "split"
    parameters.bandh_schedule = "endogenous"
    parameters.concession_rule = True
    parameters.counter_mobilization_symbolic_threat = 0.3
    parameters.death_model = "negative_binomial_split"
    samples = calibrated_samples()
    central = {name: float(np.median([sample[name] for sample in samples])) for name in samples[0]}
    apply_calibrated_sample(parameters, central)
    parameters.episode_2024_threat_relative_to_2018 = float(np.median(
        [sample["magnitude_sc_st_bharat_bandh_2024"] / sample["magnitude_sc_st_bharat_bandh_2018"] for sample in samples]))
    return parameters


def apply_calibrated_sample(parameters: ProtestModelParameters, sample: dict) -> None:
    parameters.mean_participation_threshold = sample["mean_participation_threshold"]
    parameters.participation_threshold_spread = sample["participation_threshold_spread"]
    magnitude = sample["magnitude_sc_st_bharat_bandh_2018"] * REFORM_TO_2018_SHOCK_RATIO
    parameters.reform_shock_ratio = REFORM_TO_2018_SHOCK_RATIO
    parameters.episode_2024_threat_relative_to_2018 = sample["magnitude_sc_st_bharat_bandh_2024"] / sample["magnitude_sc_st_bharat_bandh_2018"]
    parameters.symbolic_threat_by_group = BASELINE_ABRUPT_INCOME_ONLY_SWITCH.symbolic_threat_by_group * magnitude
    bandh_rate = 10 ** sample["log10_bandh_deaths_per_crore"]
    parameters.deaths_per_crore_protester_days = bandh_rate * ORDINARY_TO_BANDH_DEATH_RATE
    parameters.bandh_day_death_rate_multiplier = 1.0 / ORDINARY_TO_BANDH_DEATH_RATE
    parameters.death_dispersion = sample["death_dispersion"]


def draw_grounded_world(parameters: ProtestModelParameters, random_generator: np.random.Generator,
                        mean_threshold_standard_deviation: float = 0.0) -> ProtestModelParameters:
    """One Monte Carlo world: a surviving calibration set, plus the stylized draws for parameters no episode informs
    (loss aversion, party amplifier, contagion strengths). The mean-threshold jitter is replaced by the calibration spread."""
    world = draw_plausible_world(parameters, random_generator, 0.0)
    samples = calibrated_samples()
    sample = samples[int(random_generator.integers(len(samples)))]
    apply_calibrated_sample(world, sample)
    world.symbolic_threat_by_group = world.symbolic_threat_by_group * random_generator.uniform(0.8, 1.2)
    return world


def build_grounded_population(agent_count: int, random_generator: np.random.Generator, **options):
    samples = calibrated_samples()
    mixing = float(np.median([sample["same_group_neighbourhood_share"] for sample in samples])) if samples else DEFAULT_MIXING
    return build_synthetic_india_by_district(agent_count, random_generator, same_group_neighbourhood_share=round(mixing, 2),
                                             obc_creamy_share_of_above_line=OBC_CREAMY_SHARE,
                                             ews_asset_exclusion_share=EWS_ASSET_EXCLUSION_SHARE, **options)


if ALLOCATION_RESULTS.exists() and CALIBRATION_SAMPLES.exists():
    SPECIFICATIONS["grounded"] = Specification(
        name="grounded",
        description="District population, allocation-derived material changes, episode-calibrated dynamics, endogenous "
                    "bandhs, concession and split deaths.",
        base_parameters=grounded_base_parameters(),
        world_sampler=draw_grounded_world,
        population_builder=build_grounded_population,
    )
