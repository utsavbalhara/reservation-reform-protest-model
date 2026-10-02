"""Structural ensemble: does the lever ranking survive changes to the model's structure, not just its parameters?

Each variant switches on one structural alternative (or all of them together) from model_parameters.py and
synthetic_population.py, recalibrates the mean threshold so the baseline again hits the turnout target, and reruns the
reform and every single lever on paired worlds. The ranking is then read across variants. A result that holds only under
the reference structure is a property of that structure; one that holds across the ensemble is a property of the
mechanisms the variants share.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from protest_simulation import SCENARIO_BY_KEY
from protest_simulation.calibration import calibrate_mean_threshold
from protest_simulation.intervention_priors import SINGLE_LEVERS
from protest_simulation.monte_carlo import CAMPAIGN_SEED_OFFSET, POPULATION_SEED, paired_effects_from_runs, paired_worlds
from protest_simulation.parallel import run_campaigns
from protest_simulation.specifications import get_specification
from protest_simulation.synthetic_population import build_synthetic_india

RESULTS_FOLDER = Path(__file__).resolve().parent.parent / "results"
COMPARED = ("symbolic_only_validation",) + SINGLE_LEVERS + ("heavy_policing",)

# With the legal eligibility rules, two segments change status that the stylized material function cannot see: OBC
# families above the line but non-creamy today lose eligibility (the loss of an above-line SC/ST family, 1.0), and
# below-line General families that fail the EWS asset tests gain it (the same size of change, as a gain). Every other
# segment keeps its stylized value.
STYLIZED_LEGAL_TABLE = {"SC_above": 1.0, "SC_below": 0.35, "ST_above": 1.0, "ST_below": 0.35, "OBC_creamy": 0.0,
                        "OBC_above_ncl": 1.0, "OBC_below": 0.25, "General_above": 0.0, "General_below_ews": -0.15,
                        "General_below_asset_excluded": -1.0}

# name -> (parameter changes, population options, description)
VARIANTS = {
    "reference": ({}, {}, "Reference structure"),
    "split_streams": ({"random_streams": "split"}, {}, "Separate random streams for decisions, deaths and events"),
    "logistic_thresholds": ({"threshold_distribution": "logistic"}, {}, "Logistic (heavier-tailed) threshold distribution"),
    "activist_core": ({"threshold_distribution": "activist_mixture"}, {}, "3% activist core with low thresholds"),
    "endogenous_bandhs": ({"bandh_schedule": "endogenous"}, {}, "Organizations call bandhs when turnout is credible"),
    "concession_and_backlash": ({"concession_rule": True, "counter_mobilization_symbolic_threat": 0.3}, {},
                                "Government may concede; General category counter-mobilizes"),
    "overdispersed_split_deaths": ({"death_model": "negative_binomial_split", "death_dispersion": 1.0}, {},
                                   "Negative-binomial deaths, split into police-attributed and other"),
    "deaths_deter": ({"death_response": -0.5}, {}, "Deaths deter rather than mobilize"),
    "mixed_neighbourhoods": ({}, {"same_group_neighbourhood_share": 0.7}, "30% of neighbourhoods mixed across groups"),
    "legal_eligibility": ({"material_change_by_segment": STYLIZED_LEGAL_TABLE}, {"obc_creamy_share_of_above_line": 0.5, "ews_asset_exclusion_share": 0.15},
                          "Half of above-line OBC non-creamy today; 15% of below-line General fail EWS asset tests"),
    "all_combined": ({"random_streams": "split", "threshold_distribution": "activist_mixture", "bandh_schedule": "endogenous",
                      "concession_rule": True, "counter_mobilization_symbolic_threat": 0.3,
                      "death_model": "negative_binomial_split", "death_dispersion": 1.0, "material_change_by_segment": STYLIZED_LEGAL_TABLE},
                     {"same_group_neighbourhood_share": 0.7, "obc_creamy_share_of_above_line": 0.5, "ews_asset_exclusion_share": 0.15},
                     "All alternatives above except deterrence"),
}


# Grounded specification: it already has split streams, endogenous bandhs, concession and backlash, split deaths, mixed
# neighbourhoods and the legal eligibility rules, so its variants switch those off or change the remaining structure. The
# mean threshold is not recalibrated: it comes from the parameter sets that survive history matching, and no turnout
# target exists to recalibrate against. The level of protest therefore moves with the variant; the ranking is the object.
GROUNDED_VARIANTS = {
    "reference": ({}, "Grounded reference structure"),
    "logistic_thresholds": ({"threshold_distribution": "logistic"}, "Logistic (heavier-tailed) threshold distribution"),
    "activist_core": ({"threshold_distribution": "activist_mixture"}, "3% activist core with low thresholds"),
    "fixed_bandh_days": ({"bandh_schedule": "fixed"}, "Bandhs on fixed days 3, 13 and 26 instead of called on credible turnout"),
    "no_concession": ({"concession_rule": False, "counter_mobilization_symbolic_threat": 0.0}, "Government never concedes; no backlash"),
    "slow_concession": ({"concession_max_daily_hazard": 0.03, "concession_pressure_midpoint": 2.0},
                        "Government concedes slowly (at most 3% a day; midpoint pressure 2)"),
    "poisson_deaths": ({"death_model": "poisson"}, "Poisson deaths, all of which feed the martyr effect"),
    "deaths_deter": ({"death_response": -0.5}, "Deaths deter rather than mobilize"),
    "single_stream": ({"random_streams": "single"}, "One random stream for decisions, deaths and events"),
    "no_martyr_effect": ({"death_response": 0.0}, "Deaths have no effect on turnout"),
    "stylized_material_table": ({"material_change_by_segment": STYLIZED_LEGAL_TABLE, "lever_material_tables": None,
                                 "reform_material_change_by_segment": None},
                                "Assumed per-segment material changes (stylized values) instead of the allocation model"),
}


def sampler_with_changes(base_sampler, changes):
    """Structural changes applied after each world is drawn, so the grounded sampler's calibrated values are kept."""
    def sample(parameters, random_generator, mean_threshold_standard_deviation=0.0):
        world = base_sampler(parameters, random_generator, mean_threshold_standard_deviation)
        for field_name, value in changes.items():
            setattr(world, field_name, dict(value) if isinstance(value, dict) else value)
        return world
    return sample


def run_scenario(population, world_sampler, base, scenario, run_count):
    interventions = () if scenario == "baseline" else SCENARIO_BY_KEY[scenario].interventions
    worlds = paired_worlds(interventions, run_count, base_parameters=base, world_sampler=world_sampler)
    outcomes = run_campaigns(population, [(world, CAMPAIGN_SEED_OFFSET + index) for index, world in enumerate(worlds)])
    return {"peak_day_protesters": [o.peak_day_protesters for o in outcomes],
            "cumulative_unique_protesters": [o.cumulative_unique_protesters for o in outcomes],
            "total_deaths": [o.total_deaths for o in outcomes]}


def main():
    parser = argparse.ArgumentParser(description="Rerun the lever comparison under alternative model structures.")
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--calibration-runs", type=int, default=20)
    parser.add_argument("--agents", type=int, default=120_000)
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--specification", default="stylized", choices=("stylized", "grounded"))
    arguments = parser.parse_args()
    if arguments.specification == "grounded":
        run_grounded(arguments)
        return
    arguments.variants = arguments.variants or list(VARIANTS)

    specification = get_specification("stylized")
    results = {}
    for name in arguments.variants:
        changes, population_options, description = VARIANTS[name]
        population = build_synthetic_india(arguments.agents, np.random.default_rng(POPULATION_SEED), **population_options)
        base = specification.base_parameters.copy()
        for field, value in changes.items():
            setattr(base, field, value)
        threshold, calibrated_peak, _ = calibrate_mean_threshold(population, base, run_count=arguments.calibration_runs,
                                                                 world_sampler=specification.world_sampler)
        base.mean_participation_threshold = threshold
        baseline = run_scenario(population, specification.world_sampler, base, "baseline", arguments.runs)
        effects = {scenario: paired_effects_from_runs(run_scenario(population, specification.world_sampler, base, scenario, arguments.runs),
                                                      baseline)
                   for scenario in COMPARED}
        order = sorted(SINGLE_LEVERS, key=lambda lever: effects[lever]["peak"]["change_percent"])
        results[name] = {"description": description, "parameter_changes": changes, "population_options": population_options,
                         "recalibrated_mean_threshold": round(threshold, 4), "calibrated_median_peak_lakh": round(calibrated_peak, 1),
                         "baseline_median_peak_lakh": round(float(np.median(baseline["peak_day_protesters"])) / 1e5, 1),
                         "baseline_median_deaths": float(np.median(baseline["total_deaths"])),
                         "paired_effects": effects, "single_lever_order_by_peak_change": order}
        print(f"{name:>28}: theta {threshold:.3f} | order {' > '.join(order)} | "
              + ", ".join(f"{s} {effects[s]['peak']['change_percent']:+.0f}%" for s in COMPARED), flush=True)

    RESULTS_FOLDER.mkdir(exist_ok=True)
    path = RESULTS_FOLDER / "structural_ensemble.json"
    if set(arguments.variants) != set(VARIANTS) and path.exists():
        # A partial rerun replaces only the variants it ran and keeps the others, in catalogue order.
        previous = json.loads(path.read_text())["variants"]
        results = {name: results.get(name, previous.get(name)) for name in VARIANTS if name in results or name in previous}
    top = {name: value["single_lever_order_by_peak_change"][0] for name, value in results.items()}
    bottom = {name: value["single_lever_order_by_peak_change"][-1] for name, value in results.items()}
    output = {"runs_per_scenario": arguments.runs, "agents": arguments.agents, "variants": results,
              "strongest_lever_by_variant": top, "weakest_lever_by_variant": bottom,
              "share_of_variants_where_strongest": {lever: round(list(top.values()).count(lever) / len(top), 3) for lever in SINGLE_LEVERS}}
    path.write_text(json.dumps(output, indent=1))
    print(json.dumps({k: output[k] for k in ("strongest_lever_by_variant", "weakest_lever_by_variant")}, indent=1))
    print(f"Saved {path}")


def run_grounded(arguments):
    specification = get_specification("grounded")
    population = specification.build_population(arguments.agents)
    names = arguments.variants or list(GROUNDED_VARIANTS)
    results = {}
    for name in names:
        changes, description = GROUNDED_VARIANTS[name]
        sampler = sampler_with_changes(specification.world_sampler, changes)
        base = specification.base_parameters
        baseline = run_scenario(population, sampler, base, "baseline", arguments.runs)
        effects = {scenario: paired_effects_from_runs(run_scenario(population, sampler, base, scenario, arguments.runs), baseline)
                   for scenario in COMPARED}
        order = sorted(SINGLE_LEVERS, key=lambda lever: effects[lever]["peak"]["change_percent"])
        results[name] = {"description": description, "parameter_changes": {k: v for k, v in changes.items() if not isinstance(v, dict)},
                         "baseline_median_peak_lakh": round(float(np.median(baseline["peak_day_protesters"])) / 1e5, 1),
                         "baseline_median_deaths": float(np.median(baseline["total_deaths"])),
                         "paired_effects": effects, "single_lever_order_by_peak_change": order}
        print(f"{name:>24}: peak {results[name]['baseline_median_peak_lakh']} lakh | order {' > '.join(order)} | "
              + ", ".join(f"{s} {effects[s]['peak']['change_percent']:+.0f}%" for s in COMPARED), flush=True)
    top = {name: value["single_lever_order_by_peak_change"][0] for name, value in results.items()}
    bottom = {name: value["single_lever_order_by_peak_change"][-1] for name, value in results.items()}
    output = {"specification": "grounded", "runs_per_scenario": arguments.runs, "agents": arguments.agents, "variants": results,
              "strongest_lever_by_variant": top, "weakest_lever_by_variant": bottom,
              "share_of_variants_where_strongest": {lever: round(list(top.values()).count(lever) / len(top), 3) for lever in SINGLE_LEVERS}}
    path = RESULTS_FOLDER / "structural_ensemble_grounded.json"
    path.write_text(json.dumps(output, indent=1))
    print(json.dumps({k: output[k] for k in ("strongest_lever_by_variant", "weakest_lever_by_variant")}, indent=1))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
