"""How much do the grounded results depend on the two assumptions no episode identifies?

  R: the reform's symbolic threat relative to the 2018 dilution of the Atrocities Act (R = 1 in the main analysis).
  Party backing of the protest against the reform: 0.6 in the main analysis (the level coded for 2024, when opposition
  parties backed the bandh), against 0.2 for 2018 (locally organized).

For each combination, the grounded baseline and a set of levers are run on paired worlds. The question is whether the
level of protest (which should move a lot) and the ranking of levers (which the paper relies on) move together.

The material weight w_m is a third such assumption: the episodes involve no material change, so calibration says nothing
about it. A second sweep multiplies w_m (at R = 1, backing 0.6) and records where a material lever becomes the strongest.
"""
import argparse
import json
from pathlib import Path

from protest_simulation import SCENARIO_BY_KEY
from protest_simulation.episode_simulation import party_amplifier
from protest_simulation.monte_carlo import CAMPAIGN_SEED_OFFSET, paired_effects_from_runs, paired_worlds
from protest_simulation.parallel import run_campaigns
from protest_simulation.specifications import get_specification

RESULTS_FOLDER = Path(__file__).resolve().parent.parent / "results"
SHOCK_RATIOS = (0.5, 1.0, 1.5)
PARTY_BACKING = (0.2, 0.6)
MATERIAL_WEIGHT_MULTIPLIERS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
MATERIAL_LEVERS = ("grandfathering", "seat_expansion", "compensation")
COMPARED = ("grandfathering", "seat_expansion", "hybrid_caste_subquotas", "consensus_commission", "compensation",
            "credible_guarantees", "heavy_policing", "managed_transition")


def sampler_with(base_sampler, shock_ratio, backing, reference_amplifier):
    def sample(parameters, random_generator, mean_threshold_standard_deviation=0.0):
        world = base_sampler(parameters, random_generator, mean_threshold_standard_deviation)
        world.symbolic_threat_by_group = world.symbolic_threat_by_group * shock_ratio
        world.reform_shock_ratio *= shock_ratio
        world.opposition_party_amplifier *= party_amplifier(backing, reference_amplifier) / reference_amplifier
        return world
    return sample


def run(population, base, sampler, scenario, run_count):
    interventions = () if scenario == "baseline" else SCENARIO_BY_KEY[scenario].interventions
    worlds = paired_worlds(interventions, run_count, base_parameters=base, world_sampler=sampler)
    outcomes = run_campaigns(population, [(world, CAMPAIGN_SEED_OFFSET + index) for index, world in enumerate(worlds)])
    return {"peak_day_protesters": [o.peak_day_protesters for o in outcomes],
            "cumulative_unique_protesters": [o.cumulative_unique_protesters for o in outcomes],
            "total_deaths": [o.total_deaths for o in outcomes],
            "conceded": [o.conceded_on_day is not None for o in outcomes]}


def median(values):
    values = sorted(values)
    return values[len(values) // 2] if len(values) % 2 else 0.5 * (values[len(values) // 2 - 1] + values[len(values) // 2])


def main():
    parser = argparse.ArgumentParser(description="Vary the reform's shock size and party backing in the grounded specification.")
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--agents", type=int, default=120_000)
    arguments = parser.parse_args()
    specification = get_specification("grounded")
    population = specification.build_population(arguments.agents)
    base = specification.base_parameters
    reference_amplifier = base.opposition_party_amplifier
    cells = []
    for shock_ratio in SHOCK_RATIOS:
        for backing in PARTY_BACKING:
            sampler = sampler_with(specification.world_sampler, shock_ratio, backing, reference_amplifier)
            baseline = run(population, base, sampler, "baseline", arguments.runs)
            effects = {scenario: paired_effects_from_runs(run(population, base, sampler, scenario, arguments.runs), baseline)
                       for scenario in COMPARED}
            singles = [s for s in COMPARED if s not in ("heavy_policing", "managed_transition")]
            order = sorted(singles, key=lambda s: effects[s]["peak"]["change_percent"])
            cell = {"shock_ratio": shock_ratio, "party_backing": backing,
                    "baseline_median_peak_lakh": round(median(baseline["peak_day_protesters"]) / 1e5, 1),
                    "baseline_median_cumulative_crore": round(median(baseline["cumulative_unique_protesters"]) / 1e7, 2),
                    "baseline_median_deaths": median(baseline["total_deaths"]),
                    "baseline_share_conceded": sum(baseline["conceded"]) / len(baseline["conceded"]),
                    "paired_effects": effects, "single_lever_order_by_peak_change": order}
            cells.append(cell)
            print(f"R {shock_ratio} backing {backing}: peak {cell['baseline_median_peak_lakh']} lakh, deaths {cell['baseline_median_deaths']}, "
                  f"conceded {cell['baseline_share_conceded']:.0%} | " + " > ".join(order), flush=True)
    material_sweep = []
    sampler = sampler_with(specification.world_sampler, 1.0, 0.6, reference_amplifier)
    for multiplier in MATERIAL_WEIGHT_MULTIPLIERS:
        weighted = base.copy()
        weighted.material_loss_weight = base.material_loss_weight * multiplier
        baseline = run(population, weighted, sampler, "baseline", arguments.runs)
        singles = [s for s in COMPARED if s not in ("heavy_policing", "managed_transition")]
        effects = {scenario: paired_effects_from_runs(run(population, weighted, sampler, scenario, arguments.runs), baseline)
                   for scenario in singles}
        order = sorted(singles, key=lambda s: effects[s]["peak"]["change_percent"])
        point = {"material_weight_multiplier": multiplier, "material_loss_weight": weighted.material_loss_weight,
                 "baseline_median_peak_lakh": round(median(baseline["peak_day_protesters"]) / 1e5, 1),
                 "peak_change_percent": {s: effects[s]["peak"]["change_percent"] for s in singles},
                 "single_lever_order_by_peak_change": order, "material_lever_is_strongest": order[0] in MATERIAL_LEVERS}
        material_sweep.append(point)
        print(f"w_m x{multiplier}: peak {point['baseline_median_peak_lakh']} lakh | " + " > ".join(order), flush=True)
    RESULTS_FOLDER.mkdir(exist_ok=True)
    path = RESULTS_FOLDER / "grounded_assumptions.json"
    path.write_text(json.dumps({"runs": arguments.runs, "agents": arguments.agents, "cells": cells,
                                "material_weight_sweep": material_sweep}, indent=1))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
