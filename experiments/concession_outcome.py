"""Concession as an outcome.

In the grounded specification the government concedes under pressure, and at the reference settings it does so in
every simulated campaign against the abrupt switch. For a government that wants the reform to survive, whether and when
it has to concede is the policy-relevant outcome, so this script reports, for every scenario, the share of runs in which
the government concedes, the median day of concession, and the peak and cumulative changes, under three concession
rules: the reference rule; a slow rule (at most 3% a day, with the pressure midpoint doubled); and no concession.

The concession rule is assumed, not calibrated: the historical episodes are simulated as two-day campaigns, so nothing
after day 2 is fitted. The three rules show how much the lever comparison depends on it.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from protest_simulation import SCENARIO_BY_KEY
from protest_simulation.monte_carlo import CAMPAIGN_SEED_OFFSET, paired_effects_from_runs, paired_worlds
from protest_simulation.parallel import run_campaigns
from protest_simulation.specifications import get_specification

RESULTS_FOLDER = Path(__file__).resolve().parent.parent / "results"
RULES = {
    "reference": {},
    "slow": {"concession_max_daily_hazard": 0.03, "concession_pressure_midpoint": 2.0},
    "none": {"concession_rule": False, "counter_mobilization_symbolic_threat": 0.0},
}
SCENARIOS = ("baseline", "grandfathering", "seat_expansion", "hybrid_caste_subquotas", "sub_classification", "consensus_commission",
             "compensation", "credible_guarantees", "internet_shutdown", "heavy_policing", "managed_transition", "hybrid_package")
LEVERS = SCENARIOS[1:8]


def sampler_with(base_sampler, changes):
    def sample(parameters, random_generator, mean_threshold_standard_deviation=0.0):
        world = base_sampler(parameters, random_generator, mean_threshold_standard_deviation)
        for name, value in changes.items():
            setattr(world, name, value)
        return world
    return sample


def run(population, base, sampler, scenario, run_count):
    interventions = () if scenario == "baseline" else SCENARIO_BY_KEY[scenario].interventions
    worlds = paired_worlds(interventions, run_count, base_parameters=base, world_sampler=sampler)
    outcomes = run_campaigns(population, [(world, CAMPAIGN_SEED_OFFSET + index) for index, world in enumerate(worlds)])
    return {"peak_day_protesters": [o.peak_day_protesters for o in outcomes],
            "cumulative_unique_protesters": [o.cumulative_unique_protesters for o in outcomes],
            "total_deaths": [o.total_deaths for o in outcomes],
            "conceded_on_day": [o.conceded_on_day for o in outcomes],
            "peak_day": [int(np.argmax(o.daily_protesters)) for o in outcomes],
            "first_bandh_day": [min(o.bandh_days) if o.bandh_days else None for o in outcomes]}


def main():
    parser = argparse.ArgumentParser(description="Concession as an outcome, under three concession rules.")
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--agents", type=int, default=120_000)
    arguments = parser.parse_args()
    specification = get_specification("grounded")
    population = specification.build_population(arguments.agents)
    output = {"runs": arguments.runs, "rules": {}}
    for rule, changes in RULES.items():
        sampler = sampler_with(specification.world_sampler, changes)
        raw = {scenario: run(population, specification.base_parameters, sampler, scenario, arguments.runs) for scenario in SCENARIOS}
        baseline = raw["baseline"]
        scenarios = {}
        for scenario, runs in raw.items():
            days = [day for day in runs["conceded_on_day"] if day is not None]
            record = {"share_conceded": round(len(days) / len(runs["conceded_on_day"]), 3),
                      "median_concession_day": float(np.median(days)) if days else None,
                      "share_conceded_by_day_10": round(sum(day <= 10 for day in days) / len(runs["conceded_on_day"]), 3),
                      "median_peak_lakh": round(float(np.median(runs["peak_day_protesters"])) / 1e5, 1),
                      "median_cumulative_crore": round(float(np.median(runs["cumulative_unique_protesters"])) / 1e7, 2),
                      "median_deaths": float(np.median(runs["total_deaths"])),
                      "share_peak_on_first_bandh_day": round(float(np.mean([b is not None and p == b for p, b in
                                                                            zip(runs["peak_day"], runs["first_bandh_day"])])), 3)}
            if scenario != "baseline":
                record["paired_effects"] = paired_effects_from_runs(runs, baseline)
            scenarios[scenario] = record
        # Rank single levers by how often the government still has to concede (fewer is better), then by how late.
        order = sorted(LEVERS, key=lambda s: (scenarios[s]["share_conceded"], -(scenarios[s]["median_concession_day"] or 99)))
        peak_order = sorted(LEVERS, key=lambda s: scenarios[s]["paired_effects"]["peak"]["change_percent"])
        output["rules"][rule] = {"changes": changes, "scenarios": scenarios, "lever_order_by_concession": order,
                                 "lever_order_by_peak": peak_order}
        print(f"[{rule}] concession order: {' > '.join(order)}", flush=True)
        for scenario in SCENARIOS:
            r = scenarios[scenario]
            print(f"   {scenario:>24} conceded {r['share_conceded']:.2f} day {r['median_concession_day']} peak {r['median_peak_lakh']} "
                  f"first-bandh-peak {r['share_peak_on_first_bandh_day']}", flush=True)
    RESULTS_FOLDER.mkdir(exist_ok=True)
    path = RESULTS_FOLDER / "concession_outcome.json"
    path.write_text(json.dumps(output, indent=1))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
