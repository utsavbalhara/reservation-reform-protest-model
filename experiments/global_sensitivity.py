"""Global sensitivity: Morris elementary-effects screening over the behavioural parameters.

The one-at-a-time analysis (sensitivity_analysis.py) moves each parameter by 25% around the reference point with all
others fixed, so it cannot see interactions or effects that only appear elsewhere in the parameter space. Morris screening
(Morris 1991; Campolongo, Cariboni and Saltelli 2007) walks r random trajectories through the whole box of plausible
values, changing one parameter at a time by a fixed step, and reports for each parameter and output:
  mu_star: mean absolute elementary effect (overall importance),
  sigma:   standard deviation of the elementary effects (non-linearity or interactions),
with bootstrap intervals for mu_star over trajectories. Because the mean threshold is recalibrated in the main analysis,
each point also re-uses the same campaign seed across the trajectory's steps and across scenarios (common random
numbers), so elementary effects are not dominated by simulation noise.

Outputs: the baseline's log peak, log cumulative participation and log(1 + deaths), and the paired percentage change in
peak for the three levers whose ranking the paper discusses (L1 grandfathering, L3 caste quotas with an income filter,
L5 consensus commission).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from protest_simulation.intervention_priors import interventions_for, reference_mapping
from protest_simulation.parallel import run_campaigns
from protest_simulation.specifications import get_specification

RESULTS_FOLDER = Path(__file__).resolve().parent.parent / "results"
LEVELS = 4
STEP = LEVELS / (2 * (LEVELS - 1))
CAMPAIGN_SEED = 7000

# Parameter -> (low, high); multiplicative ranges are applied to the reference value, additive ones are absolute.
FACTORS = {
    "mean_participation_threshold": ("add", -0.6, 0.6),
    "participation_threshold_spread": ("mul", 0.75, 1.25),
    "decision_noise": ("mul", 0.67, 1.5),
    "loss_aversion": ("mul", 0.7, 1.3),
    "material_loss_weight": ("mul", 0.67, 1.5),
    "symbolic_threat_weight": ("mul", 0.67, 1.5),
    "most_deprived_tier_share_of_symbolic_threat": ("abs", 0.6, 1.0),
    "opposition_party_amplifier": ("abs", 1.0, 2.0),
    "max_neighbourhood_influence": ("mul", 0.75, 1.25),
    "neighbourhood_turnout_at_saturation": ("mul", 0.5, 2.0),
    "max_national_visibility_influence": ("mul", 0.75, 1.25),
    "fatigue_per_protest_day": ("mul", 0.6, 1.4),
    "mobilization_on_bandh_days": ("mul", 0.75, 1.25),
    "deaths_per_crore_protester_days": ("mul", 0.5, 2.0),
    "symbolic_threat_rise_per_death": ("mul", 0.5, 2.0),
}
NAMES = tuple(FACTORS)
SCENARIOS = ("baseline", "grandfathering", "hybrid_caste_subquotas", "consensus_commission")


def trajectory(random_generator, k):
    """One Morris trajectory in the unit cube on a LEVELS-point grid: k + 1 points, each step moving one factor by STEP."""
    grid = np.arange(LEVELS) / (LEVELS - 1)
    start = random_generator.choice(grid[grid <= 1 - STEP + 1e-9], size=k)
    order = random_generator.permutation(k)
    signs = random_generator.choice([-1, 1], size=k)
    points, current = [start.copy()], start.copy()
    for j in order:
        # Move up when the start is in the lower part of the grid, otherwise down; the random sign alternates directions
        # where both are possible.
        direction = 1 if current[j] + STEP <= 1 + 1e-9 and (signs[j] > 0 or current[j] - STEP < -1e-9) else -1
        current = current.copy()
        current[j] += direction * STEP
        points.append(current)
    return np.array(points), order


def parameters_at(base, unit_point):
    parameters = base.copy()
    for value, name in zip(unit_point, NAMES):
        kind, low, high = FACTORS[name]
        x = low + value * (high - low)
        if kind == "add":
            setattr(parameters, name, getattr(base, name) + x)
        elif kind == "mul":
            setattr(parameters, name, getattr(base, name) * x)
        else:
            setattr(parameters, name, x)
    return parameters


def scenario_parameters(parameters, scenario, mapping):
    world = parameters.copy()
    if scenario != "baseline":
        for intervention in interventions_for(scenario, mapping):
            intervention(world)
    return world


def outputs_of(outcomes):
    baseline = outcomes["baseline"]
    # One person is added to every count so that points where nobody protests (possible at high thresholds) stay finite.
    values = {"log_peak": np.log1p(baseline.peak_day_protesters), "log_cumulative": np.log1p(baseline.cumulative_unique_protesters),
              "log1p_deaths": np.log1p(baseline.total_deaths)}
    for scenario in SCENARIOS[1:]:
        values[f"peak_change_percent:{scenario}"] = ((1 + outcomes[scenario].peak_day_protesters) / (1 + baseline.peak_day_protesters) - 1) * 100
    return values


def bootstrap_interval(values, random_generator, resamples=2000):
    values = np.asarray(values)
    draws = [np.mean(np.abs(random_generator.choice(values, len(values)))) for _ in range(resamples)]
    return [round(float(np.percentile(draws, 2.5)), 4), round(float(np.percentile(draws, 97.5)), 4)]


def main():
    parser = argparse.ArgumentParser(description="Morris screening of the behavioural parameters.")
    parser.add_argument("--specification", default="stylized")
    parser.add_argument("--trajectories", type=int, default=20)
    parser.add_argument("--agents", type=int, default=120_000)
    parser.add_argument("--seed", type=int, default=20261001)
    arguments = parser.parse_args()

    specification = get_specification(arguments.specification)
    population = specification.build_population(arguments.agents)
    mapping = reference_mapping()
    random_generator = np.random.default_rng(arguments.seed)
    k = len(NAMES)
    trajectories = [trajectory(random_generator, k) for _ in range(arguments.trajectories)]

    jobs, index = [], []
    for t, (points, _) in enumerate(trajectories):
        for p, point in enumerate(points):
            parameters = parameters_at(specification.base_parameters, point)
            for scenario in SCENARIOS:
                jobs.append((scenario_parameters(parameters, scenario, mapping), CAMPAIGN_SEED + t))
                index.append((t, p, scenario))
    outcomes = run_campaigns(population, jobs)
    by_point = {}
    for (t, p, scenario), outcome in zip(index, outcomes):
        by_point.setdefault((t, p), {})[scenario] = outcome

    output_names = list(outputs_of(by_point[(0, 0)]))
    effects = {name: {output: [] for output in output_names} for name in NAMES}
    for t, (points, order) in enumerate(trajectories):
        values = [outputs_of(by_point[(t, p)]) for p in range(len(points))]
        for step, j in enumerate(order):
            delta = points[step + 1][j] - points[step][j]
            for output in output_names:
                effects[NAMES[j]][output].append((values[step + 1][output] - values[step][output]) / delta)

    interval_generator = np.random.default_rng(arguments.seed + 1)
    results = {output: {} for output in output_names}
    for output in output_names:
        for name in NAMES:
            values = np.array(effects[name][output])
            results[output][name] = {"mu_star": round(float(np.mean(np.abs(values))), 4), "mu": round(float(np.mean(values)), 4),
                                     "sigma": round(float(np.std(values, ddof=1)), 4),
                                     "mu_star_ci95": bootstrap_interval(values, interval_generator)}
        order = sorted(NAMES, key=lambda name: -results[output][name]["mu_star"])
        results[output] = {"ranking": order, "factors": results[output]}

    # How often does each lever keep its place? Share of design points at which L3 beats L5 and L5 beats L1 on peak.
    points = list(by_point.values())
    peak = lambda outcomes, scenario: outcomes[scenario].peak_day_protesters
    ordering = {"L3_below_L5": float(np.mean([peak(o, "hybrid_caste_subquotas") < peak(o, "consensus_commission") for o in points])),
                "L5_below_L1": float(np.mean([peak(o, "consensus_commission") < peak(o, "grandfathering") for o in points])),
                "L3_below_L1": float(np.mean([peak(o, "hybrid_caste_subquotas") < peak(o, "grandfathering") for o in points])),
                "design_points": len(points)}
    output = {"specification": arguments.specification, "method": "Morris elementary effects (trajectory design)",
              "levels": LEVELS, "step": STEP, "trajectories": arguments.trajectories, "agents": arguments.agents,
              "factors": {name: list(value) for name, value in FACTORS.items()}, "results": results,
              "lever_order_across_design_points": ordering}
    RESULTS_FOLDER.mkdir(exist_ok=True)
    path = RESULTS_FOLDER / f"global_sensitivity_{arguments.specification}.json"
    path.write_text(json.dumps(output, indent=1))
    for name in ("log_peak", "peak_change_percent:hybrid_caste_subquotas", "peak_change_percent:consensus_commission"):
        top = results[name]["ranking"][:5]
        print(name, [(factor, results[name]["factors"][factor]["mu_star"]) for factor in top])
    print(json.dumps(ordering))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
