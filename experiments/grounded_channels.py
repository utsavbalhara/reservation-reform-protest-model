"""The grounded specification, taken apart.

1. Channels. Every lever changes more than one input. Re-run L1, L3, L5, B1 and B2 with one channel at a time, and the
   baseline with its material or its symbolic grievance removed, on the same paired worlds as the main comparison.
2. Grievance composition. For the central parameter values, the mean material and symbolic components of grievance by
   eligibility segment, and the share of all positive grievance that is symbolic.
3. Who protests and where. Protester-days by group, by SC/ST tier and by state under the abrupt switch, and how
   sub-classification (L4) and caste quotas with an income filter (L3) shift them. This tests stylized fact S4 (groups
   split when a reform creates winners inside them).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from protest_simulation import SCENARIO_BY_KEY
from protest_simulation.geography import state_names
from protest_simulation.monte_carlo import CAMPAIGN_SEED_OFFSET, paired_effects_from_runs, paired_worlds
from protest_simulation.parallel import run_campaigns
from protest_simulation.policy_interventions import (SC_ST_THREAT_RETAINED_UNDER_INCOME_FILTER, _scale_segment_losses, _use_lever_table,
                                                   income_filter_threat_retained_from_episodes, remove_all_material_loss)
from protest_simulation.protest_campaign import (
    SEGMENT_NAMES, eligibility_segment_of_each_agent, material_loss_felt_by_each_agent, symbolic_threat_felt_by_each_agent)
from protest_simulation.specifications import get_specification
from protest_simulation.synthetic_population import OBC, SOCIAL_GROUP_NAMES

RESULTS_FOLDER = Path(__file__).resolve().parent.parent / "results"


def no_material(world):
    remove_all_material_loss(world)


def no_symbolic(world):
    world.symbolic_threat_by_group = world.symbolic_threat_by_group * 0.0


def l3_symbolic(world):
    symbolic = world.symbolic_threat_by_group.copy()
    symbolic[: OBC + 1] *= SC_ST_THREAT_RETAINED_UNDER_INCOME_FILTER
    from_episodes = income_filter_threat_retained_from_episodes(world)
    if from_episodes is not None:
        symbolic[:OBC] = world.symbolic_threat_by_group[:OBC] * from_episodes
    world.symbolic_threat_by_group = symbolic


def l3_allocation(world):
    _use_lever_table(world, "hybrid_caste_subquotas")


def l5_amplifier(world):
    world.opposition_party_amplifier /= 1.5


def l5_symbolic(world):
    world.symbolic_threat_by_group = world.symbolic_threat_by_group * 0.8


def l1_material(world):
    _scale_segment_losses(world, 0.3)


def l1_symbolic(world):
    world.symbolic_threat_by_group = world.symbolic_threat_by_group * 0.95


def b2_turnout_cost(world):
    world.heavy_policing_turnout_cost = 0.4


def b2_violence(world):
    share = world.police_attributed_death_share
    world.police_death_multiplier *= (3.0 - (1 - share)) / share
    world.symbolic_threat_rise_per_death *= 1.5


def b1_coordination(world):
    world.internet_shutdown_active = True
    world.internet_shutdown_violence_factor = 1.0


def full(key):
    return lambda world: [intervention(world) for intervention in SCENARIO_BY_KEY[key].interventions]


CHANNELS = {
    "baseline": {"no_material": no_material, "no_symbolic": no_symbolic},
    "L1": {"full": full("grandfathering"), "material_only": l1_material, "symbolic_only": l1_symbolic},
    "L3": {"full": full("hybrid_caste_subquotas"), "symbolic_only": l3_symbolic, "allocation_only": l3_allocation},
    "L5": {"full": full("consensus_commission"), "amplifier_only": l5_amplifier, "symbolic_only": l5_symbolic},
    "B1": {"full": full("internet_shutdown"), "coordination_only": b1_coordination},
    "B2": {"full": full("heavy_policing"), "turnout_cost_only": b2_turnout_cost, "violence_and_martyr_only": b2_violence},
}
RECORDED = {"baseline": None, "sub_classification": full("sub_classification"), "hybrid_caste_subquotas": full("hybrid_caste_subquotas")}


def run(population, specification, change, run_count, record=False):
    worlds = paired_worlds((), run_count, base_parameters=specification.base_parameters, world_sampler=specification.world_sampler)
    for world in worlds:
        if change is not None:
            change(world)
    outcomes = run_campaigns(population, [(world, CAMPAIGN_SEED_OFFSET + index, False, record) for index, world in enumerate(worlds)])
    runs = {"peak_day_protesters": [o.peak_day_protesters for o in outcomes],
            "cumulative_unique_protesters": [o.cumulative_unique_protesters for o in outcomes],
            "total_deaths": [o.total_deaths for o in outcomes],
            "protester_days": [float(o.daily_protesters.sum()) for o in outcomes]}
    return runs, (outcomes if record else None)


def with_days(effects, scenario, baseline):
    ratios = np.array(scenario["protester_days"]) / np.maximum(np.array(baseline["protester_days"]), 1.0)
    effects["protester_days"] = {"median_paired_ratio": round(float(np.median(ratios)), 4),
                                 "change_percent": round((float(np.median(ratios)) - 1) * 100, 1)}
    return effects


def composition(population, parameters):
    material = parameters.material_loss_weight * material_loss_felt_by_each_agent(population, parameters)
    symbolic = parameters.symbolic_threat_weight * population.identity_strength * symbolic_threat_felt_by_each_agent(population, parameters)
    segments = eligibility_segment_of_each_agent(population)
    table = {}
    for index, name in enumerate(SEGMENT_NAMES):
        chosen = segments == index
        if chosen.any():
            table[name] = {"share_of_population": round(float(chosen.mean()), 4),
                           "mean_material": round(float(material[chosen].mean()), 3),
                           "mean_symbolic": round(float(symbolic[chosen].mean()), 3)}
    positive_material, positive_symbolic = np.clip(material, 0, None).sum(), np.clip(symbolic, 0, None).sum()
    return {"by_segment": table, "symbolic_share_of_positive_grievance": round(float(positive_symbolic / (positive_material + positive_symbolic)), 3)}


def who_and_where(population, outcomes):
    names = state_names(population)
    agent_state = population.state_of_district[population.district]
    group, deprived = population.social_group, population.is_most_deprived_tier
    by_group, by_tier, by_state = [], [], []
    for outcome in outcomes:
        days = outcome.agent_protest_days.astype(float)
        total = max(days.sum(), 1.0)
        by_group.append([days[group == g].sum() / total for g in range(len(SOCIAL_GROUP_NAMES))])
        sc_st = group <= 1
        by_tier.append({"better_off_share_protesting": float((days[sc_st & ~deprived] > 0).mean()),
                        "most_deprived_share_protesting": float((days[sc_st & deprived] > 0).mean())})
        by_state.append(np.bincount(agent_state, weights=days, minlength=len(names)) / total)
    state_share = np.median(by_state, axis=0)
    order = np.argsort(-state_share)
    return {"protester_day_share_by_group": {name: round(float(np.median([row[g] for row in by_group])), 3) for g, name in enumerate(SOCIAL_GROUP_NAMES)},
            "sc_st_share_ever_protesting_by_tier": {key: round(float(np.median([row[key] for row in by_tier])), 4) for key in by_tier[0]},
            "protester_day_share_by_state": {names[i]: round(float(state_share[i]), 4) for i in order if state_share[i] > 0}}


def main():
    parser = argparse.ArgumentParser(description="Channels, grievance composition and turnout composition in the grounded model.")
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--agents", type=int, default=120_000)
    arguments = parser.parse_args()
    specification = get_specification("grounded")
    population = specification.build_population(arguments.agents)
    baseline, baseline_outcomes = run(population, specification, None, arguments.runs, record=True)
    output = {"run_count": arguments.runs, "agent_count": arguments.agents, "channels": {},
              "grievance_composition": composition(population, specification.base_parameters), "who_and_where": {}}
    for code, channels in CHANNELS.items():
        output["channels"][code] = {}
        for name, change in channels.items():
            runs, _ = run(population, specification, change, arguments.runs)
            effects = with_days(paired_effects_from_runs(runs, baseline), runs, baseline)
            output["channels"][code][name] = effects
            print(code, name, {k: effects[k]["change_percent"] for k in ("peak", "cumulative", "deaths", "protester_days")}, flush=True)
    output["who_and_where"]["baseline"] = who_and_where(population, baseline_outcomes)
    for key, change in RECORDED.items():
        if key == "baseline":
            continue
        _, outcomes = run(population, specification, change, arguments.runs, record=True)
        output["who_and_where"][key] = who_and_where(population, outcomes)
    print(json.dumps({key: {k: v for k, v in value.items() if k != "protester_day_share_by_state"} for key, value in output["who_and_where"].items()}, indent=1))
    print("top states:", list(output["who_and_where"]["baseline"]["protester_day_share_by_state"].items())[:6])
    print("symbolic share of positive grievance:", output["grievance_composition"]["symbolic_share_of_positive_grievance"])
    RESULTS_FOLDER.mkdir(exist_ok=True)
    path = RESULTS_FOLDER / "grounded_channels.json"
    path.write_text(json.dumps(output, indent=1))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
