"""Split composite levers into their channels, and separate what the suppression scenarios assume from what emerges.

Every lever changes more than one parameter. This experiment re-runs L1, L3 and L5 with each channel applied on its
own, using the same paired draws as the full lever, so the channels can be compared run by run.

For suppression, deaths are Poisson with mean kappa x protester-days. The death ratio against the baseline therefore
factors into the kappa multiplier the scenario assumes and the ratio of protester-days, which the model produces.
Only the second factor, and the change in cumulative participation, are findings.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from protest_simulation import BASELINE_ABRUPT_INCOME_ONLY_SWITCH, build_shared_population
from protest_simulation.monte_carlo import CAMPAIGN_SEED_OFFSET, paired_effects_from_runs, paired_worlds
from protest_simulation.parallel import run_campaigns
from protest_simulation.policy_interventions import (
    build_consensus_through_data_first_commission,
    deploy_heavy_policing,
    grandfather_current_cohorts_with_ten_year_glide,
    keep_caste_subquotas_with_income_filter,
    shut_down_internet,
)
from protest_simulation.synthetic_population import OBC

RESULTS_FOLDER = Path(__file__).resolve().parent.parent / "results"


def consensus_amplifier_channel(parameters):
    parameters.opposition_party_amplifier /= 1.5


def consensus_symbolic_channel(parameters):
    parameters.symbolic_threat_by_group = parameters.symbolic_threat_by_group * 0.8


def hybrid_symbolic_channel(parameters):
    symbolic = parameters.symbolic_threat_by_group.copy()
    symbolic[: OBC + 1] *= 0.4
    parameters.symbolic_threat_by_group = symbolic


def hybrid_material_channel(parameters):
    parameters.material_loss_sc_st_below_income_line = 0.0
    parameters.material_loss_obc_below_income_line = 0.0


def grandfathering_material_channel(parameters):
    parameters.material_loss_sc_st_above_income_line *= 0.3
    parameters.material_loss_sc_st_below_income_line *= 0.3
    parameters.material_loss_obc_below_income_line *= 0.3


def grandfathering_symbolic_channel(parameters):
    parameters.symbolic_threat_by_group = parameters.symbolic_threat_by_group * 0.95


def policing_turnout_cost_channel(parameters):
    parameters.heavy_policing_turnout_cost = 0.4


def policing_violence_channel(parameters):
    parameters.deaths_per_crore_protester_days *= 3
    parameters.symbolic_threat_rise_per_death *= 1.5


def shutdown_without_extra_violence(parameters):
    parameters.internet_shutdown_active = True
    parameters.internet_shutdown_violence_factor = 1.0


CHANNELS = {
    "L5": {"full": build_consensus_through_data_first_commission, "amplifier_only": consensus_amplifier_channel,
           "symbolic_only": consensus_symbolic_channel},
    "L3": {"full": keep_caste_subquotas_with_income_filter, "symbolic_only": hybrid_symbolic_channel, "material_only": hybrid_material_channel},
    "L1": {"full": grandfather_current_cohorts_with_ten_year_glide, "material_only": grandfathering_material_channel,
           "symbolic_only": grandfathering_symbolic_channel},
    "B2": {"full": deploy_heavy_policing, "turnout_cost_only": policing_turnout_cost_channel, "violence_and_martyr_only": policing_violence_channel},
    "B1": {"full": shut_down_internet, "coordination_only": shutdown_without_extra_violence},
}
ASSUMED_DEATH_RATE_MULTIPLIER = {"B1": BASELINE_ABRUPT_INCOME_ONLY_SWITCH.internet_shutdown_violence_factor, "B2": 3.0}


def run_scenario(population, interventions, run_count):
    worlds = paired_worlds(interventions, run_count)
    outcomes = run_campaigns(population, [(world, CAMPAIGN_SEED_OFFSET + index) for index, world in enumerate(worlds)])
    return {
        "peak_day_protesters": [outcome.peak_day_protesters for outcome in outcomes],
        "cumulative_unique_protesters": [outcome.cumulative_unique_protesters for outcome in outcomes],
        "total_deaths": [outcome.total_deaths for outcome in outcomes],
        "protester_days": [float(outcome.daily_protesters.sum()) for outcome in outcomes],
    }


def paired_with_protester_days(scenario, baseline):
    effects = paired_effects_from_runs(scenario, baseline)
    ratios = np.array(scenario["protester_days"]) / np.array(baseline["protester_days"])
    effects["protester_days"] = {"median_paired_ratio": round(float(np.median(ratios)), 4),
                                 "change_percent": round((float(np.median(ratios)) - 1) * 100, 1)}
    return effects


def main():
    parser = argparse.ArgumentParser(description="Decompose composite levers and account for suppression inputs.")
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--agents", type=int, default=120_000)
    arguments = parser.parse_args()
    population = build_shared_population(arguments.agents)
    baseline = run_scenario(population, (), arguments.runs)
    output = {"run_count": arguments.runs, "agent_count": arguments.agents, "channels": {}}
    for code, channels in CHANNELS.items():
        output["channels"][code] = {}
        for channel_name, intervention in channels.items():
            effects = paired_with_protester_days(run_scenario(population, (intervention,), arguments.runs), baseline)
            if code in ASSUMED_DEATH_RATE_MULTIPLIER and channel_name == "full":
                effects["assumed_death_rate_multiplier"] = ASSUMED_DEATH_RATE_MULTIPLIER[code]
                effects["emergent_death_factor"] = round(effects["deaths"]["median_paired_ratio"] / ASSUMED_DEATH_RATE_MULTIPLIER[code], 3)
            output["channels"][code][channel_name] = effects
            print(code, channel_name, {k: effects[k]["change_percent"] for k in ("peak", "cumulative", "deaths", "protester_days")}, flush=True)
    RESULTS_FOLDER.mkdir(exist_ok=True)
    path = RESULTS_FOLDER / "lever_decomposition.json"
    path.write_text(json.dumps(output, indent=1))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
