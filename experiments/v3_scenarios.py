"""Run the v3 reform scenarios on the retained calibration sets and report protest relative to the 2018 bandh.

Each run draws one retained parameter set and one random seed; every scenario in that run uses the same set and seed,
so comparisons are paired. A campaign is an announcement day, a national bandh the next day, and six more days of
ordinary mobilization (the episodes fit at most six consecutive days, so the week is about as far as the calibration
reaches). Outputs, per scenario:
  - bandh-day expected news events and turnout, relative to the 2018 replay on the same set ("x 2018");
  - the share of runs above the 2018 level;
  - protester-days and expected deaths over the week, relative to the 2018 replay;
  - who takes part on the bandh day, by identity group and SC/ST tier, and by income line;
  - mean share of each district's population on the street on the bandh day (for maps);
  - IIT seat changes by segment, from the allocation model.
Writes results/v3_scenarios.json.
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import experiments.v3_history_match as hm
from protest_v3.model import Shock, expected_events, reporting_intensity, simulate
from protest_v3.population import IDENTITY_GROUP_NAMES, build_population
from protest_v3.scenarios import ALLOCATION_CASE, LABELS, MODIFIER_SETS, REFORMS, draw

RESULTS = Path(__file__).resolve().parent.parent / "results"
DAYS, ACTION_DAYS = 8, (1,)
SCENARIOS = [("replay_2018", "none"), ("replay_ews_2019", "none")] + [(r, m) for r in REFORMS for m in MODIFIER_SETS]


def run_one(arguments):
    run, sample = arguments
    population = build_population(hm.AGENTS, hm.nearest_mixing(sample["mixing"]))
    base = hm.parameters_from(sample)
    district_population = population.base.district_table.population.to_numpy(float)
    out = {}
    for reform, modifier in SCENARIOS:
        rng = np.random.default_rng([90_000, run, SCENARIOS.index((reform, modifier))])
        scenario = draw(reform, MODIFIER_SETS[modifier], sample, rng)
        shock = Shock(threat=scenario.threat, deprived_tier_share=scenario.deprived_tier_share, material_by_segment=scenario.material_by_segment,
                      deprived_tier_material_gain=scenario.deprived_tier_material_gain)
        outcome = simulate(population, base.copy(party_amplifier=scenario.amplifier), shock, DAYS, ACTION_DAYS, np.random.default_rng(50_000 + run))
        events = expected_events(outcome.daily_by_state, reporting_intensity(population.state_names), sample["log10_observation_scale"],
                                 sample["observation_exponent"], sample["reporting_power"]).sum(axis=1)
        bandh = outcome.daily_by_identity[ACTION_DAYS[0]]
        out[f"{reform}|{modifier}"] = {
            "bandh_events": float(events[ACTION_DAYS[0]]), "bandh_turnout": float(outcome.daily[ACTION_DAYS[0]]),
            "week_protester_days": float(outcome.daily.sum()), "week_deaths": float(outcome.daily_expected_deaths.sum()),
            "unique": outcome.unique_participants, "bandh_by_identity": bandh.tolist(),
            "district_share": (outcome.district_protester_days / DAYS / district_population).tolist(),
        }
    return out


def interval(values):
    values = np.asarray(values, float)
    return {"median": float(np.median(values)), "p05": float(np.percentile(values, 5)), "p95": float(np.percentile(values, 95))}


def seat_changes():
    data = json.loads((RESULTS / "merged_pool_allocation.json").read_text())["summary"]
    return {reform: {segment: values["percent_change"]["median"] for segment, values in data[case].items()}
            for reform, case in ALLOCATION_CASE.items() if case in data}


def main():
    parser = argparse.ArgumentParser(description="Run the v3 reform scenarios.")
    parser.add_argument("--runs", type=int, default=300)
    arguments = parser.parse_args()
    samples = json.loads((RESULTS / "v3_nroy_samples.json").read_text())
    rng = np.random.default_rng(20261004)
    picks = [samples[i] for i in rng.integers(len(samples), size=arguments.runs)]
    with ProcessPoolExecutor() as pool:
        runs = list(pool.map(run_one, list(enumerate(picks)), chunksize=4))
    output = {"runs": arguments.runs, "retained_sets": len(samples), "identity_groups": list(IDENTITY_GROUP_NAMES), "scenarios": {}}
    for reform, modifier in SCENARIOS:
        key = f"{reform}|{modifier}"
        rel_events = [r[key]["bandh_events"] / max(r["replay_2018|none"]["bandh_events"], 1e-9) for r in runs]
        rel_turnout = [r[key]["bandh_turnout"] / max(r["replay_2018|none"]["bandh_turnout"], 1.0) for r in runs]
        rel_days = [r[key]["week_protester_days"] / max(r["replay_2018|none"]["week_protester_days"], 1.0) for r in runs]
        by_identity = np.array([r[key]["bandh_by_identity"] for r in runs])
        share_identity = by_identity.sum(axis=0) / max(by_identity.sum(), 1.0)
        output["scenarios"][key] = {
            "reform": reform, "modifier": modifier, "label": LABELS[reform],
            "events_vs_2018": interval(rel_events), "turnout_vs_2018": interval(rel_turnout), "protester_days_vs_2018": interval(rel_days),
            "share_of_runs_above_2018": float(np.mean(np.array(rel_events) > 1)),
            "bandh_turnout_lakh": interval([r[key]["bandh_turnout"] / 1e5 for r in runs]),
            "week_deaths": interval([r[key]["week_deaths"] for r in runs]),
            "share_by_identity_group": dict(zip(IDENTITY_GROUP_NAMES, share_identity.round(4).tolist())),
            "district_share_mean": np.mean([r[key]["district_share"] for r in runs], axis=0).round(7).tolist(),
        }
        print(f"{key:38} events x2018 {np.median(rel_events):6.2f} [{np.percentile(rel_events, 5):.2f}, {np.percentile(rel_events, 95):.2f}]  "
              f"P(>2018) {np.mean(np.array(rel_events) > 1):.2f}", flush=True)
    output["seat_changes_percent"] = seat_changes()
    (RESULTS / "v3_scenarios.json").write_text(json.dumps(output))
    print(f"Saved {RESULTS / 'v3_scenarios.json'}")


if __name__ == "__main__":
    main()
