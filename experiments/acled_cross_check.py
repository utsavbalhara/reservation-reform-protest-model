"""Cross-check the GDELT-based calibration against ACLED, a hand-coded event dataset.

Requires data/derived/acled_episode_summary.json (python -m data_pipelines.acled_episode_events). For every episode ACLED
covers:
  1. Counts: ACLED core-day events against the precision-corrected GDELT count.
  2. Geography: Spearman correlation across states between ACLED and GDELT core-day events.
  3. Frozen predictions: the leave-one-episode-out state distributions (frozen and hashed before any ACLED data were
     read) scored against ACLED's state distribution, with the same naive baselines as the GDELT scoring.
  4. Deaths: ACLED's reported fatalities on core days against the deaths coded from press reports.
  5. Crowds: for 2018, the sum of crowd sizes ACLED reports on the bandh day is a lower bound on turnout (many events
     report none). The share of retained calibration sets whose implied 2018 turnout lies above that bound is a test
     of the calibration that uses no GDELT data.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from experiments.score_frozen_predictions import verified
from protest_simulation.geography import district_table

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DERIVED = REPOSITORY_ROOT / "data" / "derived"
RESULTS = REPOSITORY_ROOT / "results"
# ACLED admin1 names that differ from the Census state names used by the model.
STATE_ALIASES = {"National Capital Territory of Delhi": "Delhi", "NCT of Delhi": "Delhi", "Orissa": "Odisha", "Pondicherry": "Puducherry",
                 "Uttaranchal": "Uttarakhand", "Jammu & Kashmir": "Jammu and Kashmir", "Andaman & Nicobar Islands": "Andaman and Nicobar Islands"}
MECHANISM = ("sc_st_bharat_bandh_2018", "upper_caste_bandh_2018", "sc_st_bharat_bandh_2024", "ews_quota_2019")


def normalized(by_state: dict) -> dict:
    out = {}
    for state, count in by_state.items():
        name = STATE_ALIASES.get(state, state)
        out[name] = out.get(name, 0) + count
    return out


def spearman(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return round(float(spearmanr(a, b).correlation), 3)


def main():
    acled = json.loads((DERIVED / "acled_episode_summary.json").read_text())
    targets = json.loads((DERIVED / "episode_targets.json").read_text())
    table = district_table()
    population = table.groupby("state").population.sum()
    sc_st = (table.sc + table.st).groupby(table.state).sum()
    large = [state for state in population.index if population[state] > 1e7 and state != "Delhi"]
    calibration = json.loads((RESULTS / "episode_calibration.json").read_text())
    output = {}
    for key, value in acled.items():
        if not value["covered_by_acled"] or key not in targets:
            continue
        acled_states = normalized(value["core_events_by_state"])
        gdelt_states = targets[key]["core_events_by_state"]
        record = {
            "acled_core_events": value["core_events"],
            "gdelt_corrected_core_events": targets[key]["corrected_core_events"]["median"],
            "states_correlation_acled_gdelt": spearman([acled_states.get(s, 0) for s in large], [gdelt_states.get(s, 0) for s in large]),
            "acled_core_fatalities": value["core_fatalities"], "coded_deaths": targets[key]["deaths"],
            "acled_events_reporting_crowd_size": value["core_events_reporting_crowd_size"],
            "acled_reported_crowd_sum": value["core_reported_crowd_sum"],
        }
        frozen_path = RESULTS / "frozen_predictions" / f"loeo_{key}.json"
        if frozen_path.exists():
            prediction = verified(frozen_path)
            states = prediction["states"]
            chosen = [i for i, name in enumerate(states) if name in large]
            observed = [acled_states.get(states[i], 0) for i in chosen]
            record["frozen_prediction_vs_acled"] = {
                "model": spearman(np.array(prediction["predicted_state_share"])[chosen], observed) if prediction["predicted_state_share"] else None,
                "baseline_population": spearman([population[states[i]] for i in chosen], observed),
                "baseline_sc_plus_st_population": spearman([sc_st[states[i]] for i in chosen], observed), "states": len(chosen)}
        if key == "sc_st_bharat_bandh_2018" and calibration.get("kept_samples"):
            turnout = np.array([sample["bandh_day_turnout_2018"] for sample in calibration["kept_samples"]])
            bound = value["core_reported_crowd_sum"]
            record["calibration_sets_above_acled_crowd_lower_bound"] = round(float(np.mean(turnout >= bound)), 3)
        output[key] = record
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "acled_cross_check.json"
    path.write_text(json.dumps(output, indent=1))
    print(json.dumps(output, indent=1))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
