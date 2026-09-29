"""Workstream D: calibrate the district-level model to event data from historical episodes, by history matching.

Why history matching: the model is stochastic, several parameters trade off, and each episode has its own unknown shock
size. Instead of a single best fit, history matching keeps every parameter set that is 'not ruled out yet': its predictions
for every observed quantity lie within three standard deviations of the observation once observation error, simulation
noise and a stated model discrepancy are allowed for (Vernon, Goldstein and Bower 2010; Andrianakis et al. 2015).

Observed quantities (data_pipelines/episode_targets.py):
  - GDELT protest events on each episode's core days, keyword-filtered, corrected by the audited precision and expressed
    per 1,000 GDELT events located in India on those days;
  - deaths on the bandh day (2018: 11 to 14; 2024, September 2018 and the EWS quota: none reported);
  - for the observation-model anchors (Maratha march 2017, Patidar rally 2015), the same event rate, with the crowd size
    drawn from the range of published estimates.

Observation model: expected events = scale x sum over states of (protester-days in the state / 1 lakh) ^ exponent, with
Delhi's events multiplied by a media factor (national media are concentrated there). The concave form means a bandh
spread over many states produces more events than a single rally of the same total size, as it should.

Unknowns and priors are in PRIORS. Episode magnitudes are relative to the reform's symbolic threat to SC agents.

Outputs (results/episode_calibration.json): the non-implausible parameter sets, derived quantities (magnitude ratios, the
event-data estimate of 2018 bandh-day turnout, deaths per crore protester-days on bandh days), spatial tests of the
model's state distribution against demographic baselines, and leave-one-episode-out predictions, which are written to
results/frozen_predictions/ and hashed before they are scored.
"""
import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from protest_simulation import BASELINE_ABRUPT_INCOME_ONLY_SWITCH, simulate_protest_campaign
from protest_simulation.episode_simulation import EPISODE_SHOCKS, episode_parameters, turnout_by_state
from protest_simulation.geography import build_synthetic_india_by_district, state_names

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
RESULTS_FOLDER = REPOSITORY_ROOT / "results"
TARGETS_PATH = REPOSITORY_ROOT / "data" / "derived" / "episode_targets.json"
MECHANISM_EPISODES = ("sc_st_bharat_bandh_2018", "sc_st_bharat_bandh_2024", "upper_caste_bandh_2018", "ews_quota_2019")
# Anchor episode -> (state, prior name for its crowd size in people-days on core days).
ANCHOR_EPISODES = {"maratha_march_mumbai_2017": ("Maharashtra", "log10_maratha_2017_crowd"),
                   "patidar_2015": ("Gujarat", "log10_patidar_2015_crowd")}
MIXING_GRID = np.round(np.arange(0.5, 1.0001, 0.1), 2)
AGENTS = 120_000
SEEDS = (11, 12)
MODEL_DISCREPANCY_LOG = 0.5
DEATH_DISCREPANCY = 2.0
IMPLAUSIBILITY_CUTOFF = 3.0

PRIORS = {
    "mean_participation_threshold": (5.0, 8.5),
    "participation_threshold_spread": (0.8, 2.2),
    "same_group_neighbourhood_share": (0.5, 1.0),
    "magnitude_sc_st_bharat_bandh_2018": (0.1, 2.5),
    "magnitude_sc_st_bharat_bandh_2024": (0.02, 2.5),
    "magnitude_upper_caste_bandh_2018": (0.02, 2.5),
    "magnitude_ews_quota_2019": (0.0, 1.5),
    "deprived_tier_share_2024": (0.1, 0.9),
    "log10_bandh_deaths_per_crore": (0.3, 3.0),
    "death_dispersion": (0.5, 5.0),
    "observation_exponent": (0.3, 1.0),
    "log10_observation_scale": (-4.0, 1.0),
    "delhi_media_factor": (1.0, 6.0),
    "log10_maratha_2017_crowd": (np.log10(1e5), np.log10(2e6)),
    "log10_patidar_2015_crowd": (np.log10(3e5), np.log10(2e6)),
}
NAMES = tuple(PRIORS)

_POPULATIONS = {}


def population_for(mixing: float):
    key = float(np.round(mixing, 2))
    if key not in _POPULATIONS:
        _POPULATIONS[key] = build_synthetic_india_by_district(AGENTS, np.random.default_rng(7), same_group_neighbourhood_share=key)
    return _POPULATIONS[key]


def sample_priors(random_generator, count, bounds=None):
    bounds = bounds or {name: PRIORS[name] for name in NAMES}
    return [{name: float(random_generator.uniform(*bounds[name])) for name in NAMES} for _ in range(count)]


def expected_events(state_turnout, names, sample):
    delhi = names.index("Delhi")
    weights = (np.maximum(state_turnout, 0.0) / 1e5) ** sample["observation_exponent"]
    weights[delhi] *= sample["delhi_media_factor"]
    return 10 ** sample["log10_observation_scale"] * weights.sum(), weights


def simulate_sample(sample):
    """Run every mechanism episode for one parameter set; return model outputs per episode (mean over seeds)."""
    mixing = MIXING_GRID[np.argmin(np.abs(MIXING_GRID - sample["same_group_neighbourhood_share"]))]
    population = population_for(mixing)
    names = state_names(population)
    base = BASELINE_ABRUPT_INCOME_ONLY_SWITCH.copy()
    base.mean_participation_threshold = sample["mean_participation_threshold"]
    base.participation_threshold_spread = sample["participation_threshold_spread"]
    outputs = {}
    for key in MECHANISM_EPISODES:
        shock = EPISODE_SHOCKS[key]
        parameters = episode_parameters(base, shock, sample[f"magnitude_{key}"],
                                        deprived_tier_share=sample["deprived_tier_share_2024"] if key == "sc_st_bharat_bandh_2024" else None)
        parameters.deaths_per_crore_protester_days = 0.0  # deaths are handled analytically below, from bandh-day turnout
        state_totals, bandh_day, events = [], [], []
        for seed in SEEDS:
            outcome = simulate_protest_campaign(population, parameters, np.random.default_rng(seed), record_agent_protest_days=True)
            by_state = turnout_by_state(population, outcome)
            state_totals.append(by_state)
            bandh_day.append(float(outcome.daily_protesters[1]))
            events.append(expected_events(by_state, names, sample)[0])
        mean_state = np.mean(state_totals, axis=0)
        outputs[key] = {"events": float(np.mean(events)), "log_events_sd": float(np.std(np.log(np.maximum(events, 1e-9)))),
                        "bandh_day_turnout": float(np.mean(bandh_day)), "state_turnout": mean_state.tolist()}
    return {"sample": sample, "mixing_used": float(mixing), "outputs": outputs}


def observation_log_sd(target):
    spread = np.log(max(target["p95"], 1e-6)) - np.log(max(target["p05"], 1e-6))
    return spread / (2 * 1.645)


def implausibilities(result, targets, episodes=MECHANISM_EPISODES):
    sample, outputs = result["sample"], result["outputs"]
    scores = {}
    for key in episodes:
        target = targets[key]["corrected_core_events_per_1000_india_events"]
        india_per_thousand = targets[key]["india_events_core_days"] / 1000.0
        model_rate = outputs[key]["events"] / india_per_thousand
        poisson_sd = 1.0 / np.sqrt(max(targets[key]["corrected_core_events"]["median"], 1.0))
        sd = np.sqrt(observation_log_sd(target) ** 2 + poisson_sd ** 2 + outputs[key]["log_events_sd"] ** 2 + MODEL_DISCREPANCY_LOG ** 2)
        scores[f"events:{key}"] = abs(np.log(max(model_rate, 1e-9)) - np.log(max(target["median"], 1e-9))) / sd
        deaths = targets[key]["deaths"]
        mean_deaths = 10 ** sample["log10_bandh_deaths_per_crore"] * outputs[key]["bandh_day_turnout"] / 1e7
        variance = mean_deaths + mean_deaths ** 2 / sample["death_dispersion"] + DEATH_DISCREPANCY ** 2
        observed = 0.5 * (deaths["low"] + deaths["high"])
        scores[f"deaths:{key}"] = abs(mean_deaths - observed) / np.sqrt(variance + ((deaths["high"] - deaths["low"]) / 2) ** 2)
    for key, (state, prior_name) in ANCHOR_EPISODES.items():
        crowd = 10 ** sample[prior_name]
        model_events = 10 ** sample["log10_observation_scale"] * (crowd / 1e5) ** sample["observation_exponent"]
        target = targets[key]["corrected_core_events_per_1000_india_events"]
        model_rate = model_events / (targets[key]["india_events_core_days"] / 1000.0)
        poisson_sd = 1.0 / np.sqrt(max(targets[key]["corrected_core_events"]["median"], 1.0))
        sd = np.sqrt(observation_log_sd(target) ** 2 + poisson_sd ** 2 + MODEL_DISCREPANCY_LOG ** 2)
        scores[f"events:{key}"] = abs(np.log(model_rate) - np.log(target["median"])) / sd
    return scores


def spatial_tests(result, targets, names, population):
    """Spearman correlation across states between model turnout and observed core-day events, with baselines."""
    table = population.district_table
    by_state_population = table.groupby("state").population.sum().reindex(names).to_numpy(float)
    by_state_sc_st = (table.sc + table.st).groupby(table.state).sum().reindex(names).to_numpy(float)
    tests = {}
    for key in ("sc_st_bharat_bandh_2018", "sc_st_bharat_bandh_2024", "upper_caste_bandh_2018"):
        observed_map = targets[key]["core_events_by_state"]
        chosen = [i for i, name in enumerate(names) if name not in ("Delhi",) and by_state_population[i] > 1e7]
        observed = np.array([observed_map.get(names[i], 0) for i in chosen], float)
        model = np.array(result["outputs"][key]["state_turnout"])[chosen]
        tests[key] = {"states": len(chosen),
                      "model": float(spearmanr(model, observed).correlation),
                      "baseline_population": float(spearmanr(by_state_population[chosen], observed).correlation),
                      "baseline_sc_plus_st_population": float(spearmanr(by_state_sc_st[chosen], observed).correlation)}
    return tests


CHECKPOINT_FOLDER = None


def run_wave(samples, workers=None, checkpoint=None):
    """Simulate a wave; with --checkpoint-dir, reuse a finished wave saved under the same name (runs are deterministic
    given their seeds, so a resumed calibration gives the same result as an uninterrupted one)."""
    path = CHECKPOINT_FOLDER / f"{checkpoint}.json" if CHECKPOINT_FOLDER and checkpoint else None
    if path and path.exists():
        saved = json.loads(path.read_text())
        if [result["sample"] for result in saved] == samples:
            return saved
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(simulate_sample, samples, chunksize=4))
    if path:
        path.write_text(json.dumps(results))
    return results


def bounds_of(kept, widen=0.1):
    bounds = {}
    for name in NAMES:
        values = np.array([result["sample"][name] for result in kept])
        low, high = values.min(), values.max()
        margin = widen * (high - low)
        bounds[name] = (max(PRIORS[name][0], low - margin), min(PRIORS[name][1], high + margin))
    return bounds


def frozen_write(path: Path, payload: dict) -> str:
    text = json.dumps(payload, indent=1, sort_keys=True)
    digest = hashlib.sha256(text.encode()).hexdigest()
    path.write_text(text)
    (path.parent / (path.stem + ".sha256")).write_text(digest + "\n")
    return digest


def summary(values):
    values = np.array(values, float)
    return {"median": float(np.median(values)), "p05": float(np.percentile(values, 5)), "p95": float(np.percentile(values, 95))}


def history_match(targets, episodes, wave_size, waves, seed, label, first_wave=None):
    """Waves of sampling; each later wave samples within the (slightly widened) range of the previous wave's survivors.
    Only the listed mechanism episodes (plus the observation-model anchors) are used to judge plausibility.
    first_wave: simulations of prior draws to reuse as wave 1. Wave 1 samples the prior whatever the episodes, so the
    leave-one-episode-out matches can share it with the main match; only the judging differs."""
    random_generator = np.random.default_rng(seed)
    history, kept, bounds = [], [], None
    for wave in range(waves):
        if wave == 0 and first_wave is not None:
            results = [dict(result) for result in first_wave]
        else:
            results = run_wave(sample_priors(random_generator, wave_size, bounds), checkpoint=f"{seed}_{wave_size}_wave{wave + 1}")
        for result in results:
            result["implausibility"] = implausibilities(result, targets, episodes)
            result["max_implausibility"] = max(result["implausibility"].values())
        kept = [result for result in results if result["max_implausibility"] < IMPLAUSIBILITY_CUTOFF]
        history.append({"wave": wave + 1, "samples": len(results), "kept": len(kept),
                        "bounds_used": {name: list(value) for name, value in (bounds or PRIORS).items()}})
        print(f"[{label}] wave {wave + 1}: kept {len(kept)} of {len(results)}", flush=True)
        if len(kept) < 10:
            break
        bounds = bounds_of(kept)
    return kept, history


def main():
    parser = argparse.ArgumentParser(description="History-match the district model to historical episodes.")
    parser.add_argument("--wave-size", type=int, default=3000)
    parser.add_argument("--waves", type=int, default=3)
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="save each finished wave here and resume from it")
    arguments = parser.parse_args()
    global CHECKPOINT_FOLDER
    if arguments.checkpoint_dir:
        arguments.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        CHECKPOINT_FOLDER = arguments.checkpoint_dir
    targets = json.loads(TARGETS_PATH.read_text())
    first_wave = run_wave(sample_priors(np.random.default_rng(20260928), arguments.wave_size),
                          checkpoint=f"shared_{arguments.wave_size}_wave1")
    print(f"shared first wave: {len(first_wave)} prior draws simulated", flush=True)
    kept, history = history_match(targets, MECHANISM_EPISODES, arguments.wave_size, arguments.waves, 20260929, "all episodes",
                                  first_wave)

    population = population_for(1.0)
    names = state_names(population)
    derived = []
    for result in kept:
        sample = result["sample"]
        record = {name: sample[name] for name in NAMES}
        m2018 = sample["magnitude_sc_st_bharat_bandh_2018"]
        record["ratio_2024_to_2018"] = sample["magnitude_sc_st_bharat_bandh_2024"] / m2018
        record["ratio_upper_caste_to_2018"] = sample["magnitude_upper_caste_bandh_2018"] / m2018
        record["ratio_ews_to_2018"] = sample["magnitude_ews_quota_2019"] / m2018
        record["bandh_day_turnout_2018"] = result["outputs"]["sc_st_bharat_bandh_2018"]["bandh_day_turnout"]
        record["bandh_day_turnout_2024"] = result["outputs"]["sc_st_bharat_bandh_2024"]["bandh_day_turnout"]
        record["spatial"] = spatial_tests(result, targets, names, population_for(result["mixing_used"]))
        record["event_rates_per_1000"] = {key: result["outputs"][key]["events"] / (targets[key]["india_events_core_days"] / 1000.0)
                                          for key in MECHANISM_EPISODES}
        record["max_implausibility"] = result["max_implausibility"]
        derived.append(record)

    output = {"history": history, "cutoff": IMPLAUSIBILITY_CUTOFF, "priors": {k: list(v) for k, v in PRIORS.items()},
              "model_discrepancy_log": MODEL_DISCREPANCY_LOG, "kept": len(kept)}
    if derived:
        output["posterior_summaries"] = {name: summary([record[name] for record in derived])
                                         for name in list(NAMES) + ["ratio_2024_to_2018", "ratio_upper_caste_to_2018", "ratio_ews_to_2018",
                                                                    "bandh_day_turnout_2018", "bandh_day_turnout_2024"]}
        output["spatial_tests"] = {key: {measure: summary([record["spatial"][key][measure] for record in derived])
                                         for measure in ("model", "baseline_population", "baseline_sc_plus_st_population")}
                                   for key in derived[0]["spatial"]}
        output["model_event_rates"] = {key: [record["event_rates_per_1000"][key] for record in derived] for key in MECHANISM_EPISODES}
        output["kept_samples"] = derived

    # Leave one episode out: a separate history match that never sees the held-out episode's events or deaths, then a
    # prediction of that episode's bandh-day deaths and its distribution across states. Predictions are frozen and hashed
    # before experiments/score_frozen_predictions.py compares them with the observations. The held-out episode's shock
    # magnitude is not identified without its data, so its prediction integrates over the prior for that magnitude.
    frozen_folder = RESULTS_FOLDER / "frozen_predictions"
    frozen_folder.mkdir(parents=True, exist_ok=True)
    loeo = {}
    for index, held_out in enumerate(("sc_st_bharat_bandh_2018", "sc_st_bharat_bandh_2024", "upper_caste_bandh_2018")):
        others = tuple(key for key in MECHANISM_EPISODES if key != held_out)
        retained, loeo_history = history_match(targets, others, arguments.wave_size, arguments.waves, 20260930 + index,
                                               f"without {held_out}", first_wave)
        predictions = []
        for result in retained:
            sample = result["sample"]
            mean_deaths = 10 ** sample["log10_bandh_deaths_per_crore"] * result["outputs"][held_out]["bandh_day_turnout"] / 1e7
            predictions.append({"predicted_mean_deaths": mean_deaths, "state_turnout": result["outputs"][held_out]["state_turnout"]})
        payload = {"held_out": held_out, "calibrated_on": list(others), "retained": len(retained),
                   "predicted_mean_deaths": summary([p["predicted_mean_deaths"] for p in predictions]) if predictions else None,
                   "predicted_state_share": (np.mean([np.array(p["state_turnout"]) / max(sum(p["state_turnout"]), 1e-9) for p in predictions], axis=0).round(5).tolist()
                                             if predictions else None),
                   "states": names}
        digest = frozen_write(frozen_folder / f"loeo_{held_out}.json", payload)
        loeo[held_out] = {"frozen_file": f"results/frozen_predictions/loeo_{held_out}.json", "sha256": digest,
                          "retained": len(retained), "history": loeo_history}
    output["leave_one_episode_out"] = loeo
    RESULTS_FOLDER.mkdir(exist_ok=True)
    (RESULTS_FOLDER / "episode_calibration.json").write_text(json.dumps(output, indent=1))
    # The parameter sets that survive calibration on all episodes, for the grounded reform runs.
    (RESULTS_FOLDER / "episode_calibration_nroy_samples.json").write_text(json.dumps([result["sample"] for result in kept], indent=1))
    print(json.dumps({k: v for k, v in output.items() if k not in ("kept_samples",)}, indent=1)[:6000])


if __name__ == "__main__":
    main()
