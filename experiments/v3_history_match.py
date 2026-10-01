"""History matching of the v3 model to ten reservation episodes.

Targets, for each episode (data/derived/episode_targets_v3.json):
  level    precision-corrected news-reported events on core days, per 1,000 GDELT events in India (log scale);
  profile  the share of core-day events falling on each core day (multi-day episodes);
  spatial  the distribution of core-day events across states with at least three events (shape only);
  deaths   reported protest deaths on core days, where they are protest deaths;
  crowd    published crowd sizes for the Maratha march (2017) and the Patidar rally (2015), as a range for the model's
           turnout in that state on that day.
A parameter set is implausible if any target's standardized distance exceeds 3. Waves sample within the range of the
previous wave's survivors (widened by 10%).

Two matches run on the same simulations: the main match uses every target; the "no-spatial" match drops the spatial
targets, and its survivors predict where protest happened, which is then compared with population and SC+ST baselines.

Outputs: results/v3_calibration.json, results/v3_nroy_samples.json.
"""
import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from protest_v3.episodes import EPISODE_KEYS, episode_threat, load_targets, party_amplifier, schedule
from protest_v3.model import V3Parameters, Shock, expected_events, simulate
from protest_v3.population import IDENTITY_GROUP_COUNT, build_population

RESULTS = Path(__file__).resolve().parent.parent / "results"
AGENTS = 120_000
SEED = 11
MIXING_GRID = np.round(np.arange(0.5, 1.0001, 0.1), 2)
LEVEL_DISCREPANCY = 0.5
SIMULATION_LOG_SD = 0.05
PROFILE_DISCREPANCY = 0.08
SPATIAL_DISCREPANCY = 0.8
DEATH_DISCREPANCY = 2.0
CROWD_LOG_SCALE = 0.35
CUTOFF = 3.0
NO_SPATIAL_EPISODES = {"ews_quota_2019", "kapu_2016"}

PRIORS = {
    "mean_threshold": (5.0, 9.0), "threshold_spread": (0.8, 2.2), "mixing": (0.5, 1.0), "fatigue": (0.0, 0.3),
    "neighbourhood_influence": (0.4, 1.4), "national_influence": (0.2, 1.0), "community_capacity": (0.2, 1.0),
    "kappa_urban": (-0.5, 1.0), "kappa_literacy": (-0.5, 0.5), "kappa_phone": (-0.5, 1.0),
    "log10_death_rate": (0.3, 3.0), "death_dispersion": (0.5, 5.0),
    "log10_observation_scale": (-4.0, 1.0), "observation_exponent": (0.3, 1.0), "delhi_media_factor": (1.0, 6.0),
    "deprived_tier_share_2024": (0.1, 0.9),
    **{f"magnitude_{key}": ((0.0, 1.5) if key == "ews_quota_2019" else (0.02, 3.0)) for key in EPISODE_KEYS},
}
NAMES = tuple(PRIORS)


def parameters_from(sample: dict) -> V3Parameters:
    parameters = V3Parameters(mean_threshold=sample["mean_threshold"], threshold_spread=sample["threshold_spread"],
                              fatigue=sample["fatigue"], neighbourhood_influence=sample["neighbourhood_influence"],
                              national_influence=sample["national_influence"],
                              covariate_effect=np.array([sample["kappa_urban"], sample["kappa_literacy"], sample["kappa_phone"]]),
                              action_day_death_rate=10 ** sample["log10_death_rate"], death_dispersion=sample["death_dispersion"])
    parameters.capacity[4:] = sample["community_capacity"]
    return parameters


def nearest_mixing(value: float) -> float:
    return float(MIXING_GRID[np.argmin(np.abs(MIXING_GRID - value))])


def simulate_set(sample: dict) -> dict:
    """Run every episode for one parameter set and return what the targets need."""
    targets = load_targets()
    population = build_population(AGENTS, nearest_mixing(sample["mixing"]))
    base = parameters_from(sample)
    out = {}
    for key in EPISODE_KEYS:
        target = targets[key]
        plan = schedule(target)
        parameters = base.copy(party_amplifier=party_amplifier(target["party_backing"]))
        deprived = sample["deprived_tier_share_2024"] if key == "sc_st_bharat_bandh_2024" else 0.8
        shock = Shock(threat=episode_threat(key, sample[f"magnitude_{key}"]), deprived_tier_share=deprived)
        outcome = simulate(population, parameters, shock, plan.days, plan.action_days, np.random.default_rng(SEED))
        core = list(plan.core_day_indices)
        events = expected_events(outcome.daily_by_state[core], population.state_names, sample["log10_observation_scale"],
                                 sample["observation_exponent"], sample["delhi_media_factor"])
        daily = outcome.daily
        rates = np.array([parameters.action_day_death_rate if day in plan.action_days else parameters.action_day_death_rate * parameters.other_day_death_share
                          for day in range(plan.days)])
        record = {"events_by_day": events.sum(axis=1).tolist(), "events_by_state": events.sum(axis=0).tolist(),
                  "expected_deaths_core": float((rates * daily / 1e7)[core].sum()),
                  "turnout_core": [float(daily[day]) for day in core],
                  "turnout_by_state_core": outcome.daily_by_state[core].sum(axis=0).tolist()}
        anchor = target["crowd_anchor"]
        if anchor:
            record["anchor_turnout"] = float(outcome.daily_by_state[core[anchor["day_index"]], population.state_names.index(anchor["state"])])
        out[key] = record
    return {"sample": sample, "outputs": out, "states": population.state_names}


def log_sd_from_interval(interval: dict) -> float:
    return (np.log(max(interval["p95"], 1e-6)) - np.log(max(interval["p05"], 1e-6))) / (2 * 1.645)


def implausibilities(result: dict, targets: dict, use_spatial: bool = True) -> dict:
    sample, outputs, states = result["sample"], result["outputs"], result["states"]
    scores = {}
    for key in EPISODE_KEYS:
        target, output = targets[key], outputs[key]
        observed_rate = target["core_events_per_1000"]
        model_rate = sum(output["events_by_day"]) / (target["india_events_core_days"] / 1000.0)
        poisson = 1.0 / np.sqrt(max(target["corrected_core_events"]["median"], 1.0))
        sd = np.sqrt(log_sd_from_interval(observed_rate) ** 2 + poisson ** 2 + SIMULATION_LOG_SD ** 2 + LEVEL_DISCREPANCY ** 2)
        scores[f"level:{key}"] = abs(np.log(max(model_rate, 1e-9)) - np.log(observed_rate["median"])) / sd
        counts = np.array(target["daily_core_events"], float)
        if len(counts) >= 2 and key != "ews_quota_2019":
            model = np.array(output["events_by_day"], float)
            model_share = model / max(model.sum(), 1e-12)
            observed_share = counts / counts.sum()
            sd = np.sqrt(model_share * (1 - model_share) / counts.sum() + PROFILE_DISCREPANCY ** 2)
            scores[f"profile:{key}"] = float(np.max(np.abs(observed_share - model_share) / sd))
        if use_spatial and key not in NO_SPATIAL_EPISODES:
            observed = target["core_events_by_state"]
            chosen = [state for state, value in observed.items() if value >= 3 and state in states]
            if len(chosen) >= 2:
                model = np.array(output["events_by_state"], float)
                total_observed = sum(observed.values())
                scaled = model / max(model.sum(), 1e-12) * total_observed
                z = [(np.log(observed[s] + 0.5) - np.log(scaled[states.index(s)] + 0.5)) / np.sqrt(1 / (observed[s] + 0.5) + SPATIAL_DISCREPANCY ** 2)
                     for s in chosen]
                scores[f"spatial:{key}"] = float(np.sqrt(np.mean(np.square(z))))
        deaths = target["deaths"]
        if deaths is not None:
            mean = output["expected_deaths_core"]
            variance = mean + mean ** 2 / sample["death_dispersion"] + DEATH_DISCREPANCY ** 2 + ((deaths["high"] - deaths["low"]) / 2) ** 2
            scores[f"deaths:{key}"] = abs(mean - 0.5 * (deaths["low"] + deaths["high"])) / np.sqrt(variance)
        anchor = target["crowd_anchor"]
        if anchor:
            turnout = max(output["anchor_turnout"], 1.0)
            outside = max(np.log(anchor["low"] / turnout), np.log(turnout / anchor["high"]), 0.0)
            scores[f"crowd:{key}"] = outside / CROWD_LOG_SCALE
    return scores


CHECKPOINTS = None


def run_wave(samples, label):
    path = CHECKPOINTS / f"{label}.json" if CHECKPOINTS else None
    if path and path.exists():
        saved = json.loads(path.read_text())
        if [r["sample"] for r in saved] == samples:
            return saved
    with ProcessPoolExecutor() as pool:
        results = list(pool.map(simulate_set, samples, chunksize=8))
    if path:
        path.write_text(json.dumps(results))
    return results


def sample_priors(rng, count, bounds):
    return [{name: float(rng.uniform(*bounds[name])) for name in NAMES} for _ in range(count)]


def bounds_of(kept, widen=0.1):
    bounds = {}
    for name in NAMES:
        values = np.array([r["sample"][name] for r in kept])
        margin = widen * (values.max() - values.min())
        bounds[name] = (max(PRIORS[name][0], values.min() - margin), min(PRIORS[name][1], values.max() + margin))
    return bounds


def history_match(targets, wave_size, waves, seed, use_spatial, first_wave, label):
    rng = np.random.default_rng(seed)
    bounds, history, kept = dict(PRIORS), [], []
    for wave in range(waves):
        results = first_wave if wave == 0 else run_wave(sample_priors(rng, wave_size, bounds), f"{label}_wave{wave + 1}")
        judged = []
        for r in results:
            scores = implausibilities(r, targets, use_spatial)
            judged.append({**r, "implausibility": scores, "max_implausibility": max(scores.values())})
        kept = [r for r in judged if r["max_implausibility"] < CUTOFF]
        worst = {}
        for r in judged:
            name = max(r["implausibility"], key=r["implausibility"].get)
            worst[name.split(":")[0]] = worst.get(name.split(":")[0], 0) + 1
        history.append({"wave": wave + 1, "samples": len(results), "kept": len(kept), "binding_target_counts": worst})
        print(f"[{label}] wave {wave + 1}: kept {len(kept)} of {len(results)}; binding targets {worst}", flush=True)
        if len(kept) < 10:
            break
        bounds = bounds_of(kept)
    return kept, history


def interval(values):
    values = np.asarray(values, float)
    return {"median": float(np.median(values)), "p05": float(np.percentile(values, 5)), "p95": float(np.percentile(values, 95))}


def spatial_tests(kept, targets):
    """Spearman correlation across large states between predicted and observed core-day events, with baselines."""
    population = build_population(AGENTS, 0.8)
    table = population.base.district_table
    states = population.state_names
    pop = table.groupby("state").population.sum().reindex(states).to_numpy(float)
    scst = (table.sc + table.st).groupby(table.state).sum().reindex(states).to_numpy(float)
    tests = {}
    for key in ("sc_st_bharat_bandh_2018", "sc_st_bharat_bandh_2024", "upper_caste_bandh_2018"):
        observed_map = targets[key]["core_events_by_state"]
        chosen = [i for i, s in enumerate(states) if s != "Delhi" and pop[i] > 1e7]
        observed = np.array([observed_map.get(states[i], 0) for i in chosen], float)
        model = [spearmanr(np.array(r["outputs"][key]["turnout_by_state_core"])[chosen], observed).correlation for r in kept]
        tests[key] = {"states": len(chosen), "model": interval(model),
                      "baseline_population": float(spearmanr(pop[chosen], observed).correlation),
                      "baseline_sc_plus_st": float(spearmanr(scst[chosen], observed).correlation)}
    return tests


def main():
    parser = argparse.ArgumentParser(description="History-match the v3 model to ten episodes.")
    parser.add_argument("--wave-size", type=int, default=6000)
    parser.add_argument("--waves", type=int, default=3)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    arguments = parser.parse_args()
    global CHECKPOINTS
    if arguments.checkpoint_dir:
        arguments.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        CHECKPOINTS = arguments.checkpoint_dir
    targets = load_targets()
    first = run_wave(sample_priors(np.random.default_rng(20261001), arguments.wave_size, PRIORS), "shared_wave1")
    kept, history = history_match(targets, arguments.wave_size, arguments.waves, 20261002, True, first, "main")
    kept_free, history_free = history_match(targets, arguments.wave_size, arguments.waves, 20261003, False, first, "nospatial")
    samples = [r["sample"] for r in kept]
    output = {"cutoff": CUTOFF, "priors": {k: list(v) for k, v in PRIORS.items()}, "history": history, "kept": len(kept),
              "history_no_spatial": history_free, "kept_no_spatial": len(kept_free)}
    if kept:
        output["posterior"] = {name: interval([s[name] for s in samples]) for name in NAMES}
        output["identification"] = {name: float((np.percentile([s[name] for s in samples], 95) - np.percentile([s[name] for s in samples], 5))
                                                / (PRIORS[name][1] - PRIORS[name][0])) for name in NAMES}
        output["turnout_2018_bandh"] = interval([r["outputs"]["sc_st_bharat_bandh_2018"]["turnout_core"][0] for r in kept])
        output["model_events_per_1000"] = {key: interval([sum(r["outputs"][key]["events_by_day"]) / (targets[key]["india_events_core_days"] / 1000) for r in kept])
                                           for key in EPISODE_KEYS}
        output["model_profiles"] = {key: np.median([np.array(r["outputs"][key]["events_by_day"]) / max(sum(r["outputs"][key]["events_by_day"]), 1e-12)
                                                    for r in kept], axis=0).tolist() for key in EPISODE_KEYS}
        output["model_deaths"] = {key: interval([r["outputs"][key]["expected_deaths_core"] for r in kept]) for key in EPISODE_KEYS}
        output["spatial_in_sample"] = spatial_tests(kept, targets)
        output["magnitude_ratio_to_2018"] = {key: interval([s[f"magnitude_{key}"] / s["magnitude_sc_st_bharat_bandh_2018"] for s in samples])
                                             for key in EPISODE_KEYS}
    if kept_free:
        output["spatial_out_of_sample"] = spatial_tests(kept_free, targets)
    RESULTS.mkdir(exist_ok=True)
    text = json.dumps(output, indent=1)
    (RESULTS / "v3_calibration.json").write_text(text)
    (RESULTS / "v3_nroy_samples.json").write_text(json.dumps(samples, indent=1))
    print(json.dumps({k: output[k] for k in ("kept", "kept_no_spatial", "history")}, indent=1))


if __name__ == "__main__":
    main()
