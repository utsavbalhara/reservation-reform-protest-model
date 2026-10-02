"""How much do the history-matching results depend on its tuning choices, and on the number of seeds per set?

1. Discrepancy and cutoff. The final wave's simulated sets are judged again with the model discrepancy (0.5 on the log
   scale in the main match) set to 0.3 and 0.8, and the implausibility cutoff (3) set to 2.5 and 3.5. For each setting the
   script reports how many sets survive and the 90% range of the behavioural parameters, the 2018 shock, the 2024-to-2018
   ratio (which sets the residual threat of lever L3) and the implied 2018 turnout. Only the final wave is re-judged: a
   looser setting would also have widened the earlier waves' bounds, so the loosest ranges here are lower bounds.
2. Seeds. Each set was simulated with two seeds. The retained sets are simulated again with ten seeds and judged again
   with the reference settings, to see how many stay non-implausible when simulation noise is better estimated.
3. Identification. For each behavioural parameter, the width of the retained 90% range as a share of the prior's width.

The retained sets are not a weighted posterior: every set that survives counts once, and the grounded runs draw one of
them uniformly at random.

Needs the wave checkpoints written by experiments/episode_calibration.py --checkpoint-dir.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import experiments.episode_calibration as calibration

RESULTS_FOLDER = Path(__file__).resolve().parent.parent / "results"
DISCREPANCIES = (0.3, 0.5, 0.8)
CUTOFFS = (2.5, 3.0, 3.5)
BEHAVIOURAL = ("mean_participation_threshold", "participation_threshold_spread", "same_group_neighbourhood_share")


def ranges(kept):
    def interval(values):
        values = np.asarray(values, float)
        return [round(float(np.percentile(values, 5)), 3), round(float(np.median(values)), 3), round(float(np.percentile(values, 95)), 3)]
    samples = [result["sample"] for result in kept]
    ratio = [s["magnitude_sc_st_bharat_bandh_2024"] / s["magnitude_sc_st_bharat_bandh_2018"] for s in samples]
    out = {name: interval([s[name] for s in samples]) for name in BEHAVIOURAL + ("magnitude_sc_st_bharat_bandh_2018",)}
    out["ratio_2024_to_2018"] = interval(ratio)
    out["share_ratio_above_one"] = round(float(np.mean(np.array(ratio) > 1)), 3)
    out["bandh_day_turnout_2018_lakh"] = interval([r["outputs"]["sc_st_bharat_bandh_2018"]["bandh_day_turnout"] / 1e5 for r in kept])
    return out


def judged(results, targets, discrepancy, cutoff):
    calibration.MODEL_DISCREPANCY_LOG = discrepancy
    kept = []
    for result in results:
        scores = calibration.implausibilities(result, targets)
        if max(scores.values()) < cutoff:
            kept.append(result)
    calibration.MODEL_DISCREPANCY_LOG = 0.5
    return kept


def main():
    parser = argparse.ArgumentParser(description="Sensitivity of history matching to discrepancy, cutoff and seeds.")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--wave-size", type=int, default=6000)
    parser.add_argument("--seeds", type=int, default=10)
    arguments = parser.parse_args()
    targets = json.loads(calibration.TARGETS_PATH.read_text())
    final_wave = json.loads((arguments.checkpoint_dir / f"20260929_{arguments.wave_size}_wave3.json").read_text())
    output = {"final_wave_sets": len(final_wave), "settings": [], "prior_ranges": {n: list(calibration.PRIORS[n]) for n in BEHAVIOURAL}}
    for discrepancy in DISCREPANCIES:
        for cutoff in CUTOFFS:
            kept = judged(final_wave, targets, discrepancy, cutoff)
            record = {"model_discrepancy_log": discrepancy, "cutoff": cutoff, "kept": len(kept)}
            if len(kept) >= 5:
                record.update(ranges(kept))
            output["settings"].append(record)
            print(f"discrepancy {discrepancy} cutoff {cutoff}: kept {len(kept)}", flush=True)

    reference = judged(final_wave, targets, 0.5, 3.0)
    output["identification"] = {}
    for name in BEHAVIOURAL:
        low, high = calibration.PRIORS[name]
        values = [result["sample"][name] for result in reference]
        output["identification"][name] = round((np.percentile(values, 95) - np.percentile(values, 5)) / (high - low), 3)

    # More seeds for the retained sets.
    calibration.SEEDS = tuple(range(11, 11 + arguments.seeds))
    resimulated = calibration.run_wave([result["sample"] for result in reference])
    still = [result for result in resimulated if max(calibration.implausibilities(result, targets).values()) < calibration.IMPLAUSIBILITY_CUTOFF]
    two = np.array([r["outputs"]["sc_st_bharat_bandh_2018"]["events"] for r in reference])
    many = np.array([r["outputs"]["sc_st_bharat_bandh_2018"]["events"] for r in resimulated])
    output["more_seeds"] = {
        "seeds": arguments.seeds, "retained_with_two_seeds": len(reference), "still_retained": len(still),
        "share_still_retained": round(len(still) / len(reference), 3),
        "median_abs_log_change_2018_events": round(float(np.median(np.abs(np.log(np.maximum(many, 1e-9)) - np.log(np.maximum(two, 1e-9))))), 3),
        "median_seed_log_sd_2018_events": {"two_seeds": round(float(np.median([r["outputs"]["sc_st_bharat_bandh_2018"]["log_events_sd"] for r in reference])), 3),
                                           "many_seeds": round(float(np.median([r["outputs"]["sc_st_bharat_bandh_2018"]["log_events_sd"] for r in resimulated])), 3)},
        "ranges_still_retained": ranges(still) if len(still) >= 5 else None,
    }
    RESULTS_FOLDER.mkdir(exist_ok=True)
    path = RESULTS_FOLDER / "calibration_sensitivity.json"
    path.write_text(json.dumps(output, indent=1))
    print(json.dumps(output["identification"]), json.dumps(output["more_seeds"])[:600])
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
