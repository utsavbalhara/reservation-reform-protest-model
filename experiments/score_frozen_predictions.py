"""Score the frozen leave-one-episode-out predictions against the observations.

Each prediction file in results/frozen_predictions/ was written, with its SHA-256 digest, before this script runs. The
script refuses to score a file whose contents no longer match its digest. The digest guards against editing a prediction
after seeing the observations; it is not an external timestamp. Posting the .sha256 files publicly (for example on OSF)
before scoring would make it one.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from protest_simulation.geography import district_table

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
FROZEN_FOLDER = REPOSITORY_ROOT / "results" / "frozen_predictions"
TARGETS_PATH = REPOSITORY_ROOT / "data" / "derived" / "episode_targets.json"


def verified(path: Path) -> dict:
    text = path.read_text()
    expected = (path.parent / (path.stem + ".sha256")).read_text().strip()
    actual = hashlib.sha256(text.encode()).hexdigest()
    if actual != expected:
        raise RuntimeError(f"{path.name} has changed since it was frozen")
    return json.loads(text)


def main():
    targets = json.loads(TARGETS_PATH.read_text())
    table = district_table()
    population = table.groupby("state").population.sum()
    sc_st = (table.sc + table.st).groupby(table.state).sum()
    scores = {}
    for path in sorted(FROZEN_FOLDER.glob("loeo_*.json")):
        prediction = verified(path)
        key = prediction["held_out"]
        observed_deaths = targets[key]["deaths"]
        deaths = prediction["predicted_mean_deaths"]
        states = prediction["states"]
        chosen = [i for i, name in enumerate(states) if name != "Delhi" and population.get(name, 0) > 1e7]
        observed = np.array([targets[key]["core_events_by_state"].get(states[i], 0) for i in chosen], float)
        predicted = np.array(prediction["predicted_state_share"])[chosen] if prediction["predicted_state_share"] else None
        scores[key] = {
            "calibrated_on": prediction["calibrated_on"],
            "retained_parameter_sets": prediction["retained"],
            "deaths": {"observed": [observed_deaths["low"], observed_deaths["high"]], "predicted": deaths,
                       "observed_inside_90_percent_interval": None if deaths is None else bool(
                           deaths["p05"] <= 0.5 * (observed_deaths["low"] + observed_deaths["high"]) <= deaths["p95"])},
            "spatial_spearman": {
                "model": None if predicted is None else float(spearmanr(predicted, observed).correlation),
                "baseline_population": float(spearmanr([population[states[i]] for i in chosen], observed).correlation),
                "baseline_sc_plus_st_population": float(spearmanr([sc_st[states[i]] for i in chosen], observed).correlation),
                "states": len(chosen)},
        }
    (REPOSITORY_ROOT / "results" / "loeo_scores.json").write_text(json.dumps(scores, indent=1))
    print(json.dumps(scores, indent=1))


if __name__ == "__main__":
    main()
