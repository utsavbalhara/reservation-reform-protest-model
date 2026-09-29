"""Turn GDELT counts and the relevance audit into calibration targets.

For each episode: relevant protest events on its core days (broad filter), corrected by the audited precision on core
days (Beta(1 + relevant, 1 + irrelevant) posterior), and expressed per 1,000 GDELT events located in India on those
days, which absorbs GDELT's changing coverage. State-level counts on core days are kept for the spatial tests. The strict
filter's precision and recall are measured on the audited sample (recall relative to the broad filter's relevant hits).
"""
import json

import numpy as np
import pandas as pd

from .episodes import EPISODES
from .gdelt_episode_events import DERIVED_FOLDER

TARGETS_PATH = DERIVED_FOLDER / "episode_targets.json"


def strict_match(url: str, episode: dict) -> bool:
    text = url.lower()
    keywords = episode.get("strict_keywords", episode["keywords"])
    excluded = episode.get("strict_exclude_keywords", episode["exclude_keywords"])
    return any(k in text for k in keywords) and not any(k in text for k in excluded)


def main():
    daily = pd.read_csv(DERIVED_FOLDER / "gdelt_episode_daily.csv")
    audit = pd.read_csv(DERIVED_FOLDER / "gdelt_relevance_audit.csv").dropna(subset=["relevant"])
    targets = {}
    for key, episode in EPISODES.items():
        rows = daily[daily.episode == key]
        core = rows[rows.date.isin(episode["core_days"])]
        per_day = rows.groupby("date").agg(events=("relevant_events", "sum"), strict=("strict_relevant_events", "sum"),
                                           india=("india_events", "first")).reset_index()
        core_totals = per_day[per_day.date.isin(episode["core_days"])]
        labelled = audit[(audit.episode == key) & audit.core_day]
        relevant, total = int(labelled.relevant.sum()), int(len(labelled))
        precision_draws = np.random.default_rng(1).beta(1 + relevant, 1 + total - relevant, 4000)
        broad_core = int(core_totals.events.sum())
        india_core = int(core_totals.india.sum())
        corrected = broad_core * precision_draws
        per_thousand = corrected / india_core * 1000
        state_counts = core.groupby("state").relevant_events.sum().sort_values(ascending=False)
        labelled_all = audit[audit.episode == key]
        strict_flags = labelled_all.url.map(lambda u: strict_match(u, episode))
        strict_precision = float(labelled_all[strict_flags].relevant.mean()) if strict_flags.any() else None
        strict_recall = float(strict_flags[labelled_all.relevant == 1].mean()) if (labelled_all.relevant == 1).any() else None
        targets[key] = {
            "role": episode["role"], "core_days": episode["core_days"],
            "broad_core_events": broad_core, "strict_core_events": int(core_totals.strict.sum()), "india_events_core_days": india_core,
            "audit_core_relevant": relevant, "audit_core_labelled": total,
            "precision_core_median": round(float(np.median(precision_draws)), 3),
            "corrected_core_events": {"median": round(float(np.median(corrected)), 1), "p05": round(float(np.percentile(corrected, 5)), 1),
                                      "p95": round(float(np.percentile(corrected, 95)), 1)},
            "corrected_core_events_per_1000_india_events": {"median": round(float(np.median(per_thousand)), 3),
                                                             "p05": round(float(np.percentile(per_thousand, 5)), 3),
                                                             "p95": round(float(np.percentile(per_thousand, 95)), 3)},
            "core_events_by_state": {state: int(count) for state, count in state_counts.items() if isinstance(state, str) and state},
            "strict_filter_precision_on_audit": None if strict_precision is None else round(strict_precision, 3),
            "strict_filter_recall_on_audit": None if strict_recall is None else round(strict_recall, 3),
            "deaths": episode.get("deaths"), "party_backing": episode.get("party_backing"),
        }
    TARGETS_PATH.write_text(json.dumps(targets, indent=1))
    for key, value in targets.items():
        per = value["corrected_core_events_per_1000_india_events"]
        print(f"{key:>28}: core events {value['broad_core_events']:>4} x precision {value['precision_core_median']:.2f} -> "
              f"{value['corrected_core_events']['median']:>6.1f} ({per['median']:.2f} per 1,000 India events) | strict precision "
              f"{value['strict_filter_precision_on_audit']} recall {value['strict_filter_recall_on_audit']}")


if __name__ == "__main__":
    main()
