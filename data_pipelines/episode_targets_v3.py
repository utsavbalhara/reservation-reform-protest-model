"""Calibration targets for the v3 model: all ten episodes, day by day and state by state.

Builds on data/derived/episode_targets.json (precision-corrected core-day event rates, state counts, deaths) and
data/derived/gdelt_episode_daily.csv (relevant events per day and state). For each episode it records:
  - the core days and their offsets from the first core day (the model's action days);
  - relevant events on each core day (for the day-to-day profile; the precision correction cancels in shares);
  - the precision-corrected core-day event rate per 1,000 GDELT events in India (for the level);
  - core-day events by state (for the spatial distribution);
  - reported deaths on core days, where they are protest deaths;
  - published crowd sizes, where available (Maratha march 2017, Patidar rally 2015), as turnout anchors.
Only core days are used: the relevance audit found poor precision outside them.

Writes data/derived/episode_targets_v3.json.
"""
import json
from datetime import date
from pathlib import Path

import pandas as pd

from .episodes import EPISODES

DERIVED = Path(__file__).resolve().parent.parent / "data" / "derived"

# Crowd anchors: (state, core-day index, low, high) in people on that day, from the ranges recorded in episodes.py.
CROWD_ANCHORS = {
    "maratha_march_mumbai_2017": ("Maharashtra", 0, 1e5, 2e6),   # police expectation 1-4 lakh; media 6-10 lakh; organisers 20 lakh
    "patidar_2015": ("Gujarat", 0, 3e5, 2e6),                     # 25 August 2015 Ahmedabad rally, "over 500,000"
}
# Deaths that were not protest violence are not used (Maratha 2018: mostly suicides).
DEATHS_NOT_USED = {"maratha_quota_2018"}


def main():
    targets = json.loads((DERIVED / "episode_targets.json").read_text())
    daily = pd.read_csv(DERIVED / "gdelt_episode_daily.csv")
    out = {}
    for key, episode in EPISODES.items():
        target = targets[key]
        rows = daily[daily.episode == key]
        by_day = rows.groupby("date").agg(events=("relevant_events", "sum"), india=("india_events", "first"))
        core = episode["core_days"]
        first = date.fromisoformat(core[0])
        offsets = [(date.fromisoformat(day) - first).days for day in core]
        out[key] = {
            "label": episode.get("label", key),
            "core_days": core,
            "action_day_offsets": offsets,
            "daily_core_events": [int(by_day.events.get(day, 0)) for day in core],
            "daily_india_events": [int(by_day.india.get(day, 0)) for day in core],
            "core_events_per_1000": target["corrected_core_events_per_1000_india_events"],
            "corrected_core_events": target["corrected_core_events"],
            "india_events_core_days": target["india_events_core_days"],
            "precision_core_median": target.get("precision_core_median"),
            "core_events_by_state": target["core_events_by_state"],
            "deaths": None if key in DEATHS_NOT_USED else target["deaths"],
            "party_backing": target["party_backing"],
            "crowd_anchor": None if key not in CROWD_ANCHORS else dict(zip(("state", "day_index", "low", "high"), CROWD_ANCHORS[key])),
        }
    path = DERIVED / "episode_targets_v3.json"
    path.write_text(json.dumps(out, indent=1))
    for key, value in out.items():
        print(f"{key:28} days {value['action_day_offsets']} events {value['daily_core_events']} deaths {value['deaths'] and (value['deaths']['low'], value['deaths']['high'])}")


if __name__ == "__main__":
    main()
