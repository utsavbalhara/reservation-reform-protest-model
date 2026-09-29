"""Count reservation-related protest events in ACLED for each episode, as an independent check on the GDELT counts.

ACLED (Armed Conflict Location & Event Data; Raleigh et al. 2010) codes events by hand from news and local sources, with
a location, event type, reported fatalities and, often, a reported crowd size (in the tags field). Its India coverage
starts in 2016, so it covers the 2018, 2019 and 2024 mechanism episodes and the 2016 onward observation episodes, but
not the Patidar agitation of 2015.

Input: one or more ACLED export files (CSV, the standard columns) in data/raw/acled/. ACLED's terms of use do not allow
redistributing the event data, so the raw files are not committed. Only aggregates are written:
  data/derived/acled_episode_summary.json   per episode: events on core days and in the window, by state, fatalities,
                                            reported crowd sizes (sum and count of events that report one)
  data/derived/acled_episode_daily.csv      per episode, date and state: events, fatalities

Relevance: an event is counted for an episode if it is a protest or riot in India on a day in the episode's window and
its notes mention one of the episode's keywords (ACLED_KEYWORDS below, written for ACLED's prose notes rather than for
URLs). Every rule is in this file.
"""
import csv
import json
import re
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

from .episodes import EPISODES

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
RAW_FOLDER = REPOSITORY_ROOT / "data" / "raw" / "acled"
DERIVED_FOLDER = REPOSITORY_ROOT / "data" / "derived"
EVENT_TYPES = {"Protests", "Riots"}

ACLED_KEYWORDS = {
    "sc_st_bharat_bandh_2018": ["bharat bandh", "sc/st", "sc-st", "sc and st", "scheduled caste", "atrocities act", "atrocity act",
                                "prevention of atrocities", "dalit"],
    "upper_caste_bandh_2018": ["bharat bandh", "sc/st act", "sc-st act", "atrocities act", "upper caste", "upper-caste", "savarna",
                               "general category", "brahmin", "karni sena"],
    "sc_st_bharat_bandh_2024": ["bharat bandh", "sub-classification", "subclassification", "sub classification", "creamy layer",
                                "sc/st", "sc-st", "scheduled caste", "dalit", "reservation bachao"],
    "ews_quota_2019": ["economically weaker", "ews", "10 percent reservation", "10 per cent reservation", "10% reservation",
                       "10 percent quota", "10% quota", "quota for upper caste", "upper caste quota"],
    "patidar_2015": ["patidar", "hardik patel", "paas"],
    "jat_2016": ["jat reservation", "jat quota", "jat agitation", "jat community", "jats "],
    "kapu_2016": ["kapu"],
    "maratha_march_mumbai_2017": ["maratha"],
    "maratha_quota_2018": ["maratha"],
    "gujjar_2019": ["gujjar", "gurjar"],
}
# Conventional lower bounds for verbal crowd sizes; numeric reports are used as given.
VERBAL_CROWD_SIZES = {"tens of thousands": 20_000, "hundreds of thousands": 200_000, "several thousand": 3_000, "thousands": 2_000,
                      "several hundred": 300, "hundreds": 200, "dozens": 24, "scores": 40}


def parse_date(text: str) -> date:
    text = text.strip()
    for pattern in ("%Y-%m-%d", "%d %B %Y", "%d-%B-%Y", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(text, pattern).date()
        except ValueError:
            continue
    raise ValueError(f"unrecognized ACLED date {text!r}")


def crowd_size(tags: str):
    """The crowd size reported in the tags field, as a number of people, or None when not reported."""
    match = re.search(r"crowd size=([^;]*)", tags or "", flags=re.IGNORECASE)
    if not match:
        return None
    value = match.group(1).strip().lower()
    if not value or "no report" in value:
        return None
    digits = re.findall(r"\d[\d,]*", value)
    if digits:
        # For a range ("between 200 and 300", "200-300") the first number, the lower bound, is used.
        return int(digits[0].replace(",", ""))
    for phrase, size in VERBAL_CROWD_SIZES.items():
        if phrase in value:
            return size
    return None


def read_events(folder: Path = RAW_FOLDER):
    rows = []
    for path in sorted(folder.glob("*.csv")):
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if row.get("country", "India") != "India" or row.get("event_type") not in EVENT_TYPES:
                    continue
                rows.append({"date": parse_date(row["event_date"]), "state": row.get("admin1", ""), "district": row.get("admin2", ""),
                             "event_type": row["event_type"], "sub_event_type": row.get("sub_event_type", ""),
                             "notes": (row.get("notes") or "").lower(), "fatalities": int(row.get("fatalities") or 0),
                             "crowd_size": crowd_size(row.get("tags", ""))})
    return rows


def matches(row, key):
    return any(keyword in row["notes"] for keyword in ACLED_KEYWORDS[key])


def summarize(rows):
    summary, daily = {}, defaultdict(lambda: {"events": 0, "fatalities": 0})
    for key, episode in EPISODES.items():
        start, end = (date.fromisoformat(day) for day in episode["window"])
        core = {date.fromisoformat(day) for day in episode["core_days"]}
        chosen = [row for row in rows if start <= row["date"] <= end and matches(row, key)]
        core_rows = [row for row in chosen if row["date"] in core]
        by_state = defaultdict(int)
        for row in core_rows:
            by_state[row["state"]] += 1
        sizes = [row["crowd_size"] for row in core_rows if row["crowd_size"]]
        summary[key] = {
            "window_events": len(chosen), "core_events": len(core_rows),
            "core_events_by_state": dict(sorted(by_state.items(), key=lambda item: -item[1])),
            "core_fatalities": sum(row["fatalities"] for row in core_rows),
            "window_fatalities": sum(row["fatalities"] for row in chosen),
            "core_events_reporting_crowd_size": len(sizes), "core_reported_crowd_sum": sum(sizes),
            "core_violent_share": round(sum(row["event_type"] == "Riots" for row in core_rows) / max(len(core_rows), 1), 3),
            "covered_by_acled": start.year >= 2016,
        }
        for row in chosen:
            cell = daily[(key, row["date"].isoformat(), row["state"])]
            cell["events"] += 1
            cell["fatalities"] += row["fatalities"]
    return summary, daily


def main():
    rows = read_events()
    if not rows:
        raise SystemExit(f"No ACLED protest or riot events for India found in {RAW_FOLDER}. Put the ACLED export CSV there.")
    summary, daily = summarize(rows)
    DERIVED_FOLDER.mkdir(parents=True, exist_ok=True)
    (DERIVED_FOLDER / "acled_episode_summary.json").write_text(json.dumps(summary, indent=1))
    with (DERIVED_FOLDER / "acled_episode_daily.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["episode", "date", "state", "events", "fatalities"])
        for (key, day, state), cell in sorted(daily.items()):
            writer.writerow([key, day, state, cell["events"], cell["fatalities"]])
    for key, value in summary.items():
        print(f"{key:>28}: core events {value['core_events']:>4}, fatalities {value['core_fatalities']:>3}, "
              f"crowd sizes reported {value['core_events_reporting_crowd_size']:>3} (sum {value['core_reported_crowd_sum']:,})")


if __name__ == "__main__":
    main()
