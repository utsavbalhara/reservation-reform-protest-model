"""Count reservation-related protest events in GDELT 1.0 for each episode window.

Filter: action located in India (ActionGeo_CountryCode IN), CAMEO root code 14 (protest). Relevance: an episode keyword
appears in the source article's URL or in an actor name, and no exclusion keyword does. Violent protest is CAMEO 145.
GDELT duplicates events across articles, so both unique event IDs and unique source articles are counted. Counts are
also given per 1,000 GDELT events located in India that day, which absorbs changes in GDELT's overall volume.

Outputs:
  data/derived/gdelt_episode_daily.csv      episode, date, state, relevant events, relevant articles, violent, totals
  data/derived/gdelt_episode_locations.csv  one row per relevant event: episode, date, state, geo type, lat, long, place
The keyword filter's precision is estimated on a labelled sample (data/derived/gdelt_relevance_audit.csv).
"""
import csv
import zipfile
from pathlib import Path

from .episodes import EPISODES
from .gdelt_download import GDELT_FOLDER, days

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DERIVED_FOLDER = REPOSITORY_ROOT / "data" / "derived"
COLUMNS = ["GLOBALEVENTID", "SQLDATE", "MonthYear", "Year", "FractionDate", "Actor1Code", "Actor1Name", "Actor1CountryCode",
           "Actor1KnownGroupCode", "Actor1EthnicCode", "Actor1Religion1Code", "Actor1Religion2Code", "Actor1Type1Code",
           "Actor1Type2Code", "Actor1Type3Code", "Actor2Code", "Actor2Name", "Actor2CountryCode", "Actor2KnownGroupCode",
           "Actor2EthnicCode", "Actor2Religion1Code", "Actor2Religion2Code", "Actor2Type1Code", "Actor2Type2Code",
           "Actor2Type3Code", "IsRootEvent", "EventCode", "EventBaseCode", "EventRootCode", "QuadClass", "GoldsteinScale",
           "NumMentions", "NumSources", "NumArticles", "AvgTone", "Actor1Geo_Type", "Actor1Geo_FullName",
           "Actor1Geo_CountryCode", "Actor1Geo_ADM1Code", "Actor1Geo_Lat", "Actor1Geo_Long", "Actor1Geo_FeatureID",
           "Actor2Geo_Type", "Actor2Geo_FullName", "Actor2Geo_CountryCode", "Actor2Geo_ADM1Code", "Actor2Geo_Lat",
           "Actor2Geo_Long", "Actor2Geo_FeatureID", "ActionGeo_Type", "ActionGeo_FullName", "ActionGeo_CountryCode",
           "ActionGeo_ADM1Code", "ActionGeo_Lat", "ActionGeo_Long", "ActionGeo_FeatureID", "DATEADDED", "SOURCEURL"]
INDEX = {name: position for position, name in enumerate(COLUMNS)}

# FIPS 10-4 first-level codes that GDELT 1.0 uses for Indian states and union territories.
STATE_OF_FIPS = {
    "IN01": "Andaman and Nicobar Islands", "IN02": "Andhra Pradesh", "IN03": "Assam", "IN05": "Chandigarh",
    "IN06": "Dadra and Nagar Haveli", "IN07": "Delhi", "IN09": "Gujarat", "IN10": "Haryana", "IN11": "Himachal Pradesh",
    "IN12": "Jammu and Kashmir", "IN13": "Kerala", "IN14": "Lakshadweep", "IN16": "Maharashtra", "IN17": "Manipur",
    "IN18": "Meghalaya", "IN19": "Karnataka", "IN20": "Nagaland", "IN21": "Odisha", "IN22": "Puducherry", "IN23": "Punjab",
    "IN24": "Rajasthan", "IN25": "Tamil Nadu", "IN26": "Tripura", "IN28": "West Bengal", "IN29": "Sikkim",
    "IN30": "Arunachal Pradesh", "IN31": "Mizoram", "IN32": "Daman and Diu", "IN33": "Goa", "IN34": "Bihar",
    "IN35": "Madhya Pradesh", "IN36": "Uttar Pradesh", "IN37": "Chhattisgarh", "IN38": "Jharkhand", "IN39": "Uttarakhand",
    "IN40": "Telangana", "IN": "India (country-level)",
}


def read_day(day):
    path = GDELT_FOLDER / f"{day:%Y%m%d}.export.CSV.zip"
    with zipfile.ZipFile(path) as archive:
        name = archive.namelist()[0]
        with archive.open(name) as handle:
            for raw in handle:
                fields = raw.decode("utf-8", errors="replace").rstrip("\n").split("\t")
                if len(fields) >= len(COLUMNS):
                    yield fields


def is_relevant(fields, keywords, excluded):
    text = " ".join((fields[INDEX["SOURCEURL"]], fields[INDEX["Actor1Name"]], fields[INDEX["Actor2Name"]])).lower()
    return any(keyword in text for keyword in keywords) and not any(keyword in text for keyword in excluded)


def process_episode(key, episode):
    daily, locations = [], []
    for day in days(*episode["window"]):
        if not (GDELT_FOLDER / f"{day:%Y%m%d}.export.CSV.zip").exists():
            continue
        by_state, india_events, india_protests = {}, 0, 0
        seen_events = set()
        for fields in read_day(day):
            if fields[INDEX["ActionGeo_CountryCode"]] != "IN":
                continue
            india_events += 1
            if fields[INDEX["EventRootCode"]] != "14":
                continue
            india_protests += 1
            if not is_relevant(fields, episode["keywords"], episode["exclude_keywords"]):
                continue
            strict = is_relevant(fields, episode.get("strict_keywords", episode["keywords"]),
                                 episode.get("strict_exclude_keywords", episode["exclude_keywords"]))
            event_id = fields[INDEX["GLOBALEVENTID"]]
            if event_id in seen_events:
                continue
            seen_events.add(event_id)
            state = STATE_OF_FIPS.get(fields[INDEX["ActionGeo_ADM1Code"]], fields[INDEX["ActionGeo_ADM1Code"]] or "unknown")
            record = by_state.setdefault(state, {"events": 0, "articles": set(), "violent": 0, "strict": 0})
            record["events"] += 1
            record["strict"] += strict
            record["articles"].add(fields[INDEX["SOURCEURL"]])
            record["violent"] += fields[INDEX["EventCode"]] == "145"
            locations.append({"episode": key, "date": day.isoformat(), "state": state, "geo_type": fields[INDEX["ActionGeo_Type"]],
                              "lat": fields[INDEX["ActionGeo_Lat"]], "long": fields[INDEX["ActionGeo_Long"]],
                              "place": fields[INDEX["ActionGeo_FullName"]], "event_code": fields[INDEX["EventCode"]], "strict": int(strict),
                              "url": fields[INDEX["SOURCEURL"]]})
        for state, record in sorted(by_state.items()):
            daily.append({"episode": key, "date": day.isoformat(), "state": state, "relevant_events": record["events"],
                          "relevant_articles": len(record["articles"]), "violent_relevant_events": record["violent"], "strict_relevant_events": record["strict"],
                          "india_protest_events": india_protests, "india_events": india_events})
        if not by_state:
            daily.append({"episode": key, "date": day.isoformat(), "state": "", "relevant_events": 0, "relevant_articles": 0,
                          "violent_relevant_events": 0, "strict_relevant_events": 0, "india_protest_events": india_protests, "india_events": india_events})
    return daily, locations


def main():
    DERIVED_FOLDER.mkdir(parents=True, exist_ok=True)
    all_daily, all_locations = [], []
    for key, episode in EPISODES.items():
        # One pass per episode; each episode only reads its own window's files.
        daily, locations = process_episode(key, episode)
        all_daily += daily
        all_locations += locations
        core = sum(row["relevant_events"] for row in daily if row["date"] in episode["core_days"])
        print(f"{key}: {sum(r['relevant_events'] for r in daily)} relevant events in window, {core} on core days", flush=True)
    with open(DERIVED_FOLDER / "gdelt_episode_daily.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_daily[0]))
        writer.writeheader()
        writer.writerows(all_daily)
    with open(DERIVED_FOLDER / "gdelt_episode_locations.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_locations[0]))
        writer.writeheader()
        writer.writerows(all_locations)


if __name__ == "__main__":
    main()
