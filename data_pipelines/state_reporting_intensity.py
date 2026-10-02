"""How heavily GDELT reports each state, whatever the topic.

For every downloaded GDELT day (the episode windows, 175 days from 2015 to 2024) this counts events located in each
Indian state, of all kinds and of the protest kind (root code 14). A state's reporting intensity is its share of India's
events divided by its share of India's population (Census 2011): above 1, the state generates more news events per
person than India as a whole. The v3 observation model multiplies each state's expected protest events by intensity
to a fitted power, so the model is not asked to explain media geography with protest turnout.

Output: data/derived/gdelt_state_reporting_intensity.csv
"""
from collections import Counter
from pathlib import Path

import pandas as pd

from data_pipelines.gdelt_episode_events import GDELT_FOLDER, INDEX, STATE_OF_FIPS
from data_pipelines.gdelt_episode_events import read_day as _read_day

DERIVED = Path(__file__).resolve().parent.parent / "data" / "derived"


def main():
    all_events, protest_events, days = Counter(), Counter(), 0
    for path in sorted(GDELT_FOLDER.glob("*.export.CSV.zip")):
        day = pd.Timestamp(path.name[:8]).date()
        days += 1
        for fields in _read_day(day):
            if fields[INDEX["ActionGeo_CountryCode"]] != "IN":
                continue
            state = STATE_OF_FIPS.get(fields[INDEX["ActionGeo_ADM1Code"]])
            if state is None or state == "India (country-level)":
                continue
            all_events[state] += 1
            protest_events[state] += fields[INDEX["EventRootCode"]] == "14"
    population = pd.read_csv(DERIVED / "census2011_district_sc_st.csv").groupby("state").population.sum()
    table = pd.DataFrame({"all_events": pd.Series(all_events), "protest_events": pd.Series(protest_events)}).fillna(0)
    table.index = [name.replace(" and ", " & ") for name in table.index]
    table = table.join(population, how="inner")
    for column in ("all_events", "protest_events"):
        table[f"{column}_intensity"] = (table[column] / table[column].sum()) / (table.population / table.population.sum())
    table.index.name = "state"
    table.sort_values("all_events_intensity", ascending=False).round(4).to_csv(DERIVED / "gdelt_state_reporting_intensity.csv")
    print(f"{days} days")
    print(table.sort_values("all_events_intensity", ascending=False).round(2).to_string())


if __name__ == "__main__":
    main()
