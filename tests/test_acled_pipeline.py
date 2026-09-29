"""The ACLED pipeline on a small synthetic export (no real ACLED data is committed)."""
import csv

from data_pipelines import acled_episode_events as acled

COLUMNS = ["event_id_cnty", "event_date", "year", "event_type", "sub_event_type", "country", "admin1", "admin2", "notes",
           "fatalities", "tags"]


def write_export(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in COLUMNS})


def test_crowd_sizes_are_parsed_as_lower_bounds():
    assert acled.crowd_size("crowd size=around 3,000") == 3000
    assert acled.crowd_size("crowd size=between 200 and 300") == 200
    assert acled.crowd_size("crowd size=hundreds") == 200
    assert acled.crowd_size("crowd size=no report") is None
    assert acled.crowd_size("") is None


def test_dates_in_both_export_formats():
    assert acled.parse_date("02 April 2018").isoformat() == "2018-04-02"
    assert acled.parse_date("2018-04-02").isoformat() == "2018-04-02"


def test_episode_counts_use_type_date_and_keywords(tmp_path):
    rows = [
        {"event_date": "02 April 2018", "event_type": "Protests", "country": "India", "admin1": "Punjab",
         "notes": "Dalit groups observed a Bharat Bandh against the SC/ST Act ruling.", "fatalities": "0", "tags": "crowd size=around 500"},
        {"event_date": "02 April 2018", "event_type": "Riots", "country": "India", "admin1": "Madhya Pradesh",
         "notes": "Violence during the Bharat Bandh called by Dalit organisations.", "fatalities": "3", "tags": "crowd size=no report"},
        {"event_date": "02 April 2018", "event_type": "Protests", "country": "India", "admin1": "Kerala",
         "notes": "Nurses protested for higher wages.", "fatalities": "0", "tags": ""},
        {"event_date": "02 April 2018", "event_type": "Battles", "country": "India", "admin1": "Chhattisgarh",
         "notes": "Clash with Dalit armed group.", "fatalities": "2", "tags": ""},
        {"event_date": "20 April 2018", "event_type": "Protests", "country": "India", "admin1": "Punjab",
         "notes": "Dalit protest.", "fatalities": "0", "tags": ""},
    ]
    write_export(tmp_path / "export.csv", rows)
    events = acled.read_events(tmp_path)
    assert len(events) == 4  # the Battles event is dropped
    summary, daily = acled.summarize(events)
    episode = summary["sc_st_bharat_bandh_2018"]
    assert episode["core_events"] == 2
    assert episode["window_events"] == 2  # 20 April is outside the window; the wage protest does not match
    assert episode["core_fatalities"] == 3
    assert episode["core_reported_crowd_sum"] == 500 and episode["core_events_reporting_crowd_size"] == 1
    assert episode["core_events_by_state"] == {"Punjab": 1, "Madhya Pradesh": 1}
    assert summary["patidar_2015"]["covered_by_acled"] is False
