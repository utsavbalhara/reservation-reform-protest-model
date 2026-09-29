"""Download GDELT 1.0 daily event files for the episode windows (data/raw/gdelt/, not committed).

GDELT 1.0 daily files exist from 1 April 2013: http://data.gdeltproject.org/events/YYYYMMDD.export.CSV.zip
"""
import datetime
import sys
import time
import urllib.request
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
GDELT_FOLDER = REPOSITORY_ROOT / "data" / "raw" / "gdelt"

# (first day, last day) of each download window; see episodes.py for what each episode is.
EPISODE_WINDOWS = {
    "patidar_2015": ("2015-08-18", "2015-09-06"),
    "kapu_2016": ("2016-01-25", "2016-02-08"),
    "jat_2016": ("2016-02-08", "2016-03-01"),
    "maratha_march_mumbai_2017": ("2017-08-03", "2017-08-15"),
    "sc_st_bharat_bandh_2018": ("2018-03-26", "2018-04-14"),
    "maratha_quota_2018": ("2018-07-20", "2018-08-12"),
    "upper_caste_bandh_2018": ("2018-08-31", "2018-09-14"),
    "ews_quota_2019": ("2019-01-05", "2019-01-22"),
    "gujjar_2019": ("2019-02-06", "2019-02-18"),
    "sc_st_bharat_bandh_2024": ("2024-08-14", "2024-08-28"),
}


def days(first: str, last: str):
    day = datetime.date.fromisoformat(first)
    end = datetime.date.fromisoformat(last)
    while day <= end:
        yield day
        day += datetime.timedelta(days=1)


def download(day: datetime.date) -> Path:
    GDELT_FOLDER.mkdir(parents=True, exist_ok=True)
    target = GDELT_FOLDER / f"{day:%Y%m%d}.export.CSV.zip"
    if target.exists() and target.stat().st_size > 0:
        return target
    url = f"http://data.gdeltproject.org/events/{day:%Y%m%d}.export.CSV.zip"
    for attempt in range(4):
        try:
            urllib.request.urlretrieve(url, target)
            return target
        except Exception as error:  # network hiccups through the proxy
            time.sleep(2 ** (attempt + 1))
            last_error = error
    raise RuntimeError(f"could not download {url}: {last_error}")


if __name__ == "__main__":
    chosen = sys.argv[1:] or list(EPISODE_WINDOWS)
    for key in chosen:
        first, last = EPISODE_WINDOWS[key]
        for day in days(first, last):
            path = download(day)
            print(key, day, path.stat().st_size, flush=True)
