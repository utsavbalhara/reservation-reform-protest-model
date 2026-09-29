"""Draw a random sample of extracted JEE (Advanced) rank-list entries to check by hand against the published reports.

For each year, samples candidates from data/derived/jee_advanced_ranks_<year>.csv and writes their Common Rank List
rank, category and category rank to data/coding/rank_list_spot_check.csv, with an empty 'matches_report' column. Look
up each rank in that year's JIC report (https://jeeadv.ac.in/reports/<year>.pdf; the lists are sorted by roll number,
so search the category list for the category rank, or the CRL for the rank) and put 1 if the category and ranks match,
0 if not.
"""
import argparse
from pathlib import Path

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
YEARS = (2021, 2022, 2023, 2024, 2025)


def main():
    parser = argparse.ArgumentParser(description="Sample rank-list entries for a hand check.")
    parser.add_argument("--per-year", type=int, default=20)
    arguments = parser.parse_args()
    samples = []
    for year in YEARS:
        ranks = pd.read_csv(REPOSITORY_ROOT / "data" / "derived" / f"jee_advanced_ranks_{year}.csv")
        reserved = ranks[ranks.category != "GEN"]
        drawn = pd.concat([reserved.sample(arguments.per_year // 2, random_state=year),
                           ranks.sample(arguments.per_year - arguments.per_year // 2, random_state=year + 1)])
        drawn.insert(0, "year", year)
        drawn["report"] = f"https://jeeadv.ac.in/reports/{year}.pdf"
        drawn["matches_report"] = ""
        samples.append(drawn.sort_values("crl_rank"))
    output = REPOSITORY_ROOT / "data" / "coding" / "rank_list_spot_check.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(samples).to_csv(output, index=False)
    print(f"Wrote {sum(len(s) for s in samples)} entries to {output}")


if __name__ == "__main__":
    main()
