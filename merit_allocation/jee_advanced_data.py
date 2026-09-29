"""JEE (Advanced) inputs for the allocation model, all taken from the Joint Implementation Committee reports.

REGISTERED_BY_CATEGORY: table 8.1, "Zone wise and category wise distribution of eligible Registered Candidates".
APPEARED: section 3 ("candidates appeared for both the papers").
ALLOTTED_BY_CATEGORY: table 8.3, "Zone wise and category wise distribution of allotted Candidates" (candidates, by
their own category, who were allotted an IIT seat through JoSAA; supernumerary female seats included).
The rank lists themselves come from data/derived/jee_advanced_ranks_<year>.csv (data_pipelines/jee_advanced_rank_lists.py).
Years 2019 and 2020 are excluded: their extracted lists fail the within-category rank-consistency check
(Spearman correlation between category rank and CRL rank below 0.999), which indicates parsing contamination.
"""
import csv
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DERIVED_FOLDER = REPOSITORY_ROOT / "data" / "derived"
CATEGORIES = ("GEN", "GEN-EWS", "OBC-NCL", "SC", "ST")
VALIDATED_YEARS = (2021, 2022, 2023, 2024, 2025)

# Filled in by read_report_tables(); kept here after extraction so the analysis does not need the PDFs.
REPORT_TABLES = {}


def read_rank_table(year: int):
    rows = list(csv.DictReader(open(DERIVED_FOLDER / f"jee_advanced_ranks_{year}.csv")))
    crl = sorted((int(row["crl_rank"]), row["category"]) for row in rows if row["crl_rank"])
    list_sizes = {category: sum(1 for row in rows if row["category"] == category) for category in CATEGORIES[1:]}
    ranks = np.array([rank for rank, _ in crl])
    categories = np.array([CATEGORIES.index(category) for _, category in crl])
    return ranks, categories, list_sizes


# Transcribed from each JIC report. "registered" is table 8.1 (or the stage summary for 2022, which gives appeared
# counts directly); "appeared_total" is the report's count of candidates who sat both papers; "allotted" is table 8.3
# (candidates by their own category who were allotted an IIT seat). Category order follows CATEGORIES.
REPORT_TABLES = {
    2021: {"registered": (40951, 21981, 51788, 25019, 11454), "appeared_total": 141699,
           "allotted": (6576, 1672, 4394, 2433, 1221), "source": "JIC 2021 report, section 1(9) and tables 8.1, 8.3"},
    2022: {"appeared": (38180, 24104, 56538, 25287, 11429), "appeared_total": 155538,
           "allotted": (6477, 1883, 4538, 2494, 1243), "source": "JIC 2022 report, 'Summary of Number of Candidates' (PwD and non-PwD summed)"},
    2023: {"registered": (41680, 31622, 70685, 31044, 14456), "appeared_total": 180372,
           "allotted": (6617, 2043, 4747, 2616, 1317), "source": "JIC 2023 report, section 3(i) and tables 8.1, 8.3"},
    2024: {"registered": None, "appeared_total": 180200, "allotted": None,
           "source": "JIC 2024 report, section 3; category tables are images, so category counts are imputed from 2023 and 2025 shares"},
    2025: {"registered": (41202, 29939, 70148, 31130, 14804), "appeared_total": 180422,
           "allotted": (7118, 2008, 4965, 2730, 1367), "source": "JIC 2025 report, section 3 and tables 8.1, 8.3 (columns reordered)"},
}


def appeared_by_category(year: int) -> tuple:
    """Candidates who sat both papers, by category, and whether the split was imputed."""
    table = REPORT_TABLES[year]
    if table.get("appeared") is not None:
        return np.array(table["appeared"], float), False
    if table.get("registered") is not None:
        registered = np.array(table["registered"], float)
        return registered * table["appeared_total"] / registered.sum(), False
    shares = np.mean([np.array(REPORT_TABLES[y]["registered"], float) / sum(REPORT_TABLES[y]["registered"]) for y in (2023, 2025)], axis=0)
    return shares * table["appeared_total"], True


# Transcribed from each JIC report. "registered" is table 8.1 (for 2022 the report's stage summary gives appeared
# counts directly); "appeared_total" is the report's count of candidates who sat both papers; "allotted" is table 8.3
# (candidates, by their own category, allotted an IIT seat). Category order follows CATEGORIES.
REPORT_TABLES = {
    2021: {"registered": (40951, 21981, 51788, 25019, 11454), "appeared_total": 141699,
           "allotted": (6576, 1672, 4394, 2433, 1221), "source": "JIC 2021 report, section 1(9) and tables 8.1, 8.3"},
    2022: {"appeared": (38180, 24104, 56538, 25287, 11429), "appeared_total": 155538,
           "allotted": (6477, 1883, 4538, 2494, 1243), "source": "JIC 2022 report, 'Summary of Number of Candidates' (PwD and non-PwD summed)"},
    2023: {"registered": (41680, 31622, 70685, 31044, 14456), "appeared_total": 180372,
           "allotted": (6617, 2043, 4747, 2616, 1317), "source": "JIC 2023 report, section 3(i) and tables 8.1, 8.3"},
    2024: {"registered": None, "appeared_total": 180200, "allotted": None,
           "source": "JIC 2024 report, section 3; its category tables are images, so category counts are imputed from 2023 and 2025 shares"},
    2025: {"registered": (41202, 29939, 70148, 31130, 14804), "appeared_total": 180422,
           "allotted": (7118, 2008, 4965, 2730, 1367), "source": "JIC 2025 report, section 3 and tables 8.1, 8.3 (columns reordered)"},
}


def appeared_by_category(year: int):
    """Candidates who sat both papers, by category, and whether the split had to be imputed."""
    table = REPORT_TABLES[year]
    if table.get("appeared") is not None:
        return np.array(table["appeared"], float), False
    if table.get("registered") is not None:
        registered = np.array(table["registered"], float)
        return registered * table["appeared_total"] / registered.sum(), False
    shares = np.mean([np.array(REPORT_TABLES[y]["registered"], float) / sum(REPORT_TABLES[y]["registered"]) for y in (2023, 2025)], axis=0)
    return shares * table["appeared_total"], True
