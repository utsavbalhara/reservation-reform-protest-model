"""Extract the JEE (Advanced) rank lists published in the Joint Implementation Committee (JIC) reports.

Each JIC report prints the Common Rank List (CRL) and the GEN-EWS, OBC-NCL, SC and ST category rank lists, each
sorted by roll number. Joining them on roll number gives, for every candidate in the CRL, their category. That is the
category composition at every merit rank, which is what a merged-pool allocation needs.

Output (one file per year, no roll numbers, no names): data/derived/jee_advanced_ranks_<year>.csv with columns
crl_rank (blank if the candidate qualified only on relaxed criteria), category (GEN, GEN-EWS, OBC-NCL, SC, ST) and
category_rank (blank for GEN). GEN means a CRL candidate who appears on no category list: General-category candidates
who did not claim EWS, and OBC candidates in the creamy layer.

Source: https://jeeadv.ac.in/reports/<year>.pdf (downloaded to data/raw/, which is not committed).
"""
import csv
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
RAW_FOLDER = REPOSITORY_ROOT / "data" / "raw"
DERIVED_FOLDER = REPOSITORY_ROOT / "data" / "derived"
LISTS = ("CRL", "GEN-EWS", "OBC-NCL", "SC", "ST")
# Every page of a rank list repeats a column-header row such as "AdvRollNo  SC  AdvRollNo  SC ..." (2023, 2025),
# "Adv Roll No  EWS  Adv Roll No  EWS ..." (2024) or "Roll number  CRL  Roll number  CRL ..." (2020-2022). The label
# after the roll-number column names the list. The 2019 report has no such rows; its lists are identified by the page
# ranges in its table of contents (pages Ch-10/1 onward).
COLUMN_HEADER = re.compile(r"(?:Adv\s*Roll\s*No\.?|AdvRollNo|Roll\s*[Nn]umber|Roll\s*No\.?)\s+([A-Za-z][A-Za-z_\-]*)")
LABEL_ALIASES = {"CRL": "CRL", "GEN-EWS": "GEN-EWS", "EWS": "GEN-EWS", "OBC-NCL": "OBC-NCL", "OBC": "OBC-NCL", "SC": "SC", "ST": "ST"}
PAIR = re.compile(r"(?<!\d)(\d{7}|\d{9})\s+(\d{1,6})(?!\d)")
PAGE_FOOTER_2019 = re.compile(r"^\s*Ch-10/(\d+)\s*$")
PAGE_RANGES_2019 = {"CRL": (1, 146), "GEN-EWS": (147, 175), "OBC-NCL": (176, 234), "SC": (235, 256), "ST": (257, 263)}
STATED_LABELS = {"CRL": "Common Rank List (CRL)", "GEN-EWS": "GEN-EWS Rank List", "OBC-NCL": "OBC-NCL Rank List",
                 "SC": "SC Rank List", "ST": "ST Rank List"}


def report_text(year: int) -> str:
    RAW_FOLDER.mkdir(parents=True, exist_ok=True)
    pdf, text = RAW_FOLDER / f"jee_advanced_jic_report_{year}.pdf", RAW_FOLDER / f"jee_advanced_jic_report_{year}.txt"
    if not text.exists():
        if not pdf.exists():
            urllib.request.urlretrieve(f"https://jeeadv.ac.in/reports/{year}.pdf", pdf)
        subprocess.run(["pdftotext", "-layout", str(pdf), str(text)], check=True)
    return text.read_text(errors="replace")


def parse_rank_lists(text: str) -> dict:
    lists = {name: {} for name in LISTS}
    current = None
    for line in text.splitlines():
        headers = COLUMN_HEADER.findall(line)
        if len(headers) >= 2:
            current = LABEL_ALIASES.get(headers[0].replace("_", "-").upper())
            continue
        if current is None:
            continue
        for roll, rank in PAIR.findall(line):
            lists[current][roll] = int(rank)
    return lists


def parse_rank_lists_by_page_range(text: str) -> dict:
    """2019 layout: pairs on a page belong to the list whose page range contains that page's footer number."""
    lists = {name: {} for name in LISTS}
    buffered = []
    for line in text.splitlines():
        footer = PAGE_FOOTER_2019.match(line)
        if footer:
            page = int(footer.group(1))
            for name, (first, last) in PAGE_RANGES_2019.items():
                if first <= page <= last:
                    for roll, rank in buffered:
                        lists[name][roll] = int(rank)
            buffered = []
            continue
        if "Ch-" in line or "Page" in line:
            buffered = []
        buffered.extend(PAIR.findall(line))
    return lists


def stated_list_sizes(text: str) -> dict:
    """The report's own statement of each list's size, from the summary near the front."""
    sizes = {}
    front = text[:300_000]
    for key, label in STATED_LABELS.items():
        for match in re.finditer(re.escape(label) + r"\s*:", front):
            segment = front[match.end(): match.end() + 400]
            number = re.search(r"(?:relaxation|[Cc]riteria)\s*:?\s*([\d,]+)", segment)
            if number:
                sizes[key] = int(number.group(1).replace(",", ""))
                break
    return sizes


def joined_rows(lists: dict) -> list:
    category_of = {}
    for name in LISTS[1:]:
        for roll, rank in lists[name].items():
            category_of[roll] = (name, rank)
    rows = []
    for roll, crl_rank in lists["CRL"].items():
        category, category_rank = category_of.get(roll, ("GEN", None))
        rows.append((crl_rank, category, category_rank))
    for roll, (category, category_rank) in category_of.items():
        if roll not in lists["CRL"]:
            rows.append((None, category, category_rank))
    rows.sort(key=lambda row: (row[0] is None, row[0] or 0, row[1], row[2] or 0))
    return rows


def extract(year: int) -> dict:
    text = report_text(year)
    lists = parse_rank_lists_by_page_range(text) if year == 2019 else parse_rank_lists(text)
    stated = stated_list_sizes(text)
    rows = joined_rows(lists)
    DERIVED_FOLDER.mkdir(parents=True, exist_ok=True)
    with open(DERIVED_FOLDER / f"jee_advanced_ranks_{year}.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["crl_rank", "category", "category_rank"])
        for crl_rank, category, category_rank in rows:
            writer.writerow(["" if crl_rank is None else crl_rank, category, "" if category_rank is None else category_rank])
    return {"year": year, "parsed": {name: len(values) for name, values in lists.items()}, "stated": stated}


if __name__ == "__main__":
    for year in [int(argument) for argument in sys.argv[1:]] or [2019, 2020, 2021, 2022, 2023, 2024, 2025]:
        print(extract(year))
