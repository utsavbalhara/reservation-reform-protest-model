"""NEET (UG) inputs for a second allocation domain, from NTA's official result press releases.

2023: 'Press release for final NTA score for NEET (UG) 2023' (13 June 2023): table 4 (category-wise registered, appeared,
qualified), table 7 (category-wise qualifying marks ranges and counts) and list 9(a) (top 50 candidates).
NEET categories: Gen (General, including OBC creamy layer), GEN-EWS, OBC (non-creamy layer), SC, ST.

The qualifying cut-offs are percentile scores over all candidates: 137 marks is the 50th percentile (UR/EWS) and 107
marks the 40th (OBC/SC/ST). Candidates at or above 137 marks: 1,014,372 in total. For OBC, SC and ST, the number between
107 and 136 is reported, so those three categories give two quantiles each; General and EWS give one each, plus their
combined count between 107 and 136, which follows from the 40th percentile covering 60% of all candidates.
Persons-with-disability relaxations are small (a few hundred candidates) and are netted out as reported.
"""
import numpy as np

CATEGORIES = ("GEN", "GEN-EWS", "OBC-NCL", "SC", "ST")
NEET_2023 = {
    "appeared": np.array([592110, 152197, 873173, 294995, 126121], float),
    "qualified": np.array([312405, 98322, 525194, 153674, 56381], float),
    "between_107_and_136": {"OBC-NCL": 88592 + 179, "SC": 29918 + 50, "ST": 12437 + 23},
    "at_or_above_137_total": 1014372,
    "appeared_total": 2038596,
    "top_50": {"GEN": 37, "GEN-EWS": 0, "OBC-NCL": 11, "SC": 2, "ST": 0},
    "mbbs_seats": 107948,
    "source": "NTA press release on NEET (UG) 2023 results, tables 4 and 7 and list 9(a); MBBS seat count is the 2023-24 total reported by the National Medical Commission",
}


def quantile_targets(data=NEET_2023):
    """Shares of each category at or above 137 marks and at or above 107 marks (None where not identified)."""
    appeared = data["appeared"]
    above_137 = data["qualified"].copy()
    for category, count in data["between_107_and_136"].items():
        above_137[CATEGORIES.index(category)] -= count
    above_107 = {category: data["qualified"][CATEGORIES.index(category)] / appeared[CATEGORIES.index(category)]
                 for category in data["between_107_and_136"]}
    general_and_ews_107_to_136 = 0.60 * data["appeared_total"] - data["at_or_above_137_total"] - sum(data["between_107_and_136"].values())
    return {"share_at_or_above_137": above_137 / appeared, "share_at_or_above_107": above_107,
            "general_and_ews_between_107_and_136": general_and_ews_107_to_136}
