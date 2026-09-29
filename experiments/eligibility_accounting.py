"""Who changes eligibility status under a single ₹8 lakh income test, under the actual current rules?

Current rules that the original accounting ignored:
  - The OBC creamy-layer test excludes salary and agricultural income (DoPT O.M. of 8 September 1993 and later
    revisions), so many OBC families above ₹8 lakh on gross income are non-creamy and eligible today.
  - EWS eligibility also fails families owning 5 acres or more of agricultural land, a residential flat of 1,000 sq ft
    or more, or residential plots above set sizes (DoPT O.M. of 19 January 2019), so some below-line General families
    are ineligible today.
The reform is taken to apply one test on gross family income, with no category and no asset test. Status changes are:
  lose: SC and ST above the line; OBC above the line but non-creamy today;
  gain: below-line General families who fail the EWS asset tests today.
The two shares are unknown (no public data split OBC incomes by source or General families by assets), so the table is
given over a grid of values; the central values are the grounded specification's (creamy share 0.5, asset exclusion 0.15).
"""
import json
from pathlib import Path

import numpy as np

from protest_simulation.geography import build_synthetic_india_by_district
from protest_simulation.protest_campaign import SEGMENT_NAMES, eligibility_segment_of_each_agent
from protest_simulation.synthetic_population import build_synthetic_india

RESULTS_FOLDER = Path(__file__).resolve().parent.parent / "results"
AGENTS = 600_000
CREAMY_SHARES = (0.3, 0.5, 0.7, 1.0)
ASSET_EXCLUSION_SHARES = (0.0, 0.05, 0.15, 0.25)
LOSE = ("SC_above", "ST_above", "OBC_above_ncl")
GAIN = ("General_below_asset_excluded",)


def shares(population):
    segments = eligibility_segment_of_each_agent(population)
    return {name: float(np.mean(segments == index)) for index, name in enumerate(SEGMENT_NAMES)}


def main():
    builders = {"stylized_national_shares": build_synthetic_india, "district_census_nfhs_shares": build_synthetic_india_by_district}
    output = {"agents": AGENTS, "lose_segments": LOSE, "gain_segments": GAIN, "populations": {}}
    for label, builder in builders.items():
        grid = []
        for creamy in CREAMY_SHARES:
            for asset in ASSET_EXCLUSION_SHARES:
                population = builder(AGENTS, np.random.default_rng(7), obc_creamy_share_of_above_line=creamy,
                                     ews_asset_exclusion_share=asset)
                segment_shares = shares(population)
                grid.append({"obc_creamy_share_of_above_line": creamy, "ews_asset_exclusion_share": asset,
                             "segment_shares": {k: round(v, 5) for k, v in segment_shares.items()},
                             "share_losing_eligibility": round(sum(segment_shares[s] for s in LOSE), 5),
                             "share_gaining_eligibility": round(sum(segment_shares[s] for s in GAIN), 5),
                             "share_sc_st_above_line": round(segment_shares["SC_above"] + segment_shares["ST_above"], 5)})
        central = next(row for row in grid if row["obc_creamy_share_of_above_line"] == 0.5 and row["ews_asset_exclusion_share"] == 0.15)
        prior = [row for row in grid if row["obc_creamy_share_of_above_line"] in (0.3, 0.7) and row["ews_asset_exclusion_share"] in (0.05, 0.25)]
        output["populations"][label] = {
            "grid": grid, "central": central,
            "range_over_priors": {measure: [min(row[measure] for row in prior), max(row[measure] for row in prior)]
                                  for measure in ("share_losing_eligibility", "share_gaining_eligibility")},
            "original_accounting": next(row for row in grid if row["obc_creamy_share_of_above_line"] == 1.0
                                        and row["ews_asset_exclusion_share"] == 0.0)["share_losing_eligibility"]}
        print(f"{label}: lose {central['share_losing_eligibility']:.3%} (range {output['populations'][label]['range_over_priors']['share_losing_eligibility']}), "
              f"gain {central['share_gaining_eligibility']:.3%}; original accounting {output['populations'][label]['original_accounting']:.3%}")
    RESULTS_FOLDER.mkdir(exist_ok=True)
    path = RESULTS_FOLDER / "eligibility_accounting.json"
    path.write_text(json.dumps(output, indent=1))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
