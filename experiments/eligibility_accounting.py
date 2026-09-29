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

The shares of each group above the income line (General 30%, OBC 17%, SC 19% and 5% by tier, ST 16% and 4%) are also
assumptions, ordered by survey evidence on the caste gradient of income but not estimated. The joint-prior section below
multiplies each group's above-line share by an independent factor drawn from 0.6 to 1.4 and draws the two legal shares
from their priors, and reports the resulting interval for the shares losing and gaining eligibility.
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
INCOME_SHARE_FACTOR_PRIOR = (0.6, 1.4)
CREAMY_PRIOR = (0.3, 0.7)
ASSET_PRIOR = (0.05, 0.25)
JOINT_DRAWS = 20_000
GAIN = ("General_below_asset_excluded",)


def shares(population):
    segments = eligibility_segment_of_each_agent(population)
    return {name: float(np.mean(segments == index)) for index, name in enumerate(SEGMENT_NAMES)}


def joint_prior():
    """Shares losing and gaining eligibility with priors on the income shares as well as the two legal shares."""
    from protest_simulation.synthetic_population import (ABOVE_INCOME_LINE_SHARE_IN_BETTER_OFF_TIER, ABOVE_INCOME_LINE_SHARE_IN_MOST_DEPRIVED_TIER,
                                                         ABOVE_INCOME_LINE_SHARE_IN_OTHER_GROUPS, GENERAL, OBC, SC, ST)
    population = build_synthetic_india_by_district(AGENTS, np.random.default_rng(7))
    group, deprived = population.social_group, population.is_most_deprived_tier
    share = {(g, d): float(np.mean((group == g) & (deprived == d))) for g in (SC, ST) for d in (False, True)}
    obc, general = float(np.mean(group == OBC)), float(np.mean(group == GENERAL))
    random_generator = np.random.default_rng(20260930)
    low, high = INCOME_SHARE_FACTOR_PRIOR
    lose, gain = [], []
    for _ in range(JOINT_DRAWS):
        factor = {g: random_generator.uniform(low, high) for g in (SC, ST, OBC, GENERAL)}
        creamy, asset = random_generator.uniform(*CREAMY_PRIOR), random_generator.uniform(*ASSET_PRIOR)
        sc_st = sum(share[(g, d)] * min(1.0, factor[g] * (ABOVE_INCOME_LINE_SHARE_IN_MOST_DEPRIVED_TIER if d else ABOVE_INCOME_LINE_SHARE_IN_BETTER_OFF_TIER)[g])
                    for g in (SC, ST) for d in (False, True))
        obc_above = obc * min(1.0, factor[OBC] * ABOVE_INCOME_LINE_SHARE_IN_OTHER_GROUPS[OBC])
        general_below = general * (1 - min(1.0, factor[GENERAL] * ABOVE_INCOME_LINE_SHARE_IN_OTHER_GROUPS[GENERAL]))
        lose.append(sc_st + obc_above * (1 - creamy))
        gain.append(general_below * asset)
    summary = lambda values: [round(float(np.percentile(values, q)), 5) for q in (5, 50, 95)]
    return {"income_share_factor_prior": list(INCOME_SHARE_FACTOR_PRIOR), "creamy_prior": list(CREAMY_PRIOR), "asset_prior": list(ASSET_PRIOR),
            "draws": JOINT_DRAWS, "share_losing_eligibility_p05_p50_p95": summary(lose), "share_gaining_eligibility_p05_p50_p95": summary(gain)}


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
    output["joint_prior"] = joint_prior()
    print("joint prior:", output["joint_prior"])
    RESULTS_FOLDER.mkdir(exist_ok=True)
    path = RESULTS_FOLDER / "eligibility_accounting.json"
    path.write_text(json.dumps(output, indent=1))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
