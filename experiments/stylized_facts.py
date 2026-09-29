"""Does the grounded model reproduce the stylized facts of Section 2 (S1--S5)?

Most checks read results that other experiments already produce; this script adds the ones that need daily detail from
the grounded baseline: whether the peak falls on a bandh day (S2), and how turnout on bandh days compares with ordinary
days. It writes results/stylized_facts.json, which collects every check with the file its evidence comes from.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from protest_simulation.monte_carlo import CAMPAIGN_SEED_OFFSET, paired_worlds
from protest_simulation.parallel import run_campaigns
from protest_simulation.specifications import get_specification

RESULTS_FOLDER = Path(__file__).resolve().parent.parent / "results"


def load(name):
    path = RESULTS_FOLDER / name
    return json.loads(path.read_text()) if path.exists() else None


def main():
    parser = argparse.ArgumentParser(description="Check the grounded model against the stylized facts.")
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--agents", type=int, default=120_000)
    arguments = parser.parse_args()
    specification = get_specification("grounded")
    population = specification.build_population(arguments.agents)
    worlds = paired_worlds((), arguments.runs, base_parameters=specification.base_parameters, world_sampler=specification.world_sampler)
    outcomes = run_campaigns(population, [(world, CAMPAIGN_SEED_OFFSET + index) for index, world in enumerate(worlds)])
    peak_on_bandh, bandh_to_ordinary, no_bandh, peak_after_bandh = [], [], [], []
    for outcome in outcomes:
        daily = np.asarray(outcome.daily_protesters)
        bandh_days = list(outcome.bandh_days)
        peak = int(np.argmax(daily))
        peak_on_bandh.append(peak in bandh_days)
        no_bandh.append(not bandh_days)
        peak_after_bandh.append(bool(bandh_days) and peak not in bandh_days and peak > min(bandh_days))
        if bandh_days:
            ordinary = [day for day in range(len(daily)) if day not in bandh_days and daily[day] > 0]
            if ordinary:
                bandh_to_ordinary.append(float(np.mean(daily[bandh_days]) / np.mean(daily[ordinary])))
    comparison = load("intervention_comparison_grounded_central.json")
    assumptions = load("grounded_assumptions.json")
    channels = load("grounded_channels.json")
    calibration = load("episode_calibration.json")
    facts = {
        "S1_symbolic_threat_alone_mobilizes": {
            "symbolic_only_median_peak_lakh": comparison["scenarios"]["symbolic_only_validation"]["summary"]["median_peak_lakh"] if comparison else None,
            "calibration_reproduces_2018_event_count": bool(calibration and calibration.get("kept")),
            "source": "intervention_comparison_grounded_central.json; episode_calibration.json"},
        "S2_mobilization_concentrates_on_bandh_days": {
            "share_of_runs_peak_on_a_bandh_day": round(float(np.mean(peak_on_bandh)), 3),
            "share_of_runs_without_a_bandh": round(float(np.mean(no_bandh)), 3),
            "share_of_runs_peak_after_first_bandh": round(float(np.mean(peak_after_bandh)), 3),
            "median_ratio_bandh_day_to_ordinary_day_turnout": round(float(np.median(bandh_to_ordinary)), 1) if bandh_to_ordinary else None,
            "source": "this script"},
        "S3_party_backing_is_one_amplifier_among_several": None,
        "S4_groups_split_when_a_reform_creates_winners_inside_them": None,
        "S5_deaths_and_concession": {
            "share_of_runs_conceded": comparison and round(float(np.mean([day is not None for day in comparison["scenarios"]["baseline"]["runs"]["conceded_on_day"]])), 3),
            "loeo_deaths_predicted_for_peaceful_bandhs": "see loeo_scores.json",
            "source": "intervention_comparison_grounded_central.json; loeo_scores.json"},
    }
    if assumptions:
        cells = {(cell["shock_ratio"], cell["party_backing"]): cell["baseline_median_peak_lakh"] for cell in assumptions["cells"]}
        facts["S3_party_backing_is_one_amplifier_among_several"] = {
            "peak_ratio_backing_0_6_to_0_2_at_R_1": round(cells[(1.0, 0.6)] / cells[(1.0, 0.2)], 3),
            "peak_ratio_R_1_5_to_R_0_5_at_backing_0_6": round(cells[(1.5, 0.6)] / cells[(0.5, 0.6)], 1),
            "source": "grounded_assumptions.json"}
    if channels:
        baseline = channels["who_and_where"]["baseline"]["sc_st_share_ever_protesting_by_tier"]
        split = channels["who_and_where"]["sub_classification"]["sc_st_share_ever_protesting_by_tier"]
        facts["S4_groups_split_when_a_reform_creates_winners_inside_them"] = {
            "baseline": baseline, "with_sub_classification": split,
            "most_deprived_change_percent": round((split["most_deprived_share_protesting"] / baseline["most_deprived_share_protesting"] - 1) * 100, 1),
            "better_off_change_percent": round((split["better_off_share_protesting"] / baseline["better_off_share_protesting"] - 1) * 100, 1),
            "source": "grounded_channels.json"}
    RESULTS_FOLDER.mkdir(exist_ok=True)
    path = RESULTS_FOLDER / "stylized_facts.json"
    path.write_text(json.dumps(facts, indent=1))
    print(json.dumps(facts, indent=1))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
