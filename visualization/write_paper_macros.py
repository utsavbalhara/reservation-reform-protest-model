import json

import numpy as np

from protest_simulation import BASELINE_ABRUPT_INCOME_ONLY_SWITCH

from .figure_style import REPOSITORY_ROOT, RESULTS_FOLDER
from .render_paper_figures import READABLE_PARAMETER_NAMES, SENSITIVITY_BASELINE_RUNS, reference_baseline_peak_lakh
from .write_latex_result_macros import signed_percent

GENERATED_FOLDER = REPOSITORY_ROOT / "paper" / "generated"
READABLE_SCENARIO_NAMES = {
    "grandfathering": "L1", "seat_expansion": "L2", "hybrid_caste_subquotas": "L3", "sub_classification": "L4",
    "consensus_commission": "L5", "compensation": "L6", "credible_guarantees": "L7",
}


def load_json(file_name):
    return json.loads((RESULTS_FOLDER / file_name).read_text())


def macro(name, value):
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def people_text(lakh):
    if lakh >= 100:
        return f"{lakh / 100:.1f}~crore"
    if lakh < 1:
        return f"{int(round(lakh * 1e5, -3)):,}".replace(",", "{,}")
    return f"{lakh:.1f}~lakh"


def peak_at(curve, threshold, field=None):
    for point in curve:
        if abs(point["mean_participation_threshold"] - threshold) < 1e-6:
            return point["median_peak_lakh"] if field is None else point[field]["median_peak_lakh"]
    raise KeyError(threshold)


def history_macros():
    history = load_json("model_development_history.json")
    first, second, third, final = (history[key] for key in (
        "iteration_1_unbounded_contagion", "iteration_2_narrow_thresholds", "iteration_3_reweighting", "iteration_4_final"))
    macros = [macro("HistOnePeak", people_text(float(np.median([point["median_peak_lakh"] for point in first["threshold_sweep"]]))))]
    for group, share in first["bandh_day_participation_percent_without_contagion"].items():
        macros.append(macro(f"HistOneBandh{group}", f"{share:.1f}\\%"))
    for group, share in final["bandh_day_participation_percent_without_contagion"].items():
        macros.append(macro(f"HistFinalBandh{group}", f"{share:.2f}\\%"))

    scenarios = second["scenarios"]
    baseline = scenarios["baseline"]
    macros += [
        macro("HistTwoBaseline", people_text(baseline["median_peak_lakh"])),
        macro("HistTwoBaselineRange", f"{baseline['peak_lakh_10th_percentile']}--{baseline['peak_lakh_90th_percentile']}"),
        macro("HistTwoSymbolicOnly", people_text(scenarios["symbolic_only_validation"]["median_peak_lakh"])),
        macro("HistTwoSymbolicShare", f"{scenarios['symbolic_only_validation']['median_peak_lakh'] / baseline['median_peak_lakh'] * 100:.1f}\\%"),
    ]
    for key, name in (("grandfathering", "Grandfathering"), ("hybrid_caste_subquotas", "Hybrid"), ("consensus_commission", "Consensus"),
                      ("compensation", "Compensation"), ("managed_transition", "Managed")):
        macros.append(macro(f"HistTwoChange{name}", signed_percent((scenarios[key]["median_peak_lakh"] / baseline["median_peak_lakh"] - 1) * 100)))
    early_curve = second["threshold_response_curve"]
    final_curve = load_json("threshold_response_curve.json")["response_curve"]
    macros += [
        macro("HistTwoFallFactor", f"{peak_at(early_curve, 4.9) / peak_at(early_curve, 5.0):.1f}"),
        macro("HistTwoPeakAtFourNine", people_text(peak_at(early_curve, 4.9))),
        macro("HistTwoPeakAtFive", people_text(peak_at(early_curve, 5.0))),
        macro("HistFinalFallFactor", f"{peak_at(final_curve, 6.5, 'baseline') / peak_at(final_curve, 6.75, 'baseline'):.1f}"),
    ]
    final_scenarios = final["scenarios"]
    macros.append(macro("FinalSymbolicShare", f"{final_scenarios['symbolic_only_validation']['median_peak_lakh'] / final_scenarios['baseline']['median_peak_lakh'] * 100:.0f}\\%"))

    rows = [f"{attempt['symbolic_threat_weight']} & {attempt['material_loss_weight']} & {attempt['mean_participation_threshold']} & "
            f"{attempt['baseline_median_peak_lakh']} & {attempt['symbolic_only_median_peak_lakh']} & "
            f"{attempt['symbolic_only_share_of_baseline'] * 100:.1f}\\% \\\\" for attempt in third["attempts"]]
    return macros, rows


def sensitivity_macros():
    sensitivity = load_json("sensitivity_analysis.json")
    perturbations = sensitivity["perturbations"]
    correlations = [row["rank_correlation_with_reference"] for row in perturbations]
    by_parameter = {}
    for row in perturbations:
        by_parameter.setdefault(row["parameter"], {})[row["direction"]] = row
    widest = max(by_parameter.items(), key=lambda item: abs(np.log(item[1]["higher"]["baseline_median_peak_lakh"] / item[1]["lower"]["baseline_median_peak_lakh"])))
    macros = [
        macro("SensReferencePeak", people_text(reference_baseline_peak_lakh(SENSITIVITY_BASELINE_RUNS))),
        macro("SensPerturbationCount", str(len(perturbations))),
        macro("SensRankCorrMin", f"{min(correlations):.2f}"),
        macro("SensRankCorrMedian", f"{float(np.median(correlations)):.2f}"),
        macro("SensHybridStrongestCount", str(sum(row["strongest_single_lever"] == "hybrid_caste_subquotas" for row in perturbations))),
        macro("SensCompensationWeakestCount", str(sum(row["weakest_single_lever"] == "compensation" for row in perturbations))),
        macro("SensWidestParameter", READABLE_PARAMETER_NAMES[widest[0]].split(" (")[0].lower()),
        macro("SensWidestLow", people_text(widest[1]["lower"]["baseline_median_peak_lakh"])),
        macro("SensWidestHigh", people_text(widest[1]["higher"]["baseline_median_peak_lakh"])),
    ]
    weakest_counts = {}
    for row in perturbations:
        weakest_counts[row["weakest_single_lever"]] = weakest_counts.get(row["weakest_single_lever"], 0) + 1
    macros.append(macro("SensWeakestSummary", ", ".join(f"{READABLE_SCENARIO_NAMES[key]} in {count}" for key, count in sorted(weakest_counts.items(), key=lambda item: -item[1]))))

    table_rows = []
    for parameter, directions in by_parameter.items():
        lower, higher = directions["lower"], directions["higher"]
        table_rows.append(
            f"{READABLE_PARAMETER_NAMES[parameter]} & {lower['baseline_median_peak_lakh']} & {higher['baseline_median_peak_lakh']} & "
            f"{lower['rank_correlation_with_reference']:.2f} & {higher['rank_correlation_with_reference']:.2f} & "
            f"{READABLE_SCENARIO_NAMES[lower['strongest_single_lever']]}/{READABLE_SCENARIO_NAMES[higher['strongest_single_lever']]} & "
            f"{READABLE_SCENARIO_NAMES[lower['weakest_single_lever']]}/{READABLE_SCENARIO_NAMES[higher['weakest_single_lever']]} \\\\")
    convergence_rows = [
        f"{row['agent_count']:,}".replace(",", "{,}") + f" & {people_per_agent_text(row['agent_count'])} & {row['baseline']['median_peak_lakh']} & "
        f"{row['baseline']['peak_lakh_10th_percentile']}--{row['baseline']['peak_lakh_90th_percentile']} & "
        f"{row['hybrid_caste_subquotas']['median_peak_lakh']} \\\\"
        for row in sensitivity["agent_count_convergence"]
    ]
    return macros, table_rows, convergence_rows


def people_per_agent_text(agent_count):
    return f"{int(round(146e7 / agent_count, -2)):,}".replace(",", "{,}")


def main():
    GENERATED_FOLDER.mkdir(parents=True, exist_ok=True)
    history, iteration_three_rows = history_macros()
    sensitivity, sensitivity_rows, convergence_rows = sensitivity_macros()
    parameter_macros = [
        macro("ParamLossAversion", BASELINE_ABRUPT_INCOME_ONLY_SWITCH.loss_aversion),
        macro("ParamMeanThreshold", BASELINE_ABRUPT_INCOME_ONLY_SWITCH.mean_participation_threshold),
    ]
    (GENERATED_FOLDER / "paper_macros.tex").write_text("\n".join(history + sensitivity + parameter_macros) + "\n")
    (GENERATED_FOLDER / "iteration_three_rows.tex").write_text("\n".join(iteration_three_rows) + "\n")
    (GENERATED_FOLDER / "sensitivity_rows.tex").write_text("\n".join(sensitivity_rows) + "\n")
    (GENERATED_FOLDER / "convergence_rows.tex").write_text("\n".join(convergence_rows) + "\n")
    print(f"Wrote paper macros and tables to {GENERATED_FOLDER}")


if __name__ == "__main__":
    main()
