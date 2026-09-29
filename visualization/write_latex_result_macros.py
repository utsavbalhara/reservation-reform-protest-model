import json

from .figure_style import RESULTS_FOLDER, REPOSITORY_ROOT

GENERATED_FOLDER = REPOSITORY_ROOT / "report" / "generated"
# Result file suffix -> macro prefix. The grounded specification's comparison is included when it has been run.
REGIME_MACRO_NAMES = {"central": "Central", "high-mobilization": "High", "grounded_central": "Grounded"}
SCENARIO_TABLE_ORDER = ["baseline", "symbolic_only_validation", "grandfathering", "seat_expansion", "hybrid_caste_subquotas",
                        "sub_classification", "consensus_commission", "compensation", "credible_guarantees",
                        "internet_shutdown", "heavy_policing", "managed_transition", "hybrid_package"]
TABLE_SECTION_BREAKS_AFTER = ("symbolic_only_validation", "credible_guarantees", "heavy_policing")


def macro_name_for(scenario_key: str) -> str:
    return "".join(part.capitalize() for part in scenario_key.split("_"))


def lakh_text(lakh: float) -> str:
    if lakh >= 100:
        return f"{lakh / 100:.2f}~crore"
    return f"{lakh:.1f}~lakh"


def signed_percent(change: float) -> str:
    rounded = round(change)
    if rounded == 0:
        return "0\\%"
    return f"{'+' if rounded > 0 else '$-$'}{abs(rounded)}\\%"


def signed_number(value: float) -> str:
    rounded = round(value)
    if rounded == 0:
        return "0"
    return f"{'+' if rounded > 0 else '$-$'}{abs(rounded)}"


def interval_text(interval) -> str:
    return f"{signed_number(interval[0])} to {signed_number(interval[1])}\\%"


def macro(name: str, value) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def scenario_macros(regime_name: str, key: str, scenario: dict) -> list:
    """Every percentage change is a paired statistic: the median over runs of scenario / baseline,
    where both used the same world and campaign seeds, with a percentile bootstrap 95% interval."""
    summary, paired, name = scenario["summary"], scenario["paired_effects_vs_baseline"], macro_name_for(key)
    return [
        macro(f"Peak{regime_name}{name}", lakh_text(summary["median_peak_lakh"])),
        macro(f"PeakRange{regime_name}{name}", f"{summary['peak_lakh_10th_percentile']}--{summary['peak_lakh_90th_percentile']}"),
        macro(f"PeakChange{regime_name}{name}", signed_percent(paired["peak"]["change_percent"])),
        macro(f"PeakChangeCI{regime_name}{name}", interval_text(paired["peak"]["change_percent_ci95"])),
        macro(f"PeakShareHigher{regime_name}{name}", f"{paired['peak']['share_of_runs_higher_than_baseline'] * 100:.0f}\\%"),
        macro(f"Cumulative{regime_name}{name}", f"{summary['median_cumulative_crore']:.2f}~crore"),
        macro(f"CumulativeChange{regime_name}{name}", signed_percent(paired["cumulative"]["change_percent"])),
        macro(f"CumulativeChangeCI{regime_name}{name}", interval_text(paired["cumulative"]["change_percent_ci95"])),
        macro(f"Deaths{regime_name}{name}", f"{summary['median_deaths']:g}"),
        macro(f"DeathRatio{regime_name}{name}", f"{paired['deaths']['median_paired_ratio']:.1f}"),
        macro(f"DeathChangeCI{regime_name}{name}", interval_text(paired["deaths"]["change_percent_ci95"])),
    ]


def result_macros() -> list:
    macros = []
    for regime, regime_name in REGIME_MACRO_NAMES.items():
        path = RESULTS_FOLDER / f"intervention_comparison_{regime}.json"
        if not path.exists():
            continue
        comparison = json.loads(path.read_text())
        for key, scenario in comparison["scenarios"].items():
            macros += scenario_macros(regime_name, key, scenario)
        macros.append(macro(f"RunCount{regime_name}", comparison["run_count"]))
    robustness = json.loads((RESULTS_FOLDER / "robustness_checks.json").read_text())
    retained_names = {40: "Forty", 60: "Sixty", 80: "Eighty"}
    for check in robustness["hybrid_sensitivity"]:
        retained = int(round(check["symbolic_threat_retained"] * 100))
        if retained in retained_names:
            macros.append(macro(f"HybridRetained{retained_names[retained]}", lakh_text(check["median_peak_lakh"])))
    for check in robustness["managed_transition_leave_one_out"]:
        suffix = "Full" if check["left_out"] is None else "Without" + macro_name_for(check["left_out"])
        macros.append(macro(f"ManagedTransition{suffix}", lakh_text(check["median_peak_lakh"])))
    return macros


def scenario_table_rows(regime: str) -> list:
    scenarios = json.loads((RESULTS_FOLDER / f"intervention_comparison_{regime}.json").read_text())["scenarios"]
    rows = []
    for key in SCENARIO_TABLE_ORDER:
        scenario, summary = scenarios[key], scenarios[key]["summary"]
        paired = scenario["paired_effects_vs_baseline"]["peak"]
        change = "--" if key == "baseline" else signed_percent(paired["change_percent"])
        interval = "--" if key == "baseline" else interval_text(paired["change_percent_ci95"])
        label = scenario["label"].replace("₹", "\\rupee{}").replace("&", "\\&")
        rows.append(
            f"{scenario['code']} & {label} & {summary['median_peak_lakh']:.1f} & "
            f"{summary['peak_lakh_10th_percentile']}--{summary['peak_lakh_90th_percentile']} & {change} & {interval} & "
            f"{summary['median_cumulative_crore']:.2f} & {summary['median_deaths']:g} \\\\"
        )
        if key in TABLE_SECTION_BREAKS_AFTER:
            rows.append("\\midrule")
    return rows


def main():
    GENERATED_FOLDER.mkdir(parents=True, exist_ok=True)
    (GENERATED_FOLDER / "result_macros.tex").write_text("\n".join(result_macros()) + "\n")
    for regime in REGIME_MACRO_NAMES:
        if not (RESULTS_FOLDER / f"intervention_comparison_{regime}.json").exists():
            continue
        (GENERATED_FOLDER / f"scenario_table_{regime}.tex").write_text("\n".join(scenario_table_rows(regime)) + "\n")
    print(f"Wrote LaTeX macros and tables to {GENERATED_FOLDER}")


if __name__ == "__main__":
    main()
