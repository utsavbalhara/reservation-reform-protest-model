import json
from pathlib import Path

from .figure_style import RESULTS_FOLDER, REPOSITORY_ROOT

GENERATED_FOLDER = REPOSITORY_ROOT / "report" / "generated"
REGIME_MACRO_NAMES = {"central": "Central", "high-mobilization": "High"}


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


def result_macros() -> list:
    macros = []
    for regime, regime_name in REGIME_MACRO_NAMES.items():
        scenarios = json.loads((RESULTS_FOLDER / f"intervention_comparison_{regime}.json").read_text())["scenarios"]
        baseline = scenarios["baseline"]["summary"]
        for key, scenario in scenarios.items():
            summary = scenario["summary"]
            name = macro_name_for(key)
            peak_change = (summary["median_peak_lakh"] / baseline["median_peak_lakh"] - 1) * 100
            cumulative_change = (summary["median_cumulative_crore"] / baseline["median_cumulative_crore"] - 1) * 100
            death_ratio = summary["median_deaths"] / baseline["median_deaths"] if baseline["median_deaths"] else 0
            macros += [
                f"\\newcommand{{\\Peak{regime_name}{name}}}{{{lakh_text(summary['median_peak_lakh'])}}}",
                f"\\newcommand{{\\PeakRange{regime_name}{name}}}{{{summary['peak_lakh_10th_percentile']}--{summary['peak_lakh_90th_percentile']}}}",
                f"\\newcommand{{\\PeakChange{regime_name}{name}}}{{{signed_percent(peak_change)}}}",
                f"\\newcommand{{\\Cumulative{regime_name}{name}}}{{{summary['median_cumulative_crore']:.2f}~crore}}",
                f"\\newcommand{{\\CumulativeChange{regime_name}{name}}}{{{signed_percent(cumulative_change)}}}",
                f"\\newcommand{{\\Deaths{regime_name}{name}}}{{{summary['median_deaths']:g}}}",
                f"\\newcommand{{\\DeathRatio{regime_name}{name}}}{{{death_ratio:.1f}}}",
            ]
    robustness = json.loads((RESULTS_FOLDER / "robustness_checks.json").read_text())
    for check in robustness["hybrid_sensitivity"]:
        retained = int(round(check["symbolic_threat_retained"] * 100))
        macros.append(f"\\newcommand{{\\HybridRetained{['Forty', 'Sixty', 'Eighty'][[40, 60, 80].index(retained)]}}}{{{lakh_text(check['median_peak_lakh'])}}}")
    for check in robustness["managed_transition_leave_one_out"]:
        suffix = "Full" if check["left_out"] is None else "Without" + macro_name_for(check["left_out"])
        macros.append(f"\\newcommand{{\\ManagedTransition{suffix}}}{{{lakh_text(check['median_peak_lakh'])}}}")
    return macros


def scenario_table_rows(regime: str) -> list:
    scenarios = json.loads((RESULTS_FOLDER / f"intervention_comparison_{regime}.json").read_text())["scenarios"]
    baseline = scenarios["baseline"]["summary"]
    order = ["baseline", "symbolic_only_validation", "grandfathering", "seat_expansion", "hybrid_caste_subquotas",
             "sub_classification", "consensus_commission", "compensation", "credible_guarantees",
             "internet_shutdown", "heavy_policing", "managed_transition", "hybrid_package"]
    rows = []
    for key in order:
        scenario, summary = scenarios[key], scenarios[key]["summary"]
        change = "--" if key == "baseline" else signed_percent((summary["median_peak_lakh"] / baseline["median_peak_lakh"] - 1) * 100)
        label = scenario["label"].replace("₹", "\\rupee{}").replace("&", "\\&")
        rows.append(
            f"{scenario['code']} & {label} & {summary['median_peak_lakh']:.1f} & "
            f"{summary['peak_lakh_10th_percentile']}--{summary['peak_lakh_90th_percentile']} & {change} & "
            f"{summary['median_cumulative_crore']:.2f} & {summary['median_deaths']:g} \\\\"
        )
        if key in ("symbolic_only_validation", "credible_guarantees", "heavy_policing"):
            rows.append("\\midrule")
    return rows


def main():
    GENERATED_FOLDER.mkdir(parents=True, exist_ok=True)
    (GENERATED_FOLDER / "result_macros.tex").write_text("\n".join(result_macros()) + "\n")
    for regime in REGIME_MACRO_NAMES:
        (GENERATED_FOLDER / f"scenario_table_{regime}.tex").write_text("\n".join(scenario_table_rows(regime)) + "\n")
    print(f"Wrote LaTeX macros and tables to {GENERATED_FOLDER}")


if __name__ == "__main__":
    main()
