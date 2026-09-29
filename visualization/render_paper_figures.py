import json

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from protest_simulation import BASELINE_ABRUPT_INCOME_ONLY_SWITCH

from .figure_style import (
    CATEGORY_COLOURS,
    GROUP_COLOURS,
    INK,
    MUTED_INK,
    RESULTS_FOLDER,
    SECONDARY_INK,
    SURFACE,
    apply_house_style,
    save_figure,
)

ITERATION_TWO_CALIBRATED_THRESHOLD = 5.1
SENSITIVITY_BASELINE_RUNS = 20
EARLY_ITERATION_COLOUR = GROUP_COLOURS["ST"]
FINAL_MODEL_COLOUR = GROUP_COLOURS["SC"]
COMPARED_SCENARIOS = (
    ("symbolic_only_validation", "V  Symbolic threat only"),
    ("grandfathering", "L1 Grandfathering"),
    ("hybrid_caste_subquotas", "L3 Caste sub-quotas kept"),
    ("consensus_commission", "L5 Consensus commission"),
    ("compensation", "L6 Compensation"),
    ("managed_transition", "C1 Managed transition"),
)
READABLE_PARAMETER_NAMES = {
    "loss_aversion": "Loss aversion",
    "material_loss_weight": "Material-loss weight",
    "symbolic_threat_weight": "Symbolic-threat weight",
    "opposition_party_amplifier": "Opposition-party amplifier",
    "max_neighbourhood_influence": "Neighbourhood influence",
    "max_national_visibility_influence": "National visibility influence",
    "participation_threshold_spread": "Threshold spread",
    "decision_noise": "Decision noise",
    "fatigue_per_protest_day": "Fatigue",
    "deaths_per_crore_protester_days": "Death rate",
    "symbolic_threat_rise_per_death": "Martyr effect per death",
    "mean_participation_threshold": "Mean threshold (±0.25)",
}
LAKH_LABELS = {0.1: "10,000", 1: "1 lakh", 10: "10 lakh", 100: "1 crore", 1000: "10 crore"}


def load_json(file_name):
    return json.loads((RESULTS_FOLDER / file_name).read_text())


def reference_baseline_peak_lakh(run_count: int) -> float:
    baseline_runs = load_json("intervention_comparison_central.json")["scenarios"]["baseline"]["runs"]["peak_day_protesters"]
    return round(float(np.median(baseline_runs[:run_count])) / 1e5, 1)


def lakh_formatter():
    return FuncFormatter(lambda lakh, _: LAKH_LABELS.get(round(lakh, 1), ""))


def plot_model_development():
    history = load_json("model_development_history.json")
    final_curve = load_json("threshold_response_curve.json")["response_curve"]
    early_curve = history["iteration_2_narrow_thresholds"]["threshold_response_curve"]

    figure, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1, 1.15], "wspace": 0.55})
    left.plot([point["mean_participation_threshold"] - ITERATION_TWO_CALIBRATED_THRESHOLD for point in early_curve],
              [max(point["median_peak_lakh"], 0.05) for point in early_curve], color=EARLY_ITERATION_COLOUR, marker="o", markersize=4)
    final_points = [point for point in final_curve if abs(point["mean_participation_threshold"] - BASELINE_ABRUPT_INCOME_ONLY_SWITCH.mean_participation_threshold) <= 1.01]
    left.plot([point["mean_participation_threshold"] - BASELINE_ABRUPT_INCOME_ONLY_SWITCH.mean_participation_threshold for point in final_points],
              [max(point["baseline"]["median_peak_lakh"], 0.05) for point in final_points], color=FINAL_MODEL_COLOUR, marker="o", markersize=4)
    left.axvline(0, color=INK, linewidth=0.8, linestyle=(0, (2, 2)))
    left.set_yscale("log")
    left.yaxis.set_major_locator(FixedLocator([1, 10, 100, 1000]))
    left.yaxis.set_minor_locator(NullLocator())
    left.yaxis.set_major_formatter(lakh_formatter())
    left.set_xlabel("Mean threshold relative to each model's calibrated value")
    left.set_ylabel("Peak-day protesters (median)")
    left.set_title("Narrow thresholds tipped abruptly")
    left.legend(handles=[Line2D([], [], color=EARLY_ITERATION_COLOUR, marker="o", label="Iteration 2: narrow thresholds"),
                         Line2D([], [], color=FINAL_MODEL_COLOUR, marker="o", label="Final model: wide thresholds")], loc="upper right")

    early = history["iteration_2_narrow_thresholds"]["scenarios"]
    final = history["iteration_4_final"]["scenarios"]
    rows = np.arange(len(COMPARED_SCENARIOS))
    bar_height = 0.36
    for offset, (results, colour, name) in enumerate(((early, EARLY_ITERATION_COLOUR, "Iteration 2"), (final, FINAL_MODEL_COLOUR, "Final model"))):
        changes = [(results[key]["median_peak_lakh"] / results["baseline"]["median_peak_lakh"] - 1) * 100 for key, _ in COMPARED_SCENARIOS]
        positions = rows + (offset - 0.5) * (bar_height + 0.04)
        right.barh(positions, changes, height=bar_height, color=colour, label=name)
        for position, change in zip(positions, changes):
            right.text(change - 2, position, f"{change:.0f}%".replace("-", "−"), va="center", ha="right", fontsize=8, color=INK)
    right.set_yticks(rows, [label for _, label in COMPARED_SCENARIOS])
    right.invert_yaxis()
    right.set_xlim(-118, 0)
    right.grid(axis="y", visible=False)
    right.set_xlabel("Change in peak-day turnout vs that model's baseline")
    right.set_title("…and overstated the material levers", pad=26)
    right.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.2)
    save_figure(figure, "fig08_model_development")


def plot_sensitivity_tornado():
    sensitivity = load_json("sensitivity_analysis.json")
    reference = reference_baseline_peak_lakh(SENSITIVITY_BASELINE_RUNS)
    by_parameter = {}
    for row in sensitivity["perturbations"]:
        by_parameter.setdefault(row["parameter"], {})[row["direction"]] = row
    spans = sorted(by_parameter.items(), key=lambda item: abs(np.log(item[1]["higher"]["baseline_median_peak_lakh"] / item[1]["lower"]["baseline_median_peak_lakh"])))
    figure, axis = plt.subplots(figsize=(10.5, 5.2))
    for row_index, (parameter, directions) in enumerate(spans):
        lower, higher = directions["lower"]["baseline_median_peak_lakh"], directions["higher"]["baseline_median_peak_lakh"]
        axis.plot([lower, higher], [row_index, row_index], color="#c9c8c3", linewidth=5, solid_capstyle="round", zorder=1)
        axis.scatter(lower, row_index, color=FINAL_MODEL_COLOUR, s=44, zorder=2, edgecolor=SURFACE, linewidth=1.5)
        axis.scatter(higher, row_index, color=EARLY_ITERATION_COLOUR, s=44, zorder=2, edgecolor=SURFACE, linewidth=1.5)
        correlations = [directions[d]["rank_correlation_with_reference"] for d in ("lower", "higher")]
        axis.text(1.02, row_index, f"ρ = {correlations[0]:.2f} / {correlations[1]:.2f}", transform=axis.get_yaxis_transform(),
                  va="center", fontsize=8.5, color=SECONDARY_INK)
    axis.axvline(reference, color=MUTED_INK, linestyle=(0, (4, 3)), linewidth=1.2)
    axis.set_yticks(range(len(spans)), [READABLE_PARAMETER_NAMES[parameter] for parameter, _ in spans])
    axis.set_xscale("log")
    axis.xaxis.set_major_locator(FixedLocator([1, 10, 100, 1000]))
    axis.xaxis.set_minor_locator(NullLocator())
    axis.xaxis.set_major_formatter(lakh_formatter())
    axis.grid(axis="y", visible=False)
    axis.set_xlabel("Baseline peak-day protesters (median of 20 runs)")
    axis.set_title("One-at-a-time sensitivity (±25%); ρ is the lever-ranking correlation with the reference")
    axis.legend(handles=[Line2D([], [], color=FINAL_MODEL_COLOUR, marker="o", linestyle="", label="Parameter lowered"),
                         Line2D([], [], color=EARLY_ITERATION_COLOUR, marker="o", linestyle="", label="Parameter raised"),
                         Line2D([], [], color=MUTED_INK, linestyle=(0, (4, 3)), label="Unperturbed baseline (same 20 runs)")],
                loc="lower right")
    save_figure(figure, "fig09_sensitivity_tornado")


def main():
    apply_house_style()
    plot_model_development()
    plot_sensitivity_tornado()
    print("Paper figures saved")


if __name__ == "__main__":
    main()
