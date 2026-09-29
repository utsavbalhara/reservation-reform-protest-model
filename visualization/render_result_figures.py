import json

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from protest_simulation import BASELINE_ABRUPT_INCOME_ONLY_SWITCH, build_shared_population
from protest_simulation.protest_campaign import material_loss_felt_by_each_agent, symbolic_threat_felt_by_each_agent
from protest_simulation.synthetic_population import GENERAL, OBC, SC, ST

from .figure_style import (
    CATEGORY_COLOURS,
    CATEGORY_NAMES,
    GRID_LINE,
    GROUP_COLOURS,
    HIGHLIGHT_SCENARIO_COLOURS,
    INK,
    MATERIAL_LOSS_COLOUR,
    MUTED_INK,
    RESULTS_FOLDER,
    SECONDARY_INK,
    SURFACE,
    SYMBOLIC_THREAT_COLOUR,
    apply_house_style,
    format_lakh,
    save_figure,
)

SCENARIO_DISPLAY_ORDER = (
    ("Reference", ("baseline", "symbolic_only_validation")),
    ("Single levers", ("grandfathering", "seat_expansion", "hybrid_caste_subquotas", "sub_classification",
                       "consensus_commission", "compensation", "credible_guarantees")),
    ("Packages", ("managed_transition", "hybrid_package")),
    ("Suppression", ("internet_shutdown", "heavy_policing")),
)

READABLE_INTERVENTION_NAMES = {
    "grandfather_current_cohorts_with_ten_year_glide": "Grandfathering (L1)",
    "expand_seats_so_no_group_loses": "Seat expansion (L2)",
    "build_consensus_through_data_first_commission": "Consensus commission (L5)",
    "compensate_above_line_losers": "Compensation (L6)",
    "guarantee_untouched_protections": "Guarantees (L7)",
}

LAKH_AXIS_TICKS = [1, 10, 100, 1000]


def load_json(file_name):
    return json.loads((RESULTS_FOLDER / file_name).read_text())


def lakh_tick_formatter():
    return FuncFormatter(lambda lakh, _: {1: "1 lakh", 10: "10 lakh", 100: "1 crore", 1000: "10 crore"}.get(int(round(lakh)), ""))


def plot_intervention_ranking():
    regimes = (("central", "Central regime"), ("high-mobilization", "High-mobilization regime"))
    comparisons = {regime: load_json(f"intervention_comparison_{regime}.json")["scenarios"] for regime, _ in regimes}

    row_labels, row_keys, section_breaks = [], [], []
    for section_name, keys in SCENARIO_DISPLAY_ORDER:
        section_breaks.append((len(row_keys), section_name))
        for key in keys:
            scenario = comparisons["central"][key]
            row_labels.append(f"{scenario['code']}  {scenario['label']}")
            row_keys.append(key)
    row_positions = {}
    y = 0
    for index, key in enumerate(row_keys):
        if any(index == start for start, _ in section_breaks) and index:
            y += 0.8
        row_positions[key] = y
        y += 1

    figure, axes = plt.subplots(1, 2, figsize=(11, 6.6), sharey=True, gridspec_kw={"wspace": 0.06})
    for axis, (regime, title) in zip(axes, regimes):
        results = comparisons[regime]
        baseline_median = results["baseline"]["summary"]["median_peak_lakh"]
        axis.axvline(baseline_median, color=MUTED_INK, linestyle=(0, (4, 3)), linewidth=1.2, zorder=1)
        for key in row_keys:
            summary = results[key]["summary"]
            colour = CATEGORY_COLOURS[results[key]["category"]]
            y_position = row_positions[key]
            axis.plot(
                [summary["peak_lakh_10th_percentile"], summary["peak_lakh_90th_percentile"]],
                [y_position, y_position], color=colour, alpha=0.35, linewidth=6, solid_capstyle="round", zorder=2,
            )
            axis.scatter(summary["median_peak_lakh"], y_position, s=46, color=colour, edgecolor=SURFACE, linewidth=1.5, zorder=3)
            axis.text(summary["peak_lakh_90th_percentile"] * 1.25, y_position, format_lakh(summary["median_peak_lakh"]),
                      va="center", fontsize=8, color=INK, zorder=4,
                      bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.2})
        axis.set_xscale("log")
        axis.set_xlim(0.25, 3000)
        axis.xaxis.set_major_locator(FixedLocator(LAKH_AXIS_TICKS))
        axis.xaxis.set_minor_locator(NullLocator())
        axis.xaxis.set_major_formatter(lakh_tick_formatter())
        axis.grid(axis="y", visible=False)
        axis.set_title(title)
        axis.set_xlabel("Protesters on the peak day (median, 10th–90th percentile)")
    axes[0].set_yticks([row_positions[key] for key in row_keys], row_labels, fontsize=8.5)
    axes[0].invert_yaxis()
    for start, section_name in section_breaks:
        axes[0].annotate(section_name.upper(), xy=(0, row_positions[row_keys[start]] - 0.62), xycoords=("axes fraction", "data"),
                         xytext=(-8, 0), textcoords="offset points", ha="right", fontsize=7.5, color=MUTED_INK,
                         fontweight="bold", annotation_clip=False)
    legend_handles = [Line2D([], [], marker="o", linestyle="", markersize=7, color=colour, label=CATEGORY_NAMES[category])
                      for category, colour in CATEGORY_COLOURS.items()]
    legend_handles.append(Line2D([], [], color=MUTED_INK, linestyle=(0, (4, 3)), label="Baseline median"))
    figure.legend(handles=legend_handles, loc="upper center", ncol=5, bbox_to_anchor=(0.55, 1.02))
    save_figure(figure, "fig01_intervention_ranking")


def plot_daily_turnout_paths():
    trajectories = load_json("daily_trajectories_central.json")
    days = np.arange(len(trajectories["scenarios"]["baseline"]["median_daily_lakh"]))
    figure, axis = plt.subplots(figsize=(9, 4.6))
    for bandh_day in trajectories["bandh_call_days"]:
        axis.axvline(bandh_day, color=GRID_LINE, linewidth=5, zorder=0)
        axis.text(bandh_day + 0.4, 0.97, "bandh call", transform=axis.get_xaxis_transform(), ha="left", va="top", fontsize=7.5, color=MUTED_INK)
    for key, colour in HIGHLIGHT_SCENARIO_COLOURS.items():
        path = trajectories["scenarios"][key]
        median = np.maximum(path["median_daily_lakh"], 0.01)
        axis.fill_between(days, np.maximum(path["daily_lakh_10th_percentile"], 0.01), np.maximum(path["daily_lakh_90th_percentile"], 0.01),
                          color=colour, alpha=0.12, linewidth=0)
        axis.plot(days, median, color=colour, label=f"{path['code']}  {path['label']}")
    axis.set_yscale("log")
    axis.set_ylim(0.05, 2000)
    axis.yaxis.set_major_locator(FixedLocator([0.1, 1, 10, 100]))
    axis.yaxis.set_minor_locator(NullLocator())
    axis.yaxis.set_major_formatter(FuncFormatter(lambda lakh, _: {0.1: "10,000", 1: "1 lakh", 10: "10 lakh", 100: "1 crore"}.get(round(lakh, 1), "")))
    axis.set_xlim(0, days[-1])
    axis.set_xlabel("Day of campaign")
    axis.set_ylabel("Protesters that day")
    axis.set_title("Daily turnout over a 40-day campaign (median of 20 runs, shaded 10th–90th percentile)", pad=10)
    axis.legend(loc="upper right", ncol=2, bbox_to_anchor=(1, 0.93))
    save_figure(figure, "fig02_daily_turnout_paths")


def plot_turnout_by_group():
    baseline = load_json("daily_trajectories_central.json")["scenarios"]["baseline"]
    days = np.arange(len(baseline["median_daily_lakh"]))
    figure, axis = plt.subplots(figsize=(9, 4.2))
    for group, colour in GROUP_COLOURS.items():
        path = np.array(baseline["median_daily_lakh_by_group"][group])
        axis.plot(days, path, color=colour)
        axis.annotate(group, xy=(int(np.argmax(path)), path.max()), xytext=(6, 3), textcoords="offset points", fontsize=9, color=INK, fontweight="bold")
    axis.set_xlim(0, days[-1])
    axis.set_ylim(bottom=0)
    axis.set_xlabel("Day of campaign")
    axis.set_ylabel("Protesters that day (lakh)")
    axis.set_title("Who protests under the abrupt switch (baseline, median of 20 runs)")
    save_figure(figure, "fig03_turnout_by_group")


def plot_grievance_composition():
    population = build_shared_population()
    parameters = BASELINE_ABRUPT_INCOME_ONLY_SWITCH
    material_part = parameters.material_loss_weight * material_loss_felt_by_each_agent(population, parameters)
    symbolic_part = parameters.symbolic_threat_weight * population.identity_strength * symbolic_threat_felt_by_each_agent(population, parameters)
    group, above, deprived = population.social_group, population.is_above_income_line, population.is_most_deprived_tier
    segments = [
        ("SC, above ₹8L", (group == SC) & above),
        ("SC, below ₹8L, better-off tier", (group == SC) & ~above & ~deprived),
        ("SC, below ₹8L, most-deprived tier", (group == SC) & ~above & deprived),
        ("ST, above ₹8L", (group == ST) & above),
        ("ST, below ₹8L", (group == ST) & ~above),
        ("OBC, below ₹8L", (group == OBC) & ~above),
        ("OBC, above ₹8L", (group == OBC) & above),
        ("General, below ₹8L", (group == GENERAL) & ~above),
        ("General, above ₹8L", (group == GENERAL) & above),
    ]
    figure, axis = plt.subplots(figsize=(9, 4.4))
    for row, (name, members) in enumerate(segments):
        material, symbolic = material_part[members].mean(), symbolic_part[members].mean()
        share_of_population = members.mean() * 100
        left_start = 0.0
        for value, colour in ((symbolic, SYMBOLIC_THREAT_COLOUR), (material, MATERIAL_LOSS_COLOUR)):
            if value >= 0:
                axis.barh(row, value, left=max(left_start, 0), color=colour, height=0.62, edgecolor=SURFACE, linewidth=2)
                left_start = max(left_start, 0) + value
        negative_start = 0.0
        for value, colour in ((symbolic, SYMBOLIC_THREAT_COLOUR), (material, MATERIAL_LOSS_COLOUR)):
            if value < 0:
                axis.barh(row, value, left=negative_start, color=colour, height=0.62, edgecolor=SURFACE, linewidth=2)
                negative_start += value
        axis.text(max(left_start, 0) + 0.05, row, f"{share_of_population:.1f}% of India", va="center", fontsize=8, color=SECONDARY_INK)
    axis.axvline(0, color=INK, linewidth=0.8)
    axis.set_yticks(range(len(segments)), [name for name, _ in segments])
    axis.invert_yaxis()
    axis.grid(axis="y", visible=False)
    axis.set_xlabel("Average grievance (model units)")
    axis.set_title("Where grievance comes from: symbolic threat outweighs material loss")
    axis.legend(handles=[Line2D([], [], color=SYMBOLIC_THREAT_COLOUR, linewidth=8, label="Symbolic threat × identity"),
                         Line2D([], [], color=MATERIAL_LOSS_COLOUR, linewidth=8, label="Material loss × loss aversion")],
                loc="lower right")
    save_figure(figure, "fig04_grievance_composition")


def plot_tipping_curve():
    calibration = load_json("threshold_response_curve.json")["response_curve"]
    thresholds = [point["mean_participation_threshold"] for point in calibration]
    baseline = [max(point["baseline"]["median_peak_lakh"], 0.05) for point in calibration]
    symbolic_only = [max(point["symbolic_only_validation"]["median_peak_lakh"], 0.05) for point in calibration]
    figure, axis = plt.subplots(figsize=(9, 4.6))
    axis.axhspan(20, 100, color=GROUP_COLOURS["SC"], alpha=0.08, linewidth=0)
    axis.text(8.45, 60, "calibration target\n20 lakh – 1 crore", ha="right", va="center", fontsize=8, color=SECONDARY_INK)
    axis.plot(thresholds, baseline, color=CATEGORY_COLOURS["reference"], marker="o", markersize=4)
    axis.plot(thresholds, symbolic_only, color=GROUP_COLOURS["ST"], marker="o", markersize=4)
    label_index = thresholds.index(7.25)
    axis.annotate("Full reform (material + symbolic)", xy=(thresholds[label_index], baseline[label_index]), xytext=(8, 8), textcoords="offset points", fontsize=8.5)
    pointer_index = thresholds.index(6.0)
    axis.annotate("Symbolic threat only\n(2018-type validation check)", xy=(thresholds[pointer_index], symbolic_only[pointer_index]),
                  xytext=(5.05, 3), textcoords="data", fontsize=8.5, ha="left", va="center",
                  arrowprops={"arrowstyle": "-", "color": MUTED_INK, "linewidth": 0.8, "shrinkB": 4})
    axis.axvline(BASELINE_ABRUPT_INCOME_ONLY_SWITCH.mean_participation_threshold, color=INK, linewidth=1, linestyle=(0, (2, 2)))
    axis.text(BASELINE_ABRUPT_INCOME_ONLY_SWITCH.mean_participation_threshold + 0.05, 1500, "calibrated\nthreshold 6.6", fontsize=8, va="top")
    axis.set_yscale("log")
    axis.yaxis.set_major_locator(FixedLocator([0.1, 1, 10, 100, 1000]))
    axis.yaxis.set_minor_locator(NullLocator())
    axis.yaxis.set_major_formatter(FuncFormatter(lambda lakh, _: {0.1: "10,000", 1: "1 lakh", 10: "10 lakh", 100: "1 crore", 1000: "10 crore"}.get(round(lakh, 1), "")))
    axis.set_xlabel("Mean participation threshold (higher = harder to mobilize)")
    axis.set_ylabel("Protesters on the peak day")
    axis.set_title("Turnout falls roughly 2× per quarter-step in threshold: a smooth but steep response")
    save_figure(figure, "fig05_threshold_response_curve")


def plot_suppression_backfire():
    results = load_json("intervention_comparison_central.json")["scenarios"]
    scenarios = (("baseline", "Baseline", CATEGORY_COLOURS["reference"]), ("internet_shutdown", "B1 Internet shutdowns", GROUP_COLOURS["SC"]),
                 ("heavy_policing", "B2 Heavy policing", GROUP_COLOURS["ST"]))
    measures = (("median_peak_lakh", "Peak-day protesters"), ("median_cumulative_crore", "People who ever protest"), ("median_deaths", "Deaths"))
    figure, axis = plt.subplots(figsize=(9, 4.2))
    bar_width = 0.26
    for scenario_index, (key, name, colour) in enumerate(scenarios):
        for measure_index, (measure, _) in enumerate(measures):
            index_value = results[key]["summary"][measure] / results["baseline"]["summary"][measure] * 100
            x_position = measure_index + (scenario_index - 1) * (bar_width + 0.03)
            axis.bar(x_position, index_value, width=bar_width, color=colour, label=name if measure_index == 0 else None)
            axis.text(x_position, index_value + 6, f"{index_value:.0f}", ha="center", fontsize=8.5, color=INK,
                      bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1})
    axis.axhline(100, color=MUTED_INK, linewidth=1, linestyle=(0, (4, 3)))
    axis.set_xticks(range(len(measures)), [label for _, label in measures])
    axis.grid(axis="x", visible=False)
    axis.set_ylabel("Index, baseline = 100")
    axis.set_title("Suppression barely dents turnout and multiplies deaths")
    axis.legend(loc="upper left")
    save_figure(figure, "fig06_suppression_backfire")


def plot_robustness_checks():
    robustness = load_json("robustness_checks.json")
    baseline_median = load_json("intervention_comparison_central.json")["scenarios"]["baseline"]["summary"]["median_peak_lakh"]
    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.3], "wspace": 0.75})

    sensitivity = robustness["hybrid_sensitivity"]
    labels = [f"{check['symbolic_threat_retained']:.0%}" for check in sensitivity]
    values = [check["median_peak_lakh"] for check in sensitivity]
    left.bar(labels, values, color=CATEGORY_COLOURS["lever"], width=0.55)
    for x_position, value in enumerate(values):
        left.text(x_position, value + 0.3, f"{value:.1f} L\n({(value / baseline_median - 1) * 100:.0f}%)".replace("-", "−"), ha="center", fontsize=8.5)
    left.set_xlabel("Share of symbolic threat the hybrid leaves in place")
    left.set_ylabel("Peak-day protesters (lakh)")
    left.set_title("L3 hybrid holds up under weaker assumptions")
    left.set_ylim(0, max(values) * 1.45)
    left.grid(axis="x", visible=False)

    leave_one_out = robustness["managed_transition_leave_one_out"]
    names = ["All five parts"] + [f"Without {READABLE_INTERVENTION_NAMES[check['left_out']]}" for check in leave_one_out[1:]]
    values = [check["median_peak_lakh"] for check in leave_one_out]
    colours = [CATEGORY_COLOURS["package"]] + [CATEGORY_COLOURS["suppression"] if check["left_out"] == "build_consensus_through_data_first_commission" else "#9d97c9" for check in leave_one_out[1:]]
    right.barh(range(len(values)), values, color=colours, height=0.6)
    for row, value in enumerate(values):
        right.text(value + 0.08, row, f"{value:.1f} L", va="center", fontsize=8.5)
    right.set_yticks(range(len(values)), names)
    right.invert_yaxis()
    right.grid(axis="y", visible=False)
    right.set_xlabel("Peak-day protesters (lakh)")
    right.set_title("C1 managed transition: consensus carries it")
    right.set_xlim(0, max(values) * 1.25)
    save_figure(figure, "fig07_robustness_checks")


def main():
    apply_house_style()
    plot_intervention_ranking()
    plot_daily_turnout_paths()
    plot_turnout_by_group()
    plot_grievance_composition()
    plot_tipping_curve()
    plot_suppression_backfire()
    plot_robustness_checks()
    print("Figures saved")


if __name__ == "__main__":
    main()
