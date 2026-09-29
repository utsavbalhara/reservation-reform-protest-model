"""Figures for the allocation model, episode calibration, grounded comparison and uncertainty analyses.

Colour jobs: gains and losses use the diverging pair (blue gain, orange loss); specifications and observed-versus-model
use the categorical order (blue, then orange). Every bar carries its value as text, since two palette colours fall
below 3:1 contrast with the surface. Figures whose result file is missing are skipped.
"""
import json

import matplotlib.pyplot as plt
import numpy as np

from .figure_style import INK, MUTED_INK, RESULTS_FOLDER, REPOSITORY_ROOT, SECONDARY_INK, apply_house_style, save_figure

BLUE, ORANGE, GREEN, GOLD = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
LEVER_CODES = {"grandfathering": "L1", "seat_expansion": "L2", "hybrid_caste_subquotas": "L3", "sub_classification": "L4",
               "consensus_commission": "L5", "compensation": "L6", "credible_guarantees": "L7", "internet_shutdown": "B1",
               "heavy_policing": "B2", "managed_transition": "C1", "hybrid_package": "C2", "symbolic_only_validation": "V"}
SEGMENT_LABELS = {"SC_above": "SC, above line", "SC_below": "SC, below line", "ST_above": "ST, above line", "ST_below": "ST, below line",
                  "OBC-NCL_above": "OBC non-creamy, above line", "OBC-NCL_below": "OBC, below line", "GEN-EWS": "General, EWS-eligible",
                  "GEN_below": "General, below line, not EWS", "GEN_above": "General, above line"}
SHORT_LABELS = {"symbolic_only_validation": "Symbolic threat only (check)", "grandfathering": "Grandfathering, 10-year glide",
                "seat_expansion": "Expand seats by a quarter", "hybrid_caste_subquotas": "Caste quotas with income filter",
                "sub_classification": "Sub-classification", "consensus_commission": "Data-first commission, consensus",
                "compensation": "Compensate above-line losers", "credible_guarantees": "Guarantee untouched protections",
                "internet_shutdown": "Internet shutdowns", "heavy_policing": "Heavy policing", "managed_transition": "Managed transition package",
                "hybrid_package": "Hybrid package", "baseline": "Abrupt switch"}
EPISODE_LABELS = {"sc_st_bharat_bandh_2018": "SC/ST bandh, Apr 2018", "upper_caste_bandh_2018": "Upper-caste bandh, Sep 2018",
                  "sc_st_bharat_bandh_2024": "SC/ST bandh, Aug 2024", "ews_quota_2019": "EWS amendment, Jan 2019"}


def load(name):
    path = RESULTS_FOLDER / name
    return json.loads(path.read_text()) if path.exists() else None


def value_label(axis, x, y, text, left=False):
    axis.text(x, y, text, va="center", ha="right" if left else "left", fontsize=8, color=SECONDARY_INK)


def plot_allocation():
    data = load("merged_pool_allocation.json")
    if data is None:
        return
    main = data["summary"]["reserved_first|0.595"]
    segments = list(SEGMENT_LABELS)
    figure, axis = plt.subplots(figsize=(7.2, 3.8))
    for row, segment in enumerate(segments):
        change = main[segment]["percent_change"]
        median = change["median"]
        colour = ORANGE if median < 0 else BLUE
        axis.barh(row, median, color=colour, height=0.62)
        axis.plot([change["p05"], change["p95"]], [row, row], color=INK, linewidth=1)
        value_label(axis, (change["p95"] if median >= 0 else change["p05"]) + (5 if median >= 0 else -6), row,
                    f"{median:+.0f}%".replace("-", "\u2212"), left=median < 0)
    axis.axvline(0, color=MUTED_INK, linewidth=0.8)
    axis.set_yticks(range(len(segments)), [SEGMENT_LABELS[s] for s in segments])
    axis.invert_yaxis()
    axis.set_xlim(-140, 215)
    axis.set_xlabel("Change in IIT seats under a merged income-only pool (%)")
    axis.set_title("Who gains and loses seats")
    axis.grid(axis="y", visible=False)
    save_figure(figure, "fig10_allocation_seat_changes")


def plot_calibration():
    data = load("episode_calibration.json")
    targets_path = REPOSITORY_ROOT / "data" / "derived" / "episode_targets.json"
    if data is None or not data.get("kept_samples"):
        return
    targets = json.loads(targets_path.read_text())
    samples = json.loads((RESULTS_FOLDER / "episode_calibration_nroy_samples.json").read_text())
    figure, (left, right) = plt.subplots(1, 2, figsize=(8.6, 3.4), gridspec_kw={"width_ratios": [1.35, 1]})
    episodes = list(EPISODE_LABELS)
    model_rates = data.get("model_event_rates", {})
    for row, key in enumerate(episodes):
        target = targets[key]["corrected_core_events_per_1000_india_events"]
        left.plot([target["p05"], target["p95"]], [row - 0.12, row - 0.12], color=INK, linewidth=2.2)
        left.plot(target["median"], row - 0.12, "o", color=INK, markersize=6, label="Observed (GDELT)" if row == 0 else None)
        if key in model_rates:
            rates = np.array(model_rates[key])
            left.plot([np.percentile(rates, 5), np.percentile(rates, 95)], [row + 0.12, row + 0.12], color=BLUE, linewidth=2.2)
            left.plot(np.median(rates), row + 0.12, "o", color=BLUE, markersize=6, label="Model, retained sets" if row == 0 else None)
    left.set_xscale("log")
    left.set_yticks(range(len(episodes)), [EPISODE_LABELS[k] for k in episodes])
    left.invert_yaxis()
    left.set_xlabel("Protest events per 1,000 India events (log scale)")
    left.set_title("Event rates: observed and fitted")
    left.legend(loc="upper left")
    left.grid(axis="y", visible=False)
    turnout = np.array([record["bandh_day_turnout_2018"] for record in data["kept_samples"]]) / 1e5
    bins = np.logspace(np.log10(max(turnout.min(), 0.5)), np.log10(turnout.max()), 14)
    counts, _, _ = right.hist(turnout, bins=bins, color=BLUE, edgecolor="#fcfcfb", linewidth=1.5, zorder=2)
    right.set_xscale("log")
    right.set_ylim(0, counts.max() * 1.25)
    right.axvspan(20, 100, color=GOLD, alpha=0.22, linewidth=0, zorder=0)
    right.text(np.sqrt(20 * 100), counts.max() * 1.22, "stylized target for\nthe reform peak", ha="center", va="top", fontsize=7.5,
               color=SECONDARY_INK, zorder=3)
    right.set_xlabel("2018 bandh-day turnout implied (lakh, log scale)")
    right.set_ylabel("Retained parameter sets")
    right.set_title("What the event data imply for 2018")
    save_figure(figure, "fig11_episode_calibration")


def plot_paired_effects():
    grounded = load("intervention_comparison_grounded_central.json")
    stylized = load("intervention_comparison_central.json")
    if grounded is None or stylized is None:
        return
    order = ["symbolic_only_validation", "grandfathering", "seat_expansion", "hybrid_caste_subquotas", "sub_classification",
             "consensus_commission", "compensation", "credible_guarantees", "internet_shutdown", "heavy_policing",
             "managed_transition", "hybrid_package"]
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    for offset, (data, colour, label) in ((-0.17, (grounded, BLUE, "Grounded")), (0.17, (stylized, ORANGE, "Stylized"))):
        for row, key in enumerate(order):
            effect = data["scenarios"][key]["paired_effects_vs_baseline"]["peak"]
            low, high = effect["change_percent_ci95"]
            axis.plot([low, high], [row + offset, row + offset], color=colour, linewidth=2)
            axis.plot(effect["change_percent"], row + offset, "o", color=colour, markersize=5.5, label=label if row == 0 else None,
                      markeredgecolor="#fcfcfb", markeredgewidth=1)
    axis.axvline(0, color=MUTED_INK, linewidth=0.8)
    axis.set_yticks(range(len(order)), [f"{LEVER_CODES[k]}  {SHORT_LABELS[k]}" for k in order])
    axis.invert_yaxis()
    axis.set_xlim(-102, 10)
    axis.set_xlabel("Change in peak-day turnout vs abrupt switch (median paired change, 95% CI)")
    axis.set_title("Intervention effects in both specifications")
    axis.legend(loc="lower right")
    axis.grid(axis="y", visible=False)
    save_figure(figure, "fig12_paired_effects")


def plot_rank_probabilities():
    stylized, grounded = load("intervention_mapping_uncertainty_stylized.json"), load("intervention_mapping_uncertainty_grounded.json")
    available = [(name, data, colour) for name, data, colour in (("Grounded", grounded, BLUE), ("Stylized", stylized, ORANGE)) if data]
    if not available:
        return
    levers = list(available[0][1]["probability_strongest_single_lever"])
    figure, axis = plt.subplots(figsize=(6.4, 3.4))
    height = 0.8 / len(available)
    for index, (name, data, colour) in enumerate(available):
        for row, lever in enumerate(levers):
            probability = data["probability_strongest_single_lever"][lever] * 100
            y = row + (index - (len(available) - 1) / 2) * height
            axis.barh(y, probability, height=height * 0.9, color=colour, label=name if row == 0 else None)
            value_label(axis, probability + 1.5, y, f"{probability:.0f}%")
    axis.set_yticks(range(len(levers)), [LEVER_CODES[lever] for lever in levers])
    axis.invert_yaxis()
    axis.set_xlim(0, 110)
    axis.set_xlabel("Probability of being the strongest single lever (%), over priors on effect sizes")
    axis.set_title("How often each lever comes first")
    axis.legend(loc="lower right")
    axis.grid(axis="y", visible=False)
    save_figure(figure, "fig13_rank_probabilities")


def plot_morris():
    data = load("global_sensitivity_stylized.json")
    if data is None:
        return
    from .write_grounded_macros import MORRIS_LABELS
    panels = (("log_peak", "Baseline log peak"), ("peak_change_percent:hybrid_caste_subquotas", "Effect of L3 (points)"),
              ("peak_change_percent:consensus_commission", "Effect of L5 (points)"))
    figure, axes = plt.subplots(1, 3, figsize=(9.6, 4.0), sharey=True)
    order = data["results"]["log_peak"]["ranking"]
    for axis, (key, title) in zip(axes, panels):
        factors = data["results"][key]["factors"]
        for row, name in enumerate(order):
            factor = factors[name]
            axis.barh(row, factor["mu_star"], color=BLUE, height=0.62)
            axis.plot(factor["mu_star_ci95"], [row, row], color=INK, linewidth=1)
        axis.set_title(title)
        axis.set_xlabel("Morris $\\mu^*$")
        axis.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(order)), [MORRIS_LABELS[name].replace("\\bar\\theta", "\\bar{\\theta}") for name in order])
    axes[0].invert_yaxis()
    save_figure(figure, "fig14_morris_screening")


def plot_grounded_paths():
    data = load("daily_trajectories_grounded_central.json")
    if data is None:
        return
    scenarios = (("baseline", "#6f6e69"), ("hybrid_caste_subquotas", GREEN), ("consensus_commission", BLUE), ("heavy_policing", ORANGE))
    figure, axis = plt.subplots(figsize=(8.4, 4.2))
    for key, colour in scenarios:
        path = data["scenarios"][key]
        days = np.arange(1, len(path["median_daily_lakh"]) + 1)
        axis.fill_between(days, np.maximum(path["daily_lakh_10th_percentile"], 0.01), np.maximum(path["daily_lakh_90th_percentile"], 0.01),
                          color=colour, alpha=0.12, linewidth=0)
        axis.plot(days, np.maximum(path["median_daily_lakh"], 0.01), color=colour, label=f"{path['code']}  {SHORT_LABELS[key]}")
    axis.set_yscale("log")
    axis.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:g}"))
    axis.set_xlim(1, 40)
    axis.set_xlabel("Day of campaign")
    axis.set_ylabel("Protesters that day (lakh, log scale)")
    axis.set_title(f"Grounded specification: daily turnout (median of {data['run_count']} runs, shaded 10th–90th percentile)")
    axis.legend(loc="upper right")
    save_figure(figure, "fig15_grounded_daily_paths")


SEGMENT_DISPLAY = [("SC_above", "SC, above line"), ("SC_below", "SC, below line"), ("ST_above", "ST, above line"),
                   ("ST_below", "ST, below line"), ("OBC_above_ncl", "OBC non-creamy, above line"), ("OBC_creamy", "OBC creamy layer"),
                   ("OBC_below", "OBC, below line"), ("General_above", "General, above line"), ("General_below_ews", "General, EWS-eligible"),
                   ("General_below_asset_excluded", "General, below line, not EWS")]


def plot_grievance_composition():
    data = load("grounded_channels.json")
    if data is None:
        return
    table = data["grievance_composition"]["by_segment"]
    figure, axis = plt.subplots(figsize=(7.2, 4.0))
    rows = [(key, label) for key, label in SEGMENT_DISPLAY if key in table]
    for row, (key, label) in enumerate(rows):
        material, symbolic = table[key]["mean_material"], table[key]["mean_symbolic"]
        # Positive parts stack to the right of zero, negative parts to the left, with a 2px-equivalent gap.
        right = 0.0
        for value, colour in ((symbolic, BLUE), (material, ORANGE)):
            if value > 0:
                axis.barh(row, value, left=right, color=colour, height=0.62, edgecolor="#fcfcfb", linewidth=1.2)
                right += value
        left = 0.0
        for value, colour in ((symbolic, BLUE), (material, ORANGE)):
            if value < 0:
                axis.barh(row, value, left=left, color=colour, height=0.62, edgecolor="#fcfcfb", linewidth=1.2)
                left += value
        share = table[key]["share_of_population"] * 100
        axis.text(max(right, 0) + 0.08, row, f"{share:.0f}% of India", va="center", fontsize=7.5, color=SECONDARY_INK)
    axis.axvline(0, color=MUTED_INK, linewidth=0.8)
    axis.set_yticks(range(len(rows)), [label for _, label in rows])
    axis.invert_yaxis()
    axis.set_xlim(-2.4, 4.2)
    axis.set_xlabel("Mean grievance component (positive = pushes towards protest)")
    axis.set_title("Grievance by segment, grounded model")
    from matplotlib.patches import Patch
    axis.legend(handles=[Patch(color=BLUE, label="Symbolic threat × identity"), Patch(color=ORANGE, label="Material change × loss aversion")],
                loc="lower right")
    axis.grid(axis="y", visible=False)
    save_figure(figure, "fig16_grievance_composition_grounded")


def plot_state_distribution():
    data = load("grounded_channels.json")
    targets_path = REPOSITORY_ROOT / "data" / "derived" / "episode_targets.json"
    if data is None or not targets_path.exists():
        return
    shares = data["who_and_where"]["baseline"]["protester_day_share_by_state"]
    observed = json.loads(targets_path.read_text())["sc_st_bharat_bandh_2018"]["core_events_by_state"]
    observed = {state: count for state, count in observed.items() if state in shares}
    observed_total = sum(observed.values())
    states = list(shares)[:14]
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    for row, state in enumerate(states):
        axis.barh(row, shares[state] * 100, color=BLUE, height=0.62, label="Model: reform, share of protester-days" if row == 0 else None)
        if observed_total:
            axis.plot(observed.get(state, 0) / observed_total * 100, row, "o", color=INK, markersize=5,
                      label="Observed: 2 April 2018, share of GDELT events" if row == 0 else None)
    axis.set_yticks(range(len(states)), states)
    axis.invert_yaxis()
    axis.set_xlabel("Share of national total (%)")
    axis.set_title("Where protest concentrates")
    axis.legend(loc="center right")
    axis.grid(axis="y", visible=False)
    save_figure(figure, "fig17_state_distribution")


def main():
    apply_house_style()
    for plot in (plot_allocation, plot_calibration, plot_paired_effects, plot_rank_probabilities, plot_morris, plot_grounded_paths, plot_grievance_composition, plot_state_distribution):
        plot()
    print("Rendered grounded figures")


if __name__ == "__main__":
    main()
