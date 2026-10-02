"""Figures for the v3 model (files figures/v3_*.png and .pdf).

Colour jobs follow the house style: categorical slots in fixed order (blue, orange, aqua, yellow, magenta) for series,
the sequential blue ramp for maps, the diverging orange (loss) / blue (gain) pair for seat changes. Every chart with two
or more series has a legend, and values that matter carry direct labels.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm

from .figure_style import INK, MUTED_INK, REPOSITORY_ROOT, RESULTS_FOLDER, SECONDARY_INK, TURNOUT_COLOUR_MAP, apply_house_style, save_figure

BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
EPISODE_LABELS = {"sc_st_bharat_bandh_2018": "SC/ST bandh, Apr 2018", "upper_caste_bandh_2018": "Upper-caste bandh, Sep 2018",
                  "sc_st_bharat_bandh_2024": "SC/ST bandh, Aug 2024", "ews_quota_2019": "EWS amendment, Jan 2019",
                  "patidar_2015": "Patidar, Gujarat 2015", "jat_2016": "Jat, Haryana 2016", "kapu_2016": "Kapu, Andhra 2016",
                  "maratha_march_mumbai_2017": "Maratha march, 2017", "maratha_quota_2018": "Maratha quota, 2018", "gujjar_2019": "Gujjar, Rajasthan 2019"}
PARAMETER_LABELS = {"mean_threshold": "Mean threshold", "threshold_spread": "Threshold spread", "mixing": "Neighbourhood mixing",
                    "fatigue": "Fatigue", "neighbourhood_influence": "Neighbourhood influence", "national_influence": "National influence",
                    "community_capacity": "Community organization", "kappa_urban": "Urban effect", "kappa_literacy": "Literacy effect",
                    "kappa_phone": "Phone-ownership effect", "log10_death_rate": "Death rate", "death_dispersion": "Death dispersion",
                    "log10_observation_scale": "Reporting scale", "observation_exponent": "Reporting exponent", "reporting_power": "Reporting-intensity power",
                    "death_concentration_power": "Deaths: concentration power", "martyr_effect_per_death": "Martyr effect",
                    "initial_awareness": "Initial awareness", "awareness_diffusion": "Awareness diffusion"}
MODIFIER_STYLE = {"none": ("As proposed", BLUE), "phased": ("Phased", ORANGE), "negotiated": ("Negotiated", AQUA), "all_three": ("Phased + negotiated + guaranteed", YELLOW)}
REFORM_ORDER = ("sc_st_creamy_layer", "sub_classification", "income_only_expanded", "income_only", "abolition")


def load(name):
    path = RESULTS_FOLDER / name
    return json.loads(path.read_text()) if path.exists() else None


def episode_fit(calibration, targets):
    keys = list(EPISODE_LABELS)
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    for row, key in enumerate(keys):
        observed = targets[key]["core_events_per_1000"]
        model = calibration["model_events_per_1000"][key]
        axis.plot([observed["p05"], observed["p95"]], [row - 0.15] * 2, color=INK, linewidth=1)
        axis.plot(observed["median"], row - 0.15, "o", color=INK, markersize=6)
        axis.plot([model["p05"], model["p95"]], [row + 0.15] * 2, color=BLUE, linewidth=1)
        axis.plot(model["median"], row + 0.15, "o", color=BLUE, markersize=6, markeredgecolor="white", markeredgewidth=1)
    axis.plot([], [], "o", color=INK, label="Observed (GDELT, precision-corrected)")
    axis.plot([], [], "o", color=BLUE, label="Model, retained sets (median, 90%)")
    axis.set_xscale("log")
    axis.set_yticks(range(len(keys)), [EPISODE_LABELS[k] for k in keys])
    axis.invert_yaxis()
    axis.set_xlabel("Protest events reported on core days, per 1,000 GDELT events in India")
    axis.set_title("Fit to ten reservation episodes")
    axis.legend(loc="upper center", bbox_to_anchor=(0.4, -0.13), ncol=2, fontsize=8)
    axis.grid(axis="y", visible=False)
    save_figure(figure, "v3_episode_fit")


def profiles(calibration, targets):
    keys = ("patidar_2015", "jat_2016", "maratha_quota_2018", "gujjar_2019")
    figure, axes = plt.subplots(1, 4, figsize=(7.4, 2.4), sharey=True)
    for axis, key in zip(axes, keys):
        counts = np.array(targets[key]["daily_core_events"], float)
        model = np.array(calibration["model_profiles"][key])
        x = np.arange(1, len(counts) + 1)
        axis.bar(x, counts / counts.sum(), color="#d7d6d1", width=0.66, label="Observed")
        axis.plot(x, model, "o-", color=BLUE, markersize=5, linewidth=1.6, label="Model")
        axis.set_title(EPISODE_LABELS[key], fontsize=8.5)
        axis.set_xticks(x)
        axis.set_xlabel("Core day", fontsize=8)
    axes[0].set_ylabel("Share of core-day events")
    axes[0].legend(loc="upper left", fontsize=7.5)
    figure.suptitle("Day-to-day profile of multi-day agitations", fontsize=10, x=0.02, y=1.06, ha="left", fontweight="bold")
    save_figure(figure, "v3_profiles")


def identification(calibration):
    names = list(PARAMETER_LABELS)
    shares = [calibration["identification"][n] for n in names]
    order = np.argsort(shares)
    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    axis.barh(range(len(names)), np.array(shares)[order], color=BLUE, height=0.6)
    for row, value in enumerate(np.array(shares)[order]):
        axis.text(value + 0.01, row, f"{value * 100:.0f}%", va="center", fontsize=8, color=SECONDARY_INK)
    axis.set_yticks(range(len(names)), [PARAMETER_LABELS[names[i]] for i in order])
    axis.axvline(1.0, color=MUTED_INK, linewidth=0.8, linestyle="--")
    axis.set_xlim(0, 1.15)
    axis.set_xlabel("Retained 90% range as a share of the prior range (lower = better identified)")
    axis.set_title("What the ten episodes pin down")
    axis.grid(axis="y", visible=False)
    save_figure(figure, "v3_identification")


def district_polygons():
    boundaries = json.loads((REPOSITORY_ROOT / "data" / "derived" / "district_boundaries_2011.json").read_text())["districts"]
    table = pd.read_csv(REPOSITORY_ROOT / "data" / "derived" / "census2011_district_sc_st.csv")
    polygons, owner = [], []
    for row, code in enumerate(table.district_code):
        for ring in boundaries.get(str(int(code)), []):
            polygons.append(np.array(ring))
            owner.append(row)
    return polygons, np.array(owner)


def draw_map(axis, values, polygons, owner, norm, title):
    collection = PolyCollection(polygons, array=np.asarray(values)[owner], cmap=TURNOUT_COLOUR_MAP, norm=norm, edgecolors="none")
    axis.add_collection(collection)
    axis.set_xlim(67.5, 98)
    axis.set_ylim(6.5, 37.5)
    axis.set_aspect("equal")
    axis.axis("off")
    axis.set_title(title, fontsize=9)
    return collection


def map_2018(scenarios):
    polygons, owner = district_polygons()
    values = np.maximum(np.array(scenarios["replay_2018|none"]["district_share_mean"]), 1e-6)
    locations = pd.read_csv(REPOSITORY_ROOT / "data" / "derived" / "gdelt_episode_locations.csv")
    core = locations[(locations.episode == "sc_st_bharat_bandh_2018") & (locations.date == "2018-04-02")]
    figure, axis = plt.subplots(figsize=(5.2, 5.6))
    norm = LogNorm(vmin=max(values.min(), 1e-5), vmax=values.max())
    collection = draw_map(axis, values, polygons, owner, norm, "")
    axis.scatter(core.long, core.lat, s=6, color=ORANGE, edgecolors="white", linewidths=0.3, label="Reported events, 2 April 2018 (GDELT)")
    colorbar = figure.colorbar(collection, ax=axis, shrink=0.6, pad=0.01)
    colorbar.set_label("Model: share of district population on the street, mean day", fontsize=8)
    axis.legend(loc="lower left", fontsize=7.5)
    axis.set_title("The 2018 bandh: model districts and reported events", fontsize=10)
    save_figure(figure, "v3_map_2018")


def scenario_chart(scenarios):
    figure, axis = plt.subplots(figsize=(7.2, 4.3))
    offsets = {"none": -0.27, "phased": -0.09, "negotiated": 0.09, "all_three": 0.27}
    rows = list(REFORM_ORDER)
    for row, reform in enumerate(rows):
        for modifier, offset in offsets.items():
            result = scenarios.get(f"{reform}|{modifier}")
            if result is None:
                continue
            value = result["events_vs_2018"]
            colour = MODIFIER_STYLE[modifier][1]
            axis.plot([value["p05"], value["p95"]], [row + offset] * 2, color=colour, linewidth=1.2, alpha=0.8)
            axis.plot(value["median"], row + offset, "o", color=colour, markersize=5.5, markeredgecolor="white", markeredgewidth=0.8)
        top = scenarios[f"{reform}|none"]["events_vs_2018"]["median"]
        axis.text(top, row - 0.42, f"{top:.2f}x", fontsize=7.5, color=SECONDARY_INK, ha="center")
    ews = scenarios["replay_ews_2019|none"]["events_vs_2018"]
    axis.axvline(1.0, color=INK, linewidth=1, linestyle="--")
    axis.text(1.0, len(rows) - 0.4, " 2018 bandh", fontsize=8, color=INK, va="center")
    axis.axvspan(ews["p05"], ews["p95"], color="#d7d6d1", alpha=0.5, linewidth=0)
    axis.text(ews["median"], -0.75, "EWS 2019\n(replay)", fontsize=7.5, color=SECONDARY_INK, ha="center")
    for modifier, (label, colour) in MODIFIER_STYLE.items():
        axis.plot([], [], "o", color=colour, label=label)
    axis.set_xscale("log")
    axis.set_yticks(range(len(rows)), [scenarios[f"{r}|none"]["label"] for r in rows])
    axis.invert_yaxis()
    axis.set_ylim(len(rows) - 0.5, -1.1)
    axis.set_xlabel("Reported protest on the bandh day, relative to 2 April 2018 (median, 90% range)")
    axis.set_title("How each reform compares with the 2018 bandh")
    axis.legend(loc="upper center", bbox_to_anchor=(0.4, -0.16), ncol=4, fontsize=7.5)
    axis.grid(axis="y", visible=False)
    save_figure(figure, "v3_scenarios")


def scenario_maps(scenarios):
    polygons, owner = district_polygons()
    keys = ("sc_st_creamy_layer|none", "income_only|none", "abolition|none")
    values = [np.maximum(np.array(scenarios[k]["district_share_mean"]), 1e-6) for k in keys]
    high = max(v.max() for v in values)
    norm = LogNorm(vmin=high / 1000, vmax=high)
    titles = ("SC/ST creamy layer", "Income-only test", "Abolish all reservation")
    figure, axes = plt.subplots(1, 3, figsize=(9.0, 3.9), gridspec_kw={"wspace": 0.02})
    for axis, key, value, title in zip(axes, keys, values, titles):
        collection = draw_map(axis, value, polygons, owner, norm, "")
        median = scenarios[key]["events_vs_2018"]["median"]
        axis.set_title(f"{title}\n{median:.2f}x the 2018 bandh", fontsize=9)
    colorbar = figure.colorbar(collection, ax=axes, shrink=0.75, pad=0.01)
    colorbar.set_label("Share on the street, mean day", fontsize=8)
    figure.suptitle("Where each reform would be contested (same colour scale)", fontsize=10, x=0.02, ha="left", fontweight="bold")
    save_figure(figure, "v3_scenario_maps")


def seats(scenarios_data):
    changes = scenarios_data.get("seat_changes_percent", {})
    reforms = [r for r in ("sc_st_creamy_layer", "income_only_expanded", "income_only", "abolition") if r in changes]
    segments = ["SC_above", "SC_below", "ST_above", "ST_below", "OBC-NCL_above", "OBC-NCL_below", "GEN-EWS", "GEN_below", "GEN_above"]
    labels = {"SC_above": "SC, above line", "SC_below": "SC, below line", "ST_above": "ST, above line", "ST_below": "ST, below line",
              "OBC-NCL_above": "OBC non-creamy, above", "OBC-NCL_below": "OBC, below line", "GEN-EWS": "General, EWS",
              "GEN_below": "General, below, not EWS", "GEN_above": "General, above line"}
    matrix = np.array([[changes[r][s] for r in reforms] for s in segments])
    figure, axis = plt.subplots(figsize=(6.8, 4.2))
    limit = 100
    image = axis.imshow(np.clip(matrix, -limit, limit), cmap=plt.get_cmap("RdBu"), vmin=-limit, vmax=limit, aspect="auto")
    for i in range(len(segments)):
        for j in range(len(reforms)):
            value = matrix[i, j]
            axis.text(j, i, f"{value:+.0f}%".replace("-", "−"), ha="center", va="center", fontsize=8,
                      color="white" if abs(value) > 60 else INK)
    axis.set_xticks(range(len(reforms)), [scenarios_data["scenarios"][f"{r}|none"]["label"].replace(", ", ",\n").replace(" (", "\n(") for r in reforms], fontsize=7.5)
    axis.set_yticks(range(len(segments)), [labels[s] for s in segments])
    axis.set_title("IIT seats by group: change under each reform (allocation model)")
    axis.grid(False)
    colorbar = figure.colorbar(image, ax=axis, shrink=0.8)
    colorbar.set_label("Seat change, % (clipped at ±100)", fontsize=8)
    save_figure(figure, "v3_seats")


def who(scenarios_data):
    groups = scenarios_data["identity_groups"]
    reforms = list(REFORM_ORDER)
    shares = np.array([[scenarios_data["scenarios"][f"{r}|none"]["share_by_identity_group"][g] for g in groups] for r in reforms])
    folded = np.column_stack([shares[:, :4], shares[:, 4:].sum(axis=1)])
    names = ["SC", "ST", "OBC", "General", "State communities"]
    colours = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]
    figure, axis = plt.subplots(figsize=(7.2, 3.0))
    left = np.zeros(len(reforms))
    for k, (name, colour) in enumerate(zip(names, colours)):
        axis.barh(range(len(reforms)), folded[:, k], left=left, color=colour, height=0.6, edgecolor="white", linewidth=1, label=name)
        for row in range(len(reforms)):
            if folded[row, k] > 0.08:
                axis.text(left[row] + folded[row, k] / 2, row, f"{folded[row, k] * 100:.0f}%", ha="center", va="center", fontsize=7.5, color="white")
        left += folded[:, k]
    axis.set_yticks(range(len(reforms)), [scenarios_data["scenarios"][f"{r}|none"]["label"] for r in reforms])
    axis.invert_yaxis()
    axis.set_xlim(0, 1)
    axis.set_xlabel("Share of bandh-day protesters")
    axis.set_title("Who would take part")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=5, fontsize=7.5)
    axis.grid(False)
    save_figure(figure, "v3_who")


def main():
    apply_house_style()
    calibration = load("v3_calibration.json")
    targets = json.loads((REPOSITORY_ROOT / "data" / "derived" / "episode_targets_v3.json").read_text())
    if calibration and calibration.get("kept"):
        episode_fit(calibration, targets)
        profiles(calibration, targets)
        identification(calibration)
    scenarios = load("v3_scenarios.json")
    if scenarios:
        map_2018(scenarios["scenarios"])
        scenario_chart(scenarios["scenarios"])
        scenario_maps(scenarios["scenarios"])
        seats(scenarios)
        who(scenarios)
    print("v3 figures written")


if __name__ == "__main__":
    main()
