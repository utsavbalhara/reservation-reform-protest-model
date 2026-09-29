import argparse
import json
import subprocess

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import PowerNorm
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from protest_simulation import BASELINE_ABRUPT_INCOME_ONLY_SWITCH, SCENARIO_BY_KEY, SOCIAL_GROUP_NAMES, build_shared_population, simulate_protest_campaign
from protest_simulation.monte_carlo import CAMPAIGN_SEED_OFFSET, WORLD_DRAW_SEED_OFFSET
from protest_simulation.parameter_uncertainty import draw_plausible_world

from .figure_style import (
    CATEGORY_COLOURS,
    GROUP_COLOURS,
    HIGHLIGHT_SCENARIO_COLOURS,
    INK,
    MUTED_INK,
    RESULTS_FOLDER,
    SECONDARY_INK,
    SURFACE,
    TURNOUT_COLOUR_MAP,
    VIDEOS_FOLDER,
    apply_house_style,
)

FRAMES_PER_SECOND = 24
FRAMES_PER_DAY = 6
HOLD_FRAMES_AT_END = 48
MAP_COLUMNS = 42
TURNOUT_COLOUR_NORM = PowerNorm(gamma=0.55, vmin=0, vmax=0.12)
COMPARISON_SCENARIOS = ("baseline", "heavy_policing", "consensus_commission", "hybrid_caste_subquotas")
SHORT_PANEL_TITLES = {
    "baseline": "Base · abrupt switch",
    "heavy_policing": "B2 · heavy policing",
    "consensus_commission": "L5 · consensus commission",
    "hybrid_caste_subquotas": "L3 · caste sub-quotas kept",
}
VIDEO_TEXT_SIZES = {"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 12.5, "xtick.labelsize": 11.5, "ytick.labelsize": 11.5, "legend.fontsize": 12}


def format_people(count: float) -> str:
    if count >= 1e5:
        return f"{count / 1e5:,.1f} lakh"
    return f"{int(round(count, -2)):,}"



def representative_run_index() -> int:
    baseline_peaks = np.array(json.loads((RESULTS_FOLDER / "intervention_comparison_central.json").read_text())
                              ["scenarios"]["baseline"]["runs"]["peak_day_protesters"])
    return int(np.argmin(np.abs(baseline_peaks - np.median(baseline_peaks))))


def simulate_recorded_campaign(population, scenario_key: str, run_index: int):
    world = draw_plausible_world(BASELINE_ABRUPT_INCOME_ONLY_SWITCH, np.random.default_rng(WORLD_DRAW_SEED_OFFSET + run_index))
    for intervention in SCENARIO_BY_KEY[scenario_key].interventions:
        intervention(world)
    outcome = simulate_protest_campaign(population, world, np.random.default_rng(CAMPAIGN_SEED_OFFSET + run_index),
                                        record_neighbourhood_turnout=True)
    return outcome, world


def neighbourhood_map_layout(population):
    group_of_neighbourhood = np.zeros(population.neighbourhood_count, int)
    group_of_neighbourhood[population.neighbourhood] = population.social_group
    cell_row, cell_column, group_label_rows = {}, {}, []
    current_row = 0
    for group_index, group_name in enumerate(SOCIAL_GROUP_NAMES):
        members = np.where(group_of_neighbourhood == group_index)[0]
        group_label_rows.append((current_row, group_name, len(members)))
        for position, neighbourhood_id in enumerate(members):
            cell_row[neighbourhood_id] = current_row + position // MAP_COLUMNS
            cell_column[neighbourhood_id] = position % MAP_COLUMNS
        current_row += int(np.ceil(len(members) / MAP_COLUMNS)) + 1
    rows = np.array([cell_row[i] for i in range(population.neighbourhood_count)])
    columns = np.array([cell_column[i] for i in range(population.neighbourhood_count)])
    return rows, columns, current_row - 1, group_label_rows


def turnout_image(neighbourhood_turnout, rows, columns, row_count):
    image = np.full((row_count, MAP_COLUMNS), np.nan)
    image[rows, columns] = neighbourhood_turnout
    return image


def interpolated_turnout(daily_turnout, frame):
    day, step = divmod(frame, FRAMES_PER_DAY)
    last_day = len(daily_turnout) - 1
    if day >= last_day:
        return daily_turnout[last_day], last_day
    fraction = step / FRAMES_PER_DAY
    previous = daily_turnout[day - 1] if day > 0 else np.zeros_like(daily_turnout[0])
    return previous + (daily_turnout[day] - previous) * fraction, day


def draw_neighbourhood_map(axis, rows, columns, row_count, group_label_rows, label_groups=True):
    colour_map = TURNOUT_COLOUR_MAP.copy()
    colour_map.set_bad(SURFACE)
    image = axis.imshow(turnout_image(np.zeros(len(rows)), rows, columns, row_count), cmap=colour_map, norm=TURNOUT_COLOUR_NORM,
                        interpolation="nearest", aspect="equal")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_visible(False)
    if label_groups:
        for start_row, group_name, count in group_label_rows:
            axis.text(-1.2, start_row, f"{group_name}\n{count} areas", ha="right", va="top", fontsize=9, color=SECONDARY_INK)
    return image


def mark_bandh_days(axis, parameters):
    for bandh_day in parameters.bandh_call_days:
        axis.axvspan(bandh_day - 0.5, bandh_day + 0.5, color="#ecebe7", zorder=0, linewidth=0)


def render_single_campaign_video(population, run_index: int, output_stem: str):
    outcome, world = simulate_recorded_campaign(population, "baseline", run_index)
    rows, columns, row_count, group_label_rows = neighbourhood_map_layout(population)
    days = len(outcome.daily_protesters)
    by_group_lakh = outcome.daily_protesters_by_group / 1e5
    cumulative_deaths = np.cumsum(outcome.daily_deaths)
    ever_protested_estimate = None

    figure = plt.figure(figsize=(16, 9), dpi=100)
    grid = figure.add_gridspec(2, 2, width_ratios=[1.05, 1], height_ratios=[0.9, 1.1], wspace=0.12, hspace=0.3,
                               left=0.07, right=0.97, top=0.88, bottom=0.08)
    map_axis = figure.add_subplot(grid[:, 0])
    counter_axis = figure.add_subplot(grid[0, 1])
    line_axis = figure.add_subplot(grid[1, 1])
    figure.text(0.07, 0.95, "Abrupt ₹8 lakh income-only switch: one simulated protest campaign", fontsize=18, fontweight="bold", color=INK)
    figure.text(0.07, 0.915, "Each square is a neighbourhood of about 12 lakh people; colour shows the share on the street that day. "
                "Run closest to the 50-run median.", fontsize=11, color=SECONDARY_INK)

    image = draw_neighbourhood_map(map_axis, rows, columns, row_count, group_label_rows)
    colour_bar = figure.colorbar(image, ax=map_axis, orientation="horizontal", fraction=0.035, pad=0.02, ticks=[0, 0.01, 0.03, 0.06, 0.12])
    colour_bar.ax.set_xticklabels(["0%", "1%", "3%", "6%", "12%+"])
    colour_bar.set_label("Share of the neighbourhood protesting", color=SECONDARY_INK)
    colour_bar.outline.set_visible(False)

    counter_axis.axis("off")
    day_text = counter_axis.text(0, 0.95, "", fontsize=30, fontweight="bold", va="top", color=INK)
    bandh_text = counter_axis.text(0.52, 0.93, "", fontsize=16, va="top", color=CATEGORY_COLOURS["suppression"], fontweight="bold")
    stat_texts = [counter_axis.text(0, 0.55 - 0.22 * i, "", fontsize=16, va="top", color=INK) for i in range(3)]

    mark_bandh_days(line_axis, world)
    group_lines = {}
    for group_index, group_name in enumerate(SOCIAL_GROUP_NAMES):
        (group_lines[group_name],) = line_axis.plot([], [], color=GROUP_COLOURS[group_name], linewidth=2.4, label=group_name)
    line_axis.set_xlim(0, days - 1)
    line_axis.set_ylim(0, by_group_lakh.max() * 1.15)
    line_axis.set_xlabel("Day of campaign")
    line_axis.set_ylabel("Protesters that day (lakh)")
    line_axis.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, borderaxespad=0.2)
    line_axis.set_title("Daily turnout by group (shaded columns: bandh calls)", pad=34)

    total_frames = days * FRAMES_PER_DAY + HOLD_FRAMES_AT_END
    writer = FFMpegWriter(fps=FRAMES_PER_SECOND, bitrate=5000, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    VIDEOS_FOLDER.mkdir(exist_ok=True)
    with writer.saving(figure, str(VIDEOS_FOLDER / f"{output_stem}.mp4"), dpi=100):
        for frame in range(total_frames):
            turnout, day = interpolated_turnout(outcome.daily_neighbourhood_turnout, frame)
            image.set_data(turnout_image(turnout, rows, columns, row_count))
            day_text.set_text(f"Day {day + 1}")
            bandh_text.set_text("BANDH CALL" if day in world.bandh_call_days else "")
            stat_texts[0].set_text(f"On the street today:  {format_people(outcome.daily_protesters[day])}")
            stat_texts[1].set_text(f"Peak so far:  {format_people(outcome.daily_protesters[: day + 1].max())}")
            stat_texts[2].set_text(f"Deaths so far:  {cumulative_deaths[day]}")
            for group_index, group_name in enumerate(SOCIAL_GROUP_NAMES):
                group_lines[group_name].set_data(np.arange(day + 1), by_group_lakh[: day + 1, group_index])
            writer.grab_frame()
    plt.close(figure)
    return outcome


def render_scenario_comparison_video(population, run_index: int, output_stem: str):
    campaigns = {key: simulate_recorded_campaign(population, key, run_index) for key in COMPARISON_SCENARIOS}
    rows, columns, row_count, group_label_rows = neighbourhood_map_layout(population)
    days = len(campaigns["baseline"][0].daily_protesters)

    figure = plt.figure(figsize=(16, 9), dpi=100)
    grid = figure.add_gridspec(2, 4, height_ratios=[1.25, 1], wspace=0.08, hspace=0.5, left=0.07, right=0.98, top=0.85, bottom=0.08)
    figure.text(0.06, 0.945, "Same country, same run, four policy paths", fontsize=18, fontweight="bold", color=INK)
    figure.text(0.06, 0.91, "Identical population and random draws; only the policy differs. Colour: share of each neighbourhood protesting.",
                fontsize=11, color=SECONDARY_INK)
    images, counters = {}, {}
    for column_index, key in enumerate(COMPARISON_SCENARIOS):
        axis = figure.add_subplot(grid[0, column_index])
        images[key] = draw_neighbourhood_map(axis, rows, columns, row_count, group_label_rows, label_groups=column_index == 0)
        scenario = SCENARIO_BY_KEY[key]
        axis.set_title(SHORT_PANEL_TITLES[key], fontsize=14, color=HIGHLIGHT_SCENARIO_COLOURS[key], loc="left")
        counters[key] = axis.text(0, row_count + 1.5, "", fontsize=12.5, va="top", color=INK)

    line_axis = figure.add_subplot(grid[1, :])
    mark_bandh_days(line_axis, campaigns["baseline"][1])
    lines = {}
    for key in COMPARISON_SCENARIOS:
        (lines[key],) = line_axis.plot([], [], color=HIGHLIGHT_SCENARIO_COLOURS[key], linewidth=2.6, label=SHORT_PANEL_TITLES[key])
    line_axis.set_yscale("log")
    line_axis.set_ylim(0.05, max(c[0].daily_protesters.max() for c in campaigns.values()) / 1e5 * 2)
    line_axis.set_xlim(0, days - 1)
    line_axis.set_xlabel("Day of campaign")
    line_axis.set_ylabel("Protesters that day (log scale)")
    line_axis.yaxis.set_major_locator(FixedLocator([0.1, 1, 10, 100]))
    line_axis.yaxis.set_minor_locator(NullLocator())
    line_axis.yaxis.set_major_formatter(FuncFormatter(lambda lakh, _: {0.1: "10,000", 1: "1 lakh", 10: "10 lakh", 100: "1 crore"}.get(round(lakh, 1), "")))
    line_axis.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, borderaxespad=0.2)
    day_text = figure.text(0.98, 0.945, "", fontsize=18, fontweight="bold", ha="right", color=INK)

    total_frames = days * FRAMES_PER_DAY + HOLD_FRAMES_AT_END
    writer = FFMpegWriter(fps=FRAMES_PER_SECOND, bitrate=5000, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    with writer.saving(figure, str(VIDEOS_FOLDER / f"{output_stem}.mp4"), dpi=100):
        for frame in range(total_frames):
            for key, (outcome, world) in campaigns.items():
                turnout, day = interpolated_turnout(outcome.daily_neighbourhood_turnout, frame)
                images[key].set_data(turnout_image(turnout, rows, columns, row_count))
                deaths = int(np.sum(outcome.daily_deaths[: day + 1]))
                counters[key].set_text(f"Today {format_people(outcome.daily_protesters[day])}\n"
                                       f"Peak {format_people(outcome.daily_protesters[: day + 1].max())} · deaths {deaths}")
                lines[key].set_data(np.arange(day + 1), np.maximum(outcome.daily_protesters[: day + 1] / 1e5, 0.05))
            day_text.set_text(f"Day {day + 1}" + ("   · bandh call" if day in world.bandh_call_days else ""))
            writer.grab_frame()
    plt.close(figure)
    return {key: campaign[0] for key, campaign in campaigns.items()}


def make_gif_preview(output_stem: str, width: int = 960):
    source = VIDEOS_FOLDER / f"{output_stem}.mp4"
    target = VIDEOS_FOLDER / f"{output_stem}_preview.gif"
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(source),
        "-vf", f"fps=10,scale={width}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer",
        str(target),
    ], check=True)


def main():
    parser = argparse.ArgumentParser(description="Render day-by-day simulation videos straight from the model.")
    parser.add_argument("--agents", type=int, default=120_000)
    arguments = parser.parse_args()

    apply_house_style()
    plt.rcParams.update(VIDEO_TEXT_SIZES)
    population = build_shared_population(arguments.agents)
    run_index = representative_run_index()
    print(f"Representative run: {run_index}")
    outcome = render_single_campaign_video(population, run_index, "baseline_campaign")
    print(f"baseline_campaign.mp4: peak {outcome.peak_day_protesters / 1e5:.1f} lakh, deaths {outcome.total_deaths}")
    outcomes = render_scenario_comparison_video(population, run_index, "four_policy_paths")
    for key, scenario_outcome in outcomes.items():
        print(f"four_policy_paths.mp4 {key}: peak {scenario_outcome.peak_day_protesters / 1e5:.1f} lakh, deaths {scenario_outcome.total_deaths}")
    for stem in ("baseline_campaign", "four_policy_paths"):
        make_gif_preview(stem)
    print("Videos saved")


if __name__ == "__main__":
    main()
