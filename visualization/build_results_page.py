import argparse
import json
from pathlib import Path

from .figure_style import REPOSITORY_ROOT, RESULTS_FOLDER

PAGE_FOLDER = REPOSITORY_ROOT / "artifact" / "v2"
REPOSITORY_ASSET_PREFIX = "../../"
READABLE_LEFT_OUT_NAMES = {
    None: "Managed transition, all five parts",
    "grandfather_current_cohorts_with_ten_year_glide": "Managed transition without grandfathering",
    "expand_seats_so_no_group_loses": "Managed transition without seat expansion",
    "build_consensus_through_data_first_commission": "Managed transition without consensus commission",
    "compensate_above_line_losers": "Managed transition without compensation",
    "guarantee_untouched_protections": "Managed transition without guarantees",
}


def compact_regime(regime: str) -> dict:
    scenarios = json.loads((RESULTS_FOLDER / f"intervention_comparison_{regime}.json").read_text())["scenarios"]
    return {
        key: {
            "code": scenario["code"],
            "label": scenario["label"],
            "category": scenario["category"],
            "median": scenario["summary"]["median_peak_lakh"],
            "p10": scenario["summary"]["peak_lakh_10th_percentile"],
            "p90": scenario["summary"]["peak_lakh_90th_percentile"],
            "cumulative": scenario["summary"]["median_cumulative_crore"],
            "deaths": scenario["summary"]["median_deaths"],
        }
        for key, scenario in scenarios.items()
    }


def page_data() -> dict:
    trajectories = json.loads((RESULTS_FOLDER / "daily_trajectories_central.json").read_text())
    robustness = json.loads((RESULTS_FOLDER / "robustness_checks.json").read_text())
    return {
        "regimes": {regime: compact_regime(regime) for regime in ("central", "high-mobilization")},
        "dailyPaths": {key: path["median_daily_lakh"] for key, path in trajectories["scenarios"].items()},
        "bandhDays": trajectories["bandh_call_days"],
        "robustness": {
            "hybrid": [{"retained": check["symbolic_threat_retained"], "median": check["median_peak_lakh"], "cumulative": check["median_cumulative_crore"]}
                       for check in robustness["hybrid_sensitivity"]],
            "leaveOneOut": [{"label": READABLE_LEFT_OUT_NAMES[check["left_out"]], "median": check["median_peak_lakh"], "cumulative": check["median_cumulative_crore"]}
                            for check in robustness["managed_transition_leave_one_out"]],
        },
        "explanations": json.loads((PAGE_FOLDER / "scenario_explanations.json").read_text()),
    }


def build_page(asset_prefix: str, output_path: Path):
    template = (PAGE_FOLDER / "page_template.html").read_text()
    page = template.replace("__RESULTS_DATA__", json.dumps(page_data(), ensure_ascii=False, separators=(",", ":")))
    page = page.replace("__ASSET_PREFIX__", asset_prefix)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page)
    print(f"Wrote {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Build the v2 results page from the result files.")
    parser.add_argument("--publish-copy", type=Path, help="Also write a copy whose assets sit next to it, for publishing.")
    arguments = parser.parse_args()
    build_page(REPOSITORY_ASSET_PREFIX, PAGE_FOLDER / "index.html")
    if arguments.publish_copy:
        build_page("", arguments.publish_copy)


if __name__ == "__main__":
    main()
