import argparse
import json
import time
from pathlib import Path

from protest_simulation import MOBILIZATION_REGIMES, SCENARIOS, run_paired_monte_carlo
from protest_simulation.specifications import get_specification

RESULTS_FOLDER = Path(__file__).resolve().parent.parent / "results"


def compare_all_scenarios(regime: str, run_count: int, agent_count: int, specification_name: str = "stylized") -> dict:
    specification = get_specification(specification_name)
    population = specification.build_population(agent_count)
    threshold_shift = MOBILIZATION_REGIMES[regime]
    comparison = {"specification": specification_name, "regime": regime, "run_count": run_count, "agent_count": agent_count,
                  "scenarios": {}}
    started = time.time()
    baseline_runs = None
    for scenario in SCENARIOS:
        runs = run_paired_monte_carlo(population, scenario.interventions, run_count, threshold_shift,
                                      base_parameters=specification.base_parameters, world_sampler=specification.world_sampler)
        summary = runs.summary()
        if scenario.key == "baseline":
            baseline_runs = runs
        paired = runs.paired_effects(baseline_runs) if baseline_runs is not None else None
        comparison["scenarios"][scenario.key] = {
            "code": scenario.code,
            "label": scenario.label,
            "category": scenario.category,
            "summary": summary,
            "paired_effects_vs_baseline": paired,
            "runs": {
                "peak_day_protesters": runs.peak_day_protesters,
                "cumulative_unique_protesters": runs.cumulative_unique_protesters,
                "total_deaths": runs.total_deaths,
                "bandh_days_called": runs.bandh_days_called,
                "conceded_on_day": runs.conceded_on_day,
                "campaign_days_run": runs.campaign_days_run,
            },
        }
        change = "" if paired is None else f"paired change {paired['peak']['change_percent']:>6.1f}% [{paired['peak']['change_percent_ci95'][0]}, {paired['peak']['change_percent_ci95'][1]}]  "
        print(
            f"{scenario.code:>4}  {scenario.label:<52} peak {summary['median_peak_lakh']:>7.1f} lakh "
            f"[{summary['peak_lakh_10th_percentile']}–{summary['peak_lakh_90th_percentile']}]  {change}"
            f"cumulative {summary['median_cumulative_crore']:>5.2f} crore  deaths {summary['median_deaths']:>6.1f}  "
            f"({time.time() - started:.0f}s)",
            flush=True,
        )
    return comparison


def main():
    parser = argparse.ArgumentParser(description="Compare every intervention against the abrupt income-only switch.")
    parser.add_argument("--regime", choices=sorted(MOBILIZATION_REGIMES), default="central")
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--agents", type=int, default=120_000)
    parser.add_argument("--specification", default="stylized")
    arguments = parser.parse_args()

    comparison = compare_all_scenarios(arguments.regime, arguments.runs, arguments.agents, arguments.specification)
    RESULTS_FOLDER.mkdir(exist_ok=True)
    prefix = "" if arguments.specification == "stylized" else f"{arguments.specification}_"
    output_path = RESULTS_FOLDER / f"intervention_comparison_{prefix}{arguments.regime}.json"
    output_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=1))
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
