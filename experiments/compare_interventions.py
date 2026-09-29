import argparse
import json
import time
from pathlib import Path

from protest_simulation import MOBILIZATION_REGIMES, SCENARIOS, build_shared_population, run_paired_monte_carlo

RESULTS_FOLDER = Path(__file__).resolve().parent.parent / "results"


def compare_all_scenarios(regime: str, run_count: int, agent_count: int) -> dict:
    population = build_shared_population(agent_count)
    threshold_shift = MOBILIZATION_REGIMES[regime]
    comparison = {"regime": regime, "run_count": run_count, "agent_count": agent_count, "scenarios": {}}
    started = time.time()
    for scenario in SCENARIOS:
        runs = run_paired_monte_carlo(population, scenario.interventions, run_count, threshold_shift)
        summary = runs.summary()
        comparison["scenarios"][scenario.key] = {
            "code": scenario.code,
            "label": scenario.label,
            "category": scenario.category,
            "summary": summary,
            "runs": {
                "peak_day_protesters": runs.peak_day_protesters,
                "cumulative_unique_protesters": runs.cumulative_unique_protesters,
                "total_deaths": runs.total_deaths,
            },
        }
        print(
            f"{scenario.code:>4}  {scenario.label:<52} peak {summary['median_peak_lakh']:>7.1f} lakh "
            f"[{summary['peak_lakh_10th_percentile']}–{summary['peak_lakh_90th_percentile']}]  "
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
    arguments = parser.parse_args()

    comparison = compare_all_scenarios(arguments.regime, arguments.runs, arguments.agents)
    RESULTS_FOLDER.mkdir(exist_ok=True)
    output_path = RESULTS_FOLDER / f"intervention_comparison_{arguments.regime}.json"
    output_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=1))
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
