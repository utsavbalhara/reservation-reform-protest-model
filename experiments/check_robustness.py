import argparse
import json
from pathlib import Path

from protest_simulation import build_shared_population, run_paired_monte_carlo
from protest_simulation.policy_interventions import MANAGED_TRANSITION, hybrid_with_weaker_symbolic_relief

RESULTS_FOLDER = Path(__file__).resolve().parent.parent / "results"
SYMBOLIC_THREAT_RETAINED_UNDER_HYBRID = (0.4, 0.6, 0.8)


def hybrid_sensitivity(population, run_count: int) -> list:
    checks = []
    for retained in SYMBOLIC_THREAT_RETAINED_UNDER_HYBRID:
        runs = run_paired_monte_carlo(population, (hybrid_with_weaker_symbolic_relief(retained),), run_count)
        checks.append({"test": f"Hybrid retains {retained:.0%} of symbolic threat", "symbolic_threat_retained": retained, **runs.summary()})
        print(checks[-1], flush=True)
    return checks


def managed_transition_leave_one_out(population, run_count: int) -> list:
    checks = []
    full_package = run_paired_monte_carlo(population, MANAGED_TRANSITION, run_count)
    checks.append({"test": "Managed transition, all five parts", "left_out": None, **full_package.summary()})
    print(checks[-1], flush=True)
    for left_out in MANAGED_TRANSITION:
        remaining = tuple(part for part in MANAGED_TRANSITION if part is not left_out)
        runs = run_paired_monte_carlo(population, remaining, run_count)
        checks.append({"test": f"Managed transition without {left_out.__name__}", "left_out": left_out.__name__, **runs.summary()})
        print(checks[-1], flush=True)
    return checks


def main():
    parser = argparse.ArgumentParser(description="Stress-test the two strongest findings.")
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--agents", type=int, default=120_000)
    arguments = parser.parse_args()

    population = build_shared_population(arguments.agents)
    robustness = {
        "run_count": arguments.runs,
        "hybrid_sensitivity": hybrid_sensitivity(population, arguments.runs),
        "managed_transition_leave_one_out": managed_transition_leave_one_out(population, arguments.runs),
    }
    RESULTS_FOLDER.mkdir(exist_ok=True)
    output_path = RESULTS_FOLDER / "robustness_checks.json"
    output_path.write_text(json.dumps(robustness, indent=1))
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
