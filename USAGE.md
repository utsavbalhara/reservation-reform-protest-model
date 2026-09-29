# Usage

## Requirements

Python 3.10 or newer, plus NumPy.

```bash
pip install -r requirements.txt
```

Run every command from the repository root.

## Reproduce the results

| Command | What it does | Time | Output |
|---|---|---|---|
| `python -m experiments.compare_interventions --regime central` | All 13 scenarios, 50 runs each | ~5 min | `results/intervention_comparison_central.json` |
| `python -m experiments.compare_interventions --regime high-mobilization` | Same, with the mean threshold lowered by 0.5 | ~5 min | `results/intervention_comparison_high-mobilization.json` |
| `python -m experiments.check_robustness` | Hybrid sensitivity and managed-transition leave-one-out, 30 runs each | ~4 min | `results/robustness_checks.json` |
| `python -m experiments.calibrate_participation_threshold` | Turnout against mean threshold, baseline and symbolic-only validation | ~8 min | `results/threshold_response_curve.json` |

Every script accepts `--runs` and `--agents`. Fewer runs or agents are faster but noisier:

```bash
python -m experiments.compare_interventions --regime central --runs 10 --agents 40000
```

Seeds are fixed (population seed 7, world draws 1000 + run, campaigns 5000 + run). The same command gives the same numbers on any machine with the same NumPy version.

## Run one campaign yourself

```python
import numpy as np
from protest_simulation import BASELINE_ABRUPT_INCOME_ONLY_SWITCH, SCENARIO_BY_KEY, build_shared_population, simulate_protest_campaign

population = build_shared_population()
parameters = SCENARIO_BY_KEY["hybrid_caste_subquotas"].apply_to(BASELINE_ABRUPT_INCOME_ONLY_SWITCH)
outcome = simulate_protest_campaign(population, parameters, np.random.default_rng(0))

print(f"Peak day: {outcome.peak_day_protesters / 1e5:.1f} lakh")
print(f"Ever protested: {outcome.cumulative_unique_protesters / 1e7:.2f} crore")
print(f"Deaths: {outcome.total_deaths}")
```

`outcome.daily_protesters_by_group` gives turnout per day for SC, ST, OBC and General. Pass `record_neighbourhood_turnout=True` to also keep every neighbourhood's daily turnout.

## Try your own intervention

An intervention is a function that edits a `ProtestModelParameters` in place:

```python
from protest_simulation import build_shared_population, run_paired_monte_carlo

def halve_opposition_mobilization(parameters):
    parameters.opposition_party_amplifier /= 2

runs = run_paired_monte_carlo(build_shared_population(), (halve_opposition_mobilization,), run_count=20)
print(runs.summary())
```

Paired runs share the same uncertain-parameter draws as every other scenario, so differences come from the intervention, not from chance.

## Key parameters

| Parameter | Baseline | Meaning |
|---|---|---|
| `loss_aversion` | 2.25 | How much more a loss weighs than an equal gain |
| `material_loss_weight` | 0.45 | Weight of material loss in grievance |
| `symbolic_threat_weight` | 3.3 | Weight of symbolic threat × identity strength |
| `symbolic_threat_by_group` | SC 1.0, ST 0.9, OBC 0.45, General −0.4 | Perceived threat of abolishing caste quotas |
| `opposition_party_amplifier` | 1.5 | Multiplier on organizational capacity when parties join |
| `mean_participation_threshold` | 6.6 | Calibrated so the baseline peak is about 46 lakh |
| `participation_threshold_spread` | 1.4 | Wide spread: a tail of low-threshold activists |
| `deaths_per_crore_protester_days` | 4.0 | Violence rate |

## View the results page

Open `artifact/v1/index.html` in a browser. It is self-contained apart from Google Fonts.
