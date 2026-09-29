# Usage

## Requirements

Python 3.10 or newer, plus NumPy and Matplotlib.

```bash
pip install -r requirements.txt
```

Two optional tools are needed only for the outputs that use them:

- **ffmpeg** renders the simulation videos and GIF previews.
- **XeLaTeX** (TeX Live, with the TeX Gyre fonts) builds the PDF report.

Run every command from the repository root.

## Reproduce the results

| Command | What it does | Time | Output |
|---|---|---|---|
| `python -m experiments.compare_interventions --regime central` | All 13 scenarios, 50 runs each | ~5 min | `results/intervention_comparison_central.json` |
| `python -m experiments.compare_interventions --regime high-mobilization` | Same, with the mean threshold lowered by 0.5 | ~5 min | `results/intervention_comparison_high-mobilization.json` |
| `python -m experiments.check_robustness` | Hybrid sensitivity and managed-transition leave-one-out, 30 runs each | ~4 min | `results/robustness_checks.json` |
| `python -m experiments.calibrate_participation_threshold` | Turnout against mean threshold, baseline and symbolic-only validation | ~3 min | `results/threshold_response_curve.json` |
| `python -m experiments.record_daily_trajectories` | Day-by-day turnout for every scenario, 20 runs each | ~2 min | `results/daily_trajectories_central.json` |
| `python -m experiments.replay_model_development` | Re-runs iterations 1–3 of the model's development with the current code | ~4 min | `results/model_development_history.json` |
| `python -m experiments.sensitivity_analysis` | ±25% one-at-a-time sensitivity, lever-ranking stability, agent-count convergence | ~30 min | `results/sensitivity_analysis.json` |

## Build figures, videos, report and page

All of these read only from `results/`, so they run in seconds to minutes and never re-run the model except the videos.

| Command | Output |
|---|---|
| `python -m visualization.render_result_figures` | `figures/fig01`–`fig07`, PNG and PDF |
| `python -m visualization.render_campaign_videos` | `videos/baseline_campaign.mp4`, `videos/four_policy_paths.mp4` and GIF previews (re-runs the representative simulation live, ~2 min) |
| `python -m visualization.write_latex_result_macros` | `report/generated/`: every number the report quotes |
| `cd report && xelatex research_report.tex && xelatex research_report.tex` | `report/research_report.pdf` (run twice for the contents page) |
| `xelatex mechanism_diagram.tex` in `report/`, then `pdftoppm -r 220 -png -singlefile mechanism_diagram.pdf ../figures/fig00_mechanism` | the model diagram as `figures/fig00_mechanism.png` |
| `python -m visualization.build_results_page` | `artifact/v2/index.html` with results injected from `results/` |
| `python -m visualization.render_paper_figures` | `figures/fig08_model_development`, `figures/fig09_sensitivity_tornado` |
| `python -m visualization.write_paper_macros` | `paper/generated/`: development-history, sensitivity and convergence numbers |
| `cd paper && xelatex reservation_reform_protest_paper.tex && xelatex reservation_reform_protest_paper.tex` | `paper/reservation_reform_protest_paper.pdf` |

The report never contains hand-typed results: tables and numbers come from `report/generated/`. After re-running any experiment, rebuild the macros and the report.

Every experiment script accepts `--runs` and `--agents`. Fewer runs or agents are faster but noisier:

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

## View the results pages

Open `artifact/v2/index.html` in a browser from inside the repository; it loads figures and videos from `figures/` and `videos/`. `artifact/v1/index.html` is the first, results-only version and is fully self-contained.
