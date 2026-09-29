# Usage

## Requirements

Python 3.10 or newer, plus NumPy, SciPy, pandas and Matplotlib.

```bash
pip install -r requirements.txt
```

Optional tools, only for the outputs that use them:

- **pdftotext** (poppler) extracts the JEE (Advanced) rank lists from the published reports.
- **ffmpeg** renders the simulation videos and GIF previews.
- **XeLaTeX** (TeX Live, with the TeX Gyre fonts) builds the paper and report.

Run every command from the repository root. Times are for a 4-core machine.

## Data

The derived data in `data/derived/` are committed, so the analyses run without downloading anything. To rebuild them:

| Command | What it does | Output |
|---|---|---|
| `python -m data_pipelines.jee_advanced_rank_lists` | Downloads the JIC reports and extracts rank and category for every candidate (no roll numbers or names) | `data/derived/jee_advanced_ranks_<year>.csv` |
| `python -m data_pipelines.gdelt_download` | Downloads GDELT 1.0 daily files for each episode window (about 2 GB, not committed) | `data/raw/gdelt/` |
| `python -m data_pipelines.gdelt_episode_events` | Filters protest events by episode and state | `data/derived/gdelt_episode_daily.csv`, `gdelt_episode_locations.csv` |
| `python -m data_pipelines.episode_targets` | Corrects counts by the audited precision and builds calibration targets | `data/derived/episode_targets.json` |
| `python -m data_pipelines.acled_episode_events` | Counts ACLED protest and riot events per episode from an ACLED export placed in `data/raw/acled/` (not committed; ACLED's terms forbid redistribution) | `data/derived/acled_episode_summary.json`, `acled_episode_daily.csv` |
| `python -m experiments.acled_cross_check` | Compares ACLED with GDELT, scores the frozen predictions against ACLED, and tests the 2018 turnout against ACLED's reported crowd sizes | `results/acled_cross_check.json` |

## Reproduce the results

| Command | What it does | Time | Output |
|---|---|---|---|
| `python -m experiments.eligibility_accounting` | Who changes eligibility under the real creamy-layer and EWS rules | ~10 s | `results/eligibility_accounting.json` |
| `python -m experiments.merged_pool_allocation --draws 300` | IIT seats by group under today's quotas, a merged pool, and the two allocation-rule levers | ~60 min | `results/merged_pool_allocation.json` |
| `python -m experiments.episode_calibration --wave-size 6000 --waves 3 --checkpoint-dir <dir>` | History matching to four episodes, and frozen leave-one-episode-out predictions | ~80 min | `results/episode_calibration.json`, `results/episode_calibration_nroy_samples.json`, `results/frozen_predictions/` |
| `python -m experiments.score_frozen_predictions` | Checks the hashes and scores the frozen predictions | seconds | `results/loeo_scores.json` |
| `python -m experiments.compare_interventions --specification grounded` | All 13 scenarios in the grounded model, 50 paired runs | ~3 min | `results/intervention_comparison_grounded_central.json` |
| `python -m experiments.compare_interventions --regime central` | Same, stylized model | ~3 min | `results/intervention_comparison_central.json` |
| `python -m experiments.compare_interventions --regime high-mobilization` | Stylized model, mean threshold lowered by 0.5 | ~3 min | `results/intervention_comparison_high-mobilization.json` |
| `python -m experiments.grounded_assumptions` | Grounded model under alternative shock size, party backing and material weight | ~15 min | `results/grounded_assumptions.json` |
| `python -m experiments.intervention_mapping_uncertainty --specification grounded` | Priors on every intervention effect; rank probabilities | ~10 min | `results/intervention_mapping_uncertainty_grounded.json` |
| `python -m experiments.intervention_mapping_uncertainty --specification stylized` | Same, stylized model | ~10 min | `results/intervention_mapping_uncertainty_stylized.json` |
| `python -m experiments.decompose_levers` | Splits levers into channels; separates suppression inputs from emergent effects | ~5 min | `results/lever_decomposition.json` |
| `python -m experiments.material_symbolic_break_even --specification stylized` | Sweeps the symbolic-to-material weight ratio with recalibration | ~20 min | `results/material_symbolic_break_even_stylized.json` |
| `python -m experiments.global_sensitivity` | Morris screening of 15 parameters | ~15 min | `results/global_sensitivity_stylized.json` |
| `python -m experiments.structural_ensemble` | Lever comparison under 11 structural variants, each recalibrated | ~45 min | `results/structural_ensemble.json` |
| `python -m experiments.grounded_channels` | Grounded model taken apart: lever channels, grievance composition, who protests by group, tier and state | ~5 min | `results/grounded_channels.json` |
| `python -m experiments.stylized_facts` | Checks the grounded model against the stylized facts S1–S5 (run after the two above) | ~1 min | `results/stylized_facts.json` |
| `python -m experiments.global_sensitivity --specification grounded` | Morris screening around the central calibrated values | ~10 min | `results/global_sensitivity_grounded.json` |
| `python -m experiments.record_daily_trajectories --specification grounded` | Day-by-day turnout for every scenario | ~1 min | `results/daily_trajectories_grounded_central.json` |
| `python -m experiments.sensitivity_analysis` | One-at-a-time sensitivity and agent-count convergence (stylized) | ~30 min | `results/sensitivity_analysis.json` |
| `python -m experiments.check_robustness` | L3's symbolic assumption and managed-transition leave-one-out (stylized) | ~4 min | `results/robustness_checks.json` |
| `python -m experiments.replay_model_development` | Re-runs iterations 1–3 of the stylized model's development | ~4 min | `results/model_development_history.json` |

`--checkpoint-dir` makes the calibration resumable: each finished wave is saved, and a rerun reloads it. The grounded specification needs `results/merged_pool_allocation.json` and `results/episode_calibration_nroy_samples.json`; without them only the stylized specification is available.

Most experiment scripts accept `--runs` and `--agents`. Fewer runs or agents are faster but noisier. Runs are spread across processes (set `PROTEST_WORKERS` to limit them) and give identical results to serial runs.

## Build figures, macros, documents and page

| Command | Output |
|---|---|
| `python -m visualization.render_grounded_figures` | `figures/fig10`–`fig15` |
| `python -m visualization.render_result_figures` | `figures/fig01`–`fig07` (stylized) |
| `python -m visualization.render_paper_figures` | `figures/fig08`, `fig09` |
| `python -m visualization.write_latex_result_macros` | `report/generated/`: scenario tables and paired-effect macros |
| `python -m visualization.write_paper_macros` | `paper/generated/`: development history, sensitivity and convergence |
| `python -m visualization.write_grounded_macros` | `paper/generated/`: eligibility, allocation, calibration, uncertainty and robustness macros and tables |
| `cd paper && xelatex reservation_reform_protest_paper.tex && xelatex reservation_reform_protest_paper.tex` | the paper |
| `cd report && xelatex research_report.tex && xelatex research_report.tex` | the report |
| `python -m visualization.build_results_page` | `artifact/v2/index.html` |
| `python -m visualization.render_campaign_videos` | `videos/` (stylized model) |

No result in the paper, report or page is typed by hand. After re-running any experiment, rebuild the macros and the documents.

## Tests

```bash
python -m pytest
```

The suite checks the reference model against stored outputs and an independent scalar reimplementation, model properties, the structural options, the choice rules, the rank-list extraction, the district population and the lever tables.

## Run one campaign yourself

```python
import numpy as np
from protest_simulation import SCENARIO_BY_KEY, simulate_protest_campaign
from protest_simulation.specifications import get_specification

specification = get_specification("grounded")          # or "stylized"
population = specification.build_population(120_000)
parameters = SCENARIO_BY_KEY["hybrid_caste_subquotas"].apply_to(specification.base_parameters)
outcome = simulate_protest_campaign(population, parameters, np.random.default_rng(0))

print(f"Peak day: {outcome.peak_day_protesters / 1e5:.1f} lakh")
print(f"Ever protested: {outcome.cumulative_unique_protesters / 1e7:.2f} crore")
print(f"Deaths: {outcome.total_deaths}, bandh days: {outcome.bandh_days}, conceded on day: {outcome.conceded_on_day}")
```

## Try your own intervention

An intervention is a function that edits a `ProtestModelParameters` in place:

```python
from protest_simulation import run_paired_monte_carlo
from protest_simulation.specifications import get_specification

def halve_opposition_mobilization(parameters):
    parameters.opposition_party_amplifier /= 2

specification = get_specification("grounded")
runs = run_paired_monte_carlo(specification.build_population(120_000), (halve_opposition_mobilization,), run_count=20,
                              base_parameters=specification.base_parameters, world_sampler=specification.world_sampler)
print(runs.summary())
```

Paired runs share their uncertain-parameter draws and random seeds with every other scenario, so differences come from the intervention, not from chance. Use `paired_effects_from_runs` in `protest_simulation/monte_carlo.py` for paired changes with bootstrap intervals.

## View the results page

Open `artifact/v2/index.html` in a browser from inside the repository; it loads figures and videos from `figures/` and `videos/`. `artifact/v1/index.html` is the first, results-only version of the original model.
