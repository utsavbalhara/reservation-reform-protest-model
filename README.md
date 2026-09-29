# Reservation Reform Protest Model

An agent-based simulation of street protest if India replaced caste-based reservation with a single income test (family income below ₹8 lakh). It estimates how many people would protest, and tests which policy designs and transition strategies would keep that number small.

## What it models

- **A synthetic India.** 120,000 agents stand in for 146 crore people across SC, ST, OBC and General groups. Each agent has an income position relative to the ₹8 lakh line, an identity strength, a personal protest threshold, and a neighbourhood of about 100 same-group agents. SC and ST agents are split into a better-off tier and a most-deprived tier.
- **Why people protest.** Grievance has two parts:
  - Material loss (losing eligibility, or seeing quota pools merged), weighted by loss aversion.
  - Symbolic threat (losing caste-based recognition), scaled by identity strength.
- **How protest spreads.** Each day an agent protests when grievance, pull from neighbours and national turnout, and party mobilization on bandh days together exceed its threshold. Protesting raises the threshold (fatigue). Deaths during unrest raise symbolic threat (the martyr effect).
- **What can change the outcome.** Nine interventions and two packages: grandfathering, seat expansion, a hybrid design that keeps caste sub-quotas, sub-classification, a consensus commission, compensation, guarantees, internet shutdowns and heavy policing.

## Headline results (central regime, 50 paired Monte Carlo runs)

| Code | Scenario | Peak-day protesters (median) | Change |
|---|---|---|---|
| Base | Abrupt ₹8L income-only switch | 46.1 lakh | — |
| L1 | Grandfather current cohorts + 10-year glide | 15.6 lakh | −66% |
| L2 | Expand seats so no group loses | 24.1 lakh | −48% |
| L3 | Keep caste sub-quotas, add income filter | 2.9 lakh | −94% |
| L4 | Sub-classify to favour most-deprived | 26.9 lakh | −42% |
| L5 | Data-first commission + cross-party consensus | 8.3 lakh | −82% |
| L6 | Compensate above-line losers | 34.9 lakh | −24% |
| L7 | Guarantee untouched protections | 18.2 lakh | −61% |
| B1 | Internet shutdowns | 42.8 lakh, deaths ×2.9 | −7% |
| B2 | Heavy policing and mass arrests | 32.9 lakh, deaths ×4.3 | −29% |
| C1 | Managed transition package | 2.2 lakh | −95% |
| C2 | Hybrid design package | 0.6 lakh | −99% |

The ranking is the finding; the magnitudes depend on assumed parameters. See [USAGE.md](USAGE.md) to reproduce every number.

## Repository layout

```
protest_simulation/
  synthetic_population.py    build_synthetic_india(): agents, groups, income line, neighbourhoods
  model_parameters.py        ProtestModelParameters and the calibrated baseline
  protest_campaign.py        simulate_protest_campaign(): the day-by-day protest dynamics
  parameter_uncertainty.py   draw_plausible_world(): one Monte Carlo draw of uncertain parameters
  policy_interventions.py    every intervention and the scenario catalogue
  monte_carlo.py             paired Monte Carlo runs and summaries
experiments/
  compare_interventions.py             all scenarios, one regime
  check_robustness.py                  hybrid sensitivity and package leave-one-out
  calibrate_participation_threshold.py turnout vs mean threshold, with the 2018 validation check
results/                     JSON output from the experiments
artifact/v1/index.html       interactive results page
```

## Scope

This project tests policy design, sequencing and coalition-building. Suppression tactics appear only to measure whether they backfire, and they do.
