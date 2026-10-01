# Protest model, version 3

Version 3 rebuilds the protest model around what the data can support. The grounded specification in `protest_simulation/` is kept unchanged, so the existing paper still reproduces.

## What changed and why

| | Grounded specification (v2) | Version 3 |
|---|---|---|
| Episodes used for calibration | 4 (two-day campaigns) | 10, including six multi-day state agitations (Patidar 2015, Jat 2016, Kapu 2016, Maratha 2017 and 2018, Gujjar 2019) |
| What each episode is matched on | Event count, deaths | Event count, day-by-day profile, distribution across states, deaths, and published crowd sizes where they exist |
| Dynamics after the first day | Not fitted | Fitted on agitations of up to six consecutive days |
| Who can mobilize | SC, ST, OBC, General | The same, plus state communities that mobilized for reservation (Jats, Patidars, Kapus, Marathas, Gujjars) |
| Geography | District demography only | Also district urbanization, literacy and household phone ownership (Census 2011), with effects estimated from where events happened |
| Concession | Built in, assumed | Optional, off by default; no episode identifies it |
| Main output | Headcounts (crore) | Protest relative to the 2 April 2018 bandh, on the same parameter set |
| Scenarios | Levers on one reform | Five reform designs, including abolition of all reservation, each tied to its nearest episode, with phasing, negotiation and guarantees as modifiers |

## Files

- `population.py`: district population, Census covariates and state communities.
- `model.py`: the daily simulator and the observation model.
- `episodes.py`: the ten episodes as shocks.
- `scenarios.py`: the reform designs and modifiers.
- `experiments/v3_history_match.py`: calibration to the ten episodes, plus a spatial out-of-sample test.
- `experiments/v3_scenarios.py`: the reform scenarios, reported relative to the 2018 bandh.
- `visualization/v3_figures.py`: figures `figures/v3_*`.

Data pipelines:
- `data_pipelines/district_covariates.py`: Census covariates and the district map.
- `data_pipelines/episode_targets_v3.py`: the episode targets.

## What it still cannot do

- It does not forecast headcounts. Results are relative to 2018, which is what the event data identify.
- Each scenario's threat is tied to its nearest episode where one exists. Income-only and abolition have no close episode, so their threat is a wide prior.
- The state communities' population shares are commonly cited estimates; there has been no caste census since 1931.
- The relevance labels behind the event counts were assigned by a language model from article URLs and have not been checked by hand.
