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
| Who can protest | Anyone whose threshold is crossed | Only people with a stake (a threat to their group or a material loss) who have heard of the protest |
| Build-up over days | None | Awareness spreads from those who heard the call to those who see protest nearby or in their group |
| Deaths | Proportional to protester-days | Also rise with how concentrated protest is in a state; the martyr effect is estimated |
| Media coverage | One extra factor for Delhi | Each state's reporting intensity, measured from GDELT events of every kind per head |
| Concession | Built in, assumed | Optional, off by default; no episode identifies it |
| Main output | Headcounts (crore) | Protest relative to the 2 April 2018 bandh, on the same parameter set |
| Scenarios | Levers on one reform | Five reform designs, including abolition of all reservation, each tied to its nearest episode, with phasing, negotiation and guarantees as modifiers |

## How a day works

Each person in the synthetic population (120,000 agents standing for India's 121 crore, placed in Census 2011 districts) decides each day whether to protest:

1. **Stake.** Only people whose group is threatened by the reform, or who lose materially, can take part. The first calibration did not have this rule, and it put 20 to 40 lakh unaffected people on the street in every episode. That floor squeezed every episode toward the same size and spread protest by population.
2. **Awareness.** On the announcement day a share of the stakeholders hears the call. The rest can hear of it later, by seeing protest in their neighbourhood or among their own group. This is how an agitation can grow over several days instead of peaking on day one.
3. **Decision.** An aware stakeholder protests if grievance (threat and material loss), social pull (turnout nearby and in the group) and organizational push (stronger with party backing, and on called action days) outweigh a personal threshold. Each day of protest raises the threshold (fatigue).
4. **Deaths.** Deaths are drawn from the protester-days, weighted toward states where a large share of people are on the street, and each death raises the threat felt by the groups protesting (the martyr effect).
5. **News.** Expected news events in a state are a fitted function of its turnout, multiplied by how heavily GDELT reports that state on any topic. This lets the model say how many events GDELT would record, which is what the calibration targets are.

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

## Results

### Calibration

I simulated 40,000 parameter sets in five waves: the first drawn from the priors, the rest by perturbing the best sets so far. A set is kept if its second-largest implausibility is below 3 and its largest below 4, and if it meets the two published crowd ranges and the 2018 turnout bound exactly. 909 sets pass. With a cutoff of 3 on every one of the 34 measures, almost nothing passes: each target alone rules out 10 to 30 per cent of otherwise good sets.

![Fit to ten episodes](../figures/v3_episode_fit.png)

- **Levels.** For seven of the ten episodes, the observed level is inside the model's 90% range. The three misses are the largest and the two smallest. The model puts the 2018 bandh at 9 events per 1,000 (90% range 5 to 21) against 31 observed, and the Kapu and Gujjar agitations at about three times what was reported. The model compresses the differences between episodes.
- **Turnout.** The retained sets put 2 April 2018 at 47 lakh people on the street (90% range 11 to 94 lakh). That range comes mostly from the bound and the crowd ranges; the event counts do not pin down headcounts.
- **Deaths.** For 2018 the model gives 7 (2 to 28) against 11 to 14 reported, and for the Jat agitation 9 (3 to 31) against 30. For the episodes with no deaths it gives medians of 1 to 8. One death rate per protester-day cannot separate violent agitations from peaceful ones.
- **Day profiles.** The model gets the Maratha 2018 profile roughly right, including the decline on day three, and the Patidar profile within tolerance. The Jat and Gujjar agitations peaked on their third day, and the model, which has fatigue but no escalation, stays flat for them.

![Day profiles](../figures/v3_profiles.png)
- **What is identified.** Five quantities have 90% ranges under half their prior width: the observation scale, the 2018 threat, the spread of thresholds, the death rate, and the Patidar threat. Social influence, fatigue, neighbourhood mixing and the three Census covariates all keep at least 85 per cent of their prior range. Relative to 2018, the fitted threats are about 1.5 for the Patidar and Jat agitations, 0.6 for the upper-caste bandh and the 2024 bandh, and 0.4 for the EWS amendment.
- **Where protest happens.** Across the 20 largest states, the model's ranking of states by turnout correlates with the ranking by reported events at 0.45 for 2018 (Spearman). Ranking states by their SC+ST population does as well (0.49), and for the 2024 bandh and the upper-caste bandh the simple baselines do better than the model. With the spatial targets left out of the fit, the model's correlations fall to between 0.18 and 0.36. The district covariates do not add predictive power.

![Identification](../figures/v3_identification.png)

![The 2018 bandh, model and reported events](../figures/v3_map_2018.png)

### Scenarios

Each reform is run on 300 retained sets and compared with a replay of 2018 on the same set, so the unidentified headcount scale cancels. "Phased" lowers the material loss in the first year; "negotiated" means no opposition party backs the protest; "all three" adds a credible guarantee. The seat changes come from the allocation model on JEE (Advanced) rank lists.

![Scenarios](../figures/v3_scenarios.png)

| Reform | Protest vs 2018 (median, 90% range) | Chance it exceeds 2018 | With all three modifiers |
|---|---|---|---|
| Abolish all reservation | 4.5x (1.5 to 14) | 100% | 1.6x (0.6 to 6.6) |
| Income-only, a quarter more seats | 1.9x (0.8 to 4.5) | 88% | 0.64x |
| Income-only test, merged pool | 1.8x (0.8 to 4.1) | 88% | 0.70x |
| SC/ST creamy layer | 0.57x (0.2 to 1.6) | 16% | 0.29x |
| SC/ST sub-classification | 0.45x (0.15 to 1.2) | 9% | 0.27x |
| EWS amendment, 2019 (replay) | 0.40x (0.1 to 1.0) | 5% | |

![Where each reform would be contested](../figures/v3_scenario_maps.png)

![Who would take part](../figures/v3_who.png)

![Seats by group](../figures/v3_seats.png)

What the scenarios say, and how far to trust it:

- Abolition is the only design that exceeds 2018 in every run. It is also the one with the weakest grounding: no episode resembles it, so its threat is a prior (between one and two times the income-only threat, with OBCs threatened as much as SCs). Under abolition, OBCs would be over half of those on the street. In the allocation model, SC seats fall by about 80 per cent and ST seats by about 90 per cent, while General seats rise by 40 to 44 per cent.
- Income-only designs exceed 2018 in seven runs out of eight. Adding a quarter more seats does not change that. The model gives both designs the same threat to group status, which was fitted on 2018 and the EWS amendment, and the seat changes move turnout only a little.
- The two designs tied to the 2024 episode, the creamy layer and sub-classification, stay below 2018 in most runs. Their ranges are narrower because the 2024 bandh constrains them.
- Phasing, negotiation and a guarantee together cut protest by 40 to 65 per cent, depending on the design. Each modifier's effect is a prior, not something any episode measured.
- The ranking of designs is more robust than the multiples. The multiples inherit the compression in the level fit, and the threat priors for abolition and income-only are wide.

## What it still cannot do

- It does not forecast headcounts. Results are relative to 2018, which is what the event data identify.
- It does not beat simple demographic baselines at predicting which states protest most.
- It has no escalation mechanism, so it cannot reproduce agitations that peak days after they start.
- Each scenario's threat is tied to its nearest episode where one exists. Income-only and abolition have no close episode, so their threat is a wide prior.
- The state communities' population shares are commonly cited estimates; there has been no caste census since 1931.
- The relevance labels behind the event counts were assigned by a language model from article URLs and have not been checked by hand.
