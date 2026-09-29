# Response to the review

The review recommended major revision. This document answers each point: what was done, where it is in the revised paper (section numbers refer to `paper/reservation_reform_protest_paper.pdf`), and what was not done and why. Every number in the paper is generated from result files; the commands are in the paper's Reproducibility section and in `USAGE.md`.

## Summary of the revision

The stylized model is kept as a reference specification. Alongside it, a **grounded specification** replaces its assumed inputs wherever data allow, and four components now link into one analysis (Figure 1):

1. an **eligibility accounting** under the actual creamy-layer and EWS rules (Section 2.2);
2. an **allocation model** of IIT seats built on India's over-and-above choice rule, a latent merit model fitted to five years of JEE (Advanced) rank lists, and income priors constrained by published evidence, validated against actual allotments (Section 5);
3. a **protest model on 640 Census districts** whose behavioural parameters and shock sizes are **history-matched to GDELT event counts** from four episodes, with frozen leave-one-episode-out predictions (Sections 6, 7, 10);
4. an **uncertainty layer**: priors on every intervention effect, channel decompositions, sweeps of the unidentified assumptions, an eleven-variant structural ensemble and Morris screening (Sections 11.5–12).

The revision changed three conclusions of the original analysis and confirmed the top and bottom of the lever ranking. The changed conclusions are that the reform is small, that seat expansion helps, and that heavy policing broadens protest. The level of protest the reform would provoke is now reported as not identified.

## Major points

**1. Eligibility is not allocation; the pool merger was treated loosely.**
Done. Section 5 builds an allocation model. It uses the over-and-above choice rule (Sönmez and Yenmez 2022; Aygün and Turhan 2023), which the review pointed to, rather than an invented mechanism. The latent merit model is fitted by maximum likelihood to the category composition of the JEE (Advanced) Common Rank Lists for 2021–2025, and family income enters through priors constrained by the JIC parental-income tables. Validation:
- category shares by rank decile within 3.5 percentage points;
- implied cutoffs agree across lists that share a cutoff;
- reserved-first processing reproduces actual category allotments to within 406 seats, while open-first misses by up to 2,386.

Result: under a merged pool, below-line SC candidates lose 72% of their IIT seats and below-line ST candidates 85%, although they keep eligibility. The protest model's material inputs now come from this table.

**2. The legal accounting was inaccurate, and "2.8%" was presented as a finding.**
Done. Section 2.1 states the actual rules:
- the creamy-layer income test excludes salary and agricultural income, and there are separate status criteria;
- EWS carries asset exclusions.

Section 2.2 and `experiments/eligibility_accounting.py` recompute who changes status, with priors on the two unobserved shares. Result: 6.8% of Indians lose eligibility (range 5.2–8.3%) and 3.1% gain it. The figure of 2.9% comes from counting only SC/ST families above the line, and the paper explains why that count is wrong.

**3. The ranking of interventions was baked into the intervention mappings.**
Addressed three ways (Sections 11.5, 11.6):
- **Priors.** Every lever's effect size now has a uniform prior. Each run draws one mapping, and we report the probability that each lever ranks first to last. L3 comes first with probability 95% (grounded) and 82% (stylized).
- **Channels.** Each lever is decomposed into its channels on paired worlds, which shows what drives it. L5 works mostly through its assumed symbolic cut (−48%), not party machinery (−15%).
- **Data-based levers.** Where a lever is itself an allocation rule (L2 seat expansion, L3 caste quotas with an income filter), its material effect now comes from the allocation model rather than an assumption. This is what reversed the verdict on L2.

What remains assumed: how much a commission, a guarantee or an income filter lowers perceived threat. The paper says so, and names survey experiments as the way to estimate it.

**4. Suppression findings were mostly inputs.**
Done (Sections 6.5, 11.4, 11.5, 11.6).
- Deaths are split into police-attributed and other deaths, and heavy policing raises only the former.
- The response of turnout to deaths now has a prior running from deterrence to strong backfire.
- The decomposition separates the assumed rise in deaths from the emergent change in protester-days.

Revised conclusion: suppression costs lives in every specification. Whether it broadens protest depends on the unidentified response to deaths, and it does not in the grounded specification, where governments concede quickly.

**5. The ratio of symbolic to material weight (w_s/w_m) is unidentified.**
Acknowledged and bounded (Section 11.7).
- **Stylized:** a sweep holding w_s·w_m fixed, recalibrating θ̄ at each point, finds that a material lever is strongest only when w_s/w_m ≤ 1.47 (reference 7.3).
- **Grounded:** the episodes involve no material change, so they don't identify w_m either. A direct sweep of w_m from ×0.25 to ×8 never makes a material lever the strongest; it reorders the middle of the ranking.

**6. The 2018 validation failed its own benchmarks.**
Replaced. The model is no longer calibrated to a headcount. It is history-matched to precision-corrected GDELT event counts for four episodes (Sections 7, 10):
- the April 2018 and August 2024 SC/ST bandhs;
- the September 2018 upper-caste bandh, with the threat pointing the other way;
- the EWS amendment as a null episode.

Two published crowd estimates (Maratha 2017, Patidar 2015) anchor the observation model. The 2018 bandh is coded as locally organized, following Scroll's reporting.

The results are reported honestly. Event data identify relative shock sizes far better than turnout: the implied 2018 turnout is 7–539 lakh (90%). The frozen leave-one-episode-out predictions of deaths and geography are weak. The paper draws the consequence that the reform's protest level is not identified, and presents paired comparisons between designs as the object of the analysis.

**7. The dynamics were too stylized (exogenous bandhs, no geography, single-group neighbourhoods).**
Done (Sections 6, 12.2). The grounded specification has:
- endogenous bandh calls;
- a pressure-dependent concession with a General-category backlash;
- negative-binomial, split deaths and a response parameter for deaths;
- separate random streams;
- a population on 640 Census districts with NFHS-5 shares and neighbourhood mixing estimated in calibration (median 0.78).

Each alternative can be switched on its own. The structural ensemble reruns the comparison under eleven variants, each recalibrated, and L3 comes first and compensation last in all of them.

**8. Statistics should be paired, with bootstrap intervals.**
Done throughout. Every change is the median over runs of the ratio to the baseline in the same world, with a 95% percentile-bootstrap interval and the share of runs above the baseline (Section 6.6). One earlier claim turns out to have been an artifact of comparing medians: shutdowns raising cumulative participation by 9%. It is gone.

## Minor points

- **L3 vs C2 naming.** L3 is "keep caste quotas, income filter inside them"; C2 is the "hybrid package". The word "hybrid" is no longer used for L3.
- **Brazil's Law 12.711.** Now discussed in the text (Section 5.5) as a precedent for sub-quotas inside an income-based quota.
- **Section 2.3 uncited.** The mobilization record is now cited (Scroll, Al Jazeera news and opinion), and the conflicting crowd reports for 2018 are stated.
- **S3 non-discriminating.** S3 is revised: large mobilizations don't require party backing (2018) and party backing doesn't guarantee them (2024). The grounded model agrees: raising backing from the 2018 to the 2024 level changes the peak by a factor of 1.04 (Table 10).
- **L4 "symbolic share 0.3" ambiguous.** Now stated precisely: the most-deprived tier's symbolic threat falls from 0.8 to 0.3 times its group's, and its material change falls by half the unit loss (Table 7).
- **Figure 4 overlapping labels.** Replaced by a legend; the new figures follow the same rule.

## Literature

Added and used in the text:
- Kuran (1991) and Marwell and Oliver (1993) on cascades and critical mass.
- Wilkinson (2004) on the state politics of violence, which motivates concession and policing.
- Windrum, Fagiolo and Moneta (2007) on validating agent-based models.
- Vernon et al. (2010) and Andrianakis et al. (2015) on history matching.
- Cranmer et al. (2020) on simulation-based inference, noted as an alternative.
- Morris (1991), Campolongo et al. (2007) and Saltelli et al. (2008) on global sensitivity.
- Grimm et al. (2020) on the ODD protocol; the ODD summary is in Appendix F.

Also added, from the author's own search:
- Sönmez and Yenmez (2022), and Aygün and Turhan (2020; 2023a; 2023b), on reservation choice rules.
- Bertrand, Hanna and Mullainathan (2010), and Deshpande and Ramachandran (2019a; 2019b), on caste versus income targeting.
- Deshpande et al. (2026) on perceptions under income-based affirmative action.
- Srbljinović et al. (2003), Lemos (2018), and Thron and Jackson (2015) on protest and conflict models; the last is a critique the paper answers directly in Section 3.
- Leetaru and Schrodt (2013), Hammond and Weidmann (2014), and Raleigh et al. (2010) on event data.

The novelty claim is narrowed to "the first agent-based model of reservation politics and the first comparison of reservation reform designs by the protest they would provoke". It is supported by a documented search: 18 Google Scholar queries and an SSRN search (Appendix A, `docs/literature_search.md`).

## The improvement plan (A–H)

| Plan item | Status |
|---|---|
| A. Merged-pool merit allocation | Done (Section 5). IIT only; NEET attempted and excluded after failing a tail check; government jobs not covered. |
| B. District population | Done: Census 2011 districts, NFHS-5 state OBC shares, estimated mixing. Not done: state-level income adjustment, jati-level tiers beyond two SC/ST tiers, organizational capacity by state, survey-based identity strength. Listed as limitations. |
| C. Episode dataset | Done with GDELT: 10 episodes, keyword filters, a precision audit of 246 URLs, deaths and party backing coded from cited reporting. Not done: a second independent coder and an agreement statistic (Krippendorff's α). The ACLED pipeline is built and awaits the data. |
| D. Multi-episode calibration and out-of-sample validation | Done: history matching to four episodes; leave-one-episode-out predictions frozen, hashed and pushed to the public repository before scoring. OSF preregistration was not used; the public git history serves as the timestamp. |
| E. Grounding intervention mappings | Partly: priors, rank probabilities, decompositions, and allocation-derived levers L2 and L3. The survey was skipped at the author's decision, so perceived-threat effects remain priors. |
| F. Structural upgrades | Done: endogenous bandhs, concession and backlash, split deaths, response to deaths, mixing, threshold shapes, legal eligibility flags. |
| G. Global sensitivity, ensembles, paired CIs, verification | Done: Morris screening in both specifications (not Sobol), the eleven-variant ensemble, paired bootstrap CIs, and a test suite with an independent scalar reimplementation. |
| H. Rewrite | Done: paper, report, README, usage guide and results page. |

## What the revision does not claim

- The level of protest is not identified: the baseline peak runs from 29 lakh to 6.8 crore across the assumptions the data leave open.
- The model does not predict where violence will occur, and its state distribution is a demographic baseline.
- The seat results are for IIT admissions.
- How strongly a commission, a guarantee or an income filter reduces perceived threat is a prior.

## Open items for the author

- The GDELT relevance labels were produced from URL text by a single coder and have not been double-coded. `data/derived/gdelt_relevance_audit.csv` records who coded them. Before submission they should be re-coded by the author and a second person, with an agreement statistic.
- ACLED: place the export in `data/raw/acled/` and run `python -m data_pipelines.acled_episode_events`, then `python -m experiments.acled_cross_check`.
- The simulation videos show the stylized model; the captions say so.
