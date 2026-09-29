# Response to the reviews

Thank you for the second review. This document answers it point by point, and below it summarizes my response to the first review. Section numbers refer to `paper/reservation_reform_protest_paper.pdf`. Every number in the paper is generated from the result files; the commands are in the paper's Reproducibility section and in `USAGE.md`.

## Second review

### 1. L3's residual threat, and the 2024 episode

I agree this was the most important point. In the grounded specification, L3 no longer uses an assumed share of the reform's symbolic threat for SC and ST. Each run takes the ratio of the 2024 shock to the reform's shock for the parameter set drawn in that run, so the share is drawn jointly with R and with every other calibrated parameter (Experimental design section and the scenario table).

- The share has a median of 0.57 and a 90% interval of 0.06–1.00 at R = 1.
- In 16% of the retained sets the 2024 shock exceeds the reform's (up to 1.89). I cap the share at 1, because the income filter is part of the reform and cannot threaten more than all of it.
- In 31% of the sets the share is below the old value of 0.4.

Results:
- L3's paired change in peak is now −92% (95% Monte Carlo interval −95% to −87%), against −97% before.
- Under the priors on the other levers' effects, L3 is the strongest single lever with probability 87%, against 95% before.
- It stays first across the assumption grid and in the grounded structural ensemble (Robustness section).

The headline is weaker but holds, and it now rests on the episode data rather than on my judgement. OBC, which the 2024 episode did not concern, keeps the prior.

### 2. L4 and the better-off tier

Done. L4 now raises the better-off SC/ST tier's symbolic threat by a fifth, with a prior from 0 to 50%, as well as lowering the deprived tier's. The 2024 record supports this: better-off SC groups objected to sub-classification.

The effect is large:
- L4's paired change in peak falls from −40% to −12%.
- Under the priors it is the weakest single lever in 55% of draws.
- The hybrid package, which includes L4, still changes the peak by −99%, because L3 and the other levers dominate it.

The Interventions section and the hypothesis table (H4) report the change.

### 3. Concession

I added a section on concession as an outcome ("Does the reform survive?"). You were right that the paper had not discussed a striking implication of the model: in every simulated campaign, an abrupt switch is reversed, after a median of 5 days. The section says plainly that this follows from the assumed concession rule as much as from the calibrated dynamics. Nothing after the first bandh day is fitted, and in 2018 the concession came months later. The peak falls on the first bandh day in 60% of baseline runs, so everything after it is extrapolation.

The new concession table ranks every scenario by the share of runs with a concession, under the reference rule and under a slow rule (at most 3% a day, with the pressure midpoint doubled). It also gives the peak with no concession at all.

| Scenario | Reference rule | Slow rule |
|---|---|---|
| Abrupt switch (baseline) | 100% | 70% |
| L3 | 78% | 32% |
| Grandfathering | 94% | — |
| Commission | 94% | — |
| Managed transition | 74% | 28% |
| Hybrid package | 34% | 14% |

The order of levers by concession largely follows their order by peak, because the hazard rises with turnout. What changes is the meaning of the ranking: no single lever makes concession unlikely, and in the model only a package keeps an income-only reform in place in most worlds, and only under the slow rule.

### 4. What the calibration identifies

I agree, and the paper now says so.
- **"Grounded" is defined (Introduction, Scope).** It refers to the inputs: population, eligibility rules and material changes. The behavioural parameters are prior-dominated.
- **Identification (Calibration results).** The retained 90% ranges cover 86% of the prior for the mean threshold, 83% for the spread and 89% for mixing. The claim that the data "do not support complete segregation" is gone.
- **Matching choices (new sensitivity table).** The final wave is judged again with the discrepancy at 0.3, 0.5 and 0.8 and the cutoff at 2.5, 3 and 3.5. Between 9 and 2,471 sets survive, the behavioural ranges move with the setting, and the 2024-to-2018 ratio stays wide throughout.
- **Seeds.** I resimulated the 281 retained sets with ten seeds; 275 (98%) stay non-implausible. The two-day episode outputs vary little between seeds. The heavy tails belong to the reform runs, which use 50 runs per scenario.
- **Weighting.** The paper now says that retained sets are equally weighted and drawn uniformly, not a posterior.
- **S1 and S3.** The stylized-facts table marks both as partly by construction, for the reasons you gave.

### 5. Using the IIT shock for the whole population

Partly done.
- **Allotment error by category.** The largest error, as a share of the actual allotment, is 0.6% for SC, 1.3% for ST, 1.4% for OBC-NCL, 6.1% for General and 15.1% for GEN-EWS. The paper now says that the SC and ST match is close to mechanical under reserved-first processing, so the General and GEN-EWS errors are the real test, and the GEN-EWS error is large (allocation model, Validation).
- **Scaling down.** The material-weight sweep already scales every material change to 0.25 and 0.5 of its value. The paper now presents it as the check you asked for, and the order at the top is unchanged.
- **Limitations.** The IIT extrapolation, the absence of exposure weighting and the fixed applicant pool (JEE Main cutoffs) are now listed.

Not done: a job-side bound from UPSC or SSC category-wise data. I have not yet assembled those data and say so in the limitations.

### 6. Assumed income shares behind the eligibility figures

Done. The eligibility accounting now puts a prior on each group's above-line share (a factor from 0.6 to 1.4) together with the two legal shares. The 90% interval is 4.7–9.2% losing eligibility, against 5.2–8.3% when only the legal shares varied, and 1.2–5.1% gaining it. The abstract and the eligibility section give the wider range. I have not grounded the income shares in PLFS, IHDS or SECC; that remains future work.

### 7. Robustness on the grounded specification

Done.
- **Grounded ensemble.** There is now a structural ensemble on the grounded specification. Its variants are threshold shapes, fixed bandh days, no or slow concession, Poisson deaths, deaths that deter or have no effect, a single random stream, and the stylized material table. It is not recalibrated, because the thresholds come from the episodes and there is no turnout target to recalibrate against.
- **Morris screening.** The grounded screening is now the one in the main text; the stylized one moved to an appendix.
- **Stylized checks.** The one-at-a-time sensitivity and convergence checks, which are stylized-only, moved to an appendix and are labelled as such.

### 8. Disclosure and provenance

- **Relevance labels.** Appendix B, Section 7 and the limitations now say that the labels were assigned by a large language model from the URL text and have not been checked by a human coder. A coding kit for hand-coding the sample from the articles by two independent coders, and a script that computes Krippendorff's alpha and replaces the labels, are in `data/coding/` and `data_pipelines/relevance_agreement.py`. Until that coding is done, the paper states the labels' provenance as it is.
- **Frozen predictions.** The calibration results now give the timing: the three predictions were committed 36, 18 and less than one minute before the scoring commit. The hashes show the predictions were not changed after scoring, but not that they were made independently of the analysis that scored them.
- **Literature.** Several references came from your searches in review, not from mine. My first response credited them to "my own search", which was wrong, and I apologize. Appendix A now says that several cited works were suggested in review.
- **Al Jazeera.** You were right that its same-day report of "thousands" concerned the Jantar Mantar gathering in Delhi. The paper now cites The Caravan (Donthi 2018: "thousands of Dalits across the country") and no longer describes the event data as ruling out a press figure.
- **References.** I checked the new references against their sources. The authors of the 2026 Durham working paper and the venues of the Aygün–Turhan papers are correct. The page range of Deshpande and Ramachandran (2019a) is corrected to 27–31.

### Minor points

- **Monte Carlo intervals.** The bootstrap intervals are now described as the Monte Carlo precision of the median paired change, not uncertainty about the real-world effect (Uncertainty and paired sampling).
- **What counts as protest.** The model section now defines an agent-day of protest as taking part in the day's action, not passive compliance with a shutdown, and notes that this reading is not tested against data.
- **Implausible peaks.** The section on uncertainty over interventions reports the share of runs whose baseline peak exceeds 5 crore and 10 crore people on one day. It shows that L3 is still the strongest lever in most of the runs that stay below 5 crore. I did not add a plausibility constraint to the history matching, because the episodes, not the reform, are what is matched there.

## First review (summary of my earlier response)

The first review recommended major revision.

**Main changes.** In response I:
- built the allocation model on the over-and-above choice rule you pointed to (Sönmez and Yenmez 2022; Aygün and Turhan 2023);
- recomputed eligibility under the actual creamy-layer and EWS rules;
- replaced the single headcount target with history matching to GDELT event counts from four episodes, coding 2018 as locally organized;
- put priors on every intervention effect and decomposed each lever into its channels;
- split deaths by attribution, with a prior on the response to deaths;
- reported paired effects with bootstrap intervals;
- added endogenous bandhs, concession, district geography and estimated mixing, each switchable;
- ran a structural ensemble and Morris screening.

**Conclusions that changed.** Three conclusions of the original version did not survive: that the reform is a small eligibility change, that seat expansion on the EWS model helps, and that heavy policing broadens protest.

**Minor points.**
- L3 and C2 are now named distinctly.
- Brazil's Law 12.711 is discussed.
- The mobilization record is cited.
- S3 is revised.
- L4's parameters are stated precisely.
- Overlapping figure labels were replaced by legends.

**Not done.** The survey experiment on perceived threat. The mappings of commission, guarantees and income filter onto threat remain priors, apart from L3's, which now comes from the 2024 episode.
