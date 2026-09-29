# Coding the GDELT relevance sample

The file `relevance_coding_sheet_blank.csv` lists 246 news articles that GDELT coded as protest events in India and that matched an episode's keywords. For each article, decide whether it reports a protest event that belongs to the episode. The current labels were assigned by a language model from the URL text alone; this coding replaces them.

## How to code

1. Make a copy of the sheet for each coder: `relevance_coding_sheet_coder1.csv` and `relevance_coding_sheet_coder2.csv`. Code independently: do not discuss items, and do not look at the other coder's sheet or at `data/derived/gdelt_relevance_audit.csv`.
2. Open each URL and read the article. If the link is dead, try the Internet Archive (`https://web.archive.org/web/2018*/<url>`). If no version can be read, code from the headline and write `headline only` in `notes`.
3. Put `1` in `relevant` if the article reports a protest action (street protest, bandh or shutdown, march, road or rail blockade, sit-in, gherao) that is part of the episode below, on or around the episode's dates. Put `0` otherwise, including:
   - protests about something else on the same days (for example the fuel-price bandh of 10 September 2018);
   - articles about the policy or the court case with no protest action;
   - opinion pieces and explainers that report no protest action.
4. Leave `relevant` blank only if you cannot decide; say why in `notes`.

## Episodes

| Episode | What counts | Core dates |
|---|---|---|
| sc_st_bharat_bandh_2018 | Bharat Bandh against the Supreme Court's dilution of the SC/ST (Prevention of Atrocities) Act | 2 April 2018 |
| upper_caste_bandh_2018 | Bharat Bandh by upper-caste and some OBC groups against the amendment restoring the Act | 6 September 2018 |
| sc_st_bharat_bandh_2024 | Bharat Bandh against SC/ST sub-classification and a suggested creamy layer | 21 August 2024 |
| ews_quota_2019 | Protest for or against the 10% EWS quota (103rd Amendment) | 8–9 January 2019 |
| patidar_2015 | Patidar agitation for OBC status, Gujarat | 25–27 August 2015 |
| jat_2016 | Jat agitation for OBC status, Haryana | 19–22 February 2016 |
| kapu_2016 | Kapu Garjana rally for OBC status, Tuni, Andhra Pradesh | 31 January 2016 |
| maratha_march_mumbai_2017 | Maratha Kranti Morcha silent march, Mumbai | 9 August 2017 |
| maratha_quota_2018 | Maratha quota agitation, Maharashtra | 24–25 July and 9 August 2018 |
| gujjar_2019 | Gujjar rail blockade for a 5% quota, Rajasthan | 8–13 February 2019 |

## Afterwards

Put both sheets in `data/coding/` and run

```
python -m data_pipelines.relevance_agreement --coder1 data/coding/relevance_coding_sheet_coder1.csv --coder2 data/coding/relevance_coding_sheet_coder2.csv
```

It reports Krippendorff's alpha and Cohen's kappa between the two coders, and each coder's agreement with the language-model labels. It also lists the disagreements in `data/coding/disagreements.csv`. Resolve each by discussion and fill in its `resolved` column, then run the command again with `--resolved data/coding/disagreements.csv --write`. That writes the human labels into `data/derived/gdelt_relevance_audit.csv`, with the coder column updated. Then rerun `python -m data_pipelines.episode_targets`, the calibration and everything downstream.
