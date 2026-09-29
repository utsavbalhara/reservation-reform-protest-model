# Literature search log

Searches run on 29 September 2026 to check whether any prior work models protest against reservation reform, or compares
reservation reform designs by the mobilization they would provoke. These were keyword searches, not a systematic review.

## Google Scholar

Queried directly (scholar.google.com, English interface, first page of results). Full titles and bylines for every hit are in
`literature_search_scholar_results.json`.

| Query | Reported hits | Relevant hits on first page |
|---|---|---|
| `reservation "agent-based" India protest` | About 576 results | None. Nearest: an agent-based model of food riots (Natalini et al. 2019). |
| `reservation "agent-based model" caste` | About 133 results | None. |
| `"affirmative action" "agent-based" India` | About 263 results | None. A caste residential-segregation study of Kolkata (Haque et al. 2021) is useful for the mixing parameter. |
| `reservation simulation protest India caste` | About 1,120 results | None. |
| `"reservation policy" protest simulation India` | About 91 results | None. |
| `caste protest "agent-based model"` | About 100 results | None. Nearest: an ABM of ethnic minority rule and civil war onset (Miodownik and Bhavnani 2011). |
| `"Bharat Bandh" model` | About 82 results | None. |
| `"income-based reservation" India` | 7 results | None modelling protest. Bao, Ni and Singh (2024, Management Science) model reservation policy in education markets. |
| `reservation "agent-based" "creamy layer"` | not shown | None. |
| `quota reform protest "agent-based"` | About 222 results | None. |
| `"agent-based" Dalit mobilization` | 57 results | None. |
| `"agent-based" "affirmative action" protest` | About 123 results | None. |
| `"agent-based model" "affirmative action"` | About 67 results | ABMs of affirmative action in hiring, boards and college sorting (e.g. Reardon et al. 2014); none of protest. |
| `"agent-based" quota protest Bangladesh` | About 66 results | None. |
| `simulation "reservation" caste "protest" model India` | About 963 results | None. |
| `"threshold model" caste mobilization India` | About 49 results | None. |
| `"reservation" India "computational model" protest` | 46 results | None. |
| `"Mandal" protest "simulation"` | About 286 results | None. |

## SSRN

SSRN's own search blocks automated access (HTTP 403), so it was searched through a web search restricted to ssrn.com
(`site:ssrn.com reservation India agent-based simulation protest caste quota`). Hits were legal and policy discussions of
reservation (EWS, creamy layer, reservation politics); none used simulation or modelled protest.

## OpenAlex and Semantic Scholar

Both APIs returned HTTP 429 (rate limited) from this environment, so they were not used.

## Conclusion used in the paper

No prior agent-based or other simulation model of protest against reservation reform was found. The paper therefore claims
only that it is, to the authors' knowledge, the first agent-based model of reservation politics and the first to compare
reservation reform designs by the protest they would provoke. Its mechanisms (loss aversion, symbolic threat, threshold
cascades, organization, repression backfire) are established and are cited as such.

## Related work verified for citation

- Bertrand, Hanna and Mullainathan (2010), Journal of Public Economics 94(1-2): 16-29. Caste-based targeting in engineering admissions reaches poorer students than the upper-caste applicants it displaces.
- Deshpande and Ramachandran (2019a), 'The 10% Quota: Is Caste Still an Indicator of Backwardness?', Economic and Political Weekly 54(13): 27-32. IHDS evidence that caste remains a marker of disadvantage even among the poor.
- Deshpande and Ramachandran (2019b), 'Traditional hierarchies and affirmative action in a globalizing economy: Evidence from India', World Development 118: 63-78. Cohort evidence on persistent caste gaps. (It does not itself compare poor households across castes.)
- Deshpande, Gille, Ramachandran and Sofianos (2026), 'Merit, Identity, and Redistribution: Experimental Evidence on Affirmative Action', Durham University Department of Economics Working Paper 2026_05. Incentivized experiment: SC-ST beneficiaries are judged less competent even under income-based affirmative action.
- Sonmez and Yenmez (2022), 'Affirmative Action in India via Vertical, Horizontal, and Overlapping Reservations', Econometrica 90(3): 1143-1176. Formal choice rules for vertical and horizontal reservations.
- Aygun and Turhan (2020), 'Dynamic reserves in matching markets', Journal of Economic Theory; and 'How to de-reserve reserves: Admissions to technical colleges in India', Management Science. OBC de-reservation in IIT admissions.
- Lemos (2018), Agent-Based Modeling of Social Conflict: From Mechanisms to Complex Behavior, SpringerBriefs in Complexity. Compares model output with SCAD event data (size, duration, recurrence) for eight African countries.
- Thron and Jackson (2015), 'Practicality of Agent-Based Modeling of Civil Violence: an Assessment', arXiv:1501.05838. Argues Epstein-type models may display generic self-organized-criticality features rather than real mechanisms.
- Srbljinovic, Penzar, Rodik and Kardov (2003), 'An Agent-Based Model of Ethnic Mobilisation', JASSS 6(1): 1.

## 2018 sources checked

- Scroll.in, 'The WhatsApp wires: How Dalits organised the Bharat Bandh without a central leadership' (scroll.in/article/874714): little organised backing from political parties; led by local Dalit groups; first call from a local leader in Phagwara on 27 March; spread through WhatsApp and social media.
- Al Jazeera news report, 2 April 2018: 'Thousands of people have joined mostly peaceful protests'; at least four killed in Madhya Pradesh (same-day reporting).
- Al Jazeera opinion column, 'A Dalit Spring is on the horizon', 8 April 2018: 'hundreds of thousands' on the streets; at least 11 killed.
- ACLED, 'Demonstrations in India' (Paul Swartzendruber, 20 April 2018): describes the 2 April Bharat Bandh; ACLED India coverage starts in 2016. The event dataset itself requires a registered account and was not used.
