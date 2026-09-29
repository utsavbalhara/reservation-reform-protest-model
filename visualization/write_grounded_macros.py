"""LaTeX macros and table rows for the allocation model, eligibility accounting, episode calibration, grounded
specification and the uncertainty analyses. Every number in the paper that comes from these analyses is written here
from the result files; none is typed by hand. Missing result files are skipped, so the paper can be built in stages.
"""
import json

import numpy as np

from .figure_style import REPOSITORY_ROOT, RESULTS_FOLDER
from .write_latex_result_macros import interval_text, lakh_text, macro, signed_percent

GENERATED_FOLDER = REPOSITORY_ROOT / "paper" / "generated"
DATA_FOLDER = REPOSITORY_ROOT / "data" / "derived"
LEVER_CODES = {"grandfathering": "L1", "seat_expansion": "L2", "hybrid_caste_subquotas": "L3", "sub_classification": "L4",
               "consensus_commission": "L5", "compensation": "L6", "credible_guarantees": "L7", "internet_shutdown": "B1",
               "heavy_policing": "B2", "managed_transition": "C1", "hybrid_package": "C2", "symbolic_only_validation": "V"}
CODE_WORDS = {"L1": "LOne", "L2": "LTwo", "L3": "LThree", "L4": "LFour", "L5": "LFive", "L6": "LSix", "L7": "LSeven",
              "B1": "BOne", "B2": "BTwo", "C1": "COne", "C2": "CTwo", "V": "V"}
SEGMENT_LABELS = {"SC_above": "SC, above the line", "SC_below": "SC, below the line", "ST_above": "ST, above the line",
                  "ST_below": "ST, below the line", "OBC-NCL_above": "OBC (non-creamy), above the line",
                  "OBC-NCL_below": "OBC, below the line", "GEN-EWS": "General, EWS-eligible",
                  "GEN_below": "General, below the line, not EWS-eligible", "GEN_above": "General (incl.\\ OBC creamy), above the line"}
ALLOCATION_ORDER = ("SC_above", "SC_below", "ST_above", "ST_below", "OBC-NCL_above", "OBC-NCL_below", "GEN-EWS", "GEN_below", "GEN_above")
EPISODE_LABELS = {"sc_st_bharat_bandh_2018": "SC/ST Bharat Bandh, 2 April 2018", "upper_caste_bandh_2018": "Upper-caste bandh, 6 September 2018",
                  "sc_st_bharat_bandh_2024": "SC/ST Bharat Bandh, 21 August 2024", "ews_quota_2019": "EWS amendment, January 2019",
                  "patidar_2015": "Patidar agitation, August 2015", "jat_2016": "Jat agitation, February 2016", "kapu_2016": "Kapu agitation, January 2016",
                  "maratha_march_mumbai_2017": "Maratha march, Mumbai, 9 August 2017", "maratha_quota_2018": "Maratha quota protests, 2018",
                  "gujjar_2019": "Gujjar agitation, February 2019"}
MECHANISM = ("sc_st_bharat_bandh_2018", "upper_caste_bandh_2018", "sc_st_bharat_bandh_2024", "ews_quota_2019")
ANCHORS = ("maratha_march_mumbai_2017", "patidar_2015")
MORRIS_LABELS = {"mean_participation_threshold": "Mean threshold $\\bar\\theta$", "participation_threshold_spread": "Threshold spread $\\sigma_\\theta$",
                 "decision_noise": "Decision noise $\\tau$", "loss_aversion": "Loss aversion $\\lambda$", "material_loss_weight": "Material weight $w_m$",
                 "symbolic_threat_weight": "Symbolic weight $w_s$", "most_deprived_tier_share_of_symbolic_threat": "Deprived-tier threat share",
                 "opposition_party_amplifier": "Party amplifier $a$", "max_neighbourhood_influence": "Neighbourhood influence $\\beta_n$",
                 "neighbourhood_turnout_at_saturation": "Neighbourhood saturation $n^*$", "max_national_visibility_influence": "National influence $\\beta_v$",
                 "fatigue_per_protest_day": "Fatigue $f$", "mobilization_on_bandh_days": "Bandh mobilization $b$",
                 "deaths_per_crore_protester_days": "Death rate $\\kappa$", "symbolic_threat_rise_per_death": "Martyr effect $\\delta$"}


def load(name):
    path = RESULTS_FOLDER / name
    return json.loads(path.read_text()) if path.exists() else None


def percent(value, digits=1):
    return f"{value * 100:.{digits}f}\\%"


def number(value, digits=1):
    text = f"{value:.{digits}f}"
    return text.replace("-", "$-$")


def signed(value, digits=0, suffix="\\%"):
    text = f"{abs(value):.{digits}f}"
    if float(text) == 0:
        return f"0{suffix}"
    return f"{'+' if value > 0 else '$-$'}{text}{suffix}"


def eligibility(macros, rows):
    data = load("eligibility_accounting.json")
    if data is None:
        return
    population = data["populations"]["district_census_nfhs_shares"]
    central, ranges = population["central"], population["range_over_priors"]
    macros += [macro("EligibilityLose", percent(central["share_losing_eligibility"])),
               macro("EligibilityGain", percent(central["share_gaining_eligibility"])),
               macro("EligibilityLoseRange", f"{percent(ranges['share_losing_eligibility'][0])}--{percent(ranges['share_losing_eligibility'][1])}"),
               macro("EligibilityGainRange", f"{percent(ranges['share_gaining_eligibility'][0])}--{percent(ranges['share_gaining_eligibility'][1])}"),
               macro("EligibilityOriginal", percent(population["original_accounting"])),
               macro("EligibilitySCSTAbove", percent(central["share_sc_st_above_line"]))]
    shares = central["segment_shares"]
    table = [("SC and ST families above \\rupee8 lakh", shares["SC_above"] + shares["ST_above"], "Reserved", "\\textbf{Unreserved}"),
             ("OBC above \\rupee8 lakh, non-creamy today", shares["OBC_above_ncl"], "Reserved", "\\textbf{Unreserved}"),
             ("SC and ST families below \\rupee8 lakh", shares["SC_below"] + shares["ST_below"], "Reserved (own quota)", "Reserved (merged pool)"),
             ("OBC families below \\rupee8 lakh", shares["OBC_below"], "Reserved (own quota)", "Reserved (merged pool)"),
             ("General below \\rupee8 lakh, EWS-eligible", shares["General_below_ews"], "Reserved (EWS)", "Reserved (merged pool)"),
             ("General below \\rupee8 lakh, failing EWS asset tests", shares["General_below_asset_excluded"], "Unreserved", "\\textbf{Reserved (merged pool)}"),
             ("OBC creamy layer and General above \\rupee8 lakh", shares["OBC_creamy"] + shares["General_above"], "Unreserved", "Unreserved")]
    rows["eligibility_rows.tex"] = [f"{label} & {percent(share)} & {before} & {after}\\\\" for label, share, before, after in table]


def allocation(macros, rows):
    data = load("merged_pool_allocation.json")
    if data is None:
        return
    years = data["years"]
    macros += [macro("AllocYears", f"{years[0]}--{years[-1]}"), macro("AllocDraws", f"{data['draws_per_year']:,}".replace(",", "{,}")),
               macro("AllocDrawsKept", f"{data['draws_kept']:,}".replace(",", "{,}"))]
    decile_error = max(value["largest_decile_share_error"] for value in data["validation"].values())
    macros.append(macro("AllocMaxDecileError", f"{decile_error * 100:.1f}"))
    ews_obc = {year: abs(v["pair_gaps"]["GEN-EWS minus OBC-NCL"]) for year, v in data["validation"].items()}
    sc_st = {year: abs(v["pair_gaps"]["SC minus ST"]) for year, v in data["validation"].items()}
    worst = max(sc_st, key=sc_st.get)
    macros += [macro("AllocGapEWSOBC", f"{max(ews_obc.values()):.3f}"),
               macro("AllocGapSCSTOthers", f"{max(v for y, v in sc_st.items() if y != worst):.2f}"),
               macro("AllocGapSCSTWorstYear", worst), macro("AllocGapSCSTWorst", f"{sc_st[worst]:.3f}")]
    check = data["allotment_check"]
    if check and "reserved_first" in next(iter(check.values())):
        macros.append(macro("AllocMaxAllotmentGap", f"{max(v['reserved_first']['largest_absolute_gap'] for v in check.values()):,}".replace(",", "{,}")))
        macros.append(macro("AllocMaxAllotmentGapOpenFirst", f"{max(v['open_first']['largest_absolute_gap'] for v in check.values()):,}".replace(",", "{,}")))
    main = data["summary"]["reserved_first|0.595"]
    names = {"SC_above": "SCAbove", "SC_below": "SCBelow", "ST_above": "STAbove", "ST_below": "STBelow", "OBC-NCL_above": "OBCAbove",
             "OBC-NCL_below": "OBCBelow", "GEN-EWS": "EWS", "GEN_below": "GENBelow", "GEN_above": "GENAbove"}
    for segment, name in names.items():
        macros.append(macro(f"AllocChange{name}", signed(main[segment]["percent_change"]["median"])))
        macros.append(macro(f"AllocRel{name}", number(main[segment]["relative_material_change"]["median"], 2)))
    for case, prefix in (("lever|merged_pool_expanded", "AllocExpanded"), ("lever|caste_income_filter", "AllocFilter")):
        if case in data["summary"]:
            for segment, name in names.items():
                macros.append(macro(f"{prefix}{name}", signed(data["summary"][case][segment]["percent_change"]["median"])))
    table = []
    for segment in ALLOCATION_ORDER:
        change, relative = main[segment]["percent_change"], main[segment]["relative_material_change"]
        table.append(f"{SEGMENT_LABELS[segment]} & {signed(change['median'])} & {signed(change['p05'], 0, '')} to {signed(change['p95'], 0, '')} & "
                     f"{number(relative['median'], 2)} & {number(relative['p05'], 2)} to {number(relative['p95'], 2)}\\\\")
    rows["allocation_rows.tex"] = table
    # Robustness across orders and pool sizes (appendix).
    robustness = []
    for case in data["summary"]:
        if case.startswith("lever|"):
            continue
        order, pool = case.split("|")
        values = data["summary"][case]
        robustness.append(f"{order.replace('_', '-')} & {float(pool) * 100:.1f}\\% & "
                          + " & ".join(signed(values[s]["percent_change"]["median"]) for s in ("SC_above", "SC_below", "ST_below", "OBC-NCL_below", "GEN-EWS", "GEN_below"))
                          + "\\\\")
    for case, label in (("lever|merged_pool_expanded", "L2: pool with 25\\% more seats"), ("lever|caste_income_filter", "L3: caste quotas, income filter")):
        if case in data["summary"]:
            values = data["summary"][case]
            robustness.append(f"\\multicolumn{{2}}{{@{{}}l}}{{{label}}} & "
                              + " & ".join(signed(values[s]["percent_change"]["median"]) for s in ("SC_above", "SC_below", "ST_below", "OBC-NCL_below", "GEN-EWS", "GEN_below"))
                              + "\\\\")
    rows["allocation_robustness_rows.tex"] = robustness


def episodes(macros, rows):
    path = DATA_FOLDER / "episode_targets.json"
    if not path.exists():
        return
    targets = json.loads(path.read_text())
    table = []
    for key in list(MECHANISM) + list(ANCHORS) + [k for k in targets if k not in MECHANISM and k not in ANCHORS]:
        target = targets[key]
        role = "M" if key in MECHANISM else "A" if key in ANCHORS else "O"
        rate = target["corrected_core_events_per_1000_india_events"]
        deaths = target["deaths"]
        death_text = f"{deaths['low']}" if deaths["low"] == deaths["high"] else f"{deaths['low']}--{deaths['high']}"
        backing = "--" if target.get("party_backing") is None else f"{target['party_backing']:.1f}"
        table.append(f"{EPISODE_LABELS[key]} & {role} & {target['broad_core_events']} & {target['precision_core_median']:.2f} & "
                     f"{rate['median']:.1f} ({rate['p05']:.1f}--{rate['p95']:.1f}) & {death_text} & {backing}\\\\")
    rows["episode_rows.tex"] = table
    audit = (DATA_FOLDER / "gdelt_relevance_audit.csv").read_text().strip().splitlines()
    macros.append(macro("AuditLabelled", str(len(audit) - 1)))
    macros.append(macro("EpisodeCount", str(len(targets))))


def calibration(macros, rows):
    data = load("episode_calibration.json")
    if data is None or not data.get("kept"):
        return
    history = data["history"]
    macros += [macro("CalibWaveSize", f"{history[0]['samples']:,}".replace(",", "{,}")), macro("CalibKept", str(data["kept"])),
               macro("CalibWaves", str(len(history))), macro("CalibFirstWaveKept", str(history[0]["kept"]))]
    summaries = data["posterior_summaries"]

    def interval(name, scale=1.0, digits=1):
        s = summaries[name]
        return (number(s["median"] * scale, digits), f"{number(s['p05'] * scale, digits)}--{number(s['p95'] * scale, digits)}")

    for name, label, scale, digits in (("mean_participation_threshold", "Theta", 1, 2), ("participation_threshold_spread", "Spread", 1, 2),
                                       ("same_group_neighbourhood_share", "Mixing", 1, 2),
                                       ("magnitude_sc_st_bharat_bandh_2018", "MagnitudeTwentyEighteen", 1, 2),
                                       ("ratio_2024_to_2018", "RatioTwentyFour", 1, 2), ("ratio_upper_caste_to_2018", "RatioUpperCaste", 1, 2),
                                       ("ratio_ews_to_2018", "RatioEWS", 1, 2), ("bandh_day_turnout_2018", "TurnoutTwentyEighteen", 1e-5, 0),
                                       ("bandh_day_turnout_2024", "TurnoutTwentyTwentyFour", 1e-5, 0),
                                       ("deprived_tier_share_2024", "DeprivedTierTwentyFour", 1, 2)):
        if name in summaries:
            median, span = interval(name, scale, digits)
            macros += [macro(f"Calib{label}", median), macro(f"Calib{label}Interval", span)]
    if "log10_bandh_deaths_per_crore" in summaries:
        s = summaries["log10_bandh_deaths_per_crore"]
        macros += [macro("CalibBandhDeathRate", f"{10 ** s['median']:.0f}"),
                   macro("CalibBandhDeathRateInterval", f"{10 ** s['p05']:.0f}--{10 ** s['p95']:.0f}")]
    spatial = data.get("spatial_tests", {})
    words = {"sc_st_bharat_bandh_2018": "TwentyEighteen", "sc_st_bharat_bandh_2024": "TwentyTwentyFour", "upper_caste_bandh_2018": "UpperCaste"}
    table = []
    for key, word in words.items():
        if key in spatial:
            test = spatial[key]
            macros += [macro(f"Spatial{word}Model", number(test["model"]["median"], 2)),
                       macro(f"Spatial{word}Population", number(test["baseline_population"]["median"], 2)),
                       macro(f"Spatial{word}SCST", number(test["baseline_sc_plus_st_population"]["median"], 2))]
            table.append(f"{EPISODE_LABELS[key]} & {number(test['model']['median'], 2)} ({number(test['model']['p05'], 2)} to {number(test['model']['p95'], 2)}) & "
                         f"{number(test['baseline_population']['median'], 2)} & {number(test['baseline_sc_plus_st_population']['median'], 2)}\\\\")
    rows["spatial_rows.tex"] = table
    scores = load("loeo_scores.json")
    if scores:
        for key, word in words.items():
            if key in scores:
                score = scores[key]
                predicted = score["deaths"]["predicted"]
                if predicted:
                    macros.append(macro(f"Loeo{word}Deaths", f"{predicted['median']:.0f}"))
                spatial_score = score["spatial_spearman"]
                macros += [macro(f"Loeo{word}SpatialModel", number(spatial_score["model"], 2)),
                           macro(f"Loeo{word}SpatialPopulation", number(spatial_score["baseline_population"], 2)),
                           macro(f"Loeo{word}SpatialSCST", number(spatial_score["baseline_sc_plus_st_population"], 2))]
        macros.append(macro("LoeoStates", str(next(iter(scores.values()))["spatial_spearman"]["states"])))
        table = []
        for key, score in scores.items():
            predicted = score["deaths"]["predicted"]
            observed = score["deaths"]["observed"]
            observed_text = f"{observed[0]}" if observed[0] == observed[1] else f"{observed[0]}--{observed[1]}"
            prediction = "--" if predicted is None else f"{predicted['median']:.1f} ({predicted['p05']:.1f}--{predicted['p95']:.1f})"
            inside = "--" if score["deaths"]["observed_inside_90_percent_interval"] is None else ("yes" if score["deaths"]["observed_inside_90_percent_interval"] else "no")
            spatial_score = score["spatial_spearman"]
            model = "--" if spatial_score["model"] is None else number(spatial_score["model"], 2)
            table.append(f"{EPISODE_LABELS[key]} & {score['retained_parameter_sets']} & {observed_text} & {prediction} & {inside} & "
                         f"{model} & {number(spatial_score['baseline_population'], 2)} & {number(spatial_score['baseline_sc_plus_st_population'], 2)}\\\\")
        rows["loeo_rows.tex"] = table


def mapping_uncertainty(macros, rows):
    for specification, word in (("stylized", "Stylized"), ("grounded", "Grounded")):
        data = load(f"intervention_mapping_uncertainty_{specification}.json")
        if data is None:
            continue
        macros.append(macro(f"Map{word}Runs", str(data["run_count"])))
        for lever, probability in data["probability_strongest_single_lever"].items():
            macros.append(macro(f"Map{word}Strongest{CODE_WORDS[LEVER_CODES[lever]]}", percent(probability, 0)))
        for lever, probability in data["probability_weakest_single_lever"].items():
            macros.append(macro(f"Map{word}Weakest{CODE_WORDS[LEVER_CODES[lever]]}", percent(probability, 0)))
        pairwise = data["probability_row_beats_column"]
        for a, b in (("hybrid_caste_subquotas", "consensus_commission"), ("consensus_commission", "grandfathering"),
                     ("hybrid_caste_subquotas", "grandfathering"), ("credible_guarantees", "compensation"), ("grandfathering", "seat_expansion")):
            macros.append(macro(f"Map{word}{CODE_WORDS[LEVER_CODES[a]]}Beats{CODE_WORDS[LEVER_CODES[b]]}", percent(pairwise[a][b], 0)))
        for scenario, values in data["suppression_backfire"].items():
            code = CODE_WORDS[LEVER_CODES[scenario]]
            macros += [macro(f"Map{word}{code}PeakHigher", percent(values["share_of_runs_peak_higher"], 0)),
                       macro(f"Map{word}{code}CumulativeHigher", percent(values["share_of_runs_cumulative_higher"], 0)),
                       macro(f"Map{word}{code}DaysHigher", percent(values["share_of_runs_protester_days_higher"], 0))]
            quartiles = values["by_death_response_factor_quartile"]
            macros += [macro(f"Map{word}{code}CumulativeHigherLowResponse", percent(quartiles[0]["share_cumulative_higher"], 0)),
                       macro(f"Map{word}{code}CumulativeHigherHighResponse", percent(quartiles[-1]["share_cumulative_higher"], 0))]
        table = []
        for lever in data["rank_distribution"]:
            code = LEVER_CODES[lever]
            effect = data["paired_effects"][lever]["peak"]
            ranks = data["rank_distribution"][lever]
            table.append(f"{code} & {signed(effect['change_percent'])} & {interval_text(effect['change_percent_ci95'])} & "
                         + " & ".join(percent(ranks[str(rank)], 0) for rank in range(1, len(ranks) + 1)) + "\\\\")
        rows[f"mapping_rows_{specification}.tex"] = table


def decomposition(macros, rows):
    data = load("lever_decomposition.json")
    if data is None:
        return
    channels = data["channels"]
    names = {("L5", "full"): "DecompLFiveFull", ("L5", "amplifier_only"): "DecompLFiveAmplifier", ("L5", "symbolic_only"): "DecompLFiveSymbolic",
             ("L3", "full"): "DecompLThreeFull", ("L3", "symbolic_only"): "DecompLThreeSymbolic", ("L3", "material_only"): "DecompLThreeMaterial",
             ("L1", "full"): "DecompLOneFull", ("L1", "material_only"): "DecompLOneMaterial", ("L1", "symbolic_only"): "DecompLOneSymbolic",
             ("B2", "full"): "DecompBTwoFull", ("B2", "turnout_cost_only"): "DecompBTwoTurnoutCost", ("B2", "violence_and_martyr_only"): "DecompBTwoViolence",
             ("B1", "full"): "DecompBOneFull", ("B1", "coordination_only"): "DecompBOneCoordination"}
    for (lever, channel), name in names.items():
        if lever in channels and channel in channels[lever]:
            effects = channels[lever][channel]
            macros.append(macro(name, signed(effects["peak"]["change_percent"])))
            macros.append(macro(name + "CI", interval_text(effects["peak"]["change_percent_ci95"])))
            if "cumulative" in effects:
                macros.append(macro(name + "Cumulative", signed(effects["cumulative"]["change_percent"])))
            if "deaths" in effects:
                macros.append(macro(name + "Deaths", signed(effects["deaths"]["change_percent"])))
            if "protester_days" in effects:
                macros.append(macro(name + "Days", signed(effects["protester_days"]["change_percent"])))
            if "emergent_death_factor" in effects:
                macros.append(macro(name + "EmergentDeathFactor", f"{effects['emergent_death_factor']:.1f}"))
                macros.append(macro(name + "DeathRatio", f"{effects['deaths']['median_paired_ratio']:.1f}"))


def break_even(macros, rows):
    for specification, word in (("stylized", "Stylized"), ("grounded", "Grounded")):
        data = load(f"material_symbolic_break_even_{specification}.json")
        if data is None:
            continue
        macros.append(macro(f"BreakEven{word}ReferenceRatio", number(data["reference_ratio"], 1)))
        material = data["ratios_where_a_material_lever_is_strongest"]
        admitted = data["ratios_admitted_by_validation"]
        macros.append(macro(f"BreakEven{word}MaterialStrongestBelow", "none" if not material else number(max(material), 2)))
        macros.append(macro(f"BreakEven{word}Admitted", "none" if not admitted else f"{number(min(admitted), 2)}--{number(max(admitted), 2)}"))
        table = []
        for point in data["points"]:
            effects = point["effects"]
            table.append(f"{number(point['symbolic_to_material_weight_ratio'], 2)} & {number(point['calibrated_mean_threshold'], 2)} & "
                         f"{point['baseline_median_peak_lakh']:.0f} & {point['symbolic_only_share_of_baseline'] * 100:.0f}\\% & "
                         + " & ".join(signed(effects[lever]["peak_change_percent"]) for lever in ("grandfathering", "compensation", "hybrid_caste_subquotas", "consensus_commission", "credible_guarantees"))
                         + f" & {LEVER_CODES[point['ranking_strongest_first'][0]]}\\\\")
        rows[f"break_even_rows_{specification}.tex"] = table


def global_sensitivity(macros, rows):
    data = load("global_sensitivity_stylized.json")
    if data is None:
        return
    results = data["results"]
    macros.append(macro("MorrisTrajectories", str(data["trajectories"])))
    macros.append(macro("MorrisDesignPoints", str(data["lever_order_across_design_points"]["design_points"])))
    for key, name in (("L3_below_L5", "MorrisLThreeBeatsLFive"), ("L5_below_L1", "MorrisLFiveBeatsLOne"), ("L3_below_L1", "MorrisLThreeBeatsLOne")):
        macros.append(macro(name, percent(data["lever_order_across_design_points"][key], 0)))
    outputs = ("log_peak", "peak_change_percent:grandfathering", "peak_change_percent:hybrid_caste_subquotas", "peak_change_percent:consensus_commission")
    for output, word in zip(outputs, ("Peak", "LOne", "LThree", "LFive")):
        top = results[output]["ranking"][0]
        macros.append(macro(f"MorrisTop{word}", MORRIS_LABELS[top]))
    ordering = results["log_peak"]["ranking"]
    table = []
    for name in ordering:
        cells = []
        for output in outputs:
            factor = results[output]["factors"][name]
            cells.append(f"{factor['mu_star']:.2f}" if output == "log_peak" else f"{factor['mu_star']:.0f}")
        table.append(f"{MORRIS_LABELS[name]} & " + " & ".join(cells) + f" & {results['log_peak']['factors'][name]['sigma']:.2f}\\\\")
    rows["morris_rows.tex"] = table


def structural_ensemble(macros, rows):
    data = load("structural_ensemble.json")
    if data is None:
        return
    variants = data["variants"]
    shares = data["share_of_variants_where_strongest"]
    macros.append(macro("EnsembleVariants", str(len(variants))))
    macros.append(macro("EnsembleRuns", str(data["runs_per_scenario"])))
    macros.append(macro("EnsembleLThreeStrongest", str(sum(1 for v in variants.values() if v["single_lever_order_by_peak_change"][0] == "hybrid_caste_subquotas"))))
    macros.append(macro("EnsembleLSixWeakest", str(sum(1 for v in variants.values() if v["single_lever_order_by_peak_change"][-1] == "compensation"))))
    macros.append(macro("EnsembleLFiveAboveLOne", str(sum(1 for v in variants.values()
                                                            if v["paired_effects"]["consensus_commission"]["peak"]["change_percent"] < v["paired_effects"]["grandfathering"]["peak"]["change_percent"]))))
    macros.append(macro("EnsembleBTwoCumulativeHigher", str(sum(1 for v in variants.values() if v["paired_effects"]["heavy_policing"]["cumulative"]["change_percent"] > 0))))
    table = []
    for name, variant in variants.items():
        effects = variant["paired_effects"]
        order = [LEVER_CODES[lever] for lever in variant["single_lever_order_by_peak_change"]]
        table.append(f"{variant['description']} & {variant['recalibrated_mean_threshold']:.2f} & "
                     + " & ".join(signed(effects[lever]["peak"]["change_percent"]) for lever in ("grandfathering", "hybrid_caste_subquotas", "consensus_commission", "compensation"))
                     + f" & {signed(effects['heavy_policing']['cumulative']['change_percent'])} & {order[0]} / {order[-1]}\\\\")
    rows["ensemble_rows.tex"] = table


def grounded_extras(macros, rows):
    data = load("intervention_comparison_grounded_central.json")
    if data is None:
        return
    baseline = data["scenarios"]["baseline"]["runs"]
    conceded = [day for day in baseline["conceded_on_day"]]
    macros.append(macro("GroundedConcededShare", percent(np.mean([day is not None for day in conceded]), 0)))
    days = [day for day in conceded if day is not None]
    macros.append(macro("GroundedConcessionDay", "--" if not days else f"{np.median(days):.0f}"))
    macros.append(macro("GroundedBandhCalls", f"{np.median([len(b) for b in baseline['bandh_days_called']]):.0f}"))
    first = [b[0] + 1 for b in baseline["bandh_days_called"] if b]
    macros.append(macro("GroundedFirstBandhDay", "--" if not first else f"{np.median(first):.0f}"))
    macros.append(macro("GroundedNoBandhShare", percent(np.mean([len(b) == 0 for b in baseline["bandh_days_called"]]), 0)))


def grounded_assumptions(macros, rows):
    data = load("grounded_assumptions.json")
    if data is None:
        return
    cells = data["cells"]
    peaks = [cell["baseline_median_peak_lakh"] for cell in cells]
    macros += [macro("AssumptionPeakLow", lakh_text(min(peaks))), macro("AssumptionPeakHigh", lakh_text(max(peaks))),
               macro("AssumptionCells", str(len(cells))), macro("AssumptionRuns", str(data["runs"])),
               macro("AssumptionLThreeFirst", str(sum(cell["single_lever_order_by_peak_change"][0] == "hybrid_caste_subquotas" for cell in cells))),
               macro("AssumptionLSixLast", str(sum(cell["single_lever_order_by_peak_change"][-1] == "compensation" for cell in cells)))]
    table = []
    for cell in cells:
        effects = cell["paired_effects"]
        table.append(f"{cell['shock_ratio']:.1f} & {cell['party_backing']:.1f} & {lakh_text(cell['baseline_median_peak_lakh'])} & "
                     f"{cell['baseline_median_deaths']:g} & {cell['baseline_share_conceded'] * 100:.0f}\\% & "
                     + " & ".join(signed(effects[s]["peak"]["change_percent"]) for s in ("grandfathering", "hybrid_caste_subquotas", "consensus_commission", "compensation"))
                     + f" & {LEVER_CODES[cell['single_lever_order_by_peak_change'][0]]}\\\\")
    rows["assumption_rows.tex"] = table
    sweep = data.get("material_weight_sweep", [])
    if sweep:
        material_first = [point["material_weight_multiplier"] for point in sweep if point["material_lever_is_strongest"]]
        macros.append(macro("MaterialSweepFirstMultiplier", "none" if not material_first else f"{min(material_first):g}"))
        macros.append(macro("MaterialSweepMaxMultiplier", f"{max(point['material_weight_multiplier'] for point in sweep):g}"))
        rows["material_sweep_rows.tex"] = [
            f"{point['material_weight_multiplier']:g} & {lakh_text(point['baseline_median_peak_lakh'])} & "
            + " & ".join(signed(point["peak_change_percent"][s]) for s in ("grandfathering", "seat_expansion", "hybrid_caste_subquotas", "consensus_commission", "compensation", "credible_guarantees"))
            + f" & {LEVER_CODES[point['single_lever_order_by_peak_change'][0]]}\\\\" for point in sweep]


def main():
    GENERATED_FOLDER.mkdir(parents=True, exist_ok=True)
    macros, rows = [], {}
    for section in (eligibility, allocation, episodes, calibration, mapping_uncertainty, decomposition, break_even,
                    global_sensitivity, structural_ensemble, grounded_extras, grounded_assumptions):
        section(macros, rows)
    (GENERATED_FOLDER / "grounded_macros.tex").write_text("\n".join(macros) + "\n")
    for name, lines in rows.items():
        (GENERATED_FOLDER / name).write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(macros)} macros and {len(rows)} tables to {GENERATED_FOLDER}")


if __name__ == "__main__":
    main()
