#!/usr/bin/env python
"""12_key_numbers.py: collect every number the revision's text blocks quote into
results/key_numbers.json, each with its source file, so text and RESULTS.md can
be checked mechanically against the results files."""
import json, os
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda p: json.load(open(os.path.join(W, "results", p)))
import os as _os  # public copy: the program folders (see README) are found through PAPER_PREP
PAPER_PREP = _os.environ.get("PAPER_PREP", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "program_folders"))
IMU = PAPER_PREP + "/IMU_underserved/results/mc_report_v2.json"
k = {}
def put(name, val, src): k[name] = {"value": val, "source": src}
t = R("title1_full_rule_summary.json"); s = R("small_checks.json")
for sc in ("0.5", "1.0", "1.5"):
    b = t["by_se_scale"][sc]
    put(f"t1_indeterminate_k{sc}", b["indeterminate_0p10_0p90"], "results/title1_full_rule_summary.json")
    put(f"t1_indeterminate_share_k{sc}", round(b["indeterminate_share"], 4), "results/title1_full_rule_summary.json")
    put(f"t1_knife_k{sc}", b["knife_edge_0p40_0p60"], "results/title1_full_rule_summary.json")
b = t["by_se_scale"]["1.0"]
for kk in ("expected_misclassified_flat_prior", "expected_losses", "expected_gains", "designated_with_phi_ge_0p10",
           "share_designated_with_phi_ge_0p10", "expected_changes_share_of_list"):
    put(f"t1_{kk}", round(b[kk], 4), "results/title1_full_rule_summary.json")
put("t1_by_binding_k1", b["indeterminate_by_binding_threshold"], "results/title1_full_rule_summary.json")
put("t1_n", t["n_districts"], "results/title1_full_rule_summary.json")
put("t1_point_eligible", t["point_estimate_eligible"], "results/title1_full_rule_summary.json")
put("t1_old74", t["old_count_cliff_74"], "results/title1_full_rule_summary.json")
put("t1_mc_crosscheck", t["mc_crosscheck_k1"], "results/title1_full_rule_summary.json")
put("t1_denominator", {c: v["indeterminate_0p10_0p90"] for c, v in t["denominator_error_sensitivity_k1"].items()}, "results/title1_full_rule_summary.json")
put("t1_se_fit", t["se_fit_diagnostics"], "results/title1_full_rule_summary.json")
put("t1_exhibits", [{x: e[x] for x in ("name", "formula_children", "pop_5_17", "rate", "T", "binding_threshold", "p_elig_k1.0", "p_elig_k0.5", "p_elig_k1.5")} for e in t["exhibits"]], "results/title1_full_rule_summary.json")
put("t1_small_lea", s["title1_small_lea"], "results/small_checks.json")
v = R("qct_validation.json")
put("qct_anchor_rounding", {x: v["baseline_hud_rounding"][x] for x in ("match_hud", "pct", "false_pos_replication_only", "false_neg_hud_only", "designated_replication", "designated_hud", "recall_on_hud_designated", "precision", "eligibility_boundary_10_resolved", "eligibility_boundary_10_raw_ratio_range", "n_residual", "residual_in_cap_bound_units")}, "results/qct_validation.json")
put("qct_cap_bound_units", v["cap_bound_units_baseline"], "results/qct_validation.json")
put("qct_tracts_in_cap_bound_units", v["tracts_in_cap_bound_units"], "results/qct_validation.json")
put("qct_engine_validation", {**v["baseline_folder_conventions"], **v["random_draws_five_metros"]}, "results/qct_validation.json")
rc = R("qct_ranking_conventions.json")
put("qct_ranking_conventions", {x: rc[x]["residual"] for x in rc if x.startswith("V")}, "results/qct_ranking_conventions.json")
put("qct_residual_units", {x: rc[x] for x in ("units_with_residual", "units_where_hud_list_exceeds_20pct", "hud_share_range_in_residual_units")}, "results/qct_ranking_conventions.json")
c = R("qct_canonical_summary.json")
for run in ("realistic", "indep", "xvintage_only", "xvintage_high", "realistic_ip_lo", "realistic_ip_hi", "realistic_lognormal_inc", "realistic_se075", "realistic_se125", "elig_real_rank_indep", "elig_indep_rank_real"):
    for sc in ("national", "cap_bound_units", "slack_units", "five_pilot_metros"):
        d = c["runs"][run][sc]
        put(f"qct_{run}_{sc}", {x: (round(d[x], 1) if isinstance(d[x], float) else d[x]) for x in ("designated", "expected_losses", "losses_eligibility_driven", "losses_rank_driven", "expected_gains", "gains_by_baseline_eligible", "expected_two_sided", "expected_changes_share_of_list", "phi_ge_0p10", "designated_phi_ge_0p10", "share_designated_phi_ge_0p10", "pop_designated_phi_ge_0p10", "old_signature_pelig95_phi05", "designated_rank_flip_ge_0p10", "designated_pelig95_rank_flip_ge_0p10")}, "results/qct_canonical_summary.json")
put("qct_boot_se_realistic", {sc: {x: round(y, 2) for x, y in d.items()} for sc, d in c["bootstrap_mc_se_realistic"].items()}, "results/qct_canonical_summary.json")
put("qct_rule_variants", {r: {"baseline_designated": d["baseline_designated"], **{x: (round(d["national"][x], 1) if isinstance(d["national"][x], float) else d["national"][x]) for x in ("expected_losses", "expected_gains", "expected_two_sided", "expected_changes_share_of_list", "designated_phi_ge_0p10", "share_designated_phi_ge_0p10")}} for r, d in c["rule_variants_own_baseline"].items()}, "results/qct_canonical_summary.json")
put("qct_exhibits", c["exhibits_realistic"], "results/qct_canonical_summary.json")
put("qct_residual_221_mc", c["residual_221"], "results/qct_canonical_summary.json")
dc = R("qct_decomposition/decomposition_summary.json")
put("qct_decomposition", {x: dc[x] for x in dc if x != "exhibit_06059074806"}, "results/qct_decomposition/decomposition_summary.json")
put("qct_decomposition_exhibit", dc["exhibit_06059074806"], "results/qct_decomposition/decomposition_summary.json")
put("qct_year_over_year", s["qct_year_over_year"], "results/small_checks.json")
put("qct_year_over_year_by_rule", s["qct_year_over_year_by_rule"], "results/small_checks.json")
eb = R("qct_eb/eb_summary.json")
for run in ("flat", "eb", "eb_univ1000", "eb_unit_psi"):
    for sc in ("national", "cap_bound"):
        put(f"qct_eb_{run}_{sc}", {x: (round(y, 3) if isinstance(y, float) else y) for x, y in eb[run][sc].items()}, "results/qct_eb/eb_summary.json")
put("qct_eb_near25", eb["designated_with_2019_23_rate_within_3pts_of_25pct"], "results/qct_eb/eb_summary.json")
put("qct_eb_fragile_sets", eb["fragile_sets_designated"], "results/qct_eb/eb_summary.json")
import numpy as np
P = np.array(eb["prior_fit_primary_univ_gt_500"]["psi"]); M = np.array(eb["prior_fit_primary_univ_gt_500"]["mean_sigma"]); sd = np.sqrt(np.diag(P))
put("qct_eb_prior", {"n_fit": eb["prior_fit_primary_univ_gt_500"]["n_fit"], "psi_sd": sd.round(4).tolist(), "psi_corr_12_13_23": [round(P[0,1]/sd[0]/sd[1], 3), round(P[0,2]/sd[0]/sd[2], 3), round(P[1,2]/sd[1]/sd[2], 3)], "mean_sampling_sd": np.sqrt(np.diag(M)).round(4).tolist()}, "results/qct_eb/eb_summary.json")
kt = R("qct_known_truth/known_truth_summary.json")
for des in ("T1", "T2"):
    for sc in ("national", "cap_bound"):
        d = kt[des][sc]
        put(f"qct_kt_{des}_{sc}", {**{x: round(y["mean"], 1) for x, y in d.items() if isinstance(y, dict) and "mean" in y}, "true_misclassified_sd": round(d["true_misclassified"]["sd"], 1), "flat_band_precision": round(d["flat_band_precision"], 3), "flat_band_recall": round(d["flat_band_recall"], 3), "eb_band_precision": round(d["eb_band_precision"], 3), "eb_band_recall": round(d["eb_band_recall"], 3), "brier_flat": round(d["flat_score"]["brier"], 5), "brier_eb": round(d["eb_score"]["brier"], 5)}, "results/qct_known_truth/known_truth_summary.json")
pp = R("pp_conjunctive.json")
put("pp_flag_rebuild", {x: {"n": y["n"], "agree": y["agree"]} for x, y in pp["flag_rebuild"].items()}, "results/pp_conjunctive.json")
put("pp_live_universe", pp["live_universe"], "results/pp_conjunctive.json")
put("pp_census1990_repairs", {"fips_repairs": len(pp["census1990_parse"]["fips_repairs"]), "rate_count_inconsistent_rows": len(pp["census1990_parse"]["rate_count_inconsistent_rows"])}, "results/pp_conjunctive.json")
for key, d in pp["live_rule"].items():
    put(f"pp_live_{key}", {x: (round(y, 4) if isinstance(y, float) else y) for x, y in d.items() if not isinstance(y, dict)} | ({"average_rule": {x: (round(y, 3) if isinstance(y, float) else y) for x, y in d["average_rule"].items()}} if "average_rule" in d else {}), "results/pp_conjunctive.json")
for key, d in pp["saipe_k4"].items():
    if isinstance(d, dict):
        put(f"pp_k4_{key}", {x: (round(y, 4) if isinstance(y, float) else y) for x, y in d.items()}, "results/pp_conjunctive.json")
put("pp_eb_rules", {r: {lab: {x: (round(y, 2) if isinstance(y, float) else y) for x, y in dd.items()} for lab, dd in v.items()} for r, v in pp["empirical_bayes"]["rules_under_both_priors"].items()}, "results/pp_conjunctive.json")
put("pp_eb_summary", {x: pp["empirical_bayes"][x] for x in ("flat_closed_form", "flat_arcsine_mc", "eb", "fragile_overlap_designated")}, "results/pp_conjunctive.json")
put("pp_eb_prior", {"psi_sd": [round(x, 4) for x in pp["empirical_bayes"]["psi_sd"]], "psi_corr_row1": [round(x, 3) for x in pp["empirical_bayes"]["psi_corr"][0]], "psi_corr_34": round(pp["empirical_bayes"]["psi_corr"][2][3], 3), "mean_sampling_sd_y": [round(x, 4) for x in pp["empirical_bayes"]["mean_sampling_sd_y"]]}, "results/pp_conjunctive.json")
for des, d in pp["known_truth"].items():
    put(f"pp_kt_{des}", {x: (round(y, 3) if isinstance(y, float) else y) for x, y in d.items() if not isinstance(y, dict)} | {"brier_flat": round(d["flat"]["brier"], 5), "brier_eb": round(d["eb"]["brier"], 5)}, "results/pp_conjunctive.json")
mu = json.load(open(IMU))["score_qualified_only"]
put("mua_canonical", {"n": mu["n"], "fragile_ge_0p10": mu["fragile_ge_0.10"], "fragile_ge_0p25": mu["fragile_ge_0.25"], "pov_only_ge_0p10": mu["pov_only"]["n_ge_0.10"], "fragile_rural": mu["fragile_rural"], "mean_flip_all": round(mu["all"]["mean_flip"], 4), "ndraw": json.load(open(IMU))["ndraw"]}, IMU)
put("mua_anchor_designation", R("mua_anchor_designation.json"), "results/mua_anchor_designation.json")
put("cra", s["cra"], "results/small_checks.json")
put("hubzone", s["hubzone_model_r"] | {"nonmetro_counties": s["hubzone_nonmetro_counties"]}, "results/small_checks.json")
put("estimand_closed_forms", s["estimand_closed_forms"], "results/small_checks.json")
put("common_metrics_inputs", s["common_metrics_inputs"], "results/small_checks.json")
put("mua_controlled_p65", s["mua_controlled_p65"], "results/small_checks.json")
put("qct_hud_rounding_check", s["qct_hud_rounding_check"], "results/small_checks.json")
put("mua_score_qualified_le_60", s["mua_score_qualified_le_60"], "results/small_checks.json")
_i, _e, _r, _b = (c["runs"][x]["national"]["expected_two_sided"] for x in ("indep", "elig_real_rank_indep", "elig_indep_rank_real", "realistic"))
put("qct_channel_split_national_churn", {"independent": round(_i, 1), "eligibility_channel_alone_increase": round(_e - _i, 1),
    "rank_channel_alone_increase": round(_r - _i, 1), "both_increase": round(_b - _i, 1),
    "interaction": round((_b - _i) - (_e - _i) - (_r - _i), 1)}, "results/qct_canonical_summary.json")
json.dump(k, open(os.path.join(W, "results", "key_numbers.json"), "w"), indent=1, default=float)
print(len(k), "keys written")
