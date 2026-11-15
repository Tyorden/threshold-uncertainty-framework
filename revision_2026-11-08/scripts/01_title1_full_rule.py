#!/usr/bin/env python
"""
01_title1_full_rule.py  (revision 2026-11-08, flagship threshold paper)

Title I-A Concentration-grant eligibility evaluated on the FULL rule, one
draw of the formula-child count per district (triage R1-W8, R1-Q2, R1-Q3,
R2-W4, R2-Q4, R2-Q5).

Rule (CRS R48890 wording, recorded in TITLE1_saipe/notes/thresholds.md):
  Basic:          FC >= 10 AND FC > 0.02 * N
  Concentration:  Basic AND (FC > 6,500 OR FC > 0.15 * N)
Every clause is of the form "FC above a threshold", so with N fixed
  eligible  <=>  FC > T,   T = max(10, 0.02 N, min(6500, 0.15 N)).
One count drives all clauses, so the earlier prong-by-prong scoring and the
product-of-prongs "joint" (TITLE1_saipe/scripts/03_flip_probability.py
lines 59-105) are replaced by this single threshold.

Error model: FC ~ Normal(FC_hat, (k * SE)^2), SE = A * FC_hat^B, the power
curve fitted by OLS of log SE on log count over the SAIPE 2023 county file
(TITLE1_saipe/results/se_model.json). The curve is a CENTRAL fit, not a
lower bound (diagnostics below); k in {0.5, 1.0, 1.5} is the SE-scale
sensitivity. With N fixed the eligibility probability has a closed form,
so the main figures carry no Monte Carlo error; a Monte Carlo cross-check
is run anyway. A second Monte Carlo adds error in the rate denominator N at
ASSUMED coefficients of variation (no published error exists for N).

Inputs (read only): TITLE1_saipe/results/districts_classified.csv,
  TITLE1_saipe/results/se_model.json, TITLE1_saipe/results/county_se.csv,
  TITLE1_saipe/results/flip_probabilities.csv (old prong-by-prong scores).
Outputs: results/title1_* (this folder).
"""
import json, os, sys, time
import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.dirname(HERE)
RES = os.path.join(W, "results")
import os as _os  # public copy: the program folders (see README) are found through PAPER_PREP
PAPER_PREP = _os.environ.get("PAPER_PREP", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "program_folders"))
SRC = PAPER_PREP + "/TITLE1_saipe/results"
SEED = 20261108

t0 = time.time()
df = pd.read_csv(os.path.join(SRC, "districts_classified.csv"), dtype={"leaid": str})
M = json.load(open(os.path.join(SRC, "se_model.json")))
A, B = M["A"], M["B"]

d = df[df["formula_children"].notna() & (df["pop_5_17"] > 0)].copy().reset_index(drop=True)
FC = d["formula_children"].to_numpy(float)
N = d["pop_5_17"].to_numpy(float)
SE1 = A * np.power(np.clip(FC, 1, None), B)

T_basic = np.maximum(10.0, 0.02 * N)
T_conc_or = np.minimum(6500.0, 0.15 * N)
T = np.maximum(T_basic, T_conc_or)
binding = np.where(T == T_basic, np.where(T_basic == 10.0, "basic_count_10", "basic_rate_2pct"),
                   np.where(T_conc_or == 6500.0, "conc_count_6500", "conc_rate_15pct"))
point_elig = FC > T

out = {"universe": "SAIPE 2023 school districts with a formula-child count and "
                   "positive 5-17 population",
       "n_districts": int(len(d)), "n_records_in_file": int(len(df)),
       "rule": "eligible iff FC > max(10, 0.02N, min(6500, 0.15N)); N fixed",
       "se_model": {"A": A, "B": B},
       "point_estimate_eligible": int(point_elig.sum()),
       "by_se_scale": {}}

per = d[["leaid", "name", "state_fips", "total_pop", "pop_5_17", "formula_children", "rate"]].copy()
per["T"] = T
per["binding_threshold"] = binding
per["point_eligible"] = point_elig
per["se_fc_k1"] = SE1

for k in (0.5, 1.0, 1.5):
    se = k * SE1
    p = norm.sf((T - FC) / se)                    # P(eligible)
    phi = np.where(point_elig, 1 - p, p)          # flat-prior P(true side differs)
    ind = (p > 0.10) & (p < 0.90)
    knife = (p > 0.40) & (p < 0.60)
    by_bind = {b: int((ind & (binding == b)).sum()) for b in np.unique(binding)}
    by_bind_knife = {b: int((knife & (binding == b)).sum()) for b in np.unique(binding)}
    out["by_se_scale"][str(k)] = dict(
        indeterminate_0p10_0p90=int(ind.sum()),
        indeterminate_share=float(ind.mean()),
        knife_edge_0p40_0p60=int(knife.sum()),
        indeterminate_by_binding_threshold=by_bind,
        knife_by_binding_threshold=by_bind_knife,
        expected_misclassified_flat_prior=float(phi.sum()),
        phi_ge_0p10=int((phi >= 0.10).sum()),
        phi_ge_0p25=int((phi >= 0.25).sum()),
        designated_with_phi_ge_0p10=int(((phi >= 0.10) & point_elig).sum()),
        share_designated_with_phi_ge_0p10=float(((phi >= 0.10) & point_elig).sum() / point_elig.sum()),
        expected_changes_share_of_list=float(phi.sum() / point_elig.sum()),
        expected_losses=float(phi[point_elig].sum()),
        expected_gains=float(phi[~point_elig].sum()),
        pop_5_17_in_indeterminate=float(N[ind].sum()),
        total_pop_in_indeterminate=float(d["total_pop"].to_numpy(float)[ind].sum()),
    )
    per[f"p_elig_k{k}"] = p
    per[f"phi_k{k}"] = phi

# ---- the old prong-by-prong count cliff: what the full rule says about its 74
old = pd.read_csv(os.path.join(SRC, "flip_probabilities.csv"), dtype={"leaid": str})
old["leaid"] = old["leaid"].str.zfill(7)
old74 = old[(old["p_conc_count"] > 0.10) & (old["p_conc_count"] < 0.90)]
m74 = per.set_index("leaid").loc[old74["leaid"]]
ind1 = (m74["p_elig_k1.0"] > 0.10) & (m74["p_elig_k1.0"] < 0.90)
out["old_count_cliff_74"] = dict(
    n=int(len(old74)),
    point_rate_above_15pct=int((m74["rate"] > 0.15).sum()),
    full_rule_indeterminate=int(ind1.sum()),
    full_rule_indeterminate_count_cliff_binding=int((ind1 & (m74["binding_threshold"] == "conc_count_6500")).sum()),
)
old_rate = old[(old["p_conc_rate"] > 0.10) & (old["p_conc_rate"] < 0.90)]
out["old_rate_cliff_coinflips"] = int(len(old_rate))
out["old_rate_cliff_knife"] = int(((old["p_conc_rate"] > 0.40) & (old["p_conc_rate"] < 0.60)).sum())

# ---- exhibits
ex = per[per["leaid"].isin(["4202280", "5307710"])].copy()
ex["z_k1"] = (ex["formula_children"] - ex["T"]) / ex["se_fc_k1"]
ex["p_rate_prong_15pct_k1"] = norm.sf((0.15 * ex["pop_5_17"] - ex["formula_children"]) / ex["se_fc_k1"])
ex["p_count_prong_6500_k1"] = norm.sf((6500 - ex["formula_children"]) / ex["se_fc_k1"])
ex.to_csv(os.path.join(RES, "title1_exhibits.csv"), index=False)
out["exhibits"] = ex[["leaid", "name", "total_pop", "pop_5_17", "formula_children", "rate",
                      "T", "binding_threshold", "se_fc_k1", "p_elig_k1.0", "p_elig_k0.5",
                      "p_elig_k1.5", "p_rate_prong_15pct_k1", "p_count_prong_6500_k1"]
                     ].to_dict(orient="records")

# ---- Monte Carlo cross-check of the closed form (k = 1), N fixed
rng = np.random.default_rng(SEED)
R = 20000
hits = np.zeros(len(d))
for s in range(0, R, 1000):
    Z = rng.standard_normal((len(d), 1000))
    fc_d = FC[:, None] + Z * SE1[:, None]
    hits += (fc_d > T[:, None]).sum(axis=1)
p_mc = hits / R
p_cf = per["p_elig_k1.0"].to_numpy()
mcse = np.sqrt(p_mc * (1 - p_mc) / R)
ind_mc = (p_mc > 0.10) & (p_mc < 0.90)
out["mc_crosscheck_k1"] = dict(
    draws=R, seed=SEED,
    max_abs_diff_closed_form=float(np.max(np.abs(p_mc - p_cf))),
    max_mcse=float(mcse.max()),
    indeterminate_mc=int(ind_mc.sum()),
    indeterminate_closed_form=out["by_se_scale"]["1.0"]["indeterminate_0p10_0p90"],
)

# ---- rate-denominator error at ASSUMED CVs (sensitivity only; N has no published error)
den = {}
for cv in (0.01, 0.03, 0.05):
    rng = np.random.default_rng(SEED + int(cv * 1000))
    hits = np.zeros(len(d))
    for s in range(0, R, 1000):
        Zf = rng.standard_normal((len(d), 1000))
        Zn = rng.standard_normal((len(d), 1000))
        fc_d = FC[:, None] + Zf * SE1[:, None]
        n_d = np.clip(N[:, None] * (1 + cv * Zn), 1, None)
        T_d = np.maximum(np.maximum(10.0, 0.02 * n_d), np.minimum(6500.0, 0.15 * n_d))
        hits += (fc_d > T_d).sum(axis=1)
    p = hits / R
    ind = (p > 0.10) & (p < 0.90)
    den[str(cv)] = dict(indeterminate_0p10_0p90=int(ind.sum()),
                        knife_edge_0p40_0p60=int(((p > 0.40) & (p < 0.60)).sum()),
                        draws=R, seed=SEED + int(cv * 1000))
out["denominator_error_sensitivity_k1"] = den
out["denominator_error_note"] = ("CV of N is ASSUMED (1%, 3%, 5%), independent of FC error; "
                                 "SAIPE publishes no error for the 5-17 population.")

# ---- SE-curve diagnostics (central fit, not a floor)
c = pd.read_csv(os.path.join(SRC, "county_se.csv"))
m = (c["est"] > 0) & (c["se"] > 0)
pred = A * c.loc[m, "est"] ** B
resid = np.log(c.loc[m, "se"]) - np.log(pred)
out["se_fit_diagnostics"] = dict(
    n_counties=int(m.sum()),
    share_counties_above_curve=float((c.loc[m, "se"] > pred).mean()),
    residual_sd_log=float(resid.std(ddof=2)),
    median_abs_rel_error=float((np.abs(pred - c.loc[m, "se"]) / c.loc[m, "se"]).median()),
    note="OLS of log SE on log count; a central curve, roughly half the counties lie above it.",
)

per.to_csv(os.path.join(RES, "title1_full_rule_districts.csv"), index=False)
json.dump(out, open(os.path.join(RES, "title1_full_rule_summary.json"), "w"), indent=2)
print(json.dumps(out, indent=2, default=str))
print(f"elapsed {time.time()-t0:.1f}s")
