#!/usr/bin/env python
"""11_small_checks.py: smaller regenerated numbers for the revision.

(a) CRA: flip probability across the LMI line (80 percent of area MFI) from the
    pilot's per-tract class probabilities; universe reconciliation.
(b) HUBZone: Model R (revision-calibrated normal error on the qualifying ratio,
    sigma 0.1215) applied to every nonmetropolitan county, not only the band.
(c) Title I: small LEAs (total population under 20,000) among the
    indeterminate districts of 01 (34 CFR 200.74 alternative-data option).
(d) QCT: observed year-over-year change between HUD's 2025 and 2026 lists.
(e) Estimand closed forms for a single threshold (arithmetic checks),
    verified by simulation.
"""
import json, os
import numpy as np, pandas as pd
from scipy.stats import norm
from scipy import integrate

W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import os as _os  # public copy: the program folders (see README) are found through PAPER_PREP
PAPER_PREP = _os.environ.get("PAPER_PREP", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "program_folders"))
PP = PAPER_PREP + ""
out = {}

# (a) CRA ------------------------------------------------------------------
c = pd.read_csv(f"{PP}/A1_ffiec_cra/results/flip_pilot_per_tract.csv", dtype={"GEOID": str})
lmi_base = c["inc_ind"].isin([1, 2]).to_numpy()
p_lmi = (c["p_low"] + c["p_moderate"]).to_numpy()
f_lmi = np.where(lmi_base, 1 - p_lmi, p_lmi)
R_CRA = 20000
out["cra"] = dict(
    n_tracts=int(len(c)), draws=R_CRA,
    mean_flip_four_class=float(c["p_flip_fixed_denom"].mean()),
    mean_flip_lmi_line=float(f_lmi.mean()),
    expected_lmi_flips=float(f_lmi.sum()),
    mean_flip_lmi_line_lmi_tracts=float(f_lmi[lmi_base].mean()),
    mean_flip_lmi_line_non_lmi_tracts=float(f_lmi[~lmi_base].mean()),
    n_lmi_tracts=int(lmi_base.sum()),
    lmi_flip_ge_0p10=int((f_lmi >= 0.10 - 1e-9).sum()),
    max_mc_se_per_tract=float(np.sqrt(0.25 / R_CRA)),
    universe_note="the bridge check ran on all 2,287 merged pilot tracts; the flip pilot keeps the 2,264 "
                  "with a usable MOE, a positive area MFI and a tract MFI (A1_ffiec_cra/scripts/03_flip_pilot.py lines 52-54)")

# (b) HUBZone ----------------------------------------------------------------
x = pd.read_excel(f"{PP}/A3_hubzone_laus/data/county-designations-july-2023.xlsx")
cols = list(x.columns)
out["hubzone_columns_seen"] = [str(k) for k in cols]
st = [k for k in cols if "State Unemployment Ratio" in str(k)]
us = [k for k in cols if "US Unemployment Ratio" in str(k)]
met = [k for k in cols if "etro" in str(k)]
out["hubzone_ratio_columns"] = dict(state=[str(k) for k in st], us=[str(k) for k in us], metro=[str(k) for k in met])
band = pd.read_csv(f"{PP}/A3_hubzone_laus/data/band_counties.csv")
out["hubzone_band_file_rows"] = int(len(band))
if st and us:
    nonmetro = x["2022 OMB Delineation"].astype(str).str.lower().str.replace("-", " ").str.startswith("non metropolitan")
    out["hubzone_nonmetro_counties"] = int(nonmetro.sum())
    x = x[nonmetro].copy()
    maxr = x[[st[0], us[0]]].apply(pd.to_numeric, errors="coerce").max(axis=1)
    out["hubzone_counties_with_ratio"] = int(maxr.notna().sum())
    sigma = 0.1215
    phi = norm.sf(np.abs(maxr - 1.40) / sigma)
    inb = (maxr >= 1.20) & (maxr <= 1.60)
    out["hubzone_model_r"] = dict(
        sigma=sigma, counties_with_ratio=int(maxr.notna().sum()),
        phi_ge_0p10=int((phi >= 0.10 - 1e-9).sum()), phi_ge_0p25=int((phi >= 0.25 - 1e-9).sum()),
        phi_ge_0p10_outside_band=int(((phi >= 0.10 - 1e-9) & ~inb).sum()),
        max_phi_outside_band=float(phi[~inb & maxr.notna()].max()),
        expected_misclassified_all=float(np.nansum(phi)),
        expected_misclassified_outside_band=float(np.nansum(phi[~inb])),
        band_counties=int(inb.sum()),
        share_phi_ge_0p10_of_nonmetro=float((phi >= 0.10 - 1e-9).sum() / len(x)),
        note="universe: nonmetropolitan counties (2022 OMB delineation) in SBA's July 2023 file")

# (c) Title I ----------------------------------------------------------------
t = pd.read_csv(os.path.join(W, "results", "title1_full_rule_districts.csv"), dtype={"leaid": str})
ind = (t["p_elig_k1.0"] > 0.10) & (t["p_elig_k1.0"] < 0.90)
out["title1_small_lea"] = dict(indeterminate=int(ind.sum()),
                               indeterminate_total_pop_lt_20000=int((ind & (t["total_pop"] < 20000)).sum()),
                               all_districts_total_pop_lt_20000=int((t["total_pop"] < 20000).sum()))

# (d) QCT year over year -------------------------------------------------------
q26 = pd.read_parquet(os.path.join(W, "cache", "qct_data_2026.parquet"))[["qct_id", "qct"]]
q25 = pd.read_parquet(os.path.join(W, "cache", "qct_data_2025.parquet"))[["qct_id", "qct"]]
m = q26.merge(q25, on="qct_id", suffixes=("_26", "_25"), how="outer", indicator=True)   # qct_id is unique; GEOID repeats for New England split tracts
both = m["_merge"] == "both"
a, b = m["qct_25"] == 1, m["qct_26"] == 1
out["qct_year_over_year"] = dict(
    tracts_2026=int(len(q26)), tracts_2025=int(len(q25)), matched=int(both.sum()), unmatched=int((~both).sum()),
    designated_2025=int(a.sum()), designated_2026=int(b.sum()),
    in_both=int((a & b).sum()), dropped_2025_to_2026=int((a & ~b & both).sum()),
    added_2025_to_2026=int((~a & b & both).sum()),
    turnover_share_of_2026_list=float(((a & ~b & both).sum() + (~a & b & both).sum()) / b.sum()))

# (d2) year-over-year turnover under the rule in force and under the latest vintage alone
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qct_fast as qf
d26 = pd.read_parquet(os.path.join(W, "cache", "qct_data_2026.parquet"))
d25 = pd.read_parquet(os.path.join(W, "cache", "qct_data_2025.parquet"))
ren = {}
for c in d25.columns:
    for a, b_ in (("_22", "_23"), ("_21", "_22"), ("_20", "_21")):
        if c.endswith(a) and not c.startswith("VLIL"):
            ren[c] = c[: -len(a)] + b_
            break
d25r = d25.rename(columns=ren)          # oldest/middle/latest vintages mapped to the engine's positions
D26, D25 = qf.QCTData(d26), qf.QCTData(d25r)
yy = {}
for v in ("hud_2of3", "latest_only"):
    b26, _ = qf.run(D26, rounding=True, baseline_hud_columns=False, variant=v)
    b25, _ = qf.run(D25, rounding=True, baseline_hud_columns=False, variant=v)
    s25 = pd.Series(b25, index=d25r["qct_id"].to_numpy()); s26 = pd.Series(b26, index=d26["qct_id"].to_numpy())
    j = s25.index.intersection(s26.index)
    s25, s26 = s25.loc[j], s26.loc[j]
    yy[v] = dict(designated_2025=int(s25.sum()), designated_2026=int(s26.sum()),
                 dropped=int((s25 & ~s26).sum()), added=int((~s25 & s26).sum()),
                 turnover_share_of_2026_list=float(((s25 & ~s26).sum() + (~s25 & s26).sum()) / s26.sum()))
rep25, _ = qf.run(D25, rounding=True, baseline_hud_columns=True)
yy["replication_2025_match_hud"] = int((rep25 == (D25.qct == 1)).sum())
yy["n_2025"] = int(D25.n)
yy["note"] = ("2025 file vintages 2016-20, 2017-21, 2018-22; 2026 file 2017-21, 2018-22, 2019-23; "
              "turnover mixes true change, income-limit updates and sampling error")
out["qct_year_over_year_by_rule"] = yy

# (f) MUA pilot: age-65+ shares coded as controlled (negative MOE code) get no error
mp = pd.read_csv(f"{PP}/IMU_underserved/results/pilot_flip_probs.csv")
out["mua_controlled_p65"] = dict(pilot_areas=int(len(mp)), controlled_p65_moe_code=int((mp["acs_p65_moe"] < 0).sum()),
                                 score_qualified=int((mp["imu"] <= 62.0).sum()),
                                 controlled_among_score_qualified=int(((mp["acs_p65_moe"] < 0) & (mp["imu"] <= 62.0)).sum()))

# (g) common metrics: listed units, share of listed with phi >= 0.10, expected changes as a share of the list
cm = {}
lst = t["point_eligible"].astype(bool)
phi_t = t["phi_k1.0"]
cm["title1"] = dict(listed=int(lst.sum()), listed_phi_ge_0p10=int(((phi_t >= 0.10 - 1e-9) & lst).sum()),
                    pop517_listed_phi_ge_0p10=float(t.loc[(phi_t >= 0.10 - 1e-9) & lst, "pop_5_17"].sum()),
                    total_pop_listed_phi_ge_0p10=float(t.loc[(phi_t >= 0.10 - 1e-9) & lst, "total_pop"].sum()),
                    expected_changes=float(phi_t.sum()))
cm["cra_lmi_line"] = dict(listed_lmi=int(lmi_base.sum()), listed_phi_ge_0p10=int(((f_lmi >= 0.10 - 1e-9) & lmi_base).sum()),
                          expected_changes=float(f_lmi.sum()))
if st and us:
    qual = x["Qualified by Unemplyoment?"].astype(str).str.lower().isin(["true", "yes", "1"]).to_numpy()
    cm["hubzone_unemployment"] = dict(nonmetro=int(len(x)), listed=int(qual.sum()),
                                      listed_phi_ge_0p10=int(((np.asarray(phi) >= 0.10 - 1e-9) & qual).sum()),
                                      expected_changes=float(np.nansum(phi)))
cc = json.load(open(f"{PP}/IMU_underserved/results/context_counts.json"))
mr = json.load(open(f"{PP}/IMU_underserved/results/mc_report_v2.json"))["score_qualified_only"]
cm["mua"] = dict(listed=mr["n"], listed_phi_ge_0p10=mr["fragile_ge_0.10"],
                 expected_changes=float(mr["all"]["mean_flip"] * mr["n"]),
                 designation_population_fragile=cc["fragile_designation_population_sum"],
                 designation_population_listed=cc["score_qualified_designation_population_sum"])
out["common_metrics_inputs"] = cm

# (h) HUD's stored poverty rate against candidate rounding rules, reliable tracts per vintage
q = pd.read_parquet(os.path.join(W, "cache", "qct_data_2026.parquet"))
def _moer_lt(e, m):
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.abs(m) / np.abs(e)
    r = r.replace([np.inf, -np.inf], np.nan)
    return r < 0.5
rnd = {}
for y in ("21", "22", "23"):
    u, um, pp_, pm = q[f"B17001est1_{y}"], q[f"B17001me1_{y}"], q[f"B17001est2_{y}"], q[f"B17001me2_{y}"]
    rel = _moer_lt(u, um) & _moer_lt(pp_, pm)
    x = 1000 * pp_ / u
    m = rel & x.notna() & q[f"pov_rate_{y}"].notna()
    hud = q.loc[m, f"pov_rate_{y}"]
    rnd[y] = dict(reliable_tracts=int(m.sum()),
                  mismatch_half_up=int((np.floor(x[m] + 0.5) != hud).sum()),
                  mismatch_half_even=int((np.round(x[m]) != hud).sum()),
                  mismatch_floor=int((np.floor(x[m]) != hud).sum()),
                  mismatch_ceil=int((np.ceil(x[m]) != hud).sum()),
                  unreliable_with_stored_zero=int(((~rel) & q[f"pov_rate_{y}"].notna() & (q[f"pov_rate_{y}"] == 0)).sum()),
                  unreliable_total=int(((~rel) & q[f"pov_rate_{y}"].notna()).sum()))
out["qct_hud_rounding_check"] = rnd
# (i) MUA: score-qualified areas at IMU 60.0 or less (bands of mc_report_v2.json)
bands = json.load(open(f"{PP}/IMU_underserved/results/mc_report_v2.json"))["score_qualified_only"]["flip_by_imu_band"]
out["mua_score_qualified_le_60"] = int(sum(v["n"] for k, v in bands.items() if k in ("<= 50.0", "(50.0,55.0]", "(55.0,58.0]", "(58.0,60.0]")))

# (e) estimand closed forms ---------------------------------------------------
rng = np.random.default_rng(20261112)
Z = rng.standard_normal(2_000_000)
on_line = norm.sf(np.abs(Z))                    # phi_hat for a unit whose truth is on the line
two_se = norm.sf(np.abs(2 + Z))                  # truth 2 SE above
e = {}
e["on_line_E_phi_hat"] = float(on_line.mean()); e["on_line_P_phi_hat_lt_0p10"] = float((on_line < 0.10).mean())
e["on_line_true_wrong_side_prob"] = 0.5
e["two_se_E_phi_hat"] = float(two_se.mean()); e["two_se_P_phi_hat_ge_0p10"] = float((two_se >= 0.10 - 1e-9).mean())
e["two_se_E_phi_hat_quadrature"] = float(integrate.quad(lambda z: norm.sf(abs(2 + z)) * norm.pdf(z), -12, 12)[0])
e["two_se_published_wrong_prob"] = float(norm.sf(2.0))
e["two_se_two_survey_disagreement"] = float(2 * norm.cdf(2.0) * norm.sf(2.0))
# truths spread N(c, tau^2 sigma^2): mean phi_hat vs two-survey disagreement and misclassification
taus = [0.25, 0.5, 1.0, 2.0, 4.0]
tab = []
for tau in taus:
    th = tau * rng.standard_normal(1_000_000)
    est = th + rng.standard_normal(1_000_000)
    phi_hat = norm.sf(np.abs(est))
    miscl = (np.sign(est) != np.sign(th)).mean()
    disagree = (2 * norm.cdf(th) * norm.sf(th)).mean()
    tab.append(dict(tau=tau, mean_phi_hat=float(phi_hat.mean()),
                    closed_mean_phi_hat=float(np.arctan(1 / np.sqrt(1 + tau ** 2)) / np.pi),
                    misclassification_rate=float(miscl),
                    two_survey_disagreement=float(disagree),
                    closed_disagreement=float(np.arccos(tau ** 2 / (1 + tau ** 2)) / np.pi),
                    ratio_disagreement_to_mean_phi_hat=float(disagree / phi_hat.mean())))
e["spread_truths"] = tab
e["note"] = ("mean phi_hat equals the misclassification rate (posterior calibration in aggregate under a "
             "locally flat density); the two-survey disagreement is larger by a factor between sqrt(2) and 2")
out["estimand_closed_forms"] = e
json.dump(out, open(os.path.join(W, "results", "small_checks.json"), "w"), indent=2, default=float)
print(json.dumps(out, indent=2, default=float))
