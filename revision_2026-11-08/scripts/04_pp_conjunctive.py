#!/usr/bin/env python
"""04_pp_conjunctive.py: persistent-poverty counties, corrected analysis
(triage R1-W2, R1-W3, R1-W4, R1-Q4, R1-Q5, R1-Q10, R2-W2, R2-W3, R2-Q7).

1. Rebuild the 1990 and 2000 high-poverty flags from the Census Bureau's
   county poverty tables (CPH-L-100 for 1990; the 2000 county table) and
   compare with ERS's HiPov90c / HiPov00c; test the rounding convention
   (rate rounded to one decimal before the 20.0 test) on all four periods.
2. The live ERS rule with FOUR error-bearing periods: 1990 and 2000 long-form
   sampling error from the Census 2000 SF3 technical documentation formula
   SE(p) = DF * sqrt((5/B) p (100 - p)) (Table B footnote; 5 = 6 - 1 for the
   1-in-6 sample), design factor DF bracketed at 1.0, 1.5, 2.0 because the
   poverty-specific design-factor table (Table C) was not retrievable; ACS
   2007-11 and 2017-21 from published B17001 MOEs (Census proportion formula).
   Errors independent across periods (no shared sample years) with a
   sensitivity at an assumed equicorrelation of 0.3.
   Both error directions, the single-period comparator (2017-21 alone) and
   the rule applied to the four-period average.
3. The SAIPE K = 4 analogue (1989, 1999, 2011, 2021) at rho 0, 0.3, 0.6, 0.9,
   both directions (a bracket; no claim that rho = 0 is realistic).
4. Empirical Bayes: multivariate Fay-Herriot on the arcsine-square-root
   scale, Census-division mean vectors, 4 x 4 between-county covariance by
   moments; posterior misclassification probabilities next to flat-prior ones.
5. Known-truth calibration: truth from the EB posterior (T1) or equal to the
   published estimates (T2); synthetic estimates; flat and EB probabilities
   scored against the true misclassification.
Inputs read only from A4_persistent_poverty/data and this folder's cache/.
"""
import os, json, time
import numpy as np, pandas as pd
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__)); W = os.path.dirname(HERE)
RES = os.path.join(W, "results"); CACHE = os.path.join(W, "cache")
import os as _os  # public copy: the program folders (see README) are found through PAPER_PREP
PAPER_PREP = _os.environ.get("PAPER_PREP", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "program_folders"))
A4 = PAPER_PREP + "/A4_persistent_poverty/data"
SEED = 20261108
T = 20.0
t0 = time.time()
out = {}


def num(s):
    return pd.to_numeric(s, errors="coerce")


# ------------------------------------------------------------ ERS universe
ers = pd.read_csv(os.path.join(A4, "ers_pam_sept2025.csv"), dtype=str)
cty = ers[["STUSAB", "fipstxt", "CountyName", "HiPov90c", "HiPov00c", "HIPOV0711c",
           "HiPov1721c", "PerPov1721c"]].drop_duplicates("fipstxt").reset_index(drop=True)
cty["fips"] = cty["fipstxt"].str.zfill(5)
for c in ["HiPov90c", "HiPov00c", "HIPOV0711c", "HiPov1721c", "PerPov1721c"]:
    cty[c] = num(cty[c])
cty["cname"] = cty["CountyName"].str.split(",").str[0].str.strip().str.lower()
name_key = {(r.STUSAB, r.cname): r.fips for r in cty.itertuples()}
st_of_fips = cty.groupby(cty["fips"].str[:2])["STUSAB"].first().to_dict()

# ------------------------------------------------------------ census tables
x90 = pd.read_excel(os.path.join(CACHE, "census_pov_1990_by_county.xls"), header=None, skiprows=6)
x90.columns = ["state", "county", "fips_raw", "total", "npov", "rate", "crank", "srank"]
x90 = x90[x90["fips_raw"].notna()].reset_index(drop=True)
for c in ["fips_raw", "total", "npov", "rate"]:
    x90[c] = num(x90[c])
x90["fips"] = x90["fips_raw"].astype(int).astype(str).str.zfill(5)
repairs = []
dup = x90["fips"].duplicated(keep=False)
bad_state = x90.apply(lambda r: st_of_fips.get(r["fips"][:2]) != r["state"], axis=1)
for i in x90.index[dup | bad_state]:
    key = (x90.at[i, "state"], str(x90.at[i, "county"]).strip().lower())
    new = name_key.get(key)
    if new and new != x90.at[i, "fips"]:
        repairs.append(dict(state=key[0], county=x90.at[i, "county"], fips_in_file=x90.at[i, "fips"], fips_from_ers_name=new))
        x90.at[i, "fips"] = new
# rate column: consistent with the published ranks except one row (Marion County TX,
# rate field 60.6 tied at rank 178 with 30.6 counties; counts give 30.6)
cnt_rate = 100 * x90["npov"] / x90["total"]
incons = x90[(cnt_rate - x90["rate"]).abs() > 0.051][["state", "county", "fips", "total", "npov", "rate", "crank"]]
consistent = (cnt_rate - x90["rate"]).abs() <= 0.051
x90["rate_use"] = np.where(consistent, cnt_rate, x90["rate"])     # unrounded where the counts agree
m_marion = x90["fips"] == "48315"
x90.loc[m_marion, "rate_use"] = cnt_rate[m_marion]
out["census1990_parse"] = dict(rows=int(len(x90)), fips_repairs=repairs,
                               rate_count_inconsistent_rows=incons.to_dict(orient="records"),
                               rule="unrounded count ratio where it agrees with the one-decimal rate column (within 0.05); "
                                    "otherwise the rate column (it matches the published ranks), except Marion County TX, "
                                    "where the rate field (60.6) is the typo and the count ratio (30.6) matches its rank",
                               duplicated_after_repair=int(x90["fips"].duplicated().sum()))

x00 = pd.read_excel(os.path.join(CACHE, "census_pov_2000_by_county.xls"), header=None, skiprows=5)
x00.columns = ["state", "county", "fips_raw", "total", "npov", "rate", "staterate"]
x00 = x00[x00["fips_raw"].notna() & x00["state"].notna()].reset_index(drop=True)
for c in ["fips_raw", "total", "npov", "rate"]:
    x00[c] = num(x00[c])
x00["fips"] = x00["fips_raw"].astype(int).astype(str).str.zfill(5)
x00["rate_raw"] = 100 * x00["npov"] / x00["total"]
out["census2000_parse"] = dict(rows=int(len(x00)), duplicated=int(x00["fips"].duplicated().sum()),
                               max_abs_rate_vs_counts=float((x00["rate_raw"] - x00["rate"]).abs().max()))

# ------------------------------------------------------------ ACS and SAIPE
def acs(yr):
    d = pd.read_csv(os.path.join(A4, f"acs_b17001_{yr}.csv"), dtype=str)
    d["fips"] = d["fips"].str.zfill(5)
    below, univ = num(d[f"b17_below_{yr}"]), num(d[f"b17_univ_{yr}"])
    p = below / univ
    mn, md = num(d[f"b17_below_moe_{yr}"]), num(d[f"b17_univ_moe_{yr}"])
    rad = mn ** 2 - (p ** 2) * md ** 2
    rad = np.where(rad < 0, mn ** 2 + (p ** 2) * md ** 2, rad)
    return pd.DataFrame({"fips": d["fips"], f"r{yr}": 100 * p,
                         f"s{yr}": 100 * np.sqrt(rad) / univ / 1.645, f"u{yr}": univ})


def saipe(yr):
    d = pd.read_csv(os.path.join(A4, f"saipe_{yr}.csv"), dtype=str)
    d["fips"] = d["fips"].str.zfill(5)
    return pd.DataFrame({"fips": d["fips"], f"sp{yr}": num(d[f"saipe_pct_{yr}"]),
                         f"ss{yr}": num(d[f"saipe_moe_{yr}"]) / 1.645})


df = cty.merge(x90[["fips", "total", "rate_use", "npov", "rate"]].rename(
    columns={"total": "B90", "rate_use": "r90", "npov": "n90", "rate": "r90_field"}), on="fips", how="left")
df = df.merge(x00[["fips", "total", "rate", "rate_raw"]].rename(
    columns={"total": "B00", "rate": "r00", "rate_raw": "r00_raw"}), on="fips", how="left")
df = df.merge(acs(2011), on="fips", how="left").merge(acs(2021), on="fips", how="left")
for y in (1989, 1999, 2011, 2021):
    df = df.merge(saipe(y), on="fips", how="left")


# ------------------------------------------------------------ 1. flag rebuilds
def agree(flag, ers_col, mask_extra=None):
    m = df[ers_col].isin([0, 1]) & flag.notna()
    if mask_extra is not None:
        m &= mask_extra
    f = flag[m].astype(int); e = df.loc[m, ers_col].astype(int)
    mism = df.loc[m & (flag.fillna(-9).astype(int) != df[ers_col]), ["fips", "CountyName"]]
    return dict(n=int(m.sum()), agree=int((f == e).sum()), mismatches=int((f != e).sum()),
                mismatch_fips=mism["fips"].tolist()[:30])


rb = {}
rb["1990_unrounded_ge20"] = agree((df["r90"] >= T).where(df["r90"].notna()), "HiPov90c")
rb["1990_rate_field_one_decimal_ge20"] = agree((df["r90_field"] >= T).where(df["r90_field"].notna()), "HiPov90c")
rb["2000_rate_one_decimal_ge20"] = agree((df["r00"] >= T).where(df["r00"].notna()), "HiPov00c")
rb["2000_raw_ratio_ge20"] = agree((df["r00_raw"] >= T).where(df["r00_raw"].notna()), "HiPov00c")
rb["2011_raw_ge20"] = agree((df["r2011"] >= T).where(df["r2011"].notna()), "HIPOV0711c")
rb["2011_round1_ge20"] = agree((np.round(df["r2011"], 1) >= T).where(df["r2011"].notna()), "HIPOV0711c")
rb["2021_raw_ge20"] = agree((df["r2021"] >= T).where(df["r2021"].notna()), "HiPov1721c")
rb["2021_round1_ge20"] = agree((np.round(df["r2021"], 1) >= T).where(df["r2021"].notna()), "HiPov1721c")
out["flag_rebuild"] = rb

# ------------------------------------------------------------ 2. live rule, four error-bearing periods
def lf_se(rate, B, DF):
    p = np.clip(rate, 0, 100)
    pe = np.where(p < 2, 2.0, np.where(p > 98, 98.0, p))       # documentation: use the 2-or-98 row
    return DF * np.sqrt(5.0 / B * pe * (100 - pe))


valid = df[["r90", "B90", "r00", "B00", "r2011", "s2011", "r2021", "s2021"]].notna().all(axis=1) & \
        (df["s2011"] > 0) & (df["s2021"] > 0) & df["PerPov1721c"].isin([0, 1])
L = df[valid].reset_index(drop=True)
X = L[["r90", "r00", "r2011", "r2021"]].to_numpy(float)            # ERS tests unrounded rates (flag_rebuild)
Xraw = L[["r90", "r00", "r2011", "r2021"]].to_numpy(float)
d_conj = (X >= T).all(axis=1)
out["live_universe"] = dict(n_counties=int(len(L)),
                            point_designated_rebuilt=int(d_conj.sum()),
                            official_designated_in_universe=int((L["PerPov1721c"] == 1).sum()),
                            rebuilt_vs_official_agree=int((d_conj == (L["PerPov1721c"] == 1)).sum()))
popw = L["u2021"].to_numpy(float)


def orthant_mc(Xc, S, rho, R=None, seed=None, nodes=120):
    """P(all true >= T) per county under equicorrelated (rho >= 0) Gaussian errors,
    computed exactly by one-dimensional Gauss-Hermite quadrature:
    Z_t = sqrt(rho) W + sqrt(1-rho) E_t, so P = E_W[ prod_t Phi((sqrt(rho) W - a_t)/sqrt(1-rho)) ].
    No Monte Carlo error (quadrature error below 1e-10 at 120 nodes)."""
    from numpy.polynomial.hermite_e import hermegauss
    a = (T - Xc) / S
    if rho == 0:
        return np.prod(norm.sf(a), axis=1), "exact"
    x, w = hermegauss(nodes); w = w / np.sqrt(2 * np.pi)
    J = np.zeros(len(Xc))
    for xk, wk in zip(x, w):
        J += wk * np.prod(norm.cdf((np.sqrt(rho) * xk - a) / np.sqrt(1 - rho)), axis=1)
    return J, f"Gauss-Hermite quadrature, {nodes} nodes"


def summarize(d, J, d1, p1, w, label):
    f = np.where(d, 1 - J, J)
    f1 = np.where(d1, 1 - p1, p1)
    lo, ga = float((1 - J)[d].sum()), float(J[~d].sum())
    lo1, ga1 = float((1 - p1)[d1].sum()), float(p1[~d1].sum())
    return dict(label=label, designated=int(d.sum()),
                expected_losses=lo, expected_gains=ga, expected_two_sided=lo + ga,
                loss_rate_designated=lo / max(d.sum(), 1),
                single_designated=int(d1.sum()), single_expected_losses=lo1, single_expected_gains=ga1,
                single_two_sided=lo1 + ga1, single_loss_rate=lo1 / max(d1.sum(), 1),
                ratio_two_sided=(lo + ga) / (lo1 + ga1),
                ratio_loss_rate=(lo / max(d.sum(), 1)) / (lo1 / max(d1.sum(), 1)),
                ratio_losses=lo / lo1, ratio_gains=ga / ga1 if ga1 > 0 else None,
                designated_phi_ge_0p10=int(((f >= 0.10 - 1e-9) & d).sum()),
                share_designated_phi_ge_0p10=float(((f >= 0.10 - 1e-9) & d).sum() / max(d.sum(), 1)),
                phi_ge_0p10_all=int((f >= 0.10 - 1e-9).sum()),
                expected_changes_share_of_list=(lo + ga) / max(d.sum(), 1),
                pop_in_fragile_designated=float(w[(f >= 0.10 - 1e-9) & d].sum()),
                mean_phi_designated=float(f[d].mean()))


live = {}
S_acs = L[["s2011", "s2021"]].to_numpy(float)
d1 = X[:, 3] >= T
p1 = norm.sf((T - Xraw[:, 3]) / S_acs[:, 1])
per_live = L[["fips", "CountyName", "PerPov1721c"]].copy()
for DF in (1.0, 1.5, 2.0):
    S = np.column_stack([lf_se(Xraw[:, 0], L["B90"].to_numpy(float), DF),
                         lf_se(Xraw[:, 1], L["B00"].to_numpy(float), DF), S_acs])
    J0 = np.prod(norm.sf((T - Xraw) / S), axis=1)               # rho = 0, closed form
    live[f"DF{DF}_rho0"] = summarize(d_conj, J0, d1, p1, popw, f"DF={DF}, rho=0 (closed form)")
    # rule on the four-period average
    xbar = Xraw.mean(axis=1); sbar = np.sqrt((S ** 2).sum(axis=1)) / 4
    davg = xbar >= T
    pavg = norm.sf((T - xbar) / sbar)
    favg = np.where(davg, 1 - pavg, pavg)
    live[f"DF{DF}_rho0"]["average_rule"] = dict(
        designated=int(davg.sum()), expected_losses=float((1 - pavg)[davg].sum()),
        expected_gains=float(pavg[~davg].sum()), expected_two_sided=float(favg.sum()),
        loss_rate=float((1 - pavg)[davg].sum() / max(davg.sum(), 1)),
        overlap_with_conjunction=int((davg & d_conj).sum()))
    per_live[f"phi_DF{DF}"] = np.where(d_conj, 1 - J0, J0)
    if DF == 1.5:
        J3, Rmc = orthant_mc(Xraw, S, 0.3)
        r3 = summarize(d_conj, J3, d1, p1, popw, "DF=1.5, rho=0.3 (assumed)")
        f3 = np.where(d_conj, 1 - J3, J3)
        r3["method"] = Rmc
        live["DF1.5_rho0.3"] = r3
        S15 = S.copy(); J15 = J0.copy()
# the earlier two-period model (census gates held fixed), for reconciliation
gate = (L["HiPov90c"] == 1) & (L["HiPov00c"] == 1)
g = gate.to_numpy()
Xa = Xraw[g][:, 2:]; Sa = S_acs[g]
Ja = np.prod(norm.sf((T - Xa) / Sa), axis=1)
da = (Xa >= T).all(axis=1)
d1a = Xa[:, 1] >= T
p1a = norm.sf((T - Xa[:, 1]) / Sa[:, 1])
live["old_two_period_model_census_gates_fixed"] = summarize(da, Ja, d1a, p1a, popw[g], "2 ACS periods, census gates fixed")
live["old_two_period_model_census_gates_fixed"]["n_counties"] = int(g.sum())
out["live_rule"] = live
per_live["designated_rebuilt"] = d_conj.astype(int)

# ------------------------------------------------------------ 3. SAIPE K=4 analogue
okB = df[["sp1989", "sp1999", "sp2011", "sp2021", "ss1989", "ss1999", "ss2011", "ss2021"]].notna().all(axis=1) & \
      (df[["ss1989", "ss1999", "ss2011", "ss2021"]] > 0).all(axis=1)
Bd = df[okB].reset_index(drop=True)
XB = Bd[["sp1989", "sp1999", "sp2011", "sp2021"]].to_numpy(float)
SB = Bd[["ss1989", "ss1999", "ss2011", "ss2021"]].to_numpy(float)
dB = (XB >= T).all(axis=1)
d1B = XB[:, 3] >= T
p1B = norm.sf((T - XB[:, 3]) / SB[:, 3])
k4 = {"n_counties": int(len(Bd)), "designated_point": int(dB.sum())}
for rho in (0.0, 0.3, 0.6, 0.9):
    if rho == 0.0:
        J = np.prod(norm.sf((T - XB) / SB), axis=1); Rmc = None
    else:
        J, Rmc = orthant_mc(XB, SB, rho, seed=SEED + int(10 * rho))
    r = summarize(dB, J, d1B, p1B, Bd["sp2021"].fillna(0).to_numpy() * 0 + 1, f"SAIPE K=4, rho={rho}")
    r["method"] = Rmc if Rmc else "exact (product of normal tail probabilities)"
    k4[f"rho{rho}"] = r
out["saipe_k4"] = k4

# ------------------------------------------------------------ 4. empirical Bayes (MFH, arcsine scale)
DIV = {  # Census divisions by state FIPS
    1: ["09", "23", "25", "33", "44", "50"], 2: ["34", "36", "42"], 3: ["17", "18", "26", "39", "55"],
    4: ["19", "20", "27", "29", "31", "38", "46"], 5: ["10", "11", "12", "13", "24", "37", "45", "51", "54"],
    6: ["01", "21", "28", "47"], 7: ["05", "22", "40", "48"], 8: ["04", "08", "16", "30", "32", "35", "49", "56"],
    9: ["02", "06", "15", "41", "53"]}
div_of = {s: k for k, v in DIV.items() for s in v}


def to_y(rate_pct, se_pct):
    p = np.clip(rate_pct / 100, 0.0, 1.0)
    pc = np.clip(p, 0.005, 0.995)
    return np.arcsin(np.sqrt(p)), (se_pct / 100) / (2 * np.sqrt(pc * (1 - pc)))


def fit_mfh(Y, SY, groups):
    mu = np.zeros_like(Y)
    for gval in np.unique(groups):
        m = groups == gval
        mu[m] = Y[m].mean(axis=0)
    resid = Y - mu
    Psi = np.cov(resid, rowvar=False) - np.diag((SY ** 2).mean(axis=0))
    w, V = np.linalg.eigh(Psi)
    Psi = V @ np.diag(np.clip(w, 1e-8, None)) @ V.T
    return mu, Psi


def posterior(Y, SY, mu, Psi):
    Pi = np.linalg.inv(Psi)
    n, K = Y.shape
    M = np.empty_like(Y); Lc = np.empty((n, K, K))
    for i in range(n):
        Si = np.diag(1 / SY[i] ** 2)
        V = np.linalg.inv(Pi + Si)
        M[i] = V @ (Pi @ mu[i] + Si @ Y[i])
        Lc[i] = np.linalg.cholesky((V + V.T) / 2)
    return M, Lc


def post_prob_rules(M, Lc, R=20000, seed=SEED + 7, thr=T / 100):
    """From the same posterior draws: P(all four >= thr), P(last period >= thr),
    P(four-period mean >= thr), per county."""
    Z = np.random.default_rng(seed).standard_normal((R, M.shape[1]))
    a, l, m = np.empty(len(M)), np.empty(len(M)), np.empty(len(M))
    for i in range(len(M)):
        y = M[i] + Z @ Lc[i].T
        p = np.sin(np.clip(y, 0, np.pi / 2)) ** 2
        a[i] = (p >= thr).all(axis=1).mean(); l[i] = (p[:, -1] >= thr).mean(); m[i] = (p.mean(axis=1) >= thr).mean()
    return a, l, m


def post_prob_all(M, Lc, R=20000, seed=SEED + 7, thr=T / 100):
    Z = np.random.default_rng(seed).standard_normal((R, M.shape[1]))
    out_ = np.empty(len(M))
    for i in range(len(M)):
        y = M[i] + Z @ Lc[i].T
        p = np.sin(np.clip(y, 0, np.pi / 2)) ** 2
        out_[i] = (p >= thr).all(axis=1).mean()
    return out_, R


groups = L["fips"].str[:2].map(div_of).to_numpy()
Y, SY = to_y(Xraw, S15)
mu, Psi = fit_mfh(Y, SY, groups)
M, Lc = posterior(Y, SY, mu, Psi)
J_eb, Reb = post_prob_all(M, Lc)
# flat prior on the same scale and engine (isolates the prior)
Lflat = np.zeros_like(Lc)
for i in range(len(Y)):
    Lflat[i] = np.diag(SY[i])
J_flat_y, _ = post_prob_all(Y, Lflat)
f_eb = np.where(d_conj, 1 - J_eb, J_eb)
f_fl = np.where(d_conj, 1 - J_flat_y, J_flat_y)
f_cf = np.where(d_conj, 1 - J15, J15)
sd = np.sqrt(np.diag(Psi))
out["empirical_bayes"] = dict(
    model="multivariate Fay-Herriot on arcsin(sqrt(rate)), Census-division mean vectors, "
          "4x4 between-county covariance by moments (eigenvalues clipped at 1e-8), DF=1.5, independent errors",
    psi_sd=sd.tolist(), psi_corr=(Psi / np.outer(sd, sd)).tolist(),
    mean_sampling_sd_y=SY.mean(axis=0).tolist(),
    mc_draws=Reb,
    flat_closed_form=dict(mean_phi_designated=float(f_cf[d_conj].mean()), designated_phi_ge_0p10=int(((f_cf >= 0.1 - 1e-9) & d_conj).sum()),
                          expected_losses=float((1 - J15)[d_conj].sum()), expected_gains=float(J15[~d_conj].sum())),
    flat_arcsine_mc=dict(mean_phi_designated=float(f_fl[d_conj].mean()), designated_phi_ge_0p10=int(((f_fl >= 0.1 - 1e-9) & d_conj).sum()),
                         expected_losses=float((1 - J_flat_y)[d_conj].sum()), expected_gains=float(J_flat_y[~d_conj].sum())),
    eb=dict(mean_phi_designated=float(f_eb[d_conj].mean()), designated_phi_ge_0p10=int(((f_eb >= 0.1 - 1e-9) & d_conj).sum()),
            expected_losses=float((1 - J_eb)[d_conj].sum()), expected_gains=float(J_eb[~d_conj].sum()),
            phi_ge_0p10_all=int((f_eb >= 0.1 - 1e-9).sum())),
    fragile_overlap_designated=dict(flat=int(((f_fl >= 0.1 - 1e-9) & d_conj).sum()), eb=int(((f_eb >= 0.1 - 1e-9) & d_conj).sum()),
                                    both=int(((f_fl >= 0.1 - 1e-9) & (f_eb >= 0.1 - 1e-9) & d_conj).sum())),
    eb_minus_flat_designated_mean=float((f_eb - f_fl)[d_conj].mean()),
    eb_minus_flat_undesignated_mean=float((f_eb - f_fl)[~d_conj].mean()),
)
# the three rules under both priors, from the same posterior draws
rules = {}
for lab, (Mc, Lcc) in (("flat_arcsine", (Y, Lflat)), ("eb", (M, Lc))):
    pa, pl, pm = post_prob_rules(Mc, Lcc)
    for rname, prob, dpub in (("conjunction", pa, d_conj), ("single_2017_21", pl, X[:, 3] >= T),
                              ("four_period_mean", pm, X.mean(axis=1) >= T)):
        rules.setdefault(rname, {})[lab] = dict(
            designated=int(dpub.sum()), expected_false_designations=float((1 - prob)[dpub].sum()),
            expected_missed_designations=float(prob[~dpub].sum()),
            expected_two_sided=float((1 - prob)[dpub].sum() + prob[~dpub].sum()),
            false_designation_rate=float((1 - prob)[dpub].sum() / max(dpub.sum(), 1)))
out["empirical_bayes"]["rules_under_both_priors"] = rules
per_live["phi_flat_DF1.5"] = f_cf
per_live["phi_eb_DF1.5"] = f_eb
per_live.to_csv(os.path.join(RES, "pp_live_rule_per_county.csv"), index=False)

# ------------------------------------------------------------ 5. known-truth calibration
def score(prob, truth_mis, label):
    bins = [0, 0.02, 0.05, 0.10, 0.25, 0.5, 1.0001]
    cal = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (prob >= lo) & (prob < hi)
        if m.sum():
            cal.append(dict(bin=f"[{lo},{min(hi,1)})", n=int(m.sum()), mean_prob=float(prob[m].mean()),
                            observed_rate=float(truth_mis[m].mean())))
    return dict(label=label, sum_prob=float(prob.sum()), sum_true=float(truth_mis.sum()),
                brier=float(((prob - truth_mis) ** 2).mean()), calibration=cal)


kt = {}
rng = np.random.default_rng(SEED + 99)
NS = 40
for design in ("T1_truth_from_EB_posterior", "T2_truth_equals_published"):
    agg = {"flat": [], "eb": [], "true": [], "band_flat_wrong": [], "band_eb_wrong": [], "n_true_atrisk": [],
           "true_losses": [], "flat_losses": [], "eb_losses": [], "true_gains": [], "flat_gains": [], "eb_gains": [],
           "flat_flag": [], "eb_flag": [], "flat_flag_at_risk": [], "eb_flag_at_risk": [], "disagree": []}
    pool_flat, pool_eb, pool_mis = [], [], []
    for s in range(NS):
        if design.startswith("T1"):
            Zs = rng.standard_normal(Y.shape)
            Yt = M + np.einsum("ijk,ik->ij", Lc, Zs)
            rt = 100 * np.sin(np.clip(Yt, 0, np.pi / 2)) ** 2
        else:
            rt = Xraw.copy()
        xs = rt + rng.standard_normal(rt.shape) * S15            # synthetic published estimates
        d_true = (rt >= T).all(axis=1)
        d_pub = (xs >= T).all(axis=1)
        mis = (d_true != d_pub).astype(float)
        Jf = np.prod(norm.sf((T - xs) / S15), axis=1)
        pf = np.where(d_pub, 1 - Jf, Jf)
        Ys, SYs = to_y(xs, S15)
        Ms, Lcs = posterior(Ys, SYs, mu, Psi)
        Je, _ = post_prob_all(Ms, Lcs, R=4000, seed=SEED + 1000 + s)
        pe = np.where(d_pub, 1 - Je, Je)
        # true misclassification probability given the truth (independent errors)
        q = np.prod(norm.sf((T - rt) / S15), axis=1)
        risk = np.where(d_true, 1 - q, q)
        atrisk = risk >= 0.10 - 1e-9
        agg["flat"].append(pf.sum()); agg["eb"].append(pe.sum()); agg["true"].append(mis.sum())
        agg["band_flat_wrong"].append(int(((pf >= 0.10 - 1e-9) != atrisk).sum()))
        agg["band_eb_wrong"].append(int(((pe >= 0.10 - 1e-9) != atrisk).sum()))
        agg["n_true_atrisk"].append(int(atrisk.sum()))
        agg["true_losses"].append(float(mis[d_pub].sum())); agg["true_gains"].append(float(mis[~d_pub].sum()))
        agg["flat_losses"].append(float(pf[d_pub].sum())); agg["flat_gains"].append(float(pf[~d_pub].sum()))
        agg["eb_losses"].append(float(pe[d_pub].sum())); agg["eb_gains"].append(float(pe[~d_pub].sum()))
        agg["flat_flag"].append(int((pf >= 0.10 - 1e-9).sum())); agg["eb_flag"].append(int((pe >= 0.10 - 1e-9).sum()))
        agg["flat_flag_at_risk"].append(int(((pf >= 0.10 - 1e-9) & atrisk).sum()))
        agg["eb_flag_at_risk"].append(int(((pe >= 0.10 - 1e-9) & atrisk).sum()))
        agg["disagree"].append(float((2 * q * (1 - q)).sum()))
        pool_flat.append(pf); pool_eb.append(pe); pool_mis.append(mis)
    pf, pe, pm = map(np.concatenate, (pool_flat, pool_eb, pool_mis))
    kt[design] = dict(replicates=NS, eb_inner_draws=4000,
                      mean_sum_flat=float(np.mean(agg["flat"])), mean_sum_eb=float(np.mean(agg["eb"])),
                      mean_true_misclassified=float(np.mean(agg["true"])),
                      sd_true_misclassified=float(np.std(agg["true"], ddof=1)),
                      mean_band_errors_flat=float(np.mean(agg["band_flat_wrong"])),
                      mean_band_errors_eb=float(np.mean(agg["band_eb_wrong"])),
                      mean_units_truly_at_risk=float(np.mean(agg["n_true_atrisk"])),
                      mean_true_losses=float(np.mean(agg["true_losses"])), mean_true_gains=float(np.mean(agg["true_gains"])),
                      mean_flat_losses=float(np.mean(agg["flat_losses"])), mean_flat_gains=float(np.mean(agg["flat_gains"])),
                      mean_eb_losses=float(np.mean(agg["eb_losses"])), mean_eb_gains=float(np.mean(agg["eb_gains"])),
                      flat_band_precision=float(np.sum(agg["flat_flag_at_risk"]) / np.sum(agg["flat_flag"])),
                      flat_band_recall=float(np.sum(agg["flat_flag_at_risk"]) / np.sum(agg["n_true_atrisk"])),
                      eb_band_precision=float(np.sum(agg["eb_flag_at_risk"]) / np.sum(agg["eb_flag"])),
                      eb_band_recall=float(np.sum(agg["eb_flag_at_risk"]) / np.sum(agg["n_true_atrisk"])),
                      mean_two_survey_disagreement=float(np.mean(agg["disagree"])),
                      flat=score(pf, pm, "flat prior (closed form)"), eb=score(pe, pm, "empirical Bayes (MFH)"))
out["known_truth"] = kt
out["seconds"] = round(time.time() - t0, 1)
json.dump(out, open(os.path.join(RES, "pp_conjunctive.json"), "w"), indent=2, default=float)
print(json.dumps(out, indent=2, default=float))
