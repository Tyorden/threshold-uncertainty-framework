"""qct_fast.py: vectorized re-implementation of HUD's 2026 QCT rule
(FR Doc 2025-19007, section VI.C) for Monte Carlo use.

It reproduces QCT_lihtc/scripts/qct_pipeline.py exactly (validated in
02b_qct_validate.py) and adds two options:
  rounding=True   HUD's convention, found in its own file: the tract poverty
                  rate is stored as round-half-up(1000 * inpov / univ) for
                  reliable tracts (exact for every reliable tract in all three
                  vintages), and the 25 percent test is applied to that value.
  baseline_hud_columns=True   rank on HUD's published inc_factor / pov_rate
                  columns (the folder pipeline's baseline choice).
Rule variants for the design comparison: 'hud_2of3' (the rule in force),
'latest_only' (2019-23 vintage alone), 'avg3' (test the average of the
reliable vintage values, at least two reliable vintages required).

Cap unit: metro CBSA for metro tracts, nonmetro part of the state otherwise.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from scipy.stats import rankdata

YEARS = ["21", "22", "23"]
Z90 = 1.645
MOER_MAX = 0.50
POV_T = 0.25
CAP = 0.20


class QCTData:
    def __init__(self, df: pd.DataFrame):
        df = df.reset_index(drop=True)
        self.df = df
        self.n = len(df)
        self.geoid = df["GEOID"].to_numpy()
        self.P1 = df["P1"].to_numpy(float)
        self.qct = df["qct"].to_numpy().astype(int)
        cu = np.where(df["metro"].to_numpy() == 1,
                      "M" + df["cbsa"].astype("int64").astype(str),
                      "N" + df["state"].astype("int64").astype(str))
        self.cap_unit_label, self.unit = np.unique(cu, return_inverse=True)
        self.U = len(self.cap_unit_label)
        area = np.zeros(self.U)
        area[self.unit] = df["Area_Pop"].to_numpy(float)
        self.area_pop = area
        self.cap_pop = CAP * area
        order = np.argsort(self.unit, kind="stable")
        bounds = np.searchsorted(self.unit[order], np.arange(self.U + 1))
        self.unit_index = [order[bounds[u]:bounds[u + 1]] for u in range(self.U)]
        g = lambda c: np.stack([df[f"{c}_{y}"].to_numpy(float) for y in YEARS])
        self.med, self.med_moe = g("B19013est1"), g("B19013me1")
        self.univ, self.univ_moe = g("B17001est1"), g("B17001me1")
        self.inpov, self.inpov_moe = g("B17001est2"), g("B17001me2")
        self.adj_lim = g("adj_inc_lim")
        self.inc_factor_hud = g("inc_factor")
        self.pov_rate_hud = g("pov_rate")
        self.med_se = np.nan_to_num(self.med_moe / Z90)
        self.univ_se = np.nan_to_num(self.univ_moe / Z90)
        self.inpov_se = np.nan_to_num(self.inpov_moe / Z90)

    def subset(self, mask):
        return QCTData(self.df[mask])


def _moer_lt(est, moe):
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.abs(moe) / np.abs(est)
    r[~np.isfinite(r)] = np.nan
    return r < MOER_MAX            # NaN -> False


def eligibility(D: QCTData, med=None, univ=None, inpov=None, rounding=True,
                baseline_hud_columns=False, variant="hud_2of3"):
    med = D.med if med is None else med
    univ = D.univ if univ is None else univ
    inpov = D.inpov if inpov is None else inpov
    with np.errstate(divide="ignore", invalid="ignore"):
        med0 = np.where(med == 0, np.nan, med)
        inc_ratio = D.adj_lim / med0
        u0 = np.where(univ == 0, np.nan, univ)
        pov = inpov / u0
    inc_rel = _moer_lt(med, D.med_moe)
    pov_rel = _moer_lt(univ, D.univ_moe) & _moer_lt(inpov, D.inpov_moe)
    if rounding:
        pov = np.floor(1000.0 * pov + 0.5) / 1000.0
    with np.errstate(invalid="ignore"):
        inc_met = inc_rel & (inc_ratio > 1.0)
        pov_met = pov_rel & (pov >= POV_T)
    if baseline_hud_columns:
        inc_vals = np.where(inc_rel, D.inc_factor_hud, np.nan)
        pov_vals = np.where(pov_rel, D.pov_rate_hud / 1000.0, np.nan)
    else:
        inc_vals = np.where(inc_rel, inc_ratio, np.nan)
        pov_vals = np.where(pov_rel, pov, np.nan)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        avg_inc = np.nanmean(inc_vals, axis=0)
        avg_pov = np.nanmean(pov_vals, axis=0)
    if variant == "hud_2of3":
        inc_c = inc_met.sum(axis=0) >= 2
        pov_c = pov_met.sum(axis=0) >= 2
    elif variant == "latest_only":
        inc_c, pov_c = inc_met[2], pov_met[2]
        avg_inc = np.where(inc_rel[2], inc_vals[2], np.nan)
        avg_pov = np.where(pov_rel[2], pov_vals[2], np.nan)
    elif variant == "avg3":
        with np.errstate(invalid="ignore"):
            inc_c = (inc_rel.sum(axis=0) >= 2) & (avg_inc > 1.0)
            pov_c = (pov_rel.sum(axis=0) >= 2) & (avg_pov >= POV_T)
    else:
        raise ValueError(variant)
    elig = inc_c | pov_c
    group = np.where(inc_c & pov_c, 1, np.where(elig, 2, 0))
    return elig, group, avg_inc, avg_pov


def _rank_desc_min(v):
    """pandas rank(ascending=False, method='min', na_option='keep')."""
    out = np.full(v.shape, np.nan)
    ok = ~np.isnan(v)
    if ok.any():
        out[ok] = rankdata(-v[ok], method="min")
    return out


def _ranked_order(D, e_idx, group, avg_inc, avg_pov):
    parts = []
    for gval in (1, 2):
        eg = e_idx[group[e_idx] == gval]
        if len(eg) == 0:
            continue
        comb = (_rank_desc_min(avg_inc[eg]) + _rank_desc_min(avg_pov[eg])) / 2.0
        key = np.where(np.isnan(comb), np.inf, comb)
        o = np.lexsort((-D.P1[eg], key))       # combined asc, then pop desc, stable
        parts.append(eg[o])
    return np.concatenate(parts) if parts else e_idx[:0]


def designate(D: QCTData, elig, group, avg_inc, avg_pov, return_bound=False):
    des = np.zeros(D.n, bool)
    elig_pop = np.bincount(D.unit[elig], weights=D.P1[elig], minlength=D.U)
    bound = elig_pop > D.cap_pop
    des[elig & ~bound[D.unit]] = True
    for u in np.nonzero(bound)[0]:
        idx = D.unit_index[u]
        e = idx[elig[idx]]
        ranked = _ranked_order(D, e, group, avg_inc, avg_pov)
        pops = D.P1[ranked]
        cap = D.cap_pop[u]
        cs = np.cumsum(pops)
        k = int(np.searchsorted(cs, cap, side="right"))
        while k > 0 and cs[k - 1] > cap:      # guard (cs is integer-valued)
            k -= 1
        des[ranked[:k]] = True
        cum = cs[k - 1] if k > 0 else 0.0
        pos = k + 1
        while pos < len(ranked):
            fits = np.nonzero(cum + pops[pos:] <= cap)[0]
            if len(fits) == 0:
                break
            j = pos + fits[0]
            des[ranked[j]] = True
            cum += pops[j]
            pos = j + 1
    if return_bound:
        return des, bound
    return des


def run(D, med=None, univ=None, inpov=None, rounding=True,
        baseline_hud_columns=False, variant="hud_2of3"):
    el, gr, ai, ap = eligibility(D, med, univ, inpov, rounding,
                                 baseline_hud_columns, variant)
    return designate(D, el, gr, ai, ap), el


# ---------------------------------------------------------------- draws
R_VINT_OVERLAP = np.array([[1.0, 0.8, 0.6], [0.8, 1.0, 0.8], [0.6, 0.8, 1.0]])
R_VINT_INDEP = np.eye(3)
R_VINT_HIGH = np.array([[1, .95, .9], [.95, 1, .95], [.9, .95, 1]], float)


def r_var(rip, rup):
    return np.array([[1.0, 0.0, rip], [0.0, 1.0, rup], [rip, rup, 1.0]])


def chol9(rip, rup, Rv):
    R = np.kron(r_var(rip, rup), Rv)
    return np.linalg.cholesky((R + R.T) / 2)


SCENARIOS = {
    "indep": (0.0, 0.0, R_VINT_INDEP),
    "xvintage_only": (0.0, 0.0, R_VINT_OVERLAP),
    "realistic": (-0.5, 0.5, R_VINT_OVERLAP),
    "xvintage_high": (-0.5, 0.5, R_VINT_HIGH),
    "realistic_ip_lo": (-0.3, 0.3, R_VINT_OVERLAP),
    "realistic_ip_hi": (-0.6, 0.6, R_VINT_OVERLAP),
}


def draw(D, rng, L9, scale=1.0, Z=None):
    """One correlated draw around the published estimates (flat-prior
    posterior draw), identical in form to QCT_lihtc/scripts/03_correlated_mc.py
    (dimension index = variable*3 + vintage; clipping as there)."""
    if Z is None:
        Z = rng.standard_normal((9, D.n))
    X = (L9 @ Z).reshape(3, 3, D.n) * scale
    med = D.med + X[0] * D.med_se
    med = np.where(np.isnan(D.med), np.nan, np.clip(med, 1.0, None))
    u = np.clip(D.univ + X[1] * D.univ_se, 0, None)
    p = np.clip(D.inpov + X[2] * D.inpov_se, 0, None)
    p = np.minimum(p, u)
    u = np.where(np.isnan(D.univ), np.nan, u)
    p = np.where(np.isnan(D.inpov), np.nan, p)
    return med, u, p
