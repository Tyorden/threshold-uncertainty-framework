"""qct_rate_engine.py: rate-level QCT engine used to compare priors.

The tract poverty rate of each vintage is drawn directly (on the arcsine-square-root
scale) from either the flat-prior posterior N(y_hat, Sigma_i) or the empirical-Bayes
posterior N(m_i, V_i) of a multivariate Fay-Herriot model; the poverty reliability
screen is held at its published status; median income is drawn from its flat-prior
posterior with the overlap cross-vintage correlation (no income-poverty term), or
held at its published values. Everything downstream (2-of-3 tests with HUD's
rounding, groups, rank averages, 20 percent cap fill) is qct_fast's.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import qct_fast as qf

RHO_XU = 0.5      # universe <-> persons-in-poverty error correlation (realistic scenario)


def rate_inputs(D: qf.QCTData, clip=0.005):
    """Published rates, their SEs (delta method under the realistic count model),
    arcsine-scale values and SEs, and the published poverty reliability mask."""
    U, X = D.univ, D.inpov
    su, sx = D.univ_se, D.inpov_se
    with np.errstate(divide="ignore", invalid="ignore"):
        p = X / U
        var = (sx ** 2 + p ** 2 * su ** 2 - 2 * p * RHO_XU * sx * su) / U ** 2
    ok = np.isfinite(p) & (U > 0) & np.isfinite(var)
    p = np.where(ok, np.clip(p, 0, 1), np.nan)
    sp = np.where(ok, np.sqrt(np.clip(var, 1e-10, None)), np.nan)
    pc = np.clip(p, clip, 1 - clip)
    y = np.arcsin(np.sqrt(p))
    sy = sp / (2 * np.sqrt(pc * (1 - pc)))
    rel = qf._moer_lt(U, D.univ_moe) & qf._moer_lt(X, D.inpov_moe)
    return p, sp, y, sy, rel


def fit_prior(D, y, sy, min_univ=0.0):
    """Multivariate Fay-Herriot moments: cap-unit mean vectors, common 3x3 Psi."""
    ok = np.all(np.isfinite(y), axis=0) & (np.nanmin(D.univ, axis=0) > min_univ)
    Y = y.T  # n x 3
    mu = np.full_like(Y, np.nan)
    for u in range(D.U):
        idx = D.unit_index[u]
        sel = idx[ok[idx]]
        if len(sel) == 0:
            sel = idx
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mu[idx] = np.nanmean(Y[sel], axis=0)
    glob = np.nanmean(Y[ok], axis=0)
    mu = np.where(np.isfinite(mu), mu, glob)
    Rv = qf.R_VINT_OVERLAP
    S = sy.T[ok]
    meanSigma = (S[:, :, None] * S[:, None, :] * Rv[None]).mean(axis=0)
    resid = Y[ok] - mu[ok]
    Psi = np.cov(resid, rowvar=False) - meanSigma
    w, V = np.linalg.eigh(Psi)
    Psi = V @ np.diag(np.clip(w, 1e-6, None)) @ V.T
    return mu, Psi, dict(n_fit=int(ok.sum()), psi=Psi.tolist(), mean_sigma=meanSigma.tolist(),
                         raw_cov=np.cov(resid, rowvar=False).tolist())


def fit_prior_unit(D, y, sy, min_univ=500.0, min_tracts=1000):
    """Unit-specific Psi for cap units with at least min_tracts fitting tracts,
    the national Psi (same fitting subset) elsewhere. Returns per-tract Psi."""
    mu, Psi_nat, _ = fit_prior(D, y, sy, min_univ)
    ok = np.all(np.isfinite(y), axis=0) & (np.nanmin(D.univ, axis=0) > min_univ)
    Rv = qf.R_VINT_OVERLAP
    PsiN = np.repeat(Psi_nat[None], D.n, axis=0)
    units_specific = []
    for u in range(D.U):
        idx = D.unit_index[u]; sel = idx[ok[idx]]
        if len(sel) < min_tracts:
            continue
        Y = y.T[sel]; S = sy.T[sel]
        Ms = (S[:, :, None] * S[:, None, :] * Rv[None]).mean(axis=0)
        P = np.cov(Y - mu[sel], rowvar=False) - Ms
        w, V = np.linalg.eigh(P)
        P = V @ np.diag(np.clip(w, 1e-6, None)) @ V.T
        PsiN[idx] = P
        units_specific.append(dict(unit=str(D.cap_unit_label[u]), n_fit=int(len(sel)), psi_sd=np.sqrt(np.diag(P)).tolist()))
    return mu, PsiN, units_specific


def posterior(y, sy, mu, Psi, missing_sd=10.0):
    """Per-tract posterior mean and Cholesky factor (n x 3, n x 3 x 3).
    Psi may be 3 x 3 (common) or n x 3 x 3 (per tract)."""
    Y = np.where(np.isfinite(y), y, 0.0).T
    S = np.where(np.isfinite(sy), sy, missing_sd).T
    Rv = qf.R_VINT_OVERLAP
    Sig = S[:, :, None] * S[:, None, :] * Rv[None]
    Pi = np.linalg.inv(Psi)
    if Pi.ndim == 2:
        Pi = np.repeat(Pi[None], len(Y), axis=0)
    Si = np.linalg.inv(Sig)
    V = np.linalg.inv(Pi + Si)
    m = np.einsum("nij,nj->ni", V, np.einsum("nij,nj->ni", Pi, mu) + np.einsum("nij,nj->ni", Si, Y))
    V = (V + np.transpose(V, (0, 2, 1))) / 2
    Lc = np.linalg.cholesky(V)
    Lflat = np.linalg.cholesky(Sig)
    return m, Lc, Y, Lflat


def draw_rates(center, Lc, Z):
    """center n x 3, Lc n x 3 x 3, Z 3 x n -> rates 3 x n (sin^2 of the arcsine draw)."""
    y = center + np.einsum("nij,jn->ni", Lc, Z)
    return (np.sin(np.clip(y, 0, np.pi / 2)) ** 2).T


def draw_income(D, Z3, L3):
    X = L3 @ Z3
    med = D.med + X * D.med_se
    return np.where(np.isnan(D.med), np.nan, np.clip(med, 1.0, None))


def eligibility_rates(D, med, rates, pov_rel_fixed):
    """qct_fast.eligibility with the poverty rates given directly and the
    poverty reliability screen held at its published status."""
    with np.errstate(divide="ignore", invalid="ignore"):
        med0 = np.where(med == 0, np.nan, med)
        inc_ratio = D.adj_lim / med0
    inc_rel = qf._moer_lt(med, D.med_moe)
    pov = np.floor(1000.0 * rates + 0.5) / 1000.0
    with np.errstate(invalid="ignore"):
        inc_met = inc_rel & (inc_ratio > 1.0)
        pov_met = pov_rel_fixed & (pov >= qf.POV_T)
    inc_vals = np.where(inc_rel, inc_ratio, np.nan)
    pov_vals = np.where(pov_rel_fixed, pov, np.nan)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        avg_inc = np.nanmean(inc_vals, axis=0)
        avg_pov = np.nanmean(pov_vals, axis=0)
    inc_c = inc_met.sum(axis=0) >= 2
    pov_c = pov_met.sum(axis=0) >= 2
    elig = inc_c | pov_c
    group = np.where(inc_c & pov_c, 1, np.where(elig, 2, 0))
    return elig, group, avg_inc, avg_pov


def run_rates(D, med, rates, pov_rel_fixed):
    el, gr, ai, ap = eligibility_rates(D, med, rates, pov_rel_fixed)
    return qf.designate(D, el, gr, ai, ap), el
