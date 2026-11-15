#!/usr/bin/env python
"""07_qct_eb.py: flat-prior versus empirical-Bayes posterior misclassification
probabilities for QCT designation, all 85,390 tracts, 1,000 paired draws.

Prior: multivariate Fay-Herriot on arcsin(sqrt(poverty rate)) for the three
vintages; cap-unit mean vectors (unit fixed effects); common 3x3 between-tract
covariance Psi by moments (raw covariance of residuals minus the mean sampling
covariance; sampling covariance = SEs x overlap correlation 0.8/0.8/0.6).
Primary fit: tracts with a poverty universe above 500 in all three vintages
(a fit on all tracts is degenerate: tiny-universe tracts have exploding
arcsine-scale SEs and drive Psi to zero; recorded as prior_fit_all_tracts).
Sensitivities: (a) fitting subset universe > 1000 in every vintage; (b) cap-unit
specific Psi for units with at least 1,000 fitting tracts (national Psi elsewhere).
(An all-tract fit with the derivative clipped at [0.02, 0.98] is also degenerate,
cross-vintage correlations of 1.0; recorded as prior_fit_clip02_degenerate.)
The posterior is applied to every tract in all runs. Income is drawn from its flat posterior in both runs, with the same
normals, so the prior on the poverty input is the only difference.
"""
import os, sys, json, time
sys.dont_write_bytecode = True
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__)); W = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import numpy as np, pandas as pd
from multiprocessing import Pool
import qct_fast as qf
import qct_rate_engine as qe

SEED = 20261110
R = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
OUT = os.path.join(W, "results", "qct_eb"); os.makedirs(OUT, exist_ok=True)
G = {}


def setup():
    D = qf.QCTData(pd.read_parquet(os.path.join(W, "cache", "qct_data_2026.parquet")))
    p, sp, y, sy, rel = qe.rate_inputs(D)
    _, _, diag_all = qe.fit_prior(D, y, sy)                       # degenerate; recorded only
    mu, Psi, diag = qe.fit_prior(D, y, sy, min_univ=500.0)        # primary
    m, Lc, Yc, Lflat = qe.posterior(y, sy, mu, Psi)
    p2, sp2, y2, sy2, _ = qe.rate_inputs(D, clip=0.02)
    _, _, diag_c02 = qe.fit_prior(D, y2, sy2)                     # degenerate; recorded only
    mu2, Psi2, diag2 = qe.fit_prior(D, y, sy, min_univ=1000.0)    # sensitivity (a)
    m2, Lc2, _, _ = qe.posterior(y, sy, mu2, Psi2)
    mu3, Psi3, units3 = qe.fit_prior_unit(D, y, sy, 500.0, 1000)  # sensitivity (b)
    m3, Lc3, _, _ = qe.posterior(y, sy, mu3, Psi3)
    G.update(D=D, p=p, rel=rel, m=m, Lc=Lc, Yc=Yc, Lflat=Lflat, m5=m2, Lc5=Lc2, m3=m3, Lc3=Lc3,
             L3=np.linalg.cholesky(qf.R_VINT_OVERLAP), diag=diag, diag5=diag2, diag_all=diag_all,
             diag_c02=diag_c02, units3=units3)


def worker(ids):
    D = G["D"]; n = D.n
    acc = {k: dict(des=np.zeros(n, np.int32), elig=np.zeros(n, np.int32), eld=np.zeros(n, np.int32))
           for k in ("flat", "eb", "eb_univ1000", "eb_unit_psi")}
    for r in ids:
        rng = np.random.default_rng([SEED, r])
        Zp = rng.standard_normal((3, n)); Zi = rng.standard_normal((3, n))
        med = qe.draw_income(D, Zi, G["L3"])
        for k, (c, L) in (("flat", (G["Yc"], G["Lflat"])), ("eb", (G["m"], G["Lc"])),
                          ("eb_univ1000", (G["m5"], G["Lc5"])), ("eb_unit_psi", (G["m3"], G["Lc3"]))):
            rates = qe.draw_rates(c, L, Zp)
            de, el = qe.run_rates(D, med, rates, G["rel"])
            acc[k]["des"] += de; acc[k]["elig"] += el; acc[k]["eld"] += el & ~de
    return acc


def main():
    t0 = time.time()
    setup()
    D = G["D"]
    base_rate, _ = qe.run_rates(D, D.med, np.where(np.isfinite(G["p"]), G["p"], 0.0), G["rel"])
    base_rep, _ = qf.run(D, rounding=True, baseline_hud_columns=False)
    hud = D.qct == 1
    pieces = [list(range(i, min(i + 25, R))) for i in range(0, R, 25)]
    tot = None
    with Pool(4, initializer=setup) as pool:
        for k, acc in enumerate(pool.imap_unordered(worker, pieces)):
            if tot is None:
                tot = acc
            else:
                for a in tot:
                    for f in tot[a]:
                        tot[a][f] += acc[a][f]
            print(f"  {min((k+1)*25, R)}/{R}  {time.time()-t0:.0f}s", flush=True)
    val = json.load(open(os.path.join(W, "results", "qct_validation.json")))
    bound = np.isin(D.cap_unit_label[D.unit], val["cap_bound_unit_labels"])
    res = {"R": R, "seed": SEED, "prior_fit_primary_univ_gt_500": G["diag"], "prior_fit_sensitivity_univ_gt_1000": G["diag5"],
           "prior_fit_unit_specific_psi_units": G["units3"],
           "prior_fit_all_tracts_degenerate": G["diag_all"], "prior_fit_clip02_degenerate": G["diag_c02"],
           "baseline_rate_engine_equals_replication": bool((base_rate == base_rep).all()),
           "baseline_rate_engine_vs_hud_mismatch": int((base_rate != hud).sum())}
    per = pd.DataFrame({"GEOID": D.geoid, "hud": hud.astype(int), "cap_bound": bound.astype(int), "P1": D.P1,
                        "p_hat_23": G["p"][2], "eb_mean_rate_23": np.sin(np.clip(G["m"][:, 2], 0, np.pi / 2)) ** 2})
    for k in tot:
        pdz = tot[k]["des"] / R; pel = tot[k]["elig"] / R; ped = tot[k]["eld"] / R
        phi = np.where(hud, 1 - pdz, pdz)
        per[f"phi_{k}"] = phi; per[f"p_elig_{k}"] = pel; per[f"p_eld_{k}"] = ped
        for scope, msk in (("national", np.ones(D.n, bool)), ("cap_bound", bound), ("slack", ~bound)):
            Dm = hud & msk; Um = ~hud & msk
            res.setdefault(k, {})[scope] = dict(
                designated=int(Dm.sum()),
                expected_losses=float((1 - pdz)[Dm].sum()), losses_rank_driven=float(ped[Dm].sum()),
                expected_gains=float(pdz[Um].sum()),
                expected_two_sided=float((1 - pdz)[Dm].sum() + pdz[Um].sum()),
                designated_phi_ge_0p10=int(((phi >= 0.10 - 1e-9) & Dm).sum()),
                share_designated_phi_ge_0p10=float(((phi >= 0.10 - 1e-9) & Dm).sum() / max(Dm.sum(), 1)),
                phi_ge_0p10=int(((phi >= 0.10 - 1e-9) & msk).sum()),
                mean_phi_designated=float(phi[Dm].mean()), mean_phi_undesignated=float(phi[Um].mean()))
    # where the prior moves things: designated tracts that qualify on poverty near the line
    near = hud & np.isfinite(G["p"][2]) & (np.abs(G["p"][2] - 0.25) < 0.03)
    res["designated_with_2019_23_rate_within_3pts_of_25pct"] = dict(
        n=int(near.sum()), mean_phi_flat=float(per.loc[near, "phi_flat"].mean()),
        mean_phi_eb=float(per.loc[near, "phi_eb"].mean()))
    f, e = per["phi_flat"].to_numpy(), per["phi_eb"].to_numpy()
    res["fragile_sets_designated"] = dict(flat=int(((f >= 0.1 - 1e-9) & hud).sum()), eb=int(((e >= 0.1 - 1e-9) & hud).sum()),
                                          both=int(((f >= 0.1 - 1e-9) & (e >= 0.1 - 1e-9) & hud).sum()))
    res["mc_se_note"] = "per-tract SE sqrt(phi(1-phi)/R); at phi = 0.10 and R = 1000, 0.0095"
    res["seconds"] = round(time.time() - t0, 1)
    per.to_parquet(os.path.join(OUT, "per_tract_flat_vs_eb.parquet"))
    json.dump(res, open(os.path.join(OUT, "eb_summary.json"), "w"), indent=2, default=float)
    print(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
