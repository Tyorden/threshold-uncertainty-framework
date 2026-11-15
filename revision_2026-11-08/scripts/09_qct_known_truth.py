#!/usr/bin/env python
"""09_qct_known_truth.py: known-truth calibration of the flat-prior and
empirical-Bayes misclassification probabilities through the full QCT rule
(2-of-3 tests, groups, rank averages, 20 percent cap), all 85,390 tracts.

Only the poverty input is uncertain here: median income is held at its
published values and the poverty reliability screen at its published status,
so the experiment isolates the poverty prior. Prior: the primary multivariate
Fay-Herriot fit of 07 (arcsine scale, cap-unit means, Psi from tracts with a
poverty universe above 500).

Truth designs:
  T1  true rates drawn from the EB posterior given the actual published data
  T2  true rates equal to the published estimates
In each synthetic world: synthetic published estimates = truth + sampling error
(Sigma_i, overlap correlation); published list = rule(synthetic estimates);
true list = rule(truth); flat phi from 200 draws around the synthetic estimates;
EB phi from 200 posterior draws given the synthetic estimates; the truly-at-risk
probability and the two-survey disagreement from 100 further synthetic
estimates given the truth.
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

SEED = 20261111
S_WORLDS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
R_IN, K_RS = 200, 100
OUT = os.path.join(W, "results", "qct_known_truth"); os.makedirs(OUT, exist_ok=True)
G = {}


def setup():
    D = qf.QCTData(pd.read_parquet(os.path.join(W, "cache", "qct_data_2026.parquet")))
    p, sp, y, sy, rel = qe.rate_inputs(D)
    mu, Psi, _ = qe.fit_prior(D, y, sy, min_univ=500.0)
    m, Lc, Yc, Lflat = qe.posterior(y, sy, mu, Psi)
    G.update(D=D, y=y, sy=sy, rel=rel, mu=mu, Psi=Psi, m=m, Lc=Lc, Yc=Yc, Lflat=Lflat)


def rates_from_y(Y):            # Y: n x 3 -> 3 x n rates
    return (np.sin(np.clip(Y, 0, np.pi / 2)) ** 2).T


def des_of(rates):
    D = G["D"]
    d, _ = qe.run_rates(D, D.med, rates, G["rel"])
    return d


def world(args):
    design, s = args
    D = G["D"]; n = D.n
    rng = np.random.default_rng([SEED, 0 if design == "T1" else 1, s])
    if design == "T1":
        Yt = G["m"] + np.einsum("nij,jn->ni", G["Lc"], rng.standard_normal((3, n)))
    else:
        Yt = G["Yc"].copy()
    true_des = des_of(rates_from_y(Yt))
    Ys = Yt + np.einsum("nij,jn->ni", G["Lflat"], rng.standard_normal((3, n)))   # synthetic estimates
    pub = des_of(rates_from_y(Ys))
    # flat phi
    cnt = np.zeros(n)
    for r in range(R_IN):
        cnt += des_of(qe.draw_rates(Ys, G["Lflat"], rng.standard_normal((3, n))))
    pdf = cnt / R_IN
    phi_flat = np.where(pub, 1 - pdf, pdf)
    # EB phi (same prior, synthetic data)
    ys_T = Ys.T                                          # 3 x n
    m2, L2, _, _ = qe.posterior(ys_T, G["sy"], G["mu"], G["Psi"])
    cnt = np.zeros(n)
    for r in range(R_IN):
        cnt += des_of(qe.draw_rates(m2, L2, rng.standard_normal((3, n))))
    pde = cnt / R_IN
    phi_eb = np.where(pub, 1 - pde, pde)
    # sampling distribution of the published list given the truth
    q = np.zeros(n)
    for r in range(K_RS):
        Yr = Yt + np.einsum("nij,jn->ni", G["Lflat"], rng.standard_normal((3, n)))
        q += des_of(rates_from_y(Yr))
    q /= K_RS
    risk = np.where(true_des, 1 - q, q)
    disagree = 2 * q * (1 - q)
    mis = (true_des != pub).astype(np.int8)
    return design, s, dict(mis=mis, phi_flat=phi_flat.astype(np.float32), phi_eb=phi_eb.astype(np.float32),
                           risk=risk.astype(np.float32), disagree=disagree.astype(np.float32),
                           pub=pub.astype(np.int8), true_des=true_des.astype(np.int8))


def score(prob, truth_mis):
    bins = [0, 0.02, 0.05, 0.10, 0.25, 0.5, 1.0001]
    cal = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (prob >= lo) & (prob < hi)
        if m.sum():
            cal.append(dict(bin=f"[{lo},{min(hi,1)})", n=int(m.sum()), mean_prob=float(prob[m].mean()),
                            observed_rate=float(truth_mis[m].mean())))
    return dict(brier=float(((prob - truth_mis) ** 2).mean()), calibration=cal)


def main():
    t0 = time.time()
    setup()
    D = G["D"]
    val = json.load(open(os.path.join(W, "results", "qct_validation.json")))
    bound = np.isin(D.cap_unit_label[D.unit], val["cap_bound_unit_labels"])
    tasks = [(d, s) for d in ("T1", "T2") for s in range(S_WORLDS)]
    res = {"T1": [], "T2": []}
    with Pool(4, initializer=setup) as pool:
        for k, (design, s, w) in enumerate(pool.imap_unordered(world, tasks)):
            res[design].append(w)
            print(f"  {k+1}/{len(tasks)} worlds  {time.time()-t0:.0f}s", flush=True)
    out = {"S_worlds_per_design": S_WORLDS, "R_inner": R_IN, "K_resurvey": K_RS, "seed": SEED}
    for design in ("T1", "T2"):
        ws = res[design]
        d = {}
        for scope, msk in (("national", np.ones(D.n, bool)), ("cap_bound", bound)):
            agg = lambda f: np.array([f(w) for w in ws])
            sums = dict(
                true_misclassified=agg(lambda w: w["mis"][msk].sum()),
                sum_phi_flat=agg(lambda w: w["phi_flat"][msk].sum()),
                sum_phi_eb=agg(lambda w: w["phi_eb"][msk].sum()),
                true_losses=agg(lambda w: (w["mis"] & w["pub"])[msk].sum()),
                flat_losses=agg(lambda w: (w["phi_flat"] * w["pub"])[msk].sum()),
                eb_losses=agg(lambda w: (w["phi_eb"] * w["pub"])[msk].sum()),
                true_gains=agg(lambda w: (w["mis"] & (1 - w["pub"]))[msk].sum()),
                flat_gains=agg(lambda w: (w["phi_flat"] * (1 - w["pub"]))[msk].sum()),
                eb_gains=agg(lambda w: (w["phi_eb"] * (1 - w["pub"]))[msk].sum()),
                sum_risk=agg(lambda w: w["risk"][msk].sum()),
                sum_two_survey_disagreement=agg(lambda w: w["disagree"][msk].sum()),
                units_truly_at_risk=agg(lambda w: (w["risk"] >= 0.10 - 1e-9)[msk].sum()),
                flat_flagged=agg(lambda w: (w["phi_flat"] >= 0.10 - 1e-9)[msk].sum()),
                eb_flagged=agg(lambda w: (w["phi_eb"] >= 0.10 - 1e-9)[msk].sum()),
                flat_flag_and_at_risk=agg(lambda w: ((w["phi_flat"] >= 0.10 - 1e-9) & (w["risk"] >= 0.10 - 1e-9))[msk].sum()),
                eb_flag_and_at_risk=agg(lambda w: ((w["phi_eb"] >= 0.10 - 1e-9) & (w["risk"] >= 0.10 - 1e-9))[msk].sum()),
            )
            dd = {k: dict(mean=float(v.mean()), sd=float(v.std(ddof=1))) for k, v in sums.items()}
            for lab in ("flat", "eb"):
                fl = sums[f"{lab}_flagged"]; both = sums[f"{lab}_flag_and_at_risk"]; ar = sums["units_truly_at_risk"]
                dd[f"{lab}_band_precision"] = float(both.sum() / fl.sum())
                dd[f"{lab}_band_recall"] = float(both.sum() / ar.sum())
            pm = np.concatenate([w["mis"][msk] for w in ws]).astype(float)
            dd["flat_score"] = score(np.concatenate([w["phi_flat"][msk] for w in ws]).astype(float), pm)
            dd["eb_score"] = score(np.concatenate([w["phi_eb"][msk] for w in ws]).astype(float), pm)
            d[scope] = dd
        out[design] = d
    out["seconds"] = round(time.time() - t0, 1)
    json.dump(out, open(os.path.join(OUT, "known_truth_summary.json"), "w"), indent=2, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
