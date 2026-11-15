#!/usr/bin/env python
"""06_qct_decomposition.py: conditional decomposition of designation flips in
cap-bound units (triage R1-W7, R1-Q6, R2-W7).

Targets: tracts on HUD's list (and on the replication's), in a cap-bound unit,
whose rank-driven flip probability P(eligible and not designated) is at least
0.10 in the canonical pass. For each target i and each draw:
  full         all tracts in the unit redrawn (realistic correlated errors)
  others_only  tract i held at its published inputs, all others redrawn
  own_only     all others held at their published inputs, tract i redrawn
Designation is computed for the unit only (cap units are independent).
Each target's draws come from per-(unit, draw) seeds, so the three variants
use the same draws. HUD rounding on; averages recomputed from the inputs in
every variant (the same convention as the draws in 03).
"""
import os, sys, json, time
sys.dont_write_bytecode = True
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__)); W = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import numpy as np, pandas as pd
from scipy.stats import rankdata
from multiprocessing import Pool
import qct_fast as qf

SEED = 20261109
R = int(sys.argv[1]) if len(sys.argv) > 1 else 300
OUT = os.path.join(W, "results", "qct_decomposition")
os.makedirs(OUT, exist_ok=True)
L9 = qf.chol9(-0.5, 0.5, qf.R_VINT_OVERLAP)
G = {}


def rk(v):
    out = np.full(v.shape, np.nan); ok = ~np.isnan(v)
    if ok.any():
        out[ok] = rankdata(-v[ok], method="min")
    return out


def designate_unit(P1, cap, el, gr, ai, ap):
    if P1[el].sum() <= cap:
        return el.copy()
    des = np.zeros(len(P1), bool)
    parts = []
    for gval in (1, 2):
        eg = np.nonzero(el & (gr == gval))[0]
        if len(eg) == 0:
            continue
        comb = (rk(ai[eg]) + rk(ap[eg])) / 2.0
        key = np.where(np.isnan(comb), np.inf, comb)
        parts.append(eg[np.lexsort((-P1[eg], key))])
    ranked = np.concatenate(parts)
    pops = P1[ranked]
    cs = np.cumsum(pops)
    k = int(np.searchsorted(cs, cap, side="right"))
    des[ranked[:k]] = True
    cum = cs[k - 1] if k > 0 else 0.0
    pos = k + 1
    while pos < len(ranked):
        fits = np.nonzero(cum + pops[pos:] <= cap)[0]
        if len(fits) == 0:
            break
        j = pos + fits[0]
        des[ranked[j]] = True; cum += pops[j]; pos = j + 1
    return des


def init():
    df = pd.read_parquet(os.path.join(W, "cache", "qct_data_2026.parquet"))
    G["df"] = df
    D = qf.QCTData(df)
    G["unit_lab"] = D.cap_unit_label[D.unit]


def task(args):
    unit_label, tgt_geoids, uid = args
    df = G["df"]
    m = G["unit_lab"] == unit_label
    Du = qf.QCTData(df[m])
    cap = Du.cap_pop[0]
    tg = np.nonzero(np.isin(Du.geoid, tgt_geoids))[0]
    el_b, gr_b, ai_b, ap_b = qf.eligibility(Du, rounding=True)
    full = np.zeros(len(tg)); oth = np.zeros(len(tg)); own = np.zeros(len(tg))
    for r in range(R):
        rng = np.random.default_rng([SEED, uid, r])
        med, u, p = qf.draw(Du, rng, L9)
        el_d, gr_d, ai_d, ap_d = qf.eligibility(Du, med, u, p, rounding=True)
        des_full = designate_unit(Du.P1, cap, el_d, gr_d, ai_d, ap_d)
        full += des_full[tg]
        for n_, i in enumerate(tg):
            e, g_, a1, a2 = el_d.copy(), gr_d.copy(), ai_d.copy(), ap_d.copy()
            e[i], g_[i], a1[i], a2[i] = el_b[i], gr_b[i], ai_b[i], ap_b[i]
            oth[n_] += designate_unit(Du.P1, cap, e, g_, a1, a2)[i]
            e, g_, a1, a2 = el_b.copy(), gr_b.copy(), ai_b.copy(), ap_b.copy()
            e[i], g_[i], a1[i], a2[i] = el_d[i], gr_d[i], ai_d[i], ap_d[i]
            own[n_] += designate_unit(Du.P1, cap, e, g_, a1, a2)[i]
    return pd.DataFrame({"GEOID": Du.geoid[tg], "cap_unit": unit_label,
                         "phi_full": 1 - full / R, "phi_others_only": 1 - oth / R, "phi_own_only": 1 - own / R,
                         "baseline_designated_recomputed": des_b_unit(Du, cap, el_b, gr_b, ai_b, ap_b)[tg]})


def des_b_unit(Du, cap, el, gr, ai, ap):
    return designate_unit(Du.P1, cap, el, gr, ai, ap).astype(int)


def main():
    t0 = time.time()
    tr = pd.read_parquet(os.path.join(W, "results", "qct_canonical", "per_tract_realistic.parquet"))
    val = json.load(open(os.path.join(W, "results", "qct_validation.json")))
    bound = set(val["cap_bound_unit_labels"])
    tgt = tr[(tr["hud"] == 1) & (tr["replication"] == 1) & tr["cap_unit"].isin(bound) & (tr["p_elig_not_des"] >= 0.10 - 1e-9)]
    print(f"targets: {len(tgt)} in {tgt['cap_unit'].nunique()} cap-bound units; R={R}", flush=True)
    units = sorted(tgt["cap_unit"].unique())
    uid = {u: k for k, u in enumerate(units)}
    tasks = []
    for u in units:
        g = tgt[tgt["cap_unit"] == u]["GEOID"].tolist()
        for j in range(0, len(g), 25):
            tasks.append((u, g[j:j + 25], uid[u]))
    tasks.sort(key=lambda t: -len(t[1]))
    res = []
    with Pool(4, initializer=init) as pool:
        for k, d in enumerate(pool.imap_unordered(task, tasks)):
            res.append(d)
            if (k + 1) % 10 == 0:
                print(f"  {k+1}/{len(tasks)} tasks  {time.time()-t0:.0f}s", flush=True)
    out = pd.concat(res).merge(tgt[["GEOID", "P1", "p_eligible", "p_elig_not_des", "phi_vs_hud"]], on="GEOID")
    out.to_csv(os.path.join(OUT, "decomposition_targets.csv"), index=False)
    s = dict(R=R, seed=SEED, n_targets=int(len(out)), n_units=int(out["cap_unit"].nunique()),
             others_only_ge_0p10=int((out["phi_others_only"] >= 0.10 - 1e-9).sum()),
             own_only_ge_0p10=int((out["phi_own_only"] >= 0.10 - 1e-9).sum()),
             both_ge_0p10=int(((out["phi_others_only"] >= 0.10 - 1e-9) & (out["phi_own_only"] >= 0.10 - 1e-9)).sum()),
             others_only_ge_0p05=int((out["phi_others_only"] >= 0.05 - 1e-9).sum()),
             own_only_lt_0p05_and_others_ge_0p10=int(((out["phi_own_only"] < 0.05 - 1e-9) & (out["phi_others_only"] >= 0.10 - 1e-9)).sum()),
             mean_phi_full=float(out["phi_full"].mean()), mean_phi_others_only=float(out["phi_others_only"].mean()),
             mean_phi_own_only=float(out["phi_own_only"].mean()),
             baseline_recomputed_designated=int(out["baseline_designated_recomputed"].sum()),
             seconds=round(time.time() - t0, 1))
    ex = out[out["GEOID"] == "06059074806"]
    s["exhibit_06059074806"] = ex.to_dict(orient="records")
    json.dump(s, open(os.path.join(OUT, "decomposition_summary.json"), "w"), indent=2, default=float)
    print(json.dumps(s, indent=2, default=float))


if __name__ == "__main__":
    main()
