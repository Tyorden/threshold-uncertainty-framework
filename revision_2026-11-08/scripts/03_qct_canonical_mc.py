#!/usr/bin/env python
"""03_qct_canonical_mc.py: the canonical QCT pass for the revision.

All 85,390 tracts (every cap unit, so slack units that bind in a draw are
handled), HUD's poverty-rate rounding applied, 2,000 paired draws per run
(per-draw seeds [SEED, r], so results do not depend on the worker split).
Draws are centred on the published estimates: they are posterior draws under
a flat prior, so a tract's flip share estimates the flat-prior posterior
probability that the rule applied to the true inputs gives a different
designation from the published list (the estimand; see text/estimand.tex).

Runs (all on the same standard normals per draw):
  error scenarios   indep, xvintage_only, realistic (canonical), xvintage_high,
                    realistic_ip_lo, realistic_ip_hi, realistic_lognormal_inc,
                    realistic_se075, realistic_se125
  rule variants     latest_only, avg3 (under realistic errors)
  channel split     elig_real_rank_indep, elig_indep_rank_real
Per-tract counts are saved for every run; per-draw designation and
eligibility bit matrices are saved for 'realistic' (bootstrap MC errors).
"""
import os, sys, json, time
sys.dont_write_bytecode = True
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__)); W = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import numpy as np, pandas as pd
from multiprocessing import Pool
import qct_fast as qf

SEED = 20261108
R = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
NWORK = 4
OUT = os.path.join(W, "results", "qct_canonical")
os.makedirs(OUT, exist_ok=True)

L = {k: qf.chol9(rip, rup, Rv) for k, (rip, rup, Rv) in qf.SCENARIOS.items()}
RUNS = ["indep", "xvintage_only", "realistic", "xvintage_high", "realistic_ip_lo",
        "realistic_ip_hi", "realistic_lognormal_inc", "realistic_se075", "realistic_se125",
        "latest_only", "avg3", "elig_real_rank_indep", "elig_indep_rank_real"]

D = None


def init():
    global D
    D = qf.QCTData(pd.read_parquet(os.path.join(W, "cache", "qct_data_2026.parquet")))


def lognormal_draw(Z):
    med, u, p = qf.draw(D, None, L["realistic"], Z=Z)
    X = (L["realistic"] @ Z).reshape(3, 3, D.n)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = D.med_se / D.med
    med_ln = D.med * np.exp(X[0] * np.nan_to_num(s))
    med_ln = np.where(np.isnan(D.med), np.nan, np.clip(med_ln, 1.0, None))
    return med_ln, u, p


def worker(draw_ids):
    n = D.n
    cnt = {k: dict(des=np.zeros(n, np.int32), elig=np.zeros(n, np.int32),
                   elig_not_des=np.zeros(n, np.int32)) for k in RUNS}
    bits_des, bits_el = [], []
    for r in draw_ids:
        rng = np.random.default_rng([SEED, r])
        Z = rng.standard_normal((9, n))
        draws = {k: qf.draw(D, None, L[k], Z=Z) for k in qf.SCENARIOS}
        draws["realistic_lognormal_inc"] = lognormal_draw(Z)
        draws["realistic_se075"] = qf.draw(D, None, L["realistic"], scale=0.75, Z=Z)
        draws["realistic_se125"] = qf.draw(D, None, L["realistic"], scale=1.25, Z=Z)
        el_cache = {}
        for k in RUNS:
            if k in ("latest_only", "avg3"):
                el, gr, ai, ap = qf.eligibility(D, *draws["realistic"], rounding=True, variant=k)
            elif k == "elig_real_rank_indep":
                el, gr, _, _ = el_cache.get("realistic") or qf.eligibility(D, *draws["realistic"], rounding=True)
                _, _, ai, ap = el_cache.get("indep") or qf.eligibility(D, *draws["indep"], rounding=True)
            elif k == "elig_indep_rank_real":
                el, gr, _, _ = el_cache.get("indep") or qf.eligibility(D, *draws["indep"], rounding=True)
                _, _, ai, ap = el_cache.get("realistic") or qf.eligibility(D, *draws["realistic"], rounding=True)
            else:
                el, gr, ai, ap = qf.eligibility(D, *draws[k], rounding=True)
                el_cache[k] = (el, gr, ai, ap)
            de = qf.designate(D, el, gr, ai, ap)
            cnt[k]["des"] += de
            cnt[k]["elig"] += el
            cnt[k]["elig_not_des"] += el & ~de
            if k == "realistic":
                bits_des.append(np.packbits(de)); bits_el.append(np.packbits(el))
    return draw_ids, cnt, np.array(bits_des), np.array(bits_el)


def main():
    t0 = time.time()
    chunks = [list(range(R))[i::NWORK] for i in range(NWORK)]
    # finer chunks for progress
    pieces = []
    for c in chunks:
        for j in range(0, len(c), 50):
            pieces.append(c[j:j + 50])
    init()
    n = D.n
    tot = {k: dict(des=np.zeros(n, np.int64), elig=np.zeros(n, np.int64),
                   elig_not_des=np.zeros(n, np.int64)) for k in RUNS}
    bits_des = np.zeros((R, (n + 7) // 8), np.uint8)
    bits_el = np.zeros((R, (n + 7) // 8), np.uint8)
    done = 0
    with Pool(NWORK, initializer=init) as pool:
        for ids, cnt, bd, be in pool.imap_unordered(worker, pieces):
            for k in RUNS:
                for f in ("des", "elig", "elig_not_des"):
                    tot[k][f] += cnt[k][f]
            bits_des[ids] = bd; bits_el[ids] = be
            done += len(ids)
            print(f"  {done}/{R} draws  {time.time()-t0:.0f}s", flush=True)
    np.save(os.path.join(OUT, "bits_des_realistic.npy"), bits_des)
    np.save(os.path.join(OUT, "bits_elig_realistic.npy"), bits_el)
    base = pd.DataFrame({"GEOID": D.geoid, "cap_unit": D.cap_unit_label[D.unit], "P1": D.P1,
                         "qct_hud": D.qct})
    des_rep, el_rep = qf.run(D, rounding=True, baseline_hud_columns=True)
    base["des_replication"] = des_rep.astype(int)
    base["elig_replication"] = el_rep.astype(int)
    for k in RUNS:
        for f in ("des", "elig", "elig_not_des"):
            base[f"{k}__{f}"] = tot[k][f]
    base.to_parquet(os.path.join(OUT, "per_tract_counts.parquet"))
    json.dump(dict(R=R, seed=SEED, runs=RUNS, n_tracts=int(n), seconds=round(time.time() - t0, 1),
                   seeding="np.random.default_rng([SEED, r]) per draw r, 9 x n standard normals, shared by all runs"),
              open(os.path.join(OUT, "run_meta.json"), "w"), indent=2)
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
