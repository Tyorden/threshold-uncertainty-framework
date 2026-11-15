#!/usr/bin/env python
"""16_qct_screen_fixed.py (2026-11-14): how much of the canonical QCT total comes from the
reliability-screen convention.

In the canonical pass the reliability screen (margin of error below half the estimate) is
recomputed in every draw from the drawn value and the published margin. Here the same engine
(revision_2026-11-08/scripts/qct_fast.py), the same per-draw seeds [20261108, r], the realistic
error model, HUD rounding and 2,000 draws are used, and ONLY the screen changes:
  canonical        screen recomputed in each draw (must reproduce the canonical pass; asserted)
  pov_screen_fixed poverty screen (universe and count in poverty) held at its published status
  all_screens_fixed poverty and income screens both held at their published status
All runs are scored against HUD's published list.
Output: results/qct_screen_fixed.json
Run (niced): nice -n 10 paper_prep_venv/bin/python 16_qct_screen_fixed.py"""
import os, sys, json, time
sys.dont_write_bytecode = True
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"): os.environ.setdefault(v, "1")
HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.join(HERE, "..", "revision_2026-11-08")
sys.path.insert(0, os.path.join(REV, "scripts"))
import numpy as np, pandas as pd
from multiprocessing import Pool
import qct_fast as qf
SEED = 20261108
R = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
RUNS = ["canonical", "pov_screen_fixed", "all_screens_fixed"]
L = qf.chol9(*qf.SCENARIOS["realistic"])
D = None; PUB = None; MODE = "canonical"
_orig = qf._moer_lt
def _patched(est, moe):
    # hold a screen at its published status by answering with the published mask
    if MODE != "canonical":
        if moe is D.univ_moe: return PUB["univ"]
        if moe is D.inpov_moe: return PUB["inpov"]
        if MODE == "all_screens_fixed" and moe is D.med_moe: return PUB["med"]
    return _orig(est, moe)
def init():
    global D, PUB
    D = qf.QCTData(pd.read_parquet(os.path.join(REV, "cache", "qct_data_2026.parquet")))
    PUB = {"med": _orig(D.med, D.med_moe), "univ": _orig(D.univ, D.univ_moe), "inpov": _orig(D.inpov, D.inpov_moe)}
    qf._moer_lt = _patched
def worker(ids):
    global MODE
    n = D.n
    cnt = {k: dict(des=np.zeros(n, np.int32), end=np.zeros(n, np.int32)) for k in RUNS}
    for r in ids:
        Z = np.random.default_rng([SEED, r]).standard_normal((9, n))
        dr = qf.draw(D, None, L, Z=Z)
        for k in RUNS:
            MODE = k
            el, gr, ai, ap = qf.eligibility(D, *dr, rounding=True)
            de = qf.designate(D, el, gr, ai, ap)
            cnt[k]["des"] += de; cnt[k]["end"] += el & ~de
    MODE = "canonical"
    return cnt
if __name__ == "__main__":
    t0 = time.time(); init(); n = D.n
    pieces = [list(range(R))[i::40] for i in range(40)]
    tot = {k: dict(des=np.zeros(n, np.int64), end=np.zeros(n, np.int64)) for k in RUNS}
    with Pool(4, initializer=init) as pool:
        for cnt in pool.imap_unordered(worker, pieces):
            for k in RUNS:
                for f in ("des", "end"): tot[k][f] += cnt[k][f]
    hud = D.qct == 1
    out = {"R": R, "seed": SEED, "n_tracts": int(n), "model": "realistic error model; only the reliability screen differs between runs",
           "published_screen_true_share": {a: float(PUB[a].mean()) for a in PUB}, "runs": {}}
    for k in RUNS:
        p_des = tot[k]["des"] / R; phi = np.where(hud, 1 - p_des, p_des)
        out["runs"][k] = {
            "expected_losses": float((1 - p_des[hud]).sum()), "losses_rank_driven": float((tot[k]["end"][hud] / R).sum()),
            "expected_gains": float(p_des[~hud].sum()), "expected_two_sided": float(phi.sum()),
            "designated_phi_ge_0p10": int((phi[hud] >= 0.10 - 1e-9).sum()), "designated": int(hud.sum())}
    can = json.load(open(os.path.join(REV, "results", "qct_canonical_summary.json")))["runs"]["realistic"]["national"]
    b = out["runs"]["canonical"]
    for kk in ("expected_losses", "losses_rank_driven", "expected_gains", "expected_two_sided"):
        assert abs(b[kk] - can[kk]) < 1e-6, (kk, b[kk], can[kk])
    assert b["designated_phi_ge_0p10"] == can["designated_phi_ge_0p10"]
    out["canonical_reproduces_canonical_pass"] = True
    for k in RUNS[1:]:
        d = out["runs"][k]
        d["two_sided_change_vs_canonical_pct"] = round(100 * (d["expected_two_sided"] / b["expected_two_sided"] - 1), 1)
    out["seconds"] = round(time.time() - t0, 1)
    json.dump(out, open(os.path.join(HERE, "results", "qct_screen_fixed.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
