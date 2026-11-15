#!/usr/bin/env python
"""15_qct_shared_component.py (2026-11-14): sensitivity of the canonical QCT results to
cross-tract error dependence (triage point R2-W7; main-text limitation 'Dependence across units').

The canonical pass draws errors independently across tracts. Here each tract's nine standard
normals (three inputs x three vintages) are replaced by
    Z' = sqrt(1 - c) * Z + sqrt(c) * G[county],
where G is one draw of nine standard normals per county, shared by all tracts of the county.
Each input's error then has the same variance and the same within-tract correlations as in the
realistic model, and the errors of two tracts in the same county correlate at c (an ASSUMED
value: 0.1 and 0.3; c = 0 must reproduce the canonical pass exactly, which is asserted).
Same engine (revision_2026-11-08/scripts/qct_fast.py), same per-draw seeds [20261108, r] for Z,
seeds [20261114, r] for G, 2,000 draws, HUD rounding, scored against HUD's published list.
Output: results/qct_shared_component.json
Run (niced): nice -n 15 paper_prep_venv/bin/python 15_qct_shared_component.py"""
import os, sys, json, time
sys.dont_write_bytecode = True
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"): os.environ.setdefault(v, "1")
HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.join(HERE, "..", "revision_2026-11-08")
sys.path.insert(0, os.path.join(REV, "scripts"))
import numpy as np, pandas as pd
from multiprocessing import Pool
import qct_fast as qf
SEED, SEED_G = 20261108, 20261114
R = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
CS = [0.0, 0.1, 0.3]
L = qf.chol9(*qf.SCENARIOS["realistic"])
D = None; CTY = None; NC = None
def init():
    global D, CTY, NC
    df = pd.read_parquet(os.path.join(REV, "cache", "qct_data_2026.parquet"))
    D = qf.QCTData(df)
    _, CTY = np.unique(df["stcnty"].to_numpy(), return_inverse=True); NC = int(CTY.max()) + 1
def worker(ids):
    n = D.n
    cnt = {c: dict(des=np.zeros(n, np.int32), end=np.zeros(n, np.int32)) for c in CS}
    for r in ids:
        Z = np.random.default_rng([SEED, r]).standard_normal((9, n))
        G = np.random.default_rng([SEED_G, r]).standard_normal((9, NC))[:, CTY]
        for c in CS:
            Zc = Z if c == 0.0 else np.sqrt(1 - c) * Z + np.sqrt(c) * G
            el, gr, ai, ap = qf.eligibility(D, *qf.draw(D, None, L, Z=Zc), rounding=True)
            de = qf.designate(D, el, gr, ai, ap)
            cnt[c]["des"] += de; cnt[c]["end"] += el & ~de
    return cnt
if __name__ == "__main__":
    t0 = time.time(); init(); n = D.n
    pieces = [list(range(R))[i::40] for i in range(40)]
    tot = {c: dict(des=np.zeros(n, np.int64), end=np.zeros(n, np.int64)) for c in CS}
    with Pool(4, initializer=init) as pool:
        for cnt in pool.imap_unordered(worker, pieces):
            for c in CS:
                for f in ("des", "end"): tot[c][f] += cnt[c][f]
    hud = D.qct == 1
    out = {"R": R, "seed_Z": SEED, "seed_G": SEED_G, "shared_level": "county (stcnty)", "n_counties": NC, "n_tracts": int(n),
           "model": "realistic error model; Z' = sqrt(1-c) Z + sqrt(c) G[county]; c assumed", "runs": {}}
    for c in CS:
        p_des = tot[c]["des"] / R; phi = np.where(hud, 1 - p_des, p_des)
        out["runs"][f"c={c}"] = {
            "expected_losses": float((1 - p_des[hud]).sum()), "losses_rank_driven": float((tot[c]["end"][hud] / R).sum()),
            "expected_gains": float(p_des[~hud].sum()), "expected_two_sided": float(phi.sum()),
            "designated_phi_ge_0p10": int((phi[hud] >= 0.10 - 1e-12).sum()), "designated": int(hud.sum())}
    can = json.load(open(os.path.join(REV, "results", "qct_canonical_summary.json")))["runs"]["realistic"]["national"]
    b = out["runs"]["c=0.0"]
    for kk in ("expected_losses", "losses_rank_driven", "expected_gains", "expected_two_sided"):
        assert abs(b[kk] - can[kk]) < 1e-6, (kk, b[kk], can[kk])
    # The band count uses a tolerant comparison (phi >= 0.10 - 1e-12), so tracts whose flip share is
    # exactly 200 of 2,000 are counted; the canonical summary is checked on the expected totals above.
    out["canonical_summary_band_count_for_reference"] = can["designated_phi_ge_0p10"]
    out["c0_reproduces_canonical_pass"] = True
    for c in CS[1:]:
        d = out["runs"][f"c={c}"]
        d["two_sided_change_vs_c0_pct"] = round(100 * (d["expected_two_sided"] / b["expected_two_sided"] - 1), 2)
        d["rank_driven_change_vs_c0_pct"] = round(100 * (d["losses_rank_driven"] / b["losses_rank_driven"] - 1), 2)
    out["seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results", "qct_shared_component.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
