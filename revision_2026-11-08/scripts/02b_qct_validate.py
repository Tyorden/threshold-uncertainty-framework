#!/usr/bin/env python
"""02b_qct_validate.py: check that qct_fast reproduces the QCT folder's own
pipeline (qct_pipeline.py) exactly, at baseline and on random draws, then
apply HUD's rounding convention and report what it does to the residual."""
import sys, os, time, json
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); W = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import os as _os  # public copy: the program folders (see README) are found through PAPER_PREP
PAPER_PREP = _os.environ.get("PAPER_PREP", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "program_folders"))
sys.path.insert(1, PAPER_PREP + "/QCT_lihtc/scripts")
import numpy as np, pandas as pd
import qct_pipeline as qp
import qct_fast as qf

df = pd.read_parquet(os.path.join(W, "cache", "qct_data_2026.parquet"))
D = qf.QCTData(df)
out = {}
# 1. baseline, folder conventions (no rounding, rank on HUD columns)
t = time.time(); ref = qp.run_full(df).to_numpy(); t_ref = time.time() - t
t = time.time(); mine, el = qf.run(D, rounding=False, baseline_hud_columns=True); t_mine = time.time() - t
out["baseline_folder_conventions"] = dict(
    identical_to_folder_pipeline=bool((ref == mine).all()),
    n_diff=int((ref != mine).sum()),
    match_hud=int((mine == (D.qct == 1)).sum()), n=int(D.n),
    seconds_folder=round(t_ref, 2), seconds_fast=round(t_mine, 3))
# 2. baseline with HUD's rounding convention
mine_r, el_r = qf.run(D, rounding=True, baseline_hud_columns=True)
hud = D.qct == 1
mm_old = np.nonzero(mine != hud)[0]
mm_new = np.nonzero(mine_r != hud)[0]
mism = pd.read_csv(PAPER_PREP + "/QCT_lihtc/results/mismatches.csv", dtype={"GEOID": str})
ten = mism[~mism["my_eligible"]]["GEOID"].tolist()
idx10 = np.nonzero(np.isin(D.geoid, ten))[0]
out["baseline_hud_rounding"] = dict(
    match_hud=int((mine_r == hud).sum()), pct=float(100 * (mine_r == hud).mean()),
    false_pos_replication_only=int((mine_r & ~hud).sum()),
    false_neg_hud_only=int((~mine_r & hud).sum()),
    designated_replication=int(mine_r.sum()), designated_hud=int(hud.sum()),
    recall_on_hud_designated=float((mine_r & hud).sum() / hud.sum()),
    precision=float((mine_r & hud).sum() / mine_r.sum()),
    eligibility_boundary_10_resolved=int((mine_r[idx10] == hud[idx10]).sum()),
    eligibility_boundary_10_raw_ratio_range=[float(np.nanmin((D.inpov / D.univ)[:, idx10][(D.pov_rate_hud[:, idx10] == 250)])),
                                             float(np.nanmax((D.inpov / D.univ)[:, idx10][(D.pov_rate_hud[:, idx10] == 250)]))],
    n_residual=int(len(mm_new)),
    residual_in_old_set=int(np.isin(mm_new, mm_old).sum()),
    new_mismatches_not_in_old=[str(g) for g in D.geoid[np.setdiff1d(mm_new, mm_old)]],
    eligible_tracts=int(el_r.sum()))
# residual decomposition after rounding: eligible-and-cap-margin vs not
out["baseline_hud_rounding"]["residual_eligible_in_replication"] = int(el_r[mm_new].sum())
_, bound = qf.designate(D, *qf.eligibility(D, rounding=True, baseline_hud_columns=True), return_bound=True)
out["baseline_hud_rounding"]["residual_in_cap_bound_units"] = int(bound[D.unit[mm_new]].sum())
out["cap_bound_units_baseline"] = int(bound.sum())
out["cap_bound_unit_labels"] = [str(x) for x in D.cap_unit_label[bound]]
out["tracts_in_cap_bound_units"] = int(bound[D.unit].sum())
# 3. random draws on the five pilot metros: fast engine == folder engine
metros = [35620, 31080, 32820, 26420, 12420]
sub = df[df["cbsa"].isin(metros)].reset_index(drop=True)
S = qf.QCTData(sub)
rng = np.random.default_rng(12345)
L9 = qf.chol9(-0.5, 0.5, qf.R_VINT_OVERLAP)
ok_d, ok_e, tf, tq = 0, 0, 0.0, 0.0
ND = 5
for r in range(ND):
    med, u, p = qf.draw(S, rng, L9)
    t = time.time(); des, el = qf.run(S, med, u, p, rounding=False); tf += time.time() - t
    im = {y: pd.Series(med[j], index=sub.index) for j, y in enumerate(qf.YEARS)}
    uv = {y: pd.Series(u[j], index=sub.index) for j, y in enumerate(qf.YEARS)}
    ip = {y: pd.Series(p[j], index=sub.index) for j, y in enumerate(qf.YEARS)}
    t = time.time(); e2 = qp.eligibility(sub, inc_med=im, univ=uv, inpov=ip); d2 = qp.designate_cap(sub, e2); tq += time.time() - t
    ok_d += int((d2.to_numpy() == des).all()); ok_e += int((e2["eligible"].to_numpy() == el).all())
out["random_draws_five_metros"] = dict(draws=ND, designation_identical=ok_d, eligibility_identical=ok_e,
                                       sec_per_draw_fast=round(tf / ND, 4), sec_per_draw_folder=round(tq / ND, 2))
# 4. national timing for one realistic draw
t = time.time(); med, u, p = qf.draw(D, rng, L9); des, el = qf.run(D, med, u, p); out["national_sec_per_draw_fast"] = round(time.time() - t, 3)
json.dump(out, open(os.path.join(W, "results", "qct_validation.json"), "w"), indent=2)
print(json.dumps(out, indent=2))
