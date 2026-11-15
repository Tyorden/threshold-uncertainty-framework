#!/usr/bin/env python
"""10_mua_anchor_designation.py: designation-level agreement for the MUA
anchor at the IMU <= 62.0 cut (triage R1-W9, R1-Q9, R2-W5).

Replicates IMU_underserved/scripts/01_load_and_anchor.py exactly (same data,
same 50/50 train/test split of conversion-table anchor points, seed 42, same
order), checks that the end-to-end score agreement reproduces the folder's
0.9669, then scores the binary designation (IMU <= 62.0) of the reconstructed
score against HRSA's published score, at row and area level, overall and near
the cut, and counts how many of the canonical run's fragile areas carry an
end-to-end discrepancy above one point. Read only from IMU_underserved.
"""
import json, os
from decimal import Decimal, ROUND_HALF_UP
import numpy as np, pandas as pd

W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import os as _os  # public copy: the program folders (see README) are found through PAPER_PREP
PAPER_PREP = _os.environ.get("PAPER_PREP", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "program_folders"))
IMU = PAPER_PREP + "/IMU_underserved"
RAW = {'doc': 'Providers per 1000 Population',
       'pov': 'Percent of Population with Incomes at or Below 100 Percent of the U.S. Federal Poverty Level',
       'p65': 'Percentage of Population Age 65 and Over',
       'imr': 'Infant Mortality Rate'}
SUB = {'doc': 'Ratio of Providers per 1000 Population IMU Score',
       'pov': 'Percent of Population with Incomes at or Below 100 Percent of the U.S. Federal Poverty Level Index of Medical Underservice Score',
       'p65': 'Percentage of Population Age 65 and Over IMU Score',
       'imr': 'Infant Mortality Rate IMU Score'}
CAP = {'doc': 28.7, 'imr': 26.0, 'pov': 25.1, 'p65': 20.2}
CUT = 62.0


def rhu(x):
    return float(Decimal(str(x)).quantize(Decimal('1'), rounding=ROUND_HALF_UP))


df = pd.read_csv(f"{IMU}/data/MUA_DET.csv", dtype=str, low_memory=False)
cols = {'IMU Score': 'imu', 'MUA/P Designation Date String': 'ddate', 'MUA/P Status Description': 'status',
        'Designation Type': 'dtype', 'Rural Status Description': 'rural', 'MUA/P ID': 'id'}
cols.update({v: k for k, v in RAW.items()})
cols.update({v: k + '_s' for k, v in SUB.items()})
d = df.rename(columns=cols)
for c in ['imu'] + list(RAW) + [k + '_s' for k in SUB]:
    d[c] = pd.to_numeric(d[c], errors='coerce')
full = d.dropna(subset=[k + '_s' for k in SUB] + ['imu']).copy()
rng = np.random.default_rng(42)
recon = pd.DataFrame(index=full.index)
for k in ['doc', 'pov', 'p65', 'imr']:
    sub = d.dropna(subset=[k, k + '_s'])[[k, k + '_s']].rename(columns={k: 'raw', k + '_s': 'sub'})
    uniq = sub.drop_duplicates().sort_values('raw').reset_index(drop=True)
    mask = rng.random(len(uniq)) < 0.5
    train = uniq[mask] if mask.sum() > 2 else uniq
    g = train.groupby('raw')['sub'].mean().reset_index().sort_values('raw')
    xs, ys = g['raw'].values, g['sub'].values
    recon[k] = np.clip(np.interp(full[k].values, xs, ys, left=ys[0], right=ys[-1]), 0, CAP[k])
recon['imu_e2e'] = recon[['doc', 'pov', 'p65', 'imr']].sum(axis=1).apply(rhu)
full['imu_e2e'] = recon['imu_e2e']
err = np.abs(full['imu_e2e'] - full['imu'])
out = {"rows_scored": int(len(full)), "replicates_folder_le_1.0": float((err <= 1.0).mean())}


def confusion(off, rec):
    return dict(agree=int((off == rec).sum()), both_designated=int((off & rec).sum()), official_only=int((off & ~rec).sum()),
                reconstruction_only=int((~off & rec).sum()), neither=int((~off & ~rec).sum()),
                agreement=float((off == rec).mean()),
                recall_on_official=float((off & rec).sum() / max(off.sum(), 1)),
                precision=float((off & rec).sum() / max(rec.sum(), 1)))


off = (full['imu'] <= CUT).to_numpy(); rec = (full['imu_e2e'] <= CUT).to_numpy()
near = (np.abs(full['imu'] - CUT) <= 2.0).to_numpy()
out["row_level"] = confusion(off, rec)
out["row_level_within_2_points_of_cut"] = dict(n=int(near.sum()), **confusion(off[near], rec[near]))
area = full.groupby('id').agg(imu=('imu', 'first'), imu_e2e=('imu_e2e', 'first'),
                              n_rows=('imu', 'size'), imu_nuniq=('imu', 'nunique'), e2e_nuniq=('imu_e2e', 'nunique'))
out["areas"] = int(len(area))
out["areas_with_inconsistent_rows"] = int(((area['imu_nuniq'] > 1) | (area['e2e_nuniq'] > 1)).sum())
ao = (area['imu'] <= CUT).to_numpy(); ar = (area['imu_e2e'] <= CUT).to_numpy()
out["area_level"] = confusion(ao, ar)
an = (np.abs(area['imu'] - CUT) <= 2.0).to_numpy()
out["area_level_within_2_points_of_cut"] = dict(n=int(an.sum()), **confusion(ao[an], ar[an]))
# overlap with the canonical run's fragile areas (score-qualified pilot areas, flip >= 0.10)
pil = pd.read_csv(f"{IMU}/results/pilot_flip_probs.csv", dtype={"id": str})
pil = pil[pil["imu"] <= CUT]
area_err = (area['imu_e2e'] - area['imu']).abs()
m = pil.merge(area_err.rename("e2e_abs_err").reset_index(), on="id", how="left")
frag = m[m["flip_all"] >= 0.10]
out["pilot_score_qualified"] = int(len(pil))
out["pilot_matched_to_anchor_rows"] = int(m["e2e_abs_err"].notna().sum())
out["fragile_ge_0p10"] = int(len(frag))
out["fragile_with_e2e_error_gt_1pt"] = int((frag["e2e_abs_err"] > 1.0).sum())
out["fragile_matched"] = int(frag["e2e_abs_err"].notna().sum())
out["pilot_designation_year_le_1978"] = int((pd.to_datetime(pil["ddate"], errors="coerce").dt.year <= 1978).sum())
out["fragile_designation_year_lt_2000"] = int((pd.to_datetime(frag["ddate"], errors="coerce").dt.year < 2000).sum())
json.dump(out, open(os.path.join(W, "results", "mua_anchor_designation.json"), "w"), indent=2)
print(json.dumps(out, indent=2))
