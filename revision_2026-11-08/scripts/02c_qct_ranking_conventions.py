#!/usr/bin/env python
"""02c_qct_ranking_conventions.py: test candidate ranking/fill conventions
against the 221 cap-margin residual tracts (triage R2-W5). Baseline only, HUD
rounding on. Each variant changes one convention."""
import sys, os, json
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); W = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import numpy as np, pandas as pd
from scipy.stats import rankdata
import qct_fast as qf

df = pd.read_parquet(os.path.join(W, "cache", "qct_data_2026.parquet"))
D = qf.QCTData(df)
hud = D.qct == 1

def avgs(kind):
    inc_rel = qf._moer_lt(D.med, D.med_moe)
    pov_rel = qf._moer_lt(D.univ, D.univ_moe) & qf._moer_lt(D.inpov, D.inpov_moe)
    with np.errstate(all="ignore"):
        if kind == "hud_cols_reliable":
            a = np.nanmean(np.where(inc_rel, D.inc_factor_hud, np.nan), 0); b = np.nanmean(np.where(pov_rel, D.pov_rate_hud, np.nan), 0)
        elif kind == "hud_cols_all3_zero":
            a = np.nanmean(D.inc_factor_hud, 0); b = np.nanmean(D.pov_rate_hud, 0)
        elif kind == "raw_reliable":
            a = np.nanmean(np.where(inc_rel, D.adj_lim / D.med, np.nan), 0); b = np.nanmean(np.where(pov_rel, D.inpov / D.univ, np.nan), 0)
        elif kind == "hud_cols_reliable_round_avg":
            a = np.round(np.nanmean(np.where(inc_rel, D.inc_factor_hud, np.nan), 0), 4)
            b = np.round(np.nanmean(np.where(pov_rel, D.pov_rate_hud, np.nan), 0), 0)
        elif kind == "hud_cols_reliable_float32":
            a = np.nanmean(np.where(inc_rel, D.inc_factor_hud, np.nan), 0).astype(np.float32).astype(float)
            b = np.nanmean(np.where(pov_rel, D.pov_rate_hud, np.nan), 0).astype(np.float32).astype(float)
    return a, b

def ranked(e_idx, group, ai, ap, method="min", within_group=True, tie_pop_desc=True, comb_kind="mean"):
    def rk(v):
        out = np.full(v.shape, np.nan); ok = ~np.isnan(v)
        if ok.any(): out[ok] = rankdata(-v[ok], method=method)
        return out
    parts = []
    if within_group:
        sets = [e_idx[group[e_idx] == 1], e_idx[group[e_idx] == 2]]
        ranks = []
        for eg in sets:
            ranks.append((rk(ai[eg]), rk(ap[eg])))
    else:
        ri, rp = rk(ai[e_idx]), rk(ap[e_idx]); pos = {j: n for n, j in enumerate(e_idx)}
        sets = [e_idx[group[e_idx] == 1], e_idx[group[e_idx] == 2]]
        ranks = [(ri[[pos[j] for j in eg]], rp[[pos[j] for j in eg]]) for eg in sets]
    for eg, (r1, r2) in zip(sets, ranks):
        if len(eg) == 0: continue
        comb = (r1 + r2) / 2.0 if comb_kind == "mean" else np.minimum(r1, r2)
        if comb_kind == "nan_as_worst":
            r1 = np.where(np.isnan(r1), len(eg) + 1.0, r1); r2 = np.where(np.isnan(r2), len(eg) + 1.0, r2); comb = (r1 + r2) / 2.0
        key = np.where(np.isnan(comb), np.inf, comb)
        o = np.lexsort(((-D.P1[eg]) if tie_pop_desc else D.P1[eg], key))
        parts.append(eg[o])
    return np.concatenate(parts) if parts else e_idx[:0]

def designate(elig, group, ai, ap, strict=False, popcol=None, **kw):
    P = D.P1 if popcol is None else popcol
    des = np.zeros(D.n, bool)
    elig_pop = np.bincount(D.unit[elig], weights=P[elig], minlength=D.U)
    bound = elig_pop > D.cap_pop
    des[elig & ~bound[D.unit]] = True
    for u in np.nonzero(bound)[0]:
        idx = D.unit_index[u]; e = idx[elig[idx]]
        rr = ranked(e, group, ai, ap, **kw)
        cum = 0.0; cap = D.cap_pop[u]
        for j in rr:
            if (cum + P[j] < cap) if strict else (cum + P[j] <= cap):
                des[j] = True; cum += P[j]
    return des

el, gr, _, _ = qf.eligibility(D, rounding=True, baseline_hud_columns=True)
res = {}
variants = {
    "V0_current": dict(avg="hud_cols_reliable"),
    "V1_avg_all3_with_zeros": dict(avg="hud_cols_all3_zero"),
    "V2_rank_average": dict(avg="hud_cols_reliable", method="average"),
    "V3_rank_max": dict(avg="hud_cols_reliable", method="max"),
    "V3b_rank_dense": dict(avg="hud_cols_reliable", method="dense"),
    "V3c_rank_ordinal": dict(avg="hud_cols_reliable", method="ordinal"),
    "V4_rank_across_groups": dict(avg="hud_cols_reliable", within_group=False),
    "V5_raw_unrounded_avgs": dict(avg="raw_reliable"),
    "V6_rounded_avgs": dict(avg="hud_cols_reliable_round_avg"),
    "V6b_float32_avgs": dict(avg="hud_cols_reliable_float32"),
    "V7_tie_less_populous_first": dict(avg="hud_cols_reliable", tie_pop_desc=False),
    "V8_strict_lt_cap": dict(avg="hud_cols_reliable", strict=True),
    "V9_nan_rank_as_worst": dict(avg="hud_cols_reliable", comb_kind="nan_as_worst"),
    "V10_household_pop_cap": dict(avg="hud_cols_reliable", popcol=(D.df["P1"] - D.df["P5"]).to_numpy(float)),
}
for name, kw in variants.items():
    kw = dict(kw); ai, ap = avgs(kw.pop("avg"))
    des = designate(el, gr, ai, ap, **kw)
    res[name] = dict(match=int((des == hud).sum()), residual=int((des != hud).sum()),
                     replication_only=int((des & ~hud).sum()), hud_only=int((~des & hud).sum()))
    print(name, res[name], flush=True)
# where do the 221 sit: per unit counts and the cap slack in HUD's own list
des0 = designate(el, gr, *avgs("hud_cols_reliable"))
mm = des0 != hud
units = pd.Series(D.cap_unit_label[D.unit[mm]]).value_counts()
hud_pop = np.bincount(D.unit[hud], weights=D.P1[hud], minlength=D.U)
my_pop = np.bincount(D.unit[des0], weights=D.P1[des0], minlength=D.U)
rows = []
for lab, cnt in units.items():
    u = int(np.nonzero(D.cap_unit_label == lab)[0][0])
    rows.append(dict(cap_unit=lab, residual_tracts=int(cnt), area_pop=float(D.area_pop[u]),
                     hud_designated_share=float(hud_pop[u] / D.area_pop[u]), replication_share=float(my_pop[u] / D.area_pop[u])))
pd.DataFrame(rows).to_csv(os.path.join(W, "results", "qct_residual_by_unit.csv"), index=False)
over = [r for r in rows if r["hud_designated_share"] > 0.2]
res["units_with_residual"] = len(rows)
res["units_where_hud_list_exceeds_20pct"] = len(over)
res["hud_share_range_in_residual_units"] = [min(r["hud_designated_share"] for r in rows), max(r["hud_designated_share"] for r in rows)]
json.dump(res, open(os.path.join(W, "results", "qct_ranking_conventions.json"), "w"), indent=2)
print(json.dumps({k: v for k, v in res.items() if not k.startswith("V")}, indent=2))
