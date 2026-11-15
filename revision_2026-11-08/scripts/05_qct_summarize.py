#!/usr/bin/env python
"""05_qct_summarize.py: metrics from the canonical QCT pass (03), with Monte
Carlo standard errors.

Reference list for the posterior misclassification probability phi: HUD's
published 2026 list (the list the public sees). The 221 tracts where the
replication (with HUD's rounding) differs from HUD's list are reported
separately; everything else is computed on the 85,169 tracts where the
replication's baseline equals HUD's list, and nationally.

MC errors: per-tract binomial SE sqrt(phi(1-phi)/R); for aggregates of the
'realistic' run, a bootstrap over the R draws (B = 200 multinomial
reweightings of the stored per-draw bit matrices); for other runs an
analytic approximation (Var(count of phi_hat >= c) ~ sum_i P_i(1-P_i),
P_i = Phi((phi_i - c)/se_i)), checked against the bootstrap on 'realistic'.
"""
import os, sys, json
import numpy as np, pandas as pd
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__)); W = os.path.dirname(HERE)
IN = os.path.join(W, "results", "qct_canonical")
meta = json.load(open(os.path.join(IN, "run_meta.json")))
R = meta["R"]
b = pd.read_parquet(os.path.join(IN, "per_tract_counts.parquet"))
hud = b["qct_hud"].to_numpy() == 1
rep = b["des_replication"].to_numpy() == 1
rep_el = b["elig_replication"].to_numpy() == 1
agree = hud == rep
P1 = b["P1"].to_numpy(float)
unit = b["cap_unit"].to_numpy()
val = json.load(open(os.path.join(W, "results", "qct_validation.json")))
bound_units = set(val["cap_unit_labels"] if "cap_unit_labels" in val else val["cap_bound_unit_labels"])
in_bound = np.isin(unit, list(bound_units))
pilot = np.isin(unit, ["M35620", "M31080", "M32820", "M26420", "M12420"])
SEED_BOOT = 20261108


def metrics(p_des, p_el, p_eld, mask=None, ref=None):
    ref_ = hud if ref is None else ref
    m = np.ones(len(p_des), bool) if mask is None else mask
    D = ref_ & m; U = ~ref_ & m
    phi = np.where(ref_, 1 - p_des, p_des)
    loss_el = (1 - p_el)[D].sum(); loss_rank = p_eld[D].sum()
    gain = p_des[U].sum()
    gain_rank = p_des[U & rep_el].sum()        # gains by tracts eligible at baseline (cut by the ranking)
    rank_flip = p_eld                            # P(eligible and not designated)
    with np.errstate(divide="ignore", invalid="ignore"):
        cond = np.where(p_el > 0, p_eld / p_el, np.nan)
    return dict(
        n_tracts=int(m.sum()), designated=int(D.sum()),
        expected_losses=float(loss_el + loss_rank), losses_eligibility_driven=float(loss_el),
        losses_rank_driven=float(loss_rank), expected_gains=float(gain),
        gains_by_baseline_eligible=float(gain_rank), gains_newly_eligible=float(gain - gain_rank),
        expected_two_sided=float(loss_el + loss_rank + gain),
        expected_changes_share_of_list=float((loss_el + loss_rank + gain) / max(D.sum(), 1)),
        phi_ge_0p05=int(((phi >= 0.05 - 1e-9) & m).sum()), phi_ge_0p10=int(((phi >= 0.10 - 1e-9) & m).sum()),
        phi_ge_0p25=int(((phi >= 0.25 - 1e-9) & m).sum()),
        designated_phi_ge_0p10=int(((phi >= 0.10 - 1e-9) & D).sum()),
        share_designated_phi_ge_0p10=float(((phi >= 0.10 - 1e-9) & D).sum() / max(D.sum(), 1)),
        pop_designated_phi_ge_0p10=float(P1[(phi >= 0.10 - 1e-9) & D].sum()),
        old_signature_pelig95_phi05=int(((p_el >= 0.95) & (phi >= 0.05 - 1e-9) & m).sum()),
        designated_rank_flip_ge_0p05=int(((rank_flip >= 0.05 - 1e-9) & D).sum()),
        designated_rank_flip_ge_0p10=int(((rank_flip >= 0.10 - 1e-9) & D).sum()),
        designated_pelig_ge_0p95=int(((p_el >= 0.95) & D).sum()),
        designated_pelig95_rank_flip_ge_0p10=int(((p_el >= 0.95) & (rank_flip >= 0.10 - 1e-9) & D).sum()),
        mean_cond_not_des_given_elig_designated=float(np.nanmean(cond[D])) if D.any() else None,
    )


def run_arrays(k):
    return (b[f"{k}__des"].to_numpy() / R, b[f"{k}__elig"].to_numpy() / R,
            b[f"{k}__elig_not_des"].to_numpy() / R)


def approx_count_se(phi, c, mask):
    se = np.sqrt(np.clip(phi * (1 - phi), 1e-12, None) / R)
    P = norm.cdf((phi - c) / se)
    P = np.where(phi * (1 - phi) == 0, (phi >= c).astype(float), P)
    return float(np.sqrt((P * (1 - P))[mask].sum()))


out = {"R": R, "seed": meta["seed"], "reference": "HUD published 2026 list",
       "agreement_set_n": int(agree.sum()), "residual_n": int((~agree).sum()),
       "cap_bound_units": len(bound_units), "tracts_in_cap_bound_units": int(in_bound.sum()), "runs": {}}
scopes = {"national": None, "agreement_set": agree, "cap_bound_units": in_bound, "slack_units": ~in_bound,
          "five_pilot_metros": pilot, "cap_bound_agreement_set": in_bound & agree}
for k in meta["runs"]:
    pd_, pe_, ped_ = run_arrays(k)
    phi = np.where(hud, 1 - pd_, pd_)
    out["runs"][k] = {s: metrics(pd_, pe_, ped_, m) for s, m in scopes.items()}
    out["runs"][k]["approx_mc_se"] = {f"phi_ge_{c}": approx_count_se(phi, c, np.ones(len(phi), bool)) for c in (0.05, 0.10, 0.25)}
    out["runs"][k]["approx_mc_se"]["designated_phi_ge_0.1"] = approx_count_se(phi, 0.10, hud)

# ---------------- rule variants scored against their OWN baseline lists
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)
import qct_fast as qf
Dq = qf.QCTData(pd.read_parquet(os.path.join(W, "cache", "qct_data_2026.parquet")))
assert (Dq.geoid == b["GEOID"].to_numpy()).all()
own = {}
for v, run in (("hud_2of3", "realistic"), ("latest_only", "latest_only"), ("avg3", "avg3"), ("hud_2of3_nonoverlap", "indep")):
    vv = "hud_2of3" if v.startswith("hud_2of3") else v
    base_v, _ = qf.run(Dq, rounding=True, baseline_hud_columns=False, variant=vv)
    pd_, pe_, ped_ = run_arrays(run)
    own[v] = {s_: metrics(pd_, pe_, ped_, m_, ref=base_v) for s_, m_ in
              (("national", None), ("cap_bound_units", in_bound), ("five_pilot_metros", pilot))}
    own[v]["baseline_designated"] = int(base_v.sum())
    own[v]["baseline_overlap_with_hud"] = int((base_v & hud).sum())
    own[v]["error_run"] = run
out["rule_variants_own_baseline"] = own

# ---------------- bootstrap MC errors for the canonical 'realistic' run
bd = np.unpackbits(np.load(os.path.join(IN, "bits_des_realistic.npy")), axis=1)[:, :len(b)]
be = np.unpackbits(np.load(os.path.join(IN, "bits_elig_realistic.npy")), axis=1)[:, :len(b)]
pd_r, pe_r, ped_r = run_arrays("realistic")
assert np.allclose(bd.mean(axis=0), pd_r) and np.allclose(be.mean(axis=0), pe_r)
var = ((pd_r > 0) & (pd_r < 1)) | ((pe_r > 0) & (pe_r < 1))
idx = np.nonzero(var)[0]
Md = bd[:, idx].astype(np.float32); Me = be[:, idx].astype(np.float32)
Med = (be[:, idx] & (1 - bd[:, idx])).astype(np.float32)
rng = np.random.default_rng(SEED_BOOT)
B = 200
keys = ["expected_losses", "losses_eligibility_driven", "losses_rank_driven", "expected_gains",
        "expected_two_sided", "phi_ge_0p05", "phi_ge_0p10", "phi_ge_0p25", "designated_phi_ge_0p10",
        "old_signature_pelig95_phi05", "designated_rank_flip_ge_0p10", "designated_pelig95_rank_flip_ge_0p10"]
boot = {s: {k: [] for k in keys} for s in ("national", "cap_bound_units", "five_pilot_metros")}
for _ in range(B):
    w = rng.multinomial(R, np.full(R, 1.0 / R)).astype(np.float32)
    pdb, peb, pedb = pd_r.copy(), pe_r.copy(), ped_r.copy()
    pdb[idx] = (w @ Md) / R; peb[idx] = (w @ Me) / R; pedb[idx] = (w @ Med) / R
    for s, m in (("national", None), ("cap_bound_units", in_bound), ("five_pilot_metros", pilot)):
        mm = metrics(pdb, peb, pedb, m)
        for k in keys:
            boot[s][k].append(mm[k])
out["bootstrap_mc_se_realistic"] = {s: {k: float(np.std(v, ddof=1)) for k, v in d.items()} for s, d in boot.items()}
out["bootstrap_B"] = B
out["tracts_with_any_variation_realistic"] = int(len(idx))

# ---------------- per-tract table (realistic) and exhibits
phi_r = np.where(hud, 1 - pd_r, pd_r)
tr = pd.DataFrame({"GEOID": b["GEOID"], "cap_unit": unit, "P1": P1, "hud": hud.astype(int),
                   "replication": rep.astype(int), "p_designated": pd_r, "p_eligible": pe_r,
                   "p_elig_not_des": ped_r, "phi_vs_hud": phi_r,
                   "mc_se_phi": np.sqrt(phi_r * (1 - phi_r) / R)})
with np.errstate(divide="ignore", invalid="ignore"):
    tr["p_not_des_given_elig"] = np.where(pe_r > 0, ped_r / pe_r, np.nan)
tr.to_parquet(os.path.join(IN, "per_tract_realistic.parquet"))
ex = tr[tr["GEOID"].isin(["06059074806", "36081047300"])].copy()
out["exhibits_realistic"] = ex.to_dict(orient="records")
# candidate exhibits: on both lists, eligible in every draw, largest rank-driven loss
cand = tr[(tr["hud"] == 1) & (tr["replication"] == 1) & (tr["p_eligible"] == 1.0)].sort_values("p_elig_not_des", ascending=False)
out["top_rank_driven_both_lists"] = cand.head(10).to_dict(orient="records")
# the 221 residual tracts
res = tr[tr["hud"] != tr["replication"]]
out["residual_221"] = dict(n=int(len(res)), mean_phi_vs_hud=float(res["phi_vs_hud"].mean()),
                           mean_p_designated=float(res["p_designated"].mean()),
                           hud_only=int(((res["hud"] == 1)).sum()), replication_only=int((res["replication"] == 1).sum()))
json.dump(out, open(os.path.join(W, "results", "qct_canonical_summary.json"), "w"), indent=2, default=float)
r = out["runs"]
for k in meta["runs"]:
    print(k, {s: {kk: round(v, 1) if isinstance(v, float) else v for kk, v in r[k][s].items() if kk in
                  ("designated", "expected_losses", "losses_rank_driven", "expected_gains", "expected_two_sided",
                   "phi_ge_0p10", "designated_phi_ge_0p10", "old_signature_pelig95_phi05", "designated_rank_flip_ge_0p10")}
              for s in ("national", "five_pilot_metros")})
print(json.dumps(out["bootstrap_mc_se_realistic"], indent=1))
print(json.dumps({k: out["runs"][k]["approx_mc_se"] for k in ("realistic", "indep")}, indent=1))
print(json.dumps(out["exhibits_realistic"], indent=1, default=float))
print(json.dumps(out["top_rank_driven_both_lists"][:5], indent=1, default=float))
print(out["residual_221"])
for v, d in own.items():
    print(v, d["baseline_designated"], d["baseline_overlap_with_hud"], {s_: {k: (round(x, 1) if isinstance(x, float) else x) for k, x in d[s_].items() if k in ("designated", "expected_losses", "expected_gains", "expected_two_sided", "expected_changes_share_of_list", "designated_phi_ge_0p10", "share_designated_phi_ge_0p10")} for s_ in ("national", "five_pilot_metros")})
