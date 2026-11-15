#!/usr/bin/env python
"""make_figures.py: draw the three figures of the narrowed manuscript (2026-11-14)
directly from the revision results files (no hand-typed values).
Inputs (read only): ../../revision_2026-11-08/results/
Outputs: fig1_fragility_curves.pdf, fig2_rules.pdf, fig3_calibration.pdf,
         figure_values.json (every plotted value quoted in a caption or the text).
Run: python make_figures.py"""
import json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "..", "revision_2026-11-08", "results")
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#d9d8d2"
plt.rcParams.update({"font.size": 8.5, "font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.labelcolor": INK, "text.color": INK, "legend.frameon": False, "pdf.fonttype": 42})
vals = {}
def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.5); ax.set_axisbelow(True)

# ---------- Figure 1: share of the list with phi-hat at or above a band t
q = pd.read_parquet(os.path.join(R, "qct_canonical", "per_tract_realistic.parquet"))
bound = set(json.load(open(os.path.join(R, "qct_validation.json")))["cap_bound_unit_labels"])
p = pd.read_csv(os.path.join(R, "pp_live_rule_per_county.csv"))
t = np.linspace(0.01, 0.6, 119)
def surv(x): return np.array([(x >= u - 1e-12).mean() for u in t])
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.7), sharey=True)
L = q[q.hud == 1]
ax[0].plot(t, 100 * surv(L.phi_vs_hud.to_numpy()), color=BLUE, lw=1.6, label="All listed tracts")
vals["fig1_qct_share_listed_phi_ge_0p10"] = round(100 * float((L.phi_vs_hud >= 0.10 - 1e-9).mean()), 1)
vals["fig1_qct_share_listed_phi_ge_0p25"] = round(100 * float((L.phi_vs_hud >= 0.25 - 1e-9).mean()), 1)
vals["fig1_qct_listed_phi_ge_0p25"] = int((L.phi_vs_hud >= 0.25 - 1e-9).sum())
if bound is not None:
    Lb, Ls = L[L.cap_unit.isin(bound)], L[~L.cap_unit.isin(bound)]
    ax[0].plot(t, 100 * surv(Lb.phi_vs_hud.to_numpy()), color=INK, lw=1.1, ls="--", label="Cap-bound units")
    ax[0].plot(t, 100 * surv(Ls.phi_vs_hud.to_numpy()), color=MUTED, lw=1.1, ls=":", label="Slack units")
    vals["fig1_qct_capbound_listed"] = int(len(Lb)); vals["fig1_qct_slack_listed"] = int(len(Ls))
    vals["fig1_qct_capbound_share_phi_ge_0p10"] = round(100 * float((Lb.phi_vs_hud >= 0.10 - 1e-9).mean()), 1)
    vals["fig1_qct_slack_share_phi_ge_0p10"] = round(100 * float((Ls.phi_vs_hud >= 0.10 - 1e-9).mean()), 1)
    vals["fig1_qct_capbound_share_phi_ge_0p25"] = round(100 * float((Lb.phi_vs_hud >= 0.25 - 1e-9).mean()), 1)
    vals["fig1_qct_slack_share_phi_ge_0p25"] = round(100 * float((Ls.phi_vs_hud >= 0.25 - 1e-9).mean()), 1)
ax[0].set_title("Qualified Census Tracts (flat prior)", fontsize=8.5, loc="left")
D = p[p.PerPov1721c == 1]
ax[1].plot(t, 100 * surv(D["phi_DF1.5"].to_numpy()), color=BLUE, lw=1.6, label="Flat prior")
ax[1].plot(t, 100 * surv(D["phi_eb_DF1.5"].to_numpy()), color=ORANGE, lw=1.6, ls="--", label="Empirical Bayes")
ax[1].set_title("Persistent-poverty counties", fontsize=8.5, loc="left")
vals["fig1_pp_listed"] = int(len(D))
for nm, col in (("flat", "phi_DF1.5"), ("eb", "phi_eb_DF1.5")):
    for b in (0.10, 0.25):
        vals[f"fig1_pp_{nm}_listed_phi_ge_{str(b).replace('.', 'p')}"] = int((D[col] >= b - 1e-9).sum())
for a in ax:
    a.axvline(0.10, color=MUTED, lw=0.6); grid(a); a.set_xlim(0, 0.6); a.set_ylim(0, 100)
    a.set_xlabel(r"Band $t$ on the misclassification probability $\hat\phi$"); a.legend(loc="upper right", fontsize=7.5)
ax[0].set_ylabel(r"Listed units with $\hat\phi \geq t$ (percent)")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig1_fragility_curves.pdf")); plt.close(fig)

# ---------- Figure 2: rules compared
k = {a: b["value"] for a, b in json.load(open(os.path.join(R, "key_numbers.json"))).items()}
cs = json.load(open(os.path.join(R, "qct_canonical_summary.json")))["rule_variants_own_baseline"]
pp = json.load(open(os.path.join(R, "pp_conjunctive.json")))
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.9), gridspec_kw={"width_ratios": [1, 1.15]})
rules = [("hud_2of3", "2-of-3,\noverlapping\n(in force)"), ("latest_only", "Latest\nvintage\nalone"),
         ("avg3", "Three-vintage\naverage"), ("hud_2of3_nonoverlap", "2-of-3,\nnon-\noverlapping")]
x = np.arange(len(rules)); w = 0.36
fl = [100 * cs[r]["national"]["expected_losses"] / cs[r]["baseline_designated"] for r, _ in rules]
ms = [100 * cs[r]["national"]["expected_gains"] / cs[r]["baseline_designated"] for r, _ in rules]
b1 = ax[0].bar(x - w / 2 - 0.01, fl, w, color=BLUE, label="False designations")
b2 = ax[0].bar(x + w / 2 + 0.01, ms, w, color=ORANGE, label="Missed designations")
for (r, _), a_, b_ in zip(rules, fl, ms):
    vals[f"fig2_qct_{r}_false_pct_of_list"] = round(a_, 1); vals[f"fig2_qct_{r}_missed_pct_of_list"] = round(b_, 1)
for bars in (b1, b2):
    for rect in bars:
        ax[0].text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.5, f"{rect.get_height():.0f}", ha="center", va="bottom", fontsize=7, color=MUTED)
ax[0].set_xticks(x); ax[0].set_xticklabels([l for _, l in rules], fontsize=7); ax[0].set_ylim(0, 42)
ax[0].set_ylabel("Expected count (percent of own list)"); ax[0].set_title("Qualified Census Tracts (flat prior)", fontsize=8.5, loc="left")
ax[0].legend(loc="upper right", fontsize=7.5); grid(ax[0])
rb = pp["empirical_bayes"]["rules_under_both_priors"]; lr = pp["live_rule"]["DF1.5_rho0"]
prl = [("conjunction", "All four\nperiods\n(in force)"), ("single_2017_21", "2017-21\nalone"), ("four_period_mean", "Four-period\naverage")]
flat = {"conjunction": (lr["expected_losses"], lr["expected_gains"]),
        "single_2017_21": (lr["single_expected_losses"], lr["single_expected_gains"]),
        "four_period_mean": (lr["average_rule"]["expected_losses"], lr["average_rule"]["expected_gains"])}
x = np.arange(len(prl)); w = 0.19
sers = [("False, flat prior", BLUE, None, [flat[r][0] for r, _ in prl]),
        ("False, empirical Bayes", BLUE, "///", [rb[r]["eb"]["expected_false_designations"] for r, _ in prl]),
        ("Missed, flat prior", ORANGE, None, [flat[r][1] for r, _ in prl]),
        ("Missed, empirical Bayes", ORANGE, "///", [rb[r]["eb"]["expected_missed_designations"] for r, _ in prl])]
for j, (lab, col, hatch, v) in enumerate(sers):
    off = (j - 1.5) * (w + 0.015)
    bars = ax[1].bar(x + off, v, w, color=col if hatch is None else "white", edgecolor=col, hatch=hatch, linewidth=0.8, label=lab)
    for rect, (r, _) in zip(bars, prl):
        ax[1].text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 1.2, f"{rect.get_height():.0f}", ha="center", va="bottom", fontsize=6.5, color=MUTED)
        vals["fig2_pp_" + r + "_" + lab.lower().replace(", ", "_").replace(" ", "_")] = round(float(rect.get_height()), 1)
ax[1].set_xticks(x); ax[1].set_xticklabels([l for _, l in prl], fontsize=7); ax[1].set_ylim(0, 135)
ax[1].set_ylabel("Expected number of counties"); ax[1].set_title("Persistent-poverty counties", fontsize=8.5, loc="left")
ax[1].legend(loc="upper right", fontsize=6.8, ncol=2, columnspacing=0.8, handlelength=1.4); grid(ax[1])
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig2_rules.pdf")); plt.close(fig)

# ---------- Figure 3: known-truth calibration (binned mean probability against observed rate)
kt = json.load(open(os.path.join(R, "qct_known_truth", "known_truth_summary.json")))
pk = pp["known_truth"]
panels = [("QCT, truth design T1", kt["T1"]["national"]["flat_score"]["calibration"], kt["T1"]["national"]["eb_score"]["calibration"]),
          ("QCT, truth design T2", kt["T2"]["national"]["flat_score"]["calibration"], kt["T2"]["national"]["eb_score"]["calibration"]),
          ("Persistent poverty, T1", pk["T1_truth_from_EB_posterior"]["flat"]["calibration"], pk["T1_truth_from_EB_posterior"]["eb"]["calibration"]),
          ("Persistent poverty, T2", pk["T2_truth_equals_published"]["flat"]["calibration"], pk["T2_truth_equals_published"]["eb"]["calibration"])]
fig, axs = plt.subplots(1, 4, figsize=(6.6, 2.15), sharex=True, sharey=True)
binn, totn = {}, {}
for a, (ttl, cf, ce) in zip(axs, panels):
    a.plot([0, 0.75], [0, 0.75], color=MUTED, lw=0.6)
    for cc, col in ((cf, BLUE), (ce, ORANGE)):  # 95 percent binomial interval per bin
        xs = np.array([c["mean_prob"] for c in cc]); ys = np.array([c["observed_rate"] for c in cc]); ns = np.array([c["n"] for c in cc])
        a.errorbar(xs, ys, yerr=1.96 * np.sqrt(ys * (1 - ys) / ns), fmt="none", ecolor=col, elinewidth=0.8, capsize=1.5, zorder=3)
        binn.setdefault(ttl.split(",")[0], []).append([int(v) for v in ns]); totn.setdefault(ttl.split(",")[0], []).append(int(ns.sum()))
    a.plot([c["mean_prob"] for c in cf], [c["observed_rate"] for c in cf], color=BLUE, lw=1.3, marker="o", ms=3.2, label="Flat prior")
    a.plot([c["mean_prob"] for c in ce], [c["observed_rate"] for c in ce], color=ORANGE, lw=1.3, ls="--", marker="s", ms=3.2, label="Empirical Bayes")
    a.set_title(ttl, fontsize=7.5, loc="left"); a.set_xlim(0, 0.75); a.set_ylim(0, 0.75); a.set_aspect("equal")
    a.set_xticks([0, 0.25, 0.5, 0.75]); a.set_yticks([0, 0.25, 0.5, 0.75]); a.tick_params(labelsize=7)
    a.grid(color=GRID, linewidth=0.5); a.set_axisbelow(True)
    key = ttl.replace(",", "").replace(" ", "_").lower()
    vals["fig3_" + key + "_top_bin_flat"] = [round(cf[-1]["mean_prob"], 2), round(cf[-1]["observed_rate"], 2)]
    vals["fig3_" + key + "_top_bin_eb"] = [round(ce[-1]["mean_prob"], 2), round(ce[-1]["observed_rate"], 2)]
for prog, rows in binn.items():  # bin sizes quoted in the caption
    key = "qct" if prog.startswith("QCT") else "pp"
    upper = [v for r in rows for v in r[1:]]
    vals[f"fig3_{key}_upper_bins_min_n"] = min(upper); vals[f"fig3_{key}_upper_bins_max_n"] = max(upper)
    vals[f"fig3_{key}_lowest_bin_share_min_pct"] = round(100 * min(r[0] / t_ for r, t_ in zip(rows, totn[prog])), 1)
    vals[f"fig3_{key}_lowest_bin_share_max_pct"] = round(100 * max(r[0] / t_ for r, t_ in zip(rows, totn[prog])), 1)
axs[0].set_ylabel("Observed misclassification rate", fontsize=7.5); axs[0].legend(loc="lower right", fontsize=6, handlelength=2.2, borderaxespad=0.1)
fig.supxlabel(r"Mean reported probability $\hat\phi$ in the bin", fontsize=7.5, y=0.04)
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(os.path.join(HERE, "fig3_calibration.pdf")); plt.close(fig)
json.dump(vals, open(os.path.join(HERE, "figure_values.json"), "w"), indent=1)
print(json.dumps(vals, indent=1))
