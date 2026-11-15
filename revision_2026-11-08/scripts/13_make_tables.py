#!/usr/bin/env python
"""13_make_tables.py: write LaTeX tables for the text blocks directly from the
results files (no hand transcription). Output: text/tables_generated.tex"""
import json, os
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
k = {a: b["value"] for a, b in json.load(open(os.path.join(W, "results", "key_numbers.json"))).items()}
def f0(x): return f"{x:,.0f}".replace(",", "{,}")
def f1(x): return f"{x:,.1f}".replace(",", "{,}")
L = []
L.append("% tables_generated.tex: written by scripts/13_make_tables.py from results/key_numbers.json. Do not edit by hand.")
# ---- QCT canonical results
se = k["qct_boot_se_realistic"]
L += [r"\begin{table}[t]", r"\centering",
      r"\caption{Qualified Census Tracts: flat-prior posterior misclassification relative to HUD's published 2026 list, realistic correlated error model, all 85,390 tracts, 2,000 draws. Losses are expected false designations (listed tracts that the rule applied to true inputs would not list), split into those that become ineligible and those that remain eligible but fall below the rank cut; gains are expected missed designations. Monte Carlo standard errors from a bootstrap over draws in parentheses.}",
      r"\label{tab:qct}", r"\footnotesize", r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
      r"Scope & Listed & \multicolumn{3}{c}{Expected losses} & Expected & Listed tracts & Share of \\",
      r" & tracts & All & Ineligible & Rank cut & gains & with $\hat\phi\ge0.10$ & list \\", r"\midrule"]
for lab, sc, sek in (("All units", "national", "national"), ("Cap-bound units (104)", "cap_bound_units", "cap_bound_units"), ("Slack units", "slack_units", None)):
    d = k[f"qct_realistic_{sc}"]
    s = se.get(sek) if sek else None
    ses = (lambda key: f" ({s[key]:.1f})" if s else "")
    L.append(f"{lab} & {f0(d['designated'])} & {f1(d['expected_losses'])}{ses('expected_losses')} & {f1(d['losses_eligibility_driven'])} & {f1(d['losses_rank_driven'])} & {f1(d['expected_gains'])}{ses('expected_gains')} & {f0(d['designated_phi_ge_0p10'])}{ses('designated_phi_ge_0p10')} & {100*d['designated_phi_ge_0p10']/d['designated']:.1f}\\% \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
# ---- QCT scenarios and rules
L += [r"\begin{table}[t]", r"\centering",
      r"\caption{Qualified Census Tracts: sensitivity of national results to the error model and comparison of rules, 2,000 paired draws (all runs share the same standard normals). Two-sided = expected losses plus expected gains. Rows marked $\dagger$ are scored against each rule's own list at the published estimates; all other rows against HUD's list.}",
      r"\label{tab:qctsens}", r"\footnotesize", r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
      r"Run & Two-sided & Rank-cut losses & Listed tracts with $\hat\phi\ge0.10$ \\", r"\midrule"]
rows = [("Independent errors", "indep"), ("Cross-vintage correlation only", "xvintage_only"),
        ("Realistic (canonical)", "realistic"), ("Near-comonotone vintages (stress)", "xvintage_high"),
        ("Realistic, income-poverty correlation 0.3", "realistic_ip_lo"), ("Realistic, income-poverty correlation 0.6", "realistic_ip_hi"),
        ("Realistic, log-normal median income", "realistic_lognormal_inc"), ("Realistic, SE $\\times$ 0.75", "realistic_se075"),
        ("Realistic, SE $\\times$ 1.25", "realistic_se125"),
        ("Channel: eligibility realistic, rank scores independent", "elig_real_rank_indep"),
        ("Channel: eligibility independent, rank scores realistic", "elig_indep_rank_real")]
for lab, run in rows:
    d = k[f"qct_{run}_national"]
    L.append(f"{lab} & {f1(d['expected_two_sided'])} & {f1(d['losses_rank_driven'])} & {f0(d['designated_phi_ge_0p10'])} \\\\")
L.append(r"\midrule")
rv = k["qct_rule_variants"]
for lab, key in (("2-of-3 rule in force$^\\dagger$ (", "hud_2of3"), ("Latest vintage alone$^\\dagger$ (", "latest_only"),
                 ("Three-vintage average$^\\dagger$ (", "avg3"), ("2-of-3, non-overlapping vintage errors$^\\dagger$ (", "hud_2of3_nonoverlap")):
    d = rv[key]
    L.append(f"{lab}{f0(d['baseline_designated'])} listed) & {f1(d['expected_two_sided'])} & & {f0(d['designated_phi_ge_0p10'])} \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
# ---- common metrics across programs
cm = k["common_metrics_inputs"]; q = k["qct_realistic_national"]
pp15 = k["pp_live_DF1.5_rho0"]; ppe = k["pp_eb_rules"]["conjunction"]["eb"]; ppes = k["pp_eb_summary"]["eb"]
L += [r"\begin{table}[t]", r"\centering",
      r"\caption{Common metrics across programs. Listed units are those on the published list (or eligible at the point estimates, for Title I). Expected changes are the posterior expected number of misclassified units (false plus missed designations) as a share of the list. Universes differ: national for QCT, persistent poverty, Title I and HUBZone; pilot sets for MUA (single-county areas) and CRA (five jurisdictions).}",
      r"\label{tab:common}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}p{2.6cm}p{2.1cm}rp{2.0cm}p{1.5cm}p{3.0cm}@{}}", r"\toprule",
      r"Program & Universe & Listed & Listed with $\hat\phi\ge0.10$ & Expected changes & Population in fragile listed units \\", r"\midrule"]
L.append(f"QCT & 85{{,}}390 tracts & {f0(q['designated'])} & {f0(q['designated_phi_ge_0p10'])} ({100*q['designated_phi_ge_0p10']/q['designated']:.1f}\\%) & {100*q['expected_two_sided']/q['designated']:.1f}\\% & {q['pop_designated_phi_ge_0p10']/1e6:.1f} million residents \\\\")
L.append(f"Persistent poverty, flat & 3{{,}}109 counties & {pp15['designated']} & {pp15['designated_phi_ge_0p10']} ({100*pp15['share_designated_phi_ge_0p10']:.1f}\\%) & {100*pp15['expected_changes_share_of_list']:.1f}\\% & {pp15['pop_in_fragile_designated']/1e6:.2f} million persons \\\\")
L.append(f"Persistent poverty, EB & 3{{,}}109 counties & {ppe['designated']} & {ppes['designated_phi_ge_0p10']} ({100*ppes['designated_phi_ge_0p10']/ppe['designated']:.1f}\\%) & {100*ppe['expected_two_sided']/ppe['designated']:.1f}\\% & not computed \\\\")
t1 = cm["title1"]
L.append(f"Title I concentration & {f0(k['t1_n'])} districts & {f0(t1['listed'])} & {f0(t1['listed_phi_ge_0p10'])} ({100*t1['listed_phi_ge_0p10']/t1['listed']:.1f}\\%) & {100*t1['expected_changes']/t1['listed']:.1f}\\% & {t1['pop517_listed_phi_ge_0p10']/1e6:.2f} million aged 5 to 17 \\\\")
mu = cm["mua"]
L.append(f"MUA & 631 pilot areas & {mu['listed']} & {mu['listed_phi_ge_0p10']} ({100*mu['listed_phi_ge_0p10']/mu['listed']:.1f}\\%) & {100*mu['expected_changes']/mu['listed']:.1f}\\% & {mu['designation_population_fragile']/1e6:.2f} million (designation population) \\\\")
cr = cm["cra_lmi_line"]
L.append(f"CRA, LMI line & 2{{,}}264 pilot tracts & {cr['listed_lmi']} & {cr['listed_phi_ge_0p10']} ({100*cr['listed_phi_ge_0p10']/cr['listed_lmi']:.1f}\\%) & {100*cr['expected_changes']/cr['listed_lmi']:.1f}\\% & not in pilot file \\\\")
hz = cm["hubzone_unemployment"]
L.append(f"HUBZone, unemployment & {f0(hz['nonmetro'])} counties & {hz['listed']} & {hz['listed_phi_ge_0p10']} ({100*hz['listed_phi_ge_0p10']/hz['listed']:.1f}\\%) & {100*hz['expected_changes']/hz['listed']:.1f}\\% & not computed \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
open(os.path.join(W, "text", "tables_generated.tex"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
