#!/usr/bin/env python
"""number_audit_narrow.py (2026-11-14): every number in draft.tex and supplement.tex must trace
to a results file. Same method as ../revision_2026-11-08/scripts/14_number_audit.py: collect all
numeric values from the revision's results files (plus figures/figure_values.json and the two
MUA files of the program folder), expand to the forms the text uses, and list every number in the
two .tex files (outside comments, cites, labels) that matches none. Output: number_audit_narrow.txt"""
import glob, json, os, re
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); W = os.path.join(HERE, "..", "revision_2026-11-08")
vals = set()
def add(x):
    try: x = float(x)
    except Exception: return
    for v in (x, 100 * x, x / 1e6):
        for d in (0, 1, 2, 3, 4): vals.add(round(v, d))
def walk(o):
    if isinstance(o, dict):
        for v in o.values(): walk(v)
    elif isinstance(o, list):
        for v in o: walk(v)
    elif isinstance(o, (int, float)) and not isinstance(o, bool): add(o)
    elif isinstance(o, str):
        for m in re.findall(r"-?\d[\d,]*\.?\d*", o): add(m.replace(",", ""))
for f in glob.glob(os.path.join(W, "results", "**", "*.json"), recursive=True): walk(json.load(open(f)))
walk(json.load(open(os.path.join(HERE, "figures", "figure_values.json"))))
walk(json.load(open(os.path.join(HERE, "..", "analysis_2026-11-14", "results", "qct_shared_component.json"))))
walk(json.load(open(os.path.join(HERE, "..", "analysis_2026-11-14", "results", "qct_screen_fixed.json"))))  # screen-convention run
_ex = pd.read_parquet(os.path.join(W, "results", "qct_eb", "per_tract_flat_vs_eb.parquet"))  # the exhibit tract only
for v in _ex.loc[_ex.GEOID == "06059074806", ["phi_flat", "phi_eb"]].to_numpy().ravel(): add(v)
for f in ("qct_canonical_table.csv", "qct_residual_by_unit.csv", "title1_exhibits.csv"):
    for v in pd.read_csv(os.path.join(W, "results", f)).select_dtypes("number").to_numpy().ravel(): add(v)
import os as _os  # public copy: the program folders (see README) are found through PAPER_PREP
PAPER_PREP = _os.environ.get("PAPER_PREP", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "program_folders"))
IMU = PAPER_PREP + "/IMU_underserved/results/"
walk(json.load(open(IMU + "mc_report_v2.json"))); walk(json.load(open(IMU + "context_counts.json")))
# constants of the rules, years, identifiers and stated design values (not results)
constants = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 16, 17, 20, 25, 30, 40, 50, 60, 62, 80, 90, 100, 120, 140, 200, 250, 300, 500,
             1000, 2000, 6500, 20000, 0.5, 0.25, 0.1, 0.9, 0.4, 0.6, 0.02, 0.15, 0.8, 0.3, 1.645, 1.0, 1.5, 2.0, 1.1, 1.4, 62.0,
             1.20, 1.60, 0.1215, 730, 1978, 2000, 1990, 1989, 1999, 2007, 2011, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2025, 2026,
             203, 34, 200.71, 3.13, 2.5, 1.18, 3.0, 2.28, 0.63, 0.828, 0.900, 0.379, 0.892, 12.5, 104, 221, 85390, 41, 22, 23}
for c in constants: vals.add(round(float(c), 4))
PROGRAM_SOURCES = {
    84783: "A1_ffiec_cra/results/correctness_anchor_match.csv (exact anchor)",
    84713: "A1_ffiec_cra/results/correctness_anchor_match.csv ($100-rounding row)",
    3234: "A3_hubzone_laus/results/anchor_level1.txt", 473: "A3_hubzone_laus/results/anchor_summary.md",
    318: "A4_persistent_poverty/results/anchor_report.txt (official PerPov1721c count)",
    31000: "rounded from results/small_checks.json qct_hud_rounding_check (31,142 to 32,003)",
    32000: "rounded from results/small_checks.json qct_hud_rounding_check (31,142 to 32,003)",
    256: "SHA-256 (name of a hash function, not a result)", 64: "PCG64 (name of a generator)",
}
out, resolved = [], []
for f in ("draft.tex", "supplement.tex"):
    for i, ln in enumerate(open(os.path.join(HERE, f)).read().split("\n"), 1):
        code = re.split(r"(?<!\\)%", ln)[0]
        code = re.sub(r"\\(cite[pt]?|ref|label|input|includegraphics(\[[^\]]*\])?|url|usepackage(\[[^\]]*\])?|documentclass(\[[^\]]*\])?)\{[^}]*\}", "", code)
        code = re.sub(r"p\{[\d.]+cm\}|\[\d+pt\]|PCG64|ORCID [\d-]+", "", code)  # ORCID: an identifier
        for m in re.finditer(r"(?<![\w.])(\d{1,3}(?:(?:\{,\}|,)\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?![\w])", code):
            tok = m.group(1).replace("{,}", "").replace(",", ""); v = float(tok)
            if any(round(v, d) in vals for d in (4, 3, 2, 1, 0)): continue
            if int(v) == v and int(v) in PROGRAM_SOURCES:
                resolved.append(f"{f}:{i}: {tok} -> {PROGRAM_SOURCES[int(v)]}"); continue
            out.append(f"{f}:{i}: {tok}   | {code.strip()[:110]}")
txt = ["# number_audit_narrow.txt: numbers in draft.tex and supplement.tex checked against results files", f"# untraced numbers: {len(out)}"] + out + \
      [f"# traced to program-folder results or identifiers: {len(resolved)}"] + resolved
open(os.path.join(HERE, "number_audit_narrow.txt"), "w").write("\n".join(txt) + "\n")
print(len(out), "untraced;", len(resolved), "traced to program folders or identifiers"); print("\n".join(out))
