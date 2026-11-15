# Survey error and administrative thresholds: code and results

Analysis scripts and results files for a study of what happens when a federal rule applies an exact
threshold to a survey estimate. The study covers two designation systems built on census and
American Community Survey estimates:

- **Qualified Census Tracts** (HUD, 2026 list): 85,390 tracts, a 20 percent population cap, a
  two-of-three-vintage test and a reliability screen.
- **Persistent-poverty counties** (USDA Economic Research Service): poverty of 20 percent or more in
  each of four periods (1990, 2000, 2007 to 2011, 2017 to 2021).

Each official list is first rebuilt from published inputs. Published sampling error is then
propagated through the rule as the agency applies it. The reported quantity is a posterior
misclassification probability for each unit, under a flat prior and under an empirical-Bayes prior,
checked against simulated truth. Four further programs (Title I concentration grants, Medically
Underserved Areas, Community Reinvestment Act tract income levels, HUBZone counties) are treated in
brief in an online supplement; their numbers are here too.

This repository was narrowed in November 2026 from an earlier seven-program version (July 2026).
The July files are kept in `july_2026/`.

## Main results (all in `revision_2026-11-08/results/` and `analysis_2026-11-14/results/`)

| Result | Value | File |
|---|---|---|
| Tract designations reproduced | 85,169 of 85,390 | `qct_validation.json` |
| Listed tracts with misclassification probability of 0.10 or more, flat prior, reliability screen redrawn in each draw | 7,982 of 14,496 (55.1 percent) | `qct_canonical_summary.json` |
| The same with the poverty screen held at its published status | 6,769 of 14,496 (46.7 percent) | `analysis_2026-11-14/results/qct_screen_fixed.json` |
| Expected false plus missed designations, flat prior | 6,224.3 (4,911.9 with the poverty screen held fixed) | the same two files |
| Persistent-poverty counties: rule in force, four-period average, latest period alone, flat and empirical-Bayes | see file | `pp_conjunctive.json` |
| Known-truth calibration | see file | `qct_known_truth/known_truth_summary.json` |

`revision_2026-11-08/results/key_numbers.json` collects the reported numbers, each with its source
file.

## Layout

| Path | What it holds |
|---|---|
| `revision_2026-11-08/scripts/` | The analysis. `qct_fast.py` (vectorized HUD rule) and `qct_rate_engine.py` (rate-level engine for comparing priors) are modules; the numbered scripts are run in order |
| `revision_2026-11-08/results/` | Every results file. `input_checksums.txt` has the SHA-256 of every input file; `environment.txt` has the Python package versions |
| `revision_2026-11-08/cache/` | The Census Bureau's 1990 and 2000 county poverty tables, as downloaded |
| `analysis_2026-11-14/` | Two sensitivity runs: cross-tract dependence (15) and the reliability-screen convention (16) |
| `checks_2026-11-14/` | A complete second run of the scripts compared with the first (`rerun_compare_output.txt`) |
| `paper/` | `number_audit_narrow.py` (checks every number in the manuscript against the results files) with its output, and `figures/make_figures.py` with `figure_values.json` |

### Scripts

| Script | What it does |
|---|---|
| `02a_qct_cache.py` | Loads HUD's 2026 and 2025 tract files once and caches them as parquet |
| `02b_qct_validate.py` | Checks that `qct_fast` reproduces the reference pipeline exactly, then applies HUD's rounding convention |
| `02c_qct_ranking_conventions.py` | Tests candidate ranking conventions against the 221 tracts the rebuild does not reproduce |
| `03_qct_canonical_mc.py` | The canonical pass: all 85,390 tracts, 2,000 paired draws, per-draw seeds |
| `05_qct_summarize.py` | Metrics from the canonical pass, with Monte Carlo standard errors |
| `06_qct_decomposition.py` | Own-input and other-tract components of flips at the rank cut |
| `07_qct_eb.py` | Flat prior against empirical Bayes (multivariate Fay-Herriot), 1,000 paired draws |
| `09_qct_known_truth.py` | Known-truth calibration through the full rule |
| `04_pp_conjunctive.py` | Persistent-poverty counties: rebuild of the 1990 and 2000 flags, rounding convention, exact and simulated probabilities, alternative rules |
| `01_title1_full_rule.py`, `10_mua_anchor_designation.py`, `11_small_checks.py` | The supplement's programs and smaller checks |
| `12_key_numbers.py`, `13_make_tables.py` | Collect the reported numbers; write the generated tables |
| `15_qct_shared_component.py`, `16_qct_screen_fixed.py` | The two sensitivity runs |

## Running it

Python 3.13 with NumPy 2.5.1, SciPy 1.18.0 and pandas 3.0.3 (full list in
`revision_2026-11-08/results/environment.txt`). The Monte Carlo engines use NumPy's default
generator (PCG64); seeds are written in each script. The canonical pass seeds each draw separately,
so its results do not depend on how draws are split across processes.

All inputs are public federal files. The scripts read them from the program folders they were
developed in. Set the environment variable `PAPER_PREP` to a folder that holds those folders, or
put them in `program_folders/` at the top of this repository:

| Folder name the scripts expect | Public repository with the same `data/`, `scripts/` and `results/` |
|---|---|
| `QCT_lihtc` | `github.com/Tyorden/qct-designation-uncertainty` |
| `A4_persistent_poverty` | `github.com/Tyorden/persistent-poverty-uncertainty` |
| `IMU_underserved` | `github.com/Tyorden/imu-designation-uncertainty` |
| `A3_hubzone_laus` | `github.com/Tyorden/hubzone-laus-uncertainty` |
| `TITLE1_saipe`, `A1_ffiec_cra` | Not public yet. Only the supplement's Title I and CRA numbers (`01_title1_full_rule.py` and part of `11_small_checks.py`) need them; their outputs are in `results/`, and the CRA rebuild anchor is in `program_results/A1_ffiec_cra/correctness_anchor_match.csv` |

Then run `02a_qct_cache.py` first (it writes the two tract parquet files into
`revision_2026-11-08/cache/`), and the other scripts in the order of
`checks_2026-11-14/rerun_driver.sh`. The canonical pass took about four minutes with
parallel workers on a laptop; `16_qct_screen_fixed.py` took about a minute and a half.

Two things differ from the files as run. Absolute folder paths were replaced by the `PAPER_PREP`
setting, and the one source path inside `key_numbers.json` was made relative. No number changed:
with `PAPER_PREP` set, the number audit and the figure values reproduce byte for byte.

The figures and the number audit run from the results files alone
(`paper/figures/make_figures.py`; `paper/number_audit_narrow.py` also needs the manuscript's two
`.tex` files, which are not in this repository while the paper is under review).

## Checks

- The vectorized tract engine reproduces the reference implementation exactly, at the published
  estimates and on random draws (`02b_qct_validate.py`).
- A complete second run reproduced every results file with relative differences below one part in
  a million (`checks_2026-11-14/rerun_compare_output.txt`). It repeats the same code; it is not an
  independent implementation.
- The exact persistent-poverty computations were checked against simulation.
- There are no formal unit tests.

## Use of AI

Claude (Anthropic), a large language model, developed the analysis software and ran the replication
and Monte Carlo pipelines under the author's direction.

## Author

Tyler Orden, independent researcher. ORCID 0009-0004-9205-7422.
