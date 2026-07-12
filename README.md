# threshold-uncertainty-framework

Aggregation code, the cross-program summary table, and the complete number
trace for the paper:

> Tyler Orden, "Propagating survey and model error through administrative
> bright lines: a framework and multi-program audit."

The paper develops a common framework for what happens when a statute or
regulation draws a bright-line threshold on a published statistical
estimate, and audits it across seven federal designation systems. This
repository does not contain the per-program pipelines; it aggregates their
results. Each application program has its own repository with its full
replication code, data manifest, and results:

| Program | Paper | Repository |
|---|---|---|
| LIHTC Qualified Census Tracts (HUD) | "Another Tract's Noise" | https://github.com/Tyorden/qct-designation-uncertainty |
| Medically Underserved Areas (HRSA) | "How stable are Medically Underserved Area designations?" | https://github.com/Tyorden/imu-designation-uncertainty |
| HUBZone qualified nonmetro counties, unemployment prong (SBA) | "Federal Contracting Eligibility on an Estimate With No Published Standard Error" | https://github.com/Tyorden/hubzone-laus-uncertainty |
| Persistent-poverty counties (USDA ERS) | "Clearing the Bar Every Decade" | https://github.com/Tyorden/persistent-poverty-uncertainty |

Three further case studies in the paper (CRA/FFIEC tract income classes,
Title I-A school-district cliffs on SAIPE, and CDC Chronic Disease
Indicator caution flags) come from source folders that are not yet public;
their summary-table rows are asserted and committed here in the same way,
and their inputs are public federal data described in the paper.

## Contents

- `scripts/build_summary_table.py` builds the paper's Table 1. Every cell
  in `results/cross_program_summary.csv` is read or computed from a results
  file in one of the seven per-program source folders, and every number is
  asserted against the value quoted in the paper before the CSV is written.
- `results/cross_program_summary.csv` the committed, canonical Table 1
  (7 program rows).
- `NUMBER_TRACE.md` a complete trace of every number in the paper to the
  results file it comes from.

## Reproduction

Environment: Python 3.13, standard library only (csv, json, re, pathlib).

The script expects the per-program source folders as siblings of this
repository's root directory, under their original folder names. To
regenerate the four application rows, clone the application repositories as
siblings with those names:

```
git clone https://github.com/Tyorden/threshold-uncertainty-framework
git clone https://github.com/Tyorden/qct-designation-uncertainty QCT_lihtc
git clone https://github.com/Tyorden/imu-designation-uncertainty IMU_underserved
git clone https://github.com/Tyorden/hubzone-laus-uncertainty A3_hubzone_laus
git clone https://github.com/Tyorden/persistent-poverty-uncertainty A4_persistent_poverty
cd threshold-uncertainty-framework
python scripts/build_summary_table.py
```

The script writes `results/cross_program_summary.csv` and passes only if
every assertion holds. The three not-yet-public folders (`A1_ffiec_cra`,
`TITLE1_saipe`, `B2_cdi_flags`) are also required for a full rebuild; until
they are public, the committed CSV is the canonical output and the script
documents, row by row, exactly which results files and values each cell
comes from.

## Where each paper number lands

Table 1 is `results/cross_program_summary.csv`. Every other number in the
paper is listed in `NUMBER_TRACE.md` with the results file it traces to
(paths are given relative to the sibling-folder layout above).

## License

MIT (see `LICENSE`). Author: Tyler Orden.
