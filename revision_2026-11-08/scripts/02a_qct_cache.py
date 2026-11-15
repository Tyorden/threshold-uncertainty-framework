#!/usr/bin/env python
"""02a_qct_cache.py: load HUD's 2026 and 2025 QCT files once (read only, via the
QCT folder's own loader) and cache them as parquet in this folder's cache/."""
import sys, os, time
sys.dont_write_bytecode = True
import os as _os  # public copy: the program folders (see README) are found through PAPER_PREP
PAPER_PREP = _os.environ.get("PAPER_PREP", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "program_folders"))
sys.path.insert(0, PAPER_PREP + "/QCT_lihtc/scripts")
import qct_pipeline as qp
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for yr in ("2026", "2025"):
    t = time.time()
    df = qp.load_hud(PAPER_PREP + f"/QCT_lihtc/data/qct_data_{yr}.xlsx")
    df.to_parquet(os.path.join(W, "cache", f"qct_data_{yr}.parquet"))
    print(yr, df.shape, f"{time.time()-t:.1f}s")
    print(list(df.columns))
