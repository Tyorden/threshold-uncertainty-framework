#!/bin/zsh
cd "$(dirname "$0")/rerun"
PY=${PY:-python}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 NUMEXPR_NUM_THREADS=4
for s in 01_title1_full_rule 02b_qct_validate 02c_qct_ranking_conventions 10_mua_anchor_designation 04_pp_conjunctive 03_qct_canonical_mc 05_qct_summarize 06_qct_decomposition 07_qct_eb 09_qct_known_truth 11_small_checks 12_key_numbers; do
  echo "START $s $(date '+%H:%M:%S')" >> logs_rerun/driver.log
  nice -n 15 $PY scripts/$s.py > logs_rerun/$s.log 2>&1
  echo "END $s rc=$? $(date '+%H:%M:%S')" >> logs_rerun/driver.log
done
echo ALLDONE >> logs_rerun/driver.log
