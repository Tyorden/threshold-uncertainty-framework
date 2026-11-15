#!/bin/zsh
# usage: run_capped.sh SECONDS LOGFILE command [args...]
# Runs a command under nice -n 15 with a hard wall-clock cap (perl alarm),
# at most 4 BLAS/OpenMP threads, logging stdout+stderr to LOGFILE.
secs=$1; log=$2; shift 2
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 NUMEXPR_NUM_THREADS=4
echo "[run_capped] start $(date '+%Y-%m-%d %H:%M:%S %Z') cap=${secs}s cmd=$*" > "$log"
nice -n 15 perl -e 'alarm shift; exec @ARGV' "$secs" "$@" >> "$log" 2>&1
rc=$?
echo "[run_capped] end $(date '+%Y-%m-%d %H:%M:%S %Z') rc=$rc" >> "$log"
exit $rc
