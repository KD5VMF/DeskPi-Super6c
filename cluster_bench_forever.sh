#!/usr/bin/env bash
# cluster_bench_forever.sh — looped runs with rolling logs and backoff

set -Eeuo pipefail

BACKOFF="${BACKOFF:-5}"           # seconds between runs
LABEL="${LABEL:-live}"            # label for logs
OUTDIR="${OUTDIR:-$HOME/cluster_bench_out}"

mkdir -p "$OUTDIR/logs"
LOGFILE="$OUTDIR/logs/bench_${LABEL}_$(date -u +%Y%m%d_%H%M%SZ).log"
echo "[INFO] Logging to $LOGFILE"
echo "[INFO] Ctrl-C to stop"
trap 'echo; echo "[INFO] Stopping."; exit 0' INT

RUN=0
while true; do
  echo "[RUN] $(date -u +%F' '%T)Z run=$RUN"
  if "$PWD/run_cluster_bench.sh" 2>&1 | tee -a "$LOGFILE"; then
    echo "[OK] Completed run=$RUN" | tee -a "$LOGFILE"
  else
    echo "[WARN] Run failed (run=$RUN). Retrying after $BACKOFF s..." | tee -a "$LOGFILE"
  fi
  ((RUN++)) || true
  sleep "$BACKOFF"
done
