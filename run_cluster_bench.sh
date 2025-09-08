#!/usr/bin/env bash
# run_cluster_bench.sh — single-run MPI launch, rank0=master on PI5, 1 worker per node

set -Eeuo pipefail

HOSTFILE="${HOSTFILE:-$HOME/mpi_hosts}"
PY="${PYTHON:-$HOME/envMPI/bin/python3}"
SCRIPT="${SCRIPT:-$HOME/mpi_cluster_bench.py}"

OUTDIR="${OUTDIR:-$HOME/cluster_bench_out}"
SIZES="${SIZES:-2048,3072,4096}"
REPS="${REPS:-2}"
DTYPE="${DTYPE:-float32}"

if [[ ! -f "$HOSTFILE" ]]; then
  echo "[ERR] Hostfile not found: $HOSTFILE" >&2
  exit 2
fi
if [[ ! -x "$PY" ]]; then
  echo "[ERR] Python not executable: $PY" >&2
  exit 2
fi
mkdir -p "$OUTDIR"

# Count worker nodes (non-empty, non-comment first column)
NP=$(awk 'NF && $1 !~ /^#/{c++} END{print c+0}' "$HOSTFILE")
if [[ "$NP" -le 0 ]]; then
  echo "[ERR] No worker nodes found in $HOSTFILE" >&2
  exit 2
fi

echo "[INFO] hostfile: $HOSTFILE"
echo "[INFO] python  : $PY"
echo "[INFO] script  : $SCRIPT"
echo "[INFO] np      : $NP"
echo "[INFO] outdir  : $OUTDIR"

mpirun --hostfile "$HOSTFILE" -np "$NP" \
  --map-by ppr:1:node \
  -x OMP_NUM_THREADS -x OPENBLAS_NUM_THREADS -x MKL_NUM_THREADS \
  "$PY" "$SCRIPT" \
    --sizes "$SIZES" \
    --reps "$REPS" \
    --dtype "$DTYPE" \
    --outdir "$OUTDIR"
