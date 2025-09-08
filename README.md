# Cluster Bench Pack

This pack gives you a **live, continuous MPI cluster benchmark** that:
- pins the **master** on your PI5,
- runs **one worker rank per node** from your `~/mpi_hosts` list,
- keeps your CPUs busy with large dense GEMMs (`numpy.dot`) to stress memory + BLAS,
- streams **live results** and writes **CSV + JSON** artifacts every run,
- can loop **forever** with backoff and rolling logs.

It expects:
- `mpi4py` + NumPy installed inside your `~/envMPI` venv on all nodes,
- your worker nodes listed in `~/mpi_hosts` (one host/IP per line; comments `#` OK),
- passwordless SSH already set up (you’ve done this).

## Files

- `mpi_cluster_bench.py` — MPI program (rank 0 = master, others = workers).
- `run_cluster_bench.sh` — single-run launcher (prints live results).
- `cluster_bench_forever.sh` — infinite loop with backoff + rolling logs.
- `mpi_hosts.sample` — example hostfile (workers only, not the master PI5).
- `systemd/cluster-bench.service` — optional *user* systemd unit to run forever.

## Quick Start

```bash
# On PI5 (master)
unzip cluster_bench_pack.zip -d ~/cluster_bench_pack
cd ~/cluster_bench_pack
chmod +x run_cluster_bench.sh cluster_bench_forever.sh
```

**Distribute the Python file (only needed once or after edits):**
```bash
# If you already have a helper like cluster_push_file.sh, use it.
# Otherwise, this quick one-liner will copy to all hosts in ~/mpi_hosts:
while read -r h _; do [[ -n "$h" && "$h" != \#* ]] && scp mpi_cluster_bench.py "$h:~/"; done < ~/mpi_hosts
```

**Single run:**
```bash
OUTDIR=$HOME/cluster_bench_out SIZES=2048,3072,4096 REPS=2 DTYPE=float32 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 $PWD/run_cluster_bench.sh
```

**Run forever (recommended for “live” view):**
```bash
OUTDIR=$HOME/cluster_bench_out SIZES=2048,3072,4096 REPS=2 DTYPE=float32 INTERVAL=5 LABEL=live BACKOFF=5 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 $PWD/cluster_bench_forever.sh
```

Artifacts go to `$OUTDIR`:
- `bench_<UTCSTAMP>_run<RUN>.csv` — row per (size, worker, rep)
- `bench_<UTCSTAMP>_run<RUN>.json` — full run metadata + results
- logs in `$OUTDIR/logs/` when using the forever wrapper

## Optional: systemd (user) service

```bash
mkdir -p ~/.config/systemd/user
cp systemd/cluster-bench.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now cluster-bench.service
journalctl --user -u cluster-bench -f
```

Edit the service file to adjust environment (sizes, reps, etc.) before enabling.

## Notes & tuning

- `SIZES`: comma list of matrix sizes (e.g., `1024,2048,4096`).
- `REPS`: repetitions per size per worker.
- `DTYPE`: `float32` (faster, less memory) or `float64` (heavier).
- Use `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` to tune per-process threading (often 2–4 works well when running 1 rank per node).
- The runner computes `-np` from your hostfile and maps **one rank per node** with `--map-by ppr:1:node`, keeping PI5 as the master only.

Troubleshooting:
- If you see `unrecognized arguments: --interval-sec --label`: the launcher no longer passes these to Python; they are only used by the forever wrapper for logging cadence.
- The `plm:rsh: setpgid ... Permission denied` warning is benign with OpenMPI+rsh; you can ignore it.
