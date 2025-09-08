#!/usr/bin/env python3
"""
mpi_cluster_bench.py
Master/worker MPI benchmark: rank 0 coordinates tasks, workers perform dense GEMMs.
Prints live results and writes CSV/JSON artifacts per run.

Usage (env set by run_cluster_bench.sh):
  python mpi_cluster_bench.py --sizes 2048,3072,4096 --reps 2 --dtype float32 --outdir /path

Notes:
- Master is PI5 (not listed in hostfile). Workers are one rank per node.
- Each task is (size, rep_id). Worker creates two NxN matrices and times C = A @ B.
- GFLOPs = (2 * N^3) / seconds / 1e9
"""
from __future__ import annotations

import argparse, csv, json, math, os, socket, sys, time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

from mpi4py import MPI
import numpy as np

TAG_TASK  = 1
TAG_RESULT= 2
TAG_STOP  = 3
TAG_READY = 4

@dataclass
class Result:
    host: str
    rank: int
    size: int
    secs: float
    gflops: float
    rep: int
    t_wall: float

def now_utc_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", default="2048,3072,4096", help="Comma list of matrix sizes")
    p.add_argument("--reps", type=int, default=2, help="Repetitions per size per worker")
    p.add_argument("--dtype", choices=("float32","float64"), default="float32")
    p.add_argument("--outdir", default=os.path.expanduser("~/cluster_bench_out"))
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--continuous", action="store_true", help="loop forever inside the program")
    return p.parse_args()

def dtype_of(name: str):
    return np.float32 if name == "float32" else np.float64

def flops_count(n: int) -> float:
    return 2.0 * (n**3)

def master_single_run(comm: MPI.Comm, sizes: List[int], reps: int, outdir: str, dtype: str, run_idx: int) -> None:
    world = comm.Get_size()
    if world < 2:
        print("[ERR] Need at least 2 ranks (1 master + >=1 worker).", flush=True)
        return
    workers = list(range(1, world))
    nworkers = len(workers)

    # Prepare tasks: balanced by cycling sizes across workers
    tasks: List[Tuple[int, int]] = []
    # We create reps tasks per size in total; with nworkers workers, each will do roughly same count
    for r in range(reps):
        for s in sizes:
            tasks.append((s, r))
    ntasks_total = len(tasks)

    # Bookkeeping
    next_task_idx = 0
    done = 0
    results: List[Result] = []

    # Seed workers by sending READY handshake first
    status = MPI.Status()
    ready_workers = 0
    # While not all workers have said READY, receive READY and seed initial tasks
    while ready_workers < nworkers:
        data = comm.recv(source=MPI.ANY_SOURCE, tag=TAG_READY, status=status)
        src = status.Get_source()
        ready_workers += 1
        if next_task_idx < ntasks_total:
            comm.send(tasks[next_task_idx], dest=src, tag=TAG_TASK)
            next_task_idx += 1
        else:
            # no work at all
            comm.send(None, dest=src, tag=TAG_STOP)

    # Main loop: receive results and send more tasks
    while done < ntasks_total:
        res = comm.recv(source=MPI.ANY_SOURCE, tag=TAG_RESULT, status=status)
        src = status.Get_source()
        results.append(Result(**res))
        done += 1
        rank_idx = src
        # progress line
        print(f"[RES] {done}/{ntasks_total} r={res['rep']} host={res['host']} size={res['size']} secs={res['secs']:.4f} gflops={res['gflops']:.2f}", flush=True)

        if next_task_idx < ntasks_total:
            comm.send(tasks[next_task_idx], dest=src, tag=TAG_TASK)
            next_task_idx += 1
        else:
            comm.send(None, dest=src, tag=TAG_STOP)

    # Aggregate
    by_size: Dict[int, List[float]] = {}
    for r in results:
        by_size.setdefault(r.size, []).append(r.gflops)

    print("[AGG] GFLOPS by size (mean ± std, n):", flush=True)
    for s in sorted(by_size):
        arr = np.array(by_size[s], dtype=np.float64)
        mean = float(arr.mean())
        std = float(arr.std(ddof=0)) if arr.size > 1 else 0.0
        print(f"  N={s}: {mean:.2f} ± {std:.2f} (n={arr.size})", flush=True)

    # Write artifacts
    ts = time.strftime("%Y%m%d_%H%M%SZ", time.gmtime())
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, f"bench_{ts}_run{run_idx}.csv")
    json_path = os.path.join(outdir, f"bench_{ts}_run{run_idx}.json")

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["host","rank","size","secs","gflops","rep","t_wall_iso"])
        for r in results:
            w.writerow([r.host, r.rank, r.size, f"{r.secs:.6f}", f"{r.gflops:.3f}", r.rep, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(r.t_wall))])

    meta = {
        "timestamp_utc": now_utc_iso(),
        "world_size": world,
        "nworkers": nworkers,
        "sizes": sizes,
        "reps": reps,
        "dtype": dtype,
        "results": [asdict(r) for r in results],
    }
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[DONE] wrote {csv_path} and {json_path}", flush=True)

def master_loop(comm: MPI.Comm, args):
    sizes = [int(s) for s in str(args.sizes).split(",") if s.strip()]
    run_idx = 0
    while True:
        print(f"[INFO] Starting cluster bench: world_size={comm.Get_size()} sizes={','.join(map(str,sizes))} dtype={args.dtype} reps={args.reps}", flush=True)
        master_single_run(comm, sizes, args.reps, args.outdir, args.dtype, run_idx)
        run_idx += 1
        if not args.continuous:
            break

def worker_loop(comm: MPI.Comm, args):
    # Worker loop: send READY, then receive tasks until STOP
    rng = np.random.default_rng(args.seed + comm.Get_rank() if args.seed is not None else None)
    myhost = socket.gethostname()
    dtype = dtype_of(args.dtype)

    while True:
        comm.send(None, dest=0, tag=4)  # READY
        status = MPI.Status()
        msg = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()
        if tag == TAG_STOP:
            break
        size, rep = msg
        # Allocate and time matmul
        a = rng.random((size, size), dtype=dtype)
        b = rng.random((size, size), dtype=dtype)
        t0 = MPI.Wtime()
        c = a @ b  # BLAS-backed GEMM
        t1 = MPI.Wtime()
        secs = t1 - t0
        gflops = flops_count(size) / secs / 1e9
        res = dict(host=myhost, rank=comm.Get_rank(), size=int(size), secs=float(secs), gflops=float(gflops), rep=int(rep), t_wall=time.time())
        comm.send(res, dest=0, tag=TAG_RESULT)

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        master_loop(comm, args)
    else:
        worker_loop(comm, args)

if __name__ == "__main__":
    main()
