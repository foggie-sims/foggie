"""Load-balance and resource tracking for Enzo runs.

Two data sources, both free -- no instrumentation added to the runs:

* each packed-AMR dump's .cpuNNNN files say exactly which grids (cells and
  particles) each processor owns -> per-output imbalance metrics;
* the OutputLog's timestamps say how long the run took to reach each output
  -> node-hours versus redshift, the currency of the standalone-vs-multizoom
  comparison.

Usage::

    python -m foggie.initial_conditions.multizoom.loadbalance survey \
        --run <dir> [--run <dir> ...] [--csv out.csv]
"""

import argparse
import glob
import os
import re
import sys
from datetime import datetime

import h5py
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))


def dump_balance(dump_dir):
    """Per-processor cells and particles for one packed-AMR dump.

    Returns dict with n_cpus, grids, cells, particles, and imbalance =
    max/mean of cells per cpu (1.0 = perfect).  Level information is not in
    the cpu files; cells-per-cpu is the quantity LoadBalancing acts on.
    """
    name = os.path.basename(dump_dir.rstrip("/"))
    cpu_files = sorted(glob.glob(os.path.join(dump_dir, name + ".cpu*")))
    if not cpu_files:
        return None
    cells = []
    parts = []
    grids = []
    for fn in cpu_files:
        c = p = g = 0
        with h5py.File(fn, "r") as f:
            for gname in f:
                if not gname.startswith("Grid"):
                    continue
                g += 1
                gr = f[gname]
                if "particle_position_x" in gr:
                    p += gr["particle_position_x"].shape[0]
                # active cells: use any field dataset; DM-only dumps may have
                # none, fall back to Dark_Matter_Density if written
                for field in ("Dark_Matter_Density", "Density"):
                    if field in gr:
                        c += int(np.prod(gr[field].shape))
                        break
        cells.append(c)
        parts.append(p)
        grids.append(g)
    cells = np.array(cells, float)
    parts = np.array(parts, float)
    work = cells if cells.sum() > 0 else parts   # DM-only fallback
    return dict(
        dump=name, n_cpus=len(cpu_files),
        grids=int(sum(grids)),
        cells=int(cells.sum()), particles=int(parts.sum()),
        cells_max=int(work.max()), cells_mean=float(work.mean()),
        imbalance=float(work.max() / work.mean()) if work.mean() else 0.0,
        gini=_gini(work))


def _gini(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def dump_redshift(dump_dir):
    name = os.path.basename(dump_dir.rstrip("/"))
    par = os.path.join(dump_dir, name)
    if not os.path.exists(par):
        return None
    for line in open(par, errors="ignore"):
        m = re.match(r"CosmologyCurrentRedshift\s*=\s*(\S+)", line)
        if m:
            return float(m.group(1))
    return None


def output_log_times(run_dir):
    """{dump_name: unix_time} from OutputLog write timestamps."""
    times = {}
    log = os.path.join(run_dir, "OutputLog")
    if not os.path.exists(log):
        return times
    for line in open(log):
        # DATAOUT "CosmologyOutput" ./RD0004/RD0004 <time> <cycle> <z>
        parts = line.split()
        if len(parts) >= 3 and "/" in parts[2]:
            name = os.path.basename(parts[2])
            times[name] = os.path.getmtime(
                os.path.join(run_dir, name, name)) \
                if os.path.exists(os.path.join(run_dir, name, name)) else None
    return times


def survey_run(run_dir):
    """One row per dump: redshift, wallclock since first dump, balance."""
    rows = []
    dumps = sorted(glob.glob(os.path.join(run_dir, "RD????")))
    t0 = None
    for d in dumps:
        name = os.path.basename(d)
        par = os.path.join(d, name)
        if not os.path.exists(par):
            continue
        t = os.path.getmtime(par)
        if t0 is None:
            t0 = t
        bal = dump_balance(d) or {}
        bal.pop("dump", None)
        rows.append(dict(run=os.path.basename(run_dir.rstrip("/")),
                         dump=name, z=dump_redshift(d),
                         hours=(t - t0) / 3600.0, **bal))
    return rows




def _timing_skip(path):
    """Cycles between recorded blocks (Enzo's TimingCycleSkip), from the file.

    Read from the data rather than the parameter file: a restart can change
    it mid-run, and the blocks are the ground truth.
    """
    import re as _re
    nums = []
    for block in open(path).read().split("Cycle_Number")[1:]:
        m = _re.match(r"\s*(\d+)", block)
        if m:
            nums.append(int(m.group(1)))
        if len(nums) >= 12:
            break
    if len(nums) < 3:
        return 1
    deltas = [b - a for a, b in zip(nums, nums[1:]) if b > a]
    if not deltas:
        return 1
    deltas.sort()
    return max(deltas[len(deltas) // 2], 1)


def parse_performance(run_dir, nprocs=64):
    """Cost and balance from Enzo's performance.out (ENZO_PERFORMANCE build).

    Per cycle the file reports, for each level and for the cycle total:
    mean/std/min/max time across MPI ranks plus cell updates and grid counts.
    Wallclock summed from the per-cycle totals is immune to queue gaps and
    restarts, so it is the number to compare across runs.

    Returns dict with wall_hours, cpu_hours, cycles, per-level totals
    (seconds, cell updates) and the time-weighted imbalance max/mean on each
    level (1.0 = perfect balance).
    """
    import collections
    path = os.path.join(run_dir, "performance.out")
    if not os.path.exists(path):
        return None
    # TimingCycleSkip (the ics_refactor gas template sets 10) changes how
    # OFTEN a block is written, not what it measures: Enzo's timers accumulate
    # over the skipped cycles, so summing blocks still gives the true elapsed
    # time.  Verified against PBS: 42 blocks at skip=10 summed to 2.62 h
    # against 2.92 h of job walltime.  What the skip does change is the
    # RESOLUTION of any cycle window, which matters when comparing runs.
    skip = _timing_skip(path)
    wall = 0.0
    cycles = 0
    level_time = collections.defaultdict(float)
    level_max_time = collections.defaultdict(float)
    level_updates = collections.defaultdict(float)
    for line in open(path):
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "Cycle_Number":
            cycles += 1
        elif parts[0] == "Total" and len(parts) >= 5:
            wall += float(parts[1])
        elif parts[0].startswith("Level_") and len(parts) >= 7:
            lev = int(parts[0][6:])
            level_time[lev] += float(parts[1])
            level_max_time[lev] += float(parts[4])
            level_updates[lev] += float(parts[5])
    levels = {}
    for lev in sorted(level_time):
        mean_t = level_time[lev]
        levels[lev] = dict(
            seconds=mean_t,
            imbalance=(level_max_time[lev] / mean_t) if mean_t else 0.0,
            cell_updates=level_updates[lev])
    return dict(run=run_dir, cycles=cycles, timing_skip=skip,
                sampled_cycles=cycles,
                wall_hours=wall / 3600.0,
                cpu_hours=wall / 3600.0 * nprocs,
                levels=levels)


def print_perf(perf):
    skip = perf.get("timing_skip", 1)
    note = "" if skip == 1 else "  [blocks every %d cycles]" % skip
    print("%-46s cycles %5d  wall %6.2f h  cpu %7.1f core-h%s"
          % (os.path.basename(perf["run"].rstrip("/")) or perf["run"],
             perf["cycles"] * skip, perf["wall_hours"], perf["cpu_hours"], note))
    for lev, d in perf["levels"].items():
        print("    L%02d  %8.1f s (%4.1f%%)  imbalance %5.2f  updates %.3e"
              % (lev, d["seconds"],
                 100 * d["seconds"] / (perf["wall_hours"] * 3600.0)
                 if perf["wall_hours"] else 0.0,
                 d["imbalance"], d["cell_updates"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("survey")
    p.add_argument("--run", action="append", required=True)
    p.add_argument("--csv", default=None)
    p = sub.add_parser("perf")
    p.add_argument("--run", action="append", required=True)
    p.add_argument("--nprocs", type=int, default=64)
    args = parser.parse_args(argv)

    if args.cmd == "perf":
        for run in args.run:
            perf = parse_performance(run, args.nprocs)
            if perf is None:
                print("%s: no performance.out" % run)
            else:
                print_perf(perf)
        return

    all_rows = []
    for run in args.run:
        rows = survey_run(run)
        all_rows += rows
        print("== %s" % run)
        print("%-8s %8s %8s %9s %12s %11s %7s %6s"
              % ("dump", "z", "hours", "grids", "particles", "cells",
                 "max/mean", "gini"))
        for r in rows:
            print("%-8s %8.3f %8.2f %9d %12d %11d %7.2f %6.3f"
                  % (r["dump"], r.get("z") or -1, r["hours"],
                     r.get("grids", 0), r.get("particles", 0),
                     r.get("cells", 0), r.get("imbalance", 0),
                     r.get("gini", 0)))
    if args.csv and all_rows:
        import csv
        keys = sorted({k for r in all_rows for k in r})
        with open(args.csv, "w", newline="") as fp:
            w = csv.DictWriter(fp, fieldnames=keys)
            w.writeheader()
            w.writerows(all_rows)
        print("wrote", args.csv)


if __name__ == "__main__":
    main()
