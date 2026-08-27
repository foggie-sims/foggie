"""Resource comparison: N standalone zoom ladders vs one multizoom group.

Sums per-level costs (Enzo performance.out wallclock, immune to queue gaps)
over the standalone runs of every group member and sets them against the
group's merged runs.  The saving to expect: each standalone run pays the
full root-grid evolution (~a third of an L1 run's cost) again; the group
pays it once.

Usage::

    python -m foggie.initial_conditions.multizoom.compare_costs \
        --group-dir runs/multizoom_sixpack --halos 48014,56672,... \
        --standalone-root /nobackupnfs1/jtumlins/25Mpc_new_cosmology \
        [--sim 25Mpc_DM_512] [--levels 1,2,3] [--png out.png]
"""

import argparse
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from foggie.initial_conditions.multizoom.loadbalance import parse_performance


def collect(args):
    rows = []
    for level in args.levels:
        alone = []
        for h in args.halos:
            d = os.path.join(args.standalone_root, "halo%s" % h,
                             "%s-L%d" % (args.sim, level))
            perf = parse_performance(d, args.nprocs)
            if perf:
                alone.append((h, perf))
        gd = os.path.join(args.group_dir, "%s-L%d" % (args.sim, level))
        group = parse_performance(gd, args.nprocs)
        rows.append(dict(level=level, alone=alone, group=group))
    return rows


def report(rows):
    lines = []
    lines.append("%-6s %14s %14s %8s   %s"
                 % ("level", "standalone sum", "multizoom", "ratio", "notes"))
    tot_a = tot_g = 0.0
    for r in rows:
        a = sum(p["cpu_hours"] for _, p in r["alone"])
        n = len(r["alone"])
        g = r["group"]["cpu_hours"] if r["group"] else None
        done = (r["group"] and r["group"]["cycles"] >= 1000)
        tot_a += a
        note = "%d standalone runs" % n
        if g is None:
            lines.append("L%-5d %11.1f ch %14s %8s   %s"
                         % (r["level"], a, "--", "--", note))
            continue
        tot_g += g
        if not done:
            note += "; multizoom IN PROGRESS (%d cycles)" % r["group"]["cycles"]
        lines.append("L%-5d %11.1f ch %11.1f ch %8.2f   %s"
                     % (r["level"], a, g, g / a if a else 0, note))
    if tot_g:
        lines.append("%-6s %11.1f ch %11.1f ch %8.2f"
                     % ("total", tot_a, tot_g, tot_g / tot_a if tot_a else 0))
    return "\n".join(lines)


def balance_lines(rows):
    lines = ["", "load balance (time-weighted max/mean per level of the "
                 "hierarchy; 1.0 = perfect):"]
    for r in rows:
        if not r["group"]:
            continue
        levs = r["group"]["levels"]
        top = sorted(levs.items(), key=lambda kv: -kv[1]["seconds"])[:3]
        alone_top = {}
        for h, p in r["alone"]:
            for lev, d in p["levels"].items():
                alone_top.setdefault(lev, []).append(d["imbalance"])
        parts = []
        for lev, d in sorted(top):
            al = alone_top.get(lev)
            al_s = ("%.2f" % (sum(al) / len(al))) if al else "--"
            parts.append("H%d: mz %.2f vs alone %s" % (lev, d["imbalance"], al_s))
        lines.append("  multizoom L%d:  %s" % (r["level"], ";  ".join(parts)))
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--group-dir", required=True)
    p.add_argument("--halos", required=True)
    p.add_argument("--standalone-root", required=True)
    p.add_argument("--sim", default="25Mpc_DM_512")
    p.add_argument("--levels", default="1,2,3")
    p.add_argument("--nprocs", type=int, default=64)
    p.add_argument("--png", default=None)
    args = p.parse_args(argv)
    args.halos = args.halos.split(",")
    args.levels = [int(v) for v in args.levels.split(",")]

    rows = collect(args)
    print(report(rows))
    print(balance_lines(rows))

    if args.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        fig, ax = plt.subplots(figsize=(7, 4.2))
        levels = [r["level"] for r in rows]
        x = np.arange(len(levels))
        alone = [sum(pp["cpu_hours"] for _, pp in r["alone"]) for r in rows]
        group = [r["group"]["cpu_hours"] if r["group"] else 0 for r in rows]
        w = 0.38
        ax.bar(x - w / 2, alone, w, label="6 standalone ladders (sum)",
               color="#5b8dbe")
        ax.bar(x + w / 2, group, w, label="one multizoom domain",
               color="#c96f4a")
        for r, xi in zip(rows, x):
            if r["group"] and r["group"]["cycles"] < 1000:
                ax.text(xi + w / 2, r["group"]["cpu_hours"], "in\nprogress",
                        ha="center", va="bottom", fontsize=8, color="#c96f4a")
        ax.set_xticks(x, ["L%d" % l for l in levels])
        ax.set_ylabel("core-hours (64-core nodes)")
        ax.set_title("DM ladder cost: standalone vs multizoom")
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(args.png, dpi=150)
        print("wrote", args.png)


if __name__ == "__main__":
    main()
