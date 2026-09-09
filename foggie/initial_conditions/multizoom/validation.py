"""Validation checks for multizoom IC builds.

The first battery covers a union-mode build: the single MUSIC run must
produce nested grids that enclose every halo's Lagrangian cloud, and a
RefinementMask whose refine cells form one disjoint cloud per halo with the
volume between them left unrefined -- the property that separates a
multizoom mask from a naive spanning hull.

Usage::

    python -m foggie.initial_conditions.multizoom.validation union \
        --ics /path/to/25Mpc_DM_512-L1 [--min-clouds 2]
"""

import argparse
import glob
import os
import re
import sys

import h5py
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))


def read_mask(ics_dir, level=None):
    """The RefinementMask at `level` (default: finest) as (nx, ny, nz)."""
    masks = sorted(glob.glob(os.path.join(ics_dir, "RefinementMask.*")),
                   key=lambda p: p.rsplit(".", 1)[1])
    masks = [m for m in masks if not m.endswith(".bak")]
    if not masks:
        raise RuntimeError("no RefinementMask.* in %s" % ics_dir)
    path = masks[-1] if level is None else \
        os.path.join(ics_dir, "RefinementMask.%d" % level)
    name = os.path.basename(path)
    with h5py.File(path, "r") as fp:
        return fp[name][0].T  # (1, nz, ny, nx) -> (nx, ny, nz)


def label_clouds(refined):
    """Label 6-connected components of a boolean refine field.

    Returns (labels array, count).  Uses scipy when available; the fields
    are small (a zoom patch), so the fallback flood fill is fine too.
    """
    try:
        from scipy import ndimage
        labels, n = ndimage.label(refined)
        return labels, int(n)
    except ImportError:
        labels = np.zeros(refined.shape, dtype=np.int32)
        n = 0
        for seed in zip(*np.nonzero(refined)):
            if labels[seed]:
                continue
            n += 1
            stack = [seed]
            labels[seed] = n
            while stack:
                i, j, k = stack.pop()
                for di, dj, dk in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),
                                   (0,0,1),(0,0,-1)):
                    p = (i+di, j+dj, k+dk)
                    if all(0 <= p[d] < refined.shape[d] for d in range(3)) \
                            and refined[p] and not labels[p]:
                        labels[p] = n
                        stack.append(p)
        return labels, n


def cloud_summary(mask, min_cells=8):
    """Connected refine clouds in a mask, ignoring specks below min_cells."""
    refined = mask >= 0          # MUSIC convention: -1 outside, >=0 refine
    labels, n = label_clouds(refined)
    clouds = []
    for lab in range(1, n + 1):
        cells = int((labels == lab).sum())
        if cells < min_cells:
            continue
        idx = np.nonzero(labels == lab)
        clouds.append(dict(
            cells=cells,
            center=[float(np.mean(i)) for i in idx],
            lo=[int(i.min()) for i in idx],
            hi=[int(i.max()) for i in idx]))
    clouds.sort(key=lambda c: -c["cells"])
    return dict(refined_cells=int(refined.sum()),
                total_cells=int(mask.size),
                clouds=clouds)


def read_grid_geometry(ics_dir):
    """CosmologySimulationGrid* entries from parameter_file.txt."""
    grids = {}
    pf = os.path.join(ics_dir, "parameter_file.txt")
    for line in open(pf):
        m = re.match(r"CosmologySimulationGrid(LeftEdge|RightEdge|Dimension)"
                     r"\[(\d+)\]\s*=\s*(.*)", line)
        if m:
            key, idx, vals = m.group(1), int(m.group(2)), m.group(3).split()
            grids.setdefault(idx, {})[key] = [float(v) for v in vals]
    return grids


def check_union(ics_dir, min_clouds=2, expect_point_files=None):
    """Run the union-mode battery.  Returns (ok, report_lines)."""
    lines = []
    ok = True

    grids = read_grid_geometry(ics_dir)
    for idx in sorted(grids):
        g = grids[idx]
        size = [g["RightEdge"][d] - g["LeftEdge"][d] for d in range(3)]
        lines.append("grid %d: left %s  size %s (%.2f Mpc/h max)"
                     % (idx, np.round(g["LeftEdge"], 4),
                        np.round(size, 4), 25 * max(size)))
        if max(size) >= 0.5:
            ok = False
            lines.append("  FAIL: patch spans >= half the box")

    mask = read_mask(ics_dir)
    summary = cloud_summary(mask)
    frac = summary["refined_cells"] / summary["total_cells"]
    lines.append("mask: %d of %d cells refine (%.1f%%), %d cloud(s)"
                 % (summary["refined_cells"], summary["total_cells"],
                    100 * frac, len(summary["clouds"])))
    for c in summary["clouds"]:
        lines.append("  cloud: %7d cells, center %s, extent %s"
                     % (c["cells"], np.round(c["center"], 1),
                        [h - l + 1 for l, h in zip(c["lo"], c["hi"])]))
    if len(summary["clouds"]) < min_clouds:
        ok = False
        lines.append("FAIL: expected >= %d disjoint refine clouds, found %d "
                     "(a spanning hull would look like 1)"
                     % (min_clouds, len(summary["clouds"])))

    if expect_point_files:
        for fn in expect_point_files:
            if not os.path.exists(fn):
                ok = False
                lines.append("FAIL: expected point file missing: %s" % fn)

    lines.append("RESULT: %s" % ("PASS" if ok else "FAIL"))
    return ok, lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("union", help="validate a union-mode IC directory")
    u.add_argument("--ics", required=True)
    u.add_argument("--min-clouds", type=int, default=2)
    args = parser.parse_args(argv)
    ok, lines = check_union(args.ics, args.min_clouds)
    print("\n".join(lines))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
