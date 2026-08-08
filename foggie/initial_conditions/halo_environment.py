#!/usr/bin/env python3
"""
Environment metric for every halo in a Rockstar catalog.

For each halo, finds the distance to its nearest neighbour of at least half its
own mass.  This is the number to look at when choosing a zoom target: a halo
with a comparable-mass companion a few hundred kpc away is in a merger or a
pair, not the isolated object a zoom is usually meant to follow.  halo79628 is
the case in point -- it has a companion about 150 kpc away, obvious in the
diagnostic plots and invisible in the catalog row.

The criterion is deliberately asymmetric: neighbour mass >= 0.5 x this halo's
mass.  A dwarf sitting next to a cluster is in a crowded environment; the
cluster is not, and the pair should not be reported symmetrically.

Two things that matter for correctness:

  The box is periodic.  A halo near a face has neighbours across it, so
  separations are computed with a periodic KD-tree.  Ignoring that would report
  spuriously isolated halos all around the boundary.

  The eligible neighbours of a halo are a prefix of the mass-sorted catalog,
  and that prefix only grows as mass decreases.  So instead of rebuilding a
  tree per halo (85k trees) the tree is rebuilt at doubling checkpoints and
  candidates are filtered by index, which is exact and runs in seconds.

Usage:
    python3 halo_environment.py                     # default 512 catalog
    python3 halo_environment.py --catalog <path> --out <path>
"""

import argparse
import os
import sys

import numpy as np


DEFAULT_CATALOG = "halo_catalogs_512/512/z0/out_0.list"
DEFAULT_OUTPUT = "halo_catalogs_512/512/z0/halo_environment.ecsv"

# A neighbour counts if it is at least this fraction of the halo's own mass.
MASS_FRACTION = 0.5


def read_catalog(path):
    from astropy.io import ascii as ascii_io

    # data_start=1: data_start=2 drops the first halo (ID 0).  See build.py.
    return ascii_io.read(path, header_start=0, data_start=1)


def box_size_from_header(path, default=25.0):
    """Rockstar records the box size in a comment line; trust the file."""
    with open(path) as fp:
        for line in fp:
            if not line.startswith("#"):
                break
            if "Box size" in line:
                try:
                    return float(line.split(":")[1].split()[0])
                except (IndexError, ValueError):
                    pass
    return default


def nearest_comparable_neighbour(mass, pos, boxsize, mass_fraction=MASS_FRACTION,
                                 verbose=True):
    """Distance and index of the nearest neighbour with mass >= f x own mass.

    Returns (distance, neighbour_index); distance is inf and index -1 where no
    such neighbour exists, which happens only for the most massive halos.
    """
    from scipy.spatial import cKDTree

    n = len(mass)
    order = np.argsort(-mass)              # descending
    m_sorted = mass[order]
    p_sorted = np.mod(pos[order], boxsize)  # cKDTree(boxsize=) needs [0, L)

    # Last index whose mass still qualifies for each halo.  m_sorted descends,
    # so -m_sorted ascends and searchsorted applies directly.  Non-decreasing
    # in i, which is what makes the checkpoint scheme below valid.
    threshold = mass_fraction * m_sorted
    last_eligible = np.searchsorted(-m_sorted, -threshold, side="right") - 1

    dist = np.full(n, np.inf)
    who = np.full(n, -1, dtype=np.int64)

    tree = None
    tree_size = 0
    checkpoint = 0

    for i in range(n):
        p_i = int(last_eligible[i])
        if p_i < 1:                        # only itself qualifies
            continue

        if p_i + 1 > tree_size:
            # Rebuild at doubling checkpoints, so the tree never holds more
            # than twice the eligible set and k stays small.
            checkpoint = max(1024, 1 << int(np.ceil(np.log2(p_i + 1))))
            tree_size = min(checkpoint, n)
            tree = cKDTree(p_sorted[:tree_size], boxsize=boxsize)
            if verbose:
                print("    tree rebuilt at %d halos (i=%d)" % (tree_size, i),
                      file=sys.stderr)

        k = 16
        while True:
            k_use = min(k, tree_size)
            d, j = tree.query(p_sorted[i], k=k_use)
            d = np.atleast_1d(d)
            j = np.atleast_1d(j)
            ok = (j != i) & (j <= p_i) & np.isfinite(d)
            if ok.any():
                pick = np.argmin(np.where(ok, d, np.inf))
                dist[i] = d[pick]
                who[i] = j[pick]
                break
            if k_use >= tree_size:
                break                      # genuinely nothing eligible
            k *= 4

    # Map back to the catalog's original ordering.
    out_dist = np.full(n, np.inf)
    out_who = np.full(n, -1, dtype=np.int64)
    out_dist[order] = dist
    valid = who >= 0
    out_who[order[valid]] = order[who[valid]]
    return out_dist, out_who


def nearest_more_massive(mass, pos, rvir_kpc, boxsize, k=256, verbose=True):
    """Minimum separation from a more massive halo, in units of THAT halo's Rvir.

    This is the satellite test, and the reason it is not the same as the
    comparable-mass metric: normalising by the dwarf's own virial radius
    compresses exactly the regime of interest.  halo11177 sits 0.12 of its own
    Rvir from its neighbour, which only says "close"; measured against the host
    it is at 0.03 Rvir, i.e. 3 % of the way into a halo 43 times its mass.  It
    is not near something, it is inside it.

    Field dwarfs are conventionally those beyond ~1 Rvir_host, or 2-3 Rvir to
    exclude backsplash objects.

    The minimiser is sought among the k nearest more massive halos rather than
    all of them.  A more distant halo can win if its Rvir is large enough, but
    not from outside the nearest few hundred, and the approximation only ever
    affects halos that are isolated by any measure.
    """
    from scipy.spatial import cKDTree

    n = len(mass)
    order = np.argsort(-mass)
    p_sorted = np.mod(pos[order], boxsize)
    r_sorted = rvir_kpc[order]

    frac = np.full(n, np.inf)
    who = np.full(n, -1, dtype=np.int64)

    tree = None
    tree_size = 0
    for i in range(n):
        if i < 1:
            continue                       # the most massive halo has no host
        if i > tree_size:
            tree_size = min(max(1024, 1 << int(np.ceil(np.log2(i + 1)))), n)
            tree = cKDTree(p_sorted[:tree_size], boxsize=boxsize)
            if verbose:
                print("    host tree at %d halos (i=%d)" % (tree_size, i), file=sys.stderr)
        d, j = tree.query(p_sorted[i], k=min(k, tree_size))
        d = np.atleast_1d(d); j = np.atleast_1d(j)
        ok = (j < i) & np.isfinite(d)      # strictly more massive
        if not ok.any():
            continue
        ratio = np.where(ok, d * 1000.0 / np.maximum(r_sorted[j], 1e-9), np.inf)
        pick = int(np.argmin(ratio))
        frac[i] = ratio[pick]
        who[i] = j[pick]

    out_frac = np.full(n, np.inf); out_who = np.full(n, -1, dtype=np.int64)
    out_frac[order] = frac
    valid = who >= 0
    out_who[order[valid]] = order[who[valid]]
    return out_frac, out_who


def tidal_index(mass, pos, boxsize, k=512, n_massive=500, verbose=True):
    """Karachentsev's tidal index: Theta = max_j log10(M_j / d_ij^3).

    The strongest tidal perturbation from any neighbour, in Msun/h per (Mpc/h)^3.
    Complements a nearest-neighbour measure by catching the dwarf with no single
    dominant companion that nonetheless sits inside a group and feels the summed
    pull of several.

    Evaluated over the k nearest halos plus the n_massive most massive in the
    box: M/d^3 is maximised either by proximity or by mass, and the two candidate
    sets between them cover both.
    """
    from scipy.spatial import cKDTree

    n = len(mass)
    P = np.mod(pos, boxsize)
    tree = cKDTree(P, boxsize=boxsize)
    massive = np.argsort(-mass)[:n_massive]

    theta = np.full(n, -np.inf)
    chunk = 4096
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        d, j = tree.query(P[start:stop], k=min(k, n))
        with np.errstate(divide="ignore", invalid="ignore"):
            val = np.log10(mass[j]) - 3.0 * np.log10(d)
        val[(j == np.arange(start, stop)[:, None]) | ~np.isfinite(val)] = -np.inf
        theta[start:stop] = val.max(axis=1)

        # the big ones, wherever they are
        sep = P[start:stop, None, :] - P[None, massive, :]
        sep -= boxsize * np.round(sep / boxsize)
        dm = np.sqrt((sep ** 2).sum(axis=2))
        with np.errstate(divide="ignore", invalid="ignore"):
            vm = np.log10(mass[massive])[None, :] - 3.0 * np.log10(dm)
        vm[(dm <= 0) | ~np.isfinite(vm)] = -np.inf
        theta[start:stop] = np.maximum(theta[start:stop], vm.max(axis=1))
        if verbose and start % (chunk * 8) == 0:
            print("    tidal index %d/%d" % (start, n), file=sys.stderr)
    return theta


def aperture_mass(mass, pos, boxsize, radius_mpc=1.0):
    """Total halo mass within a fixed radius, excluding the halo itself.

    Unlike the ratio-based measures this does not rescale with the halo's own
    mass, so it separates a quiet corner of the box from the outskirts of a
    filament.  Note it sums catalog Mvir over everything in the sphere, so mass
    in substructure is counted twice; it is a relative measure, not a budget.
    """
    from scipy.spatial import cKDTree

    P = np.mod(pos, boxsize)
    tree = cKDTree(P, boxsize=boxsize)
    total = np.zeros(len(mass))
    for i, neigh in enumerate(tree.query_ball_point(P, r=radius_mpc)):
        total[i] = mass[neigh].sum() - mass[i]
    return total


def build_table(catalog_path, mass_fraction=MASS_FRACTION, verbose=True):
    from astropy.table import Table

    halos = read_catalog(catalog_path)
    boxsize = box_size_from_header(catalog_path)
    if verbose:
        print("  %d halos, box %.3f Mpc/h" % (len(halos), boxsize))

    mass = np.asarray(halos["Mvir"], dtype=float)
    pos = np.vstack([np.asarray(halos[c], dtype=float) for c in ("X", "Y", "Z")]).T
    rvir = np.asarray(halos["Rvir"], dtype=float)          # kpc/h
    ids = np.asarray(halos["ID"], dtype=np.int64)

    dist, who = nearest_comparable_neighbour(mass, pos, boxsize, mass_fraction,
                                             verbose=verbose)
    if verbose:
        print("  nearest more massive halo...")
    hfrac, host = nearest_more_massive(mass, pos, rvir, boxsize, verbose=verbose)
    if verbose:
        print("  tidal index...")
    theta = tidal_index(mass, pos, boxsize, verbose=verbose)
    if verbose:
        print("  aperture mass...")
    m1mpc = aperture_mass(mass, pos, boxsize, 1.0)

    found = who >= 0
    d_kpc = np.where(found, dist * 1000.0, np.nan)         # Mpc/h -> kpc/h
    neigh_id = np.where(found, ids[np.clip(who, 0, None)], -1)
    neigh_mass = np.where(found, mass[np.clip(who, 0, None)], np.nan)
    # Separation in units of the halo's own virial radius: the dimensionless
    # form, and the one that says whether a companion is actually interacting.
    d_over_rvir = np.where(found & (rvir > 0), d_kpc / rvir, np.nan)

    table = Table({
        "ID": ids,
        "Mvir": mass,
        "Rvir": rvir,
        "X": pos[:, 0], "Y": pos[:, 1], "Z": pos[:, 2],
        "neighbor_ID": neigh_id.astype(np.int64),
        "neighbor_Mvir": neigh_mass,
        "d_neighbor_kpc": d_kpc,
        "d_neighbor_Rvir": d_over_rvir,
        "has_neighbor": found,
        "host_ID": np.where(host >= 0, ids[np.clip(host, 0, None)], -1).astype(np.int64),
        "host_Mvir": np.where(host >= 0, mass[np.clip(host, 0, None)], np.nan),
        "d_host_Rvir_host": np.where(host >= 0, hfrac, np.nan),
        "tidal_index": theta,
        "M_within_1Mpc": m1mpc,
    })
    table.meta["comments"] = [
        "Environment metric for the %s catalog." % os.path.basename(catalog_path),
        "Generated by foggie/initial_conditions/halo_environment.py -- regenerate,",
        "do not edit.",
        "",
        "d_neighbor_kpc is the distance to the nearest halo whose Mvir is at least",
        "%.2f x this halo's Mvir. Separations are periodic in a %.3f Mpc/h box."
        % (mass_fraction, boxsize),
        "The criterion is asymmetric: a dwarf beside a cluster is in a crowded",
        "environment, the cluster is not.",
        "",
        "Distances are comoving kpc/h; masses Msun/h; Rvir kpc/h, as in the catalog.",
        "has_neighbor is False where no halo is massive enough, which happens only",
        "for the most massive objects; d_neighbor_kpc is NaN there.",
        "",
        "Small d_neighbor_Rvir means a comparable-mass companion close by: a pair or",
        "a merger rather than the isolated object a zoom usually wants.",
        "",
        "d_host_Rvir_host: separation from the nearest MORE MASSIVE halo, in units of",
        "THAT halo's Rvir. Below 1 the halo is inside a bigger one, i.e. a satellite;",
        "field dwarfs are conventionally beyond 1, or beyond 2-3 to exclude backsplash.",
        "This is the satellite test and is not interchangeable with d_neighbor_Rvir,",
        "which normalises by the halo's own radius.",
        "",
        "tidal_index: Karachentsev's Theta = max_j log10(Mvir_j / d_ij^3), Msun/h per",
        "(Mpc/h)^3. Larger means more strongly perturbed. Catches a halo with no single",
        "dominant companion that still sits inside a group.",
        "",
        "M_within_1Mpc: summed catalog Mvir within 1 Mpc/h, excluding self. Substructure",
        "is double counted, so treat it as a relative measure rather than a mass budget.",
        "",
        "An isolated dwarf should satisfy all of: large d_neighbor_Rvir, d_host_Rvir_host",
        "above ~2-3, low tidal_index, low M_within_1Mpc.",
        "",
        "None of these identifies a BACKSPLASH halo -- one that passed through a host",
        "and now sits outside it. Those look isolated at z=0 and need merger trees.",
    ]
    for col, unit in (("Mvir", "Msun/h"), ("neighbor_Mvir", "Msun/h"),
                      ("host_Mvir", "Msun/h"), ("M_within_1Mpc", "Msun/h"),
                      ("Rvir", "kpc/h"), ("d_neighbor_kpc", "kpc/h"),
                      ("X", "Mpc/h"), ("Y", "Mpc/h"), ("Z", "Mpc/h")):
        table[col].unit = unit
    return table


def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", default=os.path.join(here, DEFAULT_CATALOG))
    parser.add_argument("--out", default=os.path.join(here, DEFAULT_OUTPUT))
    parser.add_argument("--mass-fraction", type=float, default=MASS_FRACTION)
    parser.add_argument("--plots", action="store_true", help="also write selection plots")
    parser.add_argument("--highlight", default="",
                        help="comma-separated halo IDs to mark on the plots")
    args = parser.parse_args(argv)

    print("Reading %s" % args.catalog)
    table = build_table(args.catalog, args.mass_fraction)
    table.write(args.out, format="ascii.ecsv", overwrite=True)
    print("Wrote %s" % args.out)

    if args.plots:
        hl = [int(x) for x in args.highlight.split(",") if x.strip()]
        d = os.path.dirname(args.out)
        p1 = make_plots(table, os.path.join(d, "halo_environment_metrics.png"), hl)
        p2 = make_correlation_plot(table, os.path.join(d, "halo_environment_pairs.png"), hl)
        print("Wrote %s\n      %s" % (p1, p2))

    found = np.asarray(table["has_neighbor"])
    d = np.asarray(table["d_neighbor_Rvir"])[found]
    print("\n  %d halos, %d with a neighbour of at least %.0f%% their mass"
          % (len(table), found.sum(), 100 * args.mass_fraction))
    if d.size:
        for q in (5, 25, 50, 75, 95):
            print("    %2dth percentile separation: %8.1f Rvir" % (q, np.percentile(d, q)))
        print("    within 5 Rvir : %d halos (%.1f%%)"
              % ((d < 5).sum(), 100.0 * (d < 5).sum() / d.size))
    return 0




# ---------------------------------------------------------------------------
# Selection plots
# ---------------------------------------------------------------------------

DWARF_RANGE = (1e9, 1e11)      # the mass range these zooms target

# Where a halo stops being a satellite, and where backsplash contamination
# stops mattering much.  Drawn on the plots as guides, not enforced anywhere.
FIELD_CUT = 1.0
CLEAN_CUT = 3.0


def make_plots(table, out_path, highlight=(), dwarf_range=DWARF_RANGE):
    """Scatter each environment metric against halo mass, to select from."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    M = np.asarray(table["Mvir"], float)
    metrics = [
        ("d_neighbor_Rvir", "nearest comparable neighbour\n[own Rvir]", True,
         [(FIELD_CUT, "1 Rvir"), (CLEAN_CUT, "3 Rvir")]),
        ("d_host_Rvir_host", "nearest MORE MASSIVE halo\n[host Rvir]  -- satellite test", True,
         [(FIELD_CUT, "inside host"), (CLEAN_CUT, "backsplash-safe")]),
        ("tidal_index", r"tidal index $\Theta$", False, []),
        ("M_within_1Mpc", r"mass within 1 Mpc/h  [M$_\odot$/h]", True, []),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5))
    for ax, (col, label, logy, guides) in zip(axes.ravel(), metrics):
        y = np.asarray(table[col], float)
        good = np.isfinite(y) & np.isfinite(M) & (M > 0)
        if logy:
            good &= y > 0

        # 85k points: bin them, then overplot the targets.
        ax.hexbin(np.log10(M[good]), np.log10(y[good]) if logy else y[good],
                  gridsize=110, bins="log", cmap="Blues", mincnt=1, linewidths=0)

        for value, text in guides:
            yy = np.log10(value) if logy else value
            ax.axhline(yy, color="#c62828", lw=1.0, ls="--", alpha=0.8)
            ax.text(0.985, yy, " " + text, color="#c62828", fontsize=8,
                    ha="right", va="bottom", transform=ax.get_yaxis_transform())

        ax.axvspan(np.log10(dwarf_range[0]), np.log10(dwarf_range[1]),
                   color="#1f7a52", alpha=0.06, zorder=0)

        for hid in highlight:
            row = table[table["ID"] == hid]
            if not len(row):
                continue
            r = row[0]
            yv = float(r[col])
            if not np.isfinite(yv) or (logy and yv <= 0):
                continue
            xv = np.log10(float(r["Mvir"]))
            yv = np.log10(yv) if logy else yv
            ax.plot(xv, yv, "o", ms=8, mfc="none", mec="#c62828", mew=1.8, zorder=5)
            ax.annotate(str(hid), (xv, yv), textcoords="offset points",
                        xytext=(7, 5), fontsize=8.5, color="#c62828", weight="bold")

        ax.set_xlabel(r"log$_{10}$ M$_{\rm vir}$ [M$_\odot$/h]")
        ax.set_ylabel(("log$_{10}$ " if logy else "") + label)
        ax.grid(alpha=0.15)

    fig.suptitle("Halo environment metrics -- shaded band is the dwarf range "
                 "%.0e to %.0e; circles are the current zoom targets"
                 % dwarf_range, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=115)
    plt.close(fig)
    return out_path


def make_correlation_plot(table, out_path, highlight=(), dwarf_range=DWARF_RANGE):
    """The metrics against each other, over the dwarf range only.

    Shows where they agree and where they do not: a halo can look isolated on
    one axis and crowded on another, which is the whole reason for having more
    than one.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    M = np.asarray(table["Mvir"], float)
    sel = (M >= dwarf_range[0]) & (M <= dwarf_range[1])
    t = table[sel]

    pairs = [("d_host_Rvir_host", "d_neighbor_Rvir", True, True),
             ("d_host_Rvir_host", "tidal_index", True, False),
             ("tidal_index", "M_within_1Mpc", False, True)]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for ax, (cx, cy, lx, ly) in zip(axes, pairs):
        x = np.asarray(t[cx], float)
        y = np.asarray(t[cy], float)
        good = np.isfinite(x) & np.isfinite(y)
        if lx:
            good &= x > 0
        if ly:
            good &= y > 0
        ax.hexbin(np.log10(x[good]) if lx else x[good],
                  np.log10(y[good]) if ly else y[good],
                  gridsize=80, bins="log", cmap="Blues", mincnt=1, linewidths=0)
        if cx == "d_host_Rvir_host":
            for v, txt in ((FIELD_CUT, "inside host"), (CLEAN_CUT, "3 Rvir")):
                ax.axvline(np.log10(v), color="#c62828", lw=1.0, ls="--", alpha=0.8)
                ax.text(np.log10(v), 0.99, " " + txt, color="#c62828", fontsize=8,
                        rotation=90, va="top", transform=ax.get_xaxis_transform())
        for hid in highlight:
            row = table[table["ID"] == hid]
            if not len(row):
                continue
            r = row[0]
            xv, yv = float(r[cx]), float(r[cy])
            if not (np.isfinite(xv) and np.isfinite(yv)):
                continue
            if (lx and xv <= 0) or (ly and yv <= 0):
                continue
            xv = np.log10(xv) if lx else xv
            yv = np.log10(yv) if ly else yv
            ax.plot(xv, yv, "o", ms=8, mfc="none", mec="#c62828", mew=1.8, zorder=5)
            ax.annotate(str(hid), (xv, yv), textcoords="offset points",
                        xytext=(7, 5), fontsize=8.5, color="#c62828", weight="bold")
        ax.set_xlabel(("log$_{10}$ " if lx else "") + cx)
        ax.set_ylabel(("log$_{10}$ " if ly else "") + cy)
        ax.grid(alpha=0.15)
    fig.suptitle("Environment metrics against each other, %.0e < Mvir < %.0e"
                 % dwarf_range, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=115)
    plt.close(fig)
    return out_path

if __name__ == "__main__":
    sys.exit(main())
