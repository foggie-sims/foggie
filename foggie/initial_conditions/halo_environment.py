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
    ]
    for col, unit in (("Mvir", "Msun/h"), ("neighbor_Mvir", "Msun/h"),
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
    args = parser.parse_args(argv)

    print("Reading %s" % args.catalog)
    table = build_table(args.catalog, args.mass_fraction)
    table.write(args.out, format="ascii.ecsv", overwrite=True)
    print("Wrote %s" % args.out)

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


if __name__ == "__main__":
    sys.exit(main())
