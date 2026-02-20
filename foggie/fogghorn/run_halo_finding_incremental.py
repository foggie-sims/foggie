#!/usr/bin/env python
# coding: utf-8
"""Incremental halo finding wrapper for Enzo simulation outputs.

Scans a simulation directory for Enzo output directories (DD####, RD####, etc.),
determines which ones do not yet have a halo catalog, and processes only those.
Can be run once or in a continuous watch mode to process new outputs as they appear.

Usage (one-shot):
    python run_halo_finding_incremental.py --directory /path/to/sim [options]

Usage (watch mode, poll every 10 minutes):
    python run_halo_finding_incremental.py --directory /path/to/sim --watch --interval 600 [options]

Written by JT and Claude February 2026 
"""

import argparse
import os
import re
import time
import sys

from quick_halo_finding import (
    prep_dataset_for_halo_finding,
    halo_finding_step,
    repair_halo_catalog,
    export_to_astropy,
    make_halo_plots,
    find_root_particles,
)


# Enzo output directories look like DD0001, RD0001, etc.
_SNAP_PATTERN = re.compile(r'^[A-Z]{2}\d{4}$')


def find_all_snapshots(simulation_dir):
    """Return a sorted list of Enzo snapshot names found in simulation_dir."""
    try:
        entries = os.listdir(simulation_dir)
    except OSError as e:
        print(f"ERROR: cannot list directory '{simulation_dir}': {e}")
        return []

    snaps = sorted(
        e for e in entries
        if _SNAP_PATTERN.match(e) and os.path.isdir(os.path.join(simulation_dir, e))
    )
    return snaps


def catalog_exists(simulation_dir, snapname):
    """Return True if a halo catalog .h5 file already exists for this snapshot."""
    catalog_path = os.path.join(
        simulation_dir, 'halo_catalogs', snapname, snapname + '.0.h5'
    )
    return os.path.isfile(catalog_path)


def process_snapshot(snapname, args):
    """Run all halo finding steps for a single snapshot.

    Parameters
    ----------
    snapname : str
        Snapshot directory name (e.g. 'DD0100').
    args : argparse.Namespace
        Parsed command-line arguments.
    """
    print(f"\n{'='*60}")
    print(f"Processing snapshot: {snapname}")
    print(f"{'='*60}")

    try:
        ds, box = prep_dataset_for_halo_finding(
            args.directory, snapname,
            trackfile=args.trackfile,
            boxwidth=args.boxwidth,
        )
        hc = halo_finding_step(
            ds, box,
            simulation_dir=args.directory,
            threshold=args.threshold,
        )
        hc = repair_halo_catalog(
            ds, args.directory, snapname,
            min_rvir=args.min_rvir,
            min_halo_mass=args.min_mass,
        )
        export_to_astropy(args.directory, snapname)

        if args.make_plots:
            make_halo_plots(ds, args.directory, snapname)

        if args.save_root:
            import yt
            hc_ds = yt.load(
                args.directory + '/halo_catalogs/' + snapname + '/' + snapname + '.0.h5'
            )
            find_root_particles(args.directory, ds, hc_ds)

        print(f"Done: {snapname}")
        return True

    except Exception as e:
        print(f"ERROR processing {snapname}: {e}", file=sys.stderr)
        return False


def run_once(args):
    """Find and process all snapshots that do not yet have a halo catalog."""
    all_snaps = find_all_snapshots(args.directory)

    if not all_snaps:
        print(f"No Enzo snapshots found in '{args.directory}'.")
        return

    new_snaps = [s for s in all_snaps if not catalog_exists(args.directory, s)]

    if not new_snaps:
        print(f"All {len(all_snaps)} snapshots already have halo catalogs. Nothing to do.")
        return

    print(f"Found {len(all_snaps)} snapshots total, {len(new_snaps)} without catalogs:")
    for s in new_snaps:
        print(f"  {s}")

    failed = []
    for snap in new_snaps:
        success = process_snapshot(snap, args)
        if not success:
            failed.append(snap)

    print(f"\nFinished. Processed {len(new_snaps) - len(failed)}/{len(new_snaps)} snapshots.")
    if failed:
        print(f"Failed snapshots: {', '.join(failed)}")


def run_watch(args):
    """Poll the simulation directory periodically and process new snapshots."""
    print(f"Watch mode enabled. Polling every {args.interval} seconds.")
    print(f"Press Ctrl-C to stop.\n")

    processed = set()

    while True:
        all_snaps = find_all_snapshots(args.directory)
        new_snaps = [
            s for s in all_snaps
            if s not in processed and not catalog_exists(args.directory, s)
        ]

        if new_snaps:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Found {len(new_snaps)} new snapshot(s) to process.")
            for snap in new_snaps:
                success = process_snapshot(snap, args)
                if success:
                    processed.add(snap)
        else:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] No new snapshots. Sleeping {args.interval}s...")

        time.sleep(args.interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Incrementally run quick_halo_finding on new Enzo outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--directory', metavar='directory', type=str, default='./',
                        help='Path to the simulation directory containing Enzo outputs')
    parser.add_argument('--trackfile', metavar='trackfile', type=str, default=None,
                        help='Track file for halo center (passed to foggie_load)')
    parser.add_argument('--boxwidth', metavar='boxwidth', type=float, default=0.04,
                        help='Width of subregion box in code units')
    parser.add_argument('--threshold', metavar='threshold', type=float, default=400.,
                        help='Overdensity threshold for HOP algorithm')
    parser.add_argument('--min_rvir', metavar='min_rvir', type=float, default=10.,
                        help='Minimum virial radius [kpc] to keep halos')
    parser.add_argument('--min_mass', metavar='min_mass', type=float, default=1e10,
                        help='Minimum halo mass [Msun] to keep halos')
    parser.add_argument('--make_plots', dest='make_plots', action='store_true', default=True,
                        help='Make projection plots of each halo')
    parser.add_argument('--no_plots', dest='make_plots', action='store_false',
                        help='Skip making halo plots')
    parser.add_argument('--save_root', dest='save_root', action='store_true', default=False,
                        help='Save root particle indices after finding halos')
    parser.add_argument('--watch', dest='watch', action='store_true', default=False,
                        help='Run in watch mode: poll periodically for new outputs')
    parser.add_argument('--interval', metavar='interval', type=int, default=300,
                        help='Polling interval in seconds (watch mode only)')

    args = parser.parse_args()

    if args.watch:
        run_watch(args)
    else:
        run_once(args)
