"""Run the pipeline's QC figures on a multizoom group's merged runs.

pipeline.qc resolves everything through two Box methods -- halo_dir (for
the conf logs that carry each level's domain shift) and stage_dir (the run
directory) -- so a proxy Box that points both at the group is enough to
reuse the fleet's density, contamination and neighbour figures unchanged.

Every level of a merge-mode group shares one domain shift; the group's
per-halo MUSIC conf logs record it, and are exposed to qc under the
standalone file names via symlinks in a per-halo QC directory.

Usage::

    python -m foggie.initial_conditions.multizoom.group_qc \
        --group sixpack --registry runs/halo_registry_multizoom.ecsv \
        [--max-level 2] [--neighbors-level 2] [--halos 48014,...]
"""

import argparse
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from foggie.initial_conditions.pipeline import config as pconfig
from foggie.initial_conditions.pipeline import qc
from foggie.initial_conditions.multizoom import pipeline_integration as pi


class GroupBox(object):
    """Box proxy: stage dirs and conf logs come from the group directory."""

    def __init__(self, box, gdir, max_level=None):
        self._box = box
        self._gdir = gdir
        self._max_level = max_level

    def __getattr__(self, name):
        return getattr(self._box, name)

    @property
    def max_level(self):
        return self._max_level or self._box.max_level

    def halo_dir(self, halo_id):
        """A per-halo QC directory whose conf logs are the group's."""
        d = os.path.join(self._gdir, "qc", "halo%s" % halo_id)
        os.makedirs(d, exist_ok=True)
        for lev in range(1, self._box.max_level + 1):
            src = os.path.join(self._gdir, "%s-L%d-h%s.conf_log.txt"
                               % (self._box.sim_name, lev, halo_id))
            dst = os.path.join(d, "%s-L%d.conf_log.txt"
                               % (self._box.sim_name, lev))
            if os.path.exists(src) and not os.path.lexists(dst):
                os.symlink(src, dst)
        return d

    def stage_dir(self, halo_id, level, phase="DM"):
        suffix = "" if phase == "DM" else "-gas"
        return os.path.join(self._gdir, "%s-L%d%s"
                            % (self._box.sim_name, level, suffix))


def run_group_qc(name, table, max_level=None, neighbors_level=None,
                 halos=None):
    ids, box = pi.resolve_group(name, table, halos)
    gdir = pi.group_dir(box, name)
    gbox = GroupBox(box, gdir, max_level)
    out_dir = os.path.join(gdir, "qc")
    os.makedirs(out_dir, exist_ok=True)
    for halo_id in ids:
        rvir_min = pi.group_rvir_min(name, halo_id, table)
        print("=== halo %s ===" % halo_id)
        for recenter in (False, True):
            out = os.path.join(out_dir, "qc_density_halo%s%s.png"
                               % (halo_id, "_recentered" if recenter else ""))
            try:
                qc.make_density_figure(gbox, halo_id, out_path=out,
                                       recenter=recenter)
                print("  wrote", out)
            except Exception as exc:
                print("  density figure failed (%s): %s" % (recenter, exc))
        out = os.path.join(out_dir, "qc_contamination_halo%s.png" % halo_id)
        try:
            qc.make_qc_figure(gbox, halo_id, out_path=out, rvir_min=rvir_min)
            print("  wrote", out)
        except Exception as exc:
            print("  contamination figure failed: %s" % exc)
        lev = neighbors_level or gbox.max_level
        out = os.path.join(out_dir, "qc_neighbors_halo%s_L%dDM.png"
                           % (halo_id, lev))
        try:
            qc.make_neighbor_projection(gbox, halo_id, level=lev, phase="DM",
                                        out_path=out, rvir_min=rvir_min)
            print("  wrote", out)
        except Exception as exc:
            print("  neighbour projection failed: %s" % exc)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--group", required=True)
    p.add_argument("--registry", default=None)
    p.add_argument("--max-level", type=int, default=None)
    p.add_argument("--neighbors-level", type=int, default=None)
    p.add_argument("--halos", default=None,
                   help="comma-separated halo IDs defining the group ad hoc "
                        "(overrides the registry column)")
    args = p.parse_args(argv)
    table = pconfig.read_registry(args.registry)
    halos = pi.parse_halo_ids(args.halos)
    run_group_qc(args.group, table, args.max_level, args.neighbors_level,
                 halos)


if __name__ == "__main__":
    main()
