"""Build gas ICs for a multizoom group from its existing DM per-halo configs.

The pipeline makes gas ICs by re-running MUSIC on the DM MUSIC config for the
same level with baryons switched on (pipeline.build.render_gas_music_config).
A merge-mode group already has one such config per halo, so the same recipe
applies per halo and the resulting gas IC sets merge exactly as the DM ones
do -- the merge tool handles GridDensity/GridVelocities generically and, with
baryons present GridDensity.0 is additionally checked outside the refinement
windows, which is the strongest available check that they share one
realization.

Usage::

    python3 gas_group.py --group sixpack --level 2 \
        --halos 48014,42502 [--out-group gaspair] [--dry-run]
"""

import argparse
import os
import shutil
import subprocess
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from foggie.initial_conditions.pipeline import config as pconfig
from foggie.initial_conditions.multizoom import config as mzconfig
from foggie.initial_conditions.multizoom import merge_music_ics
from foggie.initial_conditions.multizoom import mrp_music
from foggie.initial_conditions.multizoom import pipeline_integration as pi
from foggie.initial_conditions.multizoom import refinement_mask


def render_gas_conf(dm_conf, out_path, omega_b, ic_dir):
    """DM per-halo MUSIC config -> the gas one (baryons on, Omega_b set)."""
    out = []
    for line in open(dm_conf):
        key = line.split("=")[0].strip()
        if key == "baryons":
            line = "baryons = yes\n"
        elif key == "Omega_b":
            line = "Omega_b = %s\n" % omega_b
        elif key == "filename":
            line = "filename = %s\n" % ic_dir
        out.append(line)
    with open(out_path, "w") as fp:
        fp.write("".join(out))
    return out_path


def build_gas_group(group, level, halos, out_group=None, table=None,
                    dry_run=False):
    if table is None:
        table = pconfig.read_registry()
    halo_ids, box = pi.resolve_group(group, table, halos)
    src_dir = pi.group_dir(box, group)
    out_group = out_group or group
    dst_dir = pi.group_dir(box, out_group)
    os.makedirs(dst_dir, exist_ok=True)

    music_exe = os.path.join(
        os.environ.get("MULTIZOOM_MUSIC_EXE_DIR") or box.music_exe_dir_path(),
        "MUSIC")
    env = dict(os.environ)
    if os.environ.get("MULTIZOOM_MUSIC_LD_PATH"):
        env["LD_LIBRARY_PATH"] = os.environ["MULTIZOOM_MUSIC_LD_PATH"]
        env["DYLD_LIBRARY_PATH"] = os.environ["MULTIZOOM_MUSIC_LD_PATH"]

    run_dirs, conf_paths = [], []
    for h in halo_ids:
        dm_conf = os.path.join(src_dir, "%s-L%d-h%s.conf"
                               % (box.sim_name, level, h))
        if not os.path.exists(dm_conf):
            raise RuntimeError("missing DM config %s -- build the DM level "
                               "for this group first" % dm_conf)
        ic_dir = os.path.join(dst_dir, "%s-L%d-gas-h%s"
                              % (box.sim_name, level, h))
        conf = render_gas_conf(dm_conf, ic_dir + ".conf", box.omega_b, ic_dir)
        print("halo %s: %s" % (h, conf))
        # MUSIC refuses an existing output directory, and a 512^3 gas run is
        # expensive: reuse a complete IC set rather than rebuilding it.
        complete = os.path.exists(os.path.join(ic_dir, "parameter_file.txt"))
        if complete:
            print("  reusing existing gas ICs in %s" % ic_dir)
        if not dry_run and not complete:
            cwd = ic_dir + ".music"
            os.makedirs(cwd, exist_ok=True)
            subprocess.run([music_exe, os.path.abspath(conf)], env=env,
                           check=True, cwd=cwd)
            # the exact-Lagrangian mask, same as the DM path
            pts = os.path.join(src_dir, "initial_particle_positions-%s-%s.dat"
                               % (h, _initial_ds(src_dir, h)))
            if os.path.exists(pts):
                refinement_mask.particle_only_mask(conf, smooth_edges=True,
                                                   backup=True,
                                                   point_files=[pts])
        run_dirs.append(ic_dir)
        conf_paths.append(conf)

    if dry_run:
        return None
    merged = os.path.join(dst_dir, "%s-L%d-gas.merged" % (box.sim_name, level))
    manifest = merge_music_ics.merge_runs(run_dirs, merged,
                                          conf_paths=conf_paths,
                                          halo_ids=[str(h) for h in halo_ids])
    mrp_music.append_mrp_block(merged, level)
    final = os.path.join(dst_dir, "%s-L%d-gas" % (box.sim_name, level))
    if os.path.exists(final):
        raise RuntimeError("%s already exists" % final)
    os.rename(merged, final)
    print("MERGED gas ICs -> %s  (%d grids)" % (final, len(manifest["grids"])))
    return final


def _initial_ds(src_dir, halo_id):
    import glob
    pat = os.path.join(src_dir, "initial_particle_positions-%s-*.dat" % halo_id)
    hits = glob.glob(pat)
    if not hits:
        return ""
    base = os.path.basename(hits[0])
    return base.rsplit("-", 1)[1][:-4]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--group", required=True,
                   help="group whose DM per-halo configs are the source")
    p.add_argument("--level", type=int, required=True)
    p.add_argument("--halos", default=None,
                   help="subset of the group to build gas ICs for")
    p.add_argument("--out-group", default=None,
                   help="group directory to write into (default: --group)")
    p.add_argument("--registry", default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    table = pconfig.read_registry(args.registry)
    build_gas_group(args.group, args.level, pi.parse_halo_ids(args.halos),
                    args.out_group, table, args.dry_run)


if __name__ == "__main__":
    main()
