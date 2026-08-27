"""Drive multizoom IC builds from the ics_refactor pipeline.

The pipeline (foggie/initial_conditions/pipeline/) owns orchestration: the
halo registry, the Box definitions, staging, submission, QC and the ledger.
It still drives enzo-mrp-music as the workhorse that traces the Lagrangian
region and runs MUSIC.  Multizoom slots in at exactly that seam: instead of
one config naming one halo, it renders one config naming N halos, and hands
it to multizoom.mrp_music rather than enzo-mrp-music.py.

Nothing in pipeline/ is modified.  This module only reads from it, so a
multizoom build cannot disturb the production fleet.

Group membership comes from an optional `multizoom_group` column in the halo
registry: every enabled row sharing a non-empty value forms one group that is
built into a single domain.  Registries without the column simply have no
groups, and the pipeline behaves exactly as before.

Typical use::

    python -m multizoom.pipeline_integration groups
    python -m multizoom.pipeline_integration render  --group dwarfs --level 1
    python -m multizoom.pipeline_integration build   --group dwarfs --level 1 \
        [--mode union|merge] [--dry-run]
"""

import argparse
import os
import sys
from collections import OrderedDict

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from foggie.initial_conditions.pipeline import config as pconfig
from foggie.initial_conditions.pipeline import build as pbuild

GROUP_COLUMN = "multizoom_group"


# --------------------------------------------------------------------------
# group membership
# --------------------------------------------------------------------------

def registry_groups(table=None, box_name=None):
    """{group_name: [halo_id, ...]} from the registry's multizoom_group column.

    Returns an empty mapping when the column is absent, so this is safe to
    call against the production registry as it stands today.
    """
    if table is None:
        table = pconfig.read_registry()
    if GROUP_COLUMN not in table.colnames:
        return OrderedDict()
    groups = OrderedDict()
    for row in table:
        if not row["enabled"]:
            continue
        name = str(row[GROUP_COLUMN]).strip()
        if not name or name.lower() in ("--", "none", "n/a"):
            continue
        if box_name and str(row["box"]) != box_name:
            continue
        groups.setdefault(name, []).append(int(row["halo_id"]))
    return groups


def group_box(name, table=None):
    """The Box shared by a group; every member must sit in the same box."""
    if table is None:
        table = pconfig.read_registry()
    boxes = {str(r["box"]) for r in table
             if r["enabled"] and GROUP_COLUMN in table.colnames
             and str(r[GROUP_COLUMN]).strip() == name}
    if not boxes:
        raise RuntimeError("multizoom group %r has no enabled members" % name)
    if len(boxes) > 1:
        raise RuntimeError(
            "multizoom group %r spans several boxes (%s); a group must share "
            "one parent box and one noise realization" % (name, sorted(boxes)))
    return pconfig.get_box(boxes.pop())


def group_rvir_min(name, halo_id, table=None):
    if table is None:
        table = pconfig.read_registry()
    for row in table:
        if int(row["halo_id"]) == int(halo_id):
            return float(row["rvir_min"]) or None
    return None


def group_dir(box, name):
    """Where a group's configs and ICs live.

    Defaults to a multizoom_<name> directory beside the per-halo directories;
    MULTIZOOM_ICS_DIR overrides the root, so development builds can deposit
    everything in a private workspace while still reading the parent box from
    the shared ICs directory.
    """
    root = os.environ.get("MULTIZOOM_ICS_DIR") or pconfig.foggie_ics_dir()
    return os.path.join(root, "multizoom_%s" % name)


def group_config_name(name, level):
    return "multizoom_%s_DM_%dto%d.conf" % (name, level - 1, level)


def group_refine_center(box, halo_id, level, gdir, analytic_center,
                        search_kpc=500.0):
    """Locate the halo in the GROUP's previous merged run, at z = 0.

    The multizoom analogue of pipeline.build.refine_center_from_run, which
    hardwires the standalone per-halo directory layout.  In merge mode no
    level ever shifts the domain, so the analytic center is the catalog
    position at every level; this refines it to where the halo actually sits
    in the merged L(n-1) run.  Falls back to the analytic center rather than
    failing -- a diagnostic must never be the reason a build fails.
    """
    prev_dir = (os.path.join(pconfig.foggie_ics_dir(), "%s-L0" % box.sim_name)
                if level - 1 == 0 else
                os.path.join(gdir, "%s-L%d" % (box.sim_name, level - 1)))
    try:
        import numpy as np
        from foggie.initial_conditions.pipeline import qc as _qc
        snap, name, _, is_final = _qc.last_output(prev_dir)
        if snap is None or not is_final:
            print("    centre refinement skipped: %s has no final dump"
                  % os.path.basename(prev_dir))
            return analytic_center
        rel, mass, ds = _qc.load_particles(snap, analytic_center, search_kpc)
        if rel is None:
            print("    centre refinement skipped: no particles near the "
                  "analytic centre")
            return analytic_center
        offset = _qc.locate_halo(rel, mass, guess_radius_kpc=search_kpc,
                                 verbose=False)
        if offset is None:
            print("    centre refinement skipped: halo not located in %s"
                  % name)
            return analytic_center
        kpc_per_code = float(ds.quan(1.0, "code_length").in_units("kpc").d)
        centre = [(c + o / kpc_per_code) % 1.0
                  for c, o in zip(analytic_center, offset)]
        drift = float(np.sqrt((np.asarray(offset) ** 2).sum()))
        print("    halo %s centre refined using %s: %.0f kpc from analytic"
              % (halo_id, name, drift))
        return centre
    except Exception as exc:
        print("    centre refinement skipped (%s); using the analytic centre"
              % exc)
        return analytic_center


# --------------------------------------------------------------------------
# config rendering
# --------------------------------------------------------------------------

def render_group_config(box, name, halo_ids, level, table=None, mode="union"):
    """Render the N-halo multizoom config for one DM level.

    The multizoom analogue of pipeline.build.render_mrp_config.  Per-halo
    centers come from the pipeline's own center_for_level, so a group member
    is traced exactly as it would be as a standalone zoom -- including the
    level >= 2 refine-from-the-run correction.
    """
    if table is None:
        table = pconfig.read_registry()
    gdir = group_dir(box, name)
    ics = pconfig.foggie_ics_dir()

    lines = [
        "# Multizoom config rendered by multizoom.pipeline_integration.",
        "# Group %r, level %d, %d halos, mode %s." % (name, level,
                                                      len(halo_ids), mode),
        "# Do not edit by hand: re-render from the registry instead.",
        "",
        "[setup]",
        "music_exe_dir = %s" % box.music_exe_dir_path(),
        "simulation_name = %s" % box.sim_name,
        "template_config = %s" % os.path.join(ics, box.template_config),
        "original_config = %s" % os.path.join(ics, box.template_config),
        "# Parent level is read from the shared ICs dir; this level's ICs are",
        "# written into the group directory.",
        "simulation_run_directory = %s" % (ics if level == 1 else gdir),
        "new_ics_directory = %s" % gdir,
        "num_cores = None",
        "mode = %s" % mode,
        "",
        "[region]",
        "final_type = halo",
        "final_redshift = 0.0",
        "halo_center_units = code_length",
        "halo_radius_units = kpc",
        "radius_factor = 1.0",
        "# exact is required for multi-halo: a single convex hull spanning",
        "# several clouds would refine the volume between the halos.",
        "shape_type = exact",
        "",
    ]

    for halo_id in halo_ids:
        rvir_min = group_rvir_min(name, halo_id, table)
        # Merge mode never shifts the domain, so the analytic center is the
        # catalog position at every level; refine it against the group's own
        # previous merged run (there is no standalone ladder to consult).
        center, rvir = pbuild.halo_center_and_radius(box, halo_id, rvir_min)
        if level >= 2 and getattr(box, "refine_centers", True):
            center = group_refine_center(box, halo_id, level, gdir, center)
        lines += [
            "[halo:%s]" % halo_id,
            "halo_center = %s , %s , %s" % tuple(repr(float(c)) for c in center),
            "halo_radius = %s" % rvir,
            "",
        ]
    return "\n".join(lines) + "\n"


def write_group_config(box, name, halo_ids, level, table=None, mode="union",
                       dry_run=False):
    gdir = group_dir(box, name)
    path = os.path.join(gdir, group_config_name(name, level))
    text = render_group_config(box, name, halo_ids, level, table, mode)
    if dry_run:
        print(text)
        return path
    os.makedirs(gdir, exist_ok=True)
    with open(path, "w") as fp:
        fp.write(text)
    return path


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------

def build_group(name, level, mode="union", table=None, dry_run=False):
    """Render the group config and run the multizoom IC build for one level."""
    if table is None:
        table = pconfig.read_registry()
    groups = registry_groups(table)
    if name not in groups:
        raise RuntimeError("no multizoom group %r in the registry (found: %s)"
                           % (name, sorted(groups) or "none"))
    box = group_box(name, table)
    halo_ids = groups[name]
    path = write_group_config(box, name, halo_ids, level, table, mode, dry_run)
    print("group %r: %d halos %s -> %s" % (name, len(halo_ids), halo_ids, path))
    if dry_run:
        return path

    from . import mrp_music
    params = mrp_music.startup(path, level)
    params = mrp_music.get_previous_run_params(params)
    params = mrp_music.find_lagrangian_regions(params)
    return mrp_music.run_level(params)


RUNSCRIPT = """#!/bin/bash
#
# Multizoom group Enzo run, rendered by multizoom.pipeline_integration.
#
#PBS -N mz-{name}-L{level}
#PBS -W group_list=s3128
#PBS -l select=1:ncpus=64:mpiprocs=64:model=mil_ait
#PBS -l walltime=24:00:00
#PBS -q long
#PBS -j oe
#PBS -m abe
#PBS -V
#PBS -e pbs_error.txt
#PBS -o pbs_output.txt

module load comp-intel/2020.4.304
module load hdf5/1.8.18_serial

export HDF5_DISABLE_VERSION_CHECK=1
export LD_LIBRARY_PATH="/u/jtumlins/installs/mpich-4.0.3/usr/local/lib":"/u/jtumlins/installs/mpich-4.0.3/usr/lib":"/u/jtumlins/grackle/grackle-3.3.1-dev/build/lib64":"/u/jtumlins/installs/compat_gfortran3":$LD_LIBRARY_PATH
export PATH="/nobackup/jtumlins/anaconda3/bin:/u/scicon/tools/bin/:/u/jtumlins/installs/mpich-4.0.3/usr/local/bin:$PATH"

cd $PBS_O_WORKDIR
/u/jtumlins/installs/memory_gauge.sh $PBS_JOBID > memory.$PBS_JOBID 2>&1 &

./simrun.pl -mpi "mpiexec -np 64 /u/scicon/tools/bin/mbind.x -cs " \
            -wall 86400 \
            -exe "{enzo_exe}" \
            -pf "{sim_name}-L{level}.enzo" \
            -jf "RunScript.sh"
mv pbs_output.txt pbs_output_$PBS_JOBID.txt
"""


def assemble_group_run(name, level, enzo_exe=None, table=None):
    """Fill the Enzo parameter file and job script into a merged IC dir.

    Uses the pipeline's own .enzo renderer, then corrects
    CosmologySimulationNumberOfInitialGrids: the pipeline keyword assumes the
    single-pyramid layout (level+1 grids), while a merged multizoom directory
    holds one base grid plus one patch per halo per level.
    """
    import re as _re
    box = group_box(name, table)
    gdir = group_dir(box, name)
    ics_dir = os.path.join(gdir, "%s-L%d" % (box.sim_name, level))
    pf = os.path.join(ics_dir, "parameter_file.txt")
    grid_parameters = pbuild.read_grid_parameters(pf)
    n_grids = None
    for line in open(pf):
        m = _re.match(r"CosmologySimulationNumberOfInitialGrids\s*=\s*(\d+)",
                      line)
        if m:
            n_grids = int(m.group(1))
    if n_grids is None:
        raise RuntimeError("NumberOfInitialGrids not found in %s" % pf)

    text = pbuild.render_enzo_param(box, level, "DM", grid_parameters)
    text = _re.sub(r"CosmologySimulationNumberOfInitialGrids\s*=\s*\d+",
                   "CosmologySimulationNumberOfInitialGrids  = %d" % n_grids,
                   text)
    enzo_fn = os.path.join(ics_dir, "%s-L%d.enzo" % (box.sim_name, level))
    with open(enzo_fn, "w") as fp:
        fp.write(text)

    enzo_exe = enzo_exe or os.environ.get("MULTIZOOM_ENZO_EXE")
    if not enzo_exe or not os.path.exists(enzo_exe):
        raise RuntimeError("set MULTIZOOM_ENZO_EXE (or --enzo-exe) to the "
                           "patched enzo.exe; merged ICs need it")
    with open(os.path.join(ics_dir, "RunScript.sh"), "w") as fp:
        fp.write(RUNSCRIPT.format(name=name, level=level,
                                  sim_name=box.sim_name, enzo_exe=enzo_exe))
    import shutil, stat
    simrun_src = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "templates", "simrun.pl")
    simrun_dst = os.path.join(ics_dir, "simrun.pl")
    shutil.copyfile(simrun_src, simrun_dst)
    for fn in (simrun_dst, os.path.join(ics_dir, "RunScript.sh")):
        os.chmod(fn, os.stat(fn).st_mode | stat.S_IXUSR)
    print("assembled %s (NumberOfInitialGrids = %d)" % (ics_dir, n_grids))
    return ics_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    registry_help = ("alternate halo registry (default: the pipeline's "
                     "production registry)")
    g = sub.add_parser("groups", help="list multizoom groups in the registry")
    g.add_argument("--registry", default=None, help=registry_help)
    for cmd in ("render", "build", "assemble"):
        p = sub.add_parser(cmd)
        p.add_argument("--group", required=True)
        p.add_argument("--level", type=int, required=True)
        p.add_argument("--mode", choices=("union", "merge"), default="union")
        p.add_argument("--dry-run", action="store_true")
        p.add_argument("--registry", default=None, help=registry_help)
        p.add_argument("--enzo-exe", default=None)
    args = parser.parse_args(argv)

    table = pconfig.read_registry(args.registry)
    if args.cmd == "groups":
        groups = registry_groups(table)
        if not groups:
            print("no multizoom groups: registry has no %r column, or no "
                  "enabled row sets it" % GROUP_COLUMN)
            return
        for name, ids in groups.items():
            print("%-16s %2d halos  box=%s  %s"
                  % (name, len(ids), group_box(name, table).sim_name, ids))
        return
    if args.cmd == "render":
        box = group_box(args.group, table)
        write_group_config(box, args.group, registry_groups(table)[args.group],
                           args.level, table, args.mode, dry_run=True)
        return
    if args.cmd == "assemble":
        assemble_group_run(args.group, args.level, args.enzo_exe, table)
        return
    build_group(args.group, args.level, args.mode, table, args.dry_run)


if __name__ == "__main__":
    main()
