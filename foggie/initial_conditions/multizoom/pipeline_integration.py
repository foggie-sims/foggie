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


def parse_halo_ids(spec):
    """"48014,56672,..." -> [48014, 56672, ...]; None/empty -> None."""
    if not spec:
        return None
    ids = [int(v) for v in str(spec).replace(" ", "").split(",") if v]
    if not ids:
        return None
    seen, out = set(), []
    for h in ids:
        if h in seen:
            raise RuntimeError("halo %s listed twice" % h)
        seen.add(h)
        out.append(h)
    if len(out) < 2:
        raise RuntimeError("a multizoom group needs at least two halos "
                           "(got %s)" % out)
    return out


def resolve_group(name, table=None, halos=None):
    """(halo_ids, box) for a group, from an explicit list or the registry.

    A group can be named two ways.  Passing `halos` defines it ad hoc, so a
    run needs no registry edit and the same tooling serves a different set
    every time; `name` is then just the label the directory takes.  With no
    `halos`, membership comes from the registry's multizoom_group column.

    Either way every member must be an enabled row of the registry sharing
    one parent box -- a group is one noise realization, and rvir_min and the
    catalog position still come from that row.
    """
    if table is None:
        table = pconfig.read_registry()
    rows = {int(r["halo_id"]): r for r in table}

    if halos:
        halo_ids = list(halos)
        missing = [h for h in halo_ids if h not in rows]
        if missing:
            raise RuntimeError("halo(s) %s are not in the registry" % missing)
        disabled = [h for h in halo_ids if not rows[h]["enabled"]]
        if disabled:
            raise RuntimeError(
                "halo(s) %s are disabled in the registry; enable them or drop "
                "them from the group" % disabled)
    else:
        groups = registry_groups(table)
        if name not in groups:
            raise RuntimeError(
                "no multizoom group %r in the registry (found: %s).  Pass "
                "--halos to define a group without editing the registry."
                % (name, sorted(groups) or "none"))
        halo_ids = groups[name]

    boxes = {str(rows[h]["box"]) for h in halo_ids}
    if len(boxes) > 1:
        raise RuntimeError(
            "multizoom group %r spans several boxes (%s); a group must share "
            "one parent box and one noise realization" % (name, sorted(boxes)))
    return halo_ids, pconfig.get_box(boxes.pop())


def group_box(name, table=None, halos=None):
    """The Box shared by a group; every member must sit in the same box."""
    return resolve_group(name, table, halos)[1]


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

    music_exe_dir = os.environ.get("MULTIZOOM_MUSIC_EXE_DIR") or \
        box.music_exe_dir_path()
    shift_override = os.environ.get("MULTIZOOM_SHIFT_OVERRIDE", "").strip()

    lines = [
        "# Multizoom config rendered by multizoom.pipeline_integration.",
        "# Group %r, level %d, %d halos, mode %s." % (name, level,
                                                      len(halo_ids), mode),
        "# Do not edit by hand: re-render from the registry instead.",
        "",
        "[setup]",
        "music_exe_dir = %s" % music_exe_dir,
        "simulation_name = %s" % box.sim_name,
        "template_config = %s" % os.path.join(ics, box.template_config),
        "original_config = %s" % os.path.join(ics, box.template_config),
        "# Parent level is read from the shared ICs dir; this level's ICs are",
        "# written into the group directory.",
        "simulation_run_directory = %s" % (ics if level == 1 else gdir),
        "new_ics_directory = %s" % gdir,
        "num_cores = None",
        "mode = %s" % mode,
    ]
    if shift_override:
        lines += ["region_shift_override = %s" % shift_override]
    music_ld = os.environ.get("MULTIZOOM_MUSIC_LD_PATH", "").strip()
    if music_ld:
        lines += ["music_ld_library_path = %s" % music_ld]
    lines += [
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
        if level >= 2:
            if shift_override:
                sh = [int(v) for v in shift_override.split(",")]
                ncoarse = float(box.parent_ngrid)
                center = [(c + v / ncoarse) % 1.0 for c, v in zip(center, sh)]
            if getattr(box, "refine_centers", True):
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

def build_group(name, level, mode="union", table=None, dry_run=False,
                halos=None):
    """Render the group config and run the multizoom IC build for one level."""
    if table is None:
        table = pconfig.read_registry()
    halo_ids, box = resolve_group(name, table, halos)
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


def assemble_group_run(name, level, enzo_exe=None, table=None, halos=None,
                       phase="DM", walltime=None, nranks=None):
    """Fill the Enzo parameter file and job script into a merged IC dir.

    Uses the pipeline's own .enzo renderer, then corrects
    CosmologySimulationNumberOfInitialGrids: the pipeline keyword assumes the
    single-pyramid layout (level+1 grids), while a merged multizoom directory
    holds one base grid plus one patch per halo per level.
    """
    import re
    _re = re
    box = group_box(name, table, halos)
    gdir = group_dir(box, name)
    suffix = "" if phase == "DM" else "-gas"
    ics_dir = os.path.join(gdir, "%s-L%d%s" % (box.sim_name, level, suffix))
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

    text = pbuild.render_enzo_param(box, level, phase, grid_parameters)
    text = _re.sub(r"CosmologySimulationNumberOfInitialGrids\s*=\s*\d+",
                   "CosmologySimulationNumberOfInitialGrids  = %d" % n_grids,
                   text)
    enzo_fn = os.path.join(ics_dir, "%s-L%d%s.enzo"
                           % (box.sim_name, level, suffix))
    with open(enzo_fn, "w") as fp:
        fp.write(text)

    enzo_exe = enzo_exe or os.environ.get("MULTIZOOM_ENZO_EXE")
    if not enzo_exe or not os.path.exists(enzo_exe):
        raise RuntimeError("set MULTIZOOM_ENZO_EXE (or --enzo-exe) to the "
                           "patched enzo.exe; merged ICs need it")
    # The gas stage runs the pipeline's own two-leg script (unshielded above
    # the transition redshift, shielded below), so a multizoom gas run is the
    # same physics as the standalone runs it is compared against.
    if phase == "gas":
        script = pbuild.render_runscript(box, name, level, "gas")
        script = script.replace(box.enzo_exe, enzo_exe)
        script = script.replace("halo%s-L%d-gas" % (name, level),
                                "mz-%s-L%d-gas" % (name, level))
        if walltime:
            script = re.sub(r"#PBS -l walltime=\S+",
                            "#PBS -l walltime=%s" % walltime, script)
        if nranks:
            script = re.sub(r"select=1:ncpus=\d+:mpiprocs=\d+",
                            "select=1:ncpus=%d:mpiprocs=%d" % (nranks, nranks),
                            script)
    else:
        script = RUNSCRIPT.format(name=name, level=level,
                                  sim_name=box.sim_name, enzo_exe=enzo_exe)
    with open(os.path.join(ics_dir, "RunScript.sh"), "w") as fp:
        fp.write(script)
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
    halos_help = ("comma-separated halo IDs defining the group ad hoc, e.g. "
                  "--halos 48014,56672,75392,21246,24122,42502.  Overrides "
                  "the registry's %s column, so a different set every run "
                  "needs no registry edit; --group is then just the label "
                  "the directory takes." % GROUP_COLUMN)
    for cmd in ("render", "build", "assemble"):
        p = sub.add_parser(cmd)
        p.add_argument("--group", required=True)
        p.add_argument("--level", type=int, required=True)
        p.add_argument("--halos", default=None, help=halos_help)
        p.add_argument("--mode", choices=("union", "merge"), default="union")
        p.add_argument("--dry-run", action="store_true")
        p.add_argument("--registry", default=None, help=registry_help)
        p.add_argument("--enzo-exe", default=None)
        p.add_argument("--phase", choices=("DM", "gas"), default="DM")
        p.add_argument("--walltime", default=None,
                       help="override the PBS walltime (gas assembly)")
        p.add_argument("--nranks", type=int, default=None,
                       help="override the rank count (gas assembly)")
    args = parser.parse_args(argv)
    halos = parse_halo_ids(getattr(args, "halos", None))

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
        halo_ids, box = resolve_group(args.group, table, halos)
        write_group_config(box, args.group, halo_ids, args.level, table,
                           args.mode, dry_run=True)
        return
    if args.cmd == "assemble":
        assemble_group_run(args.group, args.level, args.enzo_exe, table, halos,
                           phase=args.phase, walltime=args.walltime,
                           nranks=args.nranks)
        return
    build_group(args.group, args.level, args.mode, table, args.dry_run, halos)


if __name__ == "__main__":
    main()
