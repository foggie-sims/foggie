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
    """Where a group's configs and ICs live, beside the per-halo directories."""
    return os.path.join(pconfig.foggie_ics_dir(), "multizoom_%s" % name)


def group_config_name(name, level):
    return "multizoom_%s_DM_%dto%d.conf" % (name, level - 1, level)


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
        halo_dir = os.path.join(ics, "halo%s" % halo_id)
        center = pbuild.center_for_level(box, halo_id, level, halo_dir,
                                         rvir_min)
        if level >= 2 and getattr(box, "refine_centers", True):
            center, _ = pbuild.refine_center_from_run(box, halo_id, level,
                                                      halo_dir, center)
        _, rvir = pbuild.halo_center_and_radius(box, halo_id, rvir_min)
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("groups", help="list multizoom groups in the registry")
    for cmd in ("render", "build"):
        p = sub.add_parser(cmd)
        p.add_argument("--group", required=True)
        p.add_argument("--level", type=int, required=True)
        p.add_argument("--mode", choices=("union", "merge"), default="union")
        p.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    table = pconfig.read_registry()
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
    build_group(args.group, args.level, args.mode, table, args.dry_run)


if __name__ == "__main__":
    main()
