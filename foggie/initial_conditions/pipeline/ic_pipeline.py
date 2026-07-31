#!/usr/bin/env python3
"""
IC pipeline driver for the 25 Mpc zoom simulations.

Replaces the hand-driven script512.py / script256.py workflow.  See
foggie/initial_conditions/REFACTOR_PLAN.md for the design, and
refactor_roadmap.html for the diagrams.

Subcommands implemented so far:

    validate-templates   prove the collapsed .enzo templates re-render the
                         per-level files they replaced
    validate-registry    check the halo registry parses and resolves

Usage:
    python -m foggie.initial_conditions.pipeline.ic_pipeline validate-templates
"""

import argparse
import difflib
import os
import subprocess
import sys

try:
    from . import build
    from . import config
    from . import ledger
    from . import report
    from . import state as stagestate
except ImportError:
    # Allow running this file directly by path:
    #     python foggie/initial_conditions/pipeline/ic_pipeline.py status
    # which is the preferred form, because importing the `foggie` package pulls
    # in yt via foggie/__init__.py and nothing here needs it.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import build
    import config
    import ledger
    import report
    import state as stagestate


# ---------------------------------------------------------------------------
# Reference lookup
# ---------------------------------------------------------------------------

# The per-level files the collapsed templates replace.  They are read from git
# so the check keeps working after the old template directory is deleted.
_REFERENCE_DIR = "foggie/initial_conditions/halo_template_512"

_REFERENCE_STAGES = [
    (1, "DM", "25Mpc_DM_512-L1.enzo"),
    (2, "DM", "25Mpc_DM_512-L2.enzo"),
    (3, "DM", "25Mpc_DM_512-L3.enzo"),
    (3, "gas", "25Mpc_DM_512-L3-gas.enzo"),
]


def _repo_root():
    """Top of the git working tree, or None."""
    try:
        out = subprocess.check_output(["git", "rev-parse", "--show-toplevel"],
                                      cwd=os.path.dirname(os.path.abspath(__file__)),
                                      stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return None


def read_reference(filename, git_ref):
    """Read an original per-level .enzo file, preferring git over the worktree."""
    root = _repo_root()
    relpath = "%s/%s" % (_REFERENCE_DIR, filename)
    if root and git_ref:
        try:
            out = subprocess.check_output(["git", "show", "%s:%s" % (git_ref, relpath)],
                                          cwd=root, stderr=subprocess.DEVNULL)
            return out.decode(), "%s:%s" % (git_ref, relpath)
        except subprocess.CalledProcessError:
            pass
    path = os.path.join(root or ".", relpath)
    if os.path.exists(path):
        with open(path) as fp:
            return fp.read(), path
    raise RuntimeError(
        "Cannot find reference %s -- not in git ref %r and not at %s" % (filename, git_ref, path))


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def _canonical_value(value):
    """Canonicalise a parameter value so formatting differences do not matter.

    Numeric values are compared as numbers, not text: the hand-maintained files
    wrote the same Hubble constant as both `0.700` and `0.7`.  Comparing
    numerically means those agree while a genuine change (0.291 -> 0.29) still
    fails.  Non-numeric values are compared verbatim.
    """
    tokens = value.split()
    if not tokens:
        return value
    try:
        return " ".join(repr(float(t)) for t in tokens)
    except ValueError:
        return " ".join(tokens)


def normalize(text):
    """Reduce an Enzo parameter file to comparable content.

    Drops comments and blank lines, collapses whitespace, and compares numeric
    values numerically, so that cosmetic differences between the
    hand-maintained per-level files (header wording, `0.700` vs `0.7`, trailing
    spaces) do not mask a real difference in a parameter value.
    """
    lines = []
    for raw in text.splitlines():
        line = raw.split("//")[0]
        line = line.split("#")[0]
        line = " ".join(line.split())
        if not line:
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            line = "%s = %s" % (key.strip(), _canonical_value(value.strip()))
        lines.append(line)
    return lines


def compare_stage(box, level, phase, filename, git_ref):
    """Render one stage and diff it against the original.  Returns a diff list."""
    rendered = build.render_enzo_param(box, level, phase, grid_parameters="")
    original, source = read_reference(filename, git_ref)
    diff = list(difflib.unified_diff(normalize(original), normalize(rendered),
                                     fromfile="original  %s" % source,
                                     tofile="rendered  L%d-%s" % (level, phase),
                                     lineterm="", n=1))
    return diff, source


def cmd_validate_templates(args):
    box = config.get_box(args.box)
    print("Validating collapsed templates for box %s" % box.sim_name)
    print("Template dir: %s\n" % box.template_dir_path())

    failures = 0
    for level, phase, filename in _REFERENCE_STAGES:
        label = "L%d-%s" % (level, phase)
        try:
            diff, source = compare_stage(box, level, phase, filename, args.git_ref)
        except Exception as exc:
            print("  %-8s ERROR  %s" % (label, exc))
            failures += 1
            continue

        if diff:
            print("  %-8s DIFFERS from %s" % (label, source))
            for line in diff:
                print("      %s" % line)
            failures += 1
        else:
            print("  %-8s matches %s" % (label, source))

    print("")
    if failures:
        print("FAILED: %d of %d stage(s) differ." % (failures, len(_REFERENCE_STAGES)))
        print("The template collapse is not faithful; do not switch over.")
        return 1
    print("OK: all %d stages re-render to their originals "
          "(ignoring comments and whitespace)." % len(_REFERENCE_STAGES))
    return 0


def collect_registry_records(table, include_gas=False, qstat=None):
    """Stage records for every enabled registry halo, in dependency order."""
    qstat = stagestate.qstat_states() if qstat is None else qstat
    records = []
    for row in config.enabled_halos(table):
        box = config.get_box(row["box"])
        halo_id = row["halo_id"]
        halo_dir = box.halo_dir(halo_id)
        prereq_done = True
        for level, phase in config.stage_plan(row, include_gas=include_gas):
            stage_dir = box.stage_dir(halo_id, level, phase)
            jobid, job_state, action = ledger.live_job(halo_dir, level, phase, qstat)
            if action == "build" and job_state in (stagestate.QUEUED, stagestate.RUNNING):
                job_state = stagestate.BUILDING
            st = stagestate.stage_state(stage_dir, job_state=job_state,
                                        prereq_done=prereq_done)
            records.append({"halo": str(halo_id), "box": box.sim_name,
                            "stage": "L%d-%s" % (level, phase), "state": st,
                            "level": level, "phase": phase,
                            "jobid": jobid, "frozen": False,
                            "stage_dir": stage_dir,
                            "enabled": bool(row["enabled"]),
                            "final_level": int(row["final_level"]),
                            "gas": bool(row["gas"]),
                            "rvir_min": (float(row["rvir_min"])
                                         if "rvir_min" in row.colnames else None)})
            prereq_done = prereq_done and st.state == stagestate.DONE
    return records


def collect_frozen_records(table):
    """Stage records for halo directories the pipeline does not manage.

    These are the hand-built runs, including the *-manual copies.  They are
    read-only ground truth: reporting them correctly without touching them is
    what validates the state detector.
    """
    ics_dir = config.foggie_ics_dir()
    managed = {"halo%s" % row["halo_id"] for row in table}
    records = []
    for entry in sorted(os.listdir(ics_dir)):
        if not entry.startswith("halo") or entry in managed:
            continue
        halo_dir = os.path.join(ics_dir, entry)
        if not os.path.isdir(halo_dir):
            continue
        for box in config.BOXES.values():
            try:
                stages = stagestate.discover_stage_dirs(halo_dir, box.sim_name)
            except OSError:
                continue
            for level, suffix, stage_dir in stages:
                st = stagestate.stage_state(stage_dir, job_state=None, prereq_done=True)
                records.append({"halo": entry.replace("halo", "", 1), "box": box.sim_name,
                                "stage": "L%d%s" % (level, suffix or "-DM"), "state": st,
                                "level": level, "phase": (suffix or "-DM").lstrip("-"),
                                "jobid": None, "frozen": True,
                                "stage_dir": stage_dir})
    return records


def cmd_status(args):
    table = config.read_registry(args.registry)
    records = collect_registry_records(table, include_gas=args.include_gas)
    if args.include_manual:
        records += collect_frozen_records(table)

    rows = report.to_rows(records)
    print(report.render_text(rows))
    print("")
    print(report.summarize(rows))

    if args.by_halo:
        print("")
        print(report.render_text(report.to_halo_rows(rows), report.HALO_COLUMNS))

    if args.write:
        # Defaults next to the simulations rather than into the repo: this is
        # regenerated on every sweep, so versioning it would churn history and
        # collide with a poller running unattended.  --out-dir overrides.
        ics_dir = args.out_dir or config.foggie_ics_dir()
        os.makedirs(ics_dir, exist_ok=True)
        stage_ecsv = os.path.join(ics_dir, "status.ecsv")
        halo_ecsv = os.path.join(ics_dir, "status_by_halo.ecsv")
        htm = os.path.join(ics_dir, "status.html")
        report.write_ecsv(rows, stage_ecsv)
        report.write_ecsv(report.to_halo_rows(rows), halo_ecsv,
                          columns=report.HALO_COLUMNS,
                          extra_comments=["One row per halo; see status.ecsv for per-stage detail."])
        report.write_html(rows, htm)
        print("\nWrote %s   (one row per stage)" % stage_ecsv)
        print("      %s   (one row per halo)" % halo_ecsv)
        print("      %s" % htm)
    return 0


def advance_halo(row, qstat, include_gas=False, dry_run=False, verbose=True):
    """Take at most one action for one halo, then stop.

    Walks the stage ladder in order and acts on the first stage that is not
    DONE.  The ladder is strictly sequential -- level N's ICs are traced from
    level N-1's Enzo outputs -- so there is never more than one actionable
    stage, and anything other than READY or BUILT means do nothing:

        READY    prerequisite done, no ICs yet  -> submit the IC build job
        BUILT    ICs exist, Enzo never started  -> submit the Enzo run
        DONE     finished                       -> move on to the next stage
        BUILDING/QUEUED/RUNNING                 -> already in the queue
        BLOCKED                                 -> waiting on the level below
        STALLED  produced output, then stopped  -> wants a human, never retried

    Returns a short description of what it did, or None.
    """
    box = config.get_box(row["box"])
    halo_id = row["halo_id"]
    halo_dir = box.halo_dir(halo_id)

    prereq_done = True
    for level, phase in config.stage_plan(row, include_gas=include_gas):
        stage_dir = box.stage_dir(halo_id, level, phase)
        jobid, job_state, action = ledger.live_job(halo_dir, level, phase, qstat)
        if action == "build" and job_state in (stagestate.QUEUED, stagestate.RUNNING):
            job_state = stagestate.BUILDING
        st = stagestate.stage_state(stage_dir, job_state=job_state, prereq_done=prereq_done)
        key = ledger.stage_key(level, phase)

        if st.state == stagestate.DONE:
            continue

        if st.state == stagestate.READY:
            if verbose:
                print("halo %s %s is READY -- submitting IC build" % (halo_id, key))
            build.submit_build_job(box, halo_id, level, phase, dry_run=dry_run)
            return "%s %s submitted IC build" % (halo_id, key)

        if st.state == stagestate.BUILT:
            if verbose:
                print("halo %s %s is BUILT -- submitting Enzo run" % (halo_id, key))
            build.submit_enzo_run(box, halo_id, level, phase, dry_run=dry_run)
            return "%s %s submitted Enzo run" % (halo_id, key)

        if verbose:
            print("halo %s %s is %s%s -- nothing to do"
                  % (halo_id, key, st.state, " (%s)" % st.note if st.note else ""))
        return None

    if verbose:
        print("halo %s: all stages DONE" % halo_id)
    return None


def cmd_advance(args):
    """Submit the next actionable stage, for one halo or for all of them.

    Called two ways, both of which must be safe at any moment: from the tail of
    a generated RunScript when an Enzo job exits, and from the poller.  Safety
    comes from re-deriving state from disk every time, holding a per-halo lock,
    and cross-checking the ledger against qstat before submitting anything.
    """
    table = config.read_registry(args.registry)
    rows = config.enabled_halos(table)
    if args.halo is not None:
        rows = [r for r in rows if int(r["halo_id"]) == int(args.halo)]
        if not rows:
            print("halo %s is not an enabled registry row; doing nothing" % args.halo)
            return 0

    qstat = stagestate.qstat_states()
    actions = []
    for row in rows:
        halo_dir = config.get_box(row["box"]).halo_dir(row["halo_id"])
        try:
            if args.dry_run:
                did = advance_halo(row, qstat, args.include_gas, dry_run=True)
            else:
                # The lock is what makes the RunScript hook and the poller safe
                # to fire at the same instant.
                with ledger.halo_lock(halo_dir):
                    did = advance_halo(row, qstat, args.include_gas, dry_run=False)
        except ledger.UnmanagedHaloError as exc:
            print("halo %s REFUSED: %s" % (row["halo_id"], exc))
            continue
        except Exception as exc:
            print("halo %s ERROR: %s" % (row["halo_id"], exc))
            continue
        if did:
            actions.append(did)

    print("\nadvance: %s" % ("; ".join(actions) if actions else "no action taken"))
    return 0


def cmd_build(args):
    """Generate ICs and the run script for one stage, and submit it."""
    table = config.read_registry(args.registry)
    match = [r for r in table if int(r["halo_id"]) == int(args.halo)]
    if not match:
        print("halo %s is not in the registry (%s).\n"
              "Add it there rather than passing it ad hoc -- the registry is the "
              "only list of halos the pipeline acts on."
              % (args.halo, args.registry or config.default_registry_path()))
        return 1
    row = match[0]
    box = config.get_box(args.box or row["box"])

    try:
        with ledger.halo_lock(box.halo_dir(args.halo)) if not args.dry_run \
                else _nullcontext():
            if args.as_job:
                # enzo-mrp-music is far too heavy for a login node, so the
                # default path pushes IC generation onto a compute node.
                extra = []
                if args.no_submit:
                    extra.append("--no-submit")
                if args.no_hook:
                    extra.append("--no-hook")
                build.submit_build_job(box, args.halo, args.level, args.phase,
                                       dry_run=args.dry_run, adopt=args.adopt,
                                       extra_args=" ".join(extra))
            else:
                build.build_stage(
                    box, args.halo, args.level, args.phase,
                    dry_run=args.dry_run, adopt=args.adopt,
                    submit=not args.no_submit, hook=not args.no_hook,
                    rvir_min=row["rvir_min"] if "rvir_min" in row.colnames else None)
    except ledger.UnmanagedHaloError as exc:
        print("REFUSED: %s" % exc)
        return 1
    return 0


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def cmd_validate_registry(args):
    table = config.read_registry(args.registry)
    print("Registry: %s" % (args.registry or config.default_registry_path()))
    print("%d row(s), %d enabled\n" % (len(table), len(config.enabled_halos(table))))

    from astropy.io import ascii as ascii_io

    problems = 0
    for row in table:
        halo_id = row["halo_id"]
        try:
            box = config.get_box(row["box"])
        except KeyError as exc:
            print("  halo %-8s BAD BOX  %s" % (halo_id, exc))
            problems += 1
            continue

        if int(row["final_level"]) > box.max_level:
            print("  halo %-8s final_level %d exceeds box max_level %d"
                  % (halo_id, int(row["final_level"]), box.max_level))
            problems += 1

        catalog = box.catalog_path()
        if not os.path.exists(catalog):
            print("  halo %-8s catalog missing: %s" % (halo_id, catalog))
            problems += 1
            continue

        halos = ascii_io.read(catalog, header_start=0, data_start=2)
        match = halos[halos["ID"] == int(halo_id)]
        if len(match) != 1:
            print("  halo %-8s NOT FOUND in %s" % (halo_id, catalog))
            problems += 1
            continue

        stages = config.stage_plan(row, include_gas=args.include_gas)
        print("  halo %-8s %-14s enabled=%-5s Rvir=%7.2f kpc  stages: %s"
              % (halo_id, row["box"], bool(row["enabled"]), match["Rvir"][0],
                 " ".join("L%d-%s" % s for s in stages)))

    print("")
    if problems:
        print("FAILED: %d problem(s)." % problems)
        return 1
    print("OK: registry is valid.")
    return 0


# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="ic_pipeline",
        description="Automated IC generation and Enzo job chaining for 25 Mpc zooms.")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("status", help="progress table over all halos and levels")
    p.add_argument("--registry", default=None)
    p.add_argument("--include-manual", action="store_true",
                   help="also report frozen halo directories the pipeline does not manage")
    p.add_argument("--include-gas", action="store_true",
                   help="include gas stages for registry halos that ask for them")
    p.add_argument("--by-halo", action="store_true",
                   help="also print a one-row-per-halo rollup")
    p.add_argument("--write", action="store_true",
                   help="write status.ecsv, status_by_halo.ecsv and status.html")
    p.add_argument("--out-dir", default=None,
                   help="where to write them (default: FOGGIE_ICS_DIR, i.e. next to "
                        "the simulations and outside version control)")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("advance",
                       help="submit the next actionable stage (the engine both triggers call)")
    p.add_argument("--halo", default=None, help="one halo; omit for every enabled halo")
    p.add_argument("--registry", default=None)
    p.add_argument("--include-gas", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="report what would be submitted without submitting it")
    p.set_defaults(func=cmd_advance)

    p = sub.add_parser("build", help="generate ICs for one stage and submit it")
    p.add_argument("--halo", required=True)
    p.add_argument("--level", type=int, required=True)
    p.add_argument("--phase", default="DM", choices=["DM", "gas"])
    p.add_argument("--box", default=None)
    p.add_argument("--registry", default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="show every file that would be written and command run, without acting")
    p.add_argument("--no-submit", action="store_true", help="build the ICs but do not qsub")
    p.add_argument("--no-hook", action="store_true",
                   help="omit the advance hook from the generated RunScript")
    p.add_argument("--adopt", action="store_true",
                   help="allow writing into a halo directory the pipeline did not create")
    p.add_argument("--as-job", action="store_true",
                   help="submit IC generation as a PBS job instead of running it here "
                        "(required in practice: enzo-mrp-music must not run on a login node)")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("validate-templates",
                       help="check the collapsed .enzo templates re-render the per-level originals")
    p.add_argument("--box", default=config.DEFAULT_BOX)
    p.add_argument("--git-ref", default="master",
                   help="git ref to read the original per-level files from (default: master)")
    p.set_defaults(func=cmd_validate_templates)

    p = sub.add_parser("validate-registry", help="check the halo registry parses and resolves")
    p.add_argument("--registry", default=None)
    p.add_argument("--include-gas", action="store_true",
                   help="include gas stages in the printed stage plan")
    p.set_defaults(func=cmd_validate_registry)

    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
