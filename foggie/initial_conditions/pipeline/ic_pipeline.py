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
    from . import notify
    from . import qc
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
    import notify
    import qc
    import report
    import state as stagestate


# ---------------------------------------------------------------------------
# Reference lookup
# ---------------------------------------------------------------------------

# The per-level files the collapsed templates replace, per box.  They are read
# from git so the check keeps working after the old template directories are
# deleted.
#
# One shared templates/ directory now serves every box: DM-LX.enzo and
# gas-LX.enzo carry __TOP_GRID__ and __OUTPUT_REDSHIFTS__, so what used to
# require a directory per parent box is two keywords.  These tables are what
# proves that collapse faithful.
_REFERENCE_DIRS = {
    "25Mpc_DM_512": "foggie/initial_conditions/halo_template_512",
    "25Mpc_DM_256": "foggie/initial_conditions/halo_template_256",
}

# Stages checked against the ORIGINAL hand-written files (--original).
#
# The 256 box lists DM only.  Its hand-written gas templates encode superseded
# physics -- MultiMetals = 1, StarMakerMinimumMass = 10000, the deprecated
# H2StarMakerMinimumMass, no self-shielding parameters and none of the PPM or
# Restrict* guards -- so the shared gas template deliberately does not
# reproduce them.  Comparing against them would report a wall of differences
# that are all intended.  The baseline check below still covers 256 gas.
_REFERENCE_STAGES = {
    "25Mpc_DM_512": [
        (1, "DM", "25Mpc_DM_512-L1.enzo"),
        (2, "DM", "25Mpc_DM_512-L2.enzo"),
        (3, "DM", "25Mpc_DM_512-L3.enzo"),
        (3, "gas", "25Mpc_DM_512-L3-gas.enzo"),
    ],
    "25Mpc_DM_256": [
        (1, "DM", "25Mpc_DM_256-L1.enzo"),
        (2, "DM", "25Mpc_DM_256-L2.enzo"),
        (3, "DM", "25Mpc_DM_256-L3.enzo"),
        (4, "DM", "25Mpc_DM_256-L4.enzo"),
    ],
}

# Stages baselined and checked by default.  Independent of the table above,
# because the baseline covers what the pipeline actually renders today rather
# than what it once replaced.
_BASELINE_STAGES = {
    "25Mpc_DM_512": [(1, "DM"), (2, "DM"), (3, "DM"), (3, "gas")],
    "25Mpc_DM_256": [(1, "DM"), (2, "DM"), (3, "DM"), (4, "DM"), (4, "gas")],
}


def reference_stages(box):
    return _REFERENCE_STAGES.get(box.sim_name, [])


def baseline_stages(box):
    return _BASELINE_STAGES.get(box.sim_name,
                                [(l, "DM") for l in range(1, box.max_level + 1)])


def _repo_root():
    """Top of the git working tree, or None."""
    try:
        out = subprocess.check_output(["git", "rev-parse", "--show-toplevel"],
                                      cwd=os.path.dirname(os.path.abspath(__file__)),
                                      stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return None


def baseline_dir(box):
    return os.path.join(box.template_dir_path(), "baseline")


def baseline_path(box, level, phase):
    return os.path.join(baseline_dir(box), "%s.enzo" % box.stage_dirname(level, phase))


def read_reference(filename, git_ref, reference_dir):
    """Read an original per-level .enzo file, preferring git over the worktree."""
    root = _repo_root()
    relpath = "%s/%s" % (reference_dir, filename)
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
    original, source = read_reference(filename, git_ref,
                                      _REFERENCE_DIRS[box.sim_name])
    diff = list(difflib.unified_diff(normalize(original), normalize(rendered),
                                     fromfile="original  %s" % source,
                                     tofile="rendered  L%d-%s" % (level, phase),
                                     lineterm="", n=1))
    return diff, source


def cmd_validate_templates(args):
    box = config.get_box(args.box)

    if args.rebaseline:
        os.makedirs(baseline_dir(box), exist_ok=True)
        for level, phase in baseline_stages(box):
            path = baseline_path(box, level, phase)
            with open(path, "w") as fp:
                fp.write(build.render_enzo_param(box, level, phase, grid_parameters=""))
            print("  baselined %s" % path)
        print("\nBaseline updated.  Future runs check against these rather than the\n"
              "original per-level files, so intentional template changes are approved\n"
              "once and accidental ones still fail.")
        return 0

    print("Validating collapsed templates for box %s" % box.sim_name)
    print("Template dir: %s" % box.template_dir_path())
    print("Checking against: %s\n"
          % ("the original per-level files in git %s" % args.git_ref if args.original
             else "the approved baseline"))

    stages = (reference_stages(box) if args.original
              else [(l, p, None) for l, p in baseline_stages(box)])
    if not stages:
        print("No stages registered for box %s." % box.sim_name)
        return 1

    failures = 0
    for level, phase, filename in stages:
        label = "L%d-%s" % (level, phase)
        try:
            if args.original:
                diff, source = compare_stage(box, level, phase, filename, args.git_ref)
            else:
                path = baseline_path(box, level, phase)
                if not os.path.exists(path):
                    raise RuntimeError("no baseline at %s -- run with --rebaseline" % path)
                rendered = build.render_enzo_param(box, level, phase, grid_parameters="")
                with open(path) as fp:
                    expected = fp.read()
                diff = list(difflib.unified_diff(
                    normalize(expected), normalize(rendered),
                    fromfile="baseline  %s" % os.path.basename(path),
                    tofile="rendered  %s" % label, lineterm="", n=1))
                source = os.path.relpath(path, box.template_dir_path())
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
        print("FAILED: %d of %d stage(s) differ." % (failures, len(stages)))
        print("The template collapse is not faithful; do not switch over.")
        return 1
    print("OK: all %d stages re-render to their %s "
          "(ignoring comments and whitespace)."
          % (len(stages), "originals" if args.original else "baseline"))
    return 0


def collect_registry_records(table, include_gas=False, qstat=None):
    """Stage records for every enabled registry halo, in dependency order."""
    qstat = stagestate.qstat_states() if qstat is None else qstat
    if qstat is None:            # reporting only: unknown queue shows as not-live
        qstat = {}
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
            # The gas stage hangs off the DM build at the same level, not off
            # the DM run, so its prerequisite is that config file existing.
            stage_prereq = (config.gas_ready(box, halo_id, level) if phase == "gas"
                            else prereq_done)
            st = stagestate.stage_state(stage_dir, job_state=job_state,
                                        prereq_done=stage_prereq)
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
            if phase != "gas":
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

        # Diff against the previous sweep BEFORE overwriting it: last sweep's
        # status.ecsv is the snapshot, so there is no separate state to keep.
        if args.notify:
            notify.notify_changes(stage_ecsv, rows, report.to_halo_rows(rows),
                                  args.notify_to or config.get_box(config.DEFAULT_BOX).email,
                                  dry_run=args.notify_dry_run)

        report.write_ecsv(rows, stage_ecsv)
        report.write_ecsv(report.to_halo_rows(rows), halo_ecsv,
                          columns=report.HALO_COLUMNS,
                          extra_comments=["One row per halo; see status.ecsv for per-stage detail."])
        report.write_html(rows, htm)
        print("\nWrote %s   (one row per stage)" % stage_ecsv)
        print("      %s   (one row per halo)" % halo_ecsv)
        print("      %s" % htm)
    return 0


def advance_halo(row, qstat, include_gas=True, dry_run=False, verbose=True):
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
    rvir_min = float(row["rvir_min"]) if "rvir_min" in row.colnames else None

    def act_on(level, phase, prereq_done):
        """Take the one action this stage warrants, or report why not."""
        stage_dir = box.stage_dir(halo_id, level, phase)
        jobid, job_state, action = ledger.live_job(halo_dir, level, phase, qstat)
        if action == "build" and job_state in (stagestate.QUEUED, stagestate.RUNNING):
            job_state = stagestate.BUILDING
        st = stagestate.stage_state(stage_dir, job_state=job_state, prereq_done=prereq_done)
        key = ledger.stage_key(level, phase)

        if st.state == stagestate.READY:
            if verbose:
                print("halo %s %s is READY -- submitting IC build" % (halo_id, key))
            build.submit_build_job(box, halo_id, level, phase, dry_run=dry_run)
            return st, "%s %s submitted IC build" % (halo_id, key)

        if st.state == stagestate.BUILT:
            if verbose:
                print("halo %s %s is BUILT -- submitting Enzo run" % (halo_id, key))
            build.submit_enzo_run(box, halo_id, level, phase, dry_run=dry_run)
            return st, "%s %s submitted Enzo run" % (halo_id, key)

        if st.state != stagestate.DONE and verbose:
            print("halo %s %s is %s%s -- nothing to do"
                  % (halo_id, key, st.state, " (%s)" % st.note if st.note else ""))
        return st, None

    actions = []
    done_levels = []

    # --- DM ladder: strictly sequential -------------------------------------
    prereq_done = True
    for level, phase in config.dm_ladder(row):
        st, did = act_on(level, phase, prereq_done)
        if did:
            actions.append(did)
        if st.state != stagestate.DONE:
            break
        done_levels.append(level)
    else:
        if verbose:
            print("halo %s: DM ladder complete" % halo_id)

    # --- gas: a parallel branch, not the next rung --------------------------
    #
    # Gas depends on the DM *build* at the same level having written its MUSIC
    # config, not on the DM *run* finishing, so it is checked independently of
    # where the ladder above got to and can be in flight at the same time.
    gas = config.gas_stage(row, include_gas)
    if gas:
        level, phase = gas
        if config.gas_ready(box, halo_id, level):
            _, did = act_on(level, phase, True)
            if did:
                actions.append(did)
        elif verbose:
            print("halo %s L%d-gas is BLOCKED -- waiting on %s"
                  % (halo_id, level,
                     os.path.basename(config.gas_prerequisite(box, halo_id, level))))

    # --- diagnostics: one density figure per newly finished level -----------
    did = maybe_submit_qc(box, row, done_levels, qstat, dry_run=dry_run,
                          verbose=verbose)
    if did:
        actions.append(did)

    return actions


def qc_due(halo_dir, done_levels, qstat):
    """Should the density figure be regenerated?  Returns the deepest level, or None.

    The figure is a ladder: one panel per level that has reached z = 0.  So it
    is stale exactly when a level has finished that the figure on disk does not
    cover, which the ledger records as `through_level`.  Comparing against that
    rather than against the file's mtime means a figure that failed to render
    is retried, while one that succeeded is not rebuilt every sweep.
    """
    if not done_levels:
        return None
    deepest = max(done_levels)
    if ledger.qc_in_flight(halo_dir, "density", qstat):
        return None
    record = ledger.last_qc(halo_dir, "density")
    if record and record.get("through_level") is not None:
        if int(record["through_level"]) >= deepest:
            return None
    return deepest


def maybe_submit_qc(box, row, done_levels, qstat, dry_run=False, verbose=True):
    """Submit the density figure if a level has finished that it does not cover.

    Deliberately fire-and-forget: it goes to the queue as its own short job, so
    a halo never waits on a diagnostic to advance, and a failure to render can
    never block a ladder.  It is also why this runs after the ladder and the gas
    branch rather than before them.
    """
    if not getattr(box, "qc_on_advance", True):
        return None
    halo_id = row["halo_id"]
    halo_dir = box.halo_dir(halo_id)
    deepest = qc_due(halo_dir, done_levels, qstat)
    if deepest is None:
        return None
    if verbose:
        print("halo %s: L%d finished -- submitting density figure" % (halo_id, deepest))
    try:
        build.submit_qc_job(box, halo_id, density=True, through_level=deepest,
                            dry_run=dry_run)
    except Exception as exc:
        # A diagnostic that cannot be submitted is not a reason to fail an
        # advance; the ladder matters and this does not.
        print("halo %s: density figure NOT submitted: %s" % (halo_id, exc))
        return None
    return "%s density figure through L%d" % (halo_id, deepest)



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
    if qstat is None:
        print("qstat unavailable: skipping this sweep so nothing is double-submitted")
        return 1
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
            print("halo %s BLOCKED: %s" % (row["halo_id"], exc))
            continue
        actions.extend(did or [])

    print("\nadvance: %s" % ("; ".join(actions) if actions else "no action taken"))
    return 0


def cmd_poll(args):
    """Sweep every enabled halo: advance what is actionable, refresh status.

    The job-chained hook is the primary trigger and fires within seconds; this
    exists for the case where that hook never runs at all, such as a hard node
    failure. It is cheap enough for a login node -- it reads OutputLog and
    RunFinished and calls qstat, nothing more.

    --install writes a self-rescheduling PBS job that re-submits itself with
    `qsub -a`, so nothing sleeps on an allocated node.
    """
    box = config.get_box(args.box)
    script_path = os.path.abspath(__file__)
    log_dir = args.out_dir or config.foggie_ics_dir()

    if args.install_at:
        # Preferred on the NAS front ends: cron is accepted but never executed
        # there, while atd runs.  Costs no allocation, unlike the PBS poller.
        text = build.render_atpoll(box, script_path, log_dir, args.interval,
                                   python=sys.executable, reschedule=not args.once,
                                   notify=args.notify, notify_to=args.notify_to)
        path = os.path.join(log_dir, "AtPoll.sh")
        if args.dry_run:
            print(text)
            print("[dry-run] would write %s and schedule it" % path)
            return 0
        with open(path, "w") as fp:
            fp.write(text)
        os.chmod(path, 0o755)
        proc = subprocess.run(["at", "now", "+", "1", "minute", "-f", path],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(proc.stdout.decode().strip())
        if proc.returncode != 0:
            return 1
        print("Wrote %s\nPoller chain started (every %d min, first sweep in ~1 min)."
              % (path, args.interval))
        print("Inspect with: atq        Stop with: atrm <job>  (and delete %s)" % path)
        return 0

    if args.install_cron:
        # A sweep is ~1.4 s and ~90 MB: it belongs in cron on a front end, not
        # in a PBS job.  Pleiades allocates whole nodes, so waking an Aitken
        # node every 30 minutes to read a few log files would burn allocation
        # for nothing.  The PBS poller (--install) stays available for sites
        # where front-end cron is not permitted.
        env = "FOGGIE_REPO=%s FOGGIE_ICS_DIR=%s" % (config.foggie_repo(),
                                                    config.foggie_ics_dir())
        notify_args = " --notify" + (" --notify-to %s" % args.notify_to
                                     if args.notify_to else "") if args.notify else ""
        line = ("*/%d * * * * %s %s %s poll%s >> %s/poll.log 2>&1"
                % (args.interval, env, sys.executable, script_path,
                   notify_args, log_dir))
        print("crontab line:\n  %s\n" % line)
        if args.dry_run:
            print("[dry-run] not installing")
            return 0

        existing = subprocess.run(["crontab", "-l"], stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL).stdout.decode()
        kept = [l for l in existing.splitlines() if "ic_pipeline.py poll" not in l]
        new = "\n".join([l for l in kept if l.strip()] +
                         ["# FOGGIE IC pipeline poller -- safety net for the RunScript hook",
                          line, ""])
        proc = subprocess.run(["crontab", "-"], input=new.encode(),
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            print("crontab install failed: %s" % proc.stdout.decode().strip())
            return 1
        print("Installed. Current crontab:")
        print(subprocess.run(["crontab", "-l"], stdout=subprocess.PIPE).stdout.decode())
        print("Remove with: crontab -l | grep -v ic_pipeline | crontab -")
        return 0

    if args.install:
        build.check_queue_fits(box.poll_queue, box.poll_walltime, "poller")
        text = build.render_pollscript(box, script_path, log_dir, args.interval,
                                       reschedule=not args.once,
                                       notify=args.notify, notify_to=args.notify_to)
        path = os.path.join(log_dir, "PollScript.sh")
        if args.dry_run:
            print(text)
            print("[dry-run] would write %s and qsub it" % path)
            return 0
        with open(path, "w") as fp:
            fp.write(text)
        os.chmod(path, 0o755)
        jobid = subprocess.check_output(["qsub", path], cwd=log_dir).decode().strip()
        print("Wrote %s\nSubmitted poller %s (every %d min)" % (path, jobid, args.interval))
        print("Stop it with: qdel %s   (and delete %s)" % (jobid.split(".")[0], path))
        return 0

    # One sweep, here and now.
    rc = cmd_advance(argparse.Namespace(
        registry=args.registry, halo=None,
        include_gas=args.include_gas, dry_run=args.dry_run))
    print("")
    return rc or cmd_status(argparse.Namespace(
        registry=args.registry, include_manual=False, include_gas=args.include_gas,
        by_halo=True, write=not args.dry_run, out_dir=args.out_dir,
        notify=args.notify, notify_to=args.notify_to, notify_dry_run=args.dry_run))


def cmd_qc(args):
    """Diagnostic plots for a halo across its refinement levels.

    Deliberately not run automatically after every stage: it needs yt and real
    memory, and the useful moment to look is once a level has finished, not
    while it is still filling in.  Run it after the fact, or with --as-job to
    push it onto a compute node.
    """
    table = config.read_registry(args.registry)

    # --halo all: every row in the registry, enabled or not.  `enabled` governs
    # what the pipeline spends machine time *running*; a halo that has been run
    # and then held is exactly as worth looking at as one still going.
    if str(args.halo).strip().lower() == "all":
        halos = [int(r["halo_id"]) for r in table]
        if not halos:
            print("registry has no rows")
            return 1
        print("%d halo(s) from %s\n" % (len(halos),
                                        args.registry or config.default_registry_path()))
        rc = 0
        for halo_id in halos:
            sub = argparse.Namespace(**vars(args))
            sub.halo = halo_id
            print("--- halo %s ---" % halo_id)
            try:
                rc |= cmd_qc(sub)
            except Exception as exc:
                # One halo that has not been built yet, or whose dump is
                # unreadable, must not stop the other thirteen.
                print("  FAILED: %s" % exc)
                rc = 1
        return rc

    match = [r for r in table if int(r["halo_id"]) == int(args.halo)]
    rvir_min = float(match[0]["rvir_min"]) if match and "rvir_min" in match[0].colnames else None
    box = config.get_box(args.box or (match[0]["box"] if match else config.DEFAULT_BOX))

    if args.as_job:
        jobid = build.submit_qc_job(box, args.halo, density=args.density,
                                    dry_run=args.dry_run)
        return 0 if (jobid or args.dry_run) else 1

    if args.density:
        print("Projected density by level for halo %s (%s)" % (args.halo, box.sim_name))
        path, rows = qc.make_density_figure(box, args.halo, out_path=args.out,
                                            width_rvir=args.width_rvir,
                                            context_mpc=args.context_mpc,
                                            include_gas=args.include_gas,
                                            recenter=args.recenter)
        print(qc.format_density_report(args.halo, rows))
        print("\nWrote %s" % path)

        # One neighbourhood panel per IC set, circling every parent-box
        # Rockstar halo at its own Rvir and labelling it with its ORIGINAL
        # catalog ID.  That numbering is the point: it is what the halo was
        # selected from, so a neighbour seen here can be looked up in the same
        # catalog, which a zoom-local AHF id cannot do.  It also shows at a
        # glance whether the zoom refined the halo it was built for -- the
        # question that cost halo79628 three days of compute before anyone
        # asked it.
        for lev in range(1, int(match[0]["final_level"]) + 1) if match else ():
            try:
                npath, note = qc.make_neighbor_projection(
                    box, args.halo, level=lev, phase="DM", rvir_min=rvir_min)
            except Exception as exc:
                print("  L%d neighbours: skipped (%s)" % (lev, exc))
                continue
            if npath is None:
                print("  L%d neighbours: skipped (%s)" % (lev, note))
            else:
                print("  L%d neighbours: %s -> %s" % (lev, note, npath))

        # A panel whose halo has drifted out of frame shows empty sky, which
        # reads as a halo that dissolved under refinement.  The drift is one to
        # two hundred kpc regardless of halo mass, so this is routine for the
        # dwarfs and never happens for the massive ones.  Render the re-centered
        # companion automatically rather than leaving a figure that has to be
        # interpreted before it can be believed -- both are kept, since the
        # uncentered one is the honest record of how far the halo moved.
        if not args.recenter and any(r["note"].startswith("OUT OF FRAME") for r in rows):
            second = (os.path.splitext(path)[0] + "_recentered.png")
            print("\nSome panels are out of frame; also rendering re-centered:")
            path2, rows2 = qc.make_density_figure(box, args.halo, out_path=second,
                                                  width_rvir=args.width_rvir,
                                                  context_mpc=args.context_mpc,
                                                  include_gas=args.include_gas,
                                                  recenter=True)
            print(qc.format_density_report(args.halo, rows2))
            print("\nWrote %s" % path2)
        return 0

    levels = [int(x) for x in args.levels.split(",")] if args.levels else None
    print("Diagnostics for halo %s (%s)" % (args.halo, box.sim_name))
    path, report = qc.make_qc_figure(box, args.halo, levels=levels,
                                     out_path=args.out, rvir_min=rvir_min)
    print(qc.format_report(args.halo, report))
    print("\nWrote %s" % path)
    return 0


def cmd_resume(args):
    """Resubmit stages that are STALLED.

    `advance` deliberately never retries a STALLED stage: normally a stall
    means the run hit something that will kill it again, and resubmitting just
    burns allocation.  But when a shared external cause stops everything at
    once -- a full filesystem, a bad node, a scheduler outage -- and that cause
    has been fixed, restarting each stage by hand is tedious and easy to get
    half-done.

    This is the human-initiated escape hatch for that case.  It is never called
    by the poller or the job-chained hook; you run it when you know why things
    stalled and that the reason is gone.
    """
    table = config.read_registry(args.registry)
    rows = config.enabled_halos(table)
    if args.halo is not None:
        rows = [r for r in rows if int(r["halo_id"]) == int(args.halo)]

    qstat = stagestate.qstat_states()
    if qstat is None:
        print("qstat unavailable: refusing to resume anything on an unknown queue state")
        return 1
    resumed, skipped = [], []
    for row in rows:
        box = config.get_box(row["box"])
        halo_id = row["halo_id"]
        halo_dir = box.halo_dir(halo_id)
        prereq_done = True
        for level, phase in config.stage_plan(row, include_gas=args.include_gas):
            stage_dir = box.stage_dir(halo_id, level, phase)
            jobid, job_state, action = ledger.live_job(halo_dir, level, phase, qstat)
            st = stagestate.stage_state(stage_dir, job_state=job_state,
                                        prereq_done=prereq_done)
            prereq_done = st.state == stagestate.DONE
            key = ledger.stage_key(level, phase)

            if st.state != stagestate.STALLED:
                continue
            if not os.path.exists(os.path.join(stage_dir, "RunScript.sh")):
                skipped.append("%s %s (no RunScript.sh -- ICs never built)" % (halo_id, key))
                continue

            print("halo %s %s STALLED at %s (%s) -- resubmitting"
                  % (halo_id, key, st.last or "nothing", st.note or "no reason recorded"))
            if args.dry_run:
                print("    [dry-run] qsub -koed RunScript.sh   (in %s)" % stage_dir)
                resumed.append("%s %s" % (halo_id, key))
                continue
            try:
                with ledger.halo_lock(halo_dir):
                    build.submit_enzo_run(box, halo_id, level, phase)
                resumed.append("%s %s" % (halo_id, key))
            except Exception as exc:
                skipped.append("%s %s (%s)" % (halo_id, key, exc))

    print("")
    print("resumed: %s" % (", ".join(resumed) if resumed else "nothing"))
    if skipped:
        print("skipped: %s" % ", ".join(skipped))
    print("\nsimrun.pl restarts each run from its last output. Anything still READY or")
    print("BUILT is picked up by `advance` or the next poll sweep.")
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

        box_issues = config.box_problems(box)
        if box_issues:
            print("  halo %-8s box %r is not usable:" % (halo_id, row["box"]))
            for issue in box_issues:
                print("      %s" % issue)
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
        catalog_rvir = float(match["Rvir"][0])
        rvir_min = float(row["rvir_min"]) if "rvir_min" in row.colnames else 0.0
        _, effective = build.halo_center_and_radius(box, halo_id, rvir_min)
        floored = " (floored)" if effective > catalog_rvir else ""
        print("  halo %-8s %-14s enabled=%-5s Rvir %6.2f -> %6.2f kpc%-10s stages: %s"
              % (halo_id, row["box"], bool(row["enabled"]), catalog_rvir, effective,
                 floored, " ".join("L%d-%s" % s for s in stages)))

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
    p.add_argument("--no-gas", dest="include_gas", action="store_false", default=True,
                   help="omit gas stages even for halos whose registry row asks for them")
    p.add_argument("--by-halo", action="store_true",
                   help="also print a one-row-per-halo rollup")
    p.add_argument("--write", action="store_true",
                   help="write status.ecsv, status_by_halo.ecsv and status.html")
    p.add_argument("--out-dir", default=None,
                   help="where to write them (default: FOGGIE_ICS_DIR, i.e. next to "
                        "the simulations and outside version control)")
    p.add_argument("--notify", action="store_true",
                   help="email stage state changes since the previous sweep (needs --write)")
    p.add_argument("--notify-to", default=None, help="recipient (default: the box email)")
    p.add_argument("--notify-dry-run", action="store_true",
                   help="print the message instead of sending it")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("advance",
                       help="submit the next actionable stage (the engine both triggers call)")
    p.add_argument("--halo", default=None, help="one halo; omit for every enabled halo")
    p.add_argument("--registry", default=None)
    p.add_argument("--no-gas", dest="include_gas", action="store_false", default=True,
                   help="skip gas stages even for halos whose registry row asks for them")
    p.add_argument("--dry-run", action="store_true",
                   help="report what would be submitted without submitting it")
    p.set_defaults(func=cmd_advance)

    p = sub.add_parser("poll", help="one sweep of every halo; --install to run it periodically")
    p.add_argument("--registry", default=None)
    p.add_argument("--box", default=config.DEFAULT_BOX)
    # Matches `status` and `advance`, which include gas unless told not to.
    # The opt-in form here meant `resume --halo X` silently did nothing for a
    # stalled gas stage -- the one command you reach for when a gas run dies.
    p.add_argument("--no-gas", dest="include_gas", action="store_false", default=True,
                   help="skip the gas stage")
    p.add_argument("--interval", type=int, default=30, help="minutes between sweeps")
    p.add_argument("--install-at", action="store_true",
                   help="self-rescheduling `at` chain on the front end (preferred: "
                        "cron is not executed on NAS front ends, and this costs no allocation)")
    p.add_argument("--install-cron", action="store_true",
                   help="install a front-end crontab entry (preferred: a sweep is ~1.4 s "
                        "and needs no allocation)")
    p.add_argument("--install", action="store_true",
                   help="submit a self-rescheduling PBS poller instead (for sites without cron)")
    p.add_argument("--once", action="store_true",
                   help="with --install, do not reschedule after the sweep")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--notify", action="store_true", help="email state changes each sweep")
    p.add_argument("--notify-to", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_poll)

    p = sub.add_parser("qc",
                       help="diagnostic plots per refinement level, centered on the halo")
    p.add_argument("--halo", required=True,
                   help="halo id, or `all` for every row in the registry")
    p.add_argument("--box", default=None)
    p.add_argument("--registry", default=None)
    p.add_argument("--levels", default=None,
                   help="comma-separated levels, e.g. 0,1,2 (default: all with data)")
    p.add_argument("--density", action="store_true",
                   help="projected DM density at every level on one color scale, "
                        "instead of the particle contamination panels")
    p.add_argument("--width-rvir", type=float, default=qc.DENSITY_WIDTH_RVIR,
                   help="with --density, panel width in Rvir (default %(default)s)")
    p.add_argument("--context-mpc", type=float, default=qc.DENSITY_CONTEXT_MPC,
                   help="with --density, width in Mpc of the context row "
                        "(default %(default)s)")
    p.add_argument("--include-gas", action="store_true",
                   help="with --density, also panel gas stages that have reached z = 0")
    p.add_argument("--recenter", action="store_true",
                   help="with --density, center each panel on the halo rather than on "
                        "the catalog position; needed for dwarfs, whose ~150 kpc drift "
                        "between levels is several Rvir and puts them out of frame")
    p.add_argument("--out", default=None, help="output PNG (default: halo dir)")
    p.add_argument("--as-job", action="store_true",
                   help="run on a compute node; this needs yt and several GB")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_qc)

    p = sub.add_parser("resume",
                       help="resubmit STALLED stages, after fixing whatever stopped them")
    p.add_argument("--halo", default=None, help="one halo; omit for every enabled halo")
    p.add_argument("--registry", default=None)
    # Matches `status` and `advance`, which include gas unless told not to.
    # The opt-in form here meant `resume --halo X` silently did nothing for a
    # stalled gas stage -- the one command you reach for when a gas run dies.
    p.add_argument("--no-gas", dest="include_gas", action="store_false", default=True,
                   help="skip the gas stage")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_resume)

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
                   help="git ref holding the original per-level files (with --original)")
    p.add_argument("--original", action="store_true",
                   help="check against the original hand-written per-level files instead "
                        "of the approved baseline")
    p.add_argument("--rebaseline", action="store_true",
                   help="approve the current templates as the new reference")
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
