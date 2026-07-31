"""
Rendering of the per-level Enzo parameter files and run scripts.

The three DM templates 25Mpc_DM_512-L{1,2,3}.enzo differed only in
CosmologySimulationNumberOfInitialGrids, MustRefineParticlesRefineToLevel and
MinimumOverDensityForRefinement, so they collapse to a single DM-LX.enzo plus
the keyword table below.  `ic_pipeline validate-templates` proves the collapse
is faithful by re-rendering each level and diffing against the originals.
"""

import os
import re
import shutil
import subprocess
from functools import lru_cache

try:
    from . import config as _config
    from . import ledger
except ImportError:  # running by path rather than as a package
    import config as _config
    import ledger


# ---------------------------------------------------------------------------
# Level-dependent values
# ---------------------------------------------------------------------------

# The tail of the MinimumOverDensityForRefinement vector, after the two
# level-dependent leading entries.  The DM and gas templates were written with
# different lengths; preserved verbatim rather than normalised, so the rendered
# files stay byte-comparable with the originals.
_OVERDENSITY_TAIL_DM = "1. 1. 1. 1. 1. 1. 1. 1."
_OVERDENSITY_TAIL_GAS = "1. 1. 1. 1. 1"


def _fmt_overdensity(value):
    """Format a power of 1/8 the way the hand-written templates did."""
    if value == 1.0:
        return "1."
    return repr(value)


def min_overdensity(level, phase="DM"):
    """MinimumOverDensityForRefinement for a given zoom level.

    Divide by 8 for each additional level, per Britton Smith's notes: level 1
    gives 1., level 2 gives 0.125, level 3 gives 0.015625.
    """
    value = 8.0 ** -(level - 1)
    lead = _fmt_overdensity(value)
    tail = _OVERDENSITY_TAIL_GAS if phase == "gas" else _OVERDENSITY_TAIL_DM
    return "%s %s %s" % (lead, lead, tail)


def enzo_keywords(box, level, phase="DM", grid_parameters=""):
    """Keyword table for the .enzo templates."""
    kw = {
        "__NUM_INITIAL_GRIDS__": str(level + 1),
        "__MRP_REFINE_TO_LEVEL__": str(level),
        "__GRID_PARAMETERS__": grid_parameters,
    }
    if phase == "gas":
        kw["__MIN_OVERDENSITY_GAS__"] = min_overdensity(level, "gas")
        kw["__MAX_REFINE_LEVEL__"] = str(box.gas_max_refine_level)
    else:
        kw["__MIN_OVERDENSITY__"] = min_overdensity(level, "DM")
    return kw


# ---------------------------------------------------------------------------
# Substitution
# ---------------------------------------------------------------------------

def replace_keywords(text, mapping):
    """Substitute __KEYWORD__ placeholders, erroring on anything left behind."""
    for key, value in mapping.items():
        text = text.replace(key, str(value))
    leftover = sorted(set(re.findall(r"__[A-Z_]+__", text)))
    if leftover:
        raise RuntimeError("Unsubstituted keyword(s): %s" % ", ".join(leftover))
    return text


def render_enzo_param(box, level, phase="DM", grid_parameters=""):
    """Render the Enzo parameter file for one stage."""
    name = "gas-LX.enzo" if phase == "gas" else "DM-LX.enzo"
    template = os.path.join(box.template_dir_path(), name)
    with open(template) as fp:
        text = fp.read()
    return replace_keywords(text, enzo_keywords(box, level, phase, grid_parameters))


def read_grid_parameters(parameter_file_txt):
    """Pull the CosmologySimulationGrid* block out of MUSIC's parameter_file.txt.

    Replaces the old `grep ... > grid_parameters.txt; cat template
    grid_parameters.txt > pars.temp; mv pars.temp template` shell sequence.
    """
    with open(parameter_file_txt) as fp:
        return "".join(l for l in fp if l.startswith("CosmologySimulationGrid"))


def render_runscript(box, halo_id, level, phase="DM", pipeline_hook=""):
    """Render the PBS run script for one stage."""
    template = os.path.join(box.template_dir_path(), "RunScript.sh")
    with open(template) as fp:
        text = fp.read()

    is_gas = phase == "gas"
    walltime = box.gas_walltime if is_gas else box.dm_walltime
    hours, minutes, seconds = (int(x) for x in walltime.split(":"))
    mapping = {
        "__JOBNAME__": "halo%s-L%d%s" % (halo_id, level, "-gas" if is_gas else ""),
        "__GROUP__": box.group_list,
        "__SELECT__": box.gas_select if is_gas else box.dm_select,
        "__WALLTIME__": walltime,
        # No -q line unless one is configured.  The hand-built RunScripts that
        # ran successfully specify no queue and let PBS route by walltime;
        # naming `normal` explicitly caps the job at 8 h and gets a 24 h request
        # rejected outright ("Job violates queue and/or server resource limits").
        "__QUEUE_LINE__": ("#PBS -q %s" % box.queue if box.queue
                           else "# no queue requested: PBS routes on walltime"),
        "__NRANKS__": box.gas_nranks if is_gas else box.dm_nranks,
        # simrun.pl needs the walltime in seconds so it knows when to resubmit.
        "__SIMRUN_WALL__": hours * 3600 + minutes * 60 + seconds,
        "__EMAIL__": box.email,
        "__ENZO_EXE__": box.enzo_exe,
        "__PARAM_FILE__": box.param_filename(level, phase),
        "__PIPELINE_HOOK__": pipeline_hook,
    }
    return replace_keywords(text, mapping)


# ---------------------------------------------------------------------------
# Halo lookup
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _read_catalog(path):
    """Read a Rockstar catalog once per process; out_0.list is ~31 MB."""
    from astropy.io import ascii as ascii_io

    return ascii_io.read(path, header_start=0, data_start=2)


def halo_center_and_radius(box, halo_id, rvir_min=None):
    """Base (L0) halo center in code units and the zoom radius in kpc.

    The radius is Rvir from the catalog, raised to a floor if one applies.  The
    floor can come from the box (script256.py used 200 kpc, script512.py used
    none -- a difference that was silent) or per-halo from the registry.

    The per-halo override exists because halo11177 was hand-built with a radius
    of 80 kpc against a catalog Rvir of 33.66 kpc, expressed as `--rvir_min=80`
    in its L1.sh -- an argument no version of script512.py ever defined.  That
    intent now lives in the registry instead of in a broken command line.
    """
    halos = _read_catalog(box.catalog_path())
    match = halos[halos["ID"] == int(halo_id)]
    if len(match) != 1:
        raise RuntimeError("halo %s not found in %s" % (halo_id, box.catalog_path()))
    center = [float(match[c][0]) / box.boxsize_mpc for c in ("X", "Y", "Z")]
    rvir = float(match["Rvir"][0])
    floor = rvir_min if rvir_min else box.rvir_floor_kpc
    if floor:
        rvir = max(rvir, float(floor))
    return center, rvir


_SHIFT_RE = re.compile(r"setup/shift_([xyz])\s*=\s*(-?\d+)")
_DOMAIN_RE = re.compile(r"Domain shifted by\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\)")


def read_shifts(conf_log):
    """MUSIC's (shift_x, shift_y, shift_z) for a level, from its .conf_log.txt.

    A mis-parse here silently displaces the zoom region and yields ICs that look
    plausible and are wrong, so the value is read two independent ways and the
    two must agree:

        setup/shift_x = -164                (settings dump, 3 lines)
        Domain shifted by ( -164, ...)      (runtime message, 2 lines)

    Every occurrence must be identical.  The old code was
    `grep shift_x ... | awk '{print $7}'`, which took whichever line came last
    with no cross-check at all.
    """
    settings, domains = {}, []
    with open(conf_log, errors="replace") as fp:
        for line in fp:
            m = _SHIFT_RE.search(line)
            if m:
                axis, value = m.group(1), int(m.group(2))
                if axis in settings and settings[axis] != value:
                    raise RuntimeError(
                        "%s: conflicting setup/shift_%s values (%d and %d); the log "
                        "may cover more than one MUSIC run"
                        % (conf_log, axis, settings[axis], value))
                settings[axis] = value
            m = _DOMAIN_RE.search(line)
            if m:
                domains.append([int(g) for g in m.groups()])

    missing = [a for a in "xyz" if a not in settings]
    if missing:
        raise RuntimeError("No setup/shift_%s in %s" % ("/".join(missing), conf_log))
    shifts = [settings["x"], settings["y"], settings["z"]]

    if not domains:
        raise RuntimeError("No 'Domain shifted by' line in %s to cross-check "
                           "setup/shift_* against" % conf_log)
    for domain in domains:
        if domain != shifts:
            raise RuntimeError("%s: 'Domain shifted by' %s disagrees with "
                               "setup/shift_* %s" % (conf_log, domain, shifts))
    return shifts


def center_for_level(box, halo_id, level, halo_dir, rvir_min=None):
    """Halo center in the frame of the level-N ICs.

    The shift MUSIC reports is the ABSOLUTE displacement of the domain from the
    original box, not an increment over the previous level, so the center is the
    catalog center plus the shift from level N-1's log alone.  Do not accumulate
    across levels.

    halo42189 makes the distinction concrete: its L1 log says -165 and its L2
    log says -164 (x axis).  The hand-built configs are

        0to1  0.79833   = catalog center
        1to2  0.475433  = 0.79833 + (-165/511)   <- L1 log
        2to3  0.47739   = 0.79833 + (-164/511)   <- L2 log, NOT L1 + L2

    Summing the two would put L3 at 0.15453, roughly a third of a box off, with
    nothing downstream to flag it.  This mirrors the old scripts, where
    set_1to2_conf and set_2to3_conf both added a single shift to the catalog
    center x0 rather than chaining.
    """
    center, _ = halo_center_and_radius(box, halo_id, rvir_min)
    if level == 1:
        return center

    conf_log = os.path.join(halo_dir, "%s-L%d.conf_log.txt" % (box.sim_name, level - 1))
    if not os.path.exists(conf_log):
        raise RuntimeError(
            "Cannot build L%d: missing %s (level %d ICs were never generated)"
            % (level, conf_log, level - 1))
    shifts = read_shifts(conf_log)
    return [c + s / box.shift_divisor for c, s in zip(center, shifts)]


# ---------------------------------------------------------------------------
# enzo-mrp-music config
# ---------------------------------------------------------------------------

def mrp_config_name(halo_id, level):
    return "halo%s_DM_%dto%d.conf" % (halo_id, level - 1, level)


def render_mrp_config(box, halo_id, level, halo_dir, rvir_min=None):
    """Render the enzo-mrp-music config for one DM level.

    Replaces set_0to1_conf / set_1to2_conf / set_2to3_conf / set_3to4_conf,
    which were four copies of the same awk/sed pipeline.
    """
    template = os.path.join(box.template_dir_path(), "halo_DM_NtoN.conf")
    with open(template) as fp:
        text = fp.read()

    center = center_for_level(box, halo_id, level, halo_dir, rvir_min)
    _, rvir = halo_center_and_radius(box, halo_id, rvir_min)

    # Level 1 reads the parent box from the shared ICs directory; deeper levels
    # read the previous zoom level, which lives in the halo directory.
    run_dir = _config.foggie_ics_dir() if level == 1 else halo_dir

    out = []
    for line in text.splitlines(True):
        stripped = line.strip()
        if stripped.startswith("halo_center "):
            line = "halo_center = %s , %s , %s\n" % tuple(repr(c) for c in center)
        elif stripped.startswith("halo_radius "):
            line = "halo_radius = %s\n" % rvir
        elif stripped.startswith("simulation_run_directory"):
            line = "simulation_run_directory = %s\n" % run_dir
        out.append(line)
    text = "".join(out)

    text = text.replace("FOGGIE_ICS_DIR/HALO_DIR", halo_dir)
    text = text.replace("FOGGIE_ICS_DIR", _config.foggie_ics_dir())
    text = text.replace("FOGGIE_REPO", _config.foggie_repo())
    return text


def render_gas_music_config(box, halo_id, level, halo_dir):
    """Turn the DM MUSIC config for a level into the gas one.

    The old convert_to_gas() hardcoded Omega_b = 0.0461 while every -gas.enzo
    declared 0.04576, so MUSIC and Enzo disagreed about the baryon fraction.
    The value now comes from the box config and is asserted below.
    """
    source = os.path.join(halo_dir, "%s-L%d.conf" % (box.sim_name, level))
    if not os.path.exists(source):
        raise RuntimeError("Cannot build gas ICs: %s does not exist "
                           "(run the DM level first)" % source)
    out = []
    with open(source) as fp:
        for line in fp:
            key = line.split("=")[0].strip()
            if key == "baryons":
                line = "baryons = yes\n"
            elif key == "Omega_b":
                line = "Omega_b = %s\n" % box.omega_b
            elif key == "filename":
                line = "filename = %s\n" % os.path.join(
                    halo_dir, "%s-L%d-gas" % (box.sim_name, level))
            out.append(line)
    return "".join(out)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def run(command, cwd=None, dry_run=False):
    """Run a shell command, raising on failure.

    The old scripts ignored return codes and carried on into a directory that
    was never created; every call site here checks.
    """
    if dry_run:
        print("    [dry-run] %s" % command)
        return 0
    print("    + %s" % command)
    result = subprocess.run(command, shell=True, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError("Command failed (exit %d): %s" % (result.returncode, command))
    return result.returncode


def write_file(path, text, dry_run=False):
    if dry_run:
        print("    [dry-run] write %s (%d bytes)" % (path, len(text)))
        return
    with open(path, "w") as fp:
        fp.write(text)
    print("    wrote %s" % path)


def pipeline_hook_line(halo_id, halo_dir):
    """The `advance` call appended to the generated RunScript."""
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ic_pipeline.py")
    return ("python3 %s advance --halo %s >> %s/pipeline.log 2>&1"
            % (script, halo_id, halo_dir))


def check_output_consistency(box, halo_id, level, phase, allow_mixed=False):
    """Refuse to build a level whose output list differs from the level below.

    Editing the redshift list in a template part-way through a ladder silently
    produces a halo whose levels are not comparable: RD0014 is z=0 under a
    15-entry list and z~7 under the 266-entry one.  Nothing downstream would
    flag that, so it is caught here, at the only point where the two can first
    disagree.
    """
    try:
        from . import state as _state
    except ImportError:
        import state as _state

    if level <= 1:
        return
    prev_dir = box.stage_dir(halo_id, level - 1, "DM")
    prev_param = _state.find_param_file(prev_dir)
    if not prev_param:
        return

    template = os.path.join(box.template_dir_path(),
                            "gas-LX.enzo" if phase == "gas" else "DM-LX.enzo")
    new_sig = _state.output_signature(template)
    old_sig = _state.output_signature(prev_param)
    if not new_sig or not old_sig or new_sig == old_sig:
        return

    message = (
        "halo %s L%d-%s would use a different output list from L%d.\n"
        "    L%d (on disk): %d outputs, final RD%04d at z=%s\n"
        "    L%d (template): %d outputs, final RD%04d at z=%s\n"
        "  The levels would not be comparable -- the same RD number means a\n"
        "  different redshift at each level. Rebuild the lower level with the\n"
        "  current template, or pass --allow-mixed-outputs if this is deliberate."
        % (halo_id, level, phase, level - 1,
           level - 1, old_sig[0], old_sig[1], old_sig[2],
           level, new_sig[0], new_sig[1], new_sig[2]))
    if allow_mixed:
        print("  WARNING: %s" % message)
        return
    raise RuntimeError(message)


def build_stage(box, halo_id, level, phase="DM", dry_run=False, adopt=False,
                submit=True, hook=True, rvir_min=None, allow_mixed=False):
    """Generate ICs and the run script for one stage, then submit it.

    Returns the PBS job id, or None for a dry run.
    """

    halo_dir = box.halo_dir(halo_id)
    stage_dir = box.stage_dir(halo_id, level, phase)
    ledger.guard_unmanaged(halo_dir, box.sim_name, adopt=adopt)
    check_output_consistency(box, halo_id, level, phase, allow_mixed=allow_mixed)

    print("Building halo %s %s" % (halo_id, ledger.stage_key(level, phase)))
    print("  halo dir : %s" % halo_dir)
    print("  stage dir: %s" % stage_dir)

    if not dry_run:
        os.makedirs(halo_dir, exist_ok=True)
        ledger.ensure_managed(halo_dir)

    # --- initial conditions -------------------------------------------------
    if phase == "gas":
        conf_name = "%s-L%d-gas.conf" % (box.sim_name, level)
        write_file(os.path.join(halo_dir, conf_name),
                   render_gas_music_config(box, halo_id, level, halo_dir), dry_run)
        music = os.path.join(box.music_exe_dir_path(), "MUSIC")
        run("%s %s" % (music, conf_name), cwd=halo_dir, dry_run=dry_run)
    else:
        conf_name = mrp_config_name(halo_id, level)
        write_file(os.path.join(halo_dir, conf_name),
                   render_mrp_config(box, halo_id, level, halo_dir, rvir_min), dry_run)
        mrp = os.path.join(_config.foggie_repo(), "initial_conditions",
                           "enzo-mrp-music", "enzo-mrp-music.py")
        # enzo-mrp-music resolves the previous level's conf_log relative to the
        # working directory, so it must run from the halo directory.
        run("python3 %s %s %d" % (mrp, conf_name, level), cwd=halo_dir, dry_run=dry_run)

    # --- Enzo parameter file ------------------------------------------------
    parameter_file_txt = os.path.join(stage_dir, "parameter_file.txt")
    if dry_run and not os.path.exists(parameter_file_txt):
        grid_parameters = "    [dry-run: grid geometry comes from MUSIC]\n"
    else:
        grid_parameters = read_grid_parameters(parameter_file_txt)
        if not grid_parameters.strip():
            raise RuntimeError("No CosmologySimulationGrid* lines in %s" % parameter_file_txt)
    write_file(os.path.join(stage_dir, box.param_filename(level, phase)),
               render_enzo_param(box, level, phase, grid_parameters), dry_run)

    # --- run script ---------------------------------------------------------
    hook_line = pipeline_hook_line(halo_id, halo_dir) if hook else "# pipeline hook disabled"
    write_file(os.path.join(stage_dir, "RunScript.sh"),
               render_runscript(box, halo_id, level, phase, hook_line), dry_run)
    simrun = os.path.join(box.template_dir_path(), "simrun.pl")
    if dry_run:
        print("    [dry-run] copy %s -> %s" % (simrun, stage_dir))
    else:
        shutil.copy2(simrun, os.path.join(stage_dir, "simrun.pl"))
        os.chmod(os.path.join(stage_dir, "RunScript.sh"), 0o755)
        os.chmod(os.path.join(stage_dir, "simrun.pl"), 0o755)

    # --- submit -------------------------------------------------------------
    if not submit:
        print("  (not submitting)")
        return None
    return submit_enzo_run(box, halo_id, level, phase, dry_run=dry_run)


def submit_enzo_run(box, halo_id, level, phase="DM", dry_run=False):
    """Submit the Enzo run for a stage whose ICs already exist.

    Separate from build_stage so `advance` can submit a BUILT stage -- one whose
    ICs were generated but whose run was never started -- without regenerating
    the initial conditions.
    """
    halo_dir = box.halo_dir(halo_id)
    stage_dir = box.stage_dir(halo_id, level, phase)
    runscript = os.path.join(stage_dir, "RunScript.sh")
    if not os.path.exists(runscript):
        raise RuntimeError("No RunScript.sh in %s; the ICs are not built" % stage_dir)
    check_queue_fits(box.queue, box.gas_walltime if phase == "gas" else box.dm_walltime,
                     "halo %s %s" % (halo_id, ledger.stage_key(level, phase)))

    if dry_run:
        print("    [dry-run] qsub -koed RunScript.sh   (in %s)" % stage_dir)
        return None

    jobid = subprocess.check_output(["qsub", "-koed", "RunScript.sh"],
                                    cwd=stage_dir).decode().strip()
    ledger.append_record(halo_dir, {
        "stage": ledger.stage_key(level, phase),
        "action": "enzo",
        "jobid": jobid,
        "stage_dir": stage_dir,
    })
    print("  submitted Enzo run %s for halo %s %s"
          % (jobid, halo_id, ledger.stage_key(level, phase)))
    return jobid


def render_buildscript(box, halo_id, level, phase, extra_args=""):
    """Render the PBS job that generates ICs for one stage on a compute node."""
    template = os.path.join(box.template_dir_path(), "BuildScript.sh")
    with open(template) as fp:
        text = fp.read()

    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ic_pipeline.py")
    cmd = ("python3 %s build --halo %s --level %d --phase %s%s"
           % (script, halo_id, level, phase, (" " + extra_args) if extra_args else ""))
    return replace_keywords(text, {
        "__JOBNAME__": "build-halo%s-L%d-%s" % (halo_id, level, phase),
        "__GROUP__": box.group_list,
        "__BUILD_SELECT__": box.build_select,
        "__BUILD_WALLTIME__": box.build_walltime,
        "__BUILD_QUEUE__": box.build_queue,
        "__HALO_DIR__": box.halo_dir(halo_id),
        "__BUILD_CMD__": cmd,
    })


def submit_build_job(box, halo_id, level, phase="DM", dry_run=False, adopt=False,
                     extra_args=""):
    """Submit the IC-generation job for one stage.  Returns the PBS job id."""
    halo_dir = box.halo_dir(halo_id)
    ledger.guard_unmanaged(halo_dir, box.sim_name, adopt=adopt)

    check_queue_fits(box.build_queue, box.build_walltime,
                     "halo %s %s IC build" % (halo_id, ledger.stage_key(level, phase)))
    text = render_buildscript(box, halo_id, level, phase, extra_args)
    path = os.path.join(halo_dir, "BuildScript-L%d-%s.sh" % (level, phase))
    print("Submitting IC build job for halo %s %s"
          % (halo_id, ledger.stage_key(level, phase)))
    if dry_run:
        print("    [dry-run] mkdir -p %s" % halo_dir)
        write_file(path, text, dry_run=True)
        print("    [dry-run] qsub %s" % path)
        return None

    os.makedirs(halo_dir, exist_ok=True)
    ledger.ensure_managed(halo_dir)
    write_file(path, text, dry_run=False)
    os.chmod(path, 0o755)

    jobid = subprocess.check_output(["qsub", os.path.basename(path)],
                                    cwd=halo_dir).decode().strip()
    ledger.append_record(halo_dir, {
        "stage": ledger.stage_key(level, phase),
        "action": "build",
        "jobid": jobid,
        "script": path,
    })
    print("  submitted %s" % jobid)
    return jobid


@lru_cache(maxsize=8)
def queue_max_walltime(queue):
    """Max walltime a PBS queue allows, in seconds, or None if unknown."""
    try:
        out = subprocess.check_output(["qstat", "-Qf", queue],
                                      stderr=subprocess.DEVNULL, timeout=30).decode()
    except Exception:
        return None
    for line in out.splitlines():
        if "resources_max.walltime" in line:
            parts = line.split("=")[-1].strip().split(":")
            if len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return None


def check_queue_fits(queue, walltime, label=""):
    """Fail early and legibly if a walltime exceeds its queue's cap.

    Without this, an over-long request comes back from qsub only as
    "Job violates queue and/or server resource limits" and exit status 188,
    which says nothing about which limit or by how much.
    """
    if not queue:
        return
    cap = queue_max_walltime(queue)
    if cap is None:
        return
    h, m, s = (int(x) for x in walltime.split(":"))
    want = h * 3600 + m * 60 + s
    if want > cap:
        raise RuntimeError(
            "%swalltime %s exceeds the '%s' queue limit of %02d:%02d:%02d. "
            "Either request a queue that allows it (long allows 120 h) or leave "
            "the queue unset so PBS routes on walltime."
            % (label and label + ": ", walltime, queue, cap // 3600,
               (cap % 3600) // 60, cap % 60))


def render_pollscript(box, script_path, log_dir, interval_minutes, reschedule=True,
                      notify=False, notify_to=None):
    """Render the self-rescheduling poller job."""
    template = os.path.join(box.template_dir_path(), "PollScript.sh")
    with open(template) as fp:
        text = fp.read()

    poll_script = os.path.join(log_dir, "PollScript.sh")
    if reschedule:
        # `date -d` computes the deferred start; qsub -a takes [[CC]YY]MMDDhhmm.
        line = ('qsub -a $(date -d "+%d minutes" +%%Y%%m%%d%%H%%M) %s'
                % (interval_minutes, poll_script))
    else:
        line = "# rescheduling disabled: this is a one-shot sweep"

    notify_args = ""
    if notify:
        notify_args = "--notify"
        if notify_to:
            notify_args += " --notify-to %s" % notify_to

    return replace_keywords(text, {
        "__GROUP__": box.group_list,
        "__POLL_SELECT__": box.poll_select,
        "__POLL_WALLTIME__": box.poll_walltime,
        "__POLL_QUEUE_LINE__": ("#PBS -q %s" % box.poll_queue if box.poll_queue
                                else "# no queue requested"),
        "__LOG_DIR__": log_dir,
        "__SCRIPT__": script_path,
        "__NOTIFY_ARGS__": notify_args,
        "__RESCHEDULE__": line,
    })


def render_atpoll(box, script_path, log_dir, interval_minutes, python=None,
                  reschedule=True, notify=False, notify_to=None):
    """Render the `at`-driven poller.

    Used in preference to both cron and PBS on the NAS front ends: cron is not
    executed there, and a PBS poller wakes a whole node for a 1.4 s sweep.
    """
    template = os.path.join(box.template_dir_path(), "AtPoll.sh")
    with open(template) as fp:
        text = fp.read()

    self_path = os.path.join(log_dir, "AtPoll.sh")
    line = ('at now + %d minutes -f %s >/dev/null 2>&1' % (interval_minutes, self_path)
            if reschedule else "# rescheduling disabled: one-shot sweep")

    notify_args = ""
    if notify:
        notify_args = "--notify" + (" --notify-to %s" % notify_to if notify_to else "")

    return replace_keywords(text, {
        "__FOGGIE_REPO__": _config.foggie_repo(),
        "__FOGGIE_ICS_DIR__": _config.foggie_ics_dir(),
        "__LOG_DIR__": log_dir,
        "__PYTHON__": python or "python3",
        "__SCRIPT__": script_path,
        "__NOTIFY_ARGS__": notify_args,
        "__RESCHEDULE__": line,
    })
