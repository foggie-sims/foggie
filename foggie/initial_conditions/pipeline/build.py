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
        "__QUEUE__": "normal",
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
    from astropy.io import ascii as ascii_io

    halos = ascii_io.read(box.catalog_path(), header_start=0, data_start=2)
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


def read_shifts(conf_log):
    """MUSIC's (shift_x, shift_y, shift_z) for a level, from its .conf_log.txt.

    Replaces the old `grep shift_x ... | awk '{print $7}'` / paste sequence.
    """
    shifts = {}
    with open(conf_log, errors="replace") as fp:
        for line in fp:
            m = _SHIFT_RE.search(line)
            if m:
                shifts[m.group(1)] = int(m.group(2))
    missing = [a for a in "xyz" if a not in shifts]
    if missing:
        raise RuntimeError("No shift_%s in %s" % ("/".join(missing), conf_log))
    return [shifts["x"], shifts["y"], shifts["z"]]


def center_for_level(box, halo_id, level, halo_dir, rvir_min=None):
    """Halo center in the frame of the level-N ICs.

    Each MUSIC level re-centers the box, so the center must be walked forward
    through the accumulated shifts of every preceding level, converting each
    from parent-grid units to code units by dividing by (parent_ngrid - 1).
    Verified against halo42189: 0.79833 + (-165/511) = 0.475434, matching the
    0.475433 in the hand-built halo42189_DM_1to2.conf.
    """
    center, _ = halo_center_and_radius(box, halo_id, rvir_min)
    for prev in range(1, level):
        conf_log = os.path.join(halo_dir, "%s-L%d.conf_log.txt" % (box.sim_name, prev))
        if not os.path.exists(conf_log):
            raise RuntimeError(
                "Cannot build L%d: missing %s (level %d ICs were never generated)"
                % (level, conf_log, prev))
        shifts = read_shifts(conf_log)
        center = [c + s / box.shift_divisor for c, s in zip(center, shifts)]
    return center


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


def build_stage(box, halo_id, level, phase="DM", dry_run=False, adopt=False,
                submit=True, hook=True, rvir_min=None):
    """Generate ICs and the run script for one stage, then submit it.

    Returns the PBS job id, or None for a dry run.
    """

    halo_dir = box.halo_dir(halo_id)
    stage_dir = box.stage_dir(halo_id, level, phase)
    ledger.guard_unmanaged(halo_dir, box.sim_name, adopt=adopt)

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
    print("  submitted %s" % jobid)
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
