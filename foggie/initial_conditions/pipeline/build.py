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
# different lengths; preserved verbatim rather than normalized, so the rendered
# files stay byte-comparable with the originals.
_OVERDENSITY_TAIL_DM = "1. 1. 1. 1. 1. 1. 1. 1."
_OVERDENSITY_TAIL_GAS = "1. 1. 1. 1. 1"


def _fmt_overdensity(value):
    """Format a power of 1/8 the way the hand-written templates did."""
    if value == 1.0:
        return "1."
    return repr(value)


def min_overdensity(level, phase="DM", root=8.0):
    """MinimumOverDensityForRefinement for a given zoom level.

    Divide by 8 for each additional level, per Britton Smith's notes: with the
    default root value of 8, level 1 gives 1., level 2 gives 0.125, level 3
    gives 0.015625.  `root` is box.overdensity_at_root -- see the comment there
    for what the number means and why it is 8.
    """
    value = float(root) * 8.0 ** -level
    lead = _fmt_overdensity(value)
    tail = _OVERDENSITY_TAIL_GAS if phase == "gas" else _OVERDENSITY_TAIL_DM
    return "%s %s %s" % (lead, lead, tail)


def restart_dump_seconds(box, phase="DM"):
    """Wallclock seconds after which Enzo checkpoints and stops.

    dtDataDump is 0, so the periodic DD dumps that runs used to restart from no
    longer exist -- and they were doing real work: halo51541 L3 restarted from
    DD0003, and DD0028/DD0012/DD0008 were each used ~10 times across the fleet.
    Without a replacement a walltime kill would lose everything back to the last
    redshift dump, which under the 15-entry output list can be a fifteenth of
    the run.

    dtRestartDump is Enzo's purpose-built answer: check the wallclock, write a
    dump, and stop cleanly.  Set below the PBS wall so Enzo finishes before the
    scheduler kills it.  The dump is recorded in OutputLog like any other
    (Group_WriteAllData.C:998), so simrun.pl restarts from it unchanged.

    One checkpoint per job instead of sixty-five per run.
    """
    walltime = box.gas_walltime if phase == "gas" else box.dm_walltime
    h, m, s = (int(x) for x in walltime.split(":"))
    total = h * 3600 + m * 60 + s
    return int(total * box.restart_dump_fraction)


def _fmt_redshift(z):
    """15.0 -> "15".  Enzo parses either, but the parameter files are read by
    people and a stray .0 invites the question of whether it matters."""
    return str(int(z)) if float(z) == int(z) else repr(float(z))


def output_redshifts(box, phase="DM"):
    """The CosmologyOutputRedshift block for this box and phase.

    Read from a file in the template directory rather than held inline, because
    it is the one place two boxes legitimately disagree while sharing every
    other line, and because it is long enough (266 entries) to bury the rest of
    the template.
    """
    name = box.gas_output_list if phase == "gas" else box.dm_output_list
    path = os.path.join(box.template_dir_path(), name)
    if not os.path.exists(path):
        raise RuntimeError("Output redshift list missing: %s" % path)
    with open(path) as fp:
        return fp.read().rstrip("\n")


def enzo_keywords(box, level, phase="DM", grid_parameters=""):
    """Keyword table for the .enzo templates."""
    kw = {
        "__RESTART_DUMP_SECONDS__": str(restart_dump_seconds(box, phase)),
        "__NUM_INITIAL_GRIDS__": str(level + 1),
        "__MRP_REFINE_TO_LEVEL__": str(level),
        "__GRID_PARAMETERS__": grid_parameters,
        # Box geometry and cadence: everything that used to force a separate
        # template directory per parent box.
        "__TOP_GRID__": "%d %d %d" % ((box.parent_ngrid,) * 3),
        "__OUTPUT_REDSHIFTS__": output_redshifts(box, phase),
    }
    if phase == "gas":
        kw["__MIN_OVERDENSITY_GAS__"] = min_overdensity(
            level, "gas", box.overdensity_at_root)
        kw["__MAX_REFINE_LEVEL__"] = str(box.gas_max_refine_level)
        # First leg only.  RunScript.sh rewrites this to 0 at the handoff, so
        # the two must come from the same place or the run either stops twice
        # or never stops at all.
        kw["__GAS_STOP_REDSHIFT__"] = _fmt_redshift(box.gas_stop_redshift)
    else:
        kw["__MIN_OVERDENSITY__"] = min_overdensity(
            level, "DM", box.overdensity_at_root)
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


TRANSITION_MARKER = "gas_transition.done"


def render_phase_transition(box, phase):
    """The shell block that carries a gas run across the cooling transition.

    Gas runs are two legs.  The first uses unshielded Grackle cooling and stops
    at box.gas_stop_redshift; the second switches self-shielding on and runs to
    z = 0.  Enzo cannot change those parameters mid-run, so the handoff is:
    stop, rewrite the restart parameter file, restart.

    Reaching CosmologyFinalRedshift makes Enzo write RunFinished, which
    simrun.pl reads as "done" and the pipeline's state machine reads as DONE.
    Both are wrong here -- the run is half over -- so RunFinished is removed as
    part of the handoff and a marker file records that the transition already
    happened.  Keying off the marker rather than off RunFinished is what makes
    this safe to re-enter: the second leg ends by writing RunFinished too, and
    without the marker the block would loop.

    The rewrite itself goes through simrun.pl's `new_pars` mechanism, which
    applies the substitutions to the restart parameter file before its run loop
    starts and then renames the file so it cannot be applied twice.

    Returns "" for DM, which is what leaves the DM path byte-identical.
    """
    if phase != "gas":
        return ""
    pars = "\n".join("%s = %s" % (k, v) for k, v in box.gas_transition_pars)
    zstr = _fmt_redshift(box.gas_stop_redshift)
    return """
# --- cooling transition at z = %(z)s -------------------------------------------
#
# Not a completed run: the first leg stops here by design, so rewrite the
# cooling parameters into the restart file and run the second leg to z = 0
# without leaving this job.  See render_phase_transition() in pipeline/build.py.
if [ ! -e %(marker)s ] && [ -e RunFinished ]; then
    echo "reached z = %(z)s: switching on self-shielded cooling for the second leg"
    cat > new_pars <<'ENZO_NEW_PARS'
%(pars)s
ENZO_NEW_PARS
    # RunFinished must go before simrun.pl runs again, or it exits immediately.
    # Write the marker before starting the leg, not after, so that a job killed
    # mid-leg does not repeat the handoff on its next attempt -- new_pars has
    # already been consumed by then.
    rm -f RunFinished
    date > %(marker)s

    # The second leg gets only the walltime this job has left.  simrun.pl
    # restarts its own clock on every invocation, so passing SIMRUN_WALL again
    # would let the two legs together overrun the PBS wall and be killed.
    REMAINING=$(( SIMRUN_WALL - SECONDS ))
    if [ "$REMAINING" -gt %(floor)d ]; then
        run_simrun "$REMAINING"
    else
        echo "only ${REMAINING}s of walltime left; resubmitting for the second leg"
        qsub -koed RunScript.sh
    fi
fi
""" % {"z": zstr, "marker": TRANSITION_MARKER,
       "pars": pars, "floor": box.gas_transition_min_seconds}


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
        "__PHASE_TRANSITION__": render_phase_transition(box, phase),
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

    # data_start=1, not 2.  The Rockstar file is one header line then data, so
    # data_start=2 skips the first halo -- ID 0 in the 512 catalog, an ordinary
    # 1.9e9 Msun object that the pipeline could therefore never select.  The
    # idiom came from script512.py and was wrong there too.
    return ascii_io.read(path, header_start=0, data_start=1)


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


def catalog_rvir(box, halo_id):
    """The halo's virial radius from the catalog, with no floor applied.

    Distinct from the radius halo_center_and_radius returns, which is the
    *zoom* radius and may have been floored up to give the Lagrangian region a
    workable size.  Diagnostics must use this one: quoting a contamination
    distance in units of an inflated radius understates how far in the coarse
    particles have come.
    """
    halos = _read_catalog(box.catalog_path())
    match = halos[halos["ID"] == int(halo_id)]
    if len(match) != 1:
        raise RuntimeError("halo %s not found in %s" % (halo_id, box.catalog_path()))
    return float(match["Rvir"][0])


def catalog_mvir(box, halo_id):
    """The halo's virial mass from the catalog, in Msun/h as Rockstar writes it.

    Carries the same h as the positions and Rvir, so it is quoted alongside them
    unconverted; the caller divides by h if it wants physical solar masses.
    """
    halos = _read_catalog(box.catalog_path())
    match = halos[halos["ID"] == int(halo_id)]
    if len(match) != 1:
        raise RuntimeError("halo %s not found in %s" % (halo_id, box.catalog_path()))
    return float(match["Mvir"][0])


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
    # Wrap into [0,1).  The box is periodic and MUSIC's shift routinely carries
    # a center past an edge: halo5348's L2 center came out as
    # (0.442, 1.412, -0.465), which enzo-mrp-music then traced into a region
    # ~27x too large, 15 Rvir off the halo, and contaminated at 3.3%.  It looks
    # like a bad zoom rather than a bad coordinate, which is why it took a
    # contamination check to find.
    return [(c + s / box.shift_divisor) % 1.0 for c, s in zip(center, shifts)]


# ---------------------------------------------------------------------------
# enzo-mrp-music config
# ---------------------------------------------------------------------------

def mrp_config_name(halo_id, level):
    return "halo%s_DM_%dto%d.conf" % (halo_id, level - 1, level)


def refine_center_from_run(box, halo_id, level, halo_dir, analytic_center,
                           search_kpc=500.0):
    """Where the halo actually is in the previous level's run, at z = 0.

    enzo-mrp-music traces the Lagrangian region from the parent-box particles
    inside a sphere at the centre we give it.  Giving it the analytic centre --
    catalog position plus the MUSIC shift -- is wrong by however far the halo
    has drifted by z = 0, which across this fleet is 100 to 200 kpc and is not
    correlated with halo mass.

    For a big halo that is a fraction of the zoom radius and the sphere still
    overlaps the object.  For a dwarf it is several Rvir, and the sphere can
    miss entirely: halo5348's caught 5 particles instead of ~1000, and the
    convex hull of 5 scattered positions produced a region 100x too large and
    15 Rvir off the halo.  The Rvir floor exists to paper over exactly this, at
    the cost of a zoom far bigger than the science needs -- 4.6M cells against
    81k for a comparable halo.

    So locate the halo first.  Returns (centre, drift_kpc), or (analytic, None)
    if the previous level's output cannot be read, since a diagnostic must never
    be the reason a build fails.
    """
    prev_dir = (os.path.join(_config.foggie_ics_dir(), "%s-L0" % box.sim_name)
                if level - 1 == 0 else box.stage_dir(halo_id, level - 1, "DM"))
    try:
        import numpy as np
        try:
            from . import qc as _qc
        except ImportError:
            import qc as _qc

        snap, name, _, is_final = _qc.last_output(prev_dir)
        if snap is None or not is_final:
            print("    centre refinement skipped: %s has no final dump"
                  % os.path.basename(prev_dir))
            return analytic_center, None
        rel, mass, ds = _qc.load_particles(snap, analytic_center, search_kpc)
        if rel is None:
            print("    centre refinement skipped: no particles near the analytic centre")
            return analytic_center, None
        offset = _qc.locate_halo(rel, mass, guess_radius_kpc=search_kpc, verbose=False)
        if offset is None:
            print("    centre refinement skipped: halo not located in %s" % name)
            return analytic_center, None
        kpc_per_code = float(ds.quan(1.0, "code_length").in_units("kpc").d)
        drift = float(np.sqrt((np.asarray(offset) ** 2).sum()))
        centre = [(c + o / kpc_per_code) % 1.0
                  for c, o in zip(analytic_center, offset)]
        print("    centre refined using %s: halo is %.0f kpc from the analytic "
              "position" % (name, drift))
        return centre, drift
    except Exception as exc:
        print("    centre refinement skipped (%s); using the analytic centre" % exc)
        return analytic_center, None


def render_mrp_config(box, halo_id, level, halo_dir, rvir_min=None):
    """Render the enzo-mrp-music config for one DM level.

    Replaces set_0to1_conf / set_1to2_conf / set_2to3_conf / set_3to4_conf,
    which were four copies of the same awk/sed pipeline.
    """
    template = os.path.join(box.template_dir_path(), "halo_DM_NtoN.conf")
    with open(template) as fp:
        text = fp.read()

    center = center_for_level(box, halo_id, level, halo_dir, rvir_min)
    # Trace the region from where the halo IS, not where the catalog says it
    # was.  See refine_center_from_run.
    #
    # Level 1 is excluded deliberately.  Its region is traced from the L0 parent,
    # which is unigrid: every particle has the same mass, so locate_halo's
    # finest-species selection matches all of them and the shrinking spheres
    # converge on the largest object nearby rather than the target.  Doing it
    # anyway moved halo39829 394 kpc onto a neighbour and produced a 37-million
    # cell region.  It is also unnecessary -- the catalog position is measured in
    # the L0 box itself, so the analytic centre there is exact to ~1 kpc.
    if level >= 2 and getattr(box, "refine_centers", True):
        center, _ = refine_center_from_run(box, halo_id, level, halo_dir, center)
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

    # The parent box this zoom is cut from.  Hard-coded to 25Mpc_DM_512 until the
    # template directories were shared between boxes, at which point the 256 box
    # silently generated ICs against the 512 parent -- MUSIC ran at levelmin 9
    # and wrote a 25Mpc_DM_512-L1 directory the build then could not find.
    text = text.replace("__SIM_NAME__", box.sim_name)
    text = text.replace("__TEMPLATE_CONFIG__", box.template_config)
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


# Fewest Lagrangian region points a healthy zoom has produced across this fleet
# is 45 (halo52675); typical is hundreds to thousands.  halo5348's broken build
# traced 5.
MIN_REGION_POINTS = 25
WARN_REGION_POINTS = 60


def check_region_points(halo_dir, level, dry_run=False):
    """Refuse a zoom whose Lagrangian region was traced from too few particles.

    enzo-mrp-music defines the region from the parent-box particles inside a
    sphere at the halo's ANALYTIC centre -- catalog position plus the MUSIC
    shift.  By z = 0 the halo has drifted from that position, typically 100-200
    kpc, and if the zoom radius is smaller than the drift the sphere misses the
    halo and traces whatever few particles happen to be there.

    That is not a subtle failure but it is a silent one: halo5348's sphere
    caught 5 particles instead of ~1000, and the convex hull of 5 scattered
    Lagrangian positions came out ~100x the size of a normal region, 15 Rvir off
    the halo, contaminated at 3.3%.  Nothing downstream complained -- the ICs
    generated, Enzo ran, and the density ladder looked entirely plausible.

    The Rvir floor (rvir_floor_kpc, 80 kpc for the 512 box and 200 for the 256)
    is what normally keeps the sphere large enough to overlap the halo despite
    the drift.  Overriding it with a small per-halo rvir_min is what removed
    that protection.
    """
    import glob
    pattern = os.path.join(halo_dir, "initial_particle_positions-*.dat")
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not files:
        if dry_run:
            return None
        raise RuntimeError(
            "enzo-mrp-music wrote no region point file matching %s" % pattern)
    n = sum(1 for line in open(files[-1]) if line.strip())
    if n < MIN_REGION_POINTS:
        raise RuntimeError(
            "L%d region traced from only %d particles (%s).\n"
            "That is far below the %d seen for any healthy zoom in this fleet, "
            "and means the traced sphere missed the halo: it is centred on the "
            "analytic position, and the halo drifts 100-200 kpc from there by "
            "z = 0.  Raise the zoom radius -- check the registry row's rvir_min, "
            "which overrides the box's Rvir floor -- and rebuild."
            % (level, n, os.path.basename(files[-1]), MIN_REGION_POINTS))
    if n < WARN_REGION_POINTS:
        print("    WARNING: L%d region traced from only %d particles; healthy "
              "zooms here start around %d.  Check the zoom radius against the "
              "halo's drift before trusting this stage." % (level, n, WARN_REGION_POINTS))
    else:
        print("    region traced from %d particles" % n)
    return n


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
        # Before anything downstream trusts this region, check it was traced
        # from a plausible number of particles.
        check_region_points(halo_dir, level, dry_run=dry_run)

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


_EXE_RE = re.compile(r'(-exe\s+")([^"]*)(")')


def sync_runscript_exe(box, runscript, dry_run=False):
    """Repoint a rendered RunScript.sh at the box's current enzo_exe.

    Rendering happens once, at build time; submission can happen much later, and
    a stage can be resubmitted many times over its life by simrun.pl.  So the
    binary named in a RunScript.sh is a snapshot of whatever config.py said when
    that one file was written, and changing config.enzo_exe does not reach the
    stages already on disk.

    That is not hypothetical.  When enzo_exe moved to enzo-cic_deposit_fix, a
    build job for halo75392 L2 was already running with the old value imported
    into memory; it wrote its RunScript.sh four minutes after the config change
    and named the unfixed binary.  A stage built during any such window would
    have quietly kept running the tree the fix was meant to retire.

    Checking here, immediately before qsub, makes config.enzo_exe the single
    point of control for every submission regardless of when the stage was
    built.  Returns True if the file was changed.
    """
    with open(runscript) as f:
        text = f.read()
    match = _EXE_RE.search(text)
    if not match:
        print("  WARNING: no -exe line in %s; cannot verify the Enzo binary" % runscript)
        return False
    current = match.group(2)
    if current == box.enzo_exe:
        return False
    if dry_run:
        print("    [dry-run] would repoint %s\n      %s -> %s"
              % (runscript, current, box.enzo_exe))
        return False
    with open(runscript, "w") as f:
        f.write(text[:match.start()] + match.group(1) + box.enzo_exe + match.group(3)
                + text[match.end():])
    print("  repointed Enzo binary in %s\n    was %s\n    now %s"
          % (runscript, current, box.enzo_exe))
    return True


def submit_enzo_run(box, halo_id, level, phase="DM", dry_run=False):
    """Submit the Enzo run for a stage whose ICs already exist.

    Separate from build_stage so `advance` can submit a BUILT stage -- one whose
    ICs were generated but whose run was never started -- without regenerating
    the initial conditions.
    """
    # NOTE: sync_runscript_exe below is what keeps the binary a single point of
    # control.  See its docstring for why rendering alone is not enough.
    halo_dir = box.halo_dir(halo_id)
    stage_dir = box.stage_dir(halo_id, level, phase)
    runscript = os.path.join(stage_dir, "RunScript.sh")
    # Under --dry-run a fresh stage has no RunScript.sh, because build_stage only
    # said it would write one.  Treating that as "the ICs are not built" made
    # `build --phase gas --dry-run` traceback on any stage that did not already
    # exist, which is precisely the case a dry run is for.  Missing is still a
    # hard error on a real submit, where `advance` relies on it to catch a stage
    # whose ICs never got generated.
    if not os.path.exists(runscript):
        if not dry_run:
            raise RuntimeError("No RunScript.sh in %s; the ICs are not built" % stage_dir)
        print("    [dry-run] RunScript.sh would have been written above")
    check_queue_fits(box.queue, box.gas_walltime if phase == "gas" else box.dm_walltime,
                     "halo %s %s" % (halo_id, ledger.stage_key(level, phase)))
    if os.path.exists(runscript):
        sync_runscript_exe(box, runscript, dry_run=dry_run)

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


def submit_qc_job(box, halo_id, density=False, through_level=None, dry_run=False):
    """Run the diagnostics on a compute node.  They need yt and several GB.

    `density` selects the projected-density ladder rather than the particle
    contamination panels.  `through_level` is recorded in the ledger so the
    automatic trigger can tell whether the figure on disk already covers every
    level that has finished -- see ic_pipeline.qc_due.
    """
    halo_dir = box.halo_dir(halo_id)
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ic_pipeline.py")
    kind = "density" if density else "contamination"
    cmd = "python3 %s qc --halo %s" % (script, halo_id)
    if density:
        # Gas stages that have not reached z = 0 are skipped by the figure
        # itself, so asking for them costs nothing and means a finished gas run
        # appears without anyone having to remember to add the flag.
        cmd += " --density --include-gas"
    text = replace_keywords(
        open(os.path.join(box.template_dir_path(), "BuildScript.sh")).read(),
        {"__JOBNAME__": "qc%s-halo%s" % ("dens" if density else "", halo_id),
         "__GROUP__": box.group_list,
         "__BUILD_SELECT__": box.build_select,
         "__BUILD_WALLTIME__": getattr(box, "qc_walltime", box.build_walltime),
         "__BUILD_QUEUE__": box.build_queue,
         "__HALO_DIR__": halo_dir,
         "__BUILD_CMD__": cmd})
    path = os.path.join(halo_dir, "QCScript%s.sh" % ("-density" if density else ""))
    if dry_run:
        print("    [dry-run] write %s and qsub it" % path)
        return None
    write_file(path, text, dry_run=False)
    os.chmod(path, 0o755)
    jobid = subprocess.check_output(["qsub", os.path.basename(path)],
                                    cwd=halo_dir).decode().strip()
    ledger.append_record(halo_dir, {"stage": "qc", "action": "qc-%s" % kind,
                                    "jobid": jobid, "through_level": through_level})
    print("  submitted %s" % jobid)
    return jobid
