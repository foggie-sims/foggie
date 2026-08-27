"""
Box definitions and the halo registry for the IC pipeline.

Everything that used to be a hardcoded literal in script512.py / script256.py
lives here: the parent box resolution, the halo catalog, the MUSIC and Enzo
binaries, the baryon fraction.  Adding a new parent box means adding an entry
to BOXES, not copying a script.
"""

import os
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def foggie_repo():
    """Root of the FOGGIE python package (the directory holding initial_conditions/)."""
    p = os.getenv("FOGGIE_REPO")
    if not p:
        raise RuntimeError(
            "FOGGIE_REPO is not set.  It must point at the foggie package directory, "
            "e.g. /nobackupnfs1/jtumlins/foggie/foggie")
    return p.rstrip("/")


def foggie_ics_dir():
    """Root of the simulation tree holding the parent box and the halo directories."""
    p = os.getenv("FOGGIE_ICS_DIR")
    if not p:
        raise RuntimeError(
            "FOGGIE_ICS_DIR is not set.  It must point at the ICs/run tree, "
            "e.g. /nobackupnfs1/jtumlins/25Mpc_new_cosmology")
    return p.rstrip("/")


# ---------------------------------------------------------------------------
# Boxes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Box:
    """A parent (L0) simulation box that zooms are cut out of."""

    sim_name: str
    # Root grid size of the parent box.  Level shifts reported by MUSIC are in
    # units of the parent grid, so converting them to code units divides by
    # (parent_ngrid - 1) -- 511 for a 512 box, 255 for a 256 box.  This was the
    # bare literal `/511.` and `/255.` in the old scripts.
    parent_ngrid: int
    boxsize_mpc: float
    # Rockstar catalog, relative to FOGGIE_REPO.
    catalog: str
    # MUSIC config for the parent box, relative to FOGGIE_ICS_DIR.
    template_config: str
    # Template directory, relative to FOGGIE_REPO.  Shared: one DM-LX.enzo and
    # one gas-LX.enzo serve every box, with the box-dependent parts -- the root
    # grid and the output cadence -- substituted at render time.
    template_dir: str
    max_level: int
    # Baryon fraction handed to MUSIC for gas ICs.  Must agree with
    # CosmologySimulationOmegaBaryonNow in the gas .enzo template; the old
    # convert_to_gas() hardcoded 0.0461 while the templates declared 0.04576.
    omega_b: float
    omega_m: float
    # Rvir floor in kpc, or None for no floor.  script256.py floored at 200 kpc
    # and script512.py did not; that difference was silent.
    rvir_floor_kpc: float = None
    # MinimumOverDensityForRefinement at the root grid, from which every level's
    # value follows by dividing by 8.  This number IS the refinement threshold
    # expressed in finest-zoom-particle masses: the (L_box/TopGridDims)^3 factor
    # Enzo applies (CosmologySimulationInitialize.C:718-731) cancels against the
    # root particle mass, so it means the same thing in any parent box.
    #
    # 8 = 2^3 is the Lagrangian choice: a cell that refines yields eight
    # children holding ~1 particle each, the same sampling the root grid had, so
    # each level reproduces the one above it.  4 would refine earlier and leave
    # each child with ~0.5 particles.
    #
    # 8 is what both 25 Mpc parent boxes use and what FOGGIE production runs --
    # halo_008508's L4 is 0.001953125 = 8**-(4-1) on a 256 root grid.  The 4s on
    # disk are resolution-test variants (25Mpc_DM_256-L0-max*), plus halo15097's
    # hand-built ladder, which mixes both conventions across its own levels.
    overdensity_at_root: float = 8.0
    # Locate the halo in the previous level's z = 0 output and trace the
    # Lagrangian region from there, rather than from the catalog position plus
    # the MUSIC shift.  The halo drifts 100-200 kpc from that analytic position
    # by z = 0 -- see build.refine_center_from_run -- which for a dwarf can be
    # several Rvir and can make the traced sphere miss the halo entirely.
    # Needs yt, which the build job has because enzo-mrp-music itself uses it.
    refine_centers: bool = True
    # Output-redshift cadence, as filenames in the template directory.  Held
    # here rather than inline in the templates because it is the one parameter
    # where two boxes legitimately disagree while sharing everything else.
    #
    # TESTING CADENCE, 2026-08-22: gas runs take the same 15 science redshifts
    # as the DM ladder instead of 266.  266 dumps is 12-17 TB per gas zoom and a
    # dozen are running at once while the physics is still being tested; 15 is
    # enough to see the growth history at a twentieth of the disk.  Put
    # gas_output_list back to "outputs_266.txt" for production.
    #
    # outputs_16_gas.txt is those 15 redshifts PLUS one at z = 15.0.  That extra
    # dump is not science, it is structural: gas runs are two legs joined at
    # gas_stop_redshift = 15, and simrun.pl restarts the second leg from the
    # LAST line of OutputLog.  outputs_266.txt happens to contain z = 15.0
    # (index 6), so the handoff has always had a dump to resume from.  The bare
    # 15-entry list jumps 50 -> 10, so the second leg would have resumed from
    # z = 50 and re-run 50 -> 15 with self-shielded cooling switched on -- a
    # silent physics change, not a cadence change.  Keep a dump at
    # gas_stop_redshift in any list this parameter points at.
    #
    # Changing this mid-ladder makes levels incomparable -- the same RD number
    # then means a different redshift -- so runs already on disk keep the
    # cadence they were built with.  Only newly built gas stages are affected.
    dm_output_list: str = "outputs_15.txt"
    gas_output_list: str = "outputs_16_gas.txt"
    # enzo-foggie-feedback-fix (branch fcf-on-cassi) carries the cic_deposit
    # out-of-bounds fix -- a subgrid deposited onto a coarser parent ran one
    # k-plane past the end of the field array when cloudsize < cellsize, which
    # is a deep-hierarchy condition and showed up as a segfault inside
    # RebuildHierarchy at level 5 on the L3 stages -- plus the mechanical
    # feedback and PPM NaN-laundering fixes on top of it.  Do not point this
    # back at a tree without both.  Keep templates/simrun.pl's fallback in sync.
    enzo_exe: str = "/home1/jtumlins/nobackup/enzo-foggie-feedback-fix/src/enzo/enzo.exe"
    music_exe_dir: str = None       # defaults to FOGGIE_REPO/initial_conditions/music
    email: str = "tumlinson@stsci.edu"
    group_list: str = "s3128"
    # Queue for the Enzo runs.  An explicit queue is now REQUIRED: a bare qsub
    # is refused with "No queue specified" and exit status 32, which is how the
    # poller silently failed to start eight runs on 2026-08-21 -- every IC build
    # succeeded (those carry build_queue) and every Enzo submission after it was
    # rejected.  Leaving this empty emits no "#PBS -q" line and no longer works,
    # whatever the hand-built scripts used to do.  'long' allows 120 h, which is
    # what gas_walltime asks for; a queue whose cap is below the walltime gets
    # the job rejected instead, so check_queue_fits still guards that.
    queue: str = "long"
    # PBS resources for the Enzo runs.
    dm_select: str = "1:ncpus=64:mpiprocs=64:model=mil_ait"
    dm_nranks: int = 64
    dm_walltime: str = "24:00:00"
    # Full node, 128 ranks.  This is what every completed gas run used:
    # halo15134's L3-gas and halo42189-manual's L3-gas-radius3 both ran -np 128
    # on 1:ncpus=128:mpiprocs=128 to 266 and 258 outputs respectively.
    #
    # Do not read halo42177's L2-gas PBS line as a counterexample.  It says
    # mpiprocs=16 but its mpiexec line says -np 128, and the mpiexec line is
    # what sets the rank count; the select line there is vestigial.  No gas run
    # on disk has ever completed on fewer than 64 ranks.
    gas_select: str = "1:ncpus=128:mpiprocs=128:model=mil_ait"
    gas_nranks: int = 128
    gas_walltime: str = "120:00:00"
    # Maximum refinement level for gas runs.  Feeds MaximumRefinementLevel,
    # MaximumGravityRefinementLevel and MaximumParticleRefinementLevel alike
    # (__MAX_REFINE_LEVEL__).  7 while the gas path is being tested, matching
    # every hand-built gas run on disk; 9 is the eventual target and should be
    # adopted deliberately, not drifted into.
    gas_max_refine_level: int = 7
    # MaximumRefinementLevel counts levels above the ROOT grid, so the value
    # that gives a particular cell size depends on BOTH the root grid and how
    # deep the zoom goes.  Holding it fixed per box was correct only while every
    # halo in a box shared one final_level; with halos at L2 and L3 in the same
    # box it silently gives them different cell-to-particle-spacing ratios.
    #
    # The invariant that actually matters is the ratio of the finest cell to the
    # zoom's mean particle spacing, and it is the same for every run on disk:
    #     512 root, L2 zoom, nref7  -> 2^5 cells per particle spacing, 545 pc
    #     256 root, L3 zoom, nref8  -> 2^5, 545 pc
    # so the rule is nref = final_level + GAS_REFINE_OFFSET.  Deriving it means
    # a halo promoted from L2 to L3 gets nref8 automatically, and one left at L2
    # keeps nref7 -- rather than a box-wide constant handing L2 halos a
    # resolution their dark matter cannot support.
    gas_refine_offset: int = 5
    # Gas output cadence depends on what the run is FOR, not just on the box.
    # 16 outputs is enough to watch a galaxy evolve and was adopted when disk
    # was short, but it is far too sparse to build a forced-refinement track
    # from: Enzo interpolates the refine box linearly between track rows, and
    # on the 16-entry list the halo moves a median 242 ckpc/h between rows
    # against a 50 ckpc/h box half-width -- every single step leaves the box
    # behind.  The 100-entry list is a strict SUBSET of outputs_266.txt, so its
    # dumps sit at redshifts the existing L2 runs already have and the two are
    # directly comparable; it gives a median step of 29 ckpc/h and the same
    # maximum (49) as the full 266, because that ceiling is set by the halo's
    # motion rather than by the sampling.
    gas_output_list_deep: str = "outputs_100_gas.txt"
    # Zoom depth at or beyond which the deep list is used.  L3 zooms are the
    # ones we intend to run forced-refinement boxes on.
    gas_deep_level: int = 3
    # Gas runs are done in two legs.  Grackle's cooling changes character at
    # z = 15, where self-shielding starts to matter, so the first leg runs with
    # unshielded cooling and stops there; the parameters below are then written
    # into the restart file and the second leg continues to z = 0.  This was
    # previously a hand edit between two submissions -- halo42189-manual's
    # radius3 run shows RD0000 at z=99 with all three shielding parameters 0 and
    # RD0257 at z=0.06 with 1/3/1, but nothing recorded that it had to be done.
    #
    # gas_stop_redshift is CosmologyFinalRedshift for the first leg.  The pairs
    # are applied verbatim to the restart parameter file via simrun.pl's
    # `new_pars` mechanism, so the names must match ReadParameterFile.C exactly:
    # a misspelling is not an error, it is silently ignored.
    gas_stop_redshift: float = 15.0
    # Least walltime worth starting the second leg with inside the same job.
    # Below this the transition still happens, but the leg is left to a fresh
    # submission rather than burning the remainder on a run that cannot reach
    # an output.
    gas_transition_min_seconds: int = 1800
    #
    # grackle_data_file is part of the transition and not an afterthought:
    # self_shielding_method = 3 needs the shielding datasets under
    # /UVBRates/CrossSections/, which CloudyData_UVB=HM2012.h5 does not carry.
    # Grackle does not fall back -- it aborts in initialize_chemistry_data with
    # "In order to use self-shielding, you must use the shielding datasets", so
    # the second leg dies seconds after restarting.  The hand-built radius3 run
    # switched the table at the same restart it switched the other four; its
    # RD0000 names the plain table and its RD0006, at z = 15, the shielded one.
    gas_transition_pars: tuple = (
        ("H2FormationOnDust", "1"),
        ("self_shielding_method", "3"),
        ("H2_self_shielding", "1"),
        ("CosmologyFinalRedshift", "0"),
        ("grackle_data_file",
         "/u/jtumlins/grackle/grackle-3.3.1-dev/grackle_data_files/input/"
         "CloudyData_UVB=HM2012_shielded.h5"),
    )
    # Fraction of the PBS walltime after which Enzo writes a restart dump and
    # stops.  Must leave room to write the dump before the scheduler kills the
    # job; 0.9 of 24 h is 21.6 h.
    restart_dump_fraction: float = 0.9
    # PBS resources for the IC-generation job.  enzo-mrp-music loads the parent
    # box in yt and traces particles back to z = 99, so it needs a whole node
    # and must never run on a login node.
    build_select: str = "1:ncpus=64:mpiprocs=1:model=mil_ait"
    build_walltime: str = "2:00:00"
    # normal, not devel.  devel allows one job per user at a time
    # (max_queued = max_run = [u:PBS_GENERIC=1]), so a batch of IC builds
    # serialises behind itself and, worse, occupies the one devel slot that
    # interactive and test work needs.  normal caps at 8 h, comfortably above
    # the 2 h a build needs, and imposes no per-user run limit here.
    build_queue: str = "normal"
    # Poller resources.  Deliberately NOT devel: a poller waking every 30
    # minutes must not consume the single devel slot, which is reserved for
    # interactive and test work rather than anything this pipeline submits.
    poll_select: str = "1:ncpus=1:model=mil_ait"
    poll_walltime: str = "00:10:00"
    poll_queue: str = "normal"
    # A qc job wants a node's worth of memory -- it loads a 134-million-particle
    # snapshot -- but only minutes of it.  Asking for the build job's two hours
    # would queue it behind work that needs them.
    qc_walltime: str = "00:30:00"
    # Regenerate the projected-density ladder whenever a DM level finishes.
    # It is one short job per completed level, submitted after the ladder has
    # been advanced so it never delays one, and it costs about a minute per
    # panel.  Set False to make the figures purely manual.
    qc_on_advance: bool = True

    @property
    def shift_divisor(self):
        return float(self.parent_ngrid - 1)

    def catalog_path(self):
        return os.path.join(foggie_repo(), self.catalog)

    def template_dir_path(self):
        return os.path.join(foggie_repo(), self.template_dir)

    def template_config_path(self):
        return os.path.join(foggie_ics_dir(), self.template_config)

    def music_exe_dir_path(self):
        if self.music_exe_dir:
            return self.music_exe_dir
        return os.path.join(foggie_repo(), "initial_conditions", "music")

    def halo_dir(self, halo_id):
        return os.path.join(foggie_ics_dir(), "halo%s" % halo_id)

    def stage_dirname(self, level, phase="DM"):
        suffix = "-gas" if phase == "gas" else ""
        return "%s-L%d%s" % (self.sim_name, level, suffix)

    def stage_dir(self, halo_id, level, phase="DM"):
        return os.path.join(self.halo_dir(halo_id), self.stage_dirname(level, phase))

    def param_filename(self, level, phase="DM"):
        return "%s.enzo" % self.stage_dirname(level, phase)

    def gas_refine_level(self, level):
        """MaximumRefinementLevel for a gas run whose zoom reaches `level`."""
        return int(level) + int(self.gas_refine_offset)

    def gas_outputs_for(self, level):
        """Which output-redshift list a gas run at this zoom depth should use."""
        if int(level) >= int(self.gas_deep_level):
            return self.gas_output_list_deep
        return self.gas_output_list


BOXES = {
    "25Mpc_DM_512": Box(
        sim_name="25Mpc_DM_512",
        parent_ngrid=512,
        boxsize_mpc=25.0,
        catalog="initial_conditions/halo_catalogs_512/512/z0/out_0.list",
        template_config="25Mpc_DM_512_planck18.conf",
        template_dir="initial_conditions/templates",
        # 4 levels, not 3: halo80181 ran a full L1-L4 ladder successfully.
        max_level=4,
        omega_b=0.04576,
        omega_m=0.291,
        # Floor the zoom radius at 80 kpc: max(catalog Rvir, 80).  Several of
        # these dwarfs have Rvir well under that (halo79628 is 25.8 kpc,
        # halo11177 is 33.7) and too small a Lagrangian region makes a poor
        # zoom.  This generalises the hand-built halo11177 config, which used
        # 80 kpc against a catalog Rvir of 33.66.  Halos already above the
        # floor are untouched, so halo42189 keeps its 88.963.
        rvir_floor_kpc=80.0,
        # Deepen gas one level past the DM-matched value (2026-08-26, user):
        # offset=6 gives L2->nref8 and L3->nref9 (2^6 cells/particle spacing),
        # the eventual production target and the natural-run precursor to the
        # nref9c cooling ceiling of the forced runs. Both are intended: L2 gas
        # is nref8, L3 gas is nref9. The L2-nref7 straggler runs already on disk
        # keep their cadence (they never rebuild); new gas builds follow this.
        gas_refine_offset=6,
    ),
    # Shares templates/ with the 512 box.  The only differences a second parent
    # box actually needs are the root grid and the output cadence, both
    # substituted at render time, so there is no templates_256/ to keep in sync.
    #
    # parent_ngrid is the one to be careful with: MUSIC's shifts divide by 255
    # here and by 511 for the 512 box, and getting it wrong displaces the zoom
    # silently.
    #
    # This box's own hand-written L3 and L4 templates disagree with each other
    # about MinimumOverDensityForRefinement -- L3 carries L4's value and L4
    # carries L3's, transposed, on a line whose comment states the divide-by-8
    # rule they break.  The rendered files follow the rule, so
    # `validate-templates --box 25Mpc_DM_256 --original` reports those two as
    # differences.  That is the check working, not a porting error.
    "25Mpc_DM_256": Box(
        sim_name="25Mpc_DM_256",
        parent_ngrid=256,
        boxsize_mpc=25.0,
        catalog="initial_conditions/halo_catalogs_256/256/z0/out_0.list",
        template_config="25Mpc_DM_256_planck18.conf",
        template_dir="initial_conditions/templates",
        # DM output cadence is the 15-entry list, as for the 512 box.  Its
        # hand-written templates carried all 266, which cost ~300 GB per DM
        # level against ~17 GB at 15 outputs, for dumps nothing reads: the DM
        # ladder exists to place the zoom region, and only its z = 0 state is
        # used.  Gas keeps 266, where the cadence is the analysis.
        # One level deeper than the 512 box, for the same reason its zooms run
        # one level deeper: MaximumRefinementLevel counts levels above the ROOT
        # grid, so the same value on a 256 root gives half the peak resolution.
        # 512 at 7 and 256 at 8 both reach 0.545 proper kpc; leaving this at 7
        # would quietly halve the resolution of every 256 gas run.
        gas_max_refine_level=8,
        max_level=4,
        omega_b=0.04576,
        omega_m=0.291,
        rvir_floor_kpc=200.0,
    ),
}

DEFAULT_BOX = "25Mpc_DM_512"


def get_box(name):
    if name not in BOXES:
        raise KeyError("Unknown box %r.  Known boxes: %s" % (name, ", ".join(sorted(BOXES))))
    return BOXES[name]


def box_problems(box):
    """Missing pieces that would make this box fail, as a list of strings.

    A box entry can reference templates or a catalog that were never put in
    place.  Reporting that up front beats a confusing failure inside a build
    job on a compute node.
    """
    problems = []
    template_dir = box.template_dir_path()
    if not os.path.isdir(template_dir):
        problems.append("template directory missing: %s" % template_dir)
    else:
        for f in ("DM-LX.enzo", "gas-LX.enzo", "RunScript.sh", "simrun.pl",
                  "halo_DM_NtoN.conf"):
            if not os.path.exists(os.path.join(template_dir, f)):
                problems.append("template missing: %s" % os.path.join(template_dir, f))
    if not os.path.exists(box.catalog_path()):
        problems.append("halo catalog missing: %s" % box.catalog_path())
    # Any file named in the gas transition has to exist before the run reaches
    # z = 15, not after.  Getting this wrong costs the whole first leg: the
    # handoff fires, Grackle refuses the table, and the second leg dies seconds
    # into a restart that already consumed hours of compute.
    for key, value in box.gas_transition_pars:
        if key.endswith("_file") and not os.path.exists(value):
            problems.append("gas transition %s missing: %s" % (key, value))
    return problems


# ---------------------------------------------------------------------------
# Halo registry
# ---------------------------------------------------------------------------

# rvir_min: per-halo floor on the zoom radius in kpc, 0 meaning "use the
# catalog Rvir".  Exists because halo11177 was hand-built at 80 kpc against a
# catalog Rvir of 33.66 kpc via an --rvir_min flag that no script implemented.
# queue/nodes/model used to live here too, but every row carried the same three
# values -- normal, 1, mil_ait -- and nothing ever read them: scheduling comes
# from the Box dataclass above (queue, dm_select, gas_select, build_select,
# build_queue, poll_select, poll_queue), which already names the model and the
# node count. Dropped 2026-08-21; a registry that still has the columns reads
# fine, the extra ones are simply ignored.
REGISTRY_COLUMNS = ("halo_id", "box", "enabled", "final_level", "gas",
                    "rvir_min", "notes")


def default_registry_path():
    """The live registry, in the run tree rather than in the repo.

    It lives under FOGGIE_ICS_DIR so that several people can drive the pipeline
    from one checkout without editing the same versioned file: each run tree
    carries its own set of halos.  The repo ships a seed copy, see
    registry_example_path().
    """
    return os.path.join(foggie_ics_dir(), "halo_registry.ecsv")


def registry_example_path():
    """The seed registry in the repo.  Never read by the pipeline; copied once."""
    return os.path.join(foggie_repo(), "initial_conditions",
                        "halo_registry.ecsv.example")


def read_registry(path=None):
    """Read the curated halo registry.  Returns an astropy Table."""
    from astropy.table import Table

    path = path or default_registry_path()
    if not os.path.exists(path):
        raise RuntimeError(
            "Halo registry not found: %s\n"
            "The registry lives in the run tree, not the repo, so that each run "
            "tree has its own set of halos.  Seed it once with:\n"
            "    cp %s \\\n        %s\n"
            "then edit that copy -- every row in it starts disabled."
            % (path, registry_example_path(), path))
    # Explicit format: astropy infers it from the extension, which fails for a
    # registry kept under any other name -- the shipped seed is .ecsv.example.
    table = Table.read(path, format="ascii.ecsv")

    missing = [c for c in REGISTRY_COLUMNS if c not in table.colnames]
    if missing:
        raise RuntimeError("Registry %s is missing column(s): %s" % (path, ", ".join(missing)))
    return table


def enabled_halos(table):
    """Rows the pipeline is allowed to act on."""
    return [row for row in table if bool(row["enabled"])]


def dm_ladder(row):
    """The sequential DM stages, L1 .. final_level."""
    return [(level, "DM") for level in range(1, int(row["final_level"]) + 1)]


def gas_stage(row, include_gas=True):
    """The gas stage for a row, or None.  Gas runs at the final level only."""
    if not include_gas or not bool(row["gas"]):
        return None
    return (int(row["final_level"]), "gas")


def gas_prerequisite(box, halo_id, level):
    """The file the gas stage needs before it can be built.

    Gas ICs are made by running MUSIC directly on the DM MUSIC config for the
    same level with baryons switched on, so the gas stage depends on that
    config existing.  The level-N DM *build* writes it, which in turn requires
    level N-1's Enzo run to have finished -- L2 must be done before L3-gas is
    possible.

    It does NOT depend on level N's own Enzo run.  That is what lets the gas
    run proceed alongside the DM run at the same level instead of queueing
    behind it.
    """
    return os.path.join(box.halo_dir(halo_id), "%s-L%d.conf" % (box.sim_name, level))


def gas_ready(box, halo_id, level):
    """True once the gas stage's prerequisite config exists."""
    return os.path.exists(gas_prerequisite(box, halo_id, level))


def stage_plan(row, include_gas=True):
    """All stages for one registry row: the DM ladder, then gas.

    These are not one sequential chain.  The gas stage hangs off the DM *build*
    at the same level, not off its run, so it can be in flight at the same time
    as the DM run.  See gas_prerequisite().
    """
    stages = dm_ladder(row)
    gas = gas_stage(row, include_gas)
    if gas:
        stages.append(gas)
    return stages
