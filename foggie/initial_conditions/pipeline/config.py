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
    # Template directory, relative to FOGGIE_REPO.
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
    # enzo-cic_deposit_fix is enzo-foggie-aitken-mpich plus 0713af80, which
    # fixes an out-of-bounds write in cic_deposit.F when cloudsize < cellsize
    # -- a subgrid deposited onto a coarser parent ran one k-plane past the end
    # of the field array.  That is a deep-hierarchy condition, which is why it
    # showed up as a segfault inside RebuildHierarchy at level 5 and only on
    # the L3 stages.  Do not point this back at the unfixed tree.
    enzo_exe: str = "/nobackupnfs1/jtumlins/enzo-cic_deposit_fix/src/enzo/enzo.exe"
    music_exe_dir: str = None       # defaults to FOGGIE_REPO/initial_conditions/music
    email: str = "tumlinson@stsci.edu"
    group_list: str = "s3128"
    # Queue for the Enzo runs.  Empty means emit no "#PBS -q" line and let PBS
    # route on walltime, which is what the hand-built scripts do.  Naming a
    # queue whose walltime cap is below dm_walltime gets the job rejected.
    queue: str = ""
    # PBS resources for the Enzo runs.
    dm_select: str = "1:ncpus=64:mpiprocs=64:model=mil_ait"
    dm_nranks: int = 64
    dm_walltime: str = "24:00:00"
    gas_select: str = "1:ncpus=128:mpiprocs=128:model=mil_ait"
    gas_nranks: int = 128
    gas_walltime: str = "120:00:00"
    # Maximum refinement level for gas runs (__MAX_REFINE_LEVEL__).
    gas_max_refine_level: int = 9
    # Fraction of the PBS walltime after which Enzo writes a restart dump and
    # stops.  Must leave room to write the dump before the scheduler kills the
    # job; 0.9 of 24 h is 21.6 h.
    restart_dump_fraction: float = 0.9
    # PBS resources for the IC-generation job.  enzo-mrp-music loads the parent
    # box in yt and traces particles back to z = 99, so it needs a whole node
    # and must never run on a login node.
    build_select: str = "1:ncpus=64:mpiprocs=1:model=mil_ait"
    build_walltime: str = "2:00:00"
    build_queue: str = "devel"
    # Poller resources.  Deliberately NOT devel: the IC build jobs use devel,
    # and a poller waking every 30 minutes should not compete with them for it.
    poll_select: str = "1:ncpus=1:model=mil_ait"
    poll_walltime: str = "00:10:00"
    poll_queue: str = "normal"

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


BOXES = {
    "25Mpc_DM_512": Box(
        sim_name="25Mpc_DM_512",
        parent_ngrid=512,
        boxsize_mpc=25.0,
        catalog="initial_conditions/halo_catalogs_512/512/z0/out_0.list",
        template_config="25Mpc_DM_512_planck18.conf",
        template_dir="initial_conditions/templates_512",
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
    ),
    # NOT YET PORTED.  The 256 box needs templates_256/ (the collapsed .enzo
    # templates, RunScript and simrun.pl, as done for 512) and its Rockstar
    # catalog put in place.  Kept here so the shape of a second box is visible,
    # but `validate-registry` will refuse a halo that names it until the files
    # exist.  The parent_ngrid difference is the important one: shifts divide by
    # 255 here and 511 for the 512 box.
    "25Mpc_DM_256": Box(
        sim_name="25Mpc_DM_256",
        parent_ngrid=256,
        boxsize_mpc=25.0,
        catalog="initial_conditions/halo_catalogs_256/256/z0/out_0.list",
        template_config="25Mpc_DM_256_planck18.conf",
        template_dir="initial_conditions/templates_256",
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
    return problems


# ---------------------------------------------------------------------------
# Halo registry
# ---------------------------------------------------------------------------

# rvir_min: per-halo floor on the zoom radius in kpc, 0 meaning "use the
# catalog Rvir".  Exists because halo11177 was hand-built at 80 kpc against a
# catalog Rvir of 33.66 kpc via an --rvir_min flag that no script implemented.
REGISTRY_COLUMNS = ("halo_id", "box", "enabled", "final_level", "gas",
                    "rvir_min", "queue", "nodes", "model", "notes")


def default_registry_path():
    return os.path.join(foggie_repo(), "initial_conditions", "halo_registry.ecsv")


def read_registry(path=None):
    """Read the curated halo registry.  Returns an astropy Table."""
    from astropy.table import Table

    path = path or default_registry_path()
    if not os.path.exists(path):
        raise RuntimeError("Halo registry not found: %s" % path)
    table = Table.read(path)

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
