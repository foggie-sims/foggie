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
    enzo_exe: str = "/nobackupnfs1/jtumlins/enzo-foggie-aitken-mpich/src/enzo/enzo.exe"
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
        max_level=3,
        omega_b=0.04576,
        omega_m=0.291,
        rvir_floor_kpc=None,
    ),
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


def stage_plan(row, include_gas=False):
    """Ordered list of (level, phase) stages for one registry row.

    The DM ladder always runs L1..final_level.  The gas stage is appended only
    when the row asks for it *and* the caller has enabled the gas phase, so the
    column can be curated before the feature is switched on.
    """
    stages = [(level, "DM") for level in range(1, int(row["final_level"]) + 1)]
    if include_gas and bool(row["gas"]):
        stages.append((int(row["final_level"]), "gas"))
    return stages
