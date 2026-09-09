"""Multi-halo configuration parsing for the multizoom pipeline.

Forked from foggie/initial_conditions/enzo-mrp-music/enzo-mrp-music.py
(parse_config); extended with [halo:<id>] sections so one config can
describe N zoom targets.  The legacy single-halo layout (halo keys in
[region]) still parses and yields a single pseudo-halo, so existing
configs work unchanged.

Config layout::

    [setup]
    music_exe_dir = /path/to/music
    simulation_name = 25Mpc_DM_512
    template_config = /path/to/25Mpc_DM_512_planck18.conf
    original_config = /path/to/25Mpc_DM_512_planck18.conf
    simulation_run_directory = /path/to/rundir
    num_cores = None
    mode = union                ; union | merge
    music_env =                 ; optional KEY=VALUE;KEY=VALUE for MUSIC env

    [region]                    ; defaults inherited by every halo
    final_redshift = 0.0
    radius_factor = 1.0
    shape_type = exact
    halo_center_units = code_length
    halo_radius_units = kpc
    halo_mass_units = Msun/h

    [halo:5016]
    halo_center = 0.493, 0.508, 0.461
    halo_radius = 205.

    [halo:5033]
    halo_center = 0.201, 0.774, 0.312
    halo_mass = 8.3e11
"""

import configparser
import multiprocessing
import os
import warnings
from collections import OrderedDict

import numpy as np

HALO_SECTION_PREFIX = "halo:"

SETUP_DEFAULTS = dict(
    music_exe_dir=".",
    simulation_name="auto-wrapper",
    template_config="template.conf",
    original_config=None,
    simulation_run_directory=".",
    # Where the NEW ICs for this level are written.  Kept separate from
    # simulation_run_directory so a zoom reads its parent level from the shared
    # ICs directory while depositing its own ICs in the group directory.
    # Matches the ics_refactor pipeline's enzo-mrp-music option of the same
    # name; defaults to "." so older configs behave as before.
    new_ics_directory=".",
    num_cores=None,
    mode="union",
    # One common domain shift (integer coarse cells, "sx,sy,sz") shared by
    # every run of a merge-mode group, in place of no_shift.  Needs a MUSIC
    # built with music_patches/0001-region-shift-override.patch.  Used when
    # a target's Lagrangian region sits too close to the periodic boundary
    # for no_shift.
    region_shift_override=None,
    music_env=None,
    # Runtime environment for the MUSIC subprocess (ported from ics_refactor,
    # which replaced the hard-coded Pleiades path with a config option).
    music_ld_library_path="/nasa/hdf5/1.8.18_serial/lib:/u/jtumlins/installs/gsl-2.4/lib",
)

REGION_DEFAULTS = dict(
    final_type="halo",
    final_redshift=0.0,
    halo_center=None,
    halo_center_units="code_length",
    halo_mass=None,
    halo_mass_units="Msun/h",
    halo_radius=None,
    halo_radius_units="kpc",
    radius_factor=3.0,
    shape_type="box",
)

VALID_SHAPES = ("box", "ellipsoid", "convex_hull", "exact")


def _clean(value):
    if isinstance(value, str) and value.strip() in ("None", ""):
        return None
    return value


def _build_halo_info(halo_id, keys):
    """Build one halo_info dict from a merged {key: value} mapping."""
    center = _clean(keys.get("halo_center"))
    mass = _clean(keys.get("halo_mass"))
    radius = _clean(keys.get("halo_radius"))
    if center is None or (mass is None and radius is None):
        raise RuntimeError(
            "Halo %s properties not set (center plus either radius or mass "
            "are required).\n\t Center: %s\n\t Mass: %s\n\t Radius: %s"
            % (halo_id, center, mass, radius))

    info = dict(
        id=halo_id,
        center=(np.array([float(p) for p in str(center).split(",")]),
                keys.get("halo_center_units", "code_length")),
        redshift=float(keys.get("final_redshift", 0.0)),
        radius_factor=float(keys.get("radius_factor", 3.0)),
    )
    if mass is not None and radius is not None:
        warnings.warn("Halo %s: mass and radius both set. Defaulting to mass."
                      % halo_id)
        radius = None
    if mass is not None:
        info["mass"] = (float(mass), keys.get("halo_mass_units", "Msun/h"))
    if radius is not None:
        info["radius"] = (float(radius), keys.get("halo_radius_units", "kpc"))
    return info


def parse_multizoom_config(config_fn):
    """Parse a multizoom config file.

    Returns a params dict with the [setup]/[region] keys flattened (as the
    legacy wrapper did) plus:

    params["halos"]   OrderedDict {halo_id(str): halo_info dict}
    params["tag"]     "+".join(halo ids), used in file/dir names
    """
    if not os.path.exists(config_fn):
        raise RuntimeError("Config file not found: %s" % config_fn)

    cf = configparser.ConfigParser()
    cf.optionxform = str.lower
    cf.read(config_fn)

    params = dict(SETUP_DEFAULTS)
    params.update(REGION_DEFAULTS)
    for section in cf.sections():
        if section.lower().startswith(HALO_SECTION_PREFIX):
            continue
        for k, v in cf.items(section):
            params[k] = _clean(v)

    params["radius_factor"] = float(params["radius_factor"])
    params["final_redshift"] = float(params["final_redshift"])
    if params["num_cores"] is not None:
        params["num_cores"] = int(params["num_cores"])
    else:
        params["num_cores"] = multiprocessing.cpu_count()
    if params["shape_type"] not in VALID_SHAPES:
        raise RuntimeError("shape_type = %s not in %s"
                           % (params["shape_type"], list(VALID_SHAPES)))

    # Per-halo defaults come from the flattened [region] keys.
    region_defaults = {k: params[k] for k in REGION_DEFAULTS}

    halos = OrderedDict()
    halo_sections = [s for s in cf.sections()
                     if s.lower().startswith(HALO_SECTION_PREFIX)]
    for section in halo_sections:
        halo_id = section.split(":", 1)[1].strip()
        if not halo_id:
            raise RuntimeError("Empty halo id in section [%s]" % section)
        if halo_id in halos:
            raise RuntimeError("Duplicate halo section [%s]" % section)
        keys = dict(region_defaults)
        for k, v in cf.items(section):
            keys[k] = _clean(v)
        halos[halo_id] = _build_halo_info(halo_id, keys)

    if not halo_sections:
        # Legacy single-halo layout: halo keys live in [region].
        halo_id = str(_clean(params.get("halo_id")) or "0")
        halos[halo_id] = _build_halo_info(halo_id, region_defaults)

    if len(halos) > 1 and params["shape_type"] != "exact":
        # Any hull/ellipsoid spanning several clouds would mask-refine the
        # space between the halos; only the exact particle mask keeps the
        # must-refine clouds per-halo.
        raise RuntimeError(
            "Multi-halo configs require shape_type = exact "
            "(got %s): a single hull spanning several Lagrangian clouds "
            "would refine the volume between the halos." % params["shape_type"])

    params["halos"] = halos
    params["tag"] = "+".join(halos.keys())
    params["mode"] = (params.get("mode") or "union").lower()
    if params["mode"] not in ("union", "merge"):
        raise RuntimeError("mode = %s (expected union or merge)" % params["mode"])
    return params


def parse_music_env(music_env):
    """Parse the optional music_env key ("KEY=VAL;KEY=VAL") into a dict."""
    env = {}
    if music_env:
        for item in music_env.split(";"):
            item = item.strip()
            if not item:
                continue
            if "=" not in item:
                raise RuntimeError("music_env entry %r is not KEY=VALUE" % item)
            k, v = item.split("=", 1)
            env[k.strip()] = v.strip()
    return env


def read_rockstar_catalog(catalog_fn):
    """Read a rockstar out_*.list into {column_name: numpy array}.

    The first line holds the column names (leading '#' stripped); all other
    '#' lines are comments.  Kept dependency-free (no astropy).
    """
    with open(catalog_fn) as fp:
        header = fp.readline().lstrip("#").split()
        rows = [line.split() for line in fp
                if line.strip() and not line.startswith("#")]
    if not rows:
        raise RuntimeError("No halo rows found in %s" % catalog_fn)
    data = np.array(rows, dtype=float)
    if data.shape[1] != len(header):
        raise RuntimeError(
            "Column mismatch in %s: %d names, %d columns"
            % (catalog_fn, len(header), data.shape[1]))
    return {name: data[:, i] for i, name in enumerate(header)}


def halo_from_catalog(catalog, halo_id, boxsize_mpch):
    """Extract one halo's center (code units) and Rvir (comoving kpc/h)."""
    match = catalog["ID"] == float(halo_id)
    if not match.any():
        raise RuntimeError("Halo ID %s not found in catalog" % halo_id)
    i = np.where(match)[0][0]
    center = np.array([catalog["X"][i], catalog["Y"][i], catalog["Z"][i]])
    return center / boxsize_mpch, float(catalog["Rvir"][i])
