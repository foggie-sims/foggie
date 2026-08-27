"""Per-level MUSIC orchestration for multi-halo zooms (union and merge).

Forked from foggie/initial_conditions/enzo-mrp-music/enzo-mrp-music.py with
these changes:

* N zoom targets per config ([halo:<id>] sections, see config.py);
* two modes:
  - union: ONE MUSIC run whose region covers the union of all Lagrangian
    point clouds; the RefinementMask is re-deposited from the individual
    clouds so only the halos (not the space between them) carry
    must-refine particles.  No MUSIC or Enzo code changes needed.
  - merge: one MUSIC run PER HALO with identical seeds and no_shift = yes,
    merged by merge_music_ics into a single multi-patch IC set (requires
    the Enzo patches in multizoom/enzo_patches/).
* the MUSIC executable comes from the config (music_exe_dir; the legacy
  script hard-coded a Pleiades path), run via subprocess with an optional
  music_env environment;
* raising-RuntimeWarning and os.system fragilities removed.

Usage: python mrp_music.py <config> <level>
"""

import configparser
import os
import subprocess
import sys

from . import config as mzconfig
from . import lagrangian_regions
from . import merge_music_ics
from . import refinement_mask

MRP_BLOCK = ("\n"
             "#\n"
             "# must-refine particle parameters\n"
             "# *** must also include method 8 in CellFlaggingMethod ***\n"
             "# *** do NOT include the RefineRegion parameters above ***\n"
             "#\n"
             "MustRefineParticlesCreateParticles = 3\n"
             "MustRefineParticlesRefineToLevel   = %d\n"
             "CosmologySimulationParticleTypeName          = RefinementMask\n")


def startup(config_fn, level):
    params = mzconfig.parse_multizoom_config(config_fn)
    params["level"] = int(level)
    if params["level"] == 0:
        raise RuntimeError("level must be >0. "
                           "Please run the unigrid simulation first.")
    music_exe = os.path.join(params["music_exe_dir"], "MUSIC")
    for f in [music_exe, params["template_config"],
              params["simulation_run_directory"]]:
        if not os.path.exists(f):
            raise RuntimeError("File/directory not found: %s" % f)
    if params["original_config"] is not None and \
            not os.path.exists(params["original_config"]):
        raise RuntimeError("File/directory not found: %s"
                           % params["original_config"])
    params["music_exe"] = music_exe
    if params.get("region_shift_override"):
        params["region_shift_override"] = [
            int(v) for v in str(params["region_shift_override"]).split(",")]
    return params


def get_previous_run_params(params):
    """Locate the previous level's run and recover levels and shift."""
    level = params["level"]
    rundir = params["simulation_run_directory"]
    params["prev_sim_dir"] = os.path.join(
        rundir, "%s-L%d" % (params["simulation_name"], level - 1))
    params["sim_dir"] = os.path.join(
        params["new_ics_directory"],
        "%s-L%d" % (params["simulation_name"], level))

    if params["original_config"] is None:
        original_config_file = "%s-L0.conf" % params["simulation_name"]
    else:
        original_config_file = params["original_config"]
    music_cf0 = configparser.ConfigParser()
    music_cf0.read(original_config_file)
    params["initial_min_level"] = music_cf0.getint("setup", "levelmin")
    params["initial_max_level"] = music_cf0.getint("setup", "levelmax")
    params["round_factor"] = 2**params["initial_max_level"]

    # Domain shift of the previous run.  In merge mode every run uses
    # no_shift, so all frames coincide and the shift is identically zero.
    params["region_shift"] = [0, 0, 0]
    if params["mode"] == "union":
        if params["original_config"] is not None and level == 1:
            prev_log = "%s_log.txt" % params["original_config"]
        else:
            prev_log = os.path.join(
                rundir, "%s-L%d.conf_log.txt"
                % (params["simulation_name"], level - 1))
        if os.path.exists(prev_log):
            with open(prev_log) as fp:
                for line in fp:
                    if line.find("Domain shifted by") >= 0:
                        inner = line.split("(")[1].split(")")[0]
                        params["region_shift"] = \
                            [int(v) for v in inner.split(",")]

    import yt
    sim_par_file = os.path.join(
        params["prev_sim_dir"],
        "%s-L%d.enzo" % (params["simulation_name"], level - 1))
    print("Opening Enzo parameter file:", sim_par_file)
    es = yt.load_simulation(sim_par_file, "Enzo", find_outputs=True)
    params["enzo_initial_fn"] = es.all_outputs[0]["filename"]
    es.get_time_series(redshifts=[params["final_redshift"]])
    ds = es[0]
    params["enzo_final_fn"] = os.path.join(ds.directory, ds.basename)
    return params


def find_lagrangian_regions(params):
    """Trace every halo and (in union mode) build the union point file."""
    os.makedirs(params["new_ics_directory"], exist_ok=True)
    params["halo_regions"] = lagrangian_regions.get_centers_and_extents(
        params["halos"], params["enzo_initial_fn"], params["enzo_final_fn"],
        round_size=params["round_factor"],
        radius_factor=params["radius_factor"],
        output_format="txt",
        output_dir=params["new_ics_directory"])
    # Trim each cloud's far-flung strays BEFORE they are unioned or handed to
    # a per-halo MUSIC run, so one halo's outliers cannot inflate the whole
    # multizoom domain (ics_refactor added this for the single-halo case).
    for halo_id, region in params["halo_regions"].items():
        lagrangian_regions.trim_lagrangian_outliers(
            region["point_file"], label=halo_id)

    if params["mode"] == "union":
        union_fn = os.path.join(
            params["new_ics_directory"],
            lagrangian_regions.union_point_file_name(
                params["tag"], os.path.basename(params["enzo_initial_fn"])))
        lagrangian_regions.write_union_point_file(
            [r["point_file"] for r in params["halo_regions"].values()],
            union_fn)
        params["union_point_file"] = union_fn
    return params


def _final_frame_shift(params):
    """Code-units shift from the previous run's frame to this run's frame.

    With a region_shift_override every level of the group shares one frame:
    level 1 traces from the unshifted L0 (shift applies), deeper levels
    trace from the group's own shifted runs (frames coincide).  Without an
    override, merge mode is no_shift everywhere and this is zero.
    """
    import numpy as np
    override = params.get("region_shift_override")
    if override and params["level"] == 1:
        return np.array(override) / 2.0**params["initial_min_level"]
    return np.zeros(3)


def check_periodic_wrap(params, margin_coarse_cells=8):
    """In merge mode, no_shift keeps every cloud in the catalog frame; a
    cloud whose padded bounding box crosses the box boundary makes MUSIC
    fail ("Internal refinement bounding box error").  Detect that before
    running and point at the region_shift_override remedy."""
    margin = margin_coarse_cells / 2.0**params["initial_min_level"]
    frame_shift = _final_frame_shift(params)
    offenders = []
    for halo_id, region in params["halo_regions"].items():
        center = (region["center"] + frame_shift) % 1.0
        low = center - 0.5 * region["size"] - margin
        high = center + 0.5 * region["size"] + margin
        if (low < 0.0).any() or (high > 1.0).any():
            offenders.append(halo_id)
    if offenders:
        raise RuntimeError(
            "Halo(s) %s have Lagrangian regions within %d coarse cells of "
            "the periodic boundary; MUSIC cannot place their patches under "
            "no_shift = yes. Apply the region_shift_override patch "
            "(multizoom/music_patches/) to a build copy of MUSIC and set "
            "one common setup/region_shift_override for every run."
            % (offenders, margin_coarse_cells))


def write_music_conf(params, point_file, output_name, region_shift,
                     no_shift=False):
    """Write one MUSIC conf from the template (never modified in place)."""
    music_cf = configparser.ConfigParser()
    music_cf.optionxform = str
    music_cf.read(params["template_config"])
    for option in ["ref_offset", "ref_center", "ref_extent",
                   "region_point_file", "region_point_shift",
                   "region_point_levelmin"]:
        if music_cf.has_option("setup", option):
            music_cf.remove_option("setup", option)

    music_cf.set("setup", "levelmax",
                 "%d" % (params["initial_min_level"] + params["level"]))
    music_cf.set("setup", "region", "convex_hull")
    music_cf.set("setup", "region_point_file", point_file)
    music_cf.set("setup", "region_point_shift",
                 "%d, %d, %d" % tuple(region_shift))
    music_cf.set("setup", "region_point_levelmin",
                 "%d" % params["initial_min_level"])
    if no_shift:
        override = params.get("region_shift_override")
        if override:
            music_cf.set("setup", "region_shift_override",
                         "%d, %d, %d" % tuple(override))
        else:
            music_cf.set("setup", "no_shift", "yes")
    os.makedirs(params["new_ics_directory"], exist_ok=True)
    output_path = os.path.join(params["new_ics_directory"], output_name)
    music_cf.set("output", "filename", output_path)

    os.makedirs(output_path, exist_ok=True)
    conf_file = output_path + ".conf"
    with open(conf_file, "w") as fp:
        music_cf.write(fp)
    return conf_file, output_path


def run_music_exe(params, conf_file, cwd=None):
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "%d" % params["num_cores"]
    if params.get("music_ld_library_path"):
        env["LD_LIBRARY_PATH"] = params["music_ld_library_path"]
        env["DYLD_LIBRARY_PATH"] = params["music_ld_library_path"]
    env.update(mzconfig.parse_music_env(params.get("music_env")))
    # Each run gets its own working directory: MUSIC caches wnoise_*.bin and
    # writes scratch files into its CWD, and per-halo merge-mode runs have
    # different windows, so a shared cache directory would collide.
    print("Running:", params["music_exe"], conf_file, "(cwd=%s)" % cwd)
    subprocess.run([params["music_exe"], os.path.abspath(conf_file)],
                   env=env, check=True, cwd=cwd)


def append_mrp_block(ic_dir, level):
    with open(os.path.join(ic_dir, "parameter_file.txt"), "a") as fp:
        fp.write(MRP_BLOCK % level)


def run_level_union(params):
    """One MUSIC run over the union region; per-halo mask deposit."""
    conf_file, ic_dir = write_music_conf(
        params, params["union_point_file"],
        "%s-L%d" % (params["simulation_name"], params["level"]),
        params["region_shift"])
    run_music_exe(params, conf_file, cwd=ic_dir)
    refinement_mask.particle_only_mask(
        conf_file, smooth_edges=True, backup=True,
        point_files=[r["point_file"]
                     for r in params["halo_regions"].values()])
    append_mrp_block(ic_dir, params["level"])
    return ic_dir


def run_level_merge(params):
    """One MUSIC run per halo (identical seeds, no_shift), then merge."""
    check_periodic_wrap(params)
    override = params.get("region_shift_override")
    # region_point_shift names the frame the point file was traced in: the
    # unshifted L0 for level 1, the group's own (shifted) runs afterwards.
    point_shift = [0, 0, 0] if (not override or params["level"] == 1)         else list(override)
    run_dirs, conf_paths, halo_ids = [], [], []
    for halo_id, region in params["halo_regions"].items():
        conf_file, ic_dir = write_music_conf(
            params, region["point_file"],
            "%s-L%d-h%s" % (params["simulation_name"], params["level"],
                            halo_id),
            point_shift, no_shift=True)
        run_music_exe(params, conf_file, cwd=ic_dir)
        refinement_mask.particle_only_mask(
            conf_file, smooth_edges=True, backup=True,
            point_files=[region["point_file"]])
        run_dirs.append(ic_dir)
        conf_paths.append(conf_file)
        halo_ids.append(halo_id)

    merged_dir = os.path.join(
        params["simulation_run_directory"],
        "%s-L%d.merged" % (params["simulation_name"], params["level"]))
    merge_music_ics.merge_runs(run_dirs, merged_dir,
                               conf_paths=conf_paths, halo_ids=halo_ids)
    append_mrp_block(merged_dir, params["level"])
    return merged_dir


def run_level(params):
    if params["mode"] == "union":
        ic_dir = run_level_union(params)
    else:
        ic_dir = run_level_merge(params)
    print("Moving initial conditions to %s" % params["sim_dir"])
    os.rename(ic_dir, params["sim_dir"])
    return params["sim_dir"]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        raise RuntimeError("usage: python -m multizoom.mrp_music "
                           "<config_file> <level>")
    params = startup(argv[0], int(argv[1]))
    params = get_previous_run_params(params)
    params = find_lagrangian_regions(params)
    run_level(params)


if __name__ == "__main__":
    main()
