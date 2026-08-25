"""Deposit N Lagrangian particle clouds into a MUSIC RefinementMask.

Forked from foggie/initial_conditions/enzo-mrp-music/particle_only_mask.py
with these changes:

* accepts a LIST of point files and deposits every cloud into the mask, so
  a union-mode run tags each halo's Lagrangian volume without refining the
  space between them;
* scipy >= 1.13 (signal.gaussian moved to signal.windows.gaussian);
* the scipy-missing fallback actually disables smoothing (the legacy except
  branch re-enabled it while claiming to turn it off).

The deposit itself is unchanged: particles are CIC-binned into a mask
coarsened by dx=2 (the finest initial grid of the previous run), optionally
smoothed with a 3x3x3 Gaussian, thresholded, and upsampled back over the
innermost RefinementMask dataset (-1 = not refined, 0 = refine).
"""

import configparser
import os
import shutil

import h5py
import numpy as np


def _gaussian_window(n, std):
    try:
        from scipy.signal.windows import gaussian
    except ImportError:            # scipy < 1.1
        from scipy.signal import gaussian
    return gaussian(n, std)


def read_music_run_geometry(music_config):
    """Read the mask geometry a deposit needs from a MUSIC conf + outputs.

    Returns dict with: data_dir, levelmin, levelmax, finest_level, origin,
    box_shift, pt_shift, pt_level, point_files (list; the conf's own
    region_point_file).
    """
    if not os.path.exists(music_config):
        raise RuntimeError("Config file %s not found." % music_config)
    cp = configparser.ConfigParser()
    cp.read(music_config)
    geometry = dict(
        point_files=[cp.get("setup", "region_point_file")],
        pt_shift=[int(s) for s in
                  cp.get("setup", "region_point_shift").split(",")],
        pt_level=cp.getint("setup", "region_point_levelmin"),
        data_dir=cp.get("output", "filename"),
        levelmin=cp.getint("setup", "levelmin"),
        levelmax=cp.getint("setup", "levelmax"),
    )
    geometry["finest_level"] = geometry["levelmax"] - geometry["levelmin"]

    origin_parameter = "CosmologySimulationGridLeftEdge[%d]" % \
        geometry["finest_level"]
    origin = None
    with open(os.path.join(geometry["data_dir"], "parameter_file.txt")) as fh:
        for line in fh:
            if line.startswith(origin_parameter):
                origin = [float(v) for v in line.split("=")[1].split()]
    if origin is None:
        raise RuntimeError("%s not found in %s/parameter_file.txt"
                           % (origin_parameter, geometry["data_dir"]))
    geometry["origin"] = origin

    box_shift = [0, 0, 0]
    log_file = "%s_log.txt" % music_config
    with open(log_file) as fh:
        for line in fh:
            for i, ax in enumerate("xyz"):
                if line.find("setup/shift_%s" % ax) >= 0:
                    box_shift[i] = int(line.split("=")[1])
    geometry["box_shift"] = box_shift
    return geometry


def particle_only_mask(music_config, smooth_edges=True, backup=True,
                       point_files=None):
    """Overwrite the innermost RefinementMask with a deposit of the clouds.

    point_files: optional list of per-halo Lagrangian point files.  When
    None, the conf's own region_point_file is deposited (legacy behavior).
    """
    have_scipy = True
    try:
        import scipy  # noqa: F401
    except ImportError:
        have_scipy = False
    if smooth_edges and not have_scipy:
        print("scipy not installed; smooth_edges requires scipy. Turning off.")
        smooth_edges = False

    print("Modifying RefinementMask: reading parameters...")
    geo = read_music_run_geometry(music_config)
    if point_files is None:
        point_files = geo["point_files"]

    # Convert particle positions to integer cell indices at 2^levelmax,
    # relative to the innermost grid origin, in the current (shifted) frame.
    shift = np.array(geo["box_shift"]) / 2.0**geo["levelmin"] - \
        np.array(geo["pt_shift"]) / 2.0**geo["pt_level"]
    clouds = []
    for fn in point_files:
        centered = np.loadtxt(fn, ndmin=2) + shift
        centered[centered < 0.0] += 1.0
        clouds.append(((centered - geo["origin"]) *
                       2.0**geo["levelmax"]).astype("int32"))
    pts = np.concatenate(clouds, axis=0)

    mask_name = "RefinementMask.%d" % geo["finest_level"]
    mask_fn = os.path.join(geo["data_dir"], mask_name)
    if backup and not os.path.exists(mask_fn + ".bak"):
        print("Modifying RefinementMask: backing up original file...")
        shutil.copyfile(mask_fn, mask_fn + ".bak")

    print("Modifying RefinementMask: calculating the new particle mask "
          "(%d particles from %d cloud(s))..." % (len(pts), len(point_files)))
    with h5py.File(mask_fn, "a") as h5p:
        mask_shape = h5p[mask_name].shape[:0:-1]

        # Deposit at 2*dx, the finest initial grid of the previous run.
        dx = 2
        newmask_shape = np.ceil(np.array(mask_shape) / float(dx)).astype("int32")
        newmask = -np.ones(newmask_shape, dtype="int32")
        H, _ = np.histogramdd(pts / float(dx), bins=newmask_shape,
                              range=[[0, n] for n in newmask_shape])
        if smooth_edges:
            print("Modifying RefinementMask: smoothing particle mask...")
            from scipy import signal
            window = _gaussian_window(3, 0.5)
            fil3d = np.outer(np.outer(window, window),
                             window).reshape((3, 3, 3))
            H = signal.fftconvolve(H, fil3d, mode="same")
            limit = window[0]
        else:
            limit = 0
        newmask[H > limit] = 0
        upsampled = np.kron(newmask, np.ones((dx, dx, dx)))
        h5p[mask_name][0, :, :, :] = \
            upsampled[:mask_shape[0], :mask_shape[1], :mask_shape[2]].T
    print("Modifying RefinementMask: complete.")


if __name__ == "__main__":
    import sys
    particle_only_mask(sys.argv[-1])
