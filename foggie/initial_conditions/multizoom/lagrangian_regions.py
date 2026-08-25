"""Trace N halos back to their z_init Lagrangian volumes in one pass.

Forked from foggie/initial_conditions/enzo-mrp-music/get_halo_initial_extent.py
with these changes:

* multi-halo: get_centers_and_extents() selects every target's particles from
  one load of the final dataset and matches all of them during a SINGLE pass
  over the initial dataset's grids (the legacy code re-read everything per
  halo, untenable for 10+ targets);
* per-halo point-file names carry the real halo id (the legacy default id=0
  made every run write the same file);
* h5py >= 3 (.value removed) and the broken method='halo' path dropped;
* yt / mpi4py are imported lazily so the module (and its tests) work without
  a full simulation stack.

The single-halo get_center_and_extent() is kept as a thin wrapper with the
legacy return signature.
"""

import os

import numpy as np

AXES = ("x", "y", "z")
RHO_CRIT_NOW = 1.8788e-29  # g/cm^3 (H0=100 km/s/Mpc), as in the legacy script


def _get_parallel_state():
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        return comm, comm.rank, comm.size, MPI
    except ImportError:
        return None, 0, 1, None


def point_file_name(halo_id, initial_ds_name, output_format="txt"):
    ext = {"txt": ".dat", "hdf5": ".h5"}[output_format]
    return "initial_particle_positions-%s-%s%s" % (halo_id, initial_ds_name, ext)


def union_point_file_name(tag, initial_ds_name):
    return "initial_particle_positions-union-%s-%s.dat" % (tag, initial_ds_name)


def write_union_point_file(point_files, output_fn):
    """Concatenate per-halo point files (bare xyz rows) into one union file."""
    positions = [np.loadtxt(fn, ndmin=2) for fn in point_files]
    union = np.concatenate(positions, axis=0)
    np.savetxt(output_fn, union)
    return output_fn


def get_halo_sphere_particles(halo_info, ds, radius_factor):
    """Select a halo's particles from an already-loaded final dataset.

    Returns (center, particle_index array, positions[3, N] in code units).
    """
    import yt

    center = ds.arr(halo_info["center"][0], halo_info["center"][1])
    if "radius" in halo_info:
        r_200 = ds.quan(halo_info["radius"][0], halo_info["radius"][1])
    else:
        rho_crit = ds.quan(RHO_CRIT_NOW, "g/cm**3") * ds.hubble_constant**2 * \
            (1 + ds.current_redshift)**3
        hmass = ds.quan(halo_info["mass"][0], halo_info["mass"][1])
        r_200 = (((3. * hmass) / (4. * np.pi * rho_crit * 200.))**(1. / 3.)).to("Mpc")

    if yt.is_root():
        print("Halo %s: center %s, r_200 %s, selection radius %s r_200"
              % (halo_info["id"], center, r_200, radius_factor))

    sphere = ds.sphere(center, radius_factor * r_200)
    positions = np.array([sphere["particle_position_%s" % ax] for ax in AXES])
    return center, np.asarray(sphere["particle_index"]), positions


def _center_and_wrap(halo_com, positions):
    """Handle a halo straddling the periodic boundary (per-axis half shift)."""
    shifted = np.zeros(3, dtype=bool)
    com = np.array(halo_com, dtype=float)
    for i in range(3):
        if positions[i].max() - positions[i].min() > 0.5:
            positions[i] = positions[i] - 0.5
            positions[i][positions[i] < 0.0] += 1.0
            com[i] -= 0.5
            if com[i] < 0.0:
                com[i] += 1.0
            shifted[i] = True
    return com, shifted


def get_centers_and_extents(halos, initial_dataset, final_dataset,
                            round_size=None, radius_factor=5.0,
                            output_format="txt", output_dir="."):
    """Trace every halo's Lagrangian region with one pass over the grids.

    Parameters
    ----------
    halos : mapping {halo_id: halo_info} (see config.parse_multizoom_config;
        each info may carry its own radius_factor, else `radius_factor`).
    initial_dataset, final_dataset : Enzo parameter-file paths.
    round_size : round region sizes up to the nearest 1/round_size.
    output_format : "txt", "hdf5", or None (no point files).

    Returns
    -------
    OrderedDict {halo_id: dict(center, size, point_file, n_particles)}
    with center/size in code units and point_file None if output_format is.
    """
    import yt
    if output_format not in ("hdf5", "txt", None):
        raise RuntimeError("output_format = %s not known (hdf5, txt, None)"
                           % output_format)

    comm, my_rank, my_size, MPI = _get_parallel_state()

    final_ds = yt.load(final_dataset)
    targets = {}
    for halo_id, info in halos.items():
        rf = info.get("radius_factor", radius_factor)
        com, indices, positions = get_halo_sphere_particles(info, final_ds, rf)
        com, shifted = _center_and_wrap(np.array(com.to("unitary")), positions)
        targets[halo_id] = dict(com=com, shifted=shifted,
                                indices=np.asarray(indices, dtype=np.int64),
                                remaining=np.asarray(indices, dtype=np.int64),
                                axis_min=np.full(3, 2.0),
                                axis_max=np.full(3, -2.0),
                                save_pos=[[], [], []])

    initial_ds = yt.load(initial_dataset)
    n_init = initial_ds.parameters["NumberOfParticles"]
    for halo_id, t in targets.items():
        n_stars = (t["indices"] >= n_init).sum()
        if n_stars and yt.is_root():
            print("Halo %s: removing %d star particles." % (halo_id, n_stars))
        t["indices"] = t["indices"][t["indices"] < n_init]
        t["remaining"] = t["indices"].copy()

    if yt.is_root():
        print("Reading initial particle positions for %d halos "
              "(single grid pass)." % len(targets))

    for grid in initial_ds.index.grids[my_rank::my_size]:
        if all(t["remaining"].size == 0 for t in targets.values()):
            break
        grid_indices = np.asarray(grid["particle_index"], dtype=np.int64)
        grid_pos = None
        for halo_id, t in targets.items():
            if t["remaining"].size == 0:
                continue
            match = np.in1d(grid_indices, t["remaining"])
            if not match.any():
                continue
            if grid_pos is None:
                grid_pos = [np.asarray(grid["particle_position_%s" % ax])
                            for ax in AXES]
            for i in range(3):
                pos = grid_pos[i][match].copy()
                t["save_pos"][i] += pos.tolist()
                if t["shifted"][i]:
                    pos -= 0.5
                    pos[pos < 0.0] += 1.0
                pos -= t["com"][i]
                pos[pos > 0.5] -= 1.0
                pos[pos < -0.5] += 1.0
                t["axis_min"][i] = min(t["axis_min"][i], pos.min())
                t["axis_max"][i] = max(t["axis_max"][i], pos.max())
            t["remaining"] = np.setdiff1d(t["remaining"],
                                          grid_indices[match])

    results = type(halos)()
    for halo_id, t in targets.items():
        axis_min, axis_max = t["axis_min"], t["axis_max"]
        save_pos = np.array([np.array(p) for p in t["save_pos"]])
        if comm is not None:
            axis_min = np.array(
                [comm.allreduce(v, op=MPI.MIN) for v in axis_min])
            axis_max = np.array(
                [comm.allreduce(v, op=MPI.MAX) for v in axis_max])
            gathered = comm.gather(save_pos)
            if my_rank == 0:
                save_pos = np.concatenate(
                    [a for a in gathered if a.size], axis=1)

        center = size = point_fn = None
        n_particles = t["indices"].size
        if my_rank == 0:
            center = 0.5 * (axis_min + axis_max) + t["com"]
            size = axis_max - axis_min
            if round_size is not None:
                size = np.ceil(round_size * size) / round_size
            print("Halo %s: %d particles, center %s, region size %s"
                  % (halo_id, n_particles, center, size))
            if output_format is not None:
                point_fn = os.path.join(
                    output_dir,
                    point_file_name(halo_id, str(initial_ds), output_format))
                if output_format == "hdf5":
                    import h5py
                    with h5py.File(point_fn, "w") as fp:
                        fp["pos"] = save_pos.T
                else:
                    np.savetxt(point_fn, save_pos.T)
        if comm is not None:
            center = comm.bcast(center)
            size = comm.bcast(size)
            point_fn = comm.bcast(point_fn)
        results[halo_id] = dict(center=center, size=size,
                                point_file=point_fn,
                                n_particles=n_particles)
    return results


def get_center_and_extent(halo_info, initial_dataset, final_dataset,
                          round_size=None, radius_factor=5.0,
                          output_format=None):
    """Single-halo wrapper with the legacy return signature."""
    halo_id = str(halo_info.get("id", 0))
    halo_info = dict(halo_info, id=halo_id)
    results = get_centers_and_extents(
        {halo_id: halo_info}, initial_dataset, final_dataset,
        round_size=round_size, radius_factor=radius_factor,
        output_format=output_format)
    r = results[halo_id]
    return r["center"], r["size"], r["point_file"]
