import os

import h5py
import numpy as np
import pytest

from multizoom import lagrangian_regions
from multizoom import refinement_mask

MASK_CONF = """[setup]
levelmin = 5
levelmax = 6
region_point_file = {point_file}
region_point_shift = 0, 0, 0
region_point_levelmin = 5

[output]
filename = {run_dir}
"""


def _make_mask_run(tmp_path, cloud_centers, n_cloud=200, sigma=0.008):
    """A minimal MUSIC-like run for the mask deposit: one L1 patch spanning
    [0.25, 0.5)^3 (16^3 cells at levelmax=6)."""
    run_dir = tmp_path / "maskrun"
    run_dir.mkdir()
    rng = np.random.default_rng(11)
    clouds = []
    for center in cloud_centers:
        pts = rng.normal(loc=center, scale=sigma, size=(n_cloud, 3))
        clouds.append(np.clip(pts, 0.26, 0.49))
    point_files = []
    for i, cloud in enumerate(clouds):
        fn = tmp_path / ("points_%d.dat" % i)
        np.savetxt(fn, cloud)
        point_files.append(str(fn))

    conf_fn = tmp_path / "maskrun.conf"
    conf_fn.write_text(MASK_CONF.format(point_file=point_files[0],
                                        run_dir=str(run_dir)))
    (tmp_path / "maskrun.conf_log.txt").write_text(
        " - setup/shift_x = 0\n - setup/shift_y = 0\n - setup/shift_z = 0\n")
    with open(run_dir / "parameter_file.txt", "w") as fp:
        fp.write("CosmologySimulationGridLeftEdge[1]       = 0.25 0.25 0.25\n"
                 "CosmologySimulationGridRightEdge[1]      = 0.5 0.5 0.5\n")
    name = "RefinementMask.1"
    with h5py.File(run_dir / name, "w") as fp:
        fp.create_dataset(name, data=np.zeros((1, 16, 16, 16), dtype="int32"))
    return str(conf_fn), str(run_dir), point_files


def _read_mask(run_dir):
    with h5py.File(os.path.join(run_dir, "RefinementMask.1"), "r") as fp:
        return fp["RefinementMask.1"][0].T  # back to (x, y, z) ordering


def test_two_cloud_deposit_leaves_gap(tmp_path):
    conf_fn, run_dir, point_files = _make_mask_run(
        tmp_path, [(0.30, 0.30, 0.30), (0.45, 0.45, 0.45)])
    refinement_mask.particle_only_mask(conf_fn, smooth_edges=False,
                                       backup=True,
                                       point_files=point_files)
    mask = _read_mask(run_dir)
    # Cells at the two cloud centers are refined...
    assert mask[3, 3, 3] == 0        # (0.30 - 0.25) * 64 = 3.2
    assert mask[12, 12, 12] == 0     # (0.45 - 0.25) * 64 = 12.8
    # ...but the volume between the clouds is not.
    assert mask[8, 8, 8] == -1       # 0.375: between the clouds
    assert (mask >= -1).all() and (mask <= 0).all()
    assert os.path.exists(os.path.join(run_dir, "RefinementMask.1.bak"))


def test_single_conf_cloud_default(tmp_path):
    conf_fn, run_dir, point_files = _make_mask_run(
        tmp_path, [(0.30, 0.30, 0.30)])
    # No explicit point_files: the conf's own region_point_file is used.
    refinement_mask.particle_only_mask(conf_fn, smooth_edges=True,
                                       backup=False)
    mask = _read_mask(run_dir)
    assert mask[3, 3, 3] == 0
    assert mask[13, 13, 13] == -1


def test_union_point_file(tmp_path):
    a = np.arange(12, dtype=float).reshape(4, 3) / 100.0
    b = np.arange(6, dtype=float).reshape(2, 3) / 10.0
    fa, fb = tmp_path / "a.dat", tmp_path / "b.dat"
    np.savetxt(fa, a)
    np.savetxt(fb, b)
    out = tmp_path / "union.dat"
    lagrangian_regions.write_union_point_file([str(fa), str(fb)], str(out))
    union = np.loadtxt(out)
    assert union.shape == (6, 3)
    np.testing.assert_allclose(union[:4], a)
    np.testing.assert_allclose(union[4:], b)


def test_point_file_names():
    assert lagrangian_regions.point_file_name("5016", "RD0000") == \
        "initial_particle_positions-5016-RD0000.dat"
    assert lagrangian_regions.union_point_file_name("5016+5033", "RD0000") == \
        "initial_particle_positions-union-5016+5033-RD0000.dat"


def test_center_and_wrap():
    positions = np.array([[0.02, 0.97, 0.99], [0.5, 0.51, 0.52],
                          [0.5, 0.5, 0.5]])
    com, shifted = lagrangian_regions._center_and_wrap(
        np.array([0.0, 0.5, 0.5]), positions)
    assert shifted[0] and not shifted[1] and not shifted[2]
    # x positions re-wrapped around the new frame: spread now < 0.5
    assert positions[0].max() - positions[0].min() < 0.5
    assert com[0] == pytest.approx(0.5)
