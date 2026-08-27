"""Tests for the union-mode validation battery, on synthetic masks."""
import os
import sys

import h5py
import numpy as np
import pytest

from foggie.initial_conditions.multizoom import validation


def _write_ics(tmp_path, mask, left=(0.4, 0.4, 0.4), right=(0.6, 0.6, 0.6)):
    d = tmp_path / "ics"
    d.mkdir()
    name = "RefinementMask.1"
    with h5py.File(d / name, "w") as fp:
        fp.create_dataset(name, data=mask.T[np.newaxis, ...])
    with open(d / "parameter_file.txt", "w") as fp:
        fp.write("CosmologySimulationGridDimension[1]      = %d %d %d\n"
                 % mask.shape)
        fp.write("CosmologySimulationGridLeftEdge[1]       = %g %g %g\n" % left)
        fp.write("CosmologySimulationGridRightEdge[1]      = %g %g %g\n" % right)
    return str(d)


def _two_cloud_mask(n=32):
    mask = -np.ones((n, n, n), dtype="int32")
    mask[4:10, 4:10, 4:10] = 0
    mask[22:28, 22:28, 22:28] = 0
    return mask


def test_two_clouds_pass(tmp_path):
    ics = _write_ics(tmp_path, _two_cloud_mask())
    ok, lines = validation.check_union(ics, min_clouds=2)
    assert ok, "\n".join(lines)
    summary = validation.cloud_summary(validation.read_mask(ics))
    assert len(summary["clouds"]) == 2
    assert summary["refined_cells"] == 2 * 6**3


def test_spanning_hull_fails(tmp_path):
    """One connected blob (what a spanning convex hull produces) must fail."""
    n = 32
    mask = -np.ones((n, n, n), dtype="int32")
    mask[4:28, 4:28, 4:28] = 0
    ics = _write_ics(tmp_path, mask)
    ok, lines = validation.check_union(ics, min_clouds=2)
    assert not ok
    assert any("spanning hull" in l for l in lines)


def test_half_box_patch_fails(tmp_path):
    ics = _write_ics(tmp_path, _two_cloud_mask(),
                     left=(0.2, 0.4, 0.4), right=(0.8, 0.6, 0.6))
    ok, lines = validation.check_union(ics, min_clouds=2)
    assert not ok
    assert any("half the box" in l for l in lines)


def test_missing_point_file_fails(tmp_path):
    ics = _write_ics(tmp_path, _two_cloud_mask())
    ok, lines = validation.check_union(
        ics, min_clouds=2,
        expect_point_files=[str(tmp_path / "nonexistent.dat")])
    assert not ok
