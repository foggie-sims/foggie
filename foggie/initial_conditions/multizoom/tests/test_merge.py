import json
import os
import re

import h5py
import numpy as np
import pytest

from multizoom import merge_music_ics as mz
from .conftest import make_run


def test_merge_two_runs(mergeable_pair):
    tmp_path, run_a, run_b = mergeable_pair
    out = os.path.join(str(tmp_path), "merged")
    manifest = mz.merge_runs([run_a, run_b], out, halo_ids=["A", "B"])

    # Level-major numbering: 0 base, 1-2 the two L1 patches, 3-4 the L2s.
    assert set(manifest["grids"]) == {"0", "1", "2", "3", "4"}
    assert manifest["grids"]["1"] == dict(
        manifest["grids"]["1"], run="sim-L2-hA", level=1, halo="A")
    assert manifest["grids"]["2"]["run"] == "sim-L2-hB"
    assert manifest["grids"]["3"]["run"] == "sim-L2-hA"
    assert manifest["grids"]["4"]["level"] == 2

    # Files exist and their datasets carry the renumbered names.
    for g in range(5):
        fn = os.path.join(out, "ParticleDisplacements_x.%d" % g)
        with h5py.File(fn, "r") as fp:
            assert "ParticleDisplacements_x.%d" % g in fp
            assert len(fp.keys()) == 1

    # Renumbered patch data matches its source (dataset renamed, not lost).
    with h5py.File(os.path.join(out, "ParticleDisplacements_x.4"), "r") as fp:
        merged = fp["ParticleDisplacements_x.4"][()]
    with h5py.File(os.path.join(run_b, "ParticleDisplacements_x.2"), "r") as fp:
        source = fp["ParticleDisplacements_x.2"][()]
    np.testing.assert_array_equal(merged, source)

    # Parameter file: 5 grids, level-major levels, no RefineRegion lines.
    pf = open(os.path.join(out, "parameter_file.txt")).read()
    assert "CosmologySimulationNumberOfInitialGrids  = 5" in pf
    assert "RefineRegion" not in pf
    levels = [int(m) for m in re.findall(
        r"CosmologySimulationGridLevel\[\d+\]\s*=\s*(\d+)", pf)]
    assert levels == [1, 1, 2, 2]
    # Grid 2's geometry is run B's level-1 patch.
    m = re.search(r"CosmologySimulationGridLeftEdge\[2\]\s*=\s*(\S+)", pf)
    assert float(m.group(1)) == pytest.approx(0.625)

    # Base-grid displacement report exists (identical here, so zero diff).
    report = manifest["base_grid_report"]["ParticleDisplacements_x"]
    assert not report["identical"]
    assert report["diff_vs_donor"]["sim-L2-hB"]["max_abs"] == 0.0

    with open(os.path.join(out, "merge_manifest.json")) as fp:
        assert json.load(fp)["base_donor"] == "sim-L2-hA"


def test_overlap_aborts(tmp_path):
    base = np.random.default_rng(7).normal(size=(8, 8, 8))
    run_a = make_run(tmp_path, "sim-L1-hA", base_field=base,
                     patches={1: ((0.10, 0.10, 0.10), (0.40, 0.40, 0.40))})
    run_b = make_run(tmp_path, "sim-L1-hB", base_field=base,
                     patches={1: ((0.35, 0.35, 0.35), (0.60, 0.60, 0.60))})
    with pytest.raises(mz.MergeError, match="overlap"):
        mz.merge_runs([run_a, run_b], os.path.join(str(tmp_path), "m"))


def test_close_patches_abort(tmp_path):
    base = np.random.default_rng(7).normal(size=(8, 8, 8))
    # Disjoint but separated by ~1 parent cell (1/32) < 4 parent cells.
    run_a = make_run(tmp_path, "sim-L1-hA", base_field=base,
                     patches={1: ((0.125, 0.125, 0.125), (0.375, 0.375, 0.375))})
    run_b = make_run(tmp_path, "sim-L1-hB", base_field=base,
                     patches={1: ((0.40625, 0.125, 0.125), (0.65625, 0.375, 0.375))})
    with pytest.raises(mz.MergeError, match="separated"):
        mz.merge_runs([run_a, run_b], os.path.join(str(tmp_path), "m"))
    # ... but allowed when the caller relaxes the gap.
    mz.merge_runs([run_a, run_b], os.path.join(str(tmp_path), "m2"),
                  min_gap_fine_cells=0)


def test_seed_mismatch_aborts(tmp_path):
    base = np.random.default_rng(7).normal(size=(8, 8, 8))
    run_a = make_run(tmp_path, "sim-L1-hA", base_field=base,
                     patches={1: ((0.125, 0.125, 0.125), (0.375, 0.375, 0.375))})
    run_b = make_run(tmp_path, "sim-L1-hB", base_field=base,
                     seeds={5: 100 + 5, 6: 999},
                     patches={1: ((0.625, 0.625, 0.625), (0.875, 0.875, 0.875))})
    with pytest.raises(mz.MergeError, match="seed"):
        mz.merge_runs([run_a, run_b], os.path.join(str(tmp_path), "m"))


def test_shift_mismatch_aborts(tmp_path):
    base = np.random.default_rng(7).normal(size=(8, 8, 8))
    run_a = make_run(tmp_path, "sim-L1-hA", base_field=base,
                     patches={1: ((0.125, 0.125, 0.125), (0.375, 0.375, 0.375))})
    run_b = make_run(tmp_path, "sim-L1-hB", base_field=base, shift=(-3, 2, 0),
                     patches={1: ((0.625, 0.625, 0.625), (0.875, 0.875, 0.875))})
    with pytest.raises(mz.MergeError, match="shift"):
        mz.merge_runs([run_a, run_b], os.path.join(str(tmp_path), "m"))


def test_grid_density_mismatch_aborts(tmp_path):
    rng = np.random.default_rng(7)
    base = rng.normal(size=(8, 8, 8))
    density = rng.normal(size=(8, 8, 8))
    run_a = make_run(tmp_path, "sim-L1-hA", base_field=base,
                     grid_density=density,
                     patches={1: ((0.125, 0.125, 0.125), (0.375, 0.375, 0.375))})
    run_b = make_run(tmp_path, "sim-L1-hB", base_field=base,
                     grid_density=density + 1e-12,
                     patches={1: ((0.625, 0.625, 0.625), (0.875, 0.875, 0.875))})
    with pytest.raises(mz.MergeError, match="bit-identical"):
        mz.merge_runs([run_a, run_b], os.path.join(str(tmp_path), "m"))
    # Identical densities pass.
    run_c = make_run(tmp_path, "sim-L1-hC", base_field=base,
                     grid_density=density,
                     patches={1: ((0.625, 0.625, 0.625), (0.875, 0.875, 0.875))})
    mz.merge_runs([run_a, run_c], os.path.join(str(tmp_path), "m2"))


def test_nesting_violation_aborts(tmp_path):
    base = np.random.default_rng(7).normal(size=(8, 8, 8))
    run_a = make_run(tmp_path, "sim-L2-hA", base_field=base,
                     patches={1: ((0.125, 0.125, 0.125), (0.375, 0.375, 0.375)),
                              2: ((0.30, 0.1875, 0.1875),
                                  (0.425, 0.3125, 0.3125))})
    run_b = make_run(tmp_path, "sim-L2-hB", base_field=base,
                     patches={1: ((0.625, 0.625, 0.625), (0.875, 0.875, 0.875)),
                              2: ((0.6875, 0.6875, 0.6875),
                                  (0.8125, 0.8125, 0.8125))})
    with pytest.raises(mz.MergeError, match="not contained"):
        mz.merge_runs([run_a, run_b], os.path.join(str(tmp_path), "m"))


def test_output_dir_must_be_empty(mergeable_pair):
    tmp_path, run_a, run_b = mergeable_pair
    out = os.path.join(str(tmp_path), "occupied")
    os.makedirs(out)
    open(os.path.join(out, "something"), "w").close()
    with pytest.raises(mz.MergeError, match="not empty"):
        mz.merge_runs([run_a, run_b], out)
