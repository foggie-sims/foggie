"""Fixtures: synthetic MUSIC output runs for the merge/mask tests.

The fixtures mimic what MUSIC's enzo output plugin writes: a run directory
of <Field>.<level> HDF5 files whose dataset carries the same name as the
file (4-D, shape (1, nz, ny, nx)), a parameter_file.txt with one nested
grid per level, and sibling <dir>.conf / <dir>.conf_log.txt files.
"""

import os
import sys

import h5py
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

LEVELMIN = 5  # 32^3 base
BASE_N = 2**LEVELMIN

HEADER_TEMPLATE = """# Relevant Section of Enzo Paramter File (NOT COMPLETE!)
ProblemType                              = 30      // cosmology simulation
TopGridRank                              = 3
TopGridDimensions                        = {n} {n} {n}
SelfGravity                              = 1       // gravity on
CosmologySimulationOmegaBaryonNow        = 0
CosmologySimulationOmegaCDMNow           = 0.291
CosmologyOmegaMatterNow                  = 0.291
CosmologyHubbleConstantNow               = 0.7     // in 100 km/s/Mpc
CosmologyComovingBoxSize                 = 25    // in Mpc/h
CosmologyInitialRedshift                 = 99      //
ParallelRootGridIO                       = 1
ParallelParticleIO                       = 1
PartitionNestedGrids                     = 1
"""

CONF_TEMPLATE = """[setup]
boxlength = 25
zstart = 99
levelmin = {levelmin}
levelmin_TF = {levelmin}
levelmax = {levelmax}
padding = 4
baryons = no
region = convex_hull
region_point_file = {point_file}
region_point_shift = 0, 0, 0
region_point_levelmin = {levelmin}

[cosmology]
Omega_m = 0.291
Omega_L = 0.709
Omega_b = 0.04576
H0 = 70.0
sigma_8 = 0.8159
nspec = 0.9667

[random]
{seed_lines}

[output]
format = enzo
filename = {run_dir}
"""


def _write_field(run_dir, field, level, data):
    """Write one Field.<level> file the way MUSIC's enzo plugin does."""
    name = "%s.%d" % (field, level)
    with h5py.File(os.path.join(run_dir, name), "w") as fp:
        fp.create_dataset(name, data=data[np.newaxis, ...])


def make_run(base_dir, name, patches, seeds=None, shift=(0, 0, 0),
             base_field=None, grid_density=None, levelmin=LEVELMIN):
    """Create one synthetic MUSIC run.

    patches: {level(int>=1): (left_edge, right_edge)} in code units; grid
    dims are derived from the edges at each level's cell size.
    base_field: base-grid ParticleDisplacements_x array (shared across runs
    for a mergeable set); defaults to a seeded random field.
    grid_density: optional GridDensity.0 array (to exercise the strict
    bit-identity check).
    """
    run_dir = os.path.join(str(base_dir), name)
    os.makedirs(run_dir)
    levelmax = levelmin + max(patches) if patches else levelmin
    if seeds is None:
        seeds = {level: 100 + level for level in
                 range(levelmin, levelmax + 1)}

    rng = np.random.default_rng(12345)
    if base_field is None:
        base_field = rng.normal(size=(8, 8, 8))
    fields = {"ParticleDisplacements_x": base_field,
              "RefinementMask": np.zeros((8, 8, 8), dtype="int32")}
    if grid_density is not None:
        fields["GridDensity"] = grid_density
    for field, data in fields.items():
        _write_field(run_dir, field, 0, data)

    grid_lines = []
    for level in sorted(patches):
        left, right = patches[level]
        h = 1.0 / 2.0**(levelmin + level)
        dims = [int(round((right[d] - left[d]) / h)) for d in range(3)]
        for field in fields:
            patch_rng = np.random.default_rng(hash((name, field, level)) %
                                              2**32)
            data = patch_rng.normal(size=(4, 4, 4))
            if field == "RefinementMask":
                data = np.zeros((4, 4, 4), dtype="int32")
            _write_field(run_dir, field, level, data)
        grid_lines += [
            "CosmologySimulationGridDimension[%d]      = %d %d %d"
            % tuple([level] + dims),
            "CosmologySimulationGridLeftEdge[%d]       = %.10g %.10g %.10g"
            % tuple([level] + list(left)),
            "CosmologySimulationGridRightEdge[%d]      = %.10g %.10g %.10g"
            % tuple([level] + list(right)),
            "CosmologySimulationGridLevel[%d]          = %d" % (level, level),
        ]

    with open(os.path.join(run_dir, "parameter_file.txt"), "w") as fp:
        fp.write(HEADER_TEMPLATE.format(n=2**levelmin))
        fp.write("CosmologySimulationNumberOfInitialGrids  = %d\n"
                 % (1 + len(patches)))
        fp.write("\n".join(grid_lines) + "\n")
        fp.write("RefineRegionLeftEdge                     = 0.4 0.4 0.4\n")
        fp.write("RefineRegionRightEdge                    = 0.6 0.6 0.6\n")

    seed_lines = "\n".join("seed[%d] = %d" % (level, seed)
                           for level, seed in sorted(seeds.items()))
    conf_fn = run_dir + ".conf"
    with open(conf_fn, "w") as fp:
        fp.write(CONF_TEMPLATE.format(
            levelmin=levelmin, levelmax=levelmax, run_dir=run_dir,
            point_file=os.path.join(str(base_dir), "points.dat"),
            seed_lines=seed_lines))
    with open(conf_fn + "_log.txt", "w") as fp:
        for i, ax in enumerate("xyz"):
            fp.write(" - setup/shift_%s = %d\n" % (ax, shift[i]))
        fp.write("  Domain shifted by (%5d, %5d, %5d)\n" % tuple(shift))
    return run_dir


@pytest.fixture
def mergeable_pair(tmp_path):
    """Two runs sharing base field and seeds, with disjoint patches."""
    rng = np.random.default_rng(7)
    base = rng.normal(size=(8, 8, 8))
    run_a = make_run(tmp_path, "sim-L2-hA", seeds=None, base_field=base,
                     patches={1: ((0.125, 0.125, 0.125), (0.375, 0.375, 0.375)),
                              2: ((0.1875, 0.1875, 0.1875), (0.3125, 0.3125, 0.3125))})
    run_b = make_run(tmp_path, "sim-L2-hB", seeds=None, base_field=base,
                     patches={1: ((0.625, 0.625, 0.625), (0.875, 0.875, 0.875)),
                              2: ((0.6875, 0.6875, 0.6875), (0.8125, 0.8125, 0.8125))})
    return tmp_path, run_a, run_b
