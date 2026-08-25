"""Merge N same-seed MUSIC runs into one multi-patch Enzo IC set.

Each input run is a normal single-target MUSIC output directory (one nested
patch per level) generated with IDENTICAL random seeds and an identical
domain frame (no_shift = yes, or one common region_shift_override).  Under
those invariants the runs are samples of the same realization: the base
grid is shared, and each run's nested patches are mutually consistent
refinements of it.  This tool verifies the invariants, then assembles one
Enzo IC directory in which every level can hold several patches:

* grid numbering is flat and level-major (grid 0 = base, then all level-1
  patches, then all level-2 patches, ...), so parents always precede
  children and Enzo's static-region ordering invariant holds;
* HDF5 files AND their datasets are renamed (Enzo opens the dataset named
  after the full filename including the grid-number suffix);
* the merged parameter_file.txt carries every patch's
  CosmologySimulationGrid{Dimension,LeftEdge,RightEdge,Level}[g] entry and
  no RefineRegion lines;
* a merge_manifest.json records the provenance of every grid.

Reading the merged ICs requires the five Enzo patches shipped in
multizoom/enzo_patches/ (multiple same-level nested grids).

The physics approximation: base-grid ParticleDisplacements/Velocities
differ slightly between runs (each run's multigrid solve feeds its own
patch back into the coarse solution).  GridDensity.0, when present, must
be bit-identical (kspace_TF transfer, shared noise); the displacement
fields are taken from --base-donor and the cross-run differences are
measured and reported so the approximation is quantified, not assumed.
"""

import argparse
import configparser
import glob
import hashlib
import json
import os
import re
import shutil

import h5py
import numpy as np

GRID_KEYS = ("Dimension", "LeftEdge", "RightEdge", "Level")
GRID_LINE_RE = re.compile(
    r"^CosmologySimulation(Grid(?:Dimension|LeftEdge|RightEdge|Level))"
    r"\[(\d+)\]\s*=\s*(.*)$")
NGRIDS_RE = re.compile(
    r"^CosmologySimulationNumberOfInitialGrids\s*=\s*(\d+)")
EXCLUDED_HEADER_PREFIXES = ("RefineRegion",)
FIELD_FILE_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_]*)\.0$")
COSMOLOGY_KEYS = ("Omega_m", "Omega_L", "Omega_b", "H0", "sigma_8", "nspec")


class MergeError(RuntimeError):
    pass


class RunInfo(object):
    """Parsed description of one MUSIC output run."""

    def __init__(self, run_dir, conf_path=None):
        self.run_dir = os.path.abspath(run_dir)
        self.name = os.path.basename(self.run_dir.rstrip("/"))
        if conf_path is None:
            conf_path = self.run_dir.rstrip("/") + ".conf"
        if not os.path.exists(conf_path):
            raise MergeError("MUSIC conf not found for run %s (expected %s)"
                             % (self.name, conf_path))
        self.conf_path = conf_path
        self.log_path = conf_path + "_log.txt"

        cf = configparser.ConfigParser()
        cf.read(conf_path)
        self.levelmin = cf.getint("setup", "levelmin")
        self.levelmax = cf.getint("setup", "levelmax")
        self.zstart = cf.getfloat("setup", "zstart")
        self.boxlength = cf.getfloat("setup", "boxlength")
        self.baryons = cf.get("setup", "baryons", fallback="no").lower() in \
            ("yes", "true", "1")
        self.cosmology = {k: cf.getfloat("cosmology", k)
                          for k in COSMOLOGY_KEYS
                          if cf.has_option("cosmology", k)}
        self.seeds = {}
        if cf.has_section("random"):
            for key, value in cf.items("random"):
                m = re.match(r"seed\[(\d+)\]$", key)
                if m:
                    self.seeds[int(m.group(1))] = value.strip()
                else:
                    raise MergeError(
                        "Run %s: unsupported [random] entry %r (file-based "
                        "noise is not supported by the merge tool)"
                        % (self.name, key))

        self.shift = self._read_shift()
        self.header_lines, self.grids = self._read_parameter_file()
        self.n_levels = self.levelmax - self.levelmin

    def _read_shift(self):
        shift = [0, 0, 0]
        if os.path.exists(self.log_path):
            with open(self.log_path) as fh:
                for line in fh:
                    for i, ax in enumerate("xyz"):
                        if line.find("setup/shift_%s" % ax) >= 0:
                            shift[i] = int(line.split("=")[1])
        return tuple(shift)

    def _read_parameter_file(self):
        pf_path = os.path.join(self.run_dir, "parameter_file.txt")
        if not os.path.exists(pf_path):
            raise MergeError("Run %s: %s not found" % (self.name, pf_path))
        header_lines = []
        grids = {}
        with open(pf_path) as fh:
            for line in fh:
                stripped = line.rstrip("\n")
                m = GRID_LINE_RE.match(stripped)
                if m:
                    key, index, values = m.group(1), int(m.group(2)), m.group(3)
                    entry = grids.setdefault(index, {})
                    if key == "GridLevel":
                        entry["Level"] = int(values.split()[0])
                    elif key == "GridDimension":
                        entry["Dimension"] = [int(v) for v in values.split()]
                    else:
                        entry[key.replace("Grid", "")] = \
                            [float(v) for v in values.split()]
                    continue
                if NGRIDS_RE.match(stripped):
                    continue
                if stripped.strip().startswith(EXCLUDED_HEADER_PREFIXES):
                    continue
                header_lines.append(stripped)
        for index, entry in grids.items():
            missing = [k for k in ("Dimension", "LeftEdge", "RightEdge", "Level")
                       if k not in entry]
            if missing:
                raise MergeError("Run %s: grid %d is missing %s"
                                 % (self.name, index, missing))
            if entry["Level"] != index:
                raise MergeError(
                    "Run %s: grid %d has level %d (single-target MUSIC "
                    "output should have grid index == level)"
                    % (self.name, index, entry["Level"]))
        return header_lines, grids

    def grid_at_level(self, level):
        return self.grids[level]

    @property
    def fields(self):
        """Field base names present in the run dir (from the .0 files)."""
        names = []
        for fn in sorted(glob.glob(os.path.join(self.run_dir, "*.0"))):
            m = FIELD_FILE_RE.match(os.path.basename(fn))
            if m:
                names.append(m.group(1))
        return names

    def field_file(self, field, level):
        return os.path.join(self.run_dir, "%s.%d" % (field, level))


def verify_runs(runs):
    """Check the cross-run invariants that make a merge physical."""
    if len(runs) < 2:
        raise MergeError("Need at least two runs to merge.")
    donor = runs[0]
    for run in runs[1:]:
        for attr in ("levelmin", "zstart", "boxlength", "baryons"):
            if getattr(run, attr) != getattr(donor, attr):
                raise MergeError(
                    "Run %s: %s = %s differs from %s (%s)"
                    % (run.name, attr, getattr(run, attr),
                       donor.name, getattr(donor, attr)))
        if run.cosmology != donor.cosmology:
            raise MergeError("Run %s: cosmology differs from %s"
                             % (run.name, donor.name))
        if run.shift != donor.shift:
            raise MergeError(
                "Run %s: domain shift %s differs from %s %s — all runs "
                "must share one frame (no_shift = yes, or one common "
                "region_shift_override)"
                % (run.name, run.shift, donor.name, donor.shift))
        shared = set(run.seeds) & set(donor.seeds)
        for level in sorted(shared):
            if run.seeds[level] != donor.seeds[level]:
                raise MergeError(
                    "Run %s: seed[%d] = %s differs from %s (%s) — merged "
                    "runs must share the identical noise realization"
                    % (run.name, level, run.seeds[level],
                       donor.name, donor.seeds[level]))
        needed = set(range(run.levelmin, run.levelmax + 1))
        if not needed <= set(run.seeds):
            raise MergeError(
                "Run %s: missing seed[] entries for levels %s"
                % (run.name, sorted(needed - set(run.seeds))))

    fields = donor.fields
    if not fields:
        raise MergeError("Run %s: no <Field>.0 files found" % donor.name)
    for run in runs:
        if run.fields != fields:
            raise MergeError("Run %s: fields %s differ from %s (%s)"
                             % (run.name, run.fields, donor.name, fields))
        for level in range(1, run.n_levels + 1):
            if level not in run.grids:
                raise MergeError("Run %s: no grid entry for level %d"
                                 % (run.name, level))
            for field in fields:
                if not os.path.exists(run.field_file(field, level)):
                    raise MergeError("Run %s: missing %s.%d"
                                     % (run.name, field, level))
    return fields


def detect_overlaps(runs, min_gap_fine_cells=4):
    """Abort on overlapping or too-close patches; verify nesting."""
    max_level = max(run.n_levels for run in runs)
    for level in range(1, max_level + 1):
        dx_parent = 1.0 / 2.0**(runs[0].levelmin + level - 1)
        present = [run for run in runs if level <= run.n_levels]
        for a in range(len(present)):
            grid_a = present[a].grid_at_level(level)
            for b in range(a + 1, len(present)):
                grid_b = present[b].grid_at_level(level)
                gaps = [max(grid_a["LeftEdge"][d] - grid_b["RightEdge"][d],
                            grid_b["LeftEdge"][d] - grid_a["RightEdge"][d])
                        for d in range(3)]
                separation = max(gaps)
                if separation <= 0.0:
                    raise MergeError(
                        "Level-%d patches of %s and %s overlap. Their "
                        "Lagrangian volumes are not separable at this "
                        "level: concatenate those halos' point files into "
                        "one union sub-run (or reduce radius_factor) and "
                        "merge that sub-run instead."
                        % (level, present[a].name, present[b].name))
                if separation < min_gap_fine_cells * dx_parent:
                    raise MergeError(
                        "Level-%d patches of %s and %s are separated by "
                        "%.3g, less than %d parent cells (%.3g). Widen the "
                        "gap (union sub-run) or lower --min-gap-cells if "
                        "you accept adjacent patches."
                        % (level, present[a].name, present[b].name,
                           separation, min_gap_fine_cells,
                           min_gap_fine_cells * dx_parent))
        for run in present:
            if level < 2:
                continue
            child = run.grid_at_level(level)
            parent = run.grid_at_level(level - 1)
            for d in range(3):
                if child["LeftEdge"][d] < parent["LeftEdge"][d] or \
                   child["RightEdge"][d] > parent["RightEdge"][d]:
                    raise MergeError(
                        "Run %s: level-%d patch is not contained in its "
                        "level-%d patch" % (run.name, level, level - 1))


def _dataset_hash(path, name, chunk_slices=64):
    digest = hashlib.sha256()
    with h5py.File(path, "r") as fp:
        dset = fp[name]
        n = dset.shape[0]
        for start in range(0, n, chunk_slices):
            digest.update(np.ascontiguousarray(
                dset[start:start + chunk_slices]).tobytes())
    return digest.hexdigest()


def _dataset_diff_stats(path_a, path_b, name, chunk_slices=64):
    max_abs = 0.0
    sq_sum = 0.0
    count = 0
    with h5py.File(path_a, "r") as fa, h5py.File(path_b, "r") as fb:
        da, db = fa[name], fb[name]
        if da.shape != db.shape:
            raise MergeError("%s: dataset %s shapes differ (%s vs %s)"
                             % (path_b, name, da.shape, db.shape))
        n = da.shape[0]
        for start in range(0, n, chunk_slices):
            diff = np.asarray(da[start:start + chunk_slices], dtype=np.float64) - \
                np.asarray(db[start:start + chunk_slices], dtype=np.float64)
            max_abs = max(max_abs, np.abs(diff).max(initial=0.0))
            sq_sum += float((diff * diff).sum())
            count += diff.size
    rms = np.sqrt(sq_sum / count) if count else 0.0
    return dict(max_abs=float(max_abs), rms=float(rms))


def check_base_grids(runs, fields, base_donor=0):
    """Verify/measure base-grid consistency across runs.

    GridDensity.0 (present only with baryons) must be bit-identical.
    Displacement/velocity base fields are measured against the donor and
    the differences reported (the irreducible merge approximation).
    RefinementMask.0 legitimately differs (it tags each run's own region).
    """
    donor = runs[base_donor]
    report = {}
    for field in fields:
        if field == "RefinementMask":
            continue
        dataset = "%s.0" % field if donor.n_levels > 0 else field
        strict = field.startswith("GridDensity")
        if strict:
            donor_hash = _dataset_hash(donor.field_file(field, 0), dataset)
            for run in runs:
                if run is donor:
                    continue
                if _dataset_hash(run.field_file(field, 0), dataset) != donor_hash:
                    raise MergeError(
                        "Run %s: %s.0 is not bit-identical to %s. The runs "
                        "do not share one realization (check seeds, shift, "
                        "kspace_TF, and cosmology)." % (run.name, field,
                                                        donor.name))
            report[field] = dict(identical=True)
        else:
            stats = {}
            for run in runs:
                if run is donor:
                    continue
                stats[run.name] = _dataset_diff_stats(
                    donor.field_file(field, 0), run.field_file(field, 0),
                    dataset)
            report[field] = dict(identical=False, diff_vs_donor=stats)
    return report


def _copy_and_renumber(src, dst, old_name, new_name):
    shutil.copyfile(src, dst)
    if old_name != new_name:
        with h5py.File(dst, "a") as fp:
            fp.move(old_name, new_name)


def _format_grid_lines(gridnum, entry):
    return [
        "CosmologySimulationGridDimension[%d]      = %16d %16d %16d"
        % tuple([gridnum] + list(entry["Dimension"])),
        "CosmologySimulationGridLeftEdge[%d]       = %.10g %.10g %.10g"
        % tuple([gridnum] + list(entry["LeftEdge"])),
        "CosmologySimulationGridRightEdge[%d]      = %.10g %.10g %.10g"
        % tuple([gridnum] + list(entry["RightEdge"])),
        "CosmologySimulationGridLevel[%d]          = %d"
        % (gridnum, entry["Level"]),
    ]


def merge_runs(run_dirs, out_dir, base_donor=0, min_gap_fine_cells=4,
               conf_paths=None, halo_ids=None):
    """Merge the MUSIC output dirs in run_dirs into out_dir.

    Returns the manifest dict (also written to out_dir/merge_manifest.json).
    """
    if conf_paths is None:
        conf_paths = [None] * len(run_dirs)
    runs = [RunInfo(d, c) for d, c in zip(run_dirs, conf_paths)]
    if halo_ids is None:
        halo_ids = [run.name for run in runs]

    fields = verify_runs(runs)
    detect_overlaps(runs, min_gap_fine_cells=min_gap_fine_cells)
    base_report = check_base_grids(runs, fields, base_donor=base_donor)

    donor = runs[base_donor]
    if os.path.exists(out_dir) and os.listdir(out_dir):
        raise MergeError("Output directory %s exists and is not empty"
                         % out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Level-major flat numbering: grid 0 = base, then every run's level-1
    # patch, then every run's level-2 patch, ...
    assignments = []  # (gridnum, run, level)
    gridnum = 0
    max_level = max(run.n_levels for run in runs)
    for level in range(1, max_level + 1):
        for run in runs:
            if level <= run.n_levels:
                gridnum += 1
                assignments.append((gridnum, run, level))
    n_grids = 1 + len(assignments)

    for field in fields:
        shutil.copyfile(donor.field_file(field, 0),
                        os.path.join(out_dir, "%s.0" % field))
    for g, run, level in assignments:
        for field in fields:
            _copy_and_renumber(
                run.field_file(field, level),
                os.path.join(out_dir, "%s.%d" % (field, g)),
                "%s.%d" % (field, level), "%s.%d" % (field, g))

    pf_lines = list(donor.header_lines)
    pf_lines += ["",
                 "CosmologySimulationNumberOfInitialGrids  = %d" % n_grids]
    for g, run, level in assignments:
        pf_lines += _format_grid_lines(g, run.grid_at_level(level))
    with open(os.path.join(out_dir, "parameter_file.txt"), "w") as fp:
        fp.write("\n".join(pf_lines) + "\n")

    manifest = dict(
        base_donor=donor.name,
        fields=fields,
        levelmin=donor.levelmin,
        shift=list(donor.shift),
        seeds={str(k): v for k, v in sorted(donor.seeds.items())},
        base_grid_report=base_report,
        grids={"0": dict(run=donor.name, level=0)},
    )
    for g, run, level in assignments:
        halo_id = halo_ids[runs.index(run)]
        entry = run.grid_at_level(level)
        manifest["grids"][str(g)] = dict(
            run=run.name, halo=halo_id, level=level,
            dimension=entry["Dimension"],
            left_edge=entry["LeftEdge"], right_edge=entry["RightEdge"])
    with open(os.path.join(out_dir, "merge_manifest.json"), "w") as fp:
        json.dump(manifest, fp, indent=2)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Merge N same-seed MUSIC runs into one multi-patch "
                    "Enzo IC directory.")
    parser.add_argument("--out", required=True, help="output IC directory")
    parser.add_argument("--runs", required=True, nargs="+",
                        help="MUSIC output directories (a sibling "
                             "<dir>.conf is expected for each)")
    parser.add_argument("--base-donor", type=int, default=0,
                        help="index into --runs whose base grid is used")
    parser.add_argument("--min-gap-cells", type=int, default=4,
                        help="minimum patch separation in parent cells")
    args = parser.parse_args(argv)
    manifest = merge_runs(args.runs, args.out, base_donor=args.base_donor,
                          min_gap_fine_cells=args.min_gap_cells)
    print("Merged %d grids into %s" % (len(manifest["grids"]), args.out))
    for field, entry in manifest["base_grid_report"].items():
        if not entry.get("identical", False):
            print("Base-grid %s differences vs donor:" % field)
            for name, stats in entry["diff_vs_donor"].items():
                print("  %s: max |d| = %.3g, rms = %.3g"
                      % (name, stats["max_abs"], stats["rms"]))


if __name__ == "__main__":
    main()
