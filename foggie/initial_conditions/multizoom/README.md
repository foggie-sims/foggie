# multizoom — multiple DM zoom regions in one 25 Mpc domain

Self-contained toolkit for generating Enzo initial conditions with **N
independent dark-matter zoom targets in a single box**.  Nothing in the
legacy IC framework (`enzo-mrp-music/`, `halo_template_256/`,
`halo_template_512/`, `music/`) is imported or modified: the wrapper
modules here began as forks of those scripts (provenance in each file
header), the Enzo changes are carried as patch files, and the templates
in `templates/` are parameterized copies.

Design, audit, and physics rationale: `AUDIT_AND_PLAN.md` in this
directory.  In short:

* **union mode** — one MUSIC run whose region covers the union of the N
  halos' Lagrangian point clouds; the `RefinementMask` is re-deposited
  per cloud, and Enzo's must-refine-particle machinery (already
  region-agnostic) refines each halo independently.  **No Enzo or MUSIC
  code changes.**  Costs wasted intermediate-resolution volume between
  the halos — use it to validate, and for 2–3 targets.
* **merge mode** — one MUSIC run per halo with **identical seeds** and
  `no_shift = yes`, merged into a single multi-patch IC set (several
  nested grids per level).  No wasted volume; scales to 10+ targets.
  Requires the five Enzo patches in `enzo_patches/`.

Both modes preserve the physics invariants across regions: identical
seeds and one common domain frame make every run a sample of the same
realization (identical base-grid density ⇒ identical large-scale tides
and noise; the merge tool *asserts* this rather than assuming it), and
the residual base-grid displacement differences from MUSIC's per-run
Poisson/2LPT solve are measured and reported by the merge tool.

## Layout

| path | what |
|---|---|
| `config.py` | `[halo:<id>]` multi-halo config parsing (legacy configs still parse) |
| `lagrangian_regions.py` | trace N halos to z_init Lagrangian volumes in ONE grid pass |
| `refinement_mask.py` | deposit N particle clouds into a MUSIC `RefinementMask` |
| `mrp_music.py` | per-level MUSIC orchestration (union + merge modes) |
| `merge_music_ics.py` | merge N same-seed MUSIC runs into one multi-patch IC set |
| `multizoom512.py` | command-line driver for the 512³ 25 Mpc/h workflow |
| `templates/` | parameterized `.enzo` / `RunScript.sh` templates + `simrun.pl` copy |
| `enzo_patches/` | five patches for enzo-foggie (apply with `git am`) |
| `music_patches/` | optional `region_shift_override` patch for MUSIC |
| `tests/` | pytest suite on synthetic MUSIC outputs (no yt/MUSIC/Enzo needed) |

## Usage

Per level, from your run directory (the L(n−1) run must exist there as
`25Mpc_DM_512-L<n-1>/` with its `.enzo` file, as this driver lays out):

```sh
python multizoom512.py --halo_ids 5016,5033,2392 --level 1 --mode union \
    --music-exe-dir /path/to/music-build \
    --music-template /path/to/25Mpc_DM_512_planck18.conf \
    --workdir $PWD [--no-submit] [--email you@example.edu]
```

* `--mode union` needs no code changes anywhere. `--mode merge` runs
  MUSIC once per halo and merges; the Enzo executable must include the
  patches below.
* The MUSIC template must list `seed[<level>]` for every level the
  deepest zoom will reach, and the SAME template must be used for every
  halo, level, and reference run.
* In merge mode nothing ever shifts: halo centers stay in the catalog
  frame at every level, and the legacy shift-tracking files are not
  used.  In union mode the driver applies the accumulated `conf_log`
  shifts to the catalog centers exactly as `script512.py` did (but with
  the correct 1/2^levelmin cell size).
* The merge tool can also be run standalone:

```sh
python merge_music_ics.py --out 25Mpc_DM_512-L2 \
    --runs 25Mpc_DM_512-L2-h5016 25Mpc_DM_512-L2-h5033 [--min-gap-cells 4]
```

It hard-fails on any seed/shift/cosmology mismatch, on overlapping or
nearly-touching patches (remedy: concatenate those halos' point files
into one union sub-run), and on non-bit-identical `GridDensity.0`
(baryon runs).  It writes a `merge_manifest.json` recording which grid
came from which run.

## Enzo patches (merge mode only)

Five minimal, upstream-friendly fixes; apply to `foggie-sims/enzo-foggie`
(`edf858c` or later) and rebuild:

```sh
cd enzo-foggie
git am /path/to/foggie/foggie/initial_conditions/multizoom/enzo_patches/*.patch
```

1. **Parent search** (`CosmologySimulationInitialize.C`,
   `NestedCosmologySimulationInitialize.C`) — containment must hold in
   every dimension; the old search was order-dependent with several
   grids per level.
2. **`MAX_INITIAL_GRIDS` 10 → 64** — survey mode (10 halos × 4 levels).
3. **Partition walk** (`InitializeNew.C`) — partition every initial
   grid, including same-level siblings.
4. **`NestedCosmologySimulationReInitialize`** — map hierarchy grids to
   initial grids by same-level edge containment instead of assuming
   grid number == level; fixes IC file assignment, sibling children,
   and the global particle count.
5. **`Grid_CosmologyInitializeParticles.C`** — a static region that
   doesn't overlap the grid must not abort the scan over the remaining
   regions (`break` → `continue`); prevents duplicate coarse particles
   under later-scanned patches.

Patch 5 fixes a latent bug worth having even without multizoom; patches
1–4 are inert for single-pyramid runs (validated by the N=1 regression
in the plan).

## MUSIC patch (optional)

`music_patches/0001-region-shift-override.patch` adds
`setup/region_shift_override = sx,sy,sz` (integer coarse cells): one
fixed shift shared by all runs, taking precedence over the auto-shift
and `no_shift`.  Needed only when a target's Lagrangian region sits
close enough to the periodic boundary that MUSIC's placement fails
under `no_shift = yes` (the driver detects this and says so).  Apply to
a BUILD COPY of MUSIC (`patch -p1 < ...` from the repo root, or `-p4`
inside the music source dir); the vendored tree in this repository is
left as-is.

## Tests

```sh
cd multizoom && python -m pytest
```

Runs on synthetic MUSIC outputs; needs numpy, h5py, pytest (scipy
optional, yt not needed).  Covers config parsing, the multi-cloud mask
deposit, and the merge tool's renumbering and every abort path.

## Not in scope (yet)

Gas/baryon multizoom (the merge tool already handles `GridDensity` /
`GridVelocities_*` files generically), per-halo refinement ceilings
(needs the MultiRefineRegion fixes listed in AUDIT_AND_PLAN.md), and
the full in-code MUSIC multi-region refactor.

**Planned — particle-tracked forced refinement** (AUDIT_AND_PLAN.md
Part III): runtime MultiRefineRegion boxes centered on per-zoom
particle-ID sets instead of precomputed `halo_track` files, with a
DM→gas ID translation utility.  Designed and estimated; to be
implemented as `enzo_patches/0006-0008` plus
`translate_particle_ids.py`.
