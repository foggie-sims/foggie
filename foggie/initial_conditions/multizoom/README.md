# multizoom — multiple DM zoom regions in one 25 Mpc domain

Self-contained toolkit for generating Enzo initial conditions with **N
independent dark-matter zoom targets in a single box**.  Nothing in the
legacy IC framework (`enzo-mrp-music/`, `halo_template_256/`,
`halo_template_512/`, `music/`) is imported or modified: the wrapper
modules here began as forks of those scripts (provenance in each file
header), the Enzo changes are carried as patch files, and the templates
in `templates/` are parameterized copies.

This package sits on top of the **ics_refactor** IC pipeline (it was rebased
onto that branch on 2026-08-26; the earlier standalone `multizoom512.py`
driver was dropped because `pipeline/build.py` already does rockstar lookup,
shift bookkeeping, `.enzo` rendering and submission — better, and in
production use for the dwarf fleet).

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
| `pipeline_integration.py` | drives multizoom builds from the ics_refactor pipeline (registry groups → N-halo config) |
| `templates/` | parameterized `.enzo` / `RunScript.sh` templates + `simrun.pl` copy |
| `enzo_patches/` | five patches for enzo-foggie (apply with `git am`) |
| `music_patches/` | optional `region_shift_override` patch for MUSIC |
| `tests/` | pytest suite on synthetic MUSIC outputs (no yt/MUSIC/Enzo needed) |

## Usage

Multizoom is driven by the **ics_refactor pipeline**, which owns the halo
registry, the `Box` definitions, staging, submission and QC.  The pipeline
already drives `enzo-mrp-music` as its workhorse; multizoom slots in at that
seam, rendering one config that names N halos instead of one.  Nothing in
`pipeline/` is modified, so a multizoom build cannot disturb the production
fleet.

Group membership comes from an optional `multizoom_group` column in the halo
registry: enabled rows sharing a non-empty value form one group built into a
single domain.  A registry without the column simply has no groups.

```sh
# what groups exist?
python -m foggie.initial_conditions.multizoom.pipeline_integration groups

# inspect the config that would be rendered
python -m foggie.initial_conditions.multizoom.pipeline_integration render \
    --group dwarfs --level 1

# build one level's ICs for the group
python -m foggie.initial_conditions.multizoom.pipeline_integration build \
    --group dwarfs --level 1 --mode union      # or --mode merge
```

* `--mode union` needs no code changes anywhere: one MUSIC run over the union
  of the clouds, with the mask re-deposited per halo.  `--mode merge` runs
  MUSIC once per halo (identical seeds, `no_shift`) and merges; it needs an
  Enzo built with the patches below.
* Per-halo centres come from the pipeline's own `center_for_level`, including
  the level ≥ 2 refine-from-the-run correction, so a group member is traced
  exactly as it would be as a standalone zoom.
* Each halo's cloud is passed through `trim_lagrangian_outliers` before it is
  unioned or handed to MUSIC, so one halo's strays cannot inflate the domain.
* The merge tool can also be run standalone:

```sh
python merge_music_ics.py --out 25Mpc_DM_512-L2 \
    --runs 25Mpc_DM_512-L2-h5016 25Mpc_DM_512-L2-h5033 [--min-gap-cells 4]
```

It hard-fails on any seed/shift/cosmology mismatch, on overlapping or
nearly-touching patches (remedy: fold those halos into one union sub-run), and
on non-bit-identical `GridDensity.0` (baryon runs).  It writes a
`merge_manifest.json` recording which grid came from which run.

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
