# Session handoff — multizoom (2026-08-25/26)

State transfer for continuing this work in a new Claude Code session (or
by a human).  Companion documents: `AUDIT_AND_PLAN.md` (full audit +
staged plan), `README.md` (usage), `CLAUDE.md` (working rules).

## Status update — 2026-08-26 (Aitken)

Rebased onto **ics_refactor** (the branch that actually builds the dwarf
fleet).  The branch is now exactly `ics_refactor` + the multizoom commits,
purely additive, and the merge back is conflict-free.

* `multizoom512.py` **deleted** — superseded by `pipeline/build.py`.
* New `pipeline_integration.py`: registry `multizoom_group` column → N-halo
  config, reusing the pipeline's `center_for_level` / `halo_center_and_radius`.
  Nothing in `pipeline/` is modified, so the live fleet cannot be disturbed.
* Ported from ics_refactor's enzo-mrp-music into the multizoom fork:
  `new_ics_directory`, `music_ld_library_path`, and `trim_lagrangian_outliers`
  (applied per halo, before union/merge).
* **Enzo built and validated on Aitken.**  See `BUILD_NOTES.md` in the
  workspace root (`/nobackupnfs1/jtumlins/foggie-multizoom`, not in git).
  Patched binary = `edf858c` + the 5 patches; a pristine baseline binary from
  the same commit sits beside it.
* **N=1 regression PASSED** (`runs/n1_regression`, PBS 25045407): identical
  nested ICs (256^3 + L1 + L2, MRP mode 3) through both binaries give a
  byte-identical `RD0000.hierarchy`, the same 192 grids, the same 16,787,107
  particles, and all 216 HDF5 datasets bit-identical.  The five patches are
  inert for a single zoom, as required.
* Fleet survey: of 120 pairs among the 16 enabled dwarfs, 119 have disjoint
  Lagrangian patches; only 52675 & 51741 overlap (0.098 Mpc/h).  The
  46615/47314/51741/52675 neighbourhood is the one cluster needing a union
  sub-run.

## Where things stand

Branch `claude/multi-zoom-single-domain-goqnjp` of foggie-sims/foggie,
three commits ahead of master (`d6aac60`):

1. `f24c5ab` — audit & plan document.
2. `bcb3137` — the complete multizoom package (this directory).

Everything requested so far is DONE and pushed: the audit of MUSIC /
enzo-mrp-music / Enzo-FOGGIE, and the implementation of both workflows
(union + N-run merge), under the hard constraint that **nothing in the
legacy IC framework is modified** — all Enzo/MUSIC changes are patch
files here, all Python is a self-contained fork.

Tests: 20/20 passing (`cd multizoom && python -m pytest`).  The Enzo
patches were verified with `git apply --check` against enzo-foggie head
`edf858c` and the four touched files syntax-check clean (g++
-fsyntax-only with HDF5 headers).  No enzo-foggie branch was pushed
(read-only access from the authoring session) — the patch files in
`enzo_patches/` are the source of truth; apply with `git am`.

## Decisions already made (do not re-litigate without the user)

- Staged strategy: Phase 1 "union" (validation, 2-3 halos, zero code
  changes) then Phase 2 "N-run merge" (production, 10+ halos).
- Survey mode: design scales to 10+ targets (`MAX_INITIAL_GRIDS = 64`).
- All zooms share one `MustRefineParticlesRefineToLevel` (no per-halo
  ceilings; the MultiRefineRegion bug fixes that per-halo levels would
  need are documented in AUDIT_AND_PLAN.md but NOT implemented).
- `halo_template_512` tree is the workflow being replaced; DM-only.
- Complete isolation from the legacy framework, own branch.

## Key physics facts (verified in source, not assumptions)

- FOGGIE zooms are confined purely by RefinementMask particle types +
  `MustRefineParticlesCreateParticles = 3` (no RefineRegion box), and
  Enzo's MRP flagging has no spatial predicate → N disjoint clouds
  refine independently with no Enzo changes (that is union mode).
- `kspace_TF` defaults to yes in MUSIC → same-seed, same-frame runs
  produce bit-identical base-grid density → identical tides and noise
  across regions by construction.  The merge tool ASSERTS this
  (GridDensity.0 hash when baryons present; seed/shift equality always).
- The only cross-run difference is the Poisson/2LPT solve's base-grid
  displacements (multigrid FAS feeds each run's patch back into its
  coarse solution).  Donor-run base is used; differences are measured
  and reported in merge_manifest.json — quantify, don't assume.
- Under `no_shift = yes`, a Lagrangian region wrapping the periodic
  boundary makes MUSIC hard-fail (mesh.hh:1503-1512, loud not silent).
  Remedy: `music_patches/0001-region-shift-override.patch` + one common
  shift for every run in the set.

## Next steps (require HPC — MUSIC/Enzo not runnable in the authoring env)

1. Apply `enzo_patches/*.patch` to an enzo-foggie build (`git am`),
   rebuild.  Patch 5 (duplicate-particle fix) is worth upstreaming
   regardless of multizoom.
2. N=1 regression: one existing single-halo IC set through the patched
   Enzo — hierarchy and particle IDs must match the unpatched build.
3. Phase 1 validation: 2 well-separated halos, small levelmin (7-8),
   L1-L2, `multizoom512.py --mode union --no-submit`, inspect, run to
   z=2.  Compare vs single-zoom references (halo centers, M_vir,
   contamination) per AUDIT_AND_PLAN.md Milestone 4.
4. Phase 2: same pair through `--mode merge`; check the merge tool's
   base-displacement report; startup regression (particle count
   analytic, patches at correct levels/parents); z=2 comparison.
5. 3-halo then 10-halo survey dry run (IC generation + Enzo startup).
6. Docs milestone: `doc/source/user_guide/multizoom_ics.rst` not yet
   written (README.md and AUDIT_AND_PLAN.md exist).
7. Milestone 5 (designed 2026-08-27, NOT implemented): particle-tracked
   forced refinement — runtime MultiRefineRegion boxes centered on
   per-zoom particle-ID sets instead of precomputed halo_track files,
   plus the `translate_particle_ids.py` DM→gas ID bridge.  Full design,
   verified building blocks, and effort estimate in AUDIT_AND_PLAN.md
   Part III; implement as enzo_patches 0006-0008 + the utility when the
   user asks.

## Gotchas for the next session

- Run pytest from THIS directory (pytest.ini here); from the repo root,
  pytest resolves tests through the top-level foggie package, which
  imports matplotlib/yt and fails on minimal environments.
- `templates/RunScript.sh.in` carries Pleiades-specific module loads
  and paths inherited from the legacy RunScript.sh — they will need
  adjusting for Aitken queues/modules (job template only; no code
  change needed).
- The legacy wrapper bugs are fixed only in the forks here; the legacy
  scripts still have them (by design — untouched).
- Particle IDs are NOT portable between runs initialized on different
  core counts (ParallelParticleIO/CalculatePositions assigns IDs by
  partition layout; verified at
  Grid_NestedCosmologySimulationInitializeGrid.C:1586).  Any particle-ID
  list must be generated per run, or translated via Lagrangian-position
  matching (AUDIT_AND_PLAN.md Part III).
- `multizoom512.py` merge mode expects the previous level's MERGED run
  in the workdir as `<simname>-L<n-1>/` with its `.enzo` file, as the
  driver lays out.  Level-0 (unigrid) comes from the user's existing
  planck18 MUSIC template/run; the same template (with all seed[]
  entries) must be used for every halo, level, and reference run.
