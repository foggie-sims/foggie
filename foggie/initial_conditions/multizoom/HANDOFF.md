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

### Phase 1 first union build — PASSED (2026-08-26, Aitken)

Group `pairA` = halos 15659 + 48014 (4.13 Mpc/h apart, straddling the
periodic y-boundary).  Built via `pipeline_integration build --group pairA
--level 1 --mode union` with a workspace registry
(`runs/halo_registry_multizoom.ecsv`, `multizoom_group` column) and
`MULTIZOOM_ICS_DIR` pointing at the workspace — nothing written into the
production tree; the shared L0 run was read-only input.

* Single-pass tracer found both clouds (852 + 176 particles); MUSIC
  recentred the pair across the boundary (shift −54, −227, 72).
* `validation.py union` PASSES: one 3.6 Mpc/h L1 patch, exactly 2 disjoint
  refine clouds (10000 + 2608 cells), 1.3% of the patch refined — the
  volume between the halos is untouched.
* Both clouds sit 22 / 38 kpc/h (≈1 mask cell) from their halos' traced
  Lagrangian centers after unshifting.
* Standalone production masks have fewer cells (6536 / 968) because the
  fleet builds with shape_type=convex_hull while multizoom uses the exact
  particle deposit — different mask construction, same regions.
* The L1 Enzo run (patched binary, pipeline-rendered .enzo, simrun.pl
  auto-restart) was submitted as PBS 25045576 in
  `runs/multizoom_pairA/25Mpc_DM_512-L1`; compare against the standalone
  `halo15659` / `halo48014` L1 runs when it reaches z=0.

### Sixpack six-halo multizoom ladder — overnight 2026-08-26/27 (Aitken)

Group `sixpack` = 48014, 56672, 75392, 21246, 24122, 42502 (merge mode,
common shift −134,234,125 via the patched MUSIC).  The ladder driver
(`runs/advance_sixpack.sh`) carried it L1 → L2 → L3 unattended.

* Two real bugs found and fixed live: Enzo patch **0006** (MRP creation
  required level == NumberOfInitialGrids−1, a sixth single-pyramid
  assumption; N=1 regression still byte-identical) and the legacy mask
  deposit wrapping only negative positions (four of six masks came out
  empty under the override; fixed with floor() wrap + an empty-mask guard
  in the merge tool).
* L1: 7 grids, 45,992 MRPs = mask cells exactly, z=0.  L2: 13 grids, z=0.
  L3: 19 grids, 2.45M MRPs, running (z≈2 at 02:23 wall).
* **Cost (core-h, performance.out):** standalone sum vs multizoom —
  L1 334.6 vs 69.6 (0.21); L2 441.4 vs 111.7 (0.25); L3 775.4 vs 153.6
  in progress.  Six halos for roughly a fifth to a quarter of the cost.
* **Load balance** (time-weighted max/mean, lower is better): multizoom is
  BETTER than the standalone runs on every AMR level, e.g. L1 hierarchy
  level 1: 1.36 vs 1.58; L2 level 7: 1.48 vs 1.81; L3 level 7: 1.07 vs
  1.35 — six dense regions distribute across ranks more evenly than one.
* pairA (union mode) reached z=0: halos within 4/16 kpc/h and 1%/7% in
  mass of the standalone runs, identical contamination; 59.3 core-h vs
  119.4 standalone.
* Report tool: `compare_costs.py`; figure at
  `runs/multizoom_sixpack/cost_comparison.png`.

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
