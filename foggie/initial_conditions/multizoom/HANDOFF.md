# Session handoff — multizoom (2026-08-25/26)

State transfer for continuing this work in a new Claude Code session (or
by a human).  Companion documents: `AUDIT_AND_PLAN.md` (full audit +
staged plan), `README.md` (usage), `CLAUDE.md` (working rules).

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
