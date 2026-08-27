# Multiple DM Zoom Regions in a Single 25 Mpc Domain

**Audit of Enzo-FOGGIE and Enzo-MRP-MUSIC, and a staged implementation plan**

This document audits the two codes FOGGIE uses to produce zoom simulations —
Enzo-FOGGIE (`foggie-sims/enzo-foggie`) and the MUSIC-based IC pipeline vendored
in this repository (`foggie/initial_conditions/music/` +
`foggie/initial_conditions/enzo-mrp-music/`) — to identify exactly what limits
each to following **one zoom region per domain**, and lays out a plan to
support **many independent DM zoom targets in the same 25 Mpc box**, with
explicit attention to the physics that must be preserved across regions:

- **Tides** — every zoom must feel the identical large-scale tidal field it
  would feel in a single-zoom run of the same realization.
- **Power-spectrum noise** — all regions must be samples of one underlying
  white-noise field (identical seeds, identical domain frame), so the noise
  realization is consistent between regions and with reference runs.
- **IC potential-solve errors** — MUSIC's Poisson/2LPT solve couples fine
  patches back into the coarse solution; the multi-zoom scheme must keep any
  cross-region error quantified and bounded.

Scope decisions (agreed 2026-08-25): staged strategy (Phase 1 "union" for
validation, Phase 2 "N-run merge" for production efficiency); design scales to
**10+ targets** (survey mode); all zooms share one
`MustRefineParticlesRefineToLevel`; the `halo_template_512` workflow is the
tree to modernize; DM-only.

---

## Part I — Audit

### 1. How the current single-zoom pipeline works

Per level *n* (driven by `halo_template_512/script512.py`, iterating L1→L3):

1. `enzo-mrp-music.py <conf> <n>` parses a single-halo config
   (`halo_center`, `halo_mass`/`halo_radius`, `radius_factor`, `shape_type`).
2. It recovers the previous level's domain recentring shift from MUSIC's
   `<name>-L<n-1>.conf_log.txt` (`"Domain shifted by (…)"`).
3. `get_halo_initial_extent.get_center_and_extent()` selects the halo's
   particles in the last output of the L(n−1) Enzo run, traces them to the
   z=99 dump, and writes one Lagrangian point file
   (`initial_particle_positions-<id>-<ds>.dat`).
4. A MUSIC conf for level *n* is generated: `levelmax = levelmin + n`,
   plus either `ref_center`/`ref_extent` (box mode) or
   `region_point_file` + `region_point_shift` + `region_point_levelmin`
   (ellipsoid / convex_hull / exact modes).
5. MUSIC runs; for `shape_type = exact`, `particle_only_mask.py` overwrites the
   innermost `RefinementMask.<finest>` with a CIC deposit of the actual
   Lagrangian point cloud (Gaussian-smoothed edges).
6. The wrapper appends the must-refine-particle (MRP) block to MUSIC's
   `parameter_file.txt`:
   `MustRefineParticlesCreateParticles = 3`,
   `MustRefineParticlesRefineToLevel = <n>`,
   `CosmologySimulationParticleTypeName = RefinementMask` —
   with the explicit note *"do NOT include the RefineRegion parameters above"*.
7. `script512.py` greps MUSIC's `CosmologySimulationGrid*` lines into the
   `25Mpc_DM_512-L<n>.enzo` template and submits via `RunScript.sh`/`simrun.pl`.

**The critical observation:** FOGGIE zoom confinement in Enzo uses **no
`RefineRegion` box at all**. It is done entirely by the `RefinementMask`
particle types + `MustRefineParticlesCreateParticles = 3` +
`CellFlaggingMethod = 4 8`.

### 2. Enzo-FOGGIE: what is already multi-region, and what is not

#### Already region-agnostic (no changes needed)

- **MRP flagging** (`src/enzo/Grid_DepositMustRefineParticles.C`): a
  per-particle CIC deposit into the flagging field with **no spatial predicate
  anywhere**. N disjoint clouds of must-refine particles each trigger
  refinement wherever they are, independently, to
  `MustRefineParticlesRefineToLevel`. *This alone supports multiple
  simultaneous zooms at runtime.*
- The single-box `RefineRegionLeftEdge/RightEdge` limiter is switched off in
  FOGGIE templates (absent = whole domain).
- `StaticRefineRegion*` supports up to `MAX_STATIC_REGIONS = 1000` regions.
- `MultiRefineRegion*` (CellFlaggingMethod 20; FOGGIE additions by A. Wright,
  2023–24) supports 1000 static + 20 track-evolving regions with per-region
  min/max level and SF mass floors — not needed for this plan (uniform levels)
  but note two latent bugs for future per-halo-level work:
  `LocalMaximumRefinementLevel`/`LocalMinimumRefinementLevel` are declared at
  function scope and never reset per cell
  (`Grid_FlagCellsToBeRefinedByMultiRefineRegion.C:43-44`), and the documented
  per-region *maximum* level cap is never enforced (no unflagging; that logic
  only existed in the orphaned 2013 `Grid_SetFlaggingFieldMultiRefineRegions.C`).
- Timestepping and load balancing: one global dt per level (min over all
  grids), so N zooms cost roughly additively in refined volume — nothing is
  hard-coded about a single dense region. `LoadBalancing = 4` (Hilbert SFC) is
  likely better than the current `1` for several spatially separated clusters.

#### Blockers for multiple nested IC patches at the same level (verified)

The IC reader assumes one nested patch per level in five places:

| # | Location | Problem |
|---|----------|---------|
| E1 | `NestedCosmologySimulationInitialize.C:410-422` and `CosmologySimulationInitialize.C:460-475` | Parent-search bug: `ParentGrid = i` is assigned as soon as dim 0 passes and is **never cleared when a later dim fails**, and the candidate loop does not stop on success. Harmless with one patch per level; order-dependent and wrong with sibling patches. |
| E2 | `CosmologySimulationInitialize.C:99`, `NestedCosmologySimulationInitialize.C:93` | `MAX_INITIAL_GRIDS = 10` (local `#define`). Survey mode needs ~41+ (10 halos × 4 levels + base). The `"%s.%1"ISYM` filename suffix is *not* a blocker — width 1 is a minimum, multi-digit grid numbers print fine. |
| E3 | `InitializeNew.C:963-987` | The partition walk descends only the first-child chain (`NextGridNextLevel`) and **skips partitioning any grid that has a sibling** (`if (CurrentGrid->NextGridThisLevel == NULL)` guard). A second same-level patch is never partitioned and never visited. |
| E4 | `NestedCosmologySimulationInitialize.C:915-1083` | `NestedCosmologySimulationReInitialize` equates grid number with level (`CurrentGrid = CurrentGrid->NextGridNextLevel`, one filename set per level). Two logically distinct level-1 patches would both be handed patch 0's IC files. |
| E5 | `Grid_CosmologyInitializeParticles.C:107` | **Newly found duplicate-particle bug** (verified in source): in the static-region particle-exclusion loop, `if (skip) break;` aborts the scan over *all* remaining regions as soon as one same-level region fails to overlap the grid. With N zoom patches per level, coarse particles underneath patch B are not removed whenever patch A is checked first → duplicated mass. Fix is one word: `break` → `continue`. |

Also: `RefineRegionAutoAdjust` is single-box by construction (shrinks one box
until no coarse particle is inside) and will collapse or fail with two
separated targets — it must stay off (it already defaults off in FOGGIE).

### 3. MUSIC: where "one region" is hard-coded

The vendored code is MUSIC 1 (hg archive, ~2015). Upstream MUSIC2 has the same
single-zoom architecture and monofonIC is unigrid-only — there is **no prior
art to port**.

| # | Subsystem | Single-region assumption | Difficulty to generalize |
|---|-----------|--------------------------|--------------------------|
| M1 | `region_generator.hh:28-44`, `main.cc:454` | One global `the_region_generator`; interface returns one AABB per level. (`region_multibox.cc` can describe disjoint volumes in `query_point` but its `get_AABB` collapses them to one bounding box.) | Easy |
| M2 | `mesh.hh:1264-1855` (`refinement_hierarchy`) | Every geometry array indexed by level only — exactly one rectangular patch per level; one global recentring shift (`xshift_`, :1367-1396) centers *the* region; the nesting cascade (:1557-1631) halves+pads a single box chain. | **Hard — the crux** |
| M3 | `mesh.hh:533-540, 1096-1107` (`GridHierarchy`) | One mesh per level; `add_patch` chains off `.back()`. | Hard |
| M4 | `densities.cc:549-897` | Strictly linear parent→child cascade (`coarse = fine`); one `zero_subgrid` hole per parent (:744, :805); `fft_interpolate` (:170-193) hard-codes one-child offset arithmetic. | Hard |
| M5 | `random.cc:1618-1756` | One noise sub-box per level; the parent-level RNG is deleted immediately after refining (:1726); one `wnoise_%04d.bin` per level. | Medium |
| M6 | `mg_solver.hh:220-223` | Multigrid V-cycle: one fine + one coarse mesh per level. | Hard (avoidable) |
| M7 | `plugins/output_enzo.cc` | One HDF5 file per field per level (suffix = level, :226-231); the `.enzo` snippet hard-wires grid index == level (:439-461); writes one `RefineRegion` box (:463-490 — FOGGIE deletes those lines anyway). | Medium |

**Why a full in-code multi-region refactor is not the right first move:** the
single-region assumption pervades four core subsystems (M2–M5) with no
upstream reference implementation, and the same physical result is obtainable
without touching them (below).

### 4. The physics escape hatch (what makes multi-zoom cheap)

Three verified properties of MUSIC's algorithm make N *separate* runs with
identical seeds mutually consistent:

1. **`kspace_TF` defaults to `yes`** (`main.cc:378-379`), so the fine→coarse
   oct-average back-propagation (`correct_avg`) is **not executed** in the
   FOGGIE configuration. The density cascade is strictly top-down: the base
   grid is convolved from the full-box noise field and is therefore
   **bit-identical across runs that share seeds and domain frame**. Identical
   base density ⇒ identical large-scale tides, by construction.
2. **The white noise is a local, deterministic refinement** of the shared
   parent-level field with per-level seeds; fine noise cubes are indexed by
   absolute position (`random_numbers::fill_subvolume`, `random.cc:712-735`).
   Disjoint windows drawn with the same seeds are consistent samples of one
   underlying field — the noise realization is common to all regions.
3. **The only cross-run difference is in the Poisson/2LPT solve**: each run's
   base-grid displacements include the multigrid FAS fine-to-coarse correction
   only within its own refined window. For disjoint regions this cross-region
   coupling is weak and, crucially, base-grid values *underneath* the patches
   are unused by Enzo (coarse particles there are removed — once E5 is fixed).
   It is the one approximation to measure, not assume (validation §9).

The Enzo IC file format itself is not a constraint:
`CosmologySimulationNumberOfInitialGrids` is a free integer and
`CosmologySimulationGridLevel[i]` an independent array — multiple grids may
share a level once blockers E1–E5 are fixed.

### 5. Every single-target assumption in the wrapper/driver layer

Cataloged for implementation (representative; full trace in the audit notes):

- `enzo-mrp-music.py`: one `halo_center`/`halo_mass`/`halo_radius`
  (:60-82); one `params["halo_info"]`; one `ref_center`/`ref_extent`/
  `region_point_file` written (:203-217); run dirs keyed only on
  `simulation_name`+level (:116-119) — two halos in one directory collide.
- `get_halo_initial_extent.py`: halo id defaults to 0 (:181-182) so every run
  writes the same `initial_particle_positions-0-<ds>.dat` (:301); one sphere,
  one bounding region, one center returned; fixed diagnostic plot filenames.
- `particle_only_mask.py`: reads only the innermost grid's edges
  (`CosmologySimulationGrid*Edge[finest_level]`, :38-47); deposits one point
  set, overwriting the whole mask.
- `script512.py`: scalar `--halo_id`; one rockstar row; awk substitution of a
  single `halo_center`/`halo_radius`; per-level shift bookkeeping assumes one
  region; `mod_param_file` appends whatever single grid set MUSIC produced.
- Fragilities to fix in passing: MUSIC exe path hard-coded to
  `/nobackupnfs1/jtumlins/...` (`enzo-mrp-music.py:226`), h5py `.value`
  (removed in h5py ≥ 3), `scipy.signal.gaussian` (moved to
  `scipy.signal.windows`), `method='halo'` NameError in
  `get_halo_initial_extent.py`, `raise RuntimeWarning` aborting when both mass
  and radius are set, `particle_only_mask.py` `except` branch enabling
  smoothing while claiming to disable it.

---

## Part II — Implementation plan

### Strategy

**Phase 1 — "union" mode (validation + small-N production).** One MUSIC run
whose region covers the union of the N halos' Lagrangian point clouds, with
the `RefinementMask` tagging each cloud individually. Enzo's MRP machinery
refines each cloud independently — **zero Enzo or MUSIC code changes**. Cost:
wasted intermediate-resolution volume between halos (the nested grids must
enclose the union), acceptable for 2–3 halos and for validating the physics.

**Phase 2 — "N-run merge" (survey production).** Run MUSIC once per halo with
**identical seeds** and `no_shift = yes` (a common frame), then merge the N
single-halo IC sets into one Enzo IC directory with multiple same-level
patches — no wasted volume between halos. Requires the five small Enzo patches
(E1–E5) and a new Python merge tool; MUSIC's core stays untouched.

The full in-code MUSIC multi-region refactor (M2–M5) is deliberately deferred:
Phase 2 achieves the same efficiency with a fraction of the risk, and the
validation battery will tell us whether the residual base-grid approximation
ever matters.

### Architecture decisions

- New package **`foggie/initial_conditions/multizoom/`** (this directory):
  driver, merge tool, validation harness, Enzo patches, templates, docs.
  `enzo-mrp-music/` stays near-vendored with a minimal diff.
- **Config**: INI with `[halo:<id>]` sections extending the existing `.conf`
  format; `[region]` keys become per-halo defaults. No `[halo:*]` sections ⇒
  exactly today's single-halo behavior (backward compatible).

  ```ini
  [setup]
  simulation_name = 25Mpc_DM_512
  ...
  [region]                 # defaults for all halos
  final_redshift = 0.0
  radius_factor = 1.0
  shape_type = exact       # required for multi-halo
  [halo:5016]
  halo_center = 0.493, 0.508, 0.461
  halo_radius = 205.
  halo_radius_units = kpc
  [halo:5033]
  halo_center = ...
  halo_mass = 8.3e11
  ```

- **Naming**: merged/union runs keep `<simname>-L<n>` (existing templates keep
  working); per-halo MUSIC dirs `<simname>-L<n>-h<halo_id>`; point files
  `initial_particle_positions-<halo_id>-<ds>.dat`; union file
  `initial_particle_positions-union-<tag>-<ds>.dat` with
  `tag = "+".join(halo_ids)`; confs `halos_<tag>_DM_<n-1>to<n>.conf`.
- **Per-level Lagrangian tracing always uses the single merged/union L(n−1)
  run** — one Enzo run per level, and the traced regions are consistent with
  the actual production realization *including mutual tides*. Single-halo runs
  survive only as validation references.
- **Enzo changes** live on branch `multizoom` of
  `foggie-sims/enzo-foggie` (one commit per fix), exported with
  `git format-patch` into `multizoom/enzo_patches/` here, so the foggie branch
  is self-contained and the patches reviewable/appliable via `git am`.

### Milestone 0 — modernization and bug fixes (shared foundation)

In `enzo-mrp-music/`:

- `enzo-mrp-music.py`: use `params["music_exe_dir"]` (+ optional
  `music_env` key) instead of hard-coded paths; `os.system` →
  `subprocess.run(check=True)`; `raise RuntimeWarning` → `warnings.warn` and
  actually default to mass; thread the real halo `id` into `halo_info`
  (fixes point-file collisions).
- `get_halo_initial_extent.py`: fix `method='halo'` NameError; `.value` →
  `[()]`.
- `particle_only_mask.py`: `scipy.signal.windows.gaussian` (with fallback);
  fix the inverted `smooth_edges` in the `except` branch.
- `script512.py` left untouched for reproducibility; deprecation note in its
  README.

**Gate:** modernized wrapper on one existing single-halo config produces a
MUSIC `.conf` and `parameter_file.txt` identical to the old script's output
(modulo paths).

### Milestone 1 — Phase 1 union mode

1. `parse_config()` → `params["halos"] = OrderedDict{id: halo_info}`.
2. New `find_lagrangian_regions(params)`: per-halo
   `get_center_and_extent(..., output_format="txt")`, then concatenate the
   per-halo `.dat` files (bare xyz rows) into the union point file; keep
   per-halo regions for the manifest.
3. New `get_centers_and_extents(halos, ...)` in `get_halo_initial_extent.py`:
   load the final dataset once and make **one pass** over `pf.index.grids`
   matching all halos' particle-index sets (the current code re-reads
   everything per halo — untenable at N=10+). Single-halo function becomes a
   wrapper.
4. `run_music()`: pass the union point file. **Force `shape_type = exact` for
   multi-halo**: `convex_hull` would build one hull spanning all clouds and
   the mask would MRP-refine the space between halos; `exact` mode's
   `particle_only_mask()` already deposits an arbitrary point set, keeping the
   MRP clouds per-halo. (Grid *placement* still encloses the union — that is
   the accepted Phase 1 cost.)
5. MRP block unchanged (shared `MustRefineParticlesRefineToLevel`).
6. New driver `multizoom/multizoom512.py` replacing the awk/sed pipeline:

   ```
   python multizoom512.py --halo_ids 5016,5033,2392 --level 2 --mode union \
          [--catalog .../halo_catalogs_512/512/z0/out_0.list] [--no-submit]
   ```

   Reads rockstar rows per ID (as `script512.py:182-191`), writes the
   multi-halo `.conf` with configparser, calls `enzo-mrp-music.py`, copies
   parameterized templates from `multizoom/templates/` (derived from
   `25Mpc_DM_512-L<n>.enzo`, `RunScript.sh`, `simrun.pl`), appends the
   `CosmologySimulationGrid*` lines from MUSIC's `parameter_file.txt` in
   Python, optionally submits. Shift arithmetic from
   `set_1to2_conf`/`set_2to3_conf` ported to
   `apply_domain_shift(centers, conf_log_path)`.
7. Template tweaks: `LoadBalancing = 4`; never emit `RefineRegion*` or
   `RefineRegionAutoAdjust`.

### Milestone 2 — Enzo-FOGGIE patch set (five minimal patches)

1. **Parent search** (E1): all-dims containment check, reset per candidate,
   stop on first success — in both `NestedCosmologySimulationInitialize.C` and
   `CosmologySimulationInitialize.C`:

   ```c
   int ParentGrid = INT_UNDEFINED;
   for (i = 0; i < gridnum && ParentGrid == INT_UNDEFINED; i++) {
     if (CosmologySimulationGridLevel[i] != CosmologySimulationGridLevel[gridnum]-1)
       continue;
     int contained = TRUE;
     for (dim = 0; dim < MetaData.TopGridRank; dim++)
       if (CosmologySimulationGridLeftEdge[gridnum][dim]  < CosmologySimulationGridLeftEdge[i][dim] ||
           CosmologySimulationGridRightEdge[gridnum][dim] > CosmologySimulationGridRightEdge[i][dim])
         { contained = FALSE; break; }
     if (contained) ParentGrid = i;
   }
   ```
2. **`MAX_INITIAL_GRIDS` 10 → 64** (E2) in both files. No filename-format
   change needed.
3. **Partition walk** (E3): snapshot the full initial-grid list first
   (recursing both `NextGridNextLevel` *and* `NextGridThisLevel` — sibling
   chains at level l+1 hang off different parents), then partition each
   collected entry; drop the sibling-skip guard.
4. **`NestedCosmologySimulationReInitialize`** (E4): replace the
   `gridnum == level` walk with a full-hierarchy walk that maps each grid to
   its initial gridnum by same-level edge containment (ε = ¼ finest cell)
   against the already-populated static `CosmologySimulationGrid*` arrays;
   iterate **gridnum-major** so particle-ID assignment is deterministic and
   the N=1 case reproduces current behavior exactly.
5. **Static-region exclusion loop** (E5): `if (skip) break;` →
   `if (skip) continue;` in `Grid_CosmologyInitializeParticles.C:107`.

Implementation-time checks: `CommunicationPartitionGrid` internals (the one
file not fully audited) must tolerate pre-existing siblings;
`Grid_NestedCosmologySimulationInitializeGrid.C:345`
(`StaticRefineRegionLevel[InitialGridNumber-1]+1`) stays correct under
level-major grid numbering — re-verified in the two-halo startup test.

### Milestone 3 — Phase 2 N-run merge

**Workflow per level n** (`multizoom512.py --mode merge`): trace all halos from
the merged L(n−1) run → per-halo point files → per-halo MUSIC confs with
`no_shift = yes`, **identical `[random]` seeds**, identical
levelmin/levelmax, `region = convex_hull` + exact-mask post-processing,
`output/filename = <simname>-L<n>-h<id>` → run MUSIC N times (serial or
`--jobs` pool) → merge tool → append MRP block, attach template, submit.

Shift handling: in merge mode the domain is never recentred
(`region_shift ≡ 0` at every level); the legacy `conf_log` shift bookkeeping
is bypassed and halo centers stay in the catalog frame. Validation reference
runs must also use `no_shift = yes` so all realizations share one base field.

**Merge tool `multizoom/merge_music_ics.py`**
(`merge_runs(run_dirs, out_dir, base_donor=0, overlap="abort",
min_gap_fine_cells=4) -> MergeManifest`):

1. **Parse & verify**: from each run, `parameter_file.txt` + `.conf` +
   `.conf_log.txt`; assert identical cosmology, levelmin, zstart, per-level
   seeds, shifts (all zero or one common override), base dims.
2. **Base-grid consistency**: chunked hash of `GridDensity.0` across runs —
   **must be bit-identical** under `kspace_TF = yes` (hard-fail catches any
   seed/config mistake and enforces the tides/noise guarantee). For
   `ParticleDisplacements/Velocities.0`, compute and report max/RMS cross-run
   differences (expected small, nonzero — the FAS/2LPT coupling); grid 0 is
   taken from `--base-donor`. A `--base-merge windowed` option (off by
   default) covers hypothetical `kspace_TF = no` configs by overwriting the
   donor base inside each run's own window.
3. **Overlap detection**: pairwise AABB intersection per level; any overlap,
   or gap < `min_gap_fine_cells × dx_parent`, aborts with guidance
   ("concatenate those halos' point files into one union sub-run") — a union
   sub-run is just a run with a bigger patch and is fully supported. Also
   assert each level-l patch lies inside its own run's level-(l−1) patch.
4. **Flat, level-major grid numbering**: g=0 base, then
   `for l in 1..Lmax: for run in runs`. Parents precede children; the
   static-region ordering invariant holds.
5. **Copy + rename files AND HDF5 datasets**: Enzo's `ReadFile.C:146` opens
   the dataset *named after the full filename including the suffix*, so
   `ParticleDisplacements_x.2` of run B becomes file `ParticleDisplacements_x.5`
   containing dataset `ParticleDisplacements_x.5` (`h5py` `f.move`). Fields:
   `ParticleDisplacements_{x,y,z}`, `ParticleVelocities_{x,y,z}`,
   `RefinementMask` (+ `GridDensity`/`GridVelocities_*` kept generic for a
   future gas mode). Dataset attributes are geometry-only and survive the
   rename.
6. **Merged `parameter_file.txt`**: donor header verbatim (cosmology, field
   base-name keys — Enzo appends `.g` itself);
   `CosmologySimulationNumberOfInitialGrids = 1 + Σ_i (levelmax_i − levelmin)`;
   per-grid `GridDimension/LeftEdge/RightEdge/Level[g]`; no `RefineRegion*`.
7. **`merge_manifest.json`**: gridnum → {run, halo_id, level, edges, dims} +
   seed/shift provenance, for downstream analysis and debugging.

**MUSIC changes: none required.** One verified caveat: under `no_shift = yes` a
Lagrangian region wrapping the periodic boundary hard-fails loudly
("Internal refinement bounding box error 1", `mesh.hh:1503-1512`). The driver
pre-checks each traced cloud; if any target sits near the box edge, apply the
optional ~15-line MUSIC patch adding
`setup/region_shift_override = sx,sy,sz` (integer coarse cells, `mesh.hh:1367-1396`):
one **common** shift shared by every run (references included) preserves all
merge invariants — a shared translation of a periodic field — while moving all
regions off the boundary.

### Milestone 4 — validation, survey dry run, docs

Test configuration: 25 Mpc box, levelmin 7–8 (128³/256³), two well-separated
halos, L1–L2, `no_shift = yes` everywhere.
Runs: **R0** (unigrid), **R_A**/**R_B** (single-zoom references), **U_AB**
(union), **M_AB** (merged).

IC-level (`multizoom/validate_multizoom.py`; h5py/numpy/yt):

1. `GridDensity.0` bit-identical across R_A, R_B, and M_AB's source runs —
   the tides + noise guarantee, asserted not assumed.
2. Base-grid displacement/velocity max & RMS |Δ| between runs, reported inside
   each window vs outside all windows — quantifies the IC potential-solve
   cross-region coupling in coarse-cell units.
3. Zoom fidelity: within window A, RMS and 99th-percentile |Δx|, |Δv| between
   M_AB and R_A; density cross-power coherence r(k) > 0.999 below the window
   Nyquist; same for U_AB vs R_A.
4. Merged base-grid P(k) vs R0 (regression).
5. `RefinementMask` cell counts per cloud equal across U_AB / M_AB / refs.

Enzo-level (DM-only to z=2, all four setups):

6. Startup regression for the patch set: hierarchy dump at t=0 (every patch at
   the correct level with the correct parent), total particle count matching
   the analytic expectation (tests E5 — no duplicates), particle mass spectrum
   (tests E3/E4 — no wrong-level masses), MRP counts per cloud.
7. z=2 science: halo A/B centers within a few finest cells and M_vir within
   1–2% of R_A/R_B; **zero** low-resolution particles within 3 R_vir; union
   vs merge agreeing to the same tolerance.
8. Cost accounting: cells/grids per level and wallclock, U_AB vs M_AB
   (quantifies the efficiency win that motivates Phase 2); one config with
   `LoadBalancing = 1` vs `4`.
9. N=1 regression through the patched Enzo: R_A's ICs produce an identical
   hierarchy and particle IDs pre/post patch set.

Then a **survey-mode dry run**: 3 halos, then 10 halos (IC generation + Enzo
startup only) exercising `MAX_INITIAL_GRIDS = 64`, static-region ordering, and
`LoadBalancing = 4`.

Documentation: `multizoom/README.md` (workflow, config schema,
`git am enzo_patches/*.patch`, validation harness);
`doc/source/user_guide/multizoom_ics.rst` (physics caveats: seed/shift
invariants, overlap policy, base-donor approximation); note in
`enzo-mrp-music/README`.

### Ordered task list

1. (M0) Wrapper fixes + regression diff against current outputs.
2. (M1) `[halo:<id>]` parsing; `find_lagrangian_regions`; union point file;
   single-pass `get_centers_and_extents`.
3. (M1) `multizoom512.py` union mode + `multizoom/templates/`; 2-halo union
   run at 128³, L1–L2.
4. (M2) Enzo patches E1–E5 on the enzo-foggie branch; export to
   `multizoom/enzo_patches/`; N=1 regression.
5. (M3) `merge_music_ics.py` + manifest; unit tests on tiny (32³) synthetic
   two-run directories (rename, overlap-abort, seed-mismatch-abort).
6. (M3) Driver merge mode (`no_shift` plumbing, per-halo confs, boundary
   pre-check); optional `region_shift_override` MUSIC patch if any test halo
   wraps.
7. (M4) Validation battery at 128³ / 2 halos; tune `min_gap_fine_cells` and
   the donor policy from measured seam errors.
8. (M4) 3-halo, then 10-halo survey dry run.
9. (M4) Docs.

### Risks and open questions

- **Periodic wrap under `no_shift`** — loud hard-fail, mitigated by the
  pre-check + common `region_shift_override`. Whether any real survey target
  wraps is decided empirically at tracing time.
- **Base-grid displacement donor approximation** (FAS back-reaction + 2LPT
  cross-terms) — expected tiny and largely unused (coarse particles beneath
  patches are removed once E5 lands); measured explicitly in validation
  item 2. Fallback if seams ever matter: a linear-superposition base merge,
  `donor + Σ_i (run_i − no-zoom reference)`, exact for the linear part —
  implement only if the metrics demand it.
- **Close halo pairs / overlapping intermediate patches** — policy is
  abort-with-guidance; the remedy (a union sub-run of the offending pair) is
  supported by construction.
- **`CommunicationPartitionGrid` sibling assumptions** — the one un-audited
  routine; check its internals while implementing E3.
- **Shared per-level timestep** — with N zooms the level-l dt is the min over
  all regions; cost is additive in refined volume. Inherent, documented, not
  changed.
- **Gas (baryon) multizoom** — out of scope (DM-only per requirements), but
  the merge tool keeps the baryon fields generic so a `-gas` mode is a config
  change, not a redesign.

### Deliverables

- **foggie-sims/foggie**, branch `multizoom`:
  this document; then (as implementation proceeds) the `multizoom/` package
  (driver, merge tool, validation harness, templates, `enzo_patches/`,
  README), the `enzo-mrp-music/` modernization, and user-guide docs.
- **foggie-sims/enzo-foggie**, companion branch of the same name: the five
  patches E1–E5, mirrored as `git format-patch` files in
  `multizoom/enzo_patches/` so the foggie branch is self-contained and the
  patches can be applied with `git am`.

---

## Part III — Milestone 5: particle-tracked forced refinement (designed, not yet implemented)

Added 2026-08-27.  Replaces the precomputed `halo_track` /
`MustRefineRegion` box of production runs with a box **re-derived at
runtime from a discrete set of particle IDs per zoom target**.  This
removes the track chicken-and-egg (a track requires a completed prior
run with similar physics; N halos would need N track files) and gives N
independently-moving forced-refinement boxes naturally — the runtime
complement to the multi-patch ICs of Milestones 1–3.

### Verified building blocks (all present in enzo-foggie)

| piece | where | role |
|---|---|---|
| Sorted-ID bisection matching against a grid's particles | `Grid_MustRefineParticlesFlagFromList.C` (Simpson & Bryan 2009) | template for the new "match and accumulate" grid method |
| Per-root-step geometry hook | `SetEvolveRefineRegion` called from `EvolveHierarchy.C:477` (and at startup, :292) | where the live box update slots in |
| Global reductions | `CommunicationSumValues` overloads, `CommunicationUtilities.C:258-366` | combine per-processor partial sums |
| N-region storage with per-region min/max level and SF mass floor | `MultiRefineRegion*` arrays (1000 static + 20 tracks; FOGGIE 2023-24) | the boxes the update writes into |

### Design

New parameters (one entry per zoom target):

```
RefineRegionParticleListFile[i]  = <sorted particle-ID file, one per zoom>
RefineRegionParticleHalfWidth[i] = <box half-width, comoving code units>
```

Each root-grid timestep (same cadence as the existing track
interpolation): scan every grid's particles against the cached ID sets
(bisection, as in FlagFromList); accumulate positions **relative to the
previous box center with ±0.5 periodic wrap**; reduce across
processors; take the **per-axis median** (or an iterative
shrinking-sphere center — not a plain mean: a fixed ID set accumulates
outliers as particles are stripped or ejected); set
`MultiRefineRegionLeftEdge/RightEdge[i] = center ± halfwidth` (and the
CoolingRefineRegion analog where wanted).  No restart state — the box
is re-derived from the IDs every step.  Choose the ID set from the
halo's most-bound inner particles (~10^2–10^3), not the full Lagrangian
volume.

Cost: one O(N_particles) bisection scan per root step — negligible at
that cadence.

### Effort estimate

Roughly 400–600 lines across ~8–10 files, no solver/gravity/IO changes:
one new grid method (~120 lines adapted from FlagFromList), one
top-level per-step routine (~150 lines), parameter plumbing
(`ReadParameterFile` / `WriteParameterFile` / `SetDefaultGlobalValues` /
`global_data.h` / `Grid.h` / `Make.config.objects`), plus the two
already-documented MultiRefineRegion fixes this depends on (the latched
per-cell level variables and the never-enforced per-region maximum
level, ~40 lines).  Planned as patches `0006–0008` in
`multizoom/enzo_patches/` when implemented.

### Particle-ID consistency between DM-only and gas runs (verified)

In FOGGIE's IO configuration (`ParallelRootGridIO=1`,
`ParallelParticleIO=1`, `CosmologySimulationCalculatePositions=1`),
particle IDs are assigned locally 0..N−1 per partitioned grid
(`Grid_NestedCosmologySimulationInitializeGrid.C:1586`) and offset in
hierarchy-traversal order (`NestedRecursivelySetParticleCount`).
**IDs therefore depend on the MPI partition layout at initialization:**

- same IC geometry + same core count at init → a DM-only run and a gas
  run assign identical IDs to the same DM particles (gas adds no
  particles at t=0; stars are appended later without renumbering);
- different core counts → the ID↔particle mapping differs, and raw ID
  reuse silently selects the wrong particles.

**Rule: ID list files are per-run generated, never copied between
runs.**  The bridge is Lagrangian position, which is one-to-one with ID
within any single run: planned utility
`multizoom/translate_particle_ids.py` (~50 lines, yt/scipy) takes the
cheap DM run's halo IDs, looks up their z≈z_init positions in that
run's first output, nearest-neighbor matches them (tolerance ≪ fine
cell) against the target gas run's first output, and emits the target
run's own IDs.  Reuses the cross-output matching pattern of
`lagrangian_regions.get_centers_and_extents`.

### Validation (when implemented)

1. Tag ~1000 inner particles of an existing single-zoom halo; run with
   the live box alongside its known `halo_track`; compare box centers
   over time (agreement within the box slack).
2. DM→gas ID translation round trip: translate, then verify the matched
   particles' z=0 positions cluster on the same halo in the gas run.
3. Restart mid-run and confirm the box is bit-identically re-derived.


---

## Part IV — Architecture update: integration with the ics_refactor pipeline (2026-08-26)

The plan above was written against the legacy `script512.py` /
`enzo-mrp-music` workflow on `master`.  Production has since moved to the
**`ics_refactor`** branch, whose `foggie/initial_conditions/pipeline/` package
(65 commits ahead of master) builds and runs the dwarf fleet.  Multizoom was
rebased onto it.

### What the pipeline did and did not replace

`pipeline/` is an **orchestration** layer: the halo registry, the `Box`
definitions, staging, submission, QC, ledger and reporting.  It did *not*
replace the workhorse — `build.py:render_mrp_config()` still writes a
`halo<id>_DM_<n-1>to<n>.conf` and `BuildScript.sh` still runs
`enzo-mrp-music.py <conf> <level>` to trace the Lagrangian region and drive
MUSIC.

That is precisely the layer multizoom extends, so the overlap was smaller than
it first appeared:

* **Dropped** — `multizoom512.py`.  `pipeline/build.py` already provides
  `halo_center_and_radius`, `read_shifts`, `center_for_level`,
  `render_enzo_param`, `render_runscript` and `read_grid_parameters`.
* **Kept** — the forked workhorse modules, the merge tool, the Enzo patches
  and the Milestone 5 design; none of these have a pipeline equivalent.

### The seam

`multizoom/pipeline_integration.py` is additive: it *reads* from `pipeline/`
and modifies nothing there, so a multizoom build cannot disturb the running
fleet.  It provides group discovery (an optional `multizoom_group` column in
the registry), an N-halo config renderer that reuses the pipeline's own
per-halo centring, and a `build` entry point that calls `multizoom.mrp_music`
in place of `enzo-mrp-music.py`.

### Ported from ics_refactor into the multizoom fork

`ics_refactor` had improved `enzo-mrp-music.py` after the fork was taken:

* `new_ics_directory` — read the parent level from the shared ICs directory
  while depositing this level's ICs elsewhere (multizoom writes into the
  group directory);
* `music_ld_library_path` — replaces the hard-coded Pleiades library path;
* `trim_lagrangian_outliers()` — drops far-flung strays that balloon the
  convex hull (halo39829 produced a 20.7-million-cell zoom from six points
  8.6 Mpc/h off the cloud).  In multizoom this runs **per halo**, before the
  clouds are unioned or handed to per-halo MUSIC runs, so one halo's strays
  cannot inflate the whole domain.

### Fleet survey (the real target)

Using each enabled halo's existing finest-patch extents, un-shifted into a
common frame via its own `Domain shifted by` log line: of the 120 pairs among
the 16 enabled dwarfs, **119 are cleanly disjoint**.  One pair overlaps
(52675 & 51741, by 0.098 Mpc/h) and three more are close but well above the
4-parent-cell threshold.  The 46615/47314/51741/52675 neighbourhood is the
single cluster that needs a union sub-run; the rest of the fleet merges
directly.

### Milestone 2 status: DONE

The five Enzo patches are built and validated on Aitken.  The N=1 regression
(`runs/n1_regression`) ran identical nested ICs (256^3 + L1 + L2, MRP mode 3)
through the patched binary and a pristine baseline built from the same commit:
byte-identical `RD0000.hierarchy`, 192 grids each, 16,787,107 particles each,
and all 216 HDF5 datasets bit-identical.  The patches are inert for a single
zoom.
