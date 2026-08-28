# Multizoom session memory

Written 2026-08-27 for whoever picks this up next -- another agent or a human
returning cold.  `HANDOFF.md` is the short version; this is the full account,
including the reasoning behind decisions and the mistakes worth not repeating.

Companion documents in this directory: `AUDIT_AND_PLAN.md` (the original code
audit and staged plan, plus Part III/IV added later), `README.md` (usage),
`CLAUDE.md` (working rules loaded automatically when editing here).

---

## 1. What this project is

FOGGIE runs cosmological zoom simulations: a 25 Mpc/h box (512^3 root grid)
with one halo refined to high resolution.  Enzo and MUSIC were both built
around **one zoom region per domain**.  This work makes **N zoom regions in
one domain** work, so several dwarfs can be simulated together.

Two motivations, in the order they emerged:

1. **Cost.**  Every standalone zoom pays the full 512^3 root grid again.  Six
   halos in one box pay it once.
2. **Load balance (the bigger one).**  The user's production audits attribute
   50-60% of gas-run time to communication waits.  Measurement here found the
   mechanism: deep AMR levels hold **fewer grids than MPI ranks**, so most
   ranks sit in level barriers.  A single deep zoom cannot produce enough
   grids to fill the machine; N zooms can.  The user's framing: *"putting
   particle and cell compute where we currently just have MPI wait time will
   be more efficient than single zooms."*

---

## 2. Where everything lives

| what | where |
|---|---|
| workspace root | `/nobackupnfs1/jtumlins/foggie-multizoom` (git worktree, branch `multizoom`) |
| this package | `foggie/initial_conditions/multizoom/` |
| patched Enzo source | `enzo-multizoom/` (branch `multizoom`, built) |
| pristine baseline Enzo | `enzo-baseline/` (worktree at unpatched `edf858ca`) |
| patched MUSIC | `scratch/music-shiftpatch/` (region_shift_override) |
| runs and plots | `runs/` (git-excluded; ~1 TB) |
| production tree (READ ONLY) | `/nobackupnfs1/jtumlins/25Mpc_new_cosmology` |
| production checkout | `/nobackupnfs1/jtumlins/foggie` (branch `ics_refactor`) -- do not disturb |

Environment: `source env_multizoom.sh` before building or running anything.
It loads comp-intel/2020.4.304 + hdf5/1.8.18_serial, MPICH 4.0.3, and
`/u/jtumlins/installs/compat_gfortran3` (grackle links `libgfortran.so.3`
while the system now ships `.so.5`).

### Published branches

* `foggie-sims/foggie` -> **`multizoom`** (based on `ics_refactor`, 32 commits,
  purely additive under `initial_conditions/multizoom/`).
* `foggie-sims/enzo-foggie` -> **`multizoom`** (based on `main` at `edf858ca`,
  6 patches + the Aitken machine file).

**Pushing:** the SSH remote resolves to a read-only deploy key.  Push over
HTTPS; a credential helper (`~/.foggie_bench_git_creds`) supplies auth.  In
this worktree `remote.origin.pushurl` is already set to the HTTPS URL.

---

## 3. How it works

Two modes, both driven from the ics_refactor pipeline:

* **union** -- ONE MUSIC run whose region covers the union of the halos'
  Lagrangian clouds; the RefinementMask is re-deposited per cloud so only the
  halos carry must-refine particles, not the space between them.  Needs no
  code changes anywhere.  Costs wasted intermediate-resolution volume, and the
  union patch grows fast, so it suits 2-3 nearby halos only.
* **merge** -- one MUSIC run per halo with **identical seeds** and one shared
  domain frame, merged into a single multi-patch IC set (several nested grids
  per level).  This is the production path.  Needs the Enzo patches.

The physics that makes merge valid: with `kspace_TF = yes` (the FOGGIE
default) the density cascade is strictly top-down, so same-seed same-frame
runs share one realization.  Verified on real data -- see section 5.

### Integration with the pipeline

`pipeline/` (on `ics_refactor`) owns orchestration: registry, `Box`
definitions, staging, submission, QC.  It still drives `enzo-mrp-music` as the
workhorse.  Multizoom slots in at exactly that seam: it renders one config
naming N halos instead of one, and hands it to `multizoom.mrp_music`.
**Nothing in `pipeline/` is modified**, so a multizoom build cannot disturb
the production fleet.

Group membership: an optional `multizoom_group` column in the halo registry,
or `--halos 1,2,3` on the command line (ad hoc, no registry edit).  Either
way every member must be an enabled registry row sharing one parent box.

### Modules

| file | role |
|---|---|
| `config.py` | `[halo:<id>]` multi-halo config parsing |
| `lagrangian_regions.py` | trace N halos to their z_init clouds in ONE grid pass; outlier trimming |
| `refinement_mask.py` | deposit N clouds into a MUSIC RefinementMask |
| `mrp_music.py` | per-level orchestration (union and merge) |
| `merge_music_ics.py` | merge N same-seed MUSIC runs into one multi-patch IC set |
| `gas_group.py` | gas ICs from a group's existing DM configs (baryons on) |
| `pipeline_integration.py` | group resolution, config rendering, run assembly; the CLI |
| `group_qc.py` | run the pipeline's QC figures on a group (proxy `Box`) |
| `loadbalance.py` | per-dump work distribution; per-cycle cost/imbalance from `performance.out` |
| `compare_costs.py` | standalone-vs-multizoom cost report + figure |
| `validation.py` | union-mode mask checks |
| `advance_group.sh` | idempotent ladder driver (build -> assemble -> submit -> next level) |

39 tests: `cd foggie/initial_conditions/multizoom && python3 -m pytest`
(needs only numpy/h5py/pytest; `pytest.ini` here is load-bearing -- running
from the repo root pulls in the top-level foggie package, which imports yt).

---

## 4. Enzo patches (branch `multizoom` of enzo-foggie)

Six, all off `edf858ca`.  Also exported as `enzo_patches/000{1..6}-*.patch`.

1. **Parent search** -- require containment in ALL dimensions, stop at first
   match.  `ParentGrid` was assigned once dim 0 passed and never cleared.
2. **`MAX_INITIAL_GRIDS` 10 -> 64.**  Count is `1 + N_halos x N_levels`; 64
   allows 21 halos to L3, 15 to L4.  Only the two cosmology initializers were
   raised; three other problem types still say 10 deliberately.
3. **Partition walk** -- snapshot the initial-grid list recursing BOTH
   hierarchy links, then partition each.  The old walk skipped any grid with a
   sibling.
4. **`ReInitialize`** -- map hierarchy grids to initial grids by same-level
   edge containment instead of assuming grid number == level.
5. **`Grid_CosmologyInitializeParticles.C`**: `break` -> `continue` in the
   static-region exclusion loop.  One non-overlapping region ended the scan,
   leaving **duplicate coarse particles** under later patches.  Latent bug,
   affects single zooms too.
6. **MRP creation on the deepest initial grid LEVEL** (new global
   `CosmologySimulationMaximumInitialLevel`).  The condition was
   `level == NumberOfInitialGrids-1`, a level no grid has in a merged set --
   the first ten-grid run initialized with **zero** must-refine particles.

Patches 1-4 are inert for single-pyramid runs; 5 and 6 are real bug fixes.
**N=1 regression run twice** (before and after patch 6): identical nested ICs
through patched and pristine binaries give a byte-identical `RD0000.hierarchy`,
the same 16,787,107 particles, and all 216 HDF5 datasets bit-identical.

Patches 1-4 came from the static audit; **5 and 6 were found by running**.

### MUSIC

One optional patch, `music_patches/0001-region-shift-override.patch`, adding
`setup/region_shift_override = sx,sy,sz`: one fixed shift shared by every run
of a group, taking precedence over the auto-shift and `no_shift`.  Needed
because under `no_shift` a Lagrangian region near the periodic boundary makes
MUSIC fail (`mesh.hh:1503-1512`).  Built at `scratch/music-shiftpatch/`
(the tipsy plugins are excluded -- they need removed SunRPC headers).

**Computing the shift:** find the widest *uncovered* run of cells on each
axis and put the boundary there.  A naive "largest gap between intervals"
heuristic degenerates when intervals wrap and will silently leave a halo
straddling the edge.

---

## 5. Measured results (all from real runs, not estimates)

### Sixpack: 6 halos (48014, 56672, 75392, 21246, 24122, 42502), DM L1-L3 to z=0

| level | 6 standalone | multizoom | ratio |
|---|---|---|---|
| L1 | 334.6 core-h | 69.6 | 0.21 |
| L2 | 441.4 | 111.7 | 0.25 |
| L3 | 775.4 | 443.3 | 0.57 |
| total | 1551.5 | **624.5** | **0.40** |

Whole ladder cost 12.3 node-hours ~= 34 SBUs.  Load balance BETTER than
standalone at every level (L3 levels 5-7: 1.04-1.06 vs 1.16-1.35).

**Cost decomposes into fixed + marginal:** root grid 17.8/18.0/19.6 core-h at
L1/L2/L3 (N-independent), marginal per halo 8.6/15.6/70.6 vs standalone
55.8/73.6/129.2.  The marginal halo is already cheaper than a standalone one.
Yield/SBU saturates fast at L3 (1.7x at N=6, 1.8x at N=40) because the root is
only ~4% there; at L1 it keeps climbing (4.8x -> 6.1x).

**Shared-timestep tax** (multizoom cell-updates / sum of standalone, refined
levels): pairA 2 halos ~1.4x, sixpack 6 halos 1.49x (L1), 1.50x (L2), 1.20x
(L3).  **It does not grow from 2 to 6 halos** -- each level's dt is set by the
single most demanding grid, so the penalty is fixed by the worst-case halo,
not by how many companions it has.  Design implication: exclude outliers, not
diversity; never mix target levels in one group.

### pairA: union mode, 2 halos (15659 + 48014) to z=0

Halo positions within 4 and 16 kpc/h of standalone, masses within 1% and 7%,
identical contamination.  59.3 vs 119.4 core-h.  Validated union mode end to
end.  Note its level-1 update tax was **20x** -- the union patch covers the
bounding volume of both clouds.  Strong argument for merge over union.

### Rank starvation (the load-balance mechanism)

Rank-time lost purely because a level has fewer grids than ranks, second half
of each run:

| run | ranks | lost |
|---|---|---|
| standalone GAS halo48014 L2 | 128 | **47.2%** |
| standalone GAS halo31260 L2 | 128 | **36.0%** |
| standalone DM halo48014 L3 | 64 | 1.9% |
| multizoom sixpack L3 (DM) | 64 | **0.0%** |

In the gas runs **level 7 alone is 43% of wallclock with ~30 grids on 128
ranks** (23% occupancy).  DM never starves (hundreds of small grids per
level), which is why DM multizoom gains came from the root grid, not
occupancy.  This is the quantitative match to the user's 50-60% audit figure.

### DM->gas particle IDs are NOT portable

Measured on production data: for ~16,800 shared IDs between halo48014's DM and
gas runs, the position each ID points to differs by a **median of 10.9 Mpc/h**
-- random pairs in a 25 Mpc box.  Cause: with `ParallelRootGridIO` +
`ParallelParticleIO`, IDs are assigned per partitioned grid then offset in
hierarchy order, so the mapping depends on the **MPI decomposition at
initialization**; the fleet runs DM at 64 ranks and gas at 128.

The bridge works: matching by z=99 position in a 100 kpc/h cube gave nearest
neighbours at 0.049 base cells median, second-nearest at 0.92 cells (19x
margin), **80/80 unique matches, 0/80 with equal IDs**.  So any DM->gas
particle tracking must translate through Lagrangian position.  This affects
existing FOGGIE analysis, not just multizoom.

### MultiRefineRegion (Enzo's multi-box forced refinement), tested as-is

Two boxes on halo46615 + companion 47330, restarting at z=0.5:

* Forced refinement **works unpatched**: level-6 cells 3.24M -> 40.3M, and the
  added coverage (1.054e-6) matches two 200 kpc/h boxes (1.024e-6) to 3%.
* The **latch bug** (`LocalMin/MaxRefinementLevel` never reset per cell) is
  only a ~3% surface effect here, because method 20 is gated off below
  `MustRefineParticlesRefineToLevel` -- exactly where grids are large enough
  for the latch to be catastrophic.
* The **max-level cap is inert AND harmful**: setting `MaximumLevel=4` on a
  halo already at 7 changed nothing at level 7 but *increased* levels 5-6,
  because a non-zero ceiling opens the flagging gate and the latch carries it.

So: uniform ceilings work today; **per-halo ceilings need patches 7-9**
(reset the latch per cell; actually unflag above the ceiling; decide whether
method 20 should be exempt from the MRP AND-clause).

---

## 6. Currently running (as of 2026-08-27 ~18:00 PDT)

### `mz-gaspair-L2-gas` (PBS 25049434) -- the open question

Two-halo **gas** multizoom, 48014 + 42502, at `runs/multizoom_gaspair/25Mpc_DM_512-L2-gas`.
128 ranks, 24 h wall, no self-resubmission (bounded ~67 SBUs).  nref7 to match
the fleet's existing L2-gas runs exactly (a **one-off** `--max-refine-level 7`;
the pipeline was not modified).  Last seen z=4, cycle 550, ~5 h in.

**Purpose:** does filling idle ranks with a second zoom's work convert the
36-47% starvation into throughput?  Interim at z=7->5: multizoom 4123 s vs
8957 s for the two standalone runs -- **2.2x faster** for 2.3x fewer
cell-updates, with deep-level grid counts almost exactly the sum of the two
standalone runs.  But idle was 16.3% vs 11.3/14.0% -- at that early epoch the
standalone runs are not yet starved, so the gain so far is root-grid sharing
plus fewer root steps, NOT starvation recovery.  The decisive window is z~3-2.

Compare with `scratch/gas_starvation.py <zhi> <zlo>` (redshift-matched).

### `mz-tenpack-L1` (PBS 25050969) -- the production build

Ten halos for resolution tests and physics sweeps, from
`25Mpc_new_cosmology/multizoom_halo_ids.txt`:
48014, 42784, 1703, 79186, 31208, 15659, 82682, 42502, 59186, 15494
(1.1e9 to 1.2e10 Msun/h, log-spaced; all ten have standalone L1/L2 runs as
baselines).

Pre-flight: all **45 pairs disjoint**; common shift **(-44, 227, 176)** leaves
every region contiguous with 2.23 Mpc/h clearance; 21 grids at L2, well inside
64.  59186 carries `rvir_min=400` so its region is much larger (4.15 Mpc/h).

L1 ICs merged (11 grids) and the run is evolving.  The driver advances
automatically: L1 -> z=0, L2 build, L2 -> z=0.

**Then the gas stage, which is what the user asked for and is NOT yet set up:**

```sh
python3 .../multizoom/gas_group.py --group tenpack --level 2 \
    --halos 48014,42784,1703,79186,31208,15659,82682,42502,59186,15494 \
    --registry runs/halo_registry_tenpack.ecsv
python3 .../multizoom/pipeline_integration.py assemble --group tenpack --level 2 \
    --phase gas --max-refine-level 9 --registry runs/halo_registry_tenpack.ecsv
```

**nref9 is required** and was verified: diffing the pipeline's rendered gas
parameter file against `halo42784`'s standalone L2-gas run shows the ONLY
differences are the refinement levels and grid geometry -- star formation,
feedback, cooling, Grackle and H2 settings are identical.  The five newer
halos already run nref9 standalone.

Driver command that resumes the DM ladder if needed:

```sh
runs/advance_group.sh --group tenpack \
  --halos 48014,42784,1703,79186,31208,15659,82682,42502,59186,15494 \
  --shift "-44, 227, 176" --max-level 2 \
  --registry runs/halo_registry_tenpack.ecsv --build-walltime 8:00:00
```

---

## 7. Traps that cost real time here

* **`performance.out` cadence.**  The gas template sets `TimingCycleSkip = 10`;
  older fleet runs use 1.  The skip changes how OFTEN a block is written, not
  what it measures -- Enzo's timers accumulate, so summing blocks still gives
  true elapsed time (verified: 42 blocks at skip=10 summed to 2.62 h against
  2.92 h of job walltime).  Rescaling by the skip would claim 26 h.  What it
  DOES change is the resolution of a cycle window.
* **Epoch matching.**  Runs use different output lists, so dump names are not
  comparable; and exact-equality boundaries fail on values like 4.9999993.
  Interpolate z(cycle) -- see `scratch/gas_starvation.py`.
* **The nested patch is NOT where the halo is at z=0.**  It encloses the
  Lagrangian (z=99) volume, which drifts ~1.3 Mpc comoving.  Use
  `qc.center_in_run` (note `shift_divisor` is 511, not 512).
* **MUSIC refuses an existing output directory** and aborts.  Worse, the
  aborted run first REWRITES the log, so a valid IC set can afterwards parse
  as unshifted.  The merge tool now takes the shift from
  `region_shift_override` in the config, which cannot be clobbered.
* **Clear a PBS log before resubmitting**, or a monitor watching for `exit=`
  fires instantly on the previous failure.
* **Overriding PBS walltime** must also rewrite `SIMRUN_WALL`, or simrun.pl
  believes it has the box's full walltime and never checkpoints.
* **The mask deposit wrapped only negative positions.**  Under a shift
  override a cloud can exceed 1.0; four of six sixpack masks came out empty
  and the run initialized with MRPs for only two halos.  Fixed with a
  two-sided wrap plus an empty-mask guard in the merge tool.

---

## 8. Open items

1. **The gas verdict** (running).  Everything else is bookkeeping by
   comparison; this is the question the investigation exists to answer.
2. **Tenpack L2 gas at nref9** -- set up once L2 DM finishes (commands above).
3. **`Box.gas_refine_offset = 6`** means any NEW L2-gas run gets nref8 while
   the fleet's existing L2-gas runs used nref7.  Production drift, not
   introduced here; worth a decision.
4. **42502 contamination** 1.32% in multizoom L2 QC vs 0.76% standalone -- the
   one halo where multizoom looks worse.  Probably the exact particle mask
   sitting tighter than the fleet's convex hull; a larger `rvir_min` is the
   fleet's usual fix.
5. **Milestone 5** (`AUDIT_AND_PLAN.md` Part III): particle-tracked forced
   refinement -- boxes centred at runtime on per-halo particle-ID sets instead
   of precomputed `halo_track` files, removing the chicken-and-egg for a new
   group.  Designed, not built.  Needs the MultiRefineRegion fixes above, plus
   `translate_particle_ids.py` (the DM->gas bridge, which section 5 shows is
   mandatory for any ID reuse).
6. **Scaling test** -- 10-12 halos to find where the timestep tax overtakes the
   occupancy gain.  The tenpack run is the natural vehicle.
