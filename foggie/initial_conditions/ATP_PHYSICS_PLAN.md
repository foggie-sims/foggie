# Implementing the ATP 2023 physics in Enzo

## Context

The ATP 2023 proposal *"Galaxy Evolution at the Edge"*
(`foggie/initial_conditions/ATP2023-DRAFT-v3-June28.pdf`) commits to three specific Enzo
developments, listed in §5.2 and scheduled for Year 1 in the Figure 6 timeline:

- **§5.2.1 Small star particles** — a hybrid scheme tracking individual high-mass stars as single
  particles while aggregating low-mass stars, with the division at ≈ 8 M☉ where α-element and Fe
  yields begin to matter, plus a prescriptive delayed double-degenerate Type Ia model for the
  aggregated population.
- **§5.2.2 Chemical evolution** — enrichment tracked separately by *process*: SNII, SNIa, AGB,
  kilonovae / r-process, and re-enrichment from stellar winds and planetary nebulae, plus a total
  metallicity field for metal-dependent cooling, built on the JINA-NuGrid pipeline and SYGMA.
- **§5.2.3 "Multi-refine" tracking** — enforcing high resolution on *many* moving dwarf galaxies
  inside one zoom, combining pre-computed tracks with bound-particle tracers.

These exist to serve the science: resolving the gas that receives supernova ejecta finely enough
that metal mixing is not artificially efficient (§3.5, Figure 3), and reaching ≲ 30 pc in dwarf
ISM with 50–200 pc CGM in nested spheres out to 25–50 kpc (§4.2), across 250–500 dwarfs rather
than one.

This plan covers **the Enzo physics only**. The IC pipeline work is complete and documented
separately on the `ics_refactor` branch.

**Decisions taken with the user:** base the work on `enzo-foggie`; implement small star particles
on the **ActiveParticle** framework; **repair `CellFlaggingMethod = 20` first**, then add
radius-graded refinement.

---

## Immediate action: nothing is implemented yet

> **SUPERSEDED 2026-08-27** — the review happened, the team decided to proceed, and
> multizoom was invented in the course of that work. See "Status addendum" below for
> what is actually built. This section is kept as the record of the decision point.

**No code is written until the team has reviewed this plan.** The only work to do now is to bank
it for review:

1. Write `foggie/initial_conditions/ATP_PHYSICS_PLAN.md` — this document, verbatim.
2. Write `foggie/initial_conditions/atp_physics_roadmap.html` — a self-contained review page.
3. Commit both to the `ics_refactor` branch and push.

No files in any `enzo-*` tree are touched. No branch is cut in the Enzo repo. No PR is opened.
Everything below §"Base fork" describes work to begin **after** review, not now.

---

## Status addendum — 2026-08-27

**This supersedes "Immediate action: nothing is implemented yet" above.** The team reviewed the
plan, decided to proceed, and in the course of that work **multizoom was invented** — a capability
that did not exist when this plan was written and that changes how §5.2.3 is delivered. It is
recorded as Component D below.

### Where the components actually stand

| | plan said (2026-08-01) | actual (2026-08-27) |
|---|---|---|
| A chemical evolution | machinery exists, half unreachable | **further along than recorded** — see D1 |
| B small star particles | greenfield | unchanged, not started |
| C multi-refine | one blocking bug | **C0-C4 still not started**; the multizoom patches are a *different* six |
| D multizoom | did not exist | built, measured, merged |

### Branch consolidation

Our production line had **diverged from `enzo-foggie/main`** — 48 commits ours, 46 theirs, sharing
only `f6255482`. Consolidated on 2026-08-27 into **`atp-dwarfs-dev`**: `fcf-on-cassi` (Cassi
mechanical feedback, cic_deposit, PPM NaN) + the six performance-audit picks + upstream `main` +
the six multizoom patches. One conflict (`star_feedback6.F`), resolved by dropping the SN-energy
validation that main had relocated into `Grid_StarParticleHandler.C` and keeping the T1.6 work.
On the FOGGIE side, `ics_refactor` now carries the multizoom package (purely additive).

### Discrepancies to carry forward

**D1. §5.2.2 is further along than this plan records.** All four per-process metal fields exist in
the consolidated tree — `MetalSNIIDensity`, `MetalSNIaDensity`, `MetalAGBDensity`,
`MetalNSMDensity` — and upstream **PR #68** added chem yield-table interpolation routines and the
table-reading infrastructure while we were on a divergent branch. Today's merge is the first time
our line has had them. **The "half unreachable" audit in §"What already exists" predates this and
should be re-run against the merged tree before any A-series work is scoped.**

**D2. The method-20 ceiling is worse than "not enforced".** This plan says the per-box maximum is
computed and ignored. Measured 2026-08-26 on halo46615 + companion: setting `MaximumLevel = 4` on
a halo already at level 7 changed nothing at 7 but **increased levels 5-6** — a non-zero ceiling
opens the flagging gate and the un-reset latch carries it, so a cap *adds* refinement. **C2 is a
correctness fix, not an enhancement.**

**D3. The latch is bounded, and this plan overstates it.** The "What already exists" section says
"the whole grid refines". That holds only for a grid that STRADDLES a region boundary, and it does
not generalise: `grid::FlagCellsToBeRefinedByMultiRefineRegion` is a per-grid method whose
`LocalMaximumRefinementLevel` / `LocalMinimumRefinementLevel` are declared with initialisers inside
the function (lines 43-44), so they are re-initialised on every call and **the latch does not
persist across grids**. A grid wholly inside a region is unaffected; a grid wholly outside keeps
both at 0 and flags nothing. Only cells following the first in-region cell in i/j/k order, on a
boundary-straddling grid, inherit the level.

That is a surface, not a volume -- which is exactly what the multizoom work measured on
halo46615 + companion: a ~3% surface effect, with forced refinement otherwise working unpatched
(level-6 cells 3.24M -> 40.3M, added coverage matching two 200 kpc/h boxes to 3%). The
`MustRefineParticlesRefineToLevel` gate at `Grid_SetFlaggingField.C:61` suppresses it further at
coarse levels, but that is second-order on a bug that is already bounded.

**Consequence for sequencing: C1 is a real bug but LOW severity and not blocking.** It costs a
shell of unnecessary refinement around region boundaries, not correctness of the science, which is
why multizoom runs correctly today on uniform floors. **C2 is the priority**, because the
unenforced ceiling is what blocks per-halo control (D-d) and particle-tracked refinement (D-e).
C1 is a three-line change that can ride along with it.

**D3b. C0 as written in Component C cannot fail.** It asks that cells outside every region "must
not exceed the outer level", but `MultiRefineRegionMaximumOuterLevel` is only read and defaulted
(`Grid_FlagCellsToBeRefinedByMultiRefineRegion.C:51-52`) and never used to unflag anything -- no
maximum is enforced anywhere in that routine. Setting it low and asserting on it would fail for
C2's reason, not C1's, conflating the two bugs the sequence exists to separate. **Recommended
instead: a differential test** -- run the same short problem twice, identical but for whether
`CellFlaggingMethod` includes 20, and assert the grid structure outside every region is unchanged.
That isolates method 20, needs no absolute level to compare against, cannot be satisfied by
over-refinement, and separates C1 from C2. Pair it with a small static region carrying a high
minimum level in an otherwise empty corner, so the latch has something to latch onto wherever the
halo sits.

**That differential already exists — build on it, do not rebuild it.** Found 2026-08-27 at
`foggie-multizoom/runs/multirefine_test/`, a three-arm experiment on halo46615 (1.3e10) plus
companion 47330 (2.6e9), 425 ckpc/h apart and both wholly in the fine species, restarted at
z=0.5 for 5 cycles (`run_multirefine_test.sh`). Config lives in the RD0012 restart files, not
in a `.enzo` deck, which is why a deck-level grep for method 20 does not find it:

| arm | `CellFlaggingMethod` | regions | tests |
|-----|----------------------|---------|-------|
| `ctrl` | `4 8` | none | the differential's control arm |
| `test` | `4 8 20` | 2 boxes, `MinimumLevel = 6` | the floor |
| `cap`  | `4 8 20` | 2 boxes, `MinimumLevel = 0` | the ceiling |

`ctrl` vs `test` is exactly the recommended differential. This is also the sole source of D2's
and D3's measurements. Note `MultiRefineRegionMaximumOuterLevel = -99999` (undefined) in all
three arms, which is the direct confirmation of this entry's point: the outer level C0 asserts
on was never even set, let alone enforced.

What it does NOT cover, and what an extension should add: live evolving tracks (all three arms
use static boxes), multiple job legs (5 cycles inside one leg, so D5's restart duplication is
untouched), and N > 2 regions.

**D3c. RETRACTED — the tenpack does not test multirefine.** This entry previously claimed the
ten-halo run was "the empirical check" for the method-20 defects, exercising "45 disjoint region
pairs". That is wrong, and wrong in a way that mattered: it credited work in flight with
validating code it never enters. Verified 2026-08-27 across every multizoom deck — sixpack
(L1/L2/L3), tenpack, pairA, gaspair, and the N=1 regression decks — **all run
`CellFlaggingMethod = 4 8` with `MustRefineParticles`, and contain zero `MultiRefineRegion`
parameters.** Nor does `halo21140-L3-gas` (`CFM = 2 4 8`), whose two-region track was built but
never wired into a deck.

Method 20 has been exercised exactly once, and not by any of the above: a dedicated **two-box
restart experiment on halo46615 + companion 47330 at z=0.5**, which is the sole source of both
measurements this plan relies on — the ~3% surface effect in D3 and the inert-and-harmful ceiling
in D2. So C1/C2 are not unmeasured. They are **measured once, in a targeted restart, and never
under production load**: never on live evolving tracks, never across multiple job legs, never at
N > 2 boxes.

The consequence for sequencing is that the sixpack's success is **no evidence at all** about
C1/C2 — different code path, never entered — and the evidence that does exist comes from a single
short experiment. Extending it needs a run whose only job is to exercise method 20 on live
disjoint tracks over several legs; the halo21140/21151 pair is the intended vehicle and is already
most of the way built.

**D7. Multizoom and multirefine are different mechanisms, and this plan conflates them.**
The distinction is structural, not a matter of degree, and `global_data.h` settles it:

```
MustRefineRegionLeftEdge[MAX_DIMENSION]                       <- ONE box, no region index
EvolveMustRefineRegionLeftEdge[MAX_REFINE_REGIONS][3]         <- the SAME box, TIME bins
MultiRefineRegionLeftEdge[MAX_STATIC_REGIONS+MAX_TRACKS][3]   <- N boxes
MultiRefineRegionMinimumLevel[] / MaximumLevel[]              <- per-region floor and ceiling
```

- **Multizoom** (Component D): N MustRefineParticles *zoom regions* in one domain. IC-side,
  `CellFlaggingMethod 4`, N tagged particle sets. Not in the ATP proposal — invented during this
  campaign. **Working**; the sixpack is real evidence.
- **Multirefine** (Component C): N *forced boxes* via method 20, independent of how many zoom
  regions exist. In the ATP proposal from the start (§5.2.3). **Unexercised.**

Today's forced production runs (`halo15659-L9c-L7f`, `halo80181-L9c-L7f`) use
`ReadEvolveRefineFile` with a `halo_track` file: **one** moving box, evolving through time bins.
Not method 20.

**The composition gap this exposes.** A ten-halo multizoom run today can force-refine **exactly
one** of its ten halos, because the evolving-track mechanism holds a single box. Giving each halo
its own forced box *requires* method 20. So the ATP production target — N halos, each with an
nref8f forced box, sharing one domain — is multizoom **plus** multirefine, and multirefine is the
unfinished half. Component C is therefore on the critical path to Component D reaching
production, not a parallel track that can lag it.


**D4. §5.2.3's "bound-particle tracers" have an unstated prerequisite.** DM->gas particle IDs are
**not portable**: for ~16,800 shared IDs between halo48014's DM and gas runs the position each ID
points to differs by a **median of 10.9 Mpc/h**, because IDs are assigned per partitioned grid and
offset in hierarchy order, so the mapping depends on the MPI decomposition at initialization (the
fleet runs DM at 64 ranks and gas at 128). Matching by z=99 position works (0.049 vs 0.92 base
cells for first vs second neighbour, 80/80 unique). **Any tracer scheme reusing IDs across phases
needs that translation bridge first.** This affects existing FOGGIE analysis, not just multizoom.

**D5. Restart duplication threatens live-track work now, not later.** C4 was scheduled as a
routine repair, but `WriteParameterFile.C:694-733` writing evolving tracks as static means any
restart duplicates every track and leaves frozen copies. The fleet's runs restart constantly
between legs, so **a live-track demonstration is only trustworthy within a single job leg until
C4 lands.**

**D6. Confirmed as written:** `Grid_SetFlaggingFieldMultiRefineRegions.C` is still absent from
`Make.config.objects` with no callers (verified 2026-08-27), so it remains a source of logic to
lift for C2/C5, not a component to enable.

---

## Component D — Multizoom (N zoom regions in one domain)

Not in the ATP proposal by name, but it is now the delivery vehicle for its §4.2 ambition of
**250-500 dwarfs** rather than one. Enzo and MUSIC were both built around one zoom region per
domain; this makes N work.

**Why it matters more than cost.** Production gas runs lose **36-47% of rank-time** purely to deep
levels holding fewer grids than MPI ranks — level 7 alone is 43% of wallclock at ~30 grids on 128
ranks. A single deep zoom cannot fill the machine; N zooms can. That is the measured mechanism
behind the 50-60% communication-wait figure from the production audits. DM never starves (1.9%),
which is why DM multizoom gains come from sharing the root grid instead.

**Measured, not estimated** (six halos, DM L1-L3 to z=0): **0.40x** the cost of six standalone
runs, with better load balance at every level. Cost separates into a fixed root grid
(17.8/18.0/19.6 core-h, N-independent) and a marginal halo (8.6/15.6/70.6 vs standalone
55.8/73.6/129.2). The **shared-timestep tax does not grow with N** — 1.4x at two halos, 1.49/1.50/
1.20x at six — because each level's dt is set by the single worst grid. Design consequence:
**group by excluding outliers, not by restricting mass diversity, and never mix target refinement
levels in one group.**

**Relationship to Component C.** They are complementary, not alternatives, and they are *different
code paths* — see D7, which corrects an earlier conflation of the two. C makes *many forced boxes*
behave correctly (method 20); D makes *many zoom regions* share one domain (MustRefineParticles).
D works today and C is unexercised, so it is tempting to sequence C late. That is backwards:
**a multizoom run using today's single-box mechanism can force-refine only one of its N halos**,
so forced production at N halos is blocked on C. Both are needed for the proposal's dwarf counts.

- **D-a. Patches 1-6 — done**, merged into `atp-dwarfs-dev`. Two are real bug fixes, not multizoom
  scaffolding: duplicate coarse particles under later-scanned static regions (latent in single
  zooms too), and must-refine creation on a grid level that no merged set has.
- **D-b. The gas verdict — running.** Does filling idle ranks with a second zoom convert
  starvation into throughput? Interim at z=7->5 was 2.2x faster, but from root sharing rather than
  occupancy; the decisive window is z~3-2.
- **D-c. Tenpack scaling test — running** (IC build stage as of 2026-08-27, not yet evolving).
  Ten halos spanning the ELVES-to-SAGA mass range, to find where the timestep tax overtakes the
  occupancy gain. Scope note per D3c: it runs `CFM = 4 8` and so tests **multizoom only** — it
  says nothing about method 20.
- **D-d. Per-halo ceilings — blocked on C2.** Note the blocking is specific: with every region at
  the same target level, per-region ceiling equals global `MaximumRefinementLevel` and the
  unenforced ceiling is a **no-op**, so the homogeneous N-halo forced run is not blocked on C2.
  It is *heterogeneous* resolution — halo A at nref9 beside halo B at nref7 — that C2 gates.
- **D-e. Particle-tracked forced refinement** (AUDIT_AND_PLAN.md Part III, Milestone 5): boxes
  centred at runtime on per-halo particle-ID sets instead of precomputed tracks, removing the
  chicken-and-egg for a new group. Designed, not built; needs C1-C2 and the D4 bridge.

### Sequencing change

Multizoom slots into the existing order without disturbing it, because D-a is already merged and
D-b/D-c are measurements rather than code. The one hard edge is that **C1-C2 now gate D-d and
D-e**, which raises their priority: the multi-refine repairs are no longer just a §5.2.3
deliverable, they are what unblocks per-halo control in the multizoom production path.

---

---

## Component E — Science-grade forced ultra-faints via multizoom

**Stated by JT, 2026-08-29.** A goal alongside the fleet, not a replacement for
it: take the ultra-faints to science-grade forced-refinement runs, in order to
study **H2 star formation and feedback in the smallest halos** with FOGGIE-style
forced boxes, delivered by **multizoom** so that a domain carries N ultra-faints
rather than one.

### Sequence

1. **Entire fleet to complete L3 DM.** In progress; most halos are there or
   close, and the ten-halo L3 set for the tenpack comparison is complete.
2. **A selection up to L4.** L4 is demonstrated: halo80181 built and ran it on
   2026-08-29 -- 251,071 particles traced, five nested grids, MUSIC at
   levelmax 13, 77 GB of ICs, ~96 GB written by z = 3.8. Build on a compute
   node; the login node aborts without a diagnostic.
3. **Up the gas refinement ladder** on that selection.
4. **Forced boxes on each**, FOGGIE-style.
5. **Grouped by multizoom.**

### The dependency this creates: Component C becomes a science blocker

Steps 4 and 5 together **require multirefine, `CellFlaggingMethod = 20`**.
Per D7, multizoom and multirefine are different code paths, and today's forced
runs use `ReadEvolveRefineFile` with a single moving `halo_track` box --
`MustRefineRegionLeftEdge[MAX_DIMENSION]`, one box, no region index. So **a
multizoom run can force-refine exactly one of its N halos.** Giving every
ultra-faint in a group its own forced box has no path that avoids method 20.

Component C therefore moves from "efficiency work that can lag" to **on the
critical path for a stated science goal**. Its state is unchanged and unhappy:

- **C1**, the per-grid latch: bounded, a ~3% surface effect on
  boundary-straddling grids, low severity (D3).
- **C2**, the ceiling that is never enforced: the blocker for per-halo control
  (D-d), and now for this.
- Method 20 has been exercised **once** -- the halo46615+47330 two-box restart
  at z = 0.5 (D3b) -- never on live evolving tracks, never across job legs,
  never at N > 2. Every one of those is required here.

### Second dependency: track construction

A forced box follows a track file, and **a track built from an incomplete
parent run freezes and loses the galaxy**. halo80181's track carries evolving
rows only to z = 0.208 and is padded thereafter; its halo left the 143 kpc box
by 357 kpc and spent the final 2.6 Gyr outside its own forced region
(\S\ref{sec:trackcaveat} of the paper draft). Tracks must be built from
completed parents. At N ultra-faints per domain this failure is N-fold and
correspondingly easier to miss -- a per-halo track/halo offset check belongs in
the group QC before any group forced run is trusted.

### Why the ultra-faints specifically

They are where the H2 prescription is least constrained and most
resolution-sensitive. halo80181 forms 2.31e4 Msun of stars forced against
1.25e3 unforced, a factor of 18, and its star particles run 5.7-101 Msun with a
median of 19 -- approaching individual-star masses, where the assumption that a
star particle is a well-sampled simple stellar population fails outright. Three
fleet halos form no stars at all without forcing; whether they light up with it
is the experiment now running.

### Order of work implied

- C2 first, then C1 (they ride together).
- A method-20 test on live evolving tracks across multiple job legs at N > 2 --
  the halo21140/21151 disjoint pair already has a track built for exactly this
  and has never been wired into a deck (D3b).
- Then a small forced multizoom group, before committing the ultra-faint set.


## What already exists

The single most important finding: **two of the three components are substantially built.** The
work is far more repair-and-extend than greenfield, and the plan is sized accordingly.

### Chemical evolution — the machinery is there, half of it is unreachable

`ReadFeedbackTable.C` already reads a SYGMA/NuGrid HDF5 table and `tabular_feedback.F` (1111
lines) already interpolates it bilinearly in (metallicity, age), with correct trapezoidal
integration across table nodes. The table is already **3-D in source**:

```c
// typedefs.h:337-342
const enum_type TabSN2 = 0, TabSN1a = 1, TabAGB = 2, TabNSM = 3;
```

`/sygma_models/ejecta_mass` and `ejecta_metal_mass` are `n_met × n_age × 4`. A second
Starburst99 table (`ReadPreSNFeedbackTable.C`) supplies pre-SN winds.

But:

- **`agb_mass`, `agb_metal`, `nsm_mass`, `nsm_metal` are defined and never called.** Confirmed:
  `tabular_feedback.F:538,577,617,656`, zero call sites in the tree. AGB and r-process enrichment
  are computed by SYGMA, shipped in the table, read into memory, and then dropped.
- **`MetalAGBDensity` (104) and `MetalNSMDensity` (105) fields exist** in `typedefs.h` and are
  registered in the cosmology initialiser, **but are missing from the fraction↔density conversion**
  in `Grid_StarParticleHandler.C:730` and `:2269`, which lists only `MetalSNIaDensity` and
  `MetalSNIIDensity`. Enabling them today would silently corrupt those fields.

So §5.2.2 is largely a matter of *connecting what is already there*, then adding the fifth source
(winds/PNe) and the IMF-variation hooks.

### `anna-branch` is already merged — and carries more than the code

`origin/anna-branch` in `enzo-foggie` has **zero commits not already in `main`** (checked against a
ref fetched the same day; `main` is 38 commits ahead). There is nothing to cherry-pick, which
reinforces the choice of `enzo-foggie` as the base. Anna Wright's work landed through PRs #26,
#39, #48, #53 and #57 between Jul 2025 and Mar 2026.

Two of those commits matter more than the code itself:

```
9c6446d8  Documentation for parameters related to MultiRefineRegions and context aware star formation
dd57aca3  Adding MultiRefineRegion and context aware star formation tests to AMR nested cosmology
```

**The parameters are already documented** in the Enzo manual — `doc/manual/source/parameters/index.rst`
lines 1119-1170 (regions, track file format, overlap semantics) and 2328-2342 (star mass). New
parameters must be added there, and the existing entries are the style to follow.

**There is already a test problem**: `run/CosmologySimulation/amr_nested_cosmology/`, a
one-minute single-core zoom with MUSIC ICs and must-refine particles, with
`test_context_aware_star_formation_and_multirefinement` in `test_amr_nested_cosmology.py`. This is
the harness to extend, not to replace.

### "Context-aware star formation" already delivers part of §5.2.1

`MultiRefineRegionSpatiallyVaryingStarMass` plus `MultiRefineRegionMinimumStarMass[#]` let the
minimum star particle mass vary per region, so dwarfs inside a refined region form *smaller* star
particles than the global `StarMakerMinimumMass`. Implemented in `Grid_SetMinimumStarMass.C`,
documented, and tested — the existing test asserts that small particles form and that all of them
fall inside the evolving region.

This is a real partial delivery of "small star particles". What remains from §5.2.1 is the
*hybrid* scheme: individual stars above ≈ 8 M☉ as single particles, which is a different thing
from a smaller aggregate mass and is where Component B's work actually lies.

### Multi-refine — exists, usable for uniform floors, with two bugs above that

> **Heading corrected 2026-08-27.** This read "with one bug that makes it unusable". Both halves
> were wrong: there are two defects (C1 latch, C2 ceiling), and neither makes the mechanism
> unusable for the uniform-floor case that production needs first. See **D3** for the latch's
> real (bounded, ~3%) scope, **D2** for the ceiling, and **D7** for why this is a different
> mechanism from multizoom (Component D) rather than a variant of it.

`CellFlaggingMethod = 20` (`MultiRefineRegion`, Anna Wright, Dec 2023) already provides up to
`MAX_TRACKS = 20` independently moving boxes read from one track file, each with its own minimum
level, maximum level and minimum star mass, linearly interpolated in time
(`SetEvolveRefineRegion.C:309-421`, `ReadEvolveRefineFile.C:248-401`). `Grid_SetMinimumStarMass.C`
already lowers `StarMakerMinimumMass` inside regions.

The blocking defect, confirmed by reading `Grid_FlagCellsToBeRefinedByMultiRefineRegion.C`:

```c
int LocalMaximumRefinementLevel = 0;   // line 43 -- FUNCTION scope
int LocalMinimumRefinementLevel = 0;   // line 44
...
    if (LocalMinimumRefinementLevel < MultiRefineRegionMinimumLevel[region])
        LocalMinimumRefinementLevel = MultiRefineRegionMinimumLevel[region];   // only ever raised
```

They are never reset per cell — `NRegions = 0` at the top of the cell loop is, but these two are
not. Once any single cell falls inside a region, **every subsequent cell in the i/j/k traversal
inherits that level and is flagged** — the whole grid refines. This is a scoping fix, not a
redesign.

**The existing test cannot catch this.** `test_context_aware_star_formation_and_multirefinement`
asserts only lower bounds:

```python
assert(np.all(MRR1['index','grid_level'] >= llim[timestep]))   # inside the evolving region
assert(np.all(MRR2['index','grid_level'] >= 2))                # inside the static region
```

Over-refinement satisfies both. Nothing anywhere asserts that cells *outside* every region are
left alone, so the bug passes a green test suite. Adding that upper-bound assertion is the first
thing to do, and it should be seen to fail before the fix and pass after.

Three further defects in the same routine and its supporting code:

- `LocalMaximumRefinementLevel` is computed and used only in a `> 0` test — **the per-box maximum
  level is read from the track file and never enforced.**
- The region loop runs `MAX_STATIC_REGIONS + NIter` = **1000+N iterations per cell**, with no
  grid-level bounding-box prefilter (method 12 has one; method 20 does not).
- `WriteParameterFile.C:694-733` writes evolving tracks out as if they were static regions, so
  **each restart duplicates every track** and leaves stale frozen copies behind.

### Radius-graded refinement — a working prototype that was never compiled

`Grid_SetFlaggingFieldMultiRefineRegions.C` (Elizabeth Tasker, 2013) contains exactly the
"resolution floor that relaxes with radius" ramp the proposal asks for, with correct
flag/unflag/leave-alone tri-state semantics — better designed than the built method 20. It is
**not in `Make.config.objects`, has no callers, and references six globals that are declared
nowhere**, so it would not compile. It is a source of logic to lift, not a component to enable.

### Small star particles — the genuine greenfield

The legacy particle-attribute route is effectively closed: `MAX_NUMBER_OF_PARTICLE_ATTRIBUTES` is
a compile-time 4 (7 with `WINDS`) and is already full, HDF5 attribute names are hard-coded in two
files that must stay in sync, and `ParticleInitialMass` had to be special-cased as a separate
array touching roughly 30 files.

The **ActiveParticle** framework is the alternative and is already in the tree.
`ParticleAttributeHandler.h` provides templated `Handler<Class, type, &Class::member>` objects
registered into a static vector, off which HDF5 IO, MPI packing and grid transfer are driven
generically. `ActiveParticle_Skeleton.C:51` is the minimal example. Nine AP types already exist,
including `PopIII` and `SmartStar`.

`enzo-foggie` is also already ahead here: `NUM_PARTICLE_TYPES = 12` (vs 11 in the production
tree) and it carries `Grid_TestMultiStarParticleInitializeGrid.C` and
`Grid_ReturnNumberOfParticlesOfThisType.C`, which the production tree lacks — somebody has begun
this work.

---

## Base fork

Work proceeds on a branch cut from **`enzo-foggie`** (`4ceb0007`, merge of PR #65), not from
`enzo-foggie-aitken-mpich` which production currently runs. The production tree is behind on
exactly the particle-side changes this work needs.

Port the Aitken/Milan/MPICH machine file (the only substantive difference, per
`c0f036d0`) onto the new branch, build, and point the IC pipeline at it by changing one line —
`enzo_exe` in `foggie/initial_conditions/pipeline/config.py`. The pipeline's box config exists
precisely so the binary is not baked into committed run scripts.

---

## Component A — Chemical evolution (§5.2.2)

Sequenced first: it is the cheapest, it is the most nearly complete, and it produces the fields
that Components B and C both write into.

**A1. Connect AGB and NSM.** Call the existing `agb_mass`/`agb_metal`/`nsm_mass`/`nsm_metal` from
`star_feedback6.F` alongside the SNII/SNIa calls, depositing into `MetalAGBDensity` and
`MetalNSMDensity`. Add both field types to the fraction↔density conversion at
`Grid_StarParticleHandler.C:730` and `:2269` — **this must land in the same commit**, or the
fields are corrupted rather than merely absent.

**A2. Add the fifth source: winds and planetary nebulae.** The proposal names "re-enrichment from
stellar winds and planetary nebulae" as a tracked process. The Starburst99 pre-SN path
(`ReadPreSNFeedbackTable.C`, `pSN_mass`/`pSN_metal`/`pSN_mom`) already supplies wind mass and
metals but deposits them into the total metal field only. Add `MetalWindDensity` following the
`MetalSNIIDensity` recipe end to end: enum in `typedefs.h` before `FieldUndefined` (currently
114, bump it), registration in `Grid_CosmologySimulationInitializeGrid.C:293`, matching
`DataLabel` in `CosmologySimulationInitialize.C:802` **in the same order**, colour-field handling
in `Grid_CorrectForRefinedFluxes.C:407,856`, and the conversion sites above.

**A3. Generalise the yield interpolation over sources.** `tabular_feedback.F` hard-codes source
indices (`iSNII = 1`, `iSNIa = 2`, …) with the comment *"I'm not clever enough to auto sync these
with typedefs.h"*. Replace with a single source-index table shared with the C++ enum, so adding a
source is one edit rather than four. Keep the interpolation maths untouched.

**A4. Total metallicity for cooling.** Confirm the Grackle-facing metallicity field is the sum
over sources rather than one of them, and that adding sources does not double-count. This is the
field the proposal calls out for metal-dependent cooling.

**A5. Audit the table-edge behaviour.** `integrate_yields` has a dead branch
(`tabular_feedback.F:807`) intended to zero yields for stars older than the table; because indices
are clamped it can never trigger, so old stars keep emitting at the last table row's rate
indefinitely. Decide whether that is intended and make it explicit either way.

**A6. IMF variation.** The proposal wants to "experiment with stellar initial mass function
variations and different stellar evolution models". SYGMA generates the tables offline, so this is
a matter of generating a table set per IMF and selecting by parameter — add
`StarFeedbackTabularFilename` variants and document the SYGMA invocation used, so tables are
reproducible rather than mystery binaries.

---

## Component B — Small star particles (§5.2.1)

**B1. New ActiveParticle type `MassiveStar`.** Copy `ActiveParticle_Skeleton.C` as the template.
Attributes registered through `ParticleAttributeHandler`: birth time, initial mass, current mass,
metallicity, and the endpoint type (CCSN / failed / stripped). Individual stars above the
threshold (≈ 8 M☉, a parameter, not a literal) get one particle each.

Reuse rather than rewrite: `Star_AssignFinalMassFromIMF.C` and
`StarParticlePopIII_IMFInitialize.C` already sample an IMF; `Star.h` already carries `FinalMass`,
`LifeTime` and `HitEndpoint` semantics for single stars.

**B2. Sampling at formation.** In the H2-regulated star maker (`star_maker_h2reg.F`, dispatched at
`Grid_StarParticleHandler.C:1303`), when a star-forming cell produces mass, draw the massive end
of the IMF explicitly and create one `MassiveStar` per draw; put the residual low-mass mass into
an ordinary aggregated star particle on the existing path. The mass budget must close exactly —
this is the first thing to unit-test.

**B3. Per-star feedback.** Individual stars inject at their own lifetime endpoints rather than
through a population-averaged rate. Give them their own `STARFEED_METHOD` bit so
`star_feedback6.F`'s `type(n) .eq. 2` guard does not silently swallow them, and so the aggregated
and individual paths can coexist without double counting.

**B4. Delayed Type Ia from the aggregated population.** The proposal specifies a prescriptive
double-degenerate model. Note SNIa is already available two ways: baked into the SYGMA table's
`sne_event_rate[:,:,1]` (DTD folded in offline by SYGMA), and as a legacy analytic `t^-1.1` fit in
`feedback_formulae.F:84`. Prefer the tabular route for consistency with A3; add an explicit DTD
only if the SYGMA DTD proves unsuitable, and say which is in use in the parameter file.

**B5. Bookkeeping.** `Grid_ReadGrid.C:761,1005` range-check particle types against
`NUM_PARTICLE_TYPES`; AP types are handled separately but the interaction needs checking on
restart. Verify particle counts survive `CommunicationUpdateStarParticleCount` and
`StarParticleFinalize.C`.

---

## Component C — Multi-refine (§5.2.3)

**C0. Close the test's blind spot first.** Add an upper-bound assertion to
`test_context_aware_star_formation_and_multirefinement` — cells outside every region must not
exceed the outer level. Confirm it **fails** on the current code before touching anything. A fix
verified only by a test that could not have caught the bug is not verified.

**C1. Fix the per-cell scoping bug.** Move `LocalMaximumRefinementLevel` and
`LocalMinimumRefinementLevel` inside the per-cell loop in
`Grid_FlagCellsToBeRefinedByMultiRefineRegion.C`, beside the existing `NRegions = 0`. A few lines,
and it unblocks everything else. C0 must now pass.

**C2. Enforce the per-box maximum level.** Currently read from the track file and ignored. Lift
the flag / unflag / leave-alone tri-state from `Grid_SetFlaggingFieldMultiRefineRegions.C:243-266`
rather than inventing new semantics.

**C3. Prefilter and bound the region loop.** Add the grid-level bounding-box test that method 12
already has (`Grid_FlagCellsToBeRefinedByMustRefineRegion.C:81-88`), and iterate over
`NumberOfStaticMultiRefineRegions + NIter` rather than `MAX_STATIC_REGIONS + NIter`. At 1000
iterations per cell this is a real cost at production grid counts.

**C4. Fix the restart duplication.** `WriteParameterFile.C:694-733` must not write evolving tracks
as static regions. Verify by restarting twice and asserting the region count is unchanged.

**C5. Radius-graded refinement for nested CGM spheres.** The prototype is still on disk
(`Grid_SetFlaggingFieldMultiRefineRegions.C`, 9.7 kB) and still declared in `Grid.h:888`, but is
absent from `Make.config.objects` and has no callers — confirmed dead, not merely unused. Lift its
ramp and tri-state logic as a *sphere* geometry: declare the six missing globals, add to
`Make.config.objects`, add a call site in `FindSubgrids.C`. Target the proposal's profile —
~100 pc floor within Rvir relaxing to ~1 kpc at ≳ 6 Rvir.

**C6. Raise the track limits.** `MAX_TRACKS = 20` caps the design at 20 dwarfs; the proposal wants
hundreds. Raising it is a `#define`, but the arrays are statically sized
`[MAX_STATIC_REGIONS+MAX_TRACKS][MAX_TIME_ENTRIES][3]` — check BSS growth and consider dynamic
allocation before raising it far.

**C7. Per-track time-varying levels.** `EvolveMultiRefineRegion{Minimum,Maximum}Level` are
`[MAX_TRACKS]`, not `[MAX_TRACKS][MAX_TIME_ENTRIES]`, and are reassigned on every parsed row so
the last row silently wins. Make them 2-D so a dwarf's floor can tighten as it assembles.

**C8. Bound-particle tracers.** The proposal specifies tracks derived from "tracer particles,
defined to be dark matter particles known to lie fully bound to a particular dwarf". The hook
already exists: `Grid_MustRefineParticlesFlagFromList.C` reads a particle-ID list, though from a
hardcoded filename (`MustRefineParticlesFlaggingList.in`). Parameterise the filename and extend to
per-dwarf lists.

**C9. Cooling-length refinement.** The criterion the proposal describes already exists as
`CellFlaggingMethod = 7` (`Grid_FlagCellsToBeRefinedByCoolingTime.C`), but has **no level cap at
all**, which is why FOGGIE contains it with a region box. Add a maximum level and per-region
masking so it can be used safely inside dwarf regions.

**C10. Document new parameters in the Enzo manual.** `doc/manual/source/parameters/index.rst`
already documents the MultiRefineRegion family at lines 1119-1170 and 2328-2342, including the
track-file column layout and the overlap semantics. Any new parameter joins them there, in the
same style. Also fix the malformed `MultiRefineRegionMinimumStarMass[0] = 1.0e+7.0` in the test
parameter file — `sscanf` stops at the second decimal point and yields 1.0e+7 by luck.

**Track generation.** The tracks themselves come from the IC pipeline side: halo catalogs plus the
environment metrics already computed in `halo_catalogs_512/512/z0/halo_environment.ecsv`. Selecting
which dwarfs to multi-refine is exactly the isolation question that table answers.

---

## Sequencing

Ordered so each stage is verifiable before the next depends on it.

0. **Branch and build.** Cut from `enzo-foggie`, port the Aitken machine file, build, confirm it
   reproduces a current production run bit-for-bit before changing any physics.
1. **C1–C4: multi-refine repairs.** Small, high-value, independently testable. C1 in particular is
   a few lines that turn an unusable feature into a working one.
2. **A1–A2: connect AGB/NSM, add winds.** Cheap, and it exercises the field-addition path end to
   end before anything harder depends on it.
3. **A3–A5: generalise sources, audit edges.**
4. **B1–B3: MassiveStar ActiveParticle and per-star feedback.** The largest single piece.
5. **C5–C9: radius grading, tracers, cooling caps.**
6. **B4, A6: Type Ia prescription and IMF variants.**

---

## Verification

Physics changes are not verifiable by "it ran", so each stage gets a specific check.

**Baseline first.** Before any physics change, confirm the new build reproduces an existing run.
`halo42189` L1 is the natural reference — the IC pipeline has already established it matches its
hand-built counterpart bit-for-bit at RD0265, cycle 1208. Same ICs, same binary behaviour.

Extend `run/CosmologySimulation/amr_nested_cosmology/` rather than building a new harness. It runs
in about a minute on one core, already exercises MUSIC ICs, must-refine particles, evolving and
static regions, and context-aware star formation, and already has a comparison-data workflow.

```bash
# multi-refine: the assertion the existing test is missing (C0)
#   outside every region, grid_level must not exceed the outer level
#   must FAIL before C1 and PASS after -- if it passes before, the test is wrong

# restart duplication (C4)
#   run, write, restart, restart again; region count must be identical each time

# mass closure for small star particles (B2)
#   sum(MassiveStar initial masses) + aggregated particle mass == mass removed from the cell

# chemical evolution (A1/A2)
#   sum of per-source metal fields == total metal field, to round-off, every step
#   a single star particle in a uniform box: integrate ejecta over 13 Gyr and compare
#   against SYGMA's own totals for the same table -- this validates the whole
#   interpolate/integrate path independently of Enzo

# conservation
#   total metal mass conserved to round-off with feedback on and cooling off
```

For the multi-refine and chemistry work, a small non-cosmological test problem is far cheaper than
a zoom, and `Grid_TestMultiStarParticleInitializeGrid.C` in `enzo-foggie` suggests that pattern is
already in use.

**Science-level validation** once the pieces are in: the proposal's own targets — dwarf ISM at
≲ 30 pc, CGM 50–200 pc in nested spheres, gas resolution elements below 100 M☉ (Figure 3), and
a mass–metallicity relation that is not artificially flattened by over-mixing.

---

## Deliverables

Banked on the **`ics_refactor`** branch of the foggie repo, alongside `REFACTOR_PLAN.md` and
`refactor_roadmap.html`, in `foggie/initial_conditions/`:

- **`ATP_PHYSICS_PLAN.md`** — this plan verbatim, reviewable in place.
- **`atp_physics_roadmap.html`** — a graphical roadmap: the three components, what already exists
  versus what is new, the dependency order, and the verification gates. Fully self-contained —
  inline CSS and hand-written SVG, no CDN, no external fonts — since it will be opened over
  `file://` on a laptop. Same constraint as `refactor_roadmap.html`, and for the same reason.

Banking it here rather than in the Enzo repo keeps the planning documents together and avoids
touching `enzo-foggie` before any code is written. It moves when the work does.

---

## Risks and things to decide later

- **Two coexisting star-particle systems.** ActiveParticle massive stars alongside legacy
  aggregated particles means feedback must sum correctly across both and neither may double count.
  This is the main correctness risk in Component B, and the reason mass closure is the first test.
- **`MAX_TRACKS` and static arrays.** Hundreds of dwarfs will not fit the current statically-sized
  arrays; dynamic allocation is likely needed and touches restart.
- **Production divergence.** Work happens on a branch off `enzo-foggie` while production runs
  `enzo-foggie-aitken-mpich`. The longer that lasts the more painful the merge; porting the
  machine file early and keeping the branch rebased is cheaper than a big-bang merge.
- **SYGMA table provenance.** The current `sygma_feedback_table_1000.h5` has no recorded
  generation recipe. Before varying the IMF, record how tables are produced or the variants will
  not be comparable.
- **Flagged for the science team, not a coding decision:** the committed gas templates carry
  `StarMakerMinimumMass = 10000.` while the runs that completed used `10`. Small star particles
  make that parameter far more consequential.
