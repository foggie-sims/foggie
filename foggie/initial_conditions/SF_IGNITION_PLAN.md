# Igniting star formation in the smallest halos

Companion to `SF_THRESHOLD_PLAN.md`. That document asked *what sets* the
threshold; this one asks *what we have to change to move it down*, and in what
order. Everything below rests on measurements already in hand (2026-09-08),
listed first so the plan can be argued with.

---

## 1. What is measured

### 1.1 The gate

In every fleet deck both optional gates in `star_maker_h2reg.F` are disabled
(`H2StarMakerNumberDensityThreshold = 0`,
`H2StarMakerMinimumH2FractionForStarFormation = 0`). The only test the code
applies to a leaf cell at the finest level that is a local density maximum is
the minimum-mass one:

```
tau    = max(t_ff, StarMakerMinimumDynamicalTime = 1 Myr, dt_root)
m_form = min(0.9, H2StarMakerEfficiency * f_H2 * dt_root/tau) * m_cell
star  <=>  m_form >= StarMakerMinimumMass = 10 Msun
```

with `t_ff = 2.10e14 s * sqrt(1e-22/rho)`. Define the readiness

```
R = m_form / StarMakerMinimumMass          ignition <=> R >= 1
```

Where `t_ff` sets `tau` this collapses to

```
R  ~  f_H2 * n_H^(3/2) * dx^3
```

because `t_ff ~ n_H^(-1/2)` and `m_cell ~ n_H * dx^3`.

### 1.2 Where the dark halos actually stand (norad arm, no background at all)

| halo | log M200c(z=6) | R peak | short by | f_H2 at the R cell | n_H at the R cell | Z of dense gas |
|---|---|---|---|---|---|---|
| 174526 | 7.32 | 0.019 | 52x | 2.0e-3 | 39 | floor |
| 543386 | 7.55 | 0.043 | 23x | 2.8e-3 | 109 | floor |
| 27158  | 7.23 | 0.050 | 20x | 3.3e-3 | 37 | floor |
| 74411  | 7.41 | 0.067 | 15x | 3.0e-3 | 48 | floor |
| 489215 | 7.69 | 0.077 | 13x | 1.9e-3 | 586 | floor |
| 537545 | 7.93 | 0.092 | 11x | 4.4e-3 | 208 | floor |
| 439991 | 7.81 | 0.158 | 6x  | 4.1e-3 | 68 | floor |
| 491413 | 7.74 | 0.449 | 2x  | 3.7e-3 | 771 | 1.1e-5 |
| 331302 | 7.93 | 0.484 | 2x  | 3.4e-3 | 720 | 5.0e-5 |
| 21246  | 7.97 | 0.739 | 1x  | 7.5e-3 | 500 | 1.1e-4 |

"floor" = the code's pristine metallicity value, 7.7e-9 Zsun.

Three facts follow, and they define the whole problem:

1. **`f_H2` is pinned at 2-4e-3 in pristine gas, independent of density.** Every
   dark halo sits there. Every igniter is at 0.03-0.28 and is enriched. The
   metals -> dust -> H2 bootstrap never starts.
2. **Removing the UV background entirely buys only 1-10x in R.** The shortfall
   is 11-52x. Reionization is a second barrier (it removes the gas), not the
   first.
3. **The gate is resolution-dependent**: `R ~ dx^3` at fixed `n_H` and `f_H2`.
   `StarMakerMinimumMass` is a fixed physical mass tested against a cell mass
   that is not. This is the single most important caveat on any result below.

### 1.3 The root-grid star formation catastrophe

Measured on `halo543386/25Mpc_DM_512-L3-gas/RD0099` (z = 0):

```
354,592 star particles in the box, 2.68e10 Msun total
  99.7% sit on the ROOT GRID (level 0) in gas at median n_H = 0.0025 cm^-3
  100%  sit in gas with n_H < 1 cm^-3
   84%  sit in gas with n_H < 0.01 cm^-3
```

The cause is arithmetic, not physics: a root cell is 48.8 ckpc/h across, so at
z = 0 it holds ~2.8e10 Msun of gas even at n_H = 2.5e-3. With `m_cell` that
large the gate `m_form >= 10 Msun` is met at `f_H2 ~ 1e-5`, which Grackle
supplies in warm diffuse filament gas. **The fleet is manufacturing 2.7e10 Msun
of stars per run in the IGM**, roughly a hundred times the stellar mass of all
the zoom halos combined, with the associated feedback energy and metals.

The science is protected -- every analysis counts stars inside R200c -- but

* it is a large spurious term in any cosmic SFH, metal budget, or IGM thermal
  history computed from these boxes, and
* **it must be fixed before stochastic star formation is switched on**, because
  stochastic sampling extends star formation to exactly the sub-threshold tail
  where these cells live.

This is the correct diagnosis of the "coarse-leaf-cell star formation" note in
the campaign memory: the cells are not dense, they are *big*.

### 1.4 The stochastic branch is dead code

`star_maker_h2reg.F`, in the minimum-mass block:

```fortran
      if (starmass_in_Msun .lt. StarFormationMinimumMass) then
         if (StochasticStarFormation .eq. 1) then
            call random_number(random)
            critval = starmass_in_Msun / StarFormationMinimumMass
            if (random .gt. critval) goto 10
            starmass_in_Msun = StarFormationMinimumMass
            starmass = starmass_in_Msun / m1
            ...
         endif
c        JT added goto here 112623
         goto 10                        <-- fires even on a successful draw
      endif
```

The unconditional `goto 10` sits after the stochastic block's `endif`, so a
cell that *wins* the lottery is still skipped. `H2StarMakerStochastic = 1` is
currently a no-op. It needs a one-line fix before it can be tested at all.

### 1.5 `H2FloorInColdGas` is unreachable in our configuration

The floor is applied only inside the `H2Method .eq. 0` branch. We run
`H2StarMakerH2FractionMethod = 1` (read Grackle's `H2I + H2II`), so
`H2StarMakerH2FloorInColdGas` does nothing today. Note also that the
`H2Method = 0` analytic model (Krumholz-McKee-Tumlinson) returns **exactly
zero** at zero metallicity -- `tau_c = 0.067 * Z_MW * Sigma -> 0`, so
`s -> inf` and `f_H2 -> 0`. Switching to method 0 without a floor would make
things strictly worse.

---

## 2. The levers, classified honestly

Three classes, and the distinction matters for what we can claim in the paper.

**Class A -- repair a resolution artifact.** The `>= 10 Msun` threshold is not
physics; it is a discretization of a continuous star formation rate. Stochastic
sampling is the standard, defensible fix and makes the expected SFR
resolution-independent. A density threshold likewise removes cells that only
qualify because they are large.

**Class B -- real physics we are currently missing.** HD cooling, a
Pop III relic metallicity floor, H2 formation on dust at low Z. These change
what the gas actually does.

**Class C -- turning the crank.** Lowering `StarMakerMinimumMass`, raising
`H2StarMakerEfficiency`, dropping `StarMakerMinimumDynamicalTime`. These move
the threshold but are pure parameter choice; they must be reported as such and
their effect is degenerate with resolution.

The plan runs A first, because if A alone ignites the dark halos then B and C
are calibration rather than necessity.

---

## 3. Testbed

Full runs are too expensive and disk is the binding constraint (450 TB quota,
~48 TB headroom on 2026-09-08).

* **Branch, do not restart.** Fork each norad run from its existing `RD0007`
  (z = 12.5) dump and run to z = 4. That covers the whole ignition epoch --
  every fleet halo that lights does so at z >= 6.6 -- at roughly a third the
  cost of a full run.
* **Four halos**, chosen to span the shortfall:
  * `174526` (52x short) -- the hardest case
  * `543386` (23x short) -- typical dark halo
  * `537545` (11x short) -- easiest dark halo
  * `235659` (already ignites in norad, R = 2.07) -- **control**: no change may
    turn this into a runaway.
* **Short output list**: 20 RDs from z = 12.5 to z = 4, no DDs. ~35 GB/dump
  measured, so ~0.7 TB per run.
* Tooling: `make_reion_run.py` already clones a run and rewrites the deck; add
  a `--branch-from <dump>` mode rather than starting from ICs.

Disk budget: Tier 0 is 3 configs x 4 halos = 12 runs = ~8 TB. Do not queue
Tier 1 until Tier 0 has reported and its runs are deleted or thinned.

---

## 4. The plan

### Tier 0 -- repair (Class A). Run this first; it may be sufficient.

| id | change | rationale | prediction |
|---|---|---|---|
| **T0.0** | Verify the root-grid census at z = 10, 8, 6, not just z = 0 | the artifact must be shown to be present during the ignition epoch, not only late | root-grid stars dominate at all z |
| **T0.1** | **Code fix**: make the `goto 10` conditional so a winning stochastic draw creates a particle | `H2StarMakerStochastic = 1` is a no-op today | bit-identical with `Stochastic = 0`; gate on that |
| **T0.2** | `H2StarMakerNumberDensityThreshold = 1` (proper cm^-3) | kills root-grid star formation, which lives at n_H < 0.01; the dark halo cores reach n_H = 37-770, so it costs them nothing | box star mass falls by ~99%; R200c star counts unchanged |
| **T0.3** | T0.1 + T0.2 + `H2StarMakerStochastic = 1` | the actual experiment | **all seven dark halos ignite** |

The T0.3 prediction is quantitative. Summing `R * (dt_dump/dt_root)` over the
measured norad histories -- the expected number of successful draws for a
single eligible cell -- gives:

| halo | R median | root steps | expected N* |
|---|---|---|---|
| 174526 | 8.7e-4 | 3358 | 7 |
| 27158 | 3.6e-3 | 6548 | 62 |
| 543386 | 1.9e-2 | 5861 | 93 |
| 74411 | 2.2e-2 | 5965 | 126 |
| 489215 | 3.3e-2 | 5873 | 173 |
| 439991 | 5.7e-3 | 5397 | 218 |
| 537545 | 4.1e-2 | 6322 | 258 |
| 491413 | 0.107 | 6402 | 778 |
| 331302 | 0.127 | 6421 | 987 |
| 21246 | 0.067 | 4205 | 671 |
| 235659 | 0.073 | 5545 | 998 |

70-2600 Msun of stars in the previously dark halos: ultrafaint territory, which
is the science target. If T0.3 reproduces this table the mechanism is settled
and the threshold was a discretization artifact all along. If it produces far
*more*, feedback or the local-density-max condition is doing something the
single-cell estimate misses; if far less, the eligible cell count or the
local-maximum requirement is the limiter, and T1.4 becomes important.

**Gates for Tier 0**: (a) `Stochastic = 0` must stay bit-identical after the
patch; (b) 235659 must not gain more than ~2x its norad stellar mass;
(c) star particles must land at n_H > 1 by construction -- verify.

### Tier 1 -- crank the existing knobs (Class C), only if Tier 0 under-delivers

One knob per run, on 543386 (typical) and 174526 (hardest) only.

| id | change | factor on R | cost |
|---|---|---|---|
| T1.1 | `StarMakerMinimumMass` 10 -> 1 | x10 | worsens the small-particle metal cap (particles < 75 Msun already under-produce metals ~6x); feedback granularity |
| T1.2 | `H2StarMakerEfficiency` 0.02 -> 0.1 | x5 | changes the SFR normalization everywhere, including the bright halos |
| T1.3 | `StarMakerMinimumDynamicalTime` 1e6 -> 1e5 yr | only where t_ff < 1 Myr, i.e. n_H > 4e3 | none of the dark halos reach that; expect ~no effect. Run it to prove the null |
| T1.4 | `H2StarMakerUseLocalDensityMax` 1 -> 0 | more eligible cells, not a higher R | tells us whether the single-cell estimate above is the limiter |

Any two of T1.1/T1.2 close the 11-52x gap on their own. That is exactly why
they are Class C: they can produce the answer we want for no physical reason.

### Tier 2 -- physics that raises f_H2 or n_H (Class B)

| id | change | mechanism | expectation |
|---|---|---|---|
| **T2.1** | `MultiSpecies` 2 -> 3 (Grackle `primordial_chemistry = 3`) | adds D, D+, HD. HD cooling works below the ~200 K H2 floor and lets pristine gas reach ~50-100 K and higher density | `R ~ n_H^(3/2)`, so a 2x density gain is 2.8x in R. Cheap and physically clean; run it early |
| **T2.2** | Pre-enrichment floor, `Z = 1e-4` then `1e-3` Zsun | Pop III relic enrichment. Turns on the dust channel and the KMT term `tau_c ~ Z_MW * Sigma`; this is the bootstrap the dark halos never start | should reproduce the igniters' f_H2 ~ 3e-2. **The key Class B test**: if a plausible relic floor ignites them, the real answer is "these halos need Pop III enrichment we do not model" |
| T2.3 | `H2FormationOnDust` behavior at low Z; `dust_chemistry = 1`, `use_dust_density_field` | only matters once T2.2 supplies metals | pair with T2.2, not alone |
| T2.4 | Code fix: apply `H2FloorInColdGas` in the `H2Method = 1` branch as `max(f_H2_grackle, floor)` for T < `ColdGasTemperature` | makes the existing parameter reachable without abandoning Grackle chemistry | a floor of 0.05 is 15x over pristine -- closes the gap by itself. Class C in disguise; report it as a knob, not a discovery |

### Tier 3 -- resolution, because `R ~ dx^3` (must be done regardless)

| id | change | why |
|---|---|---|
| T3.1 | `CellFlaggingMethod` add 6 (Jeans length) to the current `2 4 8` | mass-based refinement alone may not drive collapsing pristine gas to level 9; Jeans refinement raises the `n_H` reached at fixed max level |
| T3.2 | `MaximumRefinementLevel` 9 -> 10 -> 11 | the direct test of the `dx^3` scaling. Each level divides `m_cell` by 8 at fixed density, so `n_H` must rise by 4x per level for R to hold. Whether it does is the whole question |
| T3.3 | L4 DM rung of the nref9 sweep | the L2 rung was deleted 2026-09-08; the sweep is L3 vs L4 only. Whatever ignition recipe wins must be shown to survive a change of DM resolution |

**Tier 3 is not optional.** Any threshold we report is a threshold at nref9. If
T3.2 shows the threshold moving with resolution, the headline result of the
paper has to be stated as a resolution-dependent one, and the recipe chosen in
Tiers 0-2 must be the one that is *least* resolution-sensitive. Stochastic
sampling (T0.3) is expected to be the most robust on exactly this ground, which
is the strongest argument for running it first.

---

## 5. Measurement protocol

Identical for every run, so results are comparable:

1. `halocat/scripts/sf_gate_history.py --halo H --arm <config>` -- per-dump R,
   f_H2, n_H, Z, the reservoirs, and N* inside R200c.
2. `plot_sf_onset.py` -- the per-halo panel with all configs overlaid on R = 1.
3. Box-wide star census at 3 redshifts (the T0.0 script) to confirm the
   root-grid population stays suppressed.
4. Ledger row per run: config, binary SHA, N* in R200c, M* in R200c, box star
   mass, peak R, ignition redshift, verdict.

**Success** = the halo forms stars in R200c and keeps them to the end of the
branch. **Failure modes to watch**: runaway in the 235659 control; star
particles at n_H < 1; box star mass not falling after T0.2; M* exceeding the
abundance-matching expectation for the halo mass.

---

## 6. What "as far down as is reasonable" should mean

A stopping criterion agreed in advance, so the sweep does not become a search
for the parameters that light the smallest halo:

1. **The recipe must be one setting for the whole fleet.** No per-halo tuning.
2. **The bright halos must not change.** 47314, 1703, 24122, 42784, 52675,
   15659 already ignite; their M* must stay within a factor of ~2.
3. **The star-forming gas must be self-gravitating.** After T0.2 every star
   forms at n_H > 1; tighten the threshold until the forming cells are also
   Jeans-unstable rather than merely dense.
4. **The result must survive T3.2.** A threshold that moves by more than ~0.3
   dex between nref9 and nref10 is a resolution result, not a physical one, and
   must be reported that way.
5. **Stop at the halo mass where the gas is no longer resolved** -- when the
   R-setting cell is a single cell with no neighbors above the density
   threshold, the collapse is unresolved and the star formation is a numerical
   statement about `dx`, not about the halo.

---

## 7. Order of operations

```
T0.0 census   ->  T0.1 patch + bit-identical gate  ->  T0.2 density threshold
                                                          |
                                       T0.3 stochastic (12 runs, ~8 TB)
                                                          |
                          ignites all seven? --yes--> T2.1 HD, T2.2 pre-enrichment
                                    |                 (is it physics or sampling?)
                                    no                          |
                                    |                       T3.1-T3.3 resolution
                          T1.1-T1.4 + T2.4 knobs                |
                                    |                        report
                                    +-------------------------/
```

Tier 3 runs in parallel with whatever else is queued, because it gates the
interpretation of everything.

---

## 8. Open questions this plan does not answer

* Does the spurious root-grid star formation affect the measured IGM thermal
  history in `igm_temperature.json`? 2.7e10 Msun of stars is a real feedback
  budget. The filtering-mass comparison rests on that curve.
* Is `StarFormationOncePerRootGridTimeStep = 1` handing the maker a root step
  even when it fires on level 9? The `dt_root/tau` factor assumes so and the
  measured `dt` values are consistent, but it has not been proved from the
  code path.
* Feedback response: none of the above accounts for a first generation of stars
  suppressing the second. The dark halos have never had feedback at all, so
  their first burst may be self-limiting in a way the R estimate cannot see.
