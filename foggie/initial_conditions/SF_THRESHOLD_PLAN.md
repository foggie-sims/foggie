# Why does star formation switch on at M200c ~ 2e9 Msun? A sweep plan

JT, 2026-09-05: "Hard to believe that the star formation threshold is so
sharp at M_DM = 2e9. We need to figure out what is driving that physically.
Make a plan to do this with resolution and physics parameter sweeps.
Consider all the physics Enzo has leverage over."

Everything in sections 1 and 2 is measured on the fleet as it stands
(20 L3-gas halos at z=0, `figures_z0/ignition_scan_L3z0.json`, the runs'
`starlog_*.txt`, and the binary's `star_maker_h2reg.F`). Sections 3 to 7
are the plan.

## 1. What the fleet already says

### 1a. The relation is bimodal, not a threshold with scatter

**Tier 0 / 0.1 done 2026-09-05** (`figures_z0/ignition_branches_L3.png`,
`.tsv`; `scripts/plot_ignition_branches.py`, `scripts/progenitor_mass.py`).
Progenitor masses are now Lagrangian: the AHF particle IDs of every object
inside the catalog's R200c at the anchor, matched to the AHF object holding
most of them at each earlier snapshot. The first version of this table used
a catalog-position walk and got 52675 and 1703 wrong (both "late igniters"
were hops onto neighbours); those numbers are superseded.

L3-gas halos with M200c(z=0) > 5e8, plus the running ones below z=4
(M* at their latest catalog, a lower limit):

| halo | M200c(z=0) | M* | N* | z_ign | M200c at z_ign | M200c at z=6 | branch |
|---|---|---|---|---|---|---|---|
| 21432 | 5.8e8 | 0 | 0 | never | - | 9.6e7 | dark |
| 170570 | 6.0e8 | 3.0e2 | 25 | 5.7 | 1.2e8 | 1.1e8 | dark |
| 57194 | 8.7e8 | 1.7e3 | 133 | 6.1 | 1.2e8 | 1.2e8 | dark |
| 24122 (z=3.5) | 1.0e9 | 1.2e7 | 18717 | 10.0 | 1.4e8 | 4.8e8 | bright |
| 48014 (z=0.6) | 1.3e9 | 5.3e3 | 343 | 8.0 | 8.5e7 | 1.1e8 | dark |
| 21246 | 1.4e9 | 2.7e2 | 21 | 4.7 | 2.8e8 | 1.0e8 | dark |
| 1703 (z=3.5) | 1.5e9 | 2.1e7 | 18850 | 10.0 | 1.0e8 | 2.4e8 | bright |
| 56672 | 1.7e9 | 2.0e6 | 10020 | 9.0 | 1.5e8 | 3.7e8 | bright, quenched |
| 42784 (z=3.6) | 1.8e9 | 2.4e7 | 33269 | 15.0 | 9.6e7 | 7.9e8 | bright |
| 52675 | 1.9e9 | 1.8e7 | 45953 | 10.0 | 1.5e8 | 6.6e8 | bright |
| 47314 | 2.1e9 | 5.5e5 | 7481 | 6.6 | 1.9e8 | 2.7e8 | bright, marginal |
| 15659 (z=3.9) | 2.2e9 | 2.3e7 | 28761 | 9.0 | 1.5e8 | 7.5e8 | bright |

(The eleven ultrafaints below 5e8 have M* <= 7e2 and M200c(z=6) <= 1e8.)

There is nothing between M* = 1.5e4 and 5e5 Msun. **The two branches
separate completely in M200c at z=6**: the dark branch tops out at 1.24e8
(57194) and the bright branch starts at 2.4e8 (1703), a factor-2 gap with
no halo in it, while in z=0 mass they overlap over 1.3-2.1e9 (48014 and
21246 dark; 24122, 1703, 56672, 47314 bright). Ignition mass does NOT
separate them: dark halos ignite too, at 5e7-2.8e8, but at z <= 8; every
bright halo except 47314 (z=6.6) ignited at z >= 9. So the boundary is
"was the progenitor above ~2e8 by z ~ 6-7", i.e. the reionization-epoch
mass, and 2e9 today is just where that maps to for a typical growth
history. 47314 sits on the boundary (2.7e8 at z=6, ignited z=6.6, 5.5e5 of
stars); 21246 crossed 2.8e8 only at z=4.7 and stayed dark.

### 1b. What separates the two branches in the star logs

At their own star-formation events (level-9 cells only):

| halo | outcome | fH2 at SF (median) | Z at SF (median, mass fraction) |
|---|---|---|---|
| 52675 | 1.8e7 Msun | 0.25 | 1.3e-3 |
| 56672 | 2.0e6, quenched | 0.03 | 4e-5 |
| 47314 | 5.5e5 | 0.017 | 5e-5 |
| 174526 | 3e3 | 0.007 | 6e-6 |
| 21246 | 2.7e2 | 0.006 | 2e-6 |

Pristine gas under the UVB sits at fH2 ~ 1e-2 (gas-phase H- route,
Sobolev-shielded). Above ~1e-4 in metals, H2 formation on dust
(`H2FormationOnDust = 1`, second leg) lifts fH2 by an order of magnitude
and star formation runs away. The bimodality is the metals -> dust -> H2
-> stars loop: whoever gets the first ~10^4 Msun of stars in before the
UVB heats the gas gets to keep forming them.

## 2. The gates in the prescription as actually run

`StarParticleCreation = 2048` (H2-regulated maker, `star_maker_h2reg.F`),
`H2StarMakerH2FractionMethod = 1` (fH2 from Grackle's 9-species network),
`StarFormationOncePerRootGridTimeStep = 1`. A cell forms a star particle
when ALL of:

1. it is a leaf cell (not covered by a finer grid) -- at ANY level;
2. it is a local maximum of HI density among its six neighbours
   (`H2StarMakerUseLocalDensityMax = 1`);
3. `H2StarMakerNumberDensityThreshold = 0`: no density threshold;
   `H2StarMakerMinimumH2FractionForStarFormation = 0`: no fH2 threshold;
4. the mass it would form,
   m_form = 0.02 * fH2 * m_cell * dt_root / max(t_ff, 1 Myr, dt_root),
   is at least `StarMakerMinimumMass = 10` Msun. `H2StarMakerStochastic = 0`,
   so below 10 Msun nothing forms and nothing accumulates. (The stochastic
   branch is dead code anyway: an unconditional `goto 10` added 2023-11-26
   skips particle creation after the random draw succeeds. If a sweep wants
   stochastic creation, that line has to come out first.)

Gate 4 is where resolution enters. m_cell at nref9 (136 pc comoving) for a
cell at n_H = 100 cm^-3 at z=6 is ~2.5e4 Msun, so a star needs
fH2 * dt_root/t_ff >= 0.02. With fH2 pinned near 1e-2 in pristine gas that
is marginal: the dark halos' few events all sit at 20-45 Msun, just above
the floor. A coarser cell holds more mass and passes the gate more easily;
a finer one holds less. So the location of the threshold has a built-in
dependence on cell size and on the 10 Msun floor that has nothing to do
with the halo. The nref8 fleet (L2-gas) never igniting below 2e9 while
nref9 does is the first data point on this: coarser cells but lower peak
densities, less shielding and lower fH2.

Two more numerical facts that belong in the plan:

- **The maker runs on every leaf cell in the box.** `EvolveLevel.C` sets
  `MakeStars` on all grids at the top level, and the maker's "finest level"
  check is only "no subgrid here". In halo52675's z=0 dump, 349,000 of the
  452,000 star particles sit on the root grid (49 kpc/h cells, fH2 ~ 3e-5,
  ~300 Msun each, all > 1.8 Mpc/h from the target), and 15,600 more sit in
  level-3 cells (6 kpc/h) at 1.2-1.9 Mpc/h. They cannot touch the target's
  own star formation (the halo's cells are level 8-9), but they enrich the
  zoom region's IGM at ~1e-6 (the 2 Mpc metal-extent pass must exclude
  them), and 350,000 particles run the tabular feedback every root step.
  The fix is either `H2StarMakerNumberDensityThreshold` > 0 (a few cm^-3
  excludes every coarse cell) or a `level == MaximumRefinementLevel` check
  in the handler. This is an enzo-foggie-wide behaviour, not ours alone.
- **The two-leg switch at z=15** turns on, at once, H2 formation on dust,
  self-shielding (method 3), H2 self-shielding and the shielded UVB table.
  Above z=15 the HM2012 table has no UVB at all, so the first leg is a
  pristine, unshielded, no-dust, no-UVB universe; ignition at z >= 12.5 in
  nine of the running halos happened there. The switch redshift is a knob
  we have never varied.

## 3. Tier 0 -- analysis on the runs we have (no queue, 2-3 days)

0.1 **Ignition table and figure.** DONE 2026-09-05, section 1a: the
    branches separate completely in M200c(z=6) (gap 1.24e8-2.4e8, no halo
    in it) and not in ignition mass. The threshold is a reionization-epoch
    mass boundary; Tier 2 is the physics of its LOCATION, Tier 1 tests
    whether the floor sets it. 439991 and 491413 have no z>=6 catalogs yet
    (439991's are queued, job 25093928).
0.2 **Gate-4 history of the densest cell** for 21246, 48014, 57194 (dark)
    and 1703, 47314, 56672 (bright, marginal): from every dump, the peak
    n_H, T, fH2, Sobolev column, metallicity and the implied
    m_form/10 Msun. Says by how much the dark halos missed the floor and
    whether fH2 or m_cell was the limiting factor.
0.3 **The same at nref8** (L2-gas runs of the same halos, all finished):
    the fH2 and peak-density change from 136 to 272 pc measures the
    resolution sensitivity of gate 4 directly, before any new run.
0.4 **Coarse-cell stars**: count, mass and metal budget per run; the cost
    share of their feedback (a paired bench with the density threshold set
    to 1 cm^-3 is the clean way -- one run, three root cycles, the usual
    harness); their effect on the metal-extent profiles beyond 1 Mpc.
0.5 **Assembly**: M200c(z) for both branches from the catalogs, to see
    whether the bright branch is simply the early-forming tail at fixed
    z=0 mass (expected) and how wide the overlap window is.

## 4. Test halos and what a run costs

Matched pairs at fixed z=0 mass with opposite outcomes, all with finished
L3-gas runs to branch from:

| pair | dark | bright | note |
|---|---|---|---|
| A | 21246 (1.4e9, 2.7e2) | 1703 (1.5e9, 2.1e7, ignited z=6.3 from 8.7e7) | the sharpest contrast; 1703 is the most fragile igniter |
| B | 48014 (1.3e9, 5e3) | 56672 (1.7e9, 2e6, quenched at z~2) | quenching as the third outcome |
| C | 47314 (2.1e9, 5.5e5, marginal) | -- | a halo sitting on the boundary; the most sensitive single probe |

Wall on one node, from the finished runs' dump times: z_init to z=15
under an hour, z=15 to z=6 in 0.1-0.3 day, to z=2 in 0.3-4 days, to z=0 in
2-9 days (dark halos are the cheap ones). Ignition is decided by z ~ 5 in
every case but 52675, so **a sweep run is a z_init -> z=2 run: 1-4 node-days**.
Second-leg knobs branch from the z=15 dump (RD0006) and skip the first
leg; first-leg knobs and resolution changes rerun from z_init. Runs stop at
z=2 unless the outcome is still open.

Decision metrics per run: z_ign, M200c at ignition, M* and N* at z=2, the
gate-4 history from 0.2, and whether the pair still lands on opposite
branches. A knob "moves the threshold" when it flips a pair member.

## 5. The sweeps

One knob per run. Halos: pair A (two runs) plus 47314 (one run) unless
noted, so three runs per knob.

### Tier 1 -- numerical gates (first, because Tier 2 must run on the corrected prescription)

| # | knob | current | test | what it tests |
|---|---|---|---|---|
| 1.1 | `StarMakerMinimumMass` | 10 | 1 | the floor: does 21246 ignite when 1-Msun particles are allowed? |
| 1.2 | `H2StarMakerStochastic` (after removing the goto) | 0 | 1, floor 10 | stochastic sampling below the floor; also ends the coarse-cell stars if paired with 1.3 |
| 1.3 | `H2StarMakerNumberDensityThreshold` | 0 | 1, 10, 100 cm^-3 | a physical threshold; kills coarse-cell SF; does it move the boundary? |
| 1.4 | `MaximumRefinementLevel` | 9 | 10 | cell mass halves, peak density and shielding rise; the direct test of gate 4's resolution dependence |
| 1.5 | `RefineByJeansLengthSafetyFactor` / `MinimumPressureSupportParameter` | 4 / 100 | 8 / off | how well the dense peaks are resolved at fixed nref |
| 1.6 | `StarFormationOncePerRootGridTimeStep` | 1 | 0 | dt in gate 4 becomes the level-9 step (much shorter): m_form drops ~30x per call but the maker runs every subcycle |

Expected: 1.1 and 1.4 are the decisive pair. If both leave 21246 dark and
1703 bright, gate 4 is not the driver and the threshold is physical.

### Tier 2 -- chemistry and the radiation background

| # | knob | current | test | what it tests |
|---|---|---|---|---|
| 2.1 | `H2FormationOnDust` | 0 then 1 at z=15 | 0 throughout / 1 throughout | the metals->dust->H2 loop; with it off, does any halo run away? |
| 2.2 | `self_shielding_method`, `H2_self_shielding` | 0,0 then 3,1 at z=15 | on from z_init / off throughout | the shielding half of the switch |
| 2.3 | switch redshift | 15 | 20, 10 | does the discontinuity itself set who ignites first? |
| 2.4 | UVB table | HM2012 (starts z=15.1) | FG20 / Puchwein19 (later, softer reionization) | reionization timing vs the ignition epoch z=9-15 |
| 2.5 | `LWbackground_intensity` | 0 (LW only via the UVB table) | 1e-21, 1e-20 erg/s/cm2/Hz/sr before z=15 | LW suppression of H2 cooling in the first leg, where most ignitions happen |
| 2.6 | `MultiSpecies` | 2 | 3 (H2 three-body, HD) | minihalo cooling at z > 15 |
| 2.7 | `CMBTemperatureFloor` | 1 | 0 | floor on cold-gas temperature at z > 10 |
| 2.8 | `MetalCooling` | 1 | 0 | isolates the cooling half of the metal loop from the dust half |

Expected: 2.1 and 2.4 are decisive. If the boundary tracks the UVB
redshift and vanishes with dust formation off, the sharpness is the
reionization fossil boundary and the location is set by the UVB model.

### Tier 3 -- feedback (the loop's supply side)

| # | knob | current | test |
|---|---|---|---|
| 3.1 | no feedback | on | `StarParticleFeedback = 0` (80181 L4 already running; add 1703 and 47314) |
| 3.2 | `StarFeedbackTabularSNIIEnergy` | 1e51 | 3e50, 3e51 |
| 3.3 | `StarFeedbackPreSNFeedback` | 1 | 0 |
| 3.4 | small-particle metal cap in `star_feedback6.F` | metals capped at 0.025 m_p per event (12 Msun particles yield ~6x too few) | cap removed | the metal supply to the loop, ~6x |
| 3.5 | `StarFeedbackTrackMetalSources` yields | SYGMA table | x0.3, x3 | loop gain |

3.4 is the one with a known defect behind it (memory
`small-particle-metal-cap`): the loop runs on metals, and the fleet
under-produces them by ~6x per event for every particle below 75 Msun.

### Tier 4 -- dark-matter resolution

The nref9 DM-level sweep (80181, 47314, 56672 at L2/L3/L4, running) already
covers this axis for two of the test halos. Add L4 for 21246 and 1703 if
Tier 1 leaves the boundary where it is: L4 halves the DM particle mass
again and sharpens the z > 10 progenitor peaks that ignite first.

## 6. Physics Enzo does NOT have leverage over here

- Local Lyman-Werner and ionizing radiation from the first stars in the
  halo itself: PR63 radiation is compiled in and deliberately off; the
  ignition epoch is exactly where it matters. Not in this plan.
- Pop III: the maker forms 10 Msun "Pop II" particles with the SYGMA
  yields in pristine gas; the first events at Z=0 are Pop III in all but
  name. A Pop III yield/IMF branch is Component B territory.
- Dust as a species (`dust_chemistry = 0`): Grackle's dust-to-metal ratio
  is a fixed scaling of Z; the loop gain in 2.1 is only as good as that.

## 7. Order, budget, deliverables

1. Tier 0 now: 0.1-0.5 are scripts on data in hand; 0.4's bench is one
   node for an hour.
2. Tier 1: 6 knobs x 3 halos = 18 runs at 1-4 node-days each, plus the
   stochastic-goto and level-check code changes (a day, bit-identical at
   default settings). Two weeks of queue.
3. Tier 2 on the corrected prescription: 8 knobs x 3 halos = 24 runs;
   first-leg knobs (2.3, 2.5, 2.6) rerun from z_init, the rest branch at
   RD0006. Two to three weeks.
4. Tier 3 and 4 as the Tier 1/2 outcome directs.
5. Deliverable: one figure (M* vs M200c at z=6 with every sweep run
   overlaid), one table (which knobs flipped which pair member), and a
   paragraph stating whether the boundary is reionization physics with a
   resolution-set location or a floor artefact.

The whole programme is ~60 runs at 1-4 node-days; at 16 nodes it is about
a month of queue alongside the fleet.
