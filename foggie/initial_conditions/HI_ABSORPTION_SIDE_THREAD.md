# Ultrafaint halos as HI absorbers: a side thread

**Status: first results, 2026-09-20.** A side thread of the ultrafaint ignition
paper, not part of its main line. The main paper asks why these halos do not
form stars; this asks what a background quasar would see through them, and
whether absorption statistics independently constrain the same physics.

## The question

A sightline crosses many ultrafaint halos. Do those crossings appear in HI
absorption, and does the observed incidence of Lyman-limit systems and DLAs
constrain whether ultrafaints keep their gas?

## What was measured

Prototypes: **halo331302** and **halo74411**, the diagnostic pair, in two arms:

| arm | UV background | stars + feedback |
|---|---|---|
| `default` | HM2012, Enzo's z = 7 gate | yes |
| `norad` | none | yes |

L3 DM (2.2e4 Msun particles) and L4 DM (2.8e3) where available, hydro at nref9,
at z = 6, 4, 2, 1, 0.

**Method** (`halocat/scripts/hi_column_profiles.py`): sightlines at log-spaced
impact parameters b/R200c from 0.03 to 3, 24 per bin (3 axes x 8 azimuths),
N(HI) integrated over +-5 R200c of the halo plane from `HI_Density`. Centers and
R200c come from the `diag_arm_history.py` histories, so nothing is re-centered.
Per redshift it also records the covering fraction above each threshold and the
effective radius R_eff = sqrt(2 Int f_c(b) b db), so that sigma = pi R_eff^2
reproduces the covering fraction.

Figure: `figures_z0/hi_profiles/hi_profiles.png`; data `prof_*.json`.

## Result 1: with the background, the halos stop being absorbers

Median log N(HI), halo331302 L3 (74411 is the same picture, one step earlier):

| z | b = 0.05 | b = 0.5 | b = 1 | b = 2 | R_LLS/R200c | R_DLA/R200c |
|---|---|---|---|---|---|---|
| 6.1 | 21.8 | 17.9 | 16.1 | 15.6 | 1.00 | 0.12 |
| 4.1 | 19.8 | 15.2 | 14.7 | 14.6 | 0.21 | 0.07 |
| 2.0 | 12.6 | 12.0 | 11.9 | 11.9 | 0 | 0 |
| 1.0 | 12.5 | 12.1 | 11.8 | 11.6 | 0 | 0 |
| 0.0 | 12.8 | 12.1 | 11.7 | 11.6 | 0 | 0 |

Three regimes:

1. **z = 6, pre-evaporation.** A DLA core (10^21.8 at 0.05 R200c) with
   Lyman-limit gas to ~R200c. R_DLA = 0.12-0.18 R200c.
2. **z = 4, mid-evaporation.** 331302 keeps a small Lyman-limit disk
   (R_LLS = 0.2 R200c); 74411, the smaller halo, is already flat at 10^14.
3. **z <= 2, empty.** Flat profiles at 10^11.6-10^12.8 with no center-to-2R200c
   gradient. Below typical detection limits (~10^12.5-13): not an absorber at all.

L4 tracks L3 throughout, so this is not a DM-resolution effect.

In `norad` the same halos sit at log N ~ 20.5-21.5 in the center and 17-19 at
R200c **at every redshift**, including z = 0.

## Result 2: the incidence, and the observational constraint

Halo abundance from Tinker+08 for M200c, box cosmology (Om = 0.291, h = 0.70,
sigma8 = 0.810, ns = 0.9665), M200c = 1e8-1e9 Msun:

- n(z=0) = 7.4 per comoving Mpc^3, R200c = 10-21 kpc.
- **dN/dz for crossing R200c = 17-20**, i.e. **~19 crossings from z = 0 to 1**.
  That is comparable to the Lya forest above 10^14 and 5-10x rarer than 10^13.

Folding in the measured R_eff/R200c (prototypes applied to the whole mass range):

| arm | z | dN/dz LLS | dN/dz DLA |
|---|---|---|---|
| default | 0, 1, 2 | **0** | **0** |
| default | 4 | 0.07 | 0.008 |
| default | 6 | 2.4 | 0.06 |
| norad | 0 | 46 | 0.11 |
| norad | 1 | 97 | 0.24 |
| norad | 2 | 81 | 0.30 |
| observed, z < 1 | | 0.3-0.5 | ~0.05 |

**The two headline statements:**

1. **In the realistic arm, ~19 R200c crossings per unit redshift produce zero
   detectable absorption.** Ultrafaints are invisible to HI absorption surveys
   at z <= 2 despite being the most numerous halo population crossed.
2. **Observed LLS counts independently rule out gas retention.** If ultrafaints
   kept their gas (`norad`), they alone would give dN/dz(LLS) ~ 46-97 against an
   observed 0.3-0.5: a 100-300x overprediction. No simulation detail is needed
   for this argument, only the halo mass function and a cross-section.

This is the absorption-side counterpart of the main paper's physics ladder,
which shows the same evacuation from inside the halo ([[sf-gate-diagnostic]],
the norad/norad-nostar controls).

## Relation to prior work

The machinery used here -- halo mass function x HI cross-section -> dN/dz -- is
not new, and neither is the finding that photoionization destroys the
absorption cross-section of low-mass halos. Both have a literature, all of it
at z >~ 3. What is unclaimed is the UFD mass range, the low-redshift regime,
and the inverse use of the incidence as a constraint.

**The minihalo absorber model.** Abel & Mo (1998, astro-ph/9712119) proposed
minihalos as the origin of the z ~ 3 Lyman-limit population and computed
dN/dz = 3.7 assuming the cold gas fills r_vir. That is structurally our `norad`
calculation, and the same order as the `default` prediction of dN/dz(LLS) = 2.4
at z = 6 in the table above. Sternberg, McKee & Wolfire (2002,
astro-ph/9901313) worked out absorption-line signatures of gas in mini dark
matter halos in detail.

**The walk-back, which is Result 1 in another guise.** Maller et al. (2003,
astro-ph/0211231) found that with a realistic photoionized gas profile
minihalos contribute almost nothing to the LLS cross-section: to cover area the
gas must be low-column, and low-column gas is ionized by the background. Kohler
& Gnedin (2007, astro-ph/0605032) confirmed from simulations that LLSs span a
wide range of halo mass but that low-mass halos do not dominate the
cross-section. **"The UVB removes the low-mass-halo absorption cross-section"
is therefore published**, for 1e6-1e8 Msun at z ~ 3. We should cite it up
front: our contribution on this point is the direct measurement in a resolved
hydro zoom at UFD mass with non-equilibrium chemistry, not the discovery.

**Photoevaporation folded into absorber statistics.** The closest modern
analogue is Park, Lukic, Sexton, Alvarez & Shapiro (2023, arXiv:2309.04129),
"Impact of Self-shielding Minihalos on the Lya Forest at High Redshift": 1D
radiation-hydrodynamics photoevaporation of 1e6-1e8 Msun minihalos, converted
to N(HI) profiles and folded into a large-box forest. They find the DLA
incidence rises by 2-4x at z ~ 5.5 relative to z ~ 4.5, ~3% mean flux
suppression, and a ~5% boost to the 1D power at k ~ 0.1 h/Mpc -- the same
chain we follow (self-shielding -> photoevaporation -> columns -> incidence),
but framed as contamination of the forest during reionization rather than as a
probe of dwarf-galaxy gas. Their conclusion already contains our null result:
at post-reionization ionizing rates (Gamma >~ 0.3e-12 s^-1) the effect "becomes
much smaller". Our flat 10^11.6 profiles at z <= 2 are the endpoint of that
statement. Upstream physics: Shapiro, Iliev & Raga (2004, astro-ph/0307266) and
Nakatani et al. (2020, arXiv:2007.08149).

**The low-z census stops where our fleet starts.** Hafen et al. (2017,
arXiv:1608.05712; MNRAS 469, 2292) convolve 14 FIRE zooms spanning
M_h = 1e9-1e13 Msun with the halo mass function and find that low-redshift LLSs
live in 1e10 <~ M_h <~ 1e12 halos -- winds, cool inflows, and dwarf
*satellites*. Bhagwat et al. (2023, arXiv:2311.18000) measure the HI covering
fraction of LLSs in FIRE halos; Bird et al. (2013, arXiv:1307.6879) survey how
physics choices move LLS/DLA statistics. Their lowest zoom rung is 1e9, so the
1e8-1e9 contribution is *assumed* negligible rather than measured. That is
exactly the gap this thread fills.

**What is genuinely unclaimed here:**

1. N(HI) vs impact parameter measured in resolved 1e8-1e9 Msun zooms with
   non-equilibrium chemistry, tracked from z = 6 to z = 0. The prior work is
   either analytic/1D (Abel & Mo; Park+) or resolution-limited to M_h >= 1e9
   (Hafen+).
2. The constraint direction: observed dN/dz(LLS) as a *falsification* of gas
   retention at UFD mass. The 100-300x overprediction in `norad` is a
   quantitative version of an argument the field has only made qualitatively,
   as a census result.
3. The z <= 2 statement at all. The minihalo absorber literature stops at
   z >~ 3 because that is where minihalos were thought to matter.

**Consequence for where this goes.** The z >= 4 direction flagged in the
section above is also the regime where the literature is set up to receive the
result: our dN/dz(LLS) = 2.4 at z = 6 is directly comparable to Abel & Mo's 3.7
and to the factor 2-4 DLA enhancement of Park et al. (2023). Hafen et al.
(2017) is also the natural place to source the observed z < 1 LLS/DLA rates
that are currently quoted from memory in the caveats below, since they validate
against the same measurements.

## Where it could go in the paper

A short section or appendix: "ultrafaints are not absorbers, and absorbers say
ultrafaints are empty". It needs no new runs at z <= 2. The interesting
unexplored regime is z >= 4, where `default` still predicts dN/dz(LLS) ~ 2.4 at
z = 6 from this mass range alone -- worth comparing against high-z LLS counts.

## Caveats

- **Two halos** stand in for the whole 1e8-1e9 Msun range. R_eff/R200c is
  assumed mass-independent within it; it is not measured to be.
- **The `norad` R_LLS ~ 2.5 is a lower bound**: that gas is still above the
  Lyman limit at the 3 R200c edge of the sampled range.
- **Observed rates are quoted from memory** (Danforth+2016; Ribaudo+2011;
  Rao+2006 in the z < 1 range) and must be checked against the papers before
  publication.
- **No self-shielding correction beyond what Grackle applies**, and no
  ionizing-radiation transfer: HI comes from the run's own chemistry.
- **Fixed comoving refinement** means the densest gas is progressively
  under-resolved toward z = 0 (peak n_H falls as (1+z)^3), which biases low-z
  columns low in the dense regime. It does not affect the `default` conclusion,
  whose gas is gone regardless.
- **Subhalos are not counted** in the mass function (+10-20% on crossings).

## Reproducing

```bash
# profiles (one PBS job, ~30 min)
qsub /nobackupnfs1/jtumlins/halocat/pbs/hi_profiles.pbs      # reads figures_z0/hi_profiles/jobs.txt
python3 halocat/scripts/plot_hi_profiles.py --in figures_z0/hi_profiles \
    --out figures_z0/hi_profiles/hi_profiles
```

Single-sightline columns at one epoch (the z = 2 check quoted in the thread):
`scratchpad/hi_columns_331302.py`.
