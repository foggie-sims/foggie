# Where our ultrafaint ignition study sits in the literature

## Context

We have just shown (2026-09-08/09) that in our fleet the star formation
threshold at M200c(z=6) ~ 2.4e8 is set by a *numerical* gate, not physics: the
only live test in `star_maker_h2reg.F` is `m_form >= StarMakerMinimumMass =
10 Msun`, the stochastic branch that was meant to soften it had been dead code
since 2023, and repairing it ignites three halos that were dark in every
previous configuration (fiducial, ungated HM2012, and no radiation at all).
Before we build a paper on that, we need to know what the highly-resolved UFD
simulation literature already says about how and when ultrafaints ignite, how
our resolution and physics compare, and therefore what is actually new here.

This document is that assessment plus the work it implies. It is a research
plan, not a code plan.

---

## 1. The comparison set

| study | code | DM particle | baryon resolution | spatial | SF prescription | H2 | radiation | sample |
|---|---|---|---|---|---|---|---|---|
| **LYRA ultra-faints** (Brown+ 2025, 2511.21824) | AREPO | 74.7 Msun | **4 Msun** gas cell; 4 Msun min star particle | 5.9–14.8 pc softening | n_H > 1e3 cm^-3 AND T < 100 K, Schmidt eps=0.02 (eps=1 above 1e4) | equilibrium tables (CHIMES/Ploeckinger+25); non-eq only H/He | no RT; **two LW backgrounds** (FG20 vs Incatasciato+23) | **65 halos**, M200c 1e7–5e9, to z=0 |
| **EDGE2** (Rey+ 2025, 2503.03813) | RAMSES-RT | 950 Msun | ~150 Msun cell target | **3 pc** | rho > 300 m_p cm^-3, T < T*, eps_ff = 10%, 300 Msun particles | **non-equilibrium H2** (Nickerson+18), gas/dust/collisional | **on-the-fly M1 RT, 6 bins incl. 12–13.6 eV H2-dissociating**; FG20 UVB ramped to full at z=6 | 15 halos, M200 1e9–1e10 |
| Wheeler+ 2019 | GIZMO/FIRE-2 | — | **30 Msun** | ~pc | FIRE-2 (self-gravitating, dense) | approximate | local + UVB | ~15 UFDs |
| Applebaum+ 2021 (Mint DC Justice League) | ChaNGa | ~1e4 Msun | ~1e3 Msun | ~90 pc | density + T threshold | metal-line + H2 approx | UVB | MW-context UFDs |
| **Kuhlen+ 2012** (1105.2376) — *our prescription's origin* | **Enzo** | 3.1e6 Msun | m_min = **1e4 Msun**, **with stochastic sampling below it** | 76 proper pc at z=4 | H2-regulated, KMT09 analytic f_H2, eps=0.01, **no density threshold** | KMT09 analytic (column + metallicity) | optically thin HM01 UVB; LW background varied J/J_MW = 1–1000 | 12.5 Mpc box, z >= 4 |
| **this work** | **Enzo** | 2.21e4 (L3) / 2764 (L4) Msun | 10 Msun min star particle; cell mass ~7.6e4 Msun at n_H=100, z=4 | **136 comoving pc / 27 proper pc at z=4** (nref9) | H2-regulated, `star_maker_h2reg`, eps=0.02, **no density threshold, no f_H2 threshold** | **non-equilibrium Grackle H2** (MultiSpecies=2), H2Method=1 | tabulated HM2012, **no RT, no LW model** | **61 halos**, 27 L3-gas to z=0, **+18 reion +18 norad matched arms** |

## 2. Where we sit on resolution — mid-pack, and behind on baryons

- **Spatial**: 27 proper pc at z=4 versus EDGE's 3 pc. At fixed gas density our
  cell holds ~700x more mass than an EDGE cell and ~2e4x more than a LYRA cell.
  **We are not a high-resolution study by the standards of this subfield.**
- **Dark matter**: 2.2e4 Msun (L3) sits between EDGE (950) and Kuhlen (3.1e6);
  our L4 rung at 2764 Msun is competitive with EDGE. This is our better axis.
- **Star particles**: our 10 Msun floor is the smallest in the table apart from
  LYRA's 4 Msun — but LYRA's 4 Msun particles are *individual stars* drawn from
  a sampled IMF, while ours are unresolved populations. Not the same claim.

The consequence matters for the headline: **R ~ f_H2 * n_H^(3/2) * dx^3**, so
our gate is 700x more permissive than EDGE's would be at the same density, and
2e4x more permissive than LYRA's. Any threshold we quote is a threshold at
nref9. This is the single most important caveat in the study and it is
quantifiable, which is an opportunity rather than only a weakness.

## 3. Where we sit on physics — one clear advantage, two clear gaps

**Advantage.** We follow **non-equilibrium H2 in Grackle** and feed the actual
`H2I + H2II` field to the star maker (`H2StarMakerH2FractionMethod = 1`). LYRA
takes H2 from *equilibrium* tables. Kuhlen+ 2012 — same star maker — used the
KMT09 *analytic* column-density model, which returns **exactly zero f_H2 at
zero metallicity**. Only EDGE2 matches us here, and they add RT on top.

**Gap 1 — no Lyman-Werner treatment at all.** `LWbackground_intensity = 0`,
no local LW, no LW band. LYRA's headline result is that the early LWB moves the
dark-to-luminous transition by a full dex (M200c ~ 1e7 with weak LWB to ~1e8
with strong), and that UFDs specifically are the population sensitive to it.
EDGE2 carries a 12–13.6 eV dissociating band in its RT. **We cannot currently
say anything about the dominant uncertainty that the newest paper in the field
identifies**, and our norad arm is the maximally optimistic LW case.

**Gap 2 — no Pop III enrichment floor.** Our dark halos sit at Z = 7.7e-9 Zsun
forever. Kuhlen+ 2012 imposed `Z_floor = 1e-3 Zsun at z = 9` *precisely
because* their H2 model gives nothing at Z = 0, and treated its amplitude and
timing as a tested parameter. LYRA forms stars at primordial metallicity but
notes the absence of a Pop III model as a leading uncertainty for exactly the
single-burst, self-quenching systems we are producing. Our metals->dust->H2
bootstrap therefore never starts *by construction*, and that is a modelling
choice we have not made deliberately.

Also missing relative to EDGE2: on-the-fly RT of any kind.

## 4. Novelty assessment

**Not novel:** resolution (we are behind LYRA, EDGE and FIRE-2 on baryons);
H2-regulated star formation in Enzo (Kuhlen+ 2012 is the same routine);
"reionization quenches ultrafaints" (settled since Bullock+ 2000).

**Genuinely novel, in descending order of strength:**

1. **The minimum-star-particle-mass gate is the star formation threshold in
   the UFD regime — measured, predicted in advance, and removed.** Kuhlen+
   2012, the authors of this very prescription, wrote that a hard mass
   threshold "can lead to a significant amount of *unfulfilled* star
   formation, although this can be remedied with a stochastic SF criterion" —
   and then used stochastic sampling with m_min = 1e4 Msun. The warning is 14
   years old and, as far as this survey found, **nobody has ever measured how
   much star formation the hard threshold actually discards, or shown that it
   is what sets the apparent halo-mass threshold.** We can: R = 0.019–0.092 for
   the dark halos (short by 11–52x), predicted yields of 7/93/258 particles
   from the measured R histories, and measured 7/81/323 after repairing the
   branch. This is a numerical-methods result that applies to every code with
   a minimum star particle mass, and it is the strongest thing we have.

2. **A matched three-arm radiation-exposure ladder on the same halos**:
   fiducial (Enzo's hard-coded `RadiationRedshiftOn = 7` gate), reion
   (unmodified HM2012 from z = 15.13), norad (no background whatsoever), 18
   halos each, same ICs, same binary. LYRA varies the LWB across two arms on
   the same 65 halos, so matched arms are not unprecedented — but a **true
   zero-radiation arm** is unusual, and it is what let us separate "the UVB
   removes the fuel" from "the halo cannot make H2 anyway". Result: removing
   the background entirely buys only 1–10x of the 11–52x needed.

3. **The Enzo z=7 gate as a documented artifact.** `SetDefaultGlobalValues.C`
   silently sets `RadiationRedshiftOn = 7`, so every Enzo run using tabulated
   HM2012 has been getting a step function at z=7 rather than the tabulated
   history. Worth a short methods note on its own; it is a trap for the whole
   Enzo user base.

4. **The predictive diagnostic** R = f_H2 n_H^(3/2) dx^3 / (M_min/eps dt),
   computed from dumps and shown to predict ignition yields to ~25% before the
   experiment ran.

**The discrepancy that most needs addressing:** LYRA finds ignition in
progenitors of **M200c ~ 1e5–1e6 Msun at z >~ 8**, and a dark-to-luminous
transition at 1e7–1e8. Our dark halos are at M200c ~ 1e7–1e8 and never ignite
at all in the fiducial configuration. That is a 1–2 dex disagreement with the
newest and highest-resolution study in the field. The candidate explanations
are exactly our two gaps plus the mass gate, and we can test all three.

## 5. Plan

### Tier A — close the two gaps that the literature says dominate (highest value)

- **A1. Pop III enrichment floor.** Add `Z_floor = 1e-3 Zsun` at z = 9,
  following Kuhlen+ 2012 exactly, plus 1e-4 and 1e-2 variants. This is the
  standard of the lineage our prescription comes from and its absence is
  currently an unforced error. Branch from norad RD0007 as the stochastic runs
  did. **Expect this to be the single largest effect after stochastic SF**, and
  it directly tests whether the dark halos are dark for want of Pop III metals.
- **A2. Lyman-Werner background.** `LWbackground_intensity` and
  `TabulatedLWBackground` already exist in the deck (both currently 0). Run the
  two LYRA prescriptions (FG20-like weak, Incatasciato+23-like strong) so our
  results are directly comparable to the paper that will be the reference.
  Without this we cannot engage with the field's leading uncertainty.

### Tier B — the resolution statement, which we must make regardless

- **B1.** `MaximumRefinementLevel` 9 -> 10 -> 11 on 543386 and 537545 with
  stochastic SF on, measuring how the ignition threshold moves. Since
  R ~ dx^3, the prediction is that the deterministic threshold moves *up*
  strongly with resolution while the **stochastic** threshold does not. If that
  holds it is the cleanest possible demonstration that stochastic sampling is
  the resolution-independent choice, and it turns our resolution disadvantage
  into the experiment.
- **B2.** The L4 rung of the nref9 DM sweep (47314, 56672, 80181) — 47314-L4
  has just reached z=0. The L2 rung was deleted 2026-09-08, so this is L3 vs L4.

### Tier C — finish the stochastic result

- **C1.** Stochastic branched off the **fiducial** arms rather than norad, to
  answer whether the ignited halos re-quench when radiation is restored. The
  fiducial-R prediction says the three smallest drop to 3–6 particles formed
  almost entirely before z=6 (ignite-then-quench fossils) while the rest keep
  forming — but that estimate cannot see the enrichment bootstrap or SN
  blowout, which is the whole point of running it.
- **C2.** A bright halo (42502 or 15659) branched with stochastic SF on, to
  check the recipe does not move the well-resolved end. The control 235659
  moved by 100x; if the bright halos move similarly the recipe cannot go
  fleet-wide as-is.
- **C3.** Aperture fix: R200c is ~7 ckpc/h for a 2e7 Msun halo and its star
  count swings with merger bookkeeping (halo174526 read 7 in R200c vs 28 in a
  fixed 100 ckpc/h aperture). Adopt a fixed comoving aperture for the UFDs and
  state it.

### What we should NOT claim

- Not a resolution record. Say plainly that EDGE2 is 3 pc and LYRA is 4 Msun,
  and that our threshold is a threshold at nref9.
- Not "we discovered that H2 regulates dwarf star formation" — that is Kuhlen+
  2012 and Gnedin+ 2009.
- Not a statement about the LW background until A2 exists.

## 6. Verification

- Every run measured with the existing `halocat/scripts/sf_gate_history.py`
  (per-dump R, f_H2, n_H, Z, reservoirs, N* in R200c) and plotted with
  `plot_sf_gate.py` / `plot_sf_onset.py`; arm comparisons with
  `sixpanel_arms.py` + `plot_sixpanel_arms.py`.
- Ledger row per run: config, binary SHA, N* and M* in R200c *and* in a fixed
  100 ckpc/h aperture, peak R, ignition redshift.
- Cross-check against the literature at the two points where it is directly
  comparable: halo mass and redshift of first star formation (LYRA Fig. 4:
  M200c ~ 1e5–1e6 at z >~ 8), and the dark-to-luminous transition mass (LYRA
  Fig. 3: 1e7 weak LWB, 1e8 strong LWB; observational inference ~1e8).
- Disk: Tier A is 12 runs (~8 TB) branched at z=12.5 and stopped at z=4. Do not
  queue Tier B until Tier A reports; quota headroom is ~46 TB and the control
  arms were capped at z=2 on 2026-09-08 to buy it.

## Sources

- LYRA ultra-faints: https://www.alphaxiv.org/abs/2511.21824
- EDGE2 scaling relations: https://www.alphaxiv.org/abs/2503.03813
- LYRA III (reionization survivors): https://www.alphaxiv.org/abs/2209.03366
- Kuhlen+ 2012, H2-regulated SF in Enzo: https://arxiv.org/pdf/1105.2376
- Wheeler+ 2019, 30 Msun dwarfs: https://academic.oup.com/mnras/article/490/3/4447/5588613
- Applebaum+ 2021, Mint DC Justice League: https://arxiv.org/pdf/2008.11207
- Munshi+ 2019, uncertainty in UFD predictions: https://arxiv.org/pdf/1810.12417
- Applebaum/Brooks stochastic IMF: https://arxiv.org/pdf/1811.00022
