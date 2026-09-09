# Zooming a coarse parent box on a halo it cannot resolve

AUTHOR: JT and Claude
LAST UPDATED: 08/07/2026
STATUS: **demonstrated, not adopted.** Everything below was measured, not
estimated. Read the risks before building on it.

---

## Why this came up

`256`-at-L+1 and `512`-at-L are the same simulation inside the zoom:

| | effective grid | finest particle |
|---|---|---|
| 512 box at L2 | 512 x 2^2 = 2048 | 2.098e5 Msun |
| 256 box at L3 | 256 x 2^3 = 2048 | 2.100e5 Msun |

The 256 root grid has 8x fewer cells and particles (measured: 134,217,728 vs
16,777,216, particle masses 1.343e7 vs 1.075e8 Msun, ratio 8.004). So the 256
box is a cheaper substrate for identical science inside the zoom, at the price of
one extra level of AMR hierarchy.

The obstacle is not resolution but **halo selection**. A 256 root grid resolves a
1e9 Msun/h halo with ~13 particles, below what Rockstar reports, so the dwarfs
this project targets have no 256 catalog entry to zoom on. Measured for
halo80181 at its z = 0 position in the 256 parent run:

    within Rvir (21 kpc/h)      0 particles
    within 80 kpc/h             3
    within 200 kpc/h           21
    within 400 kpc/h          121

Zero inside the virial radius. The object has not collapsed at that resolution
at all -- Rockstar is not being conservative, there is nothing there.

For halo42189 (8.2e10) the counterpart exists and is unambiguous: 256 halo5673,
14.5 kpc away, 94% of the mass. The technique here is only needed at the dwarf
end.

## The seeds are present even when the halo is not

Both parent boxes share `seed[5]` through `seed[13]` **identically**, differing
only in `levelmin`/`levelmax` (8 vs 9). MUSIC generates per-level white noise
from those seeds, so the level-9 and level-10 modes -- the ones that build this
halo -- are the same field in both boxes. Refine the right volume in the 256 box
and the halo has to appear.

## Why the pipeline cannot do it as written

`enzo-mrp-music` defines the Lagrangian region by finding parent-box particles
inside a sphere at z = 0 and tracing them back to z = 99. With zero particles
there is nothing to trace. Widening the sphere until it catches some (400 kpc/h
for 121 particles) defines a region roughly twenty times too large and mostly not
the halo.

## What works: transplant the region

The region is not a set of particles, it is a **volume in the initial
conditions**, and MUSIC expresses it as a point file:

    region = convex_hull
    region_point_file = initial_particle_positions-0-RD0000.dat
    region_point_levelmin = 9

That file holds normalized [0,1] box coordinates -- verified, 127 points spanning
0.890-0.898 x 0.679-0.696 x 0.004-0.017 -- so it is resolution-independent and
transplants between boxes unchanged.

**Recipe.** Build the halo at L1 in the 512 box; copy its
`initial_particle_positions-0-RD0000.dat` into a 256 working directory; write a
256 MUSIC config identical to the 512 one except `levelmin = 8`,
`levelmax = 8+N`, and the output name.

---

## Results

Test material lives in `$FOGGIE_ICS_DIR/halo80181-256test/` and
`halo80181-256test-L2/`.

### 1. MUSIC accepts the transplant and derives the same region

    region centre, 256 test   (0.893808, 0.689565, 0.010886)
    region centre, 512 build  (0.893808, 0.689565, 0.010886)

    domain shift, 256 test    (-100,  -48, 125)
    domain shift, 512 build   (-201,  -96, 250)      -- a factor of two

Identical to six decimals. Bounding boxes differ only by `padding = 4` cells
scaling with cell size. `region_point_levelmin = 9` against `levelmin = 8` was
accepted rather than rejected, which was the open question going in.

### 2. The halo forms, at the right mass

Run to z = 0 (RD0265) and measured against the 512 catalog entry it was seeded
from:

    mass within Rvir   9.779e+08 Msun/h
    512 catalog Mvir   1.006e+09 Msun/h
    ratio              0.97

    located 138.7 kpc from the analytic position -- ordinary inter-level drift,
    not a placement error

    finest species 1.343e+07 Msun, exactly the 512 box's ROOT particle mass:
    the 256-at-L+1 == 512-at-L identity, in the data rather than in arithmetic

A three-way control -- unpatched 256 (N = 0), patched 256 L1 (N = 104), 512 L0
reference (N = 106) -- makes this a test rather than an assertion. At matched
particle mass the patched and reference panels are visually near-identical at
400 kpc; at 3 Mpc the reference has resolved structure everywhere while the
patched run has it only inside the transplanted region.

### 3. Radial profile: right where it matters, wrong outside the patch

Cumulative mass ratio, patched / reference, at z = 0:

| radius | ratio | reading |
|---|---|---|
| < 15 kpc | 1.15 - 1.45 | particle-count noise; ~104 particles inside Rvir means a handful inside 10 kpc, unresolved in BOTH runs |
| 15 - 100 kpc | 1.00 +/- 0.02 | 0.5 - 3 Rvir. **The halo is reproduced.** |
| > 100 kpc | falls to 0.81 by 300 kpc | the patched region's boundary |

M(<50 kpc) is 1.625e9 in both -- identical because both contain exactly 121
particles of the same mass, so agreement to within one particle rather than
exact agreement.

### 4. Assembly history: the halo is built the same way

Following each run's own z = 0 member particles by ID, backwards, measured in a
fixed comoving aperture of 4 Rvir. All members found at every one of 12 epochs
(106/106 and 104/104). The two curves track each other from z = 99 to z = 0,
with the patched run marginally above the reference at intermediate redshift.

**Caveat on what this test can carry:** a fixed comoving aperture is dominated by
near-mean-density material at high redshift, where the two runs would agree
almost by construction. It is genuinely discriminating below z ~ 5, and it agrees
there too.

**Two failed methods, recorded so they are not repeated:**

- *Forwards from z = 99 with shrinking spheres.* The halo does not exist yet, so
  the search locks onto whatever is densest nearby and never returns. This
  produced a smooth, plausible-looking curve reaching 1.2e12 Msun -- a thousand
  times this halo. A wrong answer that looks like a right one.
- *Selecting "the finest species" by `mass < min * 1.5`.* Meaningless in a
  unigrid parent box where every particle has the same mass: it selects all of
  them. This is what let the centre walk. In a **gas** run the same expression
  selects star particles, not DM.

### 5. One level deeper, agreement loosens

256-L2-patch against 512-L1, particle mass 1.679e6 in both:

    512 L1 (reference)   N(<Rvir) = 898    Mvir = 1.072e+09 Msun/h
    256 L2 (patched)     N(<Rvir) = 953    Mvir = 1.145e+09 Msun/h   +6.8%

2% at L1, 7% at L2. That is the direction expected if the coarse environment
matters -- deeper zoom, more sensitivity to what lies outside the patch. Still
close, but a trend rather than noise, and worth watching before pushing to L3.

### 6. Region size, and what expanding it costs

The patched region's bounding box is **521 x 861 x 724 proper kpc**, about 24
Rvir on a side, from a 127-point Lagrangian cloud spanning 278 x 617 x 480 kpc.
That is why resolved structure fades beyond ~300-400 kpc.

`radius_factor` in the mrp config (the pipeline pins it at 1.0) sets the traced
sphere. Refined volume goes as r^3:

| radius_factor | refined volume | level-9 particles |
|---|---|---|
| 1 | 1x | 1x |
| 2 | 8x | 8x |
| 3 | 27x | 27x |

**At factor 2 the tidally-accurate region reaches ~700 kpc for 8x the cost --
which is the entire saving from using the 256 root grid in the first place.**
The saving and the fix for its main weakness are the same size. That is the
central tension in this approach.

---

## Risks

1. **The tidal field differs.** Inside the refined volume the modes match, but
   the surrounding structure is sampled 8x more coarsely. Measured effect is
   modest (2% at L1, 7% at L2, assembly history agreeing) but it grows with zoom
   depth, and for a dwarf whose merger history is the science that is not a small
   caveat.
2. **No independent catalog to validate against.** There is no 256 Rockstar entry
   for these halos, so "did we get the right object" can only be answered by
   comparison against the 512 run.
3. **Selection bias through the back door.** Targets chosen from the 512 catalog,
   run at 256. Any statement about completeness or the halo mass function needs
   care.
4. **It decouples what the pipeline keeps together.** Every stage currently
   derives from its own parent box. This introduces a donor relationship the
   registry cannot express and `validate-registry` cannot check.

## What adopting it would require

- A registry field naming the donor box and halo; a row would no longer be
  self-describing.
- A `build_stage` path that writes the MUSIC config directly from a donor point
  file instead of calling `enzo-mrp-music`.
- The donor's 512 build as a prerequisite, which the dependency logic does not
  model.
- A validation step comparing the result against its donor -- mass, position and
  the projected-density ladder -- because nothing else would catch a region
  transplanted to the wrong place.

The physics works. The bookkeeping is where this gets expensive.
