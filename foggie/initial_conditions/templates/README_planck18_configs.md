# MUSIC realization templates

These six files define **which universe the campaign is in**. They are the MUSIC
template configs that every halo's ICs are cut from -- cosmology, box, and the
white-noise seed table.

## They are a versioned record, not the read path

The pipeline does NOT read them from here. `templates/halo_DM_NtoN.conf` carries

    template_config = FOGGIE_ICS_DIR/__TEMPLATE_CONFIG__

and `build.py` substitutes `__TEMPLATE_CONFIG__` from `Box.template_config` and
`FOGGIE_ICS_DIR` from the environment, so the file MUSIC actually reads is

    $FOGGIE_ICS_DIR/25Mpc_DM_512_planck18.conf

That indirection is deliberate: the seed table is the realization, which is data
rather than code, and it lives beside the parent box it describes. The copies
here exist so the realization is recoverable and auditable -- the working copies
sit on scratch with no history.

**If you change a seed table, change it in `$FOGGIE_ICS_DIR` and copy it here.**
Editing only this copy changes nothing; editing only the working copy leaves no
record.

## The seed table

Seeds run level 5 -> 15, identical across all three boxes so the
resolution-matched comparisons hold (256-L4 matches 512-L3 in DM particle mass,
256-L5 matches 512-L4).

`seed[14]` and `seed[15]` were added 2026-09-09, drawn from the OS CSPRNG, to
open L5/L6 without forcing a new realization. They are inert for every build at
`levelmax <= 13`: MUSIC's `compute_random_numbers` only generates levels up to
`levelmax`, so the existing L1-L4 ladder is bit-identical with or without them.
What each seed unlocks depends on the box's `levelmin`:

| box | levelmin | seed[14] | seed[15] |
|---|---|---|---|
| 25Mpc_DM_256 | 8 | L6 | L7 |
| 25Mpc_DM_512 | 9 | **L5 (345 Msun)** | L6 (43 Msun) |
| 25Mpc_DM_1024 | 10 | L4 | L5 |

Adding a finer seed cannot perturb coarser levels: the noise is built strictly
top-down, and `correct_avg` -- the one path that would push fine information
back down -- runs only under `kspace_TF = no`, while `main.cc` defaults it to
`yes` and these configs never set it. See the CORRECTION note in
`pipeline/config.py` for the source references.

## Known limitation

`kspace_TF` defaults to `yes`, so the transfer function is applied in Fourier
space. Hahn & Abel (2011) note this forces periodicity of the real-space
transfer function on the box scale and underestimates the two-point correlation
function, "particularly relevant for small cosmological boxes (L <~ 100 h^-1
Mpc)". This box is 25 Mpc/h. The suppression is a real systematic on the halos'
large-scale environments and belongs in the numerics section of any paper; it is
not worth a rebuild, and switching would change the realization for all 61
halos. Note `periodic_TF = yes` in these files is dead -- it is read only by the
real-space kernel, which `kspace_TF = yes` never reaches.
