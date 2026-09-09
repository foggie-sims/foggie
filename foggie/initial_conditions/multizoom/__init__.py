"""Self-contained multi-zoom IC toolkit for FOGGIE.

This package generates Enzo initial conditions with MULTIPLE independent
DM zoom regions in a single domain.  It is deliberately isolated from the
legacy single-zoom framework: nothing in enzo-mrp-music/, halo_template_*/,
or music/ is imported or modified.  The wrapper modules here began as forks
of the legacy scripts (provenance noted in each file header) and carry the
multi-halo extensions and modernization fixes.

Modules
-------
config              multi-halo [halo:<id>] configuration parsing
lagrangian_regions  trace N halos to their z_init Lagrangian volumes (one pass)
refinement_mask     deposit N particle clouds into a MUSIC RefinementMask
mrp_music           per-level MUSIC orchestration (union and merge modes)
merge_music_ics     merge N same-seed MUSIC runs into one multi-patch IC set
multizoom512        command-line driver for the 512 base-grid workflow
"""

__version__ = "0.1.0"
