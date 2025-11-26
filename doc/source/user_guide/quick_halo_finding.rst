
quick_halo_finding
==================

Quick halo finding helper script
--------------------------------

This document describes the quick_halo_finding.py utility provided with the
FOGGIE/FOGGHORN analysis suite. The script is a compact wrapper around
yt's HOP halo finder and the yt_astro_analysis HaloCatalog interface; it
prepares a small subvolume around a halo center, runs the finder, and
"repairs" the resulting halo catalog by adding derived halo quantities,
filters, and optional exports.

Key features
~~~~~~~~~~~~
- Prepare a cubic subregion centered on a halo center read from a trackfile.
- Run the HOP halo finder (via yt / yt_astro_analysis) on that subregion.
- Repair and enrich the produced halo catalog: add callbacks, filters, and
  additional derived quantities (SFR, baryon fraction, H2 when available, etc.).
- Export the final halo catalog to FITS / ASCII tables.

Command-line usage
~~~~~~~~~~~~~~~~~~
Run the script from the command line::

    python quick_halo_finding.py --output SNAPNAME --trackfile TRACKFILE \
        [--directory DIR] [--boxwidth 0.04] [--threshold 400] \
        [--min_rvir 10] [--min_mass 1e10]

Required arguments
- ``--output`` : snapshot name (SNAPNAME) to operate on (required).
- ``--trackfile`` : path to the track file used to provide halo centers (required).

Optional arguments
- ``--directory`` : path to the simulation directory (default: './').
- ``--boxwidth`` : cubic subregion side length in code units (default: 0.04).
- ``--threshold`` : overdensity threshold for HOP (default: 400.0).
- ``--min_rvir`` : minimum virial radius in kpc to keep halos during filtering (default: 10).
- ``--min_mass`` : minimum halo mass in Msun to keep halos (default: 1e10).

Primary functions (module-level)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- prep_dataset_for_halo_finding(simulation_dir, snapname, trackfile, boxwidth=0.04)

  Loads the dataset via ``foggie_load`` and returns (ds, box) where ``box``
  is a cubic ``yt.Region`` centered on the halo center read from the trackfile.
  The returned box is intended for use as a subvolume to supply to the halo finder.

- halo_finding_step(ds, box, simulation_dir='./', threshold=400.)

  Runs the HOP finder (via ``HaloCatalog(..., finder_method='hop')``) on the
  provided subvolume and writes a raw catalog to ``simulation_dir/halo_catalogs``.
  Returns the created ``HaloCatalog`` instance.

- repair_halo_catalog(ds, simulation_dir, snapname, min_rvir=10., min_halo_mass=1e10)

  Loads the raw halo catalog (.0.h5) for the snapshot, wraps it in a
  ``HaloCatalog`` with the provided dataset context, applies a set of filters
  (Rvir / mass), registers a collection of derived halo quantities (via
  the callbacks defined in ``foggie.utils.halo_quantity_callbacks``), and
  re-creates / saves the repaired catalog. When available, MultiSpecies
  runs also get H2-derived quantities added.

- export_to_astropy(simulation_dir, snapname)

  Reads the final halo catalog HDF5, converts halo fields to an Astropy
  QTable and writes a FITS file and an ASCII table to the same catalog
  directory.

Outputs
~~~~~~~
- Repaired halo catalog (HDF5): ``<directory>/halo_catalogs/<snapname>/<snapname>.0.h5``
- Optional table exports:
  - FITS: ``<directory>/halo_catalogs/<snapname>/<snapname>.0.fits``
  - ASCII: ``<directory>/halo_catalogs/<snapname>/<snapname>.0.txt``

Dependencies
~~~~~~~~~~~~
- yt
- yt-astro-analysis (yt_astro_analysis.halo_analysis.HaloCatalog)
- astropy
- scipy
- foggie package utilities: foggie.utils.foggie_load, foggie.utils.halo_quantity_callbacks, etc.

Notes and recommendations
~~~~~~~~~~~~~~~~~~~~~~~~~
- The script expects the dataset to carry halo center information (via the
  FOGGIE trackfile) that ``foggie_load`` can read and attach to the dataset.
- The HOP finder parameters (threshold, subvolume) should be tuned to the
  mass/resolution of your run; default threshold = 400 is conservative for
  compact subregions.
- The repair step adds many derived quantities; ensure that the callbacks in
  ``halo_quantity_callbacks.py`` are present and compatible with your run
  (e.g. MultiSpecies flag for H2).
- The script is intended for fast, local halo-catalog creation in the
  high-resolution region rather than a full cosmological halo finding run.

Example workflow
~~~~~~~~~~~~~~~~
1. Prepare a small dataset and subvolume::

       ds, box = prep_dataset_for_halo_finding('./', 'RD0014', 'track_001.txt', boxwidth=0.04)

2. Run the halo finder on the subvolume::

       hc = halo_finding_step(ds, box, simulation_dir='./', threshold=400.)

3. Repair and augment the catalog::

       hc = repair_halo_catalog(ds, './', 'RD0014', min_rvir=10., min_halo_mass=1e10)

4. Export to tables::

       export_to_astropy('./', 'RD0014')

See also
~~~~~~~~
- foggie.utils.foggie_load
- foggie.utils.halo_quantity_callbacks
- yt_astro_analysis.halo_analysis.HaloCatalog

References
~~~~~~~~~~
- yt documentation: https://yt-project.org
- yt-astro-analysis halo analysis: https://yt-astro-analysis.readthedocs.io
