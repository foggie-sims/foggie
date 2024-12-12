DIRECTORY: `gas_metallicity/test_scripts`
AUTHOR: Ayan Acharyya
LAST UPDATED: 12/12/2024

This directory is meant to contain a set of python scripts and notebooks that were used for generic tests, learning the ropes in FOGGIE,
etc., i.e. not necessarily directly related to metallicity analysis.
Each script within this directory has a detailed description of what it does, along with examples, but here is a brief list (in a somewhat alphabetical order).

| Folder/Module        | Description |
|----------------------|-------------|
| `header.py` | Imports packages/modules required for working with scripts in this directory. |
| `util.py` | Loads some common utility functions that may be invoked by scripts in this directory |
| `muse_spectra.py` | Some random emission line plotting thing I tried for a collaborator |
| `SUV.py` | Incomplete attempt at making a Stupid Useless Visualisation (SUV) GUI tool for FOGGIE; there's two days of my life I'll never get back... |
| `test_fsp_time.py.ipynb` | Timing tests for running `foggie.galaxy_mocks/mock_ifu.filter_star_particles.py` (not used since) |
| `try_watchdog.py` | Test code for using `watchdog` as a means to check when a file is produced and then trigger the next chain of commands (not used since) |
| `volume_rendering_tests.ipynb` | Test code for volume rendering (not used since) |
| `Blizzard_RD0020_datashader_young_stars.i` | Test code for plotting location of young stars (not used since) |
| `Blizzard_RD0020_gas_metallicity.ipynb` | Test code for basic metallicity projection (not used since) |
| `Blizzard_RD0020_young_stars_age.ipynb` | Test code for plotting ages of young stars (not used since) |
| `Tempest_RD0030_KSrelation.ipynb` | Test code for plotting KS relation (not used since, see `plot_spatially_resolved.py` instead for a working script) |
| `Tempest_RD0030_Zprojection.ipynb | Test code for metallicity projection (not used since) |
| `Tempest_RD0030_ism-cgm-projection.ipynb` | Test code for demarcating halo-interface boundary (not used since) |
| `Zscatter_tests.ipynb` | Test code for plotting metallicity distributions (not used since)