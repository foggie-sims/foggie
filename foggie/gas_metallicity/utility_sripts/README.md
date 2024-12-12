DIRECTORY: `gas_metallicity/utility_scripts`
AUTHOR: Ayan Acharyya
LAST UPDATED: 12/12/2024

This directory is meant to contain a set of python scripts and notebooks that were used for generic tests, job scripts,
etc., i.e. not necessarily directly related to metallicity analysis.
Each script within this directory has a detailed description of what it does, along with examples, but here is a brief list (in a somewhat alphabetical order).

File Name: `header.py`
Description: Imports packages/modules required for working with scripts in this directory.

File Name: `util.py`
Description: Loads some common utility functions that may be invoked by scripts in this directory

File Name: `binned_profile.py`
Description: Testing utility function related to binned data

File Name: `contour.py`
Description: Picks out a contour from a 2D array based on a threshold, and overplot that contour on a different plot

File Name: `datashader_mwe.py`
Description: Minimum working example (MWE) for how matplotlib's datashader library can be used to make plots

File Name: `f_resolved.py`
Description: Computes fraction of volume, mass, etc. resolved at different levels of refinement

File Name: `find_halo_center.py`
Description: Finds center of a halo, and plots

File Name: `get_halo_track.py`
Description: Computes the trackfile of a given FOGGIE halo (based on Cassi's existing code)

File Name: `get_run_status.py`
Description: Prints out how many files, images, etc have been produced by a given pleiades job, so as to track how the job is doing

File Name: `images_to_slide.py`
Description: Makes talk slides by collating images

File Name: `jobarray_template_ayan_pleiades.txt`
Description: Template for jobarray for pleiades (almost never used)

File Name: `jobscript_template_ayan_pleiades.txt`
Description: Template for jobscripts for pleiades

File Name: `make_multiple_movies.py`
Description: Makes animations from multiple given sets of images

File Name: `merge_datashader_dataframes.py`
Description: Merges multiple existing dataframes of the same type (e.g., corresponding to different snapshots or halos) in to one

File Name: `merge_datashader_plots.py`
Description: Merges the data in multiple existing plots of the same type (e.g., corresponding to different snapshots or halos) in to one plot

File Name: `muse_spectra.py`
Description: Some random emission line plotting thing I tried for a collaborator

File Name: `plot_spatially_resolved.py`
Description: Plots spatially resolved scaling relations

File Name: `plot_vdisp_frb.py`
Description: Plots velocity dispersion computed on Fixed Resolution Buffers

File Name: `run_foggie_sim.py`
Description: Automatically runs new simulations: generates ICs and then submits Enzo jobs to pleiades; based on JT's code

File Name: `submit_jobs.py`
Description: Submits a given script, with given command line arguments, for all halos/snapshots as multiple jobs to pleiades in one go

File Name: `SUV.py`
Description: Incomplete attempt at making a Stupid Useless Visualisation (SUV) GUI tool for FOGGIE; there's two days of my life I'll never get back...

File Name: `test_fsp_time.py.ipynb`
Description: Timing tests for running `foggie.galaxy_mocks/mock_ifu.filter_star_particles.py` (not used since)

File Name: `try_watchdog.py`
Description: Test code for using `watchdog` as a means to check when a file is produced and then trigger the next chain of commands (not used since)

File Name: `volume_rendering_tests.ipynb`
Description: Test code for volume rendering (not used since)

File Name: `Blizzard_RD0020_datashader_young_stars.i`
Description: Test code for plotting location of young stars (not used since)

File Name: `Blizzard_RD0020_gas_metallicity.ipynb`
Description: Test code for basic metallicity projection (not used since)

File Name: `Blizzard_RD0020_young_stars_age.ipynb`
Description: Test code for plotting ages of young stars (not used since)

File Name: `Tempest_RD0030_KSrelation.ipynb`
Description: Test code for plotting KS relation (not used since, see `plot_spatially_resolved.py` instead for a working script)

File Name: `Tempest_RD0030_Zprojection.ipynb
Description :Test code for metallicity projection (not used since)

File Name: `Tempest_RD0030_ism-cgm-projection.ipynb`
Description: Test code for demarcating halo-interface boundary (not used since)

File Name: `Zscatter_tests.ipynb`
Description: Test code for plotting metallicity distributions (not used since)