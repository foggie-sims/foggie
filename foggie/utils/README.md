DIRECTORY: `utils`
AUTHOR: Ayan Acharyya, [Please add your names when you update the list below]
LAST UPDATED: 12/12/2024

This directory is meant to contain a set of python scripts used for generic analysis/handling FOGGIE data e.g., scripts to generate jobscripts for pleiades, etc.
Most scripts within this directory has a detailed description of what it does, along with examples, but here is a brief list of some of the scripts in this
directory (in a somewhat alphabetical order).

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

File Name: `plot_spatially_resolved.py`
Description: Plots spatially resolved scaling relations

File Name: `plot_vdisp_frb.py`
Description: Plots velocity dispersion computed on Fixed Resolution Buffers

File Name: `run_foggie_sim.py`
Description: Automatically runs new simulations: generates ICs and then submits Enzo jobs to pleiades; based on JT's code

File Name: `submit_jobs.py`
Description: Submits a given script, with given command line arguments, for all halos/snapshots as multiple jobs to pleiades in one go

