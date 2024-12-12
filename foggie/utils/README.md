DIRECTORY: `utils`
AUTHOR: Ayan Acharyya, [Please add your names when you update the list below]
LAST UPDATED: 12/12/2024

This directory is meant to contain a set of python scripts used for generic analysis/handling FOGGIE data e.g., scripts to generate jobscripts for pleiades, etc.
Most scripts within this directory has a detailed description of what it does, along with examples, but here is a brief list of some of the scripts in this
directory (in a somewhat alphabetical order).

| Folder/Module        | Description |
|----------------------|-------------|

| `header.py` | Imports packages/modules required for working with scripts in this directory. |
| `util.py` | Loads some common utility functions that may be invoked by scripts in this directory |
| `binned_profile.py` | Testing utility function related to binned data |
| `contour.py` | Picks out a contour from a 2D array based on a threshold, and overplot that contour on a different plot |
| `datashader_mwe.py` | Minimum working example (MWE) for how matplotlib's datashader library can be used to make plots |
| `f_resolved.py` | Computes fraction of volume, mass, etc. resolved at different levels of refinement |
| `find_halo_center.py` | Finds center of a halo, and plots |
| `get_halo_track.py` | Computes the trackfile of a given FOGGIE halo (based on Cassi's existing code) |
| `get_run_status.py` | Prints out how many files, images, etc have been produced by a given pleiades job, so as to track how the job is doing |
| `images_to_slide.py` | Makes talk slides by collating images |
| `jobarray_template_ayan_pleiades.txt` | Template for jobarray for pleiades (almost never used) |
| `jobscript_template_ayan_pleiades.txt` | Template for jobscripts for pleiades |
| `make_multiple_movies.py` | Makes animations from multiple given sets of images |
| `merge_datashader_dataframes.py` | Merges multiple existing dataframes of the same type (e.g., corresponding to different snapshots or halos) in to one |
| `merge_datashader_plots.py` | Merges the data in multiple existing plots of the same type (e.g., corresponding to different snapshots or halos) in to one plot |
| `plot_spatially_resolved.py` | Plots spatially resolved scaling relations |
| `plot_vdisp_frb.py` | Plots velocity dispersion computed on Fixed Resolution Buffers |
| `run_foggie_sim.py` | Automatically runs new simulations: generates ICs and then submits Enzo jobs to pleiades; based on JT's code |
| `submit_jobs.py` | Submits a given script, with given command line arguments, for all halos/snapshots as multiple jobs to pleiades in one go |
