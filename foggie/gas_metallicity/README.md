DIRECTORY: `gas_metallicity`
AUTHOR: Ayan Acharyya
DATE STARTED: 01/22/2021
LAST UPDATED: 12/12/2024

This directory is meant to contain a set of python scripts to analyse all aspects of gas phase metallicity (Z; primarily in the disk) in the FOGGIE simulations.
Each script within this directory has a detailed description of what it does, along with examples, but here is a brief list (in a somewhat alphabetical order).

Some of the following scripts can be used for general purposes for any FOGGIE snapshot, i.e., not _just_ for metallicity stuff, although they were written
for metallicity-related stuff in the first place. Those scripts are marked in the list below.

[For mock IFU pipeline scripts by Ayan Acharyya, see the `galaxy_mocks/mock_ifu` directory.]

File Name: `header.py`
Description: Imports packages/modules required for working with scripts in this directory.

File Name: `util.py`
Description: Loads some common utility functions that may be invoked by scripts in this directory

File Name: `compute_MZgrad.py`
Description: Plots and measures Z gradients and saves results to file

File Name: `compute_Zscatter.py`
Description: Plots and measures Z distribution (histograms) and saves results to file

File Name: `datashader_movie.py`
Description: Makes datashader plots between any two-three dimensional parameter space in the FOGGIE dataset, and then can interactively
             use the lasso tool to determine where does the selection lie spatially, and can also make animations from multiple datashader plots

File Name: `datashader_quickplot.py`
Description: Makes datashader plots, but less flexibility than and without the interactiveness of `datashader_movie.py`

File Name: `datashader_singleplot.py`
Description: Makes one datashader plot (no loop over snapshots), but less flexibility than and without the interactiveness of `datashader_movie.py`

File Name: `flux_tracking_movie.py`
Description: Attempts to track incoming and outgoing metal mass and gas mass budget across radial shells (may not be very accurate yet)

File Name: `kodiaqz_merge_dsh_abs.py`
Description: Plotting scripts for FOGGIE comparison plots of inflowing and outflowing absorbers, used in KODIAQ-Z paper

File Name: `make_table_fitted_quant.py`
Description: Makes machine readable table, as well as the latex table, used in FOGGIE VIII paper

File Name: `nonprojected_Zgrad_evolution.py`
Description: Plots multi-panel visualisation (instantaneous radial profile, instantaneous histogram, time evolution with verticlar line denoting given instant)
             for evolution of full 3D metallicity profile

File Name: `nonprojected_Zgrad_hist_map.py`
Description: Plots histogram and 2d metallicity maps of the full 3D metallicity profile

File Name: `plot_MZgrad.py`
Description: Primarily plots mass-metallicity relation for any given halo (across redshifts), but can be used to plot any two time-varying quantities
             against each other for one or more halos

File Name: `plot_MZscatter.py`
Description: Plots the time evolution of the various parameterisations of the metallicity distribution (IQR vs time etc.)

File Name: `plot_Zevolution.py`
Description: Makes multi-panel plots (like Fig 5 in FOGGIE VIII) for comparing time-evolution of several quantities in one plot

File Name: `plot_allZ_movie.py`
Description: Plots time evolution of metallicity gradient, metallicity distribution and metallicity profile ALL in one plot and then animates the plots to make a movie

File Name: `plot_hist_obs_met.py`
Description: Plots histograms of observed metallicity maps (CLEAR survey) and overplots FOGGIE data

File Name: `plot_metallicity_evolution.py`
Description: Makes the plot for evolution of ambient gas metallicity around young (< 10Myr) stars as function of redshift

File Name: `projected_Zgrad_evolution.py`
Description: Plots multi-panel visualisation (instantaneous radial profile, instantaneous histogram, time evolution with verticlar line denoting given instant) for evolution of projected metallicity profile

File Name: `projected_Zgrad_hist_map.py`
Description: Plots histogram and 2d metallicity maps of the projected metallicity profile

File Name: `projected_metallicity_density.py`
Description: Plots projected gas density and metallicity side-by-side

File Name: `projected_vs_3d_metallicity.py`
Description: Plots comparison of projected metallicity (map, profile and distribution) vs full 3D metallicity distribution (similar to Fig 2 in FOGGIE VIII paper)

File Name: `projection_plot.py`
Description: Initial attempts to play around with FOGGIE outputs, make projection plots.

File Name: `projection_plot_nondefault.py`
Description: Plots projection plots for simulations that are not loadable by foggie_load() i.e., do not have the default paths, halo_info files etc.

File Name: `total_tracking_movie.py`
Description: Makes movie by animating multiple plots of metal flux (using Cassi's code) and metal source and sink terms in radial shells

File Name: `track_metallicity_evolution.py`
Description: Tracks ambient gas metallicity around young (< 10Myr) stars as function of redshift, writes to file

File Name: `volume_rendering_movie.py`
Description: Crude attempt to make volume rendered movie from FOGGIe snapshots; worked okay but not great

File Name: `Zgrad_batch_jobsubmission.sh`
Description: Batch file for submitting all pleiades jobs necessary to re-run the full analysis done in the FOGGIE VIII paper
