DIRECTORY: `gas_metallicity`
AUTHOR: Ayan Acharyya
DATE STARTED: 01/22/2021
LAST UPDATED: 12/12/2024

This directory is meant to contain a set of python scripts to analyse all aspects of gas phase metallicity (Z; primarily in the disk) in the FOGGIE simulations.
Each script within this directory has a detailed description of what it does, along with examples, but here is a brief list (in a somewhat alphabetical order).

Some of the following scripts can be used for general purposes for any FOGGIE snapshot, i.e., not _just_ for metallicity stuff, although they were written
for metallicity-related stuff in the first place. Those scripts are marked in the list below.

[For mock IFU pipeline scripts by Ayan Acharyya, see the `galaxy_mocks/mock_ifu` directory.]

| Folder/Module        | Description |
|----------------------|-------------|
| `header.py` | Imports packages/modules required for working with scripts in this directory. |
| `util.py` | Loads some common utility functions that may be invoked by scripts in this directory |
| `compute_MZgrad.py` | Plots and measures Z gradients and saves results to file; Contrary to what the directory claims to be, this particular script
             can also plot and fit radial stellar metallicity gradients! |
| `compute_Zscatter.py` | Plots and measures Z distribution (histograms) and saves results to file |
| `datashader_movie.py` [Can be used generally, for non-metallicity stuff too] | Makes datashader plots between any two-three dimensional parameter space in the FOGGIE dataset, and then can interactively
             use the lasso tool to determine where does the selection lie spatially, and can also make animations from multiple datashader plots |
| `datashader_quickplot.py` [Can be used generally, for non-metallicity stuff too] | Makes datashader plots, but less flexibility than and without the interactiveness of `datashader_movie.py` |
| `datashader_singleplot.py` [Can be used generally, for non-metallicity stuff too] | Makes one datashader plot (no loop over snapshots), but less flexibility than and without the interactiveness of `datashader_movie.py` |
| `flux_tracking_movie.py` | Attempts to track incoming and outgoing metal mass and gas mass budget across radial shells (may not be very accurate yet) |
| `kodiaqz_merge_dsh_abs.py` | Plotting scripts for FOGGIE comparison plots of inflowing and outflowing absorbers, used in KODIAQ-Z paper |
| `make_table_fitted_quant.py` [Can be used generally, for non-metallicity stuff too] | Makes machine readable table, as well as the latex table, used in FOGGIE VIII paper |
| `nonprojected_Zgrad_evolution.py` | Plots multi-panel visualisation (instantaneous radial profile, instantaneous histogram, time evolution with verticlar line denoting given instant)
             for evolution of full 3D metallicity profile |
| `nonprojected_Zgrad_hist_map.py` | Plots histogram and 2d metallicity maps of the full 3D metallicity profile |
| `plot_MZgrad.py` [Can be used generally, for non-metallicity stuff too] | Primarily plots mass-metallicity relation for any given halo (across redshifts), but can be used to plot any two time-varying quantities
             against each other for one or more halos; ; Contrary to what the directory claims to be, this particular script
             can also plot fitted stellar metallicity gradients vs age! |
| `plot_MZscatter.py` | Plots the time evolution of the various parameterisations of the metallicity distribution (IQR vs time etc.) |
| `plot_Zevolution.py` | Makes multi-panel plots (like Fig 5 in FOGGIE VIII) for comparing time-evolution of several quantities in one plot |
| `plot_allZ_movie.py` | Plots time evolution of metallicity gradient, metallicity distribution and metallicity profile ALL in one plot and then animates the plots to make a movie |
| `plot_hist_obs_met.py` | Plots histograms of observed metallicity maps (CLEAR survey) and overplots FOGGIE data |
| `plot_metallicity_evolution.py` | Makes the plot for evolution of ambient gas metallicity around young (< 10Myr) stars as function of redshift |
| `projected_Zgrad_evolution.py` | Plots multi-panel visualisation (instantaneous radial profile, instantaneous histogram, time evolution with verticlar line denoting given instant) for evolution of projected metallicity profile |
| `projected_Zgrad_hist_map.py` | Plots histogram and 2d metallicity maps of the projected metallicity profile |
| `projected_metallicity_density.py` | Plots projected gas density and metallicity side-by-side |
| `projected_vs_3d_metallicity.py` | Plots comparison of projected metallicity (map, profile and distribution) vs full 3D metallicity distribution (similar to Fig 2 in FOGGIE VIII paper) |
| `projection_plot.py` [Can be used generally, for non-metallicity stuff too] | Initial attempts to play around with FOGGIE outputs, make projection plots. |
| `projection_plot_nondefault.py` [Can be used generally, for non-metallicity stuff too] | Plots projection plots for simulations that are not loadable by foggie_load() i.e., do not have the default paths, halo_info files etc. |
| `total_tracking_movie.py` | Makes movie by animating multiple plots of metal flux (using Cassi's code) and metal source and sink terms in radial shells |
| `track_metallicity_evolution.py` | Tracks ambient gas metallicity around young (< 10Myr) stars as function of redshift, writes to file |
| `volume_rendering_movie.py` [Can be used generally, for non-metallicity stuff too] | Crude attempt to make volume rendered movie from FOGGIe snapshots; worked okay but not great |
| `Zgrad_batch_jobsubmission.sh` | Batch file for submitting all pleiades jobs necessary to re-run the full analysis done in the FOGGIE VIII paper
