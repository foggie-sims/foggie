DIRECTORY: `paper_plots/Z_gradient`
AUTHOR: Ayan Acharyya
LAST UPDATED: 12/12/2024

This directory is meant to contain a set of python scripts used to reproduce all plots in the `FOGGIE VIII paper`.
Each script within this directory has a description of what it does at the start of the script.

Below is a list of commands to run in order to reproduce these plots:

### Code for reproducing paper plots

---

Before making any evolution plots, to measure Z distribution for *all* snapshots:

```bash
## (from inside /nobackupp19/aachary2/foggie_outputs/pleiades_workdir/ directory in pleiades) [for running on LDANs]
run /nobackupp19/aachary2/ayan_codes/foggie/foggie/ayan_acharyya_codes/submit_jobs.py --call compute_MZgrad --system ayan_pleiades --do_all_halos --queue ldan --mem 1500GB --prefix cmzg --opt_args "--do_all_sims --use_onlyDD --upto_kpc 10 --xcol rad --write_file --forpaper"
run /nobackupp19/aachary2/ayan_codes/foggie/foggie/ayan_acharyya_codes/submit_jobs.py --call compute_Zscatter --system ayan_pleiades --do_all_halos --queue ldan --mem 1500GB --prefix cmzs --opt_args "--do_all_sims --use_onlyDD --upto_kpc 10 --nbins 100 --write_file --forpaper"
## OR for running on Endeavour nodes
run /nobackupp19/aachary2/ayan_codes/foggie/foggie/ayan_acharyya_codes/submit_jobs.py --call compute_MZgrad --system ayan_pleiades --do_all_halos --queue e_normal --prefix cmzg --opt_args "--do_all_sims --use_onlyDD --upto_kpc 10 --xcol rad --write_file --forpaper" --nnodes 2 --ncpus 28
run /nobackupp19/aachary2/ayan_codes/foggie/foggie/ayan_acharyya_codes/submit_jobs.py --call compute_Zscatter --system ayan_pleiades --do_all_halos --queue e_normal --prefix cmzs --opt_args "--do_all_sims --use_onlyDD --upto_kpc 10 --nbins 100 --write_file --forpaper" --nnodes 2 --ncpus 28
```

For Fig 1:

```jsx
run projection_plot.py --system ayan_local --halo 2392 --upto_kpc 10 --output DD2335 --do metal --forpaper --proj z
run compute_MZgrad.py --system ayan_local --halo 2392 --output DD2335 --upto_kpc 10 --xcol rad --noweight_forfit --forpaper
run compute_Zscatter.py --system ayan_local --halo 2392 --output DD2335 --upto_kpc 10 --nbins 100 --forpaper --ymax 7.5
```

For Fig 2 (projected vs 3D metallicity comparison plot):

```jsx
run projected_vs_3d_metallicity.py --system ayan_local --halo 2392 --Zgrad_den kpc --upto_kpc 10 --forpaper --output DD2335 --noweight_forfit --clobber_plot
```

For two panels of Fig 3 (redshift evolution):

```bash
run plot_MZgrad.py --system ayan_local --halo 8508,5036,5016,4123,2878,2392 --Zgrad_den kpc --upto_kpc 10 --ycol Zgrad --xcol redshift --overplot_observations --forpaper --xmax 4 --xmin 0.5
run plot_MZgrad.py --system ayan_local --halo 8508,5036,5016,4123,2878,2392 --Zgrad_den kpc --upto_kpc 10 --ycol Zgrad --xcol redshift --overplot_observations --forpaper --xmax 4 --xmin 0.5 --overplot_theory
```

For Fig 4 top panel (time fraction):

```bash
run plot_MZgrad.py --system ayan_local --halo 2392 --Zgrad_den kpc --upto_kpc 10 --weight mass --ycol Zgrad --xcol time --zhighlight --plot_timefraction --Zgrad_allowance 0.03 --upto_z 1 --overplot_cadence 500 --keep --snaphighlight DD0526,DD0535 --forpaper --xmin 1.5
```

For Fig 4 bottom panels:

```bash
run nonprojected_Zgrad_hist_map.py --system ayan_pleiades --halo 2392 --upto_kpc 10 --forpaper --vcol vlos --output DD0526,DD0535 --nofit --nbins 20 --clobber_plot
```

For Fig 5 (time series):

```bash
run plot_Zevolution.py --system ayan_local --halo 2392 --upto_kpc 10 --forpaper --clobber
```

For Fig 6 (Z profile gradient and Z histogram):
(to make narrow figures for insets in Fig 4, use the option `--narrowfig` with below)

```bash
run nonprojected_Zgrad_hist_map.py --system ayan_pleiades --halo 2392 --upto_kpc 10 --forpaper --vcol vlos --output DD0445 --nofit --nbins 20 --clobber_plot
run projected_Zgrad_hist_map.py --system ayan_pleiades --halo 2392 --upto_kpc 10 --forpaper --vcol vlos --output DD0445 --res_arc 0.1 --nofit --nbins 20 --proj y --clobber_plot
```

For making the master table as well as smaller latex table:

```jsx
run make_table_fitted_quant.py --system ayan_local --halo 8508,5036,5016,4123,2878,2392 --Zgrad_den kpc --upto_kpc 10 --forpaper
```

For appendix plots (for each halo; then replace halo ID to get other halos):

```bash
run plot_MZgrad.py --system ayan_local --halo 8508 --Zgrad_den kpc --upto_kpc 10 --weight mass --ycol Zgrad --xcol time --zhighlight --plot_timefraction --Zgrad_allowance 0.03 --upto_z 1 --overplot_cadence 500 --keep --forpaper --xmin 1.5
run plot_Zevolution.py --system ayan_local --halo 8508 --upto_kpc 10 --forpaper --clobber
```

For appendix plot for snapshots (for each halo):

```bash
run projection_plot.py --system ayan_pleiades --halo 8508 --upto_kpc 10 --output RD0020,RD0027,RD0042 --do metal --forpaper --proj y --forappendix
run compute_MZgrad.py --system ayan_pleiades --halo 8508 --output RD0020,RD0027,RD0042 --upto_kpc 10 --xcol rad --forpaper --forappendix
run compute_Zscatter.py --system ayan_pleiades --halo 8508 --output RD0020,RD0027,RD0042 --upto_kpc 10 --nbins 100 --forpaper --forappendix
```

### Codes used for talk plots:

- Same as paper plots but with `--fortalk` option instead of `--forpaper`