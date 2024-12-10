# foggie

This repository contains code for analyzing the FOGGIE simulations.
It may be installed as a package using e.g. `pip install .` for 
easier access to imports such as `foggie.utils.consistency`
or you may directly use the scripts within.

## Documentation

The `doc` directory contains documentation on both the FOGGIE simulations
and how to use scripts within this repository. The docs can be built as HTML using:
```
cd doc
make html
```

## Analysis Modules

The `foggie` directory contains several subdirectories
with scripts from various FOGGIE analysis projects. A description of their contents is included below.

| Folder/Module        | Description |
|----------------------|-------------|
| `angular_momentum`   | Characterizing galaxy angular momentum. Scripts used for [Simons et al. 2020](https://iopscience.iop.org/article/10.3847/1538-4357/abc5b8). |
| `cgm_absorption`     | CGM absorption specctra generation and analysis. Scripts used for [Peeples et al. 2019](https://iopscience.iop.org/article/10.3847/1538-4357/ab0654). |
| `cgm_emission`       | Scripts for post-processing CGM emission using [CLOUDY](https://gitlab.nublado.org/cloudy/cloudy). |
| `clumps`             | Finding & analyzing clumps. Scripts used for Augustin et al. in prep.|
| `deprecated`         | Deprecated scripts. |
| `edges`              | Basic code for edge analysis of FOGGIE regions. |
| `examples`           |  |
| `flux_tracking`      | Scripts and Notebooks for estimating fluxes such as accretion. |
| `fogghorn`           | Diagnostic plot pipeline for planning new simulations. |
| `galaxy_mocks`       | Scripts for making mock observations. |
| `halo_infos`         | Catalog files; see accompanying README. |
| `halo_tracks`        | Halo tracks used for running the FOGGIE simulations. |
| `initial_conditions` | Old initial conditions for a 25 Mpc simulation box. |
| `interns`            | Previous student intern projects. |
| `movies`             | Scripts for making movies. |
| `notebooks`          | A folder of Jupyter Notebooks to contain `git` change tracking chaos. |
| `paper_plots`        | Scripts for making plots in the various FOGGIE publications. |
| `plots`              | Generate basic sanity-check plots. |
| `pressure_support`   | Thermal and non-thermal pressure estimations. Scripts used for [Lochhaas et al. 2023](https://iopscience.iop.org/article/10.3847/1538-4357/acbb06). |
| `radial_quantities`  | Radial profiles of common quantities. |
| `render`             | Scripts using the `datashader` package. |
| `satellites`         |  |
| `scripts`            | A random assortment of things. |
| `turbulence`         | Calculate velocity structure functions and velocity dispersions. |
| `utils`              | Utility scripts. |
