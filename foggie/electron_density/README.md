DIRECTORY: `electron_density`
AUTHOR: Ayan Acharyya
LAST UPDATED: 12/12/2024

This directory is meant to contain a set of python scripts to analyse electron density profile,s projections etc. in the FOGGIE simulations.
These scripts were triggered by the FOGGIE-Curtin collaboration.
Each script within this directory has a detailed description of what it does, along with examples, but here is a brief list (in alphabetical order).
[For gas phase metallicity analysis scripts by Ayan Acharyya, see the `gas_metallicity` directory.]

| Folder/Module        | Description |
|----------------------|-------------|
| `header.py` | Imports packages/modules required for working with scripts in this directory. |
| `util.py` | Loads some common utility functions that may be invoked by scripts in this directory |
| `electron_density_projections.py` | Plots projected gas density, electron density, etc. |
| `electron_density_spherical.py` | Plots full 3D gas density, electron density, etc. |
| `make_3D_FRB_electron_density.py` | Makes Fixed Resolution Buffer of gas and electron density |
| `make_himass_df.py` | Makes list of HI mass within a certain distance from the center |
| `plot_projected_electron_density.py` | Plots parameterised projected electron density profile as a function of SFR and stellar mass |
| `plot_spherical_electron_density.py` | Plost parameterised 3D electron density profile as a function of SFR and stellar mass

