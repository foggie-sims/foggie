DIRECTORY: `galaxy_mocks/mock_ifu`
AUTHOR: Ayan Acharyya
LAST UPDATED: 12/12/2024

This directory is meant to contain a set of python scripts to produce and analyse mock IFU datacubes from FOGGIE simulations.
Each script within this directory has a detailed description of what it does, along with examples, but here is a brief list (in alphabetical order).
[For gas phase metallicity analysis scripts by Ayan Acharyya, see the `gas_metallicity` directory.]

| Folder/Module        | Description |
|----------------------|-------------|
| `header.py` | Imports packages/modules required for working with scripts in this directory. |
| `util.py` | Loads some common utility functions that may be invoked by scripts in this directory |
| `compute_hiir_radii.py` | Models HII regions, based on their age, driving stellar source, and ambient gas properties, and then computes the HII region radius |
| `filter_star_properties.py` | Selects young stars from FOGGIE snapshots and grabs some attributes of these stars |
| `fit_mock_spectra.py` | Fits the 1D (synthetic) spectra along each pixel (LoS) of a datacube |
| `investigate_metallicity.py` | Test script to investigate why and how metallicities were getting too high after co-adding modeled HII regions along LoS |
| `lookup_flux.py` | Looks up emission line fluxes from a 4D MAPPINGS phototionisation model grid |
| `make_ideal_datacube.py` | Produces the ideal (without noise or smearing) IFU datacube |
| `make_mappings_grid.py` | Makes the 4D MAPPINGS photoionisation model grid, based on input Starburst99 stellar spectra (need to have MAPPINGs installed as an executable to run this script) |
| `make_mock_datacube.py` | Makes the synthetic (with noise and PSF smearing) IFU datacube |
| `make_mock_measurements.py` | Derives physical quantities from a given measured datacube (that contains emission line fluxes) |
| `plot_mock_observables.py` | Plots various quantities measured from the synthetic datacube |
| `test_kit.py` | Test functions for debugging, particularly how/what emission line fluxes are being assigned to each modeled HII region |
| starburst11` | Contains the Starburst99 model used to produce the MAPPINGS grid of models (the number 11 is just an identifier, specified in foggie.gas_metallicity.header.py). Provided here for reproducibility. |
| MAPPINGS` | Contains the MAPPINGS grid of models. Provided here for reproducibility.