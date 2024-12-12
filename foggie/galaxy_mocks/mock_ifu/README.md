DIRECTORY: `galaxy_mocks/mock_ifu`
AUTHOR: Ayan Acharyya
LAST UPDATED: 12/12/2024

This directory is meant to contain a set of python scripts to produce and analyse mock IFU datacubes from FOGGIE simulations.
Each script within this directory has a detailed description of what it does, along with examples, but here is a brief list (in alphabetical order).
[For gas phase metallicity analysis scripts by Ayan Acharyya, see the `gas_metallicity` directory.]

File Name: `header.py`
Description: Imports packages/modules required for working with scripts in this directory.

File Name: `util.py`
Description: Loads some common utility functions that may be invoked by scripts in this directory

File Name: `compute_hiir_radii.py`
Description: Models HII regions, based on their age, driving stellar source, and ambient gas properties, and then computes the HII region radius

File Name: `filter_star_properties.py`
Description: Selects young stars from FOGGIE snapshots and grabs some attributes of these stars

File Name: `fit_mock_spectra.py`
Description: Fits the 1D (synthetic) spectra along each pixel (LoS) of a datacube

File Name: `investigate_metallicity.py`
Description: Test script to investigate why and how metallicities were getting too high after co-adding modeled HII regions along LoS

File Name: `lookup_flux.py`
Description: Looks up emission line fluxes from a 4D MAPPINGS phototionisation model grid

File Name: `make_ideal_datacube.py`
Description: Produces the ideal (without noise or smearing) IFU datacube

File Name: `make_mappings_grid.py`
Description: Makes the 4D MAPPINGS photoionisation model grid, based on input Starburst99 stellar spectra (need to have MAPPINGs installed as an executable to run this script)

File Name: `make_mock_datacube.py`
Description: Makes the synthetic (with noise and PSF smearing) IFU datacube

File Name: `make_mock_measurements.py`
Description: Derives physical quantities from a given measured datacube (that contains emission line fluxes)

File Name: `plot_mock_observables.py`
Description: Plots various quantities measured from the synthetic datacube

File Name: `test_kit.py`
Description: Test functions for debugging, particularly how/what emission line fluxes are being assigned to each modeled HII region

Folder Name: starburst11`
Description: Contains the Starburst99 model used to produce the MAPPINGS grid of models (the number 11 is just an identifier, specified in foggie.gas_metallicity.header.py). Provided here for reproducibility.

Folder Name: MAPPINGS`
Description: Contains the MAPPINGS grid of models. Provided here for reproducibility.