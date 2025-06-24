DIRECTORY: `galaxy_mocks/mock_hi_imager`
AUTHOR: Cameron Trapp
DATE STARTED: 06/24/2025
LAST UPDATED: 06/24/2025

This directory contains a set of python and cython scripts to create a ideal and mock HI datacubes from the FOGGIE simulations.

To use for the first time you may need to run 'python setup.py build_ext --inplace' to compile the cython code.

This code currently only works by running make_hi_datacube.py from the command line (See below for usage example).

make_hi_datacube.py calls functions from a few different files to make an ideal, noiseless HI cube (make_ideal_datacubes.py), add noise and spatial smoothing (make_mock_datacubes.py), and spatially filter the the datacubes to simulate the effects of missing short baselines in interferometric studies (make_spatially_filtered_datacubes.py).

For creating the ideal datacube, each cell in the refine_box has it's corresponding emission profile (Voigt profile) calculated based on it's doppler velocity and is then projected onto the output ifu based on the viewing angle. Hubble flow is accounted for, however, we assume the gas is optically thin and neglect self shielding.

For creating the mock datacube, we start by adding a Gaussian noise profile with a standard deviation equal the survey sensitivity divided by args.sigma_noise_level. For example, if a survey has a claimed sensitivity of 1e18 cm-2, and you want a 5 sigma detection to correspond to that, the noise stdv would be set to 0.2e18 cm-2. From here, the ideal cube and separate noise cube is smoothed with a gaussian filter, with the FWHM equal to the survey beam size. The standard deviation of the noise cube is renormalied as needed.

Finally, for creating the spatially filtered datacube, we take the mock cube, fourier transform it, and apply a high-pass filter to filter out the spatially diffuse components. There are a few different filtering options in the code, but the Gaussian HPF seems to work the best. This is that FFT'd back into image space to create the dirty image. Unfortunatley, most survey parameters have a very tight cut in the center of the UV plane, meaning signficiant filtering artifacts are still introduced regardless of the filter type. To remove these, we deconvolve the dirty beam with the CLEAN algorithm. The cleaning follows the basic Hogbom 1984 algorithm, and identifies the brightest point in the image, convolves it with the dirty beam (calculated from a point source), and removes it from the image. The final (clean) image is comprised of the sum of these clean components (convolved with a clean beam, which is a Gaussian with FWHM equal to the survey beam size) and the remaining residuals in the dirty image.

Below are basic usage examples, a full list of IO arguments, and a list of files in this directory.

    
Basic Example usage:
python make_hi_datacube.py --output outputs/Tempest_MhongooseHR_10Mpc_i40 --z .002 --halo 008508 --mock_suffix _NHI5e18_RD0042 --memory_chunks 1000 --survey MHONGOOSE_HR  --min_column_density 5e18   --high_pass_filter_type gaussian  --nthreads 15 --clean_gain 0.1 --set_res_auto 1 --fov_kpc 350 --inclination 40  


The args are parsed as follows:    
IO Arguments:
    --code_dir: Where is the foggie analysis directory?
    --data_dir: Where are the simulation outputs?
    --refinement_level: To which refinement_level should the uniform covering grid be made. Defaults to the maximum refinement level in the box.
    --halo: Which halo should be analyzed. Default is 008508 (Tempest)
    --snapshot: Which snapshot should be analyzed? Default is RD0042
    --run: What refinement run should be analyzed? Default is nref11c_nref9f  

    --output: Where should the clump data be written? Default is ./output/clump_test
    --mock_suffix: Suffix to append to the non-ideal datacubes. Default is None.

    --system: Set the system to get data paths from get_run_loc_etc if not None Overrides --code_dir and --data_dir. Default is None. 
    --pwd: Use pwd arguments in get_run_loc_etc. Default is False.
    --forcepath: Use forcepath in get_run_loc_etc. Default is False.

    --force_ideal_ifu: By default, the code will try to reload an ideal ifu if it exists (as multiple surveys may use the same ideal ifu). This forces it to regenerate that ifu. Default is False.

    --nthreads: How many threads to use. Defaults to what is available on your system.
    
Survey Arguments:

    --survey: Which HI survey are you targeting. Choose from MHONGOOSE_HR, MHONGOOSE_LR, THINGS, FAST. Custom surveys can be modified from this by changing other arguments or by setting manually. Default is None.

    --obs_freq_range: What is the freq range you are analyzing in microns. Set by survey.
    --obs_spec_res: What is the observed spectral resolution in km/s. Set by survey.
    --obs_spatial_res: What is the observed spatial resolution in arcseconds. Set by survey.
    --primary_beam_FWHM_deg: What is the FWHM of the primary beam. Set by survey.

    --base_spatial_res: What is your pixel resolution? Can be set to automatically oversample the psf by set_res_auto.
    --fov_kpc: What you want the physical fov to be in kpc? Can be set automatically with set_fov_auto
    --set_res_auto: Set pixel resolution to oversample the psf by a factor of 4
    --set_fov_auto: Set the FOV to oversample the missing small spatial frequencies by a factor of 4

    --min_column_density: What is the minimum column density sensitivity you want to simulate?
    --log_min_column_density: Use to set the min column density in log instead.
    --sigma_noise_level: This sets the actual noise level as follows. Noise stdv = min_column_density / sigma_noise_level. Default is 5 (i.e. a 5 sigma detection corresponds to the min_column_density).



    --inclination: What inclination do you want to observe the galaxy at in degrees?
    --position_angle: What position angle do you want to observe the galaxy at in degrees.

    --z: What is the observed redshift? Currently used to set distance...

Spatial Filtering and CLEAN Arguments:
    --high_pass_filter_type: What type of spatial filter do you want to use? (Gaussian or Butterworth). Default is None, which corresponds to an ideal, sharp filter.
    --clean_gain: By what factor should your clean components be multiplied by when removing them. Default is 0.1. Must be less than 1.
    --max_clean_iterations: Do you want to set a maximum amount of cleaning iterations? 
    --min_baseline: What is the shortest baseline you want to simulate  in meters. (e.g. 29 m, 35 m). This sets the width of the high pass filter.
    --use_clean_mask: Do you only want to clean pixels where the ideal image has pixels above your sensitivity limit (Probably should not use).
    --clean_sigma: To what noise level do you want to clean to? Default is 3 (i.e. 3 sigma).
    

Alternate Use Cases:
    --make_simple_moment_maps: Optional module to just make simple projections instead of full datacubes.
    --skip_full_datacube: Force skipping the full datacube. Probably don't need to ever use this
    --skip_mock_datacube: Skip making the mock datacubes.
    --skip_spatial_filtering: Skip filtering spatially (can also be set by setting --min_baseline to 0)
    --mask_disk: Mask the disk out (using the clump finders disk definition) for your synthetic images.
    --test_calibartor_clean: Make additional figures testing the CLEAN algorithm on a point source.
    --make_disk_cut_mask: Make a projected mask of the disk for the ideal and smoothed datacubes. (Useful for masking out the disk in projected images).

  

| Folder/Module        | Description |
|----------------------|-------------|
| `make_hi_datacube.py` | Contains the main functions for runing the synethtic image generator. Call this to run the pipeline.|
| `make_ideal_datacubes.py` | Contains functions to create the ideal datacubes. |
| `make_mock_datacubes.py` | Contains functions to create the mock datacubes, as well as calling make_spatially_filtered_datacubes.py |
| `make_spatially_filtered_datacubes.py` | Contains functions to to spatially filter the datacubes and run the CLEAN algorithm. |
| `custom_clean.pyx` | Cython code to quickly run the CLEAN algorithm. |
| `hi_datacube_arg_parser.py` | Handles the input arguments for running the image generator. |
| `HICubeHeader.py` | Header file with constants. |
| `line_properties.py` | Defines the HI emission and associated constants. |
| `radio_telescopes.py` | Contains classes for various survey parameters. Defines things such as sensitivity, resolution, fov, etc. |
| `utils_hi_datacube.py` | Various utility functions and yt field definitions. |
| `convert_to_fits.py` | Converts the .hdf5 files generated by this pipeline to fits images. Called by RunSofia.sh. |
| `RunSofia.sh` | Bash script to that calls convert_to_fits.py on the output .hdf5 files and runs the SOFIA source extractor to identify signficant signals in the output datacubes. Requires SOFIA to be installed|
| `setup.py` | Code used to compile the cython code into custom_clean.c. Run as 'python setup.py build_ext --inplace'. |
| `README.md` | Me. |