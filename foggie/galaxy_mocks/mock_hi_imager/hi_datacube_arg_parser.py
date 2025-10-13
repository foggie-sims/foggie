import argparse

def parse_args():
    '''Parse command line arguments. Returns args object.'''
    parser = argparse.ArgumentParser()
    # Optional arguments:
    parser.add_argument('--halo', metavar='halo', type=str, action='store', \
                        help='Which halo id? Default is 008508 (Tempest).')
    parser.set_defaults(halo="008508")
    
    parser.add_argument('--snapshot', metavar='snapshot', type=str, action='store', \
                        help='Which snapshot? Default is RD0042 (redshift 0).')
    parser.set_defaults(snapshot="RD0042")
    
    parser.add_argument('--run', metavar='run', type=str, action='store', \
                        help='Which run? Default is nref11c_nref9f.')
    parser.set_defaults(run="nref11c_nref9f")

    parser.add_argument('--survey', metavar='survey', type=str, action='store', \
                        help='What instrument/survey do you want to simulate? Default is THINGS')
    parser.set_defaults(survey="THINGS")

    parser.add_argument('--obs_freq_range', metavar='obs_freq_range', type=float, action='store', \
                        help='Observed wavelength range in microns. Will be overwritten if survey is not None. Used for manually defining telescopes only.')
    parser.set_defaults(obs_freq_range=None)

    parser.add_argument('--obs_spec_res', metavar='obs_spec_res', type=float, action='store', \
                        help='Observed spectral resolution in km/s. Will be overwritten if survey is not None. Used for manually defining telescopes only.')
    parser.set_defaults(obs_spec_res=None)

    parser.add_argument('--obs_spatial_res', metavar='obs_spatial_res', type=float, action='store', \
                        help='Observed spatial resolution in arcseconds. Will be overwritten if survey is not None. Used for manually defining telescopes only.')
    parser.set_defaults(obs_spatial_res=None)

    parser.add_argument('--base_freq_range', metavar='base_freq_range', type=float, action='store', \
                        help='Base wavelength range in microns')
    parser.set_defaults(base_freq_range=None) #THIS SHOULD BE NONE, REMOVE ARGUMENT

    parser.add_argument('--base_spatial_res', metavar='base_spatial_res', type=float, action='store', \
                        help='Base pixel resolution in arcseconds. Recommend to use set_res_auto instead of manually setting this.')
    parser.set_defaults(base_spatial_res=None)

    parser.add_argument('--base_channels', metavar='base_channels', type=float, action='store', \
                        help='Base number of spectral channels')
    parser.set_defaults(base_channels=None) #THIS NEEDS TO BE NONE, REMOVE ARGUMENT

    parser.add_argument('--inclination', metavar='inclination', type=float, action='store', \
                        help='Observation inclination in degrees. Default is 45.')
    parser.set_defaults(inclination=45)

    parser.add_argument('--position_angle', metavar='position_angle', type=float, action='store', \
                        help='Observation position angle in degrees. Default is 0')
    parser.set_defaults(position_angle=0)

    parser.add_argument('--z', metavar='z', type=float, action='store', \
                        help='Redshift. Use to set distance. Default is 0.002.')
    parser.set_defaults(z=0.002)

    parser.add_argument('--fov_kpc', metavar='fov_kpc', type=float, action='store', \
                        help='Dimensions of image in kpc. Default is 250.')
    parser.set_defaults(fov_kpc=250)    

    parser.add_argument('--set_fov_auto', metavar='set_fov_auto', type=bool, action='store', \
                        help='Set the fov automatically such that the minimum spatial resolution is resolved in the UV plane by a factor of 5. Default is False.')
    parser.set_defaults(set_fov_auto=False)    

    parser.add_argument('--set_res_auto', metavar='set_res_auto', type=bool, action='store', \
                        help='Set the pixel resolution automatically such that the maximum spatial resolution is resolved in the image plane by a factor of 4. Default is False.')
    parser.set_defaults(set_res_auto=True)  

    parser.add_argument('--system', metavar='system', type=str, action='store', \
                        help='What system are you running on?')
    parser.set_defaults(system='cameron_local')

    parser.add_argument('--data_dir', metavar='data_dir', type=str, action='store', \
                        help='Where is the data_dir? Will overwrite system if not None.')
    parser.set_defaults(data_dir=None)

    parser.add_argument('--code_dir', metavar='code_dir', type=str, action='store', \
                        help='Where is the code_dir? Will overwrite system if not None.')
    parser.set_defaults(code_dir=None)

    parser.add_argument('--pwd', metavar='pwd', type=bool, action='store', \
                        help='Use pwd arguments in get_run_loc_etc. Default is False.')
    parser.set_defaults(pwd=False) 

    parser.add_argument('--forcepath', metavar='forcepath', type=bool, action='store', \
                        help='Use forcepath in get_run_loc_etc. Default is False.')
    parser.set_defaults(forcepath=False) 

    parser.add_argument('--nthreads', metavar='nthreads', type=int, action='store', \
                        help='How many threads to use? Defaults to available. Set to 1 to not parallelize')
    parser.set_defaults(nthreads=None) 

    parser.add_argument('--make_simple_moment_maps', metavar='make_simple_moment_maps', type=bool, action='store', \
                        help='Do you want to create and save simple moment maps for these projections? Default is False.')
    parser.set_defaults(make_simple_moment_maps=False) #Make this more intuitive...

    parser.add_argument('--skip_full_datacube', metavar='skip_full_datacube', type=bool, action='store', \
                        help='Do you want to skip making the full datacube? Default is False.')
    parser.set_defaults(skip_full_datacube=False) #Do you want to do this?

    parser.add_argument('--skip_mock_datacube', metavar='skip_mock_datacube', type=bool, action='store', \
                        help='Do you want to skip making the mock datacube? Default is False.')
    parser.set_defaults(skip_mock_datacube=False) #Do you want to actually do this?

    parser.add_argument('--output', metavar='output', type=str, action='store', \
                        help='File for ideal datacube to load or write to. Will load if possible.')
    parser.set_defaults(output=None) 

    parser.add_argument('--mock_suffix', metavar='mock_suffix', type=str, action='store', \
                        help='Suffix to append to non-ideal datacube. Default is None.')
    parser.set_defaults(mock_suffix=None) 

    parser.add_argument('--high_pass_filter_type', metavar='high_pass_filter_type', type=str, action='store', \
                        help='Type of high pass filter to apply to datacube. Options are "gaussian" or "butterworth". Default is a simple cut (infintely sharp filter).')
    parser.set_defaults(high_pass_filter_type=None) 

    parser.add_argument('--clean_gain', metavar='clean_gain', type=float, action='store', \
                        help='What is the loop gain for the clean algorithm? Recommended 0.1-0.5. Default is 0.1')
    parser.set_defaults(clean_gain=0.1) 

    parser.add_argument('--max_clean_iterations', metavar='max_clean_iterations', type=float, action='store', \
                        help='What is the maximum number of iterations for the clean algorithm? Default is the number of pixels in the image.')
    parser.set_defaults(max_clean_iterations=None) 

    parser.add_argument('--min_column_density', metavar='min_column_density', type=float, action='store', \
                        help='What is the minimum column density the noise model should be sensitive to? Default is 0.')
    parser.set_defaults(min_column_density=0) 

    parser.add_argument('--log_min_column_density', metavar='log_min_column_density', type=float, action='store', \
                        help='What is the log of the minimum column density the noise model should be sensitive to? Default is 0. Will override min_column_density')
    parser.set_defaults(log_min_column_density=0) 

    parser.add_argument('--skip_spatial_filtering', metavar='skip_spatial_filtering', type=bool, action='store', \
                        help='Do you want to skip spatial filtering? Default is False')
    parser.set_defaults(skip_spatial_filtering=False) 

    parser.add_argument('--min_baseline', metavar='min_baseline', type=float, action='store', \
                        help='What is the minimum baseline you are simulating (in km)? Will override the survey if not None. Default is None.')
    parser.set_defaults(min_baseline=None) 

    parser.add_argument('--mask_disk', metavar='mask_disk', type=bool, action='store', \
                        help='Mask the disk from your analysis? Default is False')
    parser.set_defaults(mask_disk=False) 

    parser.add_argument('--use_clean_mask', metavar='use_clean_mask', type=bool, action='store', \
                        help="Use a clean mask to mask the data cube? Default is False")
    parser.set_defaults(use_clean_mask=False) 

    parser.add_argument('--sigma_noise_level', metavar='sigma_noise_level', type=float, action='store', \
                        help="What sigma level to count as detection (used to set noise level with column density sensitivity)? Default is 5")
    parser.set_defaults(sigma_noise_level=5) 

    parser.add_argument('--test_calibrator_clean', metavar='test_calibrator_clean', type=bool, action='store', \
                        help="Do you want to test the cleaning on a point source? Default is False.")
    parser.set_defaults(test_calibrator_clean=False) 

    parser.add_argument('--freq_range_mult', metavar='freq_range_mult', type=float, action='store', \
                        help="Do you want to pad the bandwidth by a factor to account for abnormally high rotation? Default is 1 (no padding)")
    parser.set_defaults(freq_range_mult=1) ##This currently just gets set automatically....

    parser.add_argument('--force_ideal_ifu', metavar='force_ideal_ifu', type=bool, action='store', \
                        help="Do you want to regenerate the ideal ifu even if it already exists? Default is False.")
    parser.set_defaults(force_ideal_ifu=False) 


    parser.add_argument('--memory_chunks', metavar='memory_chunks', type=int, action='store', \
                        help="Do you calculate the datacube in chunks to limit memory usage? Default is 0 (no chunking).")
    parser.set_defaults(memory_chunks=0) 

    parser.add_argument('--primary_beam_FWHM_deg', metavar='primary_beam_FWHM_deg', type=float, action='store', \
                        help="What is the FWHM of the primary beam in degrees? Default is instrument dependent.")
    parser.set_defaults(primary_beam_FWHM_deg=None) 

    parser.add_argument('--max_projected_velocity', metavar='max_projected_velocity', type=float, action='store', \
                        help="Alternative way to set the bandwidth. Defaults to the maximum for all gas in fov above a certain column density.")
    parser.set_defaults(max_projected_velocity=None) 

    parser.add_argument('--make_disk_cut_mask', metavar='make_disk_cut_mask', type=bool, action='store', \
                        help="Do you want to make a mask for the disk projection? Default is False")
    parser.set_defaults(make_disk_cut_mask=False) 

    parser.add_argument('--clean_sigma', metavar='clean_sigma', type=float, action='store', \
                        help="To what noise level do you want to clean to? Default is 3 sigma.")
    parser.set_defaults(clean_sigma=3) 



    args = parser.parse_args()
    return args
