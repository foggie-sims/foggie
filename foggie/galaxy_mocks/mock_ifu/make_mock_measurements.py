##!/usr/bin/env python3

"""

    Title :      make_mock_measurements
    Notes :      Takes a measured datacube (which has flux, velocity and velocity dispersion maps) and derives physical quantities e.g., metallicity maps, etc
    Output :     FITS cube with 2D map of desired parameter
    Author :     Ayan Acharyya
    Started :    May 2021
    Example :    run make_mock_measurements.py --system ayan_local --halo 5036 --output RD0020 --mergeHII 0.04 --galrad 6 --base_spatial_res 0.04 --z 0.25 --obs_wave_range 0.8,0.85 --obs_spatial_res 0.1 --obs_spec_res 60 --exptime 1200 --snr 5 --compute_property metallicity --write_property --plot_property

"""
from header import *
from util import *
from fit_mock_spectra import fit_mock_spectra

# -----------------------------------------------------------------------------
def wrap_write_fitsobj(data, data_u, property_name, measured_cube, args):
    '''
    Wrapper function to organise 2D property map in to something that can be passed on to write_fitsobj() to be written to file
    '''
    linelist_dummy = read_linelist(mappings_lab_dir + 'targetlines.txt')  # list of emission lines
    instrument = telescope(args)  # declare the instrument
    if args.snr == 0: new_measured_cube = idealcube(args, instrument, linelist_dummy)  # declare a cube object
    else: new_measured_cube = noisycube(args, instrument, linelist_dummy)  # declare the noisy mock datacube object

    new_measured_cube.data = np.dstack([measured_cube.data, data]) # appending newly derived property to existing measured cube
    new_measured_cube.error = np.dstack([measured_cube.error, data_u])

    myprint('Adding ' + property_name + ' map and re-writing measured cube..', args)
    write_fitsobj(args.measured_cube_filename, new_measured_cube, instrument, args, for_qfits=True, measured_cube=True, measured_quantities=measured_cube.measured_quantities, derived_quantities=[property_name])  # writing into FITS file

# -----------------------------------------------------------------------------
def get_metallicity(measured_cube, args):
    '''
    Function to derive metallicity maps
    '''
    fluxes, fluxes_u = measured_cube.get_all_lines('flux')
    flux_dict = {}

    for thisline in measured_cube.fitted_lines:
        thisline_index = np.where(measured_cube.fitted_lines == thisline)[0][0]
        flux_dict[thisline] = fluxes[:, :, thisline_index]

    metallicity = get_D16_metallicity(flux_dict)
    #metallicity = get_PPN2_metallicity(flux_dict)
    metallicity_u = np.zeros(np.shape(metallicity)) # placeholder for future brainwave

    if args.write_property:
        wrap_write_fitsobj(metallicity, metallicity_u, 'metallicity', measured_cube, args)
        measured_cube = read_measured_cube(args.measured_cube_filename, args)
        metallicity, metallicity_u = measured_cube.get_derived_prop('metallicity')

    return metallicity, metallicity_u

# ----------------------------------------------------------------------------
def compute_properties(measured_cube, args):
    '''
    Wrapper function to call the appropriate function to compute a given property
    :return: 2D map of property, along with associated uncertainty map
    '''
    if args.compute_property == 'metallicity': property, property_u = get_metallicity(measured_cube, args)

    return property, property_u

# -----------------------------------------------------------------------------
if __name__ == '__main__':
    start_time = time.time()

    args = parse_args('8508', 'RD0042')
    if type(args) is tuple: args = args[0] # if the sim has already been loaded in, in order to compute the box center (via utils.pull_halo_center()), then no need to do it again
    if not args.keep: plt.close('all')

    if args.doideal: fitted_file = args.idealcube_filename
    elif args.snr == 0: fitted_file = args.smoothed_cube_filename
    else: fitted_file = args.mockcube_filename
    measured_cube_filename = get_measured_cube_filename(fitted_file)
    if not os.path.exists(measured_cube_filename):
        myprint('measured_cube does not exist, so calling fit_mock_spectra.py to create one..', args)
        measured_cube_dummy = fit_mock_spectra(args)

    measured_cube = read_measured_cube(measured_cube_filename, args)
    property, property_u = compute_properties(measured_cube, args)

    print('Complete in %s minutes' % ((time.time() - start_time) / 60))
