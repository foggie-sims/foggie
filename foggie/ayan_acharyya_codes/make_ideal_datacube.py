##!/usr/bin/env python3

"""

    Title :      make_ideal_datacube
    Notes :      Produces an idealised IFU datacube by projecting a list of HII region emissions on to a 2D grid, for a given inc & PA, and a base (high) spatial + spectral resolution
    Output :     FITS cube
    Author :     Ayan Acharyya
    Started :    February 2021
    Example :    run make_ideal_datacube.py --system ayan_local --halo 5036 --output RD0030 --mergeHII 0.04 --base_spatial_res 0.4 --z 0.25 --base_wave_range 0.64,0.68 --projection z --obs_wave_range 0.8,0.85 --instrument dummy

"""
from header import *
from util import *
from compute_hiir_radii import *
from filter_star_properties import get_star_properties
import make_mappings_grid as mmg

# -------------------------------------------------------------------------------------------------
def get_erf(lambda_array, height, centre, width, delta_lambda):
    '''
    Integral of a Gaussian function; used by gauss(), to 'stick' emission lines on top of stellar spectra
    '''
    return np.sqrt(np.pi / 2) * height * width * (erf((centre + delta_lambda / 2 - lambda_array) / (np.sqrt(2) * width)) - \
            erf((centre - delta_lambda / 2 - lambda_array) / (np.sqrt(2) * width))) / delta_lambda  # https://www.wolframalpha.com/input/?i=integrate+a*exp(-(x-b)%5E2%2F(2*c%5E2))*dx+from+(w-d%2F2)+to+(w%2Bd%2F2)

# -------------------------------------------------------------------------------------------------
def gauss(wave_arr, flux_arr, this_wave_cen, this_flux, vel_disp, vel_z):
    '''
    Gaussian function, to 'stick' an emission line at a given central wavelength with a given flux (area), on top of stellar spectra (flux_arr)
    '''
    this_wave_cen = this_wave_cen * (1 + vel_z / c)  # shift central wavelength wrt w0 due to LoS velocity (vel_z) of HII region as compared to systemic velocity
    sigma = this_wave_cen * vel_disp / c # converting velocity dispersion (km/s) to sigma (Angstrom)
    amplitude = this_flux / np.sqrt(2 * np.pi * sigma ** 2)  # height of Gaussian, such that area = this_flux
    delta_wave = wave_arr[np.where(wave_arr >= this_wave_cen)[0][0]] - wave_arr[np.where(wave_arr >= this_wave_cen)[0][0] - 1]

    # gaussian = amplitude * np.exp(-((wave_arr - this_wave_cen)**2)/(2 * sigma**2))
    gaussian = get_erf(wave_arr, amplitude, this_wave_cen, sigma, delta_wave) # compute what the gaussian should look like if the area has to be equal to the given flux

    flux_arr = flux_arr + gaussian
    return flux_arr

# -----------------------------------------------------------------------
def shift_ref_frame(paramlist, args):
    '''
    Function to shift spatial coordinates to a reference frame with respect to the halo center
    :param paramlist: Adds new columns to paramlist, corresponding to every spatial and velocity coordinate
    :return: paramlist
    '''

    paramlist['pos_x_cen'] = paramlist['pos_x'] - args.halo_center[0]  # in kpc
    paramlist['pos_y_cen'] = paramlist['pos_y'] - args.halo_center[1]  # in kpc
    paramlist['pos_z_cen'] = paramlist['pos_z'] - args.halo_center[2]  # in kpc

    paramlist['vel_x_cen'] = paramlist['vel_x'] - args.halo_velocity[0]  # in km/s
    paramlist['vel_y_cen'] = paramlist['vel_y'] - args.halo_velocity[1]  # in km/s
    paramlist['vel_z_cen'] = paramlist['vel_z'] - args.halo_velocity[2]  # in km/s

    return paramlist

# -------------------------------------------------------------------
def incline(paramlist, args):
    '''
    Function to incline i.e. assuming xy projection, rotate the galaxy in YZ plane (keeping X fixed) for a given angle of inclination
    :return: modified dataframe with new positions and velocities as new columns
    '''
    inc = float(args.inclination) * np.pi/180 # converting degrees to radians

    # now doing coordinate transformation to get new coordinates
    paramlist['pos_' + projection_dict[args.projection][1] + '_inc'] = paramlist['pos_' + projection_dict[args.projection][1] + '_cen'] * np.cos(inc) + paramlist['pos_' + args.projection + '_cen'] * np.sin(inc)
    paramlist['pos_' + args.projection + '_inc'] = -paramlist['pos_' + projection_dict[args.projection][1] + '_cen'] * np.sin(inc) + paramlist['pos_' + args.projection + '_cen'] * np.cos(inc)
    paramlist['pos_' + projection_dict[args.projection][0] + '_inc'] = paramlist['pos_' + projection_dict[args.projection][0] + '_cen']

    # now doing coordinate transformation to get new velocities
    paramlist['vel_' + projection_dict[args.projection][1] + '_inc'] = paramlist['vel_' + projection_dict[args.projection][1] + '_cen'] * np.cos(inc) + paramlist['vel_' + args.projection + '_cen'] * np.sin(inc)
    paramlist['vel_' + args.projection + '_inc'] = -paramlist['vel_' + projection_dict[args.projection][1] + '_cen'] * np.sin(inc) + paramlist['vel_' + args.projection + '_cen'] * np.cos(inc)
    paramlist['vel_' + projection_dict[args.projection][0] + '_inc'] = paramlist['vel_' + projection_dict[args.projection][0] + '_cen']

    return paramlist

# -----------------------------------------------------------------------
def get_grid_coord(paramlist, args):
    '''
    Function to convert physical coordinates to grid (pixel) coordinates
    :param paramlist: (1) Curtails paramlist to keep only those HII regions that are within 2 x args.galrad kpx box
                      (2) Adds new columns to paramlist, corresponding to every spatial coordinate
    :return: paramlist
    '''

    initial_length = len(paramlist)
    paramlist = paramlist[(paramlist['pos_x_cen'].between(-args.galrad, args.galrad)) & (paramlist['pos_y_cen'].between(-args.galrad, args.galrad)) & (paramlist['pos_z_cen'].between(-args.galrad, args.galrad))].reset_index(drop=True)  # Curtails paramlist to keep only those HII regions that are within 2 x args.galrad kpx box
    myprint(str(len(paramlist)) + ' HII regions remain out of ' + str(initial_length) + ', within the ' + str(2*args.galrad) + ' kpc^3 box', args)

    paramlist['pos_x_grid'] = ((paramlist['pos_x_inc'] + args.galrad)/args.base_spatial_res).astype(np.int)
    paramlist['pos_y_grid'] = ((paramlist['pos_y_inc'] + args.galrad)/args.base_spatial_res).astype(np.int)
    paramlist['pos_z_grid'] = ((paramlist['pos_z_inc'] + args.galrad)/args.base_spatial_res).astype(np.int)

    return paramlist

# -------------------------------------------------------------------------------------------------
def get_SB99continuum(wmin, wmax):
    '''
    Function to read in Starburst99 file, interpolate the continuum vs wavelength for a given stellar age
    :param wmin: starting wavelength to compute continuum
    :param wmax: ending wavelength to compute continuum
    :return: interpolated function, for every age
    '''

    SB_data = pd.read_table(sb99_dir + sb99_model + '/' + sb99_model + '.spectrum', delim_whitespace=True, comment='#', \
                            skiprows=5, header=None, names=('age', 'wavelength', 'log_total', 'log_stellar', 'log_nebular'))

    cont_interp_func_arr = [] # array for interpolation functions, interpolating the continuum vs wavelength for a given stellar age
    for thisage in pd.unique(SB_data['age']):
        SB_subdata = SB_data[SB_data['age'] == thisage] # taking a subset = spectra for only a given age
        cont_interp_func_arr.append(interp1d(SB_subdata['wavelength'], 10**SB_subdata['log_stellar'], kind='cubic'))

    return np.array(cont_interp_func_arr)

# ----------------------------------------------------------------------
def get_HII_list(args):
    '''
    Function to read in HII region parameter list
    :return: dataframe with list of HII region parameters
    '''
    emission_path = args.output_dir + 'txtfiles/' + args.output + '_emission_list' + '_' + args.diag + mmg.outtag
    emission_file = emission_path + '/emission_list_Om' + str(args.Om) + args.mergeHII_text + '.txt'
    if os.path.exists(emission_file):
        myprint('Reading HII region emission file ' + emission_file, args)
    else:
        myprint('HII region emission file does not exist, calling filter_star_properties.py..', args)
        args.automate = True  # so that it cascades to subsequent scripts as necessary
        clobber_holder = args.clobber # temporary variable to hold value of args.clobber
        args.clobber = False # clobber set to false before calling chronologically previous scripts
        paramlist = get_star_properties(args) # calling chronologically previous script
        args.clobber = clobber_holder # restore whatever was the original value of args.clobber
    paramlist = pd.read_table(emission_file, delim_whitespace=True, comment='#')  # reading in the list of HII region emission fluxes
    return paramlist

# -----------------------------------------------------------------------
def get_ideal_datacube(args, linelist):
    '''
    Function to produce ideal IFU datacube, for a given base spatial and spectral resolution;
    :param paramlist: Computes spectra (stellar continuum + emission lines) for each HII region (in paramlist), then adds all spectra along each LoS, binned spatially
    at a given spatial resolution, to produce a x - y - lambda datacube.
    :return: writes datacube in to FITS file
    '''
    start_time = time.time()

    paramlist = get_HII_list(args)
    paramlist = shift_ref_frame(paramlist, args)
    paramlist = incline(paramlist, args)  # coordinate transformation by args.inclination degrees in the YZ plane (keeping X fixed)
    paramlist = get_grid_coord(paramlist, args)

    instrument = telescope(args) # declare an instrument
    ifu = idealcube(args, instrument, linelist) # declare a cube object

    cube_output_path = get_cube_output_path(args)
    args.idealcube_filename = cube_output_path + 'ideal_ifu' + args.mergeHII_text + '.fits'

    if os.path.exists(args.idealcube_filename) and not args.clobber:
        myprint('Reading from already existing file ' + args.idealcube_filename + ', use --args.clobber to overwrite', args)
    else:
        myprint('Ideal cube file does not exist. Creating now..', args)
        cont_interp_func_arr = get_SB99continuum(ifu.rest_wave_range[0], ifu.rest_wave_range[1])

        # -------iterating over each HII region now, to compute LoS spectra (stellar continuum + nebular emission) for each-------------
        for index, HIIregion in paramlist.iterrows():
            myprint('Particle ' + str(index + 1) + ' of ' + str(len(paramlist)) + '..', args)

            age_rounded = int(round(HIIregion['age']))
            flux = np.multiply(cont_interp_func_arr[age_rounded](ifu.base_wave_arr), (HIIregion['mass'] / sb99_mass))  # to scale the continuum by HII region mass, as the ones produced by SB99 was for sb99_mass; ergs/s/A

            for line_index, thisline in ifu.linelist.iterrows():
                try:
                    flux = gauss(ifu.base_wave_arr, flux, thisline['wave_vacuum'], HIIregion[thisline['label']], args.vel_disp, HIIregion['vel_z_inc'])  # adding every line flux on top of continuum; gaussians are in ergs/s/A
                except KeyError:
                    ifu.linelist = ifu.linelist.drop(line_index)  # discarding label from linelist if it is not present in HIIRegion dataframe
                    pass
            ifu.linelist = ifu.linelist.reset_index(drop=True)

            flux = np.array([flux[ifu.bin_index == ii].mean() for ii in range(1, len(ifu.dispersion_arr) + 1)])  # spectral smearing i.e. rebinning of spectrum                                                                                                                             #mean() is used here to conserve flux; as f is in units of ergs/s/A, we want integral of f*dlambda to be preserved (same before and after resampling)
            # this can be checked as np.sum(f[1:]*np.diff(wavelength_array))

            ifu.data[int(HIIregion['pos_' + projection_dict[args.projection][0] + '_grid'])][int(HIIregion['pos_' + projection_dict[args.projection][1] + '_grid'])][:] += flux  # flux is ergs/s/A

        ifu.data = ifu.data / (4 * np.pi * (ifu.distance * Mpc_to_cm)**2) # converting from ergs/s/A to ergs/s/cm^2/A
        write_fitsobj(args.idealcube_filename, ifu, instrument, args, for_qfits=True) # writing into FITS file

    ifu = readcube(args.idealcube_filename, args)
    myprint('Done in %s minutes' % ((time.time() - start_time) / 60), args)

    return ifu, paramlist, args

# -----------------------------------------------------------------------------
if __name__ == '__main__':
    dummy_args = parse_args('8508', 'RD0042') # default simulation to work upon when comand line args not provided
    if not dummy_args.keep: plt.close('all')

    linelist = read_linelist(mappings_lab_dir + 'targetlines.txt')  # list of emission lines

    if dummy_args.do_all_sims: list_of_sims = all_sims
    else: list_of_sims = [(dummy_args.halo, dummy_args.output)]

    for index, this_sim in enumerate(list_of_sims):
        myprint('Doing halo ' + this_sim[0] + ' snapshot ' + this_sim[1] + ', which is ' + str(index + 1) + ' out of ' + str(len(list_of_sims)) + '..', dummy_args)
        args = parse_args(this_sim[0], this_sim[1])
        # ----------iterating over diag_arr and Om_ar to find the correct file with emission line fluxes-----------------------
        for diag in args.diag_arr:
            args.diag = diag
            for Om in args.Om_arr:
                args.Om = Om
                ideal_ifu, paramlist, args = get_ideal_datacube(args, linelist)

    myprint('Done making ideal datacubes for all given args.diag_arr and args.Om_arr', args)
