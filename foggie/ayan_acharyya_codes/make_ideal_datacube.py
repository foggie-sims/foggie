##!/usr/bin/env python3

"""

    Title :      make_ideal_datacube
    Notes :      Produces an idealised IFU datacube by projecting a list of HII region emissions on to a 2D grid, for a given inc & PA, and a base (high) spatial + spectral resolution
    Output :     FITS cube
    Author :     Ayan Acharyya
    Started :    January 2021
    Example :    run make_ideal_datacube.py --system ayan_local --halo 5036 --output RD0030 --mergeHII 0.04 --base_spatial_res 0.4 --z 0.25 --obs_wave_range 0.8,0.85 --instrument dummy

"""
from header import *
from compute_hiir_radii import *
from filter_star_properties import *
import make_mappings_grid as mmg


# ---------------------object containing info about the instrument being simulated----------------------------------------
class telescope(object):
    # ---------initialise object-----------
    def __init__(self, args):
        self.name = args.instrument.lower()
        self.get_instrument_properties(args)
        self.get_instrument_path()

    # ---------deduce folder corresponding to this instrument--------
    def get_instrument_properties(self, args):
        '''
        Function to set attributes of the given instrument either based on pre-saved list or the user inputs
        For the list of pre-saved instrument attributes intrument_dict:
        name: must be in lowercase
        obs_wave_range: in microns
        obs_spec_res: in km/s
        obs_spatial_res: in arcseconds
        '''
        self.instrument_dict = {'wfc3_grism': {'obs_wave_range': (0.8, 1.7), 'obs_spec_res': 200., 'obs_spatial_res': 0.1}, \
                           'sami': {'obs_wave_range': (0.8, 1.7), 'obs_spec_res': 30., 'obs_spatial_res': 1.0}, \
                           }

        if self.name in self.instrument_dict: # assigning known parameters (by overriding user input parameters, if any) for a known instrument
            myprint('Known instrument: ' + self.name + '; using pre-assigned attributes (over-riding user inputs, if any)', args)
            for key in self.instrument_dict[self.name]:
                setattr(self, key, self.instrument_dict[self.name][key])
            self.obs_wave_range = np.array(self.obs_wave_range)

        else: # assigning user input parameters for this unknown instrument
            myprint('Unknown instrument: ' + self.name + '; using user input/default attributes', args)
            self.name = 'dummy'
            self.obs_wave_range = np.array(args.obs_wave_range) # in microns
            self.obs_spec_res = args.obs_spec_res # in km/s
            self.obs_spatial_res = args.obs_spatial_res # in arcseconds

    # ---------deduce folder corresponding to this instrument--------
    def get_instrument_path(self):
        '''
        Function to deduce which specific instrument directory (in this jungle of folders) a given ifu datacube should be stored
        '''
        if self.name in self.instrument_dict:
            self.path = 'instrument_' + self.name + '/'
        else:
            self.path = 'dummy_instrument' + '_obs_wave_range_' + str(self.obs_wave_range[0]) + '-' + str(self.obs_wave_range[1]) + \
                        'mu_spectral_res_' + str(self.obs_spec_res) + 'kmps_spatial_res_' + str(self.obs_spatial_res) + 'arcsec' + '/'

# ---------------------object containing info about the ideal ifu datacube----------------------------------------
class idealcube(object):
    # ---------initialise object-----------
    def __init__(self, args, instrument, linelist):
        self.instrument_name = instrument.name # (mock) instrument with which the cube has been "observed"
        self.z = args.z # redshift of the cube
        self.distance = get_distance(self.z)  # distance to object; in Mpc
        self.inclination = args.inclination

        self.rest_wave_range = instrument.obs_wave_range / (1 + self.z) # converting from obs to rest frame
        self.rest_wave_range *= 1e4 # converting from microns to Angstroms

        delta_lambda = args.vel_highres_win / (mmg.c / 1e3)  # buffer on either side of central wavelength, for judging which lines should be included in the given wavelength range
        self.linelist = linelist[linelist['wave_vacuum'].between(self.rest_wave_range[0] * (1 + delta_lambda), self.rest_wave_range[1] * (1 - delta_lambda))].reset_index(drop=True)  # curtailing list of lines based on the restframe wavelength range
        self.get_dispersion_arr(args)

        # base resolutions with which the ideal cube is made
        self.base_spatial_res = args.base_spatial_res # kpc
        self.base_spec_res = args.base_spec_res # km/s

        self.box_size_in_pix = int(round(2 * args.galrad / self.base_spatial_res))  # the size of the ideal ifu cube in pixels
        self.data = np.zeros((self.box_size_in_pix, self.box_size_in_pix, self.ndisp)) # initialise datacube with zeroes
        self.declare_dimensions(args) # printing the dimensions

    # ---------compute dispersion array-----------
    def get_dispersion_arr(self, args):
        '''
        Function to compute the dispersion array (spectral dimension) for the idal ifu datacube
        :param args: It first creates a linear, uniform grid of length args.nbin_cont within self.rest_wave_range
        and then 'adds' further refinement of args.nbin_highres_cont bins to the continuum within +/- args.vel_highres_win window
        around each emission line; and then finally bin the entire wavelength array to a spectral resolution of args.base_spec_res

        For the default values (in header.py): within a wavelength range of ~ 3000 - 6000 A,
        the base array (1000 bins) has 3 A pixels => 300 (blue end) to 150 (red end) km/s spectral resolution,
        and around each emission line, a +/- 500 km/s window has 100 bins => 2*500/100 = 10 km/s spectral resolution;
        at the end, the whole wavelength array is binned to 30 km/s i.e. continuum is generally oversampled, except near the emission lines, where it is undersampled
        :return: args (with new variables included)
        '''
        wave_arr = np.linspace(self.rest_wave_range[0], self.rest_wave_range[1], args.nbin_cont) # base wavelength array (uniformly binned)
        wave_highres_win = args.vel_highres_win / (mmg.c/1e3) # wavelength window within which to inject finer refinement around each emission line

        # -------injecting finer refinement around emission lines------
        for this_cen in self.linelist['wave_vacuum']:
            this_leftedge = this_cen * (1 - wave_highres_win)
            this_rightedge = this_cen * (1 + wave_highres_win)
            wave_highres_arr = np.linspace(this_leftedge, this_rightedge, args.nbin_highres_cont)  # just a cutout of high spectral resolution array around this given emission line
            blueward_of_this_leftedge = wave_arr[:np.where(wave_arr < this_leftedge)[0][-1] + 1]
            redward_of_this_rightedge = wave_arr[np.where(wave_arr > this_rightedge)[0][0]:]
            wave_arr = np.hstack((blueward_of_this_leftedge, wave_highres_arr, redward_of_this_rightedge))  # slicing the original (coarse) wavelength array around the emission line and sticking in the highres wavelength array there

        # --------spectral binning as per args.base_spec_res-----------------------------------
        binned_wave_arr = [wave_arr[0]]
        while binned_wave_arr[-1] <= wave_arr[-1]:
            binned_wave_arr.append(binned_wave_arr[-1] * (1 + args.base_spec_res / (mmg.c/1e3))) # creating spectrally binned wavelength array
            # by appending new wavelength at delta_lambda interval, where delta_lambda = lambda * velocity_resolution / c

        self.base_wave_arr = wave_arr
        self.bin_index = np.digitize(wave_arr, binned_wave_arr)
        self.dispersion_arr = np.array(binned_wave_arr[1:]) # why are we omiting the first cell, again?
        self.ndisp = len(self.dispersion_arr)
        #self.delta_lambda = np.array([args.dispersion_arr[1] - args.dispersion_arr[0]] + [(args.dispersion_arr[i + 1] - args.dispersion_arr[i - 1]) / 2 for i in range(1, len(args.dispersion_arr) - 1)] + [args.dispersion_arr[-1] - args.dispersion_arr[-2]])  # wavelength spacing for each wavecell; in Angstrom
        self.delta_lambda = np.diff(self.dispersion_arr)  # wavelength spacing for each wavecell; in Angstrom

    # ---------print length os dispersion array-----------
    def declare_dimensions(self, args):
        myprint('For spectral res= ' + str(self.base_spec_res) + ' km/s, and wavelength range of ' + str(self.rest_wave_range[0]) + ' to ' + str(self.rest_wave_range[1]) + ' A, length of dispersion axis= ' + str(self.ndisp) + ' pixels', args)
        myprint('For spatial res= ' + str(self.base_spatial_res) + ' kpc, size of box= ' + str(self.box_size_in_pix) + ' pixels', args)

# -------------------------------------------------------------------------------------------
def get_distance(z, H0=70.):
    '''
    Function to ~approximately~ convert redshift (z) in to distance
    :param H0: default is 70 km/s/Mpc
    :return: distance in Mpc
    '''
    c = 3e5 # km/s
    dist = z * c / H0  # Mpc
    return dist

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
    c = mmg.c/1e3  # c = 3e5 km/s
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
    Function to incline i.e. rotate the galaxy in YZ plane (keeping X fixed) for a given angle of inclination
    :return: modified dataframe with new positions and velocities as new columns
    '''
    inc = float(args.inclination) * np.pi/180 # converting degrees to radians

    # now doing coordinate transformation to get new coordinates
    paramlist['pos_y_inc'] = paramlist['pos_y_cen'] * np.cos(inc) + paramlist['pos_z_cen'] * np.sin(inc)
    paramlist['pos_z_inc'] = -paramlist['pos_y_cen'] * np.sin(inc) + paramlist['pos_z_cen'] * np.cos(inc)
    paramlist['pos_x_inc'] = paramlist['pos_x_cen']

    # now doing coordinate transformation to get new velocities
    paramlist['vel_y_inc'] = paramlist['vel_y_cen'] * np.cos(inc) + paramlist['vel_z_cen'] * np.sin(inc)
    paramlist['vel_z_inc'] = -paramlist['vel_y_cen'] * np.sin(inc) + paramlist['vel_z_cen'] * np.cos(inc)
    paramlist['vel_x_inc'] = paramlist['vel_x_cen']

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
    paramlist = paramlist[(paramlist['pos_x_cen'].between(-args.galrad, args.galrad)) & (paramlist['pos_y_cen'].between(-args.galrad, args.galrad))].reset_index(drop=True)  # Curtails paramlist to keep only those HII regions that are within 2 x args.galrad kpx box
    myprint(str(len(paramlist)) + ' HII regions remain out of ' + str(initial_length) + ', within the ' + str(2*args.galrad) + ' kpc^2 box', args)

    paramlist['pos_x_grid'] = ((paramlist['pos_x_inc'] + args.galrad)/args.base_spatial_res).astype(np.int)
    paramlist['pos_y_grid'] = ((paramlist['pos_y_inc'] + args.galrad)/args.base_spatial_res).astype(np.int)
    paramlist['pos_z_grid'] = ((paramlist['pos_z_inc'] + args.galthick/2.)/args.base_spatial_res).astype(np.int)

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
        paramlist = get_star_properties(args)
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
    Path(cube_output_path + instrument.path).mkdir(parents=True, exist_ok=True)  # creating the directory structure, if doesn't exist already
    args.idealcube_filename = cube_output_path + instrument.path + 'ideal_ifu' + '_z' + str(ifu.z) + args.mergeHII_text + '.fits'

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

            for dummy, thisline in ifu.linelist.iterrows():
                flux = gauss(ifu.base_wave_arr, flux, thisline['wave_vacuum'], HIIregion[thisline['label']], args.vel_disp, HIIregion['vel_z_inc'])  # adding every line flux on top of continuum; gaussians are in ergs/s/A

            flux = np.array([flux[ifu.bin_index == ii].mean() for ii in range(1, len(ifu.dispersion_arr) + 1)])  # spectral smearing i.e. rebinning of spectrum                                                                                                                             #mean() is used here to conserve flux; as f is in units of ergs/s/A, we want integral of f*dlambda to be preserved (same before and after resampling)
            # this can be checked as np.sum(f[1:]*np.diff(wavelength_array))

            ifu.data[int(HIIregion['pos_x_grid'])][int(HIIregion['pos_y_grid'])][:] += flux  # flux is ergs/s/A, ifucube becomes ergs/s/A/pixel

        write_fitsobj(args.idealcube_filename, ifu, args, for_qfits=True) # writing into FITS file

    ifu = readcube(args.idealcube_filename, args)
    myprint('Done in %s minutes' % ((time.time() - start_time) / 60), args)

    return ifu, paramlist, args

# -----------------------------------------------------------------------------
if __name__ == '__main__':
    args = parse_args('8508', 'RD0042')
    if not args.keep: plt.close('all')

    linelist = mmg.read_linelist(mappings_lab_dir + 'targetlines.txt')  # list of emission lines

    # ----------iterating over diag_arr and Om_ar to find the correct file with emission line fluxes-----------------------
    for diag in args.diag_arr:
        args.diag = diag
        for Om in args.Om_arr:
            args.Om = Om
            ifu, paramlist, args = get_ideal_datacube(args, linelist)

    myprint('Done making ideal datacubes for all given args.diag_arr and args.Om_arr', args)
