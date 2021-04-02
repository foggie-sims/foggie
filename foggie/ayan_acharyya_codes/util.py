##!/usr/bin/env python3

"""

    Title :      util
    Notes :      Contains various generic utility functions and classes used by the other scripts, including a 'master' function to parse args
    Author :     Ayan Acharyya
    Started :    March 2021

"""
from header import *

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
        self.instrument_dict = {'wfc3_grism': {'obs_wave_range': (0.8, 1.7), 'obs_spec_res': 200., 'obs_spatial_res': 0.1, 'el_per_phot': 1, 'radius': 2.0}, \
                           'sami': {'obs_wave_range': (0.8, 1.7), 'obs_spec_res': 30., 'obs_spatial_res': 1.0, 'el_per_phot': 1, 'radius': 2.0}, \
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
            self.el_per_phot = args.el_per_phot # dimensionless
            self.radius = args.tel_radius # metres

    # ---------deduce folder corresponding to this instrument--------
    def get_instrument_path(self):
        '''
        Function to deduce which specific instrument directory (in this jungle of folders) a given ifu datacube should be stored
        '''
        if self.name in self.instrument_dict:
            self.path = 'instrument_' + self.name + '/'
        else:
            self.path = 'dummy_instrument' + '_obs_wave_range_' + str(self.obs_wave_range[0]) + '-' + str(self.obs_wave_range[1]) + \
                        'mu_spectral_res_' + str(self.obs_spec_res) + 'kmps_spatial_res_' + str(self.obs_spatial_res) + 'arcsec' + \
                        '_rad' + str(self.radius) + 'm_epp_' + str(self.el_per_phot) + '/'

# ---------------------object containing info about the ideal ifu datacube----------------------------------------
class idealcube(object):
    # ---------initialise object-----------
    def __init__(self, args, instrument, linelist):
        self.z = args.z # redshift of the cube
        self.distance = get_distance(self.z)  # distance to object; in Mpc
        self.inclination = args.inclination

        self.rest_wave_range = args.base_wave_range * 1e4 # Angstroms, rest frame wavelength range for the ideal datacube

        delta_lambda = args.vel_highres_win / c  # buffer on either side of central wavelength, for judging which lines should be included in the given wavelength range
        self.linelist = linelist[linelist['wave_vacuum'].between(self.rest_wave_range[0] * (1 + delta_lambda), self.rest_wave_range[1] * (1 - delta_lambda))].reset_index(drop=True)  # curtailing list of lines based on the restframe wavelength range

        # base resolutions with which the ideal cube is made
        self.base_spatial_res = args.base_spatial_res # kpc
        self.base_spec_res = args.base_spec_res # km/s
        self.get_base_dispersion_arr(args)

        self.box_size_in_pix = int(round(2 * args.galrad / self.base_spatial_res))  # the size of the ideal ifu cube in pixels
        self.pixel_size_kpc = 2 * args.galrad / self.box_size_in_pix # pixel size in kpc
        self.data = np.zeros((self.box_size_in_pix, self.box_size_in_pix, self.ndisp)) # initialise datacube with zeroes
        self.declare_dimensions(args) # printing the dimensions

    # ---------compute dispersion array-----------
    def get_base_dispersion_arr(self, args):
        '''
        Function to compute the dispersion array (spectral dimension) for the ideal ifu datacube
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
        wave_highres_win = args.vel_highres_win / c # wavelength window within which to inject finer refinement around each emission line

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
            binned_wave_arr.append(binned_wave_arr[-1] * (1 + self.base_spec_res / c)) # creating spectrally binned wavelength array
            # by appending new wavelength at delta_lambda interval, where delta_lambda = lambda * velocity_resolution / c

        self.base_wave_arr = wave_arr
        self.bin_index = np.digitize(wave_arr, binned_wave_arr)
        self.dispersion_arr = np.array(binned_wave_arr)
        self.delta_lambda = np.diff(self.dispersion_arr)  # wavelength spacing for each wavecell; in Angstrom
        self.dispersion_arr = self.dispersion_arr[1:] # why are we omiting the first cell, again?
        self.ndisp = len(self.dispersion_arr)

    # -------------------------------------------------
    def declare_dimensions(self, args):
        '''
        Function to print the box dimensions
        '''
        myprint('For base spectral res= ' + str(self.base_spec_res) + ' km/s, and wavelength range of ' + str(self.rest_wave_range[0]) + ' to ' + str(self.rest_wave_range[1]) + ' A, length of dispersion axis= ' + str(self.ndisp) + ' pixels', args)
        myprint('For base spatial res= ' + str(self.base_spatial_res) + ' kpc, size of box= ' + str(self.box_size_in_pix) + ' pixels', args)

# ---------------------object containing info about the noiseless mock ifu datacube: sub class of ideal datacube----------------------------------------
class mockcube(idealcube):
    # ---------initialise object-----------
    def __init__(self, args, instrument, linelist):
        idealcube.__init__(self, args, instrument, linelist) # getting most attributes from the parent class 'idealcube'

        # new wavelength range depending on instrument's observed wavelength range and object's redshift
        self.rest_wave_range = instrument.obs_wave_range / (1 + self.z) # converting from obs to rest frame
        self.rest_wave_range *= 1e4 # converting from microns to Angstroms

        delta_lambda = args.vel_highres_win / c  # buffer on either side of central wavelength, for judging which lines should be included in the given wavelength range
        self.linelist = linelist[linelist['wave_vacuum'].between(self.rest_wave_range[0] * (1 + delta_lambda), self.rest_wave_range[1] * (1 - delta_lambda))].reset_index(drop=True)  # curtailing list of lines based on the restframe wavelength range
        self.get_base_dispersion_arr(args) # re-computing dispersion array with base spectral resolution, but with the instrument observed wavelength range and redshift

        # observed resolutions with which the mock cube is made, depending on the instrument being simulated
        self.obs_spatial_res = instrument.obs_spatial_res * np.pi / (3600 * 180) * (self.distance * 1e3) # since instrument resolution is in arcsec, converting to kpc (self.distance is in Mpc)
        self.obs_spec_res = instrument.obs_spec_res # km/s
        self.pix_per_beam = args.pix_per_beam

        base_box_size = self.box_size_in_pix # temporary variable to store the ideal datacube's box size
        self.box_size_in_pix = int(round(2 * args.galrad / (self.obs_spatial_res/self.pix_per_beam))) # the size of the mock ifu cube in pixels

        if self.box_size_in_pix > base_box_size: # lest we have to rebin to a "finer" grid, thereby doing unnecessary interpolations
            intended_pix_per_beam = self.pix_per_beam
            self.pix_per_beam = self.obs_spatial_res / (2 * args.galrad / base_box_size) # reducing pix_per_beam as far as necessary to not enlarge the box size in pixels
            myprint('To sample spatial resolution = ' +str(self.obs_spatial_res) + ' kpc with ' +str(intended_pix_per_beam) + \
                    ' beams, required box size (= ' + str(self.box_size_in_pix) + ' pix) is larger than base box size (= ' + str(base_box_size) + \
                    ' pix). In order to keep the box size and spatial resolution unchanged, your beam will now be sampled by only ' + str(self.pix_per_beam) + ' pixels. ' + \
                    'To avoid this scenario, either aim for a coarser obs_spatial_res or use a finer base_spatial_res or target a smaller pix_per_beam.', args)
            self.box_size_in_pix = base_box_size

        self.pixel_size_kpc = 2 * args.galrad / self.box_size_in_pix # pixel size in kpc
        self.achieved_spatial_res = self.pixel_size_kpc * self.pix_per_beam # kpc (this can vary very slightly from the 'intended' obs_spatial_res

        self.get_obs_dispersion_arr(args)

        # compute kernels to convolve by
        if args.kernel == 'gauss':
            self.sigma = gf2s * self.pix_per_beam
            self.ker_size = int((self.sigma * args.ker_size_factor) // 2 * 2 + 1) # rounding off to nearest odd integer because kernels need odd integer as size
            self.kernel = con.Gaussian2DKernel(self.sigma, x_size=self.ker_size, y_size=self.ker_size)
        elif args.ker == 'moff':
            self.sigma = self.pix_per_beam / (2 * np.sqrt(2 ** (1. / args.moff_beta) - 1.))
            self.ker_size = int((self.sigma * args.ker_size_factor) // 2 * 2 + 1) # rounding off to nearest odd integer because kernels need odd integer as size
            self.kernel = con.Moffat2DKernel(self.sigma, args.moff_beta, x_size=self.ker_size, y_size=self.ker_size)

        self.declare_obs_param(args)

    # ------------------------------------------------
    def get_obs_dispersion_arr(self, args):
        '''
        Function to compute the (mock) observed dispersion array (spectral dimension) for the mock ifu datacube
        :param args: It rebins (modifies in situ) cube.dispersion_arr (which is the dispersion array based on the base_spec_res) and to an array with the given obs_spec_res
        '''
        # --------spectral binning as per args.base_spec_res-----------------------------------
        binned_wave_arr = [self.rest_wave_range[0]]
        while binned_wave_arr[-1] <= self.rest_wave_range[1]:
            binned_wave_arr.append(binned_wave_arr[-1] * (1 + self.obs_spec_res / c)) # creating spectrally binned wavelength array
            # by appending new wavelength at delta_lambda interval, where delta_lambda = lambda * velocity_resolution / c

        self.bin_index = np.digitize(self.dispersion_arr, binned_wave_arr)
        self.dispersion_arr = np.array(binned_wave_arr)
        self.delta_lambda = np.diff(self.dispersion_arr)  # wavelength spacing for each wavecell; in Angstrom
        self.dispersion_arr = self.dispersion_arr[1:] # why are we omiting the first cell, again?
        self.ndisp = len(self.dispersion_arr)

    # -------------------------------------------------
    def declare_obs_param(self, args):
        '''
        Function to print observation parameters
        '''
        myprint('For obs spectral res= ' + str(self.obs_spec_res) + ' km/s, and wavelength range of ' + str(self.rest_wave_range[0]) + ' to ' + str(self.rest_wave_range[1]) + ' A, length of dispersion axis= ' + str(self.ndisp) + ' pixels', args)
        myprint('For obs spatial res= ' + str(args.obs_spatial_res) + ' arcsec on sky => physical res at redshift ' + str(self.z) + ' galaxy= ' + str(self.obs_spatial_res) + ' kpc, and pixel per beam = ' + str(self.pix_per_beam) + ', size of smoothed box= ' + str(self.box_size_in_pix) + ' pixels', args)
        myprint('Going to convolve with ' + args.kernel + ' kernel with FWHM = ' + str(self.pix_per_beam) + ' pixels (' + str(self.achieved_spatial_res) + ' kpc) => sigma = ' + str(self.sigma) + ' pixels, and total size of smoothing kernel = ' + str(self.ker_size) + ' pixels', args)

# ---------------------object containing info about the noisy mock ifu datacube: sub class of mock datacube----------------------------------------
class noisycube(mockcube):
    # ---------initialise object-----------
    def __init__(self, args, instrument, linelist):
        mockcube.__init__(self, args, instrument, linelist) # getting most attributes from the parent class 'mockcube'

        self.exptime = args.exptime # in seconds
        self.snr = args.snr # target SNR per pixel
        self.declare_noise_param(args)

    # -------------------------------------------------
    def declare_noise_param(self, args):
        '''
        Function to print noise parameters
        '''
        myprint('Noisy cube initialised with exposure time = ' + str(self.exptime) + ' s, and target SNR/pixel = ' + str(self.snr), args)

# -------------------------------------------------------------------------------------------
def myprint(text, args):
    '''
    Function to direct the print output to stdout or a file, depending upon user args
    '''
    if not text[-1] == '\n': text += '\n'
    if not args.silent:
        if args.print_to_file:
            ofile = open(args.printoutfile, 'a')
            ofile.write(text)
            ofile.close()
        else:
            print(text)

# -------------------------------------------------------------------------
def isfloat(str):
    '''
    Function to check if input is float
    '''

    try:
        float(str)
    except ValueError:
        return False
    return True

# -------------------------------------------------------------------------------
def num(s):
    '''
    Function to check if input is a number
    '''

    if s[-1].isdigit():
        return str(format(float(s), '0.2e'))
    else:
        return str(format(float(s[:-1]), '0.2e'))

# -------------------------------------------------------------------------
def read_linelist(linelistfile):
    '''
    Function to read pre-defined line list
    '''

    lines_to_pick = pd.read_table(linelistfile, comment='#', delim_whitespace=True, skiprows=3, names=('wave_vacuum', 'label', 'wave_air'))
    lines_to_pick = lines_to_pick.sort_values(by=('wave_vacuum')).reset_index(drop=True)
    return lines_to_pick

# -------------------------------------------------------------------------------------------
def get_distance(z, H0=70.):
    '''
    Function to ~approximately~ convert redshift (z) in to distance
    :param H0: default is 70 km/s/Mpc
    :return: distance in Mpc
    '''
    dist = z * c / H0  # Mpc
    return dist

# -----------------------------------------------------------------
def poly(x, R, k):
    '''
    Function to use KD02 R23 diagnostic for the upper Z branch
    '''

    return np.abs(np.poly1d(k)(x) - np.log10(R))

# -----------------------------------------------------------------
def rebin(array, dimensions=None, scale=None):
    """ Return the array ``array`` to the new ``dimensions`` conserving flux the flux in the bins
    The sum of the array will remain the same

    >>> ar = numpy.array([
        [0,1,2],
        [1,2,3],
        [2,3,4]
        ])
    >>> rebin(ar, (2,2))
    array([
        [1.5, 4.5]
        [4.5, 7.5]
        ])
    Raises
    ------

    AssertionError
        If the totals of the input and result array don't agree, raise an error because computation may have gone wrong

    Reference
    =========
    +-+-+-+
    |1|2|3|
    +-+-+-+
    |4|5|6|
    +-+-+-+
    |7|8|9|
    +-+-+-+
    """
    if dimensions is not None:
        if isinstance(dimensions, float):
            dimensions = [int(dimensions)] * len(array.shape)
        elif isinstance(dimensions, int):
            dimensions = [dimensions] * len(array.shape)
        elif len(dimensions) != len(array.shape):
            raise RuntimeError('')
    elif scale is not None:
        if isinstance(scale, float) or isinstance(scale, int):
            dimensions = map(int, map(round, map(lambda x: x * scale, array.shape)))
        elif len(scale) != len(array.shape):
            raise RuntimeError('')
    else:
        raise RuntimeError('Incorrect parameters to rebin.\n\trebin(array, dimensions=(x,y))\n\trebin(array, scale=a')
    if np.shape(array) == dimensions: return array  # no rebinning actually needed
    import itertools
    # dY, dX = map(divmod, map(float, array.shape), dimensions)

    result = np.zeros(dimensions)
    for j, i in itertools.product(*map(range, array.shape)):
        (J, dj), (I, di) = divmod(j * dimensions[0], array.shape[0]), divmod(i * dimensions[1], array.shape[1])
        (J1, dj1), (I1, di1) = divmod(j + 1, array.shape[0] / float(dimensions[0])), divmod(i + 1,
                                                                                            array.shape[1] / float(
                                                                                                dimensions[1]))

        # Moving to new bin
        # Is this a discrete bin?
        dx, dy = 0, 0
        if (I1 - I == 0) | ((I1 - I == 1) & (di1 == 0)):
            dx = 1
        else:
            dx = 1 - di1
        if (J1 - J == 0) | ((J1 - J == 1) & (dj1 == 0)):
            dy = 1
        else:
            dy = 1 - dj1
        # Prevent it from allocating outide the array
        I_ = np.min([dimensions[1] - 1, I + 1])
        J_ = np.min([dimensions[0] - 1, J + 1])
        result[J, I] += array[j, i] * dx * dy
        result[J_, I] += array[j, i] * (1 - dy) * dx
        result[J, I_] += array[j, i] * dy * (1 - dx)
        result[J_, I_] += array[j, i] * (1 - dx) * (1 - dy)
    allowError = 0.1
    if array.sum() > 0: assert (array.sum() < result.sum() * (1 + allowError)) & (array.sum() > result.sum() * (1 - allowError))
    return result

# --------------------------------------------------------------------------
def get_KD02_metallicity(photgrid):
    '''
    Function to compute KD02 metallicity from an input pandas dataframe with line fluxes as columns
    '''

    log_ratio = np.log10(np.divide(photgrid['NII6584'], (photgrid['OII3727'] + photgrid['OII3729'])))
    logOH = 1.54020 + 1.26602 * log_ratio + 0.167977 * log_ratio ** 2
    Z = 10 ** logOH  # converting to Z (in units of Z_sol) from log(O/H) + 12
    return Z

# ---------------------------------------------------------------------------------
def get_D16_metallicity(photgrid):
    '''
    Function to compute D16 metallicity from an input pandas dataframe with line fluxes as columns
    '''

    log_ratio = np.log10(np.divide(photgrid['NII6584'], (photgrid['SII6730'] + photgrid['SII6717']))) + 0.264 * np.log10(np.divide(photgrid['NII6584'], photgrid['H6562']))
    logOH = log_ratio + 0.45 * (log_ratio + 0.3) ** 5  # + 8.77
    Z = 10 ** logOH  # converting to Z (in units of Z_sol) from log(O/H) + 12
    return Z

# -------------------------------------------------------------------------------------------
def get_cube_output_path(args):
    '''
    Function to deduce which specific directory (in this jungle of folders) a given ifu datacube should be stored
    '''
    cube_output_path = args.output_dir + 'fits/' + args.output + '/diagnostic_' + args.diag + '/Om_' + str(args.Om) + '/proj_los_' + args.projection + '/boxsize_' + str(2*args.galrad) + \
        '/inc_' + str(args.inclination) + '/base_spectral_res_' + str(args.base_spec_res) + '/base_spatial_res_' + str(args.base_spatial_res) + '/'
    Path(cube_output_path).mkdir(parents=True, exist_ok=True) # creating the directory structure, if doesn't exist already

    return cube_output_path

# ---------------------object containing info about ifu datacubes that have been read in-----------------------------
class readcube(object):
    # ---------initialise object-----------
    def __init__(self, filename, args):
        '''
        Function to read a fits file (that has been written by write_fitsobj) and store the fits data in an object
        :param filename:
        :return:
        '''
        myprint('Reading in cube file ' + filename, args)
        cube = fits.open(filename)
        self.data = cube[0].data # reading in just the 3D data cube
        self.wavelength = cube[1].data
        self.header = cube[0].header

        self.data = np.nan_to_num(self.data) # replacing all NaN values with 0, otherwise calculations get messed up
        self.data = self.data.swapaxes(0, 2) # switching from (wave, pos, pos) arrangement (QFitsView requires) to (pos, pos, wave) arrangement (traditional)

# -------------------------------------------------------------------------------------------
def write_fitsobj(filename, cube, instrument, args, fill_val=np.nan, for_qfits=True):
    '''
    Function to write a ifu cube.data to a FITS file, along with all other attributes of the cube
    '''
    if for_qfits and np.shape(cube.data)[0] == np.shape(cube.data)[1]:
        cube.data = cube.data.swapaxes(0,2) # QFitsView requires (wave, pos, pos) arrangement rather than (pos, pos, wave)  arrangement
    if filename[-5:] != '.fits':
        filename += '.fits'

    flux = np.ma.filled(cube.data, fill_value=fill_val)
    wavelength = cube.dispersion_arr

    flux_header = fits.Header({'CRPIX1': 1, \
                               'CRVAL1': -2 * args.galrad, \
                               'CDELT1': 2 * args.galrad / np.shape(cube.data)[1], \
                               'CTYPE1': 'kpc', \
                               'CRPIX2': 1, \
                               'CRVAL2': -2 * args.galrad, \
                               'CDELT2': 2 * args.galrad / np.shape(cube.data)[1], \
                               'CTYPE2': 'kpc', \
                               'CRPIX3': 1, \
                               'CRVAL3': cube.dispersion_arr[0], \
                               'CDELT3': cube.delta_lambda[0], \
                               'CTYPE3': 'A', \
                               'data_unit(flambda)': 'ergs/s/cm^2/A', \
                               'simulation': args.halo, \
                               'snapshot': args.output, \
                               'metallicity_diagnostic': args.diag, \
                               'Omega': args.Om, \
                               'cutout_from_sim(kpc)': 2 * args.galrad, \
                               'inclination(deg)': cube.inclination, \
                               'redshift': cube.z, \
                               'distance(Mpc)': cube.distance, \
                               'base_spatial_res(kpc)':cube.base_spatial_res, \
                               'base_spec_res(km/s)': cube.base_spec_res, \
                               'box_size': ','.join(np.array([cube.box_size_in_pix, cube.box_size_in_pix, cube.ndisp]).astype(str)), \
                               'pixel_size(kpc)': cube.pixel_size_kpc, \
                               'rest_wave_range(A)': ','.join(cube.rest_wave_range.astype(str)), \
                               'labels': ','.join(cube.linelist['label']), \
                               'lambdas': ','.join(cube.linelist['wave_vacuum'].astype(str)), \
                               'instrument_name': instrument.name, \
                                }) # all the CRPIX3 etc. keywords are to make sure qfits has the wavelength information upon loading

    if hasattr(cube, 'obs_spatial_res'): # i.e. it is a smoothed mock datacube rather than an ideal datacube
        flux_header.update({'obs_spatial_res(arcsec)': instrument.obs_spatial_res, \
                            'obs_spatial_res(kpc)': cube.obs_spatial_res, \
                            'obs_spec_res(km/s)': instrument.obs_spec_res, \
                            'smoothing_kernel': args.kernel, \
                            'pix_per_beam': cube.pix_per_beam, \
                            'kernel_fwhm(pix)': cube.pix_per_beam, \
                            'kernel_fwhm(kpc)': cube.achieved_spatial_res, \
                            'kernel_sigma(pix)': cube.sigma, \
                            'kernel_size(pix)': cube.ker_size, \
                            })

    if hasattr(cube, 'snr'):  # i.e. it is a smoothed AND noisy mock datacube rather than an ideal datacube
        flux_header.update({'exptime(s)': cube.exptime, \
                            'target_snr': cube.snr, \
                            'electrons_per_photon': instrument.el_per_phot, \
                            'telescope_radius(m)': instrument.radius, \
                            })


    flux_hdu = fits.PrimaryHDU(flux, header=flux_header)
    wavelength_hdu = fits.ImageHDU(wavelength)
    hdulist = fits.HDUList([flux_hdu, wavelength_hdu])
    hdulist.writeto(filename, clobber=True)
    myprint('Written file ' + filename + '\n', args)

# ----------------------------------------------------------------------------------------------
def pull_halo_center(args):
    '''
    Function to pull halo center from halo catalogue, if exists, otherwise compute halo center
    Adapted from utils.foggie_load()
    '''

    foggie_dir, output_dir, run_loc, code_path, trackname, haloname, spectra_dir, infofile = get_run_loc_etc(args)
    args.output_dir = output_dir # so that output_dir is automatically propagated henceforth as args
    halos_df_name = code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/' + 'halo_c_v'

    if os.path.exists(halos_df_name):
        halos_df = pd.read_table(halos_df_name, sep='|')
        halos_df.columns = halos_df.columns.str.strip() # trimming column names of extra whitespace
        halos_df['name'] = halos_df['name'].str.strip() # trimming column 'name' of extra whitespace

        if halos_df['name'].str.contains(args.output).any():
            print("Pulling halo center from catalog file")
            halo_ind = halos_df.index[halos_df['name'] == args.output][0]
            args.halo_center = halos_df.loc[halo_ind, ['xc', 'yc', 'zc']].values # in kpc units
            args.halo_velocity = halos_df.loc[halo_ind, ['xv', 'yv', 'zv']].values # in km/s units
            calc_hc = False
        else:
            print('This snapshot is not in the halos_df file, calculating halo center...')
            calc_hc = True
    else:
        print("This halos_df file doesn't exist, calculating halo center...")
        calc_hc = True
    if calc_hc:
        ds, refine_box = load_sim(args, region='refine_box')
        args.halo_center = ds.halo_center_kpc
        args.halo_velocity = ds.halo_velocity_kms
    return args

# --------------------------------------------------------------------------------------------------------------
def parse_args(haloname, RDname):
    '''
    Function to parse keyword arguments
    '''

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='''identify satellites in FOGGIE simulations''')
    # ---- common args used widely over the full codebase ------------
    parser.add_argument('--system', metavar='system', type=str, action='store', help='Which system are you on? Default is Jase')
    parser.set_defaults(system='ayan_local')

    parser.add_argument('--do', metavar='do', type=str, action='store', help='Which particles do you want to plot? Default is gas')
    parser.set_defaults(do='gas')

    parser.add_argument('--run', metavar='run', type=str, action='store', help='which run? default is natural')
    parser.set_defaults(run="nref11c_nref9f")

    parser.add_argument('--halo', metavar='halo', type=str, action='store', help='which halo? default is 8508 (Tempest)')
    parser.set_defaults(halo=haloname)

    parser.add_argument('--projection', metavar='projection', type=str, action='store', help='Which projection do you want to plot, i.e., which axis is your line of sight? Default is x')
    parser.set_defaults(projection='x')

    parser.add_argument('--output', metavar='output', type=str, action='store', help='which output? default is RD0020')
    parser.set_defaults(output=RDname)

    parser.add_argument('--pwd', dest='pwd', action='store_true', help='Just use the current working directory?, default is no')
    parser.set_defaults(pwd=False)

    parser.add_argument('--do_all_sims', dest='do_all_sims', action='store_true', help='Run the code on all simulation snapshots available?, default is no')
    parser.set_defaults(do_all_sims=False)

    parser.add_argument('--silent', dest='silent', action='store_true', help='Suppress all print statements?, default is no')
    parser.set_defaults(silent=False)

    # ------- args added for filter_star_properties.py ------------------------------
    parser.add_argument('--plot_proj', dest='plot_proj', action='store_true', help='plot projection map? default is no')
    parser.set_defaults(plot_proj=False)

    parser.add_argument('--clobber', dest='clobber', action='store_true', help='overwrite existing outputs with same name?, default is no')
    parser.set_defaults(clobber=False)

    parser.add_argument('--automate', dest='automate', action='store_true', help='automatically execute the next script?, default is no')
    parser.set_defaults(automate=False)

    # ------- args added for compute_hii_radii.py ------------------------------
    parser.add_argument('--galrad', metavar='galrad', type=float, action='store', help='the radial extent (in each spatial dimension) to which computations will be done, in kpc; default is 20')
    parser.set_defaults(galrad=20.)

    parser.add_argument('--mergeHII', metavar='mergeHII', type=float, action='store', help='separation btwn HII regions below which to merge them, in kpc; default is None i.e., do not merge')
    parser.set_defaults(mergeHII=None)

    # ------- args added for lookup_flux.py ------------------------------
    parser.add_argument('--diag_arr', metavar='diag_arr', type=str, action='store', help='list of metallicity diagnostics to use')
    parser.set_defaults(diag_arr='D16')

    parser.add_argument('--Om_arr', metavar='Om_arr', type=str, action='store', help='list of Omega values to use')
    parser.set_defaults(Om_arr='0.5')

    parser.add_argument('--nooutliers', dest='nooutliers', action='store_true', help='discard outlier HII regions (according to D16 diagnostic)?, default is no')
    parser.set_defaults(nooutliers=False)

    parser.add_argument('--xratio', metavar='xratio', type=str, action='store', help='ratio of lines to plot on X-axis; default is None')
    parser.set_defaults(xratio=None)

    parser.add_argument('--yratio', metavar='yratio', type=str, action='store', help='ratio of lines to plot on Y-axis; default is None')
    parser.set_defaults(yratio=None)

    parser.add_argument('--fontsize', metavar='fontsize', type=int, action='store', help='fontsize of plot labels, etc.; default is 15')
    parser.set_defaults(fontsize=15)

    parser.add_argument('--plot_metgrad', dest='plot_metgrad', action='store_true', help='make metallicity gradient plot?, default is no')
    parser.set_defaults(plot_metgrad=False)

    parser.add_argument('--plot_phase_space', dest='plot_phase_space', action='store_true', help='make P-r phase space plot?, default is no')
    parser.set_defaults(plot_phase_space=False)

    parser.add_argument('--plot_obsv_phase_space', dest='plot_obsv_phase_space', action='store_true', help='overlay observed P-r phase space on plot?, default is no')
    parser.set_defaults(plot_obsv_phase_space=False)

    parser.add_argument('--plot_fluxgrid', dest='plot_fluxgrid', action='store_true', help='make flux ratio grid plot?, default is no')
    parser.set_defaults(plot_fluxgrid=False)

    parser.add_argument('--annotate', dest='annotate', action='store_true', help='annotate grid plot?, default is no')
    parser.set_defaults(annotate=False)

    parser.add_argument('--pause', dest='pause', action='store_true', help='pause after annotating each grid?, default is no')
    parser.set_defaults(pause=False)

    parser.add_argument('--plot_Zin_Zout', dest='plot_Zin_Zout', action='store_true', help='make input vs output metallicity plot?, default is no')
    parser.set_defaults(plot_Zin_Zout=False)

    parser.add_argument('--saveplot', dest='saveplot', action='store_true', help='save the plot?, default is no')
    parser.set_defaults(saveplot=False)

    parser.add_argument('--keep', dest='keep', action='store_true', help='keep previously displayed plots on screen?, default is no')
    parser.set_defaults(keep=False)

    parser.add_argument('--use_RGI', dest='use_RGI', action='store_true', help='kuse RGI interpolation vs LND?, default is no')
    parser.set_defaults(use_RGI=False)

    # ------- args added for make_ideal_datacube.py ------------------------------
    parser.add_argument('--obs_wave_range', metavar='obs_wave_range', type=str, action='store', help='observed wavelength range for the simulated instrument, in micron; default is (0.8, 1.7) microns')
    parser.set_defaults(obs_wave_range='0.65,0.68')

    parser.add_argument('--z', metavar='z', type=float, action='store', help='redshift of the mock datacube; default is 0.0001 (not 0, so as to avoid flux unit conversion issues)')
    parser.set_defaults(z=0.0001)

    parser.add_argument('--base_wave_range', metavar='base_wave_range', type=str, action='store', help='wavelength range for the ideal datacube, in micron; default is (0.64, 0.68) microns')
    parser.set_defaults(base_wave_range='0.64,0.68')

    parser.add_argument('--inclination', metavar='inclination', type=float, action='store', help='inclination angle to rotate the galaxy by, on a plane perpendicular to projection plane, i.e. if projection is xy, rotation is on yz, in degrees; default is 0')
    parser.set_defaults(inclination=0.)

    parser.add_argument('--vel_disp', metavar='vel_disp', type=float, action='store', help='intrinsic velocity dispersion for each emission line, in km/s; default is 15 km/s')
    parser.set_defaults(vel_disp=15.)

    parser.add_argument('--nbin_cont', metavar='nbin_cont', type=int, action='store', help='no. of spectral bins to bin the continuum (witout emission lines) in to; default is 1000')
    parser.set_defaults(nbin_cont=1000)

    parser.add_argument('--vel_highres_win', metavar='vel_highres_win', type=float, action='store', help='velocity window on either side of each emission line, in km/s, within which the continuum is resolved into finer (nbin_highres_cont) spectral elements; default is 500 km/s')
    parser.set_defaults(vel_highres_win=500.)

    parser.add_argument('--nbin_highres_cont', metavar='nbin_highres_cont', type=int, action='store', help='no. of additonal spectral bins to introduce around each emission line; default is 100')
    parser.set_defaults(nbin_highres_cont=100)

    parser.add_argument('--base_spec_res', metavar='base_spec_res', type=float, action='store', help='base spectral resolution, in km/s, i.e. to be employed while making the ideal datacube; default is 30 km/s')
    parser.set_defaults(base_spec_res=30.)

    parser.add_argument('--base_spatial_res', metavar='base_spatial_res', type=float, action='store', help='base spatial resolution, in kpc, i.e. to be employed while making the ideal datacube; default is 0.04 kpc = 40 pc')
    parser.set_defaults(base_spatial_res=0.04)

    parser.add_argument('--print_to_file', dest='print_to_file', action='store_true', help='Redirect all print statements to a file?, default is no')
    parser.set_defaults(print_to_file=False)

    parser.add_argument('--printoutfile', metavar='printoutfile', type=str, action='store', help='file to write all print statements to; default is ./logfile.out')
    parser.set_defaults(printoutfile='./logfile.out')

    parser.add_argument('--instrument', metavar='instrument', type=str, action='store', help='which instrument to simulate?; default is dummy')
    parser.set_defaults(instrument='dummy')

    parser.add_argument('--debug', dest='debug', action='store_true', help='run in debug mode (lots of print checks)?, default is no')
    parser.set_defaults(debug=False)

    # ------- args added for make_mock_datacube.py ------------------------------
    parser.add_argument('--obs_spec_res', metavar='obs_spec_res', type=float, action='store', help='observed spectral resolution of the instrument, in km/s; default is 60 km/s')
    parser.set_defaults(obs_spec_res=30.)

    parser.add_argument('--obs_spatial_res', metavar='obs_spatial_res', type=float, action='store', help='observed spatial resolution of the instrument, in arcsec; default is 1.0"')
    parser.set_defaults(obs_spatial_res=1.0)

    parser.add_argument('--pix_per_beam', metavar='pix_per_beam', type=int, action='store', help='number of pixels to sample the resolution element (PSF) by; default is 6"')
    parser.set_defaults(pix_per_beam=6)

    parser.add_argument('--kernel', metavar='kernel', type=str, action='store', help='which kernel to simulate for seeing, gauss or moff?; default is gauss')
    parser.set_defaults(kernel='gauss')

    parser.add_argument('--ker_size_factor', metavar='ker_size_factor', type=int, action='store', help='factor to multiply kernel sigma by to get kernel size, e.g. if PSF sigma=5 pixel and ker_size_factor=5, kernel size=25 pixel; default is 5"')
    parser.set_defaults(ker_size_factor=5)

    parser.add_argument('--moff_beta', metavar='moff_beta', type=float, action='store', help='beta (power index) in moffat kernel; default is 4.7"')
    parser.set_defaults(moff_beta=4.7)

    parser.add_argument('--snr', metavar='snr', type=float, action='store', help='target SNR of the datacube; default is 0, i.e. noiseless"')
    parser.set_defaults(snr=0)

    parser.add_argument('--tel_radius', metavar='tel_radius', type=float, action='store', help='radius of telescope, in metres; default is 1 m')
    parser.set_defaults(tel_radius=1)

    parser.add_argument('--exptime', metavar='exptime', type=float, action='store', help='exposure time of observation, in sec; default is 1200 sec')
    parser.set_defaults(exptime=1200)

    parser.add_argument('--el_per_phot', metavar='el_per_phot', type=float, action='store', help='how many electrons do each photon trigger in the instrument; default is 1"')
    parser.set_defaults(el_per_phot=1)

    # ------- wrap up and processing args ------------------------------
    args = parser.parse_args()

    args.diag_arr = [item for item in args.diag_arr.split(',')]
    args.Om_arr = [float(item) for item in args.Om_arr.split(',')]
    args.obs_wave_range = np.array([float(item) for item in args.obs_wave_range.split(',')])
    args.base_wave_range = np.array([float(item) for item in args.base_wave_range.split(',')])
    args.mergeHII_text = '_mergeHII=' + str(args.mergeHII) + 'kpc' if args.mergeHII is not None else '' # to be used as filename suffix to denote whether HII regions have been merged
    args.without_outlier = '_no_outlier' if args.nooutliers else '' # to be used as filename suffix to denote whether outlier HII regions (as per D16 density criteria) have been discarded

    args = pull_halo_center(args) # pull details about center of the snapshot
    return args

