##!/usr/bin/env python3

"""

    Title :      make_mock_datacube
    Notes :      Produces a mock IFU datacube by spatial convolution & spectral binning & noise addition to an ideal datacube, for a given seeing + spectral resolution + SNR
    Output :     FITS cube
    Author :     Ayan Acharyya
    Started :    January 2021
    Example :    run make_mock_datacube.py --system ayan_local --halo 8508 --output RD0042 --mergeHII 0.04 --base_spatial_res 0.4 --z 0.25 --obs_wave_range 0.8,0.85 --obs_spatial_res 1 --obs_spec_res 60 --exptime 1200 --snr 5 --debug

"""
from header import *
import make_mappings_grid as mmg
from make_ideal_datacube import *

# ---------------------object containing info about the noiseless mock ifu datacube: sub class of ideal datacube----------------------------------------
class mockcube(idealcube):
    # ---------initialise object-----------
    def __init__(self, args, instrument, linelist):
        idealcube.__init__(self, args, instrument, linelist) # getting most attributes from the parent class 'idealcube'

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

        self.pixel_size_kpc = 2 * args.galrad / self.box_size_in_pix # pixel size in kpc
        self.achieved_spatial_res = self.pixel_size_kpc * self.pix_per_beam # kpc (this can vary very slightly from the 'intended' obs_spatial_res

        self.data = np.zeros((self.box_size_in_pix, self.box_size_in_pix, self.ndisp)) # initialise datacube with zeroes
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
        binned_wave_arr = [self.dispersion_arr[0]]
        while binned_wave_arr[-1] <= self.dispersion_arr[-1]:
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
    assert (array.sum() < result.sum() * (1 + allowError)) & (array.sum() > result.sum() * (1 - allowError))
    return result

# ---------------------------------------------------------------------
def spatial_convolve(ideal_ifu, mock_ifu, args):
    '''
    Function to spatially (rebin and) convolve ideal data cube, with a given PSF, wavelength slice by wavelength slice
    :return: mockcube object: mock_ifu
    '''
    start_time = time.time()

    for slice in range(np.shape(ideal_ifu.data)[2]):
        myprint('Rebinning & convolving slice ' + str(slice + 1) + ' of ' + str(np.shape(ideal_ifu.data)[2]) + '..', args)
        rebinned_slice = rebin(ideal_ifu.data[:, :, slice], (mock_ifu.box_size_in_pix, mock_ifu.box_size_in_pix))  # rebinning before convolving
        mock_ifu.data[:, :, slice] = con.convolve_fft(rebinned_slice, mock_ifu.kernel, normalize_kernel=True)  # convolving with kernel

    myprint('Completed spatial convolution in %s minutes' % ((time.time() - start_time) / 60), args)
    return mock_ifu

# ---------------------------------------------------------------------
def spectral_bin(mock_ifu, args):
    '''
    Function to spectrally rebin given data cube, with a given spectral resolution, pixel by pixel
    :return: mockcube object: mock_ifu
    '''
    start_time = time.time()

    smoothed_data = mock_ifu.data # smoothed_data is only spatially smoothed but as yet spectrally unbinned
    xlen, ylen = np.shape(smoothed_data)[0], np.shape(smoothed_data)[1]
    mock_ifu.data = np.zeros((mock_ifu.box_size_in_pix, mock_ifu.box_size_in_pix, mock_ifu.ndisp))  # initialise datacube with zeroes

    for index in range(xlen*ylen - 1):
        i, j = int(index / ylen), int(index % ylen) # cell position
        myprint('Spectral rebinning pixel (' + str(i) + ',' + str(j) + '), i.e. ' + str(index + 1) + ' out of ' + str(xlen * ylen) + '..', args)
        unbinned_spectra = smoothed_data[i, j, :]
        mock_ifu.data[i, j, :] = np.array([unbinned_spectra[mock_ifu.bin_index == ii].mean() for ii in range(1, len(mock_ifu.dispersion_arr) + 1)])  # spectral smearing i.e. rebinning of spectrum                                                                                                                             #mean() is used here to conserve flux; as f is in units of ergs/s/A, we want integral of f*dlambda to be preserved (same before and after resampling)

    myprint('Completed spectral binning in %s minutes' % ((time.time() - start_time) / 60), args)
    return mock_ifu

# ---------------------------------------------------------------------
def add_noise(mock_ifu, instrument, args):
    '''
    Function to add noise to a data cube, with a given target SNR, voxel by voxel (i.e. the noise is spatially and spectrally variable)
    :return: mockcube object: mock_ifu
    '''
    start_time = time.time()

    clean_data = mock_ifu.data # clean_data has no noise, in flux density units
    (xlen, ylen, zlen) = np.shape(clean_data)
    mock_ifu.data = np.zeros(np.shape(clean_data))  # initialise datacube with zeroes

    for index in range(xlen * ylen * zlen - 1):
        i, j, k = int(index / (ylen * zlen)), int((index / zlen) % ylen), int(index % zlen) # cell position
        myprint('Adding noise to voxel (' + str(i) + ',' + str(j) + ',' + str(k) + '), i.e. ' + str(index + 1) + ' out of ' + str(xlen * ylen * zlen) + '..', args)
        flux = clean_data[i, j, k]
        wavelength = mock_ifu.dispersion_arr[k]
        delta_lambda = mock_ifu.delta_lambda[k]

        if args.debug: myprint('Deb225: flux = ' + str(flux) + ' ergs/s/cm^2/A; wavelength = ' + str(wavelength) + ' A; delta lambda = ' + str(delta_lambda) + ' A', args)
        # compute conversion factor from flux density units to photon count (will be used to add noise), based on telescope properties
        flux_density_to_counts = np.pi * (instrument.radius * 1e2)**2 * mock_ifu.exptime * instrument.el_per_phot * delta_lambda / (planck * (c * 1e3) / (wavelength * 1e-10))  # to bring ergs/s/A/pixel to units of counts/pixel (ADUs)

        flux = flux * flux_density_to_counts  # converting flux density units to counts (photons)

        if args.debug: myprint('Deb231: flux = ' + str(flux) + ' electrons/pix; using factor = ' + str(flux_density_to_counts) + ' A.s.cm^2/ergs', args)

        noisyflux = flux + get_noise_in_voxel(flux, wavelength, mock_ifu.snr, args) # adding noise to the flux, in counts unit                                                                                                                           #mean() is used here to conserve flux; as f is in units of ergs/s/A, we want integral of f*dlambda to be preserved (same before and after resampling)
        mock_ifu.data[i, j, k] = noisyflux / flux_density_to_counts # converting counts to flux density units

        if args.debug: myprint('Deb236: noisy flux = ' + str(noisyflux) + ' electrons/pix = ' + str(mock_ifu.data[i, j, k]) + ' ergs/s/cm^2/A', args)

    myprint('Completed adding noise in %s minutes' % ((time.time() - start_time) / 60), args)
    return mock_ifu

# ---------------------------------------------------------------------
def get_noise_in_voxel(data, wavelength, target_SNR, args):
    '''
    Function to compute the noise to add to a single voxel, given the data (flux) in photon counts, wavelength and target SNR
    :return: initial data + randomly generated noise
    '''
    absolute_noise = data / target_SNR
    random_noise = np.random.poisson(lam=absolute_noise ** 2, size=np.shape(data)) - absolute_noise ** 2

    if args.debug: myprint('Deb250: data = ' + str(data) + ' electrons; absolute_noise = ' + str(absolute_noise) + '; random noise = ' + str(random_noise) + ' electrons', args)

    return random_noise
# -----------------------------------------------------------------------
def get_mock_datacube(ideal_ifu, args, linelist, cube_output_path):
    '''
    Function to produce ideal IFU datacube, for a given base spatial and spectral resolution;
    :param paramlist: Computes spectra (stellar continuum + emission lines) for each HII region (in paramlist), then adds all spectra along each LoS, binned spatially
    at a given spatial resolution, to produce a x - y - lambda datacube.
    :return: writes datacube in to FITS file and returns fits object too
    '''
    start_time = time.time()

    instrument = telescope(args)  # declare the instrument
    args.smoothed_cube_filename = cube_output_path + instrument.path + 'smoothed_ifu' + '_z' + str(args.z) + args.mergeHII_text + '_ppb' + str(args.pix_per_beam) + '.fits'
    args.mockcube_filename = cube_output_path + instrument.path + 'mock_ifu' + '_z' + str(args.z) + args.mergeHII_text + '_ppb' + str(args.pix_per_beam) + '_exp' + str(args.exptime) + 's_snr' + str(args.snr) + '.fits'

    if os.path.exists(args.mockcube_filename) and not args.clobber:
        myprint('Reading noisy mock ifu from already existing file ' + args.mockcube_filename + ', use --args.clobber to overwrite', args)
    else:
        if os.path.exists(args.smoothed_cube_filename) and not args.clobber:
            myprint('Reading from already existing no-noise cube file ' + args.mockcube_filename + ', use --args.clobber to overwrite', args)
        else:
            myprint('Noisy or no-noise mock cube file does not exist. Creating now..', args)
            ifu = mockcube(args, instrument, linelist)  # declare the noiseless mock datacube object
            ifu = spatial_convolve(ideal_ifu, ifu, args) # spatial convolution based on obs_spatial_res
            ifu = spectral_bin(ifu, args) # spectral rebinning based on obs_spec_res
            write_fitsobj(args.smoothed_cube_filename, ifu, instrument, args, for_qfits=True) # writing smoothed, no-noise cube into FITS file

        ifu = noisycube(args, instrument, linelist)  # declare the noisy mock datacube object
        ifu.data = readcube(args.smoothed_cube_filename, args).data # reading in and paste no-noise ifu data in to 'data' attribute of ifu, so that all other attributes of ifu can be used too

        if ifu.snr > 0: # otherwise no point of adding noise
            ifu = add_noise(ifu, instrument, args) # adding noise based on snr

        write_fitsobj(args.mockcube_filename, ifu, instrument, args, for_qfits=True) # writing into FITS file

    ifu = readcube(args.mockcube_filename, args)
    myprint('Mock cube ready in %s minutes' % ((time.time() - start_time) / 60), args)

    return ifu, args

# ----------------------------------------------------------------------------
def wrap_get_mock_datacube(args):
    '''
    Function to wrap get_mock_datacube(), so that this script can be used in a fully callable manner if necessary;
    This makes basic gatekeeping checks, calls previous functions in the chronology if necessary, and then calls
    get_mock_datacube() directly passes on the returned variables
    '''
    linelist = mmg.read_linelist(mappings_lab_dir + 'targetlines.txt')  # list of emission lines

    instrument = telescope(args)  # declare an instrument
    cube_output_path = get_cube_output_path(args)
    args.idealcube_filename = cube_output_path + instrument.path + 'ideal_ifu' + '_z' + str(args.z) + args.mergeHII_text + '.fits'

    if os.path.exists(args.idealcube_filename):
        myprint('Ideal cube file exists ' + args.idealcube_filename, args)
    else:
        myprint('Ideal cube file does not exist, calling make_ideal_cube.py..', args)
        ideal_ifu, paramlist, args = get_ideal_datacube(args, linelist)

    ideal_ifu = readcube(args.idealcube_filename, args) # read in the ideal datacube

    mock_ifu, args = get_mock_datacube(ideal_ifu, args, linelist, cube_output_path)
    return mock_ifu, ideal_ifu, args

# -----------------------------------------------------------------------------
if __name__ == '__main__':
    args = parse_args('8508', 'RD0042')
    if not args.keep: plt.close('all')

    # ----------iterating over diag_arr and Om_ar to find the correct file with emission line fluxes-----------------------
    for diag in args.diag_arr:
        args.diag = diag
        for Om in args.Om_arr:
            args.Om = Om
            mock_ifu, ideal_ifu, args = wrap_get_mock_datacube(args)

    myprint('Done making ideal datacubes for all given args.diag_arr and args.Om_arr', args)





