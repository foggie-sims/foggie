##!/usr/bin/env python3

"""

    Title :      make_mock_datacube
    Notes :      Produces a mock IFU datacube by spatial convolution & spectral binning & noise addition to an ideal datacube, for a given seeing + spectral resolution + SNR
    Output :     FITS cube
    Author :     Ayan Acharyya
    Started :    January 2021
    Example :    run make_mock_datacube.py --system ayan_local --halo 8508 --output RD0042 --mergeHII 0.04 --inc 0 --arc 1.0 --vres 60

"""
from header import *
import make_mappings_grid as mmg
from make_ideal_datacube import *

# ---------------------object containing info about the mock ifu datacube: sub class of ideal datacube----------------------------------------
class mockcube(idealcube):
    # ---------initialise object-----------
    def __init__(self, args, instrument, linelist):
        idealcube.__init__(self, args, instrument, linelist) # getting most attributes from the parent class 'idealcube'

        # observed resolutions with which the mock cube is made, depending on the instrument being simulated
        self.obs_spatial_res = instrument.obs_spatial_res * np.pi / (3600 * 180) * (self.distance * 1e3) # since instrument resolution is in arcsec, converting to kpc (self.distance is in Mpc)
        self.obs_spec_res = instrument.obs_spec_res # km/s
        self.pix_per_beam = args.pix_per_beam

        self.smoothed_ndisp = self.ndisp
        self.smoothed_box_size_in_pix = int(round(2 * args.galrad / (self.obs_spatial_res/self.pix_per_beam))) # the size of the mock ifu cube in pixels
        self.pixel_size = 2 * args.galrad / self.smoothed_box_size_in_pix # pixel size in kpc
        self.achieved_spatial_res = self.pixel_size * self.pix_per_beam # kpc (this can vary very slightly from the 'intended' obs_spatial_res

        self.data = np.zeros((self.smoothed_box_size_in_pix, self.smoothed_box_size_in_pix, self.smoothed_ndisp)) # initialise datacube with zeroes

        # compute kernels to convolve by
        if args.kernel == 'gauss':
            self.sigma = gf2s * self.pix_per_beam
            self.ker_size = int((self.sigma * args.ker_factor) // 2 * 2 + 1) # rounding off to nearest odd integer because kernels need odd integer as size
            self.kernel = con.Gaussian2DKernel(self.sigma, x_size=self.ker_size, y_size=self.ker_size)
        elif args.ker == 'moff':
            self.sigma = self.pix_per_beam / (2 * np.sqrt(2 ** (1. / args.moff_beta) - 1.))
            self.ker_size = int((self.sigma * args.ker_factor) // 2 * 2 + 1) # rounding off to nearest odd integer because kernels need odd integer as size
            self.kernel = con.Moffat2DKernel(self.sigma, args.moff_beta, x_size=self.ker_size, y_size=self.ker_size)

        self.declare_obs_param(args)

    # ---------print observation parameters-----------
    def declare_obs_param(self, args):
        myprint('For spectral res= ' + str(self.obs_spec_res) + ' km/s, and wavelength range of ' + str(self.rest_wave_range[0]) + ' to ' + str(self.rest_wave_range[1]) + ' A, length of dispersion axis= ' + str(self.smoothed_ndisp) + ' pixels', args)
        myprint('For spatial res= ' + str(args.obs_spatial_res) + ' arcsec on sky => physical res at redshift ' + str(self.z) + ' galaxy= ' + str(self.obs_spatial_res) + ' kpc, and pixel per beam = ' + str(self.pix_per_beam) + ', size of smoothed box= ' + str(self.smoothed_box_size_in_pix) + ' pixels', args)
        myprint('Going to convolve with ' + args.kernel + ' kernel with FWHM = ' + str(self.pix_per_beam) + ' pixels (' + str(self.achieved_spatial_res) + ' kpc) => sigma = ' + str(self.sigma) + ' pixels, and total size of smoothing kernel = ' + str(self.ker_size) + ' pixels', args)

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
    ifu = mockcube(args, instrument, linelist) # declare the datacube object
    args.mockcube_filename = cube_output_path + instrument.path + 'mock_ifu' + '_z' + str(ifu.z) + args.mergeHII_text + '_ppb' + str(ifu.pix_per_beam) + '.fits'

    if os.path.exists(args.mockcube_filename) and not args.clobber:
        myprint('Reading from already existing file ' + args.mockcube_filename + ', use --args.clobber to overwrite', args)
    else:
        myprint('Mock cube file does not exist. Creating now..', args)

        # -------just for testing: simple rebinning of ideal data cube------------------
        for slice in range(np.shape(ideal_ifu)[2]):
            myprint('Rebinning & convolving slice ' + str(slice + 1) + ' of ' + str(np.shape(ideal_ifu)[2]) + '..', args)
            rebinned_slice = rebin(ideal_ifu[:, :, slice], (ifu.smoothed_box_size_in_pix, ifu.smoothed_box_size_in_pix)) # rebinning before convolving
            ifu.data[:, :, slice] = con.convolve_fft(rebinned_slice, ifu.kernel, normalize_kernel=True) # convolving with kernel

        write_fitsobj(args.mockcube_filename, ifu, args, for_qfits=True) # writing into FITS file

    ifu = readcube(args.mockcube_filename, args)
    myprint('Done in %s minutes' % ((time.time() - start_time) / 60), args)

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

    mock_ifu, args = get_mock_datacube(ideal_ifu.data, args, linelist, cube_output_path)
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





