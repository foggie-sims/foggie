##!/usr/bin/env python3

"""

    Title :      util
    Notes :      Contains various generic utility functions and classes used by the other scripts, including a 'master' function to parse args
    Author :     Ayan Acharyya
    Started :    March 2021

"""
from header import *

# ------------------------------------------------------------------
class BlitManager:
    def __init__(self, canvas, animated_artists=()):
        """
        This class was borrowed from https://matplotlib.org/devdocs/tutorials/advanced/blitting.html?highlight=blitting
        Parameters
        ----------
        canvas : FigureCanvasAgg
            The canvas to work with, this only works for sub-classes of the Agg
            canvas which have the `~FigureCanvasAgg.copy_from_bbox` and
            `~FigureCanvasAgg.restore_region` methods.

        animated_artists : Iterable[Artist]
            List of the artists to manage
        """
        self.canvas = canvas
        self._bg = None
        self._artists = []

        for a in animated_artists:
            self.add_artist(a)
        # grab the background on every draw
        self.cid = canvas.mpl_connect("draw_event", self.on_draw)

    def on_draw(self, event):
        """Callback to register with 'draw_event'."""
        cv = self.canvas
        if event is not None:
            if event.canvas != cv:
                raise RuntimeError
        self._bg = cv.copy_from_bbox(cv.figure.bbox)
        self._draw_animated()

    def add_artist(self, art):
        """
        Add an artist to be managed.

        Parameters
        ----------
        art : Artist

            The artist to be added.  Will be set to 'animated' (just
            to be safe).  *art* must be in the figure associated with
            the canvas this class is managing.

        """
        if art.figure != self.canvas.figure:
            raise RuntimeError
        art.set_animated(True)
        self._artists.append(art)

    def _draw_animated(self):
        """Draw all of the animated artists."""
        fig = self.canvas.figure
        for a in self._artists:
            fig.draw_artist(a)

    def update(self):
        """Update the screen with animated artists."""
        cv = self.canvas
        fig = cv.figure
        # paranoia in case we missed the draw event,
        if self._bg is None:
            self.on_draw(None)
        else:
            # restore the background
            cv.restore_region(self._bg)
            # draw all of the animated artists
            self._draw_animated()
            # update the GUI state
            cv.blit(fig.bbox)
        # let the GUI event loop process anything it has to do
        cv.flush_events()

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
            #myprint('Unknown instrument: ' + self.name + '; using user input/default attributes', args)
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
        self.error = np.zeros((self.box_size_in_pix, self.box_size_in_pix, self.ndisp)) # initialise errorcube with zeroes
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
    Function to re-direct to print_mpi(), for backwards compatibility
    '''
    print_mpi(text, args)

# -------------------------------------------------------------------------------------------
def print_mpi(string, args):
    '''
    Function to print corresponding to each mpi thread
    '''
    comm = MPI.COMM_WORLD
    myprint_orig('[' + str(comm.rank) + '] {' + subprocess.check_output(['uname -n'],shell=True)[:-1].decode("utf-8") + '} ' + string + '\n', args)

# -------------------------------------------------------------------------------------------
def print_master(string, args):
    '''
    Function to print only if on the head node/thread
    '''
    comm = MPI.COMM_WORLD
    if comm.rank == 0: myprint_orig('[' + str(comm.rank) + '] ' + string + '\n', args)

# -------------------------------------------------------------------------------------------
def myprint_orig(text, args):
    '''
    Function to direct the print output to stdout or a file, depending upon user args
    '''
    if not isinstance(text, list) and not text[-1] == '\n': text += '\n'
    if 'minutes' in text: text = fix_time_format(text, 'minutes')
    elif 'mins' in text: text = fix_time_format(text, 'mins')

    if not args.silent:
        if args.print_to_file:
            ofile = open(args.printoutfile, 'a')
            ofile.write(text)
            ofile.close()
        else:
            print(text)

# --------------------------------------------------------------------------------------------
def fix_time_format(text, keyword):
    '''
     Function to modify the way time is formatted in print statements
    '''
    arr = text.split(' ' + keyword)
    pre_time = ' '.join(arr[0].split(' ')[:-1])
    this_time = float(arr[0].split(' ')[-1])
    post_time = ' '.join(arr[1].split(' '))
    text = pre_time + ' %s' % (datetime.timedelta(minutes=this_time)) + post_time

    return text

# ------------------------------------------------------------------------
def insert_line_in_file(line, pos, filename, output=None):
    '''
    Function for nserting a line in a file
    '''
    f = open(filename, 'r')
    contents = f.readlines()
    f.close()

    if pos == -1: pos = len(contents)  # to append to end of file
    contents.insert(pos, line)

    if output is None: output = filename
    f = open(output, 'w')
    contents = ''.join(contents)
    f.write(contents)
    f.close()
    return

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

    photgrid['N2O2'] = np.divide(photgrid['NII6584'], (photgrid['OII3727'] + photgrid['OII3729']))
    Z =  get_KD02_metallicity_from_ratios(photgrid)
    return Z

# ---------------------------------------------------------------------------------
def get_D16_metallicity(photgrid):
    '''
    Function to compute D16 metallicity from an input pandas dataframe with line fluxes as columns
    '''
    photgrid['N2S2'] = np.divide(photgrid['NII6584'], (photgrid['SII6730'] + photgrid['SII6717']))
    photgrid['N2Ha'] = np.divide(photgrid['NII6584'], photgrid['H6562'])
    Z =  get_D16_metallicity_from_ratios(photgrid)
    return Z

# ---------------------------------------------------------------------------------
def get_PPN2_metallicity(photgrid):
    '''
    Function to compute PP04 N2/Ha metallicity from an input pandas dataframe with line fluxes as columns
    '''

    photgrid['N2Ha'] = np.divide(photgrid['NII6584'], photgrid['H6562'])
    Z = get_PPN2_metallicity_from_ratios(photgrid)
    return Z

# ---------------------------------------------------------------------------------
def get_D16_metallicity_from_ratios(photgrid):
    '''
    Function to compute D16 metallicity from an input pandas dataframe with line ratios as columns
    '''

    log_ratio = np.log10(photgrid['N2S2']) + 0.264 * np.log10(photgrid['N2Ha'])
    logOH = log_ratio + 0.45 * (log_ratio + 0.3) ** 5  # + 8.77
    Z = 10 ** logOH  # converting to Z (in units of Z_sol) from log(O/H) + 12
    return Z

# ---------------------------------------------------------------------------------
def get_PPN2_metallicity_from_ratios(photgrid):
    '''
    Function to compute PP04 N2/Ha metallicity from an input pandas dataframe with line ratios as columns
    '''

    log_ratio = np.log10(photgrid['N2Ha'])
    logOH = 9.37 + 2.03 * log_ratio + 1.26 * log_ratio ** 2 + 0.32 * log_ratio ** 3 # from Pettini & Pagel 2004 eq 2
    logOHsol = 8.66 # from PP04
    Z = 10 ** (logOH - logOHsol)  # converting to Z (in units of Z_sol) from log(O/H) + 12
    return Z

# ---------------------------------------------------------------------------------
def get_KD02_metallicity_from_ratios(photgrid):
    '''
    Function to compute KD02 metallicity from an input pandas dataframe with line ratios as columns
    '''

    log_ratio = np.log10(photgrid['N2O2'])
    logOH = 1.54020 + 1.26602 * log_ratio + 0.167977 * log_ratio ** 2
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

# -------------------------------------------------------------------------------------------------
def get_measured_cube_filename(parentfile):
    '''
    Function to derive the name of the measured cube file, based on the name of the fits file being fitted for emission lines
    '''
    return os.path.split(parentfile)[0] + '/measured_cube_' + os.path.split(parentfile)[1]

# ------------------------------------------------------------------
def saveplot(fig, args, plot_suffix, outputdir=None):
    '''
    Function to save plots with a consistent nomenclature
    '''
    if not outputdir: outputdir = args.output_dir + 'figs/' + args.output + '/'
    Path(outputdir).mkdir(parents=True, exist_ok=True)  # creating the directory structure, if doesn't exist already

    outputname = plot_suffix
    if args.mergeHII_text not in outputname: outputname = outputname + args.mergeHII_text
    if args.without_outlier not in outputname: outputname = outputname + args.without_outlier

    outplotname = outputdir + outputname + '.pdf'
    fig.savefig(outplotname)
    myprint('Saved plot as ' + outplotname, args)

# ------------------------------------------------------------------
def movefiles():
    '''
    Function to move files in to appropriate directories
    '''
    start_time = time.time()

    for (thishalo, thissim) in all_sims:
        print('Starting', thishalo, thissim, end=' ')

        source = '/Users/acharyya/Work/astro/foggie_outputs/plots_halo_00'+thishalo+'/nref11c_nref9f/figs'
        destination = source + '/' + thissim
        Path(destination).mkdir(parents=True, exist_ok=True)

        files = os.listdir(source)
        count = 0

        for thisfile in files:
            if thisfile.startswith(thissim) and not os.path.isdir(source + '/' + thisfile): # making sure it is a file (not a directory) starting with the snapshot's name
                shutil.move(source + '/' + thisfile, destination + '/' + thisfile)
                count += 1
        print(count, 'files moved')

    print('Moving done in %s minutes' % ((time.time() - start_time) / 60))

# ---------------------object containing info about ifu datacubes that have been read in-----------------------------
class readcube(object):
    # ---------initialise object-----------
    def __init__(self, filename, args):
        '''
        Function to read a fits file (that has been written by write_fitsobj) and store the fits data in an object
        '''
        myprint('Reading in cube file ' + filename, args)
        cube = fits.open(filename)
        self.data = cube[0].data # reading in just the 3D data cube
        self.wavelength = cube[1].data
        self.header = cube[0].header

        self.data = np.nan_to_num(self.data) # replacing all NaN values with 0, otherwise calculations get messed up
        self.data = self.data.swapaxes(0, 2) # switching from (wave, pos, pos) arrangement (QFitsView requires) to (pos, pos, wave) arrangement (traditional)
        self.error = np.zeros_like(self.data) # assigning to with zeros for now, to be filled in later IF error array is available in the fits file

        if len(cube) > 2:
            additional_data = cube[2].data
            if len(np.shape(additional_data)) == 2: # then this is the 2D counts map attribute
                self.counts = additional_data
                myprint('Reading in available counts map', args)
            elif len(np.shape(additional_data)) == 3: # then this is the 3D error cube attribute
                self.error = additional_data
                self.error = np.nan_to_num(self.error)  # replacing all NaN values with 0, otherwise calculations get messed up
                self.error = self.error.swapaxes(0, 2)  # switching from (wave, pos, pos) arrangement (QFitsView requires) to (pos, pos, wave) arrangement (traditional)
                myprint('Reading in available error cube', args)

# ---------------------object containing info about measured data products that have been read in-----------------------------
class read_measured_cube(object):
    # ---------initialise object-----------
    def __init__(self, filename, args):
        '''
        Function to read a fits file (that has been written by write_fitsobj) and store the fits data in an object
        '''
        myprint('Reading in cube file ' + filename, args)
        cube = fits.open(filename)
        self.data = cube[0].data # reading in the measured quantities: flux, velocity and velocity dispersion
        self.error = cube[1].data # associated uncertainties
        self.header = cube[0].header

        self.data = np.nan_to_num(self.data) # replacing all NaN values with 0, otherwise calculations get messed up
        self.data = self.data.swapaxes(0, 2) # switching from (wave, pos, pos) arrangement (QFitsView requires) to (pos, pos, wave) arrangement (traditional)

        self.error = np.nan_to_num(self.error)  # replacing all NaN values with 0, otherwise calculations get messed up
        self.error = self.error.swapaxes(0, 2)  # switching from (wave, pos, pos) arrangement (QFitsView requires) to (pos, pos, wave) arrangement (traditional)

        self.fitted_lines = np.array(self.header['labels'].split(','))
        self.measured_quantities = np.array(self.header['measured_quantities'].split(','))
        if 'derived_quantities' in self.header: self.derived_quantities = np.array(self.header['derived_quantities'].split(','))

        args.pixel_size_kpc = self.header['pixel_size(kpc)']

    # ----------------------------------
    def get_line_prop(self, line):
        '''
        Function to assimilate all properties of a given emission line into a cube and uncertainty_cube
        '''
        line_index = np.where(self.fitted_lines == line)[0][0]
        n_mq = len(self.measured_quantities)

        line_prop = self.data[:, :, n_mq * line_index : n_mq * (line_index + 1)]
        line_prop_u = self.error[:, :, n_mq * line_index : n_mq * (line_index + 1)]

        return line_prop, line_prop_u

    # ----------------------------------
    def get_all_lines(self, property):
        '''
        Function to assimilate a given property for all emission lines into a cube and uncertainty_cube
        '''
        property_index = np.where(self.measured_quantities == property)[0][0]
        this_prop = self.data[:, :, property_index] # property for first line, i.e.e index= 0
        this_prop_u = self.error[:, :, property_index]

        for index in range(1, len(self.fitted_lines)):
            this_prop = np.dstack([this_prop, self.data[:, :, index * len(self.measured_quantities) + property_index]])
            this_prop_u = np.dstack([this_prop_u, self.error[:, :, index * len(self.measured_quantities) + property_index]])

        return this_prop, this_prop_u

    # ----------------------------------
    def get_derived_prop(self, property):
        '''
        Function to extract a derived (not associated to a particular emission line) property into a 2D map and uncertainty_map
        '''
        property_index = np.where(self.derived_quantities == property)[0][0]
        n_mq = len(self.measured_quantities)
        n_fl = len(self.fitted_lines)

        derived_prop = self.data[:, :, n_mq * n_fl + property_index]
        derived_prop_u = self.error[:, :, n_mq * n_fl + property_index]

        return derived_prop, derived_prop_u

# -------------------------------------------------------------------------------------------
def write_fitsobj(filename, cube, instrument, args, fill_val=np.nan, for_qfits=True, measured_cube=False, measured_quantities=None, derived_quantities=None):
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
                               'CRVAL1': -1 * args.galrad, \
                               'CDELT1': 2 * args.galrad / np.shape(cube.data)[1], \
                               'CTYPE1': 'kpc', \
                               'CRPIX2': 1, \
                               'CRVAL2': -1 * args.galrad, \
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

    if hasattr(cube, 'snr'):  # i.e. it is a smoothed AND noisy mock datacube rather than an ideal datacube, and therefore has an additional errorcube associated with the datacube
        flux_header.update({'exptime(s)': cube.exptime, \
                            'target_snr': cube.snr, \
                            'electrons_per_photon': instrument.el_per_phot, \
                            'telescope_radius(m)': instrument.radius, \
                            })

    if measured_quantities is not None: # i.e. it is a measured data cube (post line fitting) and we want to store what the measured quantities actually are, in the header
        flux_header.update({'measured_quantities': ','.join(measured_quantities)})

    if derived_quantities is not None: # i.e. consists of derived (in addition to measured) quantities e.g. metallicity
        flux_header.update({'derived_quantities': ','.join(derived_quantities)})

    flux_hdu = fits.PrimaryHDU(flux, header=flux_header)
    wavelength_hdu = fits.ImageHDU(wavelength)

    if hasattr(cube, 'counts'): # number of hii regions contributing to each pixel; ONLY for ideal data cubes
        counts = np.ma.filled(cube.counts, fill_value=fill_val)
        counts_hdu = fits.ImageHDU(counts)
        hdulist = fits.HDUList([flux_hdu, wavelength_hdu, counts_hdu])
    elif hasattr(cube, 'error'): # error sttribute is only available for noisy data cubes
        error = np.ma.filled(cube.error, fill_value=fill_val)
        error = error.swapaxes(0,2) # QFitsView requires (wave, pos, pos) arrangement rather than (pos, pos, wave)  arrangement
        error_hdu = fits.ImageHDU(error)
        if measured_cube: hdulist = fits.HDUList([flux_hdu, error_hdu]) # this is a measured parameters file, so no need to save the dispersion array
        else: hdulist = fits.HDUList([flux_hdu, wavelength_hdu, error_hdu])
    else:
        hdulist = fits.HDUList([flux_hdu, wavelength_hdu])
    hdulist.writeto(filename, clobber=True)
    myprint('Written file ' + filename + '\n', args)

# --------------------------------------------------------------------------------------------
def get_all_sims(args):
    '''
    Function assimilate the names of all halos and snapshots available in the given directory
    '''
    if args.do_all_halos: halos = get_all_halos(args)
    else: halos = args.halo_arr

    all_sims = []
    for index, thishalo in enumerate(halos):
        args.halo = thishalo
        thishalo_sims = get_all_sims_for_this_halo(args, given_path = args.foggie_dir + args.run_loc)
        all_sims = np.vstack([all_sims, thishalo_sims]) if index else thishalo_sims

    return all_sims

# --------------------------------------------------------------------------------------------
def get_all_halos(args):
    '''
    Function assimilate the names of all halos in the given directory
    '''
    if 'ayan_' in args.system:
        halos = ['8508', '5036', '5016', '4123', '2392', '2878'] # we exactly know the list of halos present in ayan's local (HD) or pleiades sytems and want them to be accessed in this specific order
    else:
        halo_paths = glob.glob(args.foggie_dir + 'halo_*')
        halo_paths.sort(key=os.path.getmtime)
        halos = [item.split('/')[-1][7:] for item in halo_paths]
    return halos

# --------------------------------------------------------------------------------------------
def get_all_sims_for_this_halo(args, given_path=None):
    '''
    Function assimilate the names of all snapshots available for the given halo
    '''
    all_sims = []
    if given_path is None: given_path = args.foggie_dir + args.run_loc
    snapshot_paths = glob.glob(given_path + '*/')
    if args.use_onlyDD or args.use_onlyRD: snapshot_paths.sort() # alpha-numeric sort if it is just DDs or just RDs because that ensures monotonicity in redshift
    else: snapshot_paths.sort(key=os.path.getmtime) # sort by timestamp
    snapshots = [item.split('/')[-2] for item in snapshot_paths]
    for thissnap in snapshots:
        if len(thissnap) == 6:
            if args.use_onlyRD:
                if thissnap[:2] == 'RD' and float(thissnap[2:]) >= args.snap_start and float(thissnap[2:]) <= args.snap_stop: all_sims.append([args.halo, thissnap])
            elif args.use_onlyDD:
                if thissnap[:2] == 'DD' and float(thissnap[2:]) >= args.snap_start and float(thissnap[2:]) <= args.snap_stop: all_sims.append([args.halo, thissnap])
            else:
                all_sims.append([args.halo, thissnap])
    all_sims = all_sims[:: args.nevery]
    return all_sims

# ---------------------------------------------------------------------------------------------
def pull_halo_redshift(args):
    '''
    Function to pull the current redshift of the halo (WITHOUT having to load the simulation), by matching with the corresponding halo_c_v file
    '''
    halo_cat_file = args.code_path + 'halo_infos/00' + args.halo + '/nref11c_nref9f/halo_c_v'
    #df = pd.read_csv(halo_cat_file, comment='#', sep='\s+|')
    #try: z = df.loc[df['name']==args.output, 'redshift'].values[0]
    # shifted from pandas to astropy because pandas runs in to weird error on pleiades
    df = Table.read(halo_cat_file, format='ascii')
    try: z = float(df['col2'][np.where(df['col3']==args.output)[0][0]])
    except IndexError: # if this snapshot is not yet there in halo_c_v file
        if args.halo == '4123' and args.output == 'RD0038': z = 0.15
        else: z = -99
    return z

# ----------------------------------------------------------------------------------------------
def pull_halo_center(args, fast=False):
    '''
    Function to pull halo center from halo catalogue, if exists, otherwise compute halo center
    Adapted from utils.foggie_load()
    '''

    #halos_df_name = args.code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/' + 'halo_c_v'
    halos_df_name = args.code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/' + 'halo_cen_smoothed' # changed on Aug 30, after Cassi updated smoothed halo centers

    if os.path.exists(halos_df_name):
        halos_df = pd.read_table(halos_df_name, sep='|')
        halos_df.columns = halos_df.columns.str.strip() # trimming column names of extra whitespace
        try:
            halos_df['name'] = halos_df['name'].str.strip() # trimming column 'name' of extra whitespace
        except:
            halos_df['name'] = halos_df['snap'].str.strip() # trimming column 'name' of extra whitespace

        if halos_df['name'].str.contains(args.output).any():
            myprint('Pulling halo center from catalog file', args)
            halo_ind = halos_df.index[halos_df['name'] == args.output][0]
            args.halo_center = halos_df.loc[halo_ind, ['xc', 'yc', 'zc']].values # in kpc units
            try: args.halo_velocity = halos_df.loc[halo_ind, ['xv', 'yv', 'zv']].values # in km/s units
            except: pass
            calc_hc = False
        elif not fast:
            myprint('This snapshot is not in the halos_df file, calculating halo center...', args)
            calc_hc = True
    elif not fast:
        myprint("This halos_df file doesn't exist, calculating halo center...", args)
        calc_hc = True
    if calc_hc:
        ds, refine_box = load_sim(args, region='refine_box', halo_c_v_name=halos_df_name)
        args.halo_center = ds.halo_center_kpc
        args.halo_velocity = ds.halo_velocity_kms
        return args, ds, refine_box
    else:
        return args

# --------------------------------------------------------------------------------------------------------------
def parse_args(haloname, RDname, fast=False):
    '''
    Function to parse keyword arguments
    '''

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='''identify satellites in FOGGIE simulations''')
    # ---- common args used widely over the full codebase ------------
    parser.add_argument('--system', metavar='system', type=str, action='store', default='ayan_local', help='Which system are you on? Default is Jase')
    parser.add_argument('--do', metavar='do', type=str, action='store', default='gas', help='Which particles do you want to plot? Default is gas')
    parser.add_argument('--run', metavar='run', type=str, action='store', default='nref11c_nref9f', help='which run? default is natural')
    parser.add_argument('--halo', metavar='halo', type=str, action='store', default=haloname, help='which halo? default is 8508 (Tempest)')
    parser.add_argument('--projection', metavar='projection', type=str, action='store', default='x', help='Which projection do you want to plot, i.e., which axis is your line of sight? Default is x')
    parser.add_argument('--output', metavar='output', type=str, action='store', default=RDname, help='which output? default is RD0020')
    parser.add_argument('--foggie_dir', metavar='foggie_dir', type=str, action='store', default=None, help='Specify which directory the dataset lies in, otherwise, by default it will use the args.system variable to determine the FOGGIE data location')
    parser.add_argument('--pwd', dest='pwd', action='store_true', default=False, help='Just use the current working directory?, default is no')
    parser.add_argument('--do_all_sims', dest='do_all_sims', action='store_true', default=False, help='Run the code on all simulation snapshots available for a given halo?, default is no')
    parser.add_argument('--use_onlyRD', dest='use_onlyRD', action='store_true', default=False, help='Use only the RD snapshots available for a given halo?, default is no')
    parser.add_argument('--use_onlyDD', dest='use_onlyDD', action='store_true', default=False, help='Use only the DD snapshots available for a given halo?, default is no')
    parser.add_argument('--nevery', metavar='nevery', type=int, action='store', default=1, help='use every nth snapshot when do_all_sims is specified; default is 1 i.e., all snapshots will be used')
    parser.add_argument('--snap_start', metavar='snap_start', type=int, action='store', default=0, help='index of the DD or RD snapshots to start from, when using --do_all_sims; default is 0')
    parser.add_argument('--snap_stop', metavar='snap_stop', type=int, action='store', default=10000, help='index of the DD or RD snapshots to stop at, when using --do_all_sims; default is 10000')
    parser.add_argument('--do_all_halos', dest='do_all_halos', action='store_true', default=False, help='loop over all available halos (and all snapshots each halo has)?, default is no')
    parser.add_argument('--silent', dest='silent', action='store_true', default=False, help='Suppress all print statements?, default is no')
    parser.add_argument('--galrad', metavar='galrad', type=float, action='store', default=20., help='the radial extent (in each spatial dimension) to which computations will be done, in kpc; default is 20')
    parser.add_argument('--fullbox', dest='fullbox', action='store_true', default=False, help='Use full refine box, ignoring args.galrad?, default is no')

    # ------- args added for projection_plot.py ------------------------------
    parser.add_argument('--age_thresh', metavar='age_thresh', type=float, action='store', default=10., help='age threshold to decide young/not-young stars, in Myr; default is 10')
    parser.add_argument('--noplot', dest='noplot', action='store_true', default=False, help='Skip all plotting steps?, default is no')
    parser.add_argument('--noweight', dest='noweight', action='store_true', default=False, help='Skip weighting the projection?, default is no')
    parser.add_argument('--makerotmovie', dest='makerotmovie', action='store_true', default=False, help='Make a rotation projection movie?, default is no')
    parser.add_argument('--nframes', metavar='nframes', type=int, action='store', default=200, help='total number of frames in movie, i.e. number of parts to divide the full 2*pi into; default is 200')
    parser.add_argument('--rot_normal_by', metavar='rot_normal_by', type=float, action='store', default=0, help='rotation (in degrees) for the normal vector? default is 0, i.e. first slice = no rotation')
    parser.add_argument('--rot_normal_about', metavar='rot_normal_about', type=str, action='store', default='x', help='Which axis to rotate the normal vector about? Default is x')
    parser.add_argument('--rot_north_by', metavar='rot_north_by', type=float, action='store', default=0, help='rotation (in degrees) for the north vector? default is 0, i.e. first slice = no rotation')
    parser.add_argument('--rot_north_about', metavar='rot_north_about', type=str, action='store', default='y', help='Which axis to rotate the north vector about? Default is x')
    parser.add_argument('--do_central', dest='do_central', action='store_true', default=False, help='Do central refine box projection?, default is no')
    parser.add_argument('--add_arrow', dest='add_arrow', action='store_true', default=False, help='Add arrows?, default is no')
    parser.add_argument('--add_velocity', dest='add_velocity', action='store_true', default=False, help='Add velocity?, default is no')
    parser.add_argument('--hide_axes', dest='hide_axes', action='store_true', default=False, help='Hide all axes?, default is no')
    parser.add_argument('--annotate_grids', dest='annotate_grids', action='store_true', default=False, help='annotate grids?, default is no')
    parser.add_argument('--annotate_box', metavar='annotate_box', type=str, action='store', default=None, help='comma separated comoving kpc values for annotate boxes, default is None')
    parser.add_argument('--min_level', dest='min_level', type=int, action='store', default=3, help='annotate grids min level, default is 3')

    # ------- args added for filter_star_properties.py ------------------------------
    parser.add_argument('--plot_proj', dest='plot_proj', action='store_true', default=False, help='plot projection map? default is no')
    parser.add_argument('--clobber', dest='clobber', action='store_true', default=False, help='overwrite existing outputs with same name?, default is no')
    parser.add_argument('--automate', dest='automate', action='store_true', default=False, help='automatically execute the next script?, default is no')
    parser.add_argument('--dryrun', dest='dryrun', action='store_true', default=False, help='if this is just a dryrun only loop over and print statements will be carried out, none of the actual loading datasets and all; useful for debugging on pleiades, default is no')

    # ------- args added for compute_hii_radii.py ------------------------------
    parser.add_argument('--mergeHII', metavar='mergeHII', type=float, action='store', default=None, help='separation btwn HII regions below which to merge them, in kpc; default is None i.e., do not merge')

    # ------- args added for lookup_flux.py ------------------------------
    parser.add_argument('--diag_arr', metavar='diag_arr', type=str, action='store', default='D16', help='list of metallicity diagnostics to use')
    parser.add_argument('--Om_arr', metavar='Om_arr', type=str, action='store', default='0.5', help='list of Omega values to use')
    parser.add_argument('--nooutliers', dest='nooutliers', action='store_true', default=False, help='discard outlier HII regions (according to D16 diagnostic)?, default is no')
    parser.add_argument('--xratio', metavar='xratio', type=str, action='store', default=None, help='ratio of lines to plot on X-axis; default is None')
    parser.add_argument('--yratio', metavar='yratio', type=str, action='store', default=None, help='ratio of lines to plot on Y-axis; default is None')
    parser.add_argument('--fontsize', metavar='fontsize', type=int, action='store', default=15, help='fontsize of plot labels, etc.; default is 15')
    parser.add_argument('--plot_metgrad', dest='plot_metgrad', action='store_true', default=False, help='make metallicity gradient plot?, default is no')
    parser.add_argument('--plot_phase_space', dest='plot_phase_space', action='store_true', default=False, help='make P-r phase space plot?, default is no')
    parser.add_argument('--plot_obsv_phase_space', dest='plot_obsv_phase_space', action='store_true', default=False, help='overlay observed P-r phase space on plot?, default is no')
    parser.add_argument('--plot_fluxgrid', dest='plot_fluxgrid', action='store_true', default=False, help='make flux ratio grid plot?, default is no')
    parser.add_argument('--annotate', dest='annotate', action='store_true', default=False, help='annotate grid plot?, default is no')
    parser.add_argument('--pause', dest='pause', action='store_true', default=False, help='pause after annotating each grid?, default is no')
    parser.add_argument('--plot_Zin_Zout', dest='plot_Zin_Zout', action='store_true', default=False, help='make input vs output metallicity plot?, default is no')
    parser.add_argument('--saveplot', dest='saveplot', action='store_true', default=False, help='save the plot?, default is no')
    parser.add_argument('--keep', dest='keep', action='store_true', default=False, help='keep previously displayed plots on screen?, default is no')
    parser.add_argument('--use_RGI', dest='use_RGI', action='store_true', default=False, help='kuse RGI interpolation vs LND?, default is no')

    # ------- args added for make_ideal_datacube.py ------------------------------
    parser.add_argument('--center_wrt_halo', metavar='center_wrt_halo', type=str, action='store', default='0,0,0', help='where to center the mock ifu data cube relative to the halo center? (x,y,z) position coordinates in kpc; default is (0,0,0) i.e. no offset from halo center')
    parser.add_argument('--obs_wave_range', metavar='obs_wave_range', type=str, action='store', default='0.65,0.68', help='observed wavelength range for the simulated instrument, in micron; default is (0.8, 1.7) microns')
    parser.add_argument('--z', metavar='z', type=float, action='store', default=0.0001, help='redshift of the mock datacube; default is 0.0001 (not 0, so as to avoid flux unit conversion issues)')
    parser.add_argument('--base_wave_range', metavar='base_wave_range', type=str, action='store', default='0.64,0.68', help='wavelength range for the ideal datacube, in micron; default is (0.64, 0.68) microns')
    parser.add_argument('--inclination', metavar='inclination', type=float, action='store', default=0., help='inclination angle to rotate the galaxy by, on a plane perpendicular to projection plane, i.e. if projection is xy, rotation is on yz, in degrees; default is 0')
    parser.add_argument('--vel_disp', metavar='vel_disp', type=float, action='store', default=15., help='intrinsic velocity dispersion for each emission line, in km/s; default is 15 km/s')
    parser.add_argument('--nbin_cont', metavar='nbin_cont', type=int, action='store', default=1000, help='no. of spectral bins to bin the continuum (witout emission lines) in to; default is 1000')
    parser.add_argument('--vel_highres_win', metavar='vel_highres_win', type=float, action='store', default=500., help='velocity window on either side of each emission line, in km/s, within which the continuum is resolved into finer (nbin_highres_cont) spectral elements; default is 500 km/s')
    parser.add_argument('--nbin_highres_cont', metavar='nbin_highres_cont', type=int, action='store', default=100, help='no. of additonal spectral bins to introduce around each emission line; default is 100')
    parser.add_argument('--base_spec_res', metavar='base_spec_res', type=float, action='store', default=30., help='base spectral resolution, in km/s, i.e. to be employed while making the ideal datacube; default is 30 km/s')
    parser.add_argument('--base_spatial_res', metavar='base_spatial_res', type=float, action='store', default=0.04, help='base spatial resolution, in kpc, i.e. to be employed while making the ideal datacube; default is 0.04 kpc = 40 pc')
    parser.add_argument('--print_to_file', dest='print_to_file', action='store_true', default=False, help='Redirect all print statements to a file?, default is no')
    parser.add_argument('--printoutfile', metavar='printoutfile', type=str, action='store', default='./logfile.out', help='file to write all print statements to; default is ./logfile.out')
    parser.add_argument('--instrument', metavar='instrument', type=str, action='store', default='dummy', help='which instrument to simulate?; default is dummy')
    parser.add_argument('--debug', dest='debug', action='store_true', default=False, help='run in debug mode (lots of print checks)?, default is no')

    # ------- args added for make_mock_datacube.py ------------------------------
    parser.add_argument('--obs_spec_res', metavar='obs_spec_res', type=float, action='store', default=30., help='observed spectral resolution of the instrument, in km/s; default is 60 km/s')
    parser.add_argument('--obs_spatial_res', metavar='obs_spatial_res', type=float, action='store', default=1.0, help='observed spatial resolution of the instrument, in arcsec; default is 1.0"')
    parser.add_argument('--pix_per_beam', metavar='pix_per_beam', type=int, action='store', default=6, help='number of pixels to sample the resolution element (PSF) by; default is 6"')
    parser.add_argument('--kernel', metavar='kernel', type=str, action='store', default='gauss', help='which kernel to simulate for seeing, gauss or moff?; default is gauss')
    parser.add_argument('--ker_size_factor', metavar='ker_size_factor', type=int, action='store', default=5, help='factor to multiply kernel sigma by to get kernel size, e.g. if PSF sigma=5 pixel and ker_size_factor=5, kernel size=25 pixel; default is 5"')
    parser.add_argument('--moff_beta', metavar='moff_beta', type=float, action='store', default=4.7, help='beta (power index) in moffat kernel; default is 4.7"')
    parser.add_argument('--snr', metavar='snr', type=float, action='store', default=0, help='target SNR of the datacube; default is 0, i.e. noiseless"')
    parser.add_argument('--tel_radius', metavar='tel_radius', type=float, action='store', default=1, help='radius of telescope, in metres; default is 1 m')
    parser.add_argument('--exptime', metavar='exptime', type=float, action='store', default=1200., help='exposure time of observation, in sec; default is 1200 sec')
    parser.add_argument('--el_per_phot', metavar='el_per_phot', type=float, action='store', default=1, help='how many electrons do each photon trigger in the instrument; default is 1"')

    # ------- args added for test_kit.py ------------------------------
    parser.add_argument('--plot_hist', dest='plot_hist', action='store_true', default=False, help='make histogram plot?, default is no')

    # ------- args added for fit_mock_spectra.py ------------------------------
    parser.add_argument('--test_pixel', metavar='test_pixel', type=str, action='store', default=None, help='pixel to test continuum and flux fitting on; default is None')
    parser.add_argument('--vel_mask', metavar='vel_mask', type=float, action='store', default=500., help='velocity window on either side of emission line, for masking, in km/s; default is 500')
    parser.add_argument('--nres_elements', metavar='nres_elements', type=int, action='store', default=20, help='how many resolution elements to consider on either side of each emission line, for determining group of lines; default is 20')
    parser.add_argument('--testcontfit', dest='testcontfit', action='store_true', default=False, help='run a test of continuum fitting)?, default is no')
    parser.add_argument('--testlinefit', dest='testlinefit', action='store_true', default=False, help='run a test of line fitting)?, default is no')
    parser.add_argument('--doideal', dest='doideal', action='store_true', default=False, help='run on the ideal data cube (and corresponding products)?, default is no')

    # ------- args added for make_mock_measurements.py ------------------------------
    parser.add_argument('--compute_property', metavar='compute_property', type=str, action='store', default='metallicity', help='which property to compute?; default is metallicity')
    parser.add_argument('--write_property', dest='write_property', action='store_true', default=False, help='append the property to existing measured cube file?; default is no')

    # ------- args added for plot_mock_observables.py ------------------------------
    parser.add_argument('--line', metavar='line', type=str, action='store', default='H6562', help='which emission line?; default is H6562')
    parser.add_argument('--get_property', metavar='get_property', type=str, action='store', default='flux', help='which property to get?; default is flux')
    parser.add_argument('--plot_property', dest='plot_property', action='store_true', default=False, help='plot the property?, default is no')
    parser.add_argument('--compare_property', dest='compare_property', action='store_true', default=False, help='plot a comparison of the derived property with corresponding intrinsic values from yt?, default is no')
    parser.add_argument('--iscolorlog', dest='iscolorlog', action='store_true', default=False, help='set color of property to log scale?, default is no')
    parser.add_argument('--cmin', metavar='cmin', type=float, action='store', default=None, help='minimum value for plotting imshow colorbar; default is None')
    parser.add_argument('--cmax', metavar='cmax', type=float, action='store', default=None, help='maximum value for plotting imshow colorbar; default is None')

    # ------- args added for investigate)metallicity.py ------------------------------
    parser.add_argument('--lookup_metallicity', metavar='lookup_metallicity', type=int, action='store', default=None, help='index of HII region (in the dataframe) to lookup metallicity values for; default is None')
    parser.add_argument('--Zout', metavar='Zout', type=str, action='store', default='D16', help='which metallicity diagnostic to be plotted as Zout; default is D16')
    parser.add_argument('--islog', dest='islog', action='store_true', default=False, help='set x- and y- scale as log?, default is no')
    parser.add_argument('--which_df', metavar='which_df', type=str, action='store', default='H2_summed', help='which dataframe to plot (out of H2_summed, H2_all, 4D_grid & 3D_grid; default is H2_summed')
    parser.add_argument('--plot3d', dest='plot3d', action='store_true', default=False, help='plot 3D grid plots?, default is no')
    parser.add_argument('--age_slice', metavar='age_slice', type=float, action='store', default=1, help='slice of age (in Myr) value for which to make the 3D grid plot; default is 1')
    parser.add_argument('--plotstyle', metavar='plotstyle', type=str, action='store', default='scatter', help='which plot style to use out of map, hexbin, contour, hist and scatter? default is scatter')
    parser.add_argument('--correlation', dest='correlation', action='store_true', default=False, help='compute and display the Spearman correlation on plot?, default is no')
    parser.add_argument('--swap_axes', dest='swap_axes', action='store_true', default=False, help='swan x and y axes on plot?, default is no')

    # ------- args added for volume_rendering_movie.py ------------------------------
    parser.add_argument('--makemovie', dest='makemovie', action='store_true', default=False, help='Accumulate all pngs at the end into a movie?, default is no')
    parser.add_argument('--nmovframes', metavar='nmovframes', type=int, action='store', default=0, help='total number of frames in movie, i.e. number of parts to divide the full 2*pi into; default is 200')
    parser.add_argument('--starting_frot', metavar='starting_frot', type=float, action='store', default=0, help='what fraction of full rotation (0=0, 1 = 2*pi) to start with? default is 0')
    parser.add_argument('--max_frot', metavar='max_frot', type=float, action='store', default=0, help='what fraction of full rotation (0=0, 1 = 2*pi)? default is 0')
    parser.add_argument('--max_zoom', metavar='max_zoom', type=float, action='store', default=1, help='what factor of zoom? default is 1 i.e. no zoom')
    parser.add_argument('--imres', metavar='imres', type=int, action='store', default=256, help='image resolution in pixels x pixels? default is 256 x 256')
    parser.add_argument('--sigma', metavar='sigma', type=int, action='store', default=8, help='sigma clipping? default is 8')
    parser.add_argument('--delay_frame', metavar='delay_frame', type=float, action='store', default=0.1, help='duration per frame of the movie, in sec; default is 0.1 sec')
    parser.add_argument('--annotate_domain', dest='annotate_domain', action='store_true', default=False, help='annotate domain boundaries?, default is no')
    parser.add_argument('--annotate_axes', dest='annotate_axes', action='store_true', default=False, help='annotate coordinate axes?, default is no')
    parser.add_argument('--annotate_redshift', dest='annotate_redshift', action='store_true', default=False, help='annotate current redshift?, default is no')
    parser.add_argument('--move_to', metavar='move_to', type=str, action='store', default='0,0,0', help='move camera position to (x,y,z) position coordinates in domain_length units, relative to starting position; default is (0,0,0) i.e. no movement')
    parser.add_argument('--use_full_colormap', dest='use_full_colormap', action='store_true', default=False, help='map the entire colormap within the given bounds?, default is no')

    # ------- args added for track_metallicity_evolution.py ------------------------------
    parser.add_argument('--nocallback', dest='nocallback', action='store_true', default=False, help='callback previous functions if a file is not found?, default is no')

    # ------- args added for plot_metallicity_evolution.py ------------------------------
    parser.add_argument('--weight', metavar='weight', type=str, action='store', default=None, help='column name of quantity to weight the metallicity by; default is None i.e. no weight')
    parser.add_argument('--nzbins', metavar='nzbins', type=int, action='store', default=10, help='number of redshift bins on heatmap? default is 10')

    # ------- args added for datashader_movie.py ------------------------------
    parser.add_argument('--xcol', metavar='xcol', type=str, action='store', default='rad', help='x axis quantity; default is rad')
    parser.add_argument('--ycol', metavar='ycol', type=str, action='store', default='metal', help='y axis quantity; default is metal')
    parser.add_argument('--colorcol', metavar='colorcol', type=str, action='store', default='vrad', help='x axis quantity; default is vrad')
    parser.add_argument('--clobber_plot', dest='clobber_plot', action='store_true', default=False, help='overwrite existing plots with same name?, default is no')
    parser.add_argument('--overplot_stars', dest='overplot_stars', action='store_true', default=False, help='overplot young stars?, default is no')
    parser.add_argument('--overplot_absorbers', dest='overplot_absorbers', action='store_true', default=False, help='overplot HI absorbers (from Claire)?, default is no')
    parser.add_argument('--start_index', metavar='start_index', type=int, action='store', default=0, help='index of the list of snapshots to start from; default is 0')
    parser.add_argument('--interactive', dest='interactive', action='store_true', default=False, help='interactive mode?, default is no')
    parser.add_argument('--combine', dest='combine', action='store_true', default=False, help='combine all outputs from lasso selection?, default is no')
    parser.add_argument('--selcol', dest='selcol', action='store_true', default=False, help='make a selection in the color space too?, default is no')
    parser.add_argument('--use_cvs_log', dest='use_cvs_log', action='store_true', default=False, help='make the datashader canvas itself in log-scale as opposed to converting the data to log?, default is no')
    parser.add_argument('--inflow_only', dest='inflow_only', action='store_true', default=False, help='only consider gas with negative radial velocity?, default is no')
    parser.add_argument('--outflow_only', dest='outflow_only', action='store_true', default=False, help='only consider gas with positive radial velocity?, default is no')
    parser.add_argument('--use_old_dsh', dest='use_old_dsh', action='store_true', default=False, help='use the old way of making datashader plots?, default is no')
    parser.add_argument('--quick', dest='quick', action='store_true', default=False, help='proceed with only the relevant properties and not store all properties?, default is no')
    parser.add_argument('--nofoggie', dest='nofoggie', action='store_true', default=False, help='skip plotting foggie data?, default is no')

    # ------- args added for flux_tracking_movie.py ------------------------------
    parser.add_argument('--units_kpc', dest='units_kpc', action='store_true', default=False, help='the inner and outer radii of the sphere are in kpc units?, default is no')
    parser.add_argument('--units_rvir', dest='units_rvir', action='store_true', default=False, help='the inner and outer radii of the sphere are in fraction of Rvir?, default is no')
    parser.add_argument('--temp_cut', dest='temp_cut', action='store_true', default=False, help='compute everything broken into cold, cool, warm, and hot gas?, default is no')
    parser.add_argument('--nchunks', metavar='nchunks', type=int, action='store', default=20, help='number of chunks to break up in to; default is 100')
    parser.add_argument('--overplot_source_sink', dest='overplot_source_sink', action='store_true', default=False, help='overplot source and sink terms on flux plots?, default is no')

    # ------- args added for datashader_quickplot.py ------------------------------
    parser.add_argument('--xmin', metavar='xmin', type=float, action='store', default=None, help='minimum xaxis limit; default is None')
    parser.add_argument('--xmax', metavar='xmax', type=float, action='store', default=None, help='maximum xaxis limit; default is None')
    parser.add_argument('--ymin', metavar='ymin', type=float, action='store', default=None, help='minimum yaxis limit; default is None')
    parser.add_argument('--ymax', metavar='ymax', type=float, action='store', default=None, help='maximum yaxis limit; default is None')
    parser.add_argument('--cmap', metavar='cmap', type=str, action='store', default=None, help='colormap to use; default is None')
    parser.add_argument('--ncolbins', metavar='ncolbins', type=int, action='store', default=None, help='number of bins in color space the data shader categories would be split across; default is None')
    parser.add_argument('--nodiskload', dest='nodiskload', action='store_true', default=False, help='skip loading disk-relative stuff in foggie load (saves time)?, default is no')
    parser.add_argument('--diskload', dest='diskload', action='store_true', default=False, help='load disk-relative stuff in foggie load?, default is no')

    # ------- args added for compute_MZgrad.py ------------------------------
    parser.add_argument('--upto_re', metavar='upto_re', type=float, action='store', default=2.0, help='fit metallicity gradient out to what multiple of Re? default is 2')
    parser.add_argument('--upto_kpc', metavar='upto_kpc', type=float, action='store', default=None, help='fit metallicity gradient out to what absolute kpc? default is None')
    parser.add_argument('--docomoving', dest='docomoving', action='store_true', default=False, help='consider the input upto_kpc as a comoving quantity?, default is no')
    parser.add_argument('--write_file', dest='write_file', action='store_true', default=False, help='write the list of measured gradients, mass and size to file?, default is no')
    parser.add_argument('--get_gasmass', dest='get_gasmass', action='store_true', default=False, help='save gas mass profile, in addition to stellar mass profile, to hdf5 file?, default is no')
    parser.add_argument('--snapnumber', metavar='snapnumber', type=int, action='store', default=None, help='identifier for the snapshot (what follows DD or RD); default is None, in which case it takes the snapshot from args.output')

    # ------- args added for get_halo_track.py ------------------------------
    parser.add_argument('--refsize', metavar='refsize', type=float, action='store', default=200, help='width of refine box, in kpc, to make the halo track file; default is 200 kpc')
    parser.add_argument('--reflevel', metavar='reflevel', type=int, action='store', default=7, help='forced refinement level to put in the halo track file; default is 7')
    parser.add_argument('--z_interval', metavar='z_interval', type=float, action='store', default=0.005, help='redshift interval on which to interpolate the halo center track file; default is 0.005')
    parser.add_argument('--root_dir', metavar='root_dir', type=str, action='store', default='', help='root directory path where your foggie directory is (automatically grabs the correct path if you are on ayans system); default is empty string')
    parser.add_argument('--last_center_guess', metavar='last_center_guess', type=str, action='store', default=None, help='initial guess for the center of desired halo in code units')
    parser.add_argument('--compare_tracks', dest='compare_tracks', action='store_true', default=False, help='compare (plot) multiple existing center track files?, default is no')
    parser.add_argument('--width', metavar='width', type=float, action='store', default=500, help='the width of projection plots, in kpc; default is 500 kpc')
    parser.add_argument('--search_radius', metavar='search_radius', type=float, action='store', default=50, help='the radius within which to search for density peak, in comoving kpc; default is 50 ckpc')

    # ------- args added for plot_MZgrad.py ------------------------------
    parser.add_argument('--binby', metavar='binby', type=str, action='store', default=None, help='bin the plot by either redshift or mass; default is empty string = no binning')
    parser.add_argument('--nbins', metavar='nbins', type=int, action='store', default=200, help='no. of bins to bin the binby column in to; default is 200')
    parser.add_argument('--overplot_manga', dest='overplot_manga', action='store_true', default=False, help='overplot MaNGA observed MZGR?, default is no')
    parser.add_argument('--overplot_clear', dest='overplot_clear', action='store_true', default=False, help='overplot CLEAR observed MZGR?, default is no')
    parser.add_argument('--overplot_belfiore', dest='overplot_belfiore', action='store_true', default=False, help='overplot Belfiore+17 observed MZGR?, default is no')
    parser.add_argument('--overplot_mingozzi', dest='overplot_mingozzi', action='store_true', default=False, help='overplot Mongozzi+19 observed MZGR?, default is no')
    parser.add_argument('--overplot_smoothed', metavar='overplot_smoothed', type=float, action='store', default=None, help='overplot temporally smoothed data with <X> Myr smoothing scale, default is None')
    parser.add_argument('--overplot_cadence', metavar='overplot_cadence', type=float, action='store', default=None, help='overplot with only the outputs after every <X> Myr, default is None')
    parser.add_argument('--manga_diag', metavar='manga_diag', type=str, action='store', default='n2', help='which metallicity diagnostic to extract from manga? options are: n2, o3n2, ons, pyqz, t2, m08, t04; default is n2')
    parser.add_argument('--zhighlight', dest='zhighlight', action='store_true', default=False, help='highlight a few integer-ish redshift points on the MZGR?, default is no')
    parser.add_argument('--use_gasre', dest='use_gasre', action='store_true', default=False, help='use measurements based on Re estimated from cold gas clumps (instead of that measured from stellar mass profile)?, default is no')
    parser.add_argument('--use_binnedfit', dest='use_binnedfit', action='store_true', default=False, help='use gradient measurements from radially binned Z profile (as opposed to the fit to individual cells)?, default is no')
    parser.add_argument('--Zgrad_den', metavar='Zgrad_den', type=str, action='store', default='kpc', help='normaliser of Zgrad, either kpc or re; default is kpc')
    parser.add_argument('--plot_deviation', dest='plot_deviation', action='store_true', default=False, help='make additional plot of deviation in gradient vs things like SFR?, default is no')
    parser.add_argument('--plot_timefraction', dest='plot_timefraction', action='store_true', default=False, help='make additional plot of deviation in gradient vs things like SFR?, default is no')
    parser.add_argument('--upto_z', metavar='upto_z', type=float, action='store', default=0, help='calculation of the fraction of time spent outside range will made only up to this lower limit redshift; default is 0')
    parser.add_argument('--Zgrad_allowance', metavar='Zgrad_allowance', type=float, action='store', default=0.05, help='allowance for Zgrad (in dex/kpc) outside the smoothed behaviour; default is +/- 0.05 dex/kpc')
    parser.add_argument('--zcol', metavar='zcol', type=str, action='store', default='sfr', help='x axis quantity for plotting against deviation in MZGR; default is sfr')
    parser.add_argument('--zmin', metavar='zmin', type=float, action='store', default=None, help='minimum xaxis limit; default is None')
    parser.add_argument('--zmax', metavar='zmax', type=float, action='store', default=None, help='maximum xaxis limit; default is None')
    parser.add_argument('--snaphighlight', metavar='snaphighlight', type=str, action='store', default=None, help='highlight any given array of snapshots? default is None')
    parser.add_argument('--nocolorcoding', dest='nocolorcoding', action='store_true', default=False, help='Make the plots without any colorcoding (this ignores even if --colorcol <> is passed)?, default is no')
    parser.add_argument('--hiderawdata', dest='hiderawdata', action='store_true', default=False, help='Hide the main relation (so that only the overplotted or z-highlighted lines remain)?, default is no')
    parser.add_argument('--glasspaper', dest='glasspaper', action='store_true', default=False, help='Set plot axis etc to match the GLASS paper plot?, default is no')

    # ------- args added for compute_Zscatter.py ------------------------------
    parser.add_argument('--res', metavar='res', type=str, action='store', default='0.1', help='spatial sampling resolution, in kpc, to compute the Z statistics; default is 0.1 kpc')
    parser.add_argument('--fit_multiple', dest='fit_multiple', action='store_true', default=False, help='fit one gaussian + one skewed guassian?, default is no')

    # ------- args added for plot_Zevolution.py ------------------------------
    parser.add_argument('--forpaper', dest='forpaper', action='store_true', default=False, help='make plot with certain set panels, specifically for the paper?, default is no')
    parser.add_argument('--doft', dest='doft', action='store_true', default=False, help='make new plot for Fourier Transform?, default is no')
    parser.add_argument('--docorr', metavar='docorr', type=str, action='store', default=None, help='make new plot for time delay cross-correlation with respect to a column?, default is no')

    # ------- wrap up and processing args ------------------------------
    args = parser.parse_args()

    args.diag_arr = [item for item in args.diag_arr.split(',')]
    args.diag = args.diag_arr[0]
    args.Om_arr = [float(item) for item in args.Om_arr.split(',')]
    args.Om = args.Om_arr[0]
    args.halo_arr = [item for item in args.halo.split(',')]
    args.halo = args.halo_arr[0] if len(args.halo_arr) == 1 else haloname
    if args.snapnumber is not None:
        if args.use_onlyDD: args.output = 'DD%04d' % (args.snapnumber)
        else: args.output = 'RD%04d' % (args.snapnumber)
    args.output_arr = [item for item in args.output.split(',')]
    args.res_arr = [float(item) for item in args.res.split(',')]
    args.output = args.output_arr[0] if len(args.output_arr) == 1 else RDname
    args.move_to = np.array([float(item) for item in args.move_to.split(',')])  # kpc
    args.center_wrt_halo = np.array([float(item) for item in args.center_wrt_halo.split(',')])  # kpc
    args.obs_wave_range = np.array([float(item) for item in args.obs_wave_range.split(',')])
    args.base_wave_range = np.array([float(item) for item in args.base_wave_range.split(',')])
    args.test_pixel = np.array([int(item) for item in args.test_pixel.split(',')]) if args.test_pixel is not None else None
    args.colorcol = [item for item in args.colorcol.split(',')]

    args.mergeHII_text = '_mergeHII=' + str(args.mergeHII) + 'kpc' if args.mergeHII is not None else '' # to be used as filename suffix to denote whether HII regions have been merged
    args.without_outlier = '_no_outlier' if args.nooutliers else '' # to be used as filename suffix to denote whether outlier HII regions (as per D16 density criteria) have been discarded

    args.foggie_dir, args.output_dir, args.run_loc, args.code_path, args.trackname, args.haloname, args.spectra_dir, args.infofile = get_run_loc_etc(args)
    try:
        args = pull_halo_center(args, fast=fast) # pull details about center of the snapshot
        if type(args) is tuple: args, ds, refine_box = args
        args.halo_center = args.halo_center + args.center_wrt_halo # kpc # offsetting center of ifu data cube wrt halo center, if any
    except Exception as e:
        print('Error being overlooked in utils.parse_args():', e)
        pass

    instrument_dummy = telescope(args) # declare a dummy instrument; just to set proper paths
    args.cube_output_path = get_cube_output_path(args)

    args.idealcube_filename = args.cube_output_path + 'ideal_ifu' + args.mergeHII_text + '.fits'
    args.smoothed_cube_filename = args.cube_output_path + instrument_dummy.path + 'smoothed_ifu' + '_z' + str(args.z) + args.mergeHII_text + '_ppb' + str(args.pix_per_beam) + '.fits'
    args.mockcube_filename = args.cube_output_path + instrument_dummy.path + 'mock_ifu' + '_z' + str(args.z) + args.mergeHII_text + '_ppb' + str(args.pix_per_beam) + '_exp' + str(args.exptime) + 's_snr' + str(args.snr) + '.fits'

    if 'ds' in locals(): return args, ds, refine_box # if ds has already been loaded then return it, so that subsequent functions won't need to re-load ds
    else: return args
