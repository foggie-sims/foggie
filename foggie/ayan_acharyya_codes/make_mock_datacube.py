##!/usr/bin/env python3

"""

    Title :      make_mock_datacube
    Notes :      Produces a mock IFU datacube by spatial convolution & spectral binning & noise addition to an ideal datacube, for a given seeing + spectral resolution + SNR
    Output :     FITS cube
    Author :     Ayan Acharyya
    Started :    March 2021
    Example :    run make_mock_datacube.py --system ayan_local --halo 8508 --output RD0042 --mergeHII 0.04 --base_spatial_res 0.4 --z 0.25 --obs_wave_range 0.8,0.85 --obs_spatial_res 1 --obs_spec_res 60 --exptime 1200 --snr 5 --debug

"""
from header import *
from util import *
from make_ideal_datacube import get_ideal_datacube

# ---------------------------------------------------------------------
def spatial_convolve(ideal_ifu, mock_ifu, args):
    '''
    Function to spatially (rebin and) convolve ideal data cube, with a given PSF, wavelength slice by wavelength slice
    :return: mockcube object: mock_ifu
    '''
    start_time = time.time()

    wlen = np.shape(ideal_ifu.data)[2] # length of dispersion axis
    mock_ifu.data = np.zeros((mock_ifu.box_size_in_pix, mock_ifu.box_size_in_pix, wlen))  # initialise datacube with zeroes

    for slice in range(wlen):
        myprint('Rebinning & convolving slice ' + str(slice + 1) + ' of ' + str(wlen) + '..', args)
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
            myprint('Reading from already existing no-noise cube file ' + args.smoothed_cube_filename + ', use --args.clobber to overwrite', args)
        else:
            myprint('Noisy or no-noise mock cube file does not exist. Creating now..', args)
            ifu = mockcube(args, instrument, linelist)  # declare the noiseless mock datacube object

            # ----- cut wavelength slice from ideal_ifu, depending on mock_ifu's 'observed' wavelength range ------
            start_wave_index = np.where(ideal_ifu.wavelength >= ifu.rest_wave_range[0])[0][0]
            end_wave_index = np.where(ideal_ifu.wavelength >= ifu.rest_wave_range[1])[0][0]
            ideal_ifu.data = ideal_ifu.data[:, :, start_wave_index : end_wave_index]

            ifu = spatial_convolve(ideal_ifu, ifu, args) # spatial convolution based on obs_spatial_res
            ifu = spectral_bin(ifu, args) # spectral rebinning based on obs_spec_res
            write_fitsobj(args.smoothed_cube_filename, ifu, instrument, args, for_qfits=True) # writing smoothed, no-noise cube into FITS file

        if args.snr > 0: # otherwise no point of adding noise
            ifu = noisycube(args, instrument, linelist)  # declare the noisy mock datacube object
            ifu.data = readcube(args.smoothed_cube_filename, args).data # reading in and paste no-noise ifu data in to 'data' attribute of ifu, so that all other attributes of ifu can be used too
            ifu = add_noise(ifu, instrument, args) # adding noise based on snr

            write_fitsobj(args.mockcube_filename, ifu, instrument, args, for_qfits=True) # writing into FITS file
        else:
            args.mockcube_filename = args.smoothed_cube_filename # so that in no-noise case, merely the smoothed datacube is read in

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
    linelist = read_linelist(mappings_lab_dir + 'targetlines.txt')  # list of emission lines

    instrument = telescope(args)  # declare an instrument
    cube_output_path = get_cube_output_path(args)
    #args.idealcube_filename = cube_output_path + instrument.path + 'ideal_ifu' + '_z' + str(args.z) + args.mergeHII_text + '.fits'
    args.idealcube_filename = cube_output_path + 'ideal_ifu' + args.mergeHII_text + '.fits'

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

    myprint('Done making mock datacubes for all given args.diag_arr and args.Om_arr', args)





