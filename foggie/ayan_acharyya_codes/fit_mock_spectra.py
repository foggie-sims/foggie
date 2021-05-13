##!/usr/bin/env python3

"""

    Title :      fit_mock_spectra
    Notes :      Takes a mock IFU datacube and fits the spectra along every pixel
    Output :     FITS cube with flux, vel, etc. of each emission line map as a 2D slice, aong with uncertainties
    Author :     Ayan Acharyya
    Started :    March 2021
    Example :    run fit_mock_spectra.py --system ayan_local --halo 5036 --output RD0020 --mergeHII 0.04 --galrad 6 --testcontfit --test_pixel 143,144
    OR           run fit_mock_spectra.py --system ayan_local --halo 8508 --output RD0042 --mergeHII 0.04 --galrad 6 --base_spatial_res 0.4 --z 0.25 --obs_wave_range 0.8,0.85 --obs_spatial_res 1 --obs_spec_res 60 --exptime 1200 --snr 5

"""
from header import *
from util import *
from make_ideal_datacube import get_ideal_datacube
from make_mock_datacube import wrap_get_mock_datacube

# -------------------------------------------------------------------------------------------------
def fixcont_gauss(x, cont, n, *p):
    '''
    Function to evaluate/fit total (summed) gaussian for multiple lines, given n sets of fitted parameters
    '''
    try: x = x.values
    except: pass

    result = cont
    for xx in range(0, n):
        result += p[3 * xx + 0] * exp(-((x - p[3 * xx + 1]) ** 2) / (2 * p[3 * xx + 2] ** 2))

    return result

# -------------------------------------------------------------------------------------------
def fitcont(spec, fits_header, args):
    '''
    Function for fitting continuum
    :param spec:
    :param args:
    :return:
    '''
    wave_list = [float(item) for item in fits_header['lambdas'].split(',')]
    rest_wave_range = [float(item) for item in fits_header['rest_wave_range(A)'].split(',')]

    for thiswwave in wave_list:
        linemask_width = thiswwave * 1.5 * args.vel_mask / c
        spec.loc[spec['wave'].between(thiswwave - linemask_width, thiswwave + linemask_width), 'linemask'] = True
    masked_spec = spec[~spec['linemask']].reset_index(drop=True).drop('linemask', axis=1)

    wave_bins = np.linspace(rest_wave_range[0], rest_wave_range[1], int(args.nbin_cont/10))
    wave_index = np.digitize(masked_spec['wave'], wave_bins, right=True)

    smoothed_spec = pd.DataFrame()
    for thiscol in masked_spec.columns:
        smoothed_spec[thiscol] = np.array([masked_spec[thiscol].values[np.where(wave_index == ind)].mean() for ind in range(1, len(wave_bins) + 1)])
    smoothed_spec = smoothed_spec.dropna().reset_index(drop=True)  # dropping those flames where mean wavelength value in the bin is nan

    if args.testcontfit:
        print(smoothed_spec)
        fig = plt.figure(figsize=(17, 5))
        fig.subplots_adjust(hspace=0.7, top=0.85, bottom=0.1, left=0.05, right=0.95)

        plt.plot(masked_spec['wave'], masked_spec['flam'], c='r', label='masked flam')
        plt.scatter(masked_spec['wave'], masked_spec['flam'], c='r', lw=0)
        plt.plot(masked_spec['wave'], masked_spec['flam_u'], c='orange', linestyle='dashed', label='masked flam_u')

        plt.plot(smoothed_spec['wave'], smoothed_spec['flam'], c='k', label='smoothly sampled flam')
        plt.scatter(smoothed_spec['wave'], smoothed_spec['flam'], c='k', lw=0)
        plt.plot(smoothed_spec['wave'], smoothed_spec['flam_u'], c='gray', linestyle='dashed', label='smoothly sampled flam_u')

        plt.legend()
        plt.xlabel('Obs wavelength (A)')
        plt.ylabel('flambda (ergs/s/A/pc^2)')
        plt.title('Testing continuum fit for pixel (' + str(args.test_pixel[0]) + ',' + str(args.test_pixel[1]) + ')')
        plt.xlim(rest_wave_range[0], rest_wave_range[1])
        #plt.ylim(-2e-19, 2e-18)
        plt.show(block=False)

    # ----to estimate continuum by interpolation--------
    contfunc = interp1d(smoothed_spec['wave'], smoothed_spec['flam'], kind='cubic', fill_value='extrapolate')
    spec['cont'] = contfunc(spec['wave'])

    # ----to estimate uncertainty in continuum: option 2: measure RMS of fitted cont w.r.t input flam--------
    masked_spec['cont'] = contfunc(masked_spec['wave'])
    spec['cont_u'] = np.ones(len(spec)) * np.sqrt(np.sum((masked_spec['cont'] - masked_spec['flam']) ** 2) / len(spec))

    if args.testcontfit:
        print(spec)
        plt.plot(spec['wave'], spec['cont'], c='g', label='cont')
        plt.plot(spec['wave'], spec['cont_u'], c='g', label='cont_u', linestyle='dotted')
        plt.legend()
        plt.show(block=False)

    return spec

# --------------------------------------------------------------------------------
def fit_all_lines(spec, fits_header, args, which_pixel=None):
    '''
    Fucntion for fitting multiple lines
    :param spec:
    :param fits_header:
    :param args:
    :return:
    '''
    if which_pixel is None: which_pixel = args.test_pixel
    wave_list = [float(item) for item in fits_header['lambdas'].split(',')]
    label_list = [item for item in fits_header['labels'].split(',')]
    
    if 'obs_spec_res(km/s)' in fits_header: args.vel_res = fits_header['obs_spec_res(km/s)']  # in km/s
    else: args.vel_res = fits_header['base_spec_res(km/s)']
    args.resolution = c / args.vel_res

    #spec = spec[spec['flam'] >= 0]  # keep only those rows (wavelength elements) where flux is non-negative
    if 'cont' not in spec: spec = fitcont(spec, fits_header, args)
    spec['flam_norm'] = spec['flam'] / spec['cont']
    spec['flam_u_norm'] = np.sqrt((spec['flam_u'] / spec['cont'])**2 + (spec['flam'] * spec['cont_u'] / spec['cont']**2)**2) # error propagation

    kk, failed_fitting = 1, False
    fitted_df = pd.DataFrame(columns=('label', 'wave', 'flux', 'flux_u', 'vel', 'vel_u', 'vel_disp', 'vel_disp_u')) # to store the results of the line fits
    args.ndlambda_left, args.ndlambda_right = [args.nres_elements] * 2  # how many delta-lambda wide will the window (for line fitting) be on either side of the central wavelength, default 5

    try:
        count = 1
        first, last = [wave_list[0]] * 2
        label_first, label_last = [label_list[0]] * 2
    except IndexError:
        pass

    while kk <= len(label_list):
        center1, label1 = last, label_last
        if kk == len(label_list):
            center2, label2 = 1e10, 'dummy'  # insanely high number, required to capture reddest line
        else:
            center2, label2 = wave_list[kk], label_list[kk]
        if center2 * (1. - args.ndlambda_left / args.resolution) > center1 * (1. + args.ndlambda_right / args.resolution): # if the left-edge of the redder neighbour is beyond (greater than) the right edge of the bluer neighbour, then we can stop adding further neighbours to the current group and fit the current group
            left_edge = first * (1. - args.ndlambda_left / args.resolution)
            right_edge = last * (1. + args.ndlambda_right / args.resolution)
            if args.debug: myprint('Deb131: leftmost edge of next group ' + str(center2 * (1. - args.ndlambda_left / args.resolution)) + ' > rightmost edge of this group ' + str(center1 * (1. + args.ndlambda_right / args.resolution)) + '; therefore ' + str(center2) + ' (' + label2 + ') is beyond neighbourhood of ' + str(center1) + ' (' + label1 + ') which includes ' + str(count) + ' lines from ' + str(first) + ' (' + label_first + ') to ' + str(last) + ' (' + label_last + ') and is bounded by (' + str(left_edge) + ',' + str(right_edge) + ') A', args)

            spec_thisgroup = spec[spec['wave'].between(left_edge, right_edge)] # sliced only the current group of neighbouring lines

            if args.debug: myprint('Trying to fit ' + str(label_list[kk - count:kk]) + ' line/s at once. Total ' + str(count) + '\n', args)

            if args.testlinefit:
                plt.figure(figsize=(10,4))
                plt.plot(spec['wave'], spec['flam_norm'], lw=1, c='k', label='flam')
                plt.plot(spec['wave'], spec['flam_u_norm'], c='gray', label='flam_u')

            try:
                popt, pcov = fitline(spec_thisgroup, wave_list[kk - count:kk], args)
                popt, pcov = np.array(popt), np.array(pcov)
                popt = np.concatenate(([1], popt))  # for fitting after continuum normalised, so continuum is fixed=1 and has to be inserted to popt[] by hand after fitting
                pcov = np.hstack((np.zeros((np.shape(pcov)[0] + 1, 1)), np.vstack((np.zeros(np.shape(pcov)[1]), pcov))))  # for fitting after continuum normalised, so error in continuum is fixed=0 and has to be inserted to pcov[] by hand after fitting
                
                if args.testlinefit:
                    plt.axvline(left_edge, linestyle='--', c='g')
                    plt.axvline(right_edge, linestyle='--', c='g')

                if args.debug: myprint('Done this fitting!' + '\n', args)

            except TypeError:
                if args.debug: myprint('Trying to re-do this fit with broadened wavelength window..\n', args)
                args.ndlambda_left += 1
                args.ndlambda_right += 1
                continue

            except (RuntimeError, ValueError, IndexError) as e:
                popt = np.concatenate(([1], np.zeros(count * 3)))  # if could not fit the line/s fill popt with zeros so flam_array gets zeros
                pcov = np.zeros((count * 3 + 1, count * 3 + 1))  # if could not fit the line/s fill popt with zeros so flam_array gets zeros
                failed_fitting = True
                if args.debug: myprint('Could not fit lines ' + str(label_list[kk - count:kk]) + ' for pixel ' + str(which_pixel[0]) + ', ' + str(which_pixel[1]) + ' due to \n' + repr(e), args)
                pass

            for xx in range(0, count):
                # in popt for every bunch of lines,
                # elements (0,1,2) or (3,4,5) etc. are the height(b), mean(c) and width(d)
                # so, for each line the elements (cont=a,0,1,2) or (cont=a,3,4,5) etc. make the full suite of (a,b,c,d) gaussian parameters
                # so, for each line, flam f (area under gaussian) = sqrt(2pi)*(b-a)*d
                # also the full covariance matrix pcov looks like:
                # |00 01 02 03 04 05 06 .....|
                # |10 11 12 13 14 15 16 .....|
                # |20 21 22 23 24 25 26 .....|
                # |30 31 32 33 34 35 36 .....|
                # |40 41 42 43 44 45 46 .....|
                # |50 51 52 53 54 55 56 .....|
                # |60 61 62 63 64 65 66 .....|
                # |.. .. .. .. .. .. .. .....|
                # |.. .. .. .. .. .. .. .....|
                #
                # where, 00 = var_00, 01 = var_01 and so on.. (var = sigma^2)
                # let var_aa = vaa (00), var_bb = vbb(11), var_ab = vab(01) = var_ba = vba(10) and so on..
                # for a single gaussian, f = const * (b-a)*d
                # i.e. sigma_f^2 = d^2*(saa^2 + sbb^2) + (b-a)^2*sdd^2 (dropping the constant for clarity of explanation)
                # i.e. var_f = d^2*(vaa + vbb) + (b-a)^2*vdd
                # the above holds if we assume covariance matrix to be diagonal (off diagonal terms=0) but thats not the case here
                # so we also need to include off diagnoal covariance terms while propagating flam errors
                # so now, for each line, var_f = d^2*(vaa + vbb) + (b-a)^2*vdd + 2d^2*vab + 2d*(b-a)*(vbd - vad)
                # i.e. in terms of element indices,
                # var_f = 3^2(00 + 11) + (1-0)^2*33 - (2)*3^2*10 + (2)*3*(1-0)*(13-03),
                # var_f = 6^2(00 + 44) + (4-0)^2*66 - (2)*6^2*40 + (2)*6*(4-0)*(46-06),
                # var_f = 9^2(00 + 77) + (1-0)^2*99 - (2)*9^2*70 + (2)*9*(7-0)*(79-09), etc.
                # similarly, for a single gaussian, velocity dispersion vd = speed_of_light * d/c
                # i.e. sigma_vd^2 = sdd^2/c^2 + d^2*scc^2/c^4 for a diagonal-only covariance matrix
                # i.e. sigma_vd^2 = sdd^2/c^2 + d^2*scc^2/c^4 + 2*d*scd^2/c^3
                # i.e. var_vd = vdd/c^2 + d^2*vcc/c^4 + 2*d*vcd/c^3
                #
                popt_single = np.concatenate(([popt[0]], popt[3 * xx + 1:3 * (xx + 1) + 1]))
                cont_at_line = spec[spec['wave'] >= wave_list[kk + xx - count]]['cont'].values[0]
                
                if args.debug: print('Deb198: linefit param at (', which_pixel[0], ',', which_pixel[1], ') for ', label_list[kk + xx - count], '(ergs/s/pc^2/A) =', popt_single)  #
                if args.testlinefit and count > 1: # the total fit will already be plotted by fitlines() but this is now for plotting the individual lines in a group
                    plt.plot(spec_thisgroup['wave'], fixcont_gauss(spec_thisgroup['wave'], popt[0], 1, *popt_single[1:]), c='orange', ls='dashed', label='individual line')

                flux = np.sqrt(2 * np.pi) * (popt_single[1] - popt_single[0]) * popt_single[3] * cont_at_line # total flam = integral of guassian fit ; resulting flam in ergs/s/pc^2 units
                flux_error = np.sqrt(2 * np.pi * (popt_single[3] ** 2 * (pcov[0][0] + pcov[3 * xx + 1][3 * xx + 1]) \
                                                  + (popt_single[1] - popt_single[0]) ** 2 * pcov[3 * (xx + 1)][3 * (xx + 1)] \
                                                  - 2 * popt_single[3] ** 2 * pcov[3 * xx + 1][0] \
                                                  + 2 * (popt_single[1] - popt_single[0]) * popt_single[3] * (pcov[3 * xx + 1][3 * (xx + 1)] - pcov[0][3 * (xx + 1)]) \
                                                  )) * cont_at_line # var_f = 3^2(00 + 11) + (1-0)^2*33 - (2)*3^2*10 + (2)*3*(1-0)*(13-03)

                vel_disp = c * popt_single[3]/popt_single[2] # in km/s
                vel_disp_error = c * np.sqrt(pcov[3 * xx + 3][3 * xx + 3]/(popt_single[2] ** 2) + popt_single[3]**2 * pcov[3 * xx + 2][3 * xx + 2] / (popt_single[2] ** 4) + 2 * popt_single[3] * pcov[3 * xx + 2][3 * xx + 3]/ (popt_single[2] ** 3)) # in km/s

                velocity = c * (wave_list[kk + xx - count] - popt_single[2]) / popt_single[2]  # in km/s
                velocity_error = c * wave_list[kk + xx - count] * pcov[3 * xx + 2][3 * xx + 2] / popt_single[2] ** 2  # in km/s

                fitted_df.loc[len(fitted_df)] = [label_list[kk + xx - count], wave_list[kk + xx - count], flux, flux_error, velocity, velocity_error, vel_disp, vel_disp_error] # store measured quantities in df

            if args.testlinefit:
                plt.legend()
                plt.xlabel('Rest wavelength (A)')
                plt.ylabel('flambda (ergs/s/A/pc^2)')
                plt.title('Testing line fit for pixel (' + str(which_pixel[0]) + ',' + str(which_pixel[1]) + ')')
                plt.show(block=False)

            first, last = [center2] * 2
            label_first, label_last = [label2] * 2
            count = 1 # initialising count back to 1, for the next potential group of lines
        else:
            last, label_last = center2, label2
            count += 1
            if args.debug: myprint('Deb237: leftmost edge of next group ' + str(center2 * (1. - args.ndlambda_left / args.resolution)) + ' <= rightmost edge of this group ' + str(center1 * (1. + args.ndlambda_right / args.resolution)) + '; therefore ' + str(center2) + ' (' + label2 + ') is included in the current group of lines which now includes ' + str(count) + ' lines from ' + str(first) + ' (' + label_first + ') to ' + str(last) + ' (' + label_last + ')', args)

        kk += 1

    if args.debug: print('Deb234:\n', fitted_df)

    return fitted_df, failed_fitting

# -------------------------------------------------------------------------------------------
def fitline(spec, waves_to_fit, args):
    '''
    Function to fit one group of neighbouring emission line
    '''
    v_maxwidth = 10 * args.vel_res
    z_allow = 3e-3  # wavelengths are at restframe; assumed error in redshift

    p_init, lbound, ubound = [], [], []
    for xx in range(0, len(waves_to_fit)):
        left_edge_single = waves_to_fit[xx] * (1. - 10 / args.resolution)
        right_edge_single = waves_to_fit[xx] * (1. + 10 / args.resolution)
        spec_single = spec[spec['wave'].between(left_edge_single, right_edge_single)] # extract a small region around a _single_ line, in order to be able to provide _better_ initial guesses to curve_fit(), e.g., the central wavelength to fit for
        if args.debug: print('Deb266:', xx, ' out of', len(waves_to_fit), ', between', left_edge_single, right_edge_single, 'spec_single=', spec_single) #

        p_init = np.append(p_init, [np.max(spec_single['flam_norm']) - np.min(spec_single['flam_norm']), spec_single[spec_single['flam_norm'] == np.max(spec_single['flam_norm'])]['wave'].values[0], waves_to_fit[xx] * 2. * gf2s / args.resolution])
        lbound = np.append(lbound, [1, waves_to_fit[xx] * (1. - z_allow), waves_to_fit[xx] * 1. * gf2s / args.resolution])
        ubound = np.append(ubound, [1e5, waves_to_fit[xx] * (1. + z_allow), waves_to_fit[xx] * v_maxwidth * gf2s / c])

    if args.debug: print('Deb269: p_init=', p_init, 'lbound=', lbound, 'ubound=', ubound) #

    if spec['flam_u_norm'].any():
        popt, pcov = curve_fit(lambda x, *p: fixcont_gauss(x, 1, len(waves_to_fit), *p), spec['wave'].values, spec['flam_norm'].values, method='trf', p0=p_init, maxfev=10000, bounds=(lbound, ubound), sigma=spec['flam_u_norm'], absolute_sigma=True)
    else:
        popt, pcov = curve_fit(lambda x, *p: fixcont_gauss(x, 1, len(waves_to_fit), *p), spec['wave'], spec['flam_norm'], p0=p_init, max_nfev=10000, bounds=(lbound, ubound))

    if args.testlinefit:
        plt.plot(spec['wave'], fixcont_gauss(spec['wave'], 1, len(waves_to_fit), *p_init), c='b', label='initial guess')
        plt.plot(spec['wave'], fixcont_gauss(spec['wave'], 1, len(waves_to_fit), *popt), c='r', label='best fit')

    return popt, pcov

# -----------------------------------------------------------------------------
def fit_mock_spectra(args):
    '''
    Function to load mock data cube and fit spectra along each line of sight, and write measured fitted quantities for each emission line into a FITS data cube
    :return: measured_datacube
    '''
    cube_output_path = get_cube_output_path(args)
    linelist = read_linelist(mappings_lab_dir + 'targetlines.txt')  # list of emission lines
    instrument = telescope(args)  # declare the instrument
    if args.snr == 0:
        file_to_fit = cube_output_path + 'ideal_ifu' + args.mergeHII_text + '.fits'
        if not os.path.exists(file_to_fit):
            myprint('Ideal cube file does not exist, calling make_ideal_cube.py..', args)
            dummy = get_ideal_datacube(args, linelist)
        measured_cube = idealcube(args, instrument, linelist)  # declare a cube object
    else:
        file_to_fit = cube_output_path + instrument.path + 'mock_ifu' + '_z' + str(args.z) + args.mergeHII_text + '_ppb' + str(args.pix_per_beam) + '_exp' + str(args.exptime) + 's_snr' + str(args.snr) + '.fits'
        if not os.path.exists(file_to_fit):
            myprint('Mock cube file does not exist, calling make_ideal_cube.py..', args)
            dummy = wrap_get_mock_datacube(args)
        measured_cube = noisycube(args, instrument, linelist)  # declare the noisy mock datacube object

    failed_pixel_count = 0
    measured_quantities = ['flux', 'vel', 'vel_disp']

    ifu = readcube(file_to_fit, args)
    wave_list = [float(item) for item in ifu.header['lambdas'].split(',')]
    measured_cube.data = np.zeros((np.shape(ifu.data)[0], np.shape(ifu.data)[1], 3 * len(wave_list))) # declare new cube to store measured quantities (3 quantities per emission line: flux, vel and vel_disp)
    measured_cube.error = np.zeros((np.shape(ifu.data)[0], np.shape(ifu.data)[1], 3 * len(wave_list)))  # associated uncertainties

    if args.test_pixel is not None: xlen, ylen = 1, 1
    else: xlen, ylen = np.shape(ifu.data)[0], np.shape(ifu.data)[1]

    for index in range(xlen*ylen):
        if args.test_pixel is not None: i, j = args.test_pixel
        else: i, j = int(index / ylen), int(index % ylen) # cell position
        myprint('Line fitting pixel (' + str(i) + ',' + str(j) + '), i.e. ' + str(index + 1) + ' out of ' + str(xlen * ylen) + '..', args)

        spec = pd.DataFrame({'wave': np.array(ifu.wavelength).byteswap().newbyteorder(), \
                            'flam': np.array(ifu.data[i, j, :]).byteswap().newbyteorder(), \
                            'flam_u': np.array(ifu.error[i, j, :]) if (ifu.error == 0).all() else np.array(ifu.error[i, j, :]).byteswap().newbyteorder(), \
                            'linemask': False})
                            # flam_u will have newbyteorder (or not) depending on whether it is a genuine error cube i.e. read in from saved fits file Or just a dummy array of zeros
                            # the above is to avoid "ValueError: Big-endian buffer not supported on little-endian compiler" while trying to slice the spec dataframe

        if spec['flam'].any():
            fitted_df, failed_fitting = fit_all_lines(spec, ifu.header, args, which_pixel=(i, j))
            if failed_fitting: failed_pixel_count += 1
            measured_cube.data[i, j, :] = fitted_df[measured_quantities].to_numpy().flatten() # store measured quantities for a given pixel in a 1D array
            measured_cube.error[i, j, :] = fitted_df[[item + '_u' for item in measured_quantities]].to_numpy().flatten() # store associated uncertainties
        elif args.debug:
            myprint('Pixel (' + str(i) + ',' + str(j) + '), has no flux at all, hence not bothering to fit spectra', args)

    myprint(str(failed_pixel_count) + ' out of ' + str(xlen * ylen) + ' pixels, i.e., ' + str(failed_pixel_count * 100. / (xlen * ylen)) + '% pixels failed to fit.', args)
    args.measured_cube_filename = file_to_fit.replace('ideal_ifu', 'measured_cube').replace('mock_ifu', 'measured_cube')
    write_fitsobj(args.measured_cube_filename, measured_cube, instrument, args, for_qfits=True, measured_cube=True, measured_quantities=measured_quantities)  # writing into FITS file
    return measured_cube

# -----------------------------------------------------------------------------
if __name__ == '__main__':
    start_time = time.time()

    args = parse_args('8508', 'RD0042')
    args.diag = args.diag_arr[0]
    args.Om = args.Om_arr[0]
    if not args.keep: plt.close('all')
    if (args.testlinefit or args.testcontfit) and args.test_pixel is None: raise AssertionError('Cannot test fitting without a specified test_pixel')

    measured_cube = fit_mock_spectra(args)
    print('Complete in %s minutes' % ((time.time() - start_time) / 60))
