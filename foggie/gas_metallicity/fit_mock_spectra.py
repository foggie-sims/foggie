##!/usr/bin/env python3

"""

    Title :      fit_mock_spectra
    Notes :      Takes a mock IFU datacube and fits the spectra along every pixel
    Output :     FITS cube with flux, vel, etc. of each emission line map as a 2D slice, aong with uncertainties
    Author :     Ayan Acharyya
    Started :    March 2021
    Example :    run fit_mock_spectra.py --system ayan_local --halo 5036 --output RD0020 --mergeHII 0.04 --galrad 6 --testcontfit --test_pixel 143,144
run fit_mock_spectra.py --system ayan_local --halo 8508 --output RD0042 --mergeHII 0.04 --galrad 6 --base_spatial_res 0.4 --z 0.25 --obs_wave_range 0.8,0.85 --obs_spatial_res 1 --obs_spec_res 60 --exptime 1200 --snr 5
run fit_mock_spectra.py --system ayan_local --halo 4123 --output RD0020 --mergeHII 0.04 --galrad 2 --base_spatial_res 0.06 --z 0.25 --obs_wave_range 0.8,0.85 --obs_spatial_res 0.07 --obs_spec_res 60 --snr 0 --keep --doideal --debug --testcontfit --testlinefit --test_pixel 34,34
"""
from header import *
from util import *
from make_ideal_datacube import get_ideal_datacube
from make_mock_datacube import wrap_get_mock_datacube
from scipy.signal import find_peaks

# -------------------------------------------------------------------------------------------------
def fixcont_fixgroupz_gauss(x, cont, obswave, *p):
    '''
    Function to evaluate/fit total (summed) gaussian for multiple lines with the SAME redshift (i.e. velocity), given n sets of fitted parameters
    '''
    try: x = x.values
    except: pass

    result = cont
    for xx in range(0, len(obswave)):
        result += p[2 * xx + 1] * exp(-((x - obswave[xx] * (1. + p[0]/c)) ** 2) / (2 * p[2 * xx + 2] ** 2))
    return result

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
        if args.debug: print('Deb: testcontfit 87: smoothed spec\n', smoothed_spec)
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

    spec['flam_norm'] = spec['flam'] / spec['cont']
    spec['flam_u_norm'] = np.sqrt((spec['flam_u'] / spec['cont']) ** 2 + (spec['flam'] * spec['cont_u'] / spec['cont'] ** 2) ** 2)  # error propagation

    if args.testcontfit:
        if args.debug: print('Deb: testcontfit 87: full spec\n', spec)
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
                plt.plot(spec['wave'], spec['flam_u_norm'], lw=1, c='gray', label='flam_u')

            try:
                popt, pcov = fit_group_of_lines(spec_thisgroup, wave_list[kk - count:kk], args)
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
                # elements (1,2,3) or (4,5,6) etc. are the net-height(b), mean(c) and width(d) and element[0] is continuum(a)
                # so, for each line the elements (a,1,2,3) or (a,4,5,6) etc. make the full suite of (a,b,c,d) gaussian parameters
                # so, for each line, flam f (area under gaussian) = sqrt(2pi)*b*d
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
                popt_single = np.concatenate(([popt[0]], popt[3 * xx + 1 : 3 * xx + 4])) # popt_single = [continuum, amplitude, mean, width] = [a,b,c,d]
                cont_at_line = spec[spec['wave'] >= wave_list[kk + xx - count]]['cont'].values[0]
                
                if args.debug: print('Deb198: linefit param at (', which_pixel[0], ',', which_pixel[1], ') for ', label_list[kk + xx - count], '(ergs/s/pc^2/A) =', popt_single)  #
                if args.testlinefit and count > 1: # the total fit will already be plotted by fit_group_of_lines() but this is now for plotting the individual lines in a group
                    plt.plot(spec_thisgroup['wave'], fixcont_gauss(spec_thisgroup['wave'], popt[0], 1, *popt_single[1:]), c='orange', ls='dashed', label='individual line' if not xx else None, lw=1)

                #flux = np.sqrt(2 * np.pi) * (popt_single[1] - popt_single[0]) * popt_single[3] * cont_at_line # total flam = integral of guassian fit ; resulting flam in ergs/s/pc^2 units
                flux = np.sqrt(2 * np.pi) * popt_single[1] * popt_single[3] * cont_at_line
                flux_error = np.sqrt(2 * np.pi * (popt_single[3] ** 2 * pcov[3 * xx + 1][3 * xx + 1] \
                                                  + popt_single[1] ** 2 * pcov[3 * xx + 3][3 * xx + 3] \
                                                  )) * cont_at_line # var_f = 3^2(00 + 11) + (1-0)^2*33 - (2)*3^2*10 + (2)*3*(1-0)*(13-03)

                vel_disp = c * popt_single[3]/popt_single[2] # in km/s
                vel_disp_error = c * np.sqrt(pcov[3 * xx + 3][3 * xx + 3]/(popt_single[2] ** 2) + popt_single[3]**2 * pcov[3 * xx + 2][3 * xx + 2] / (popt_single[2] ** 4) + 2 * popt_single[3] * pcov[3 * xx + 2][3 * xx + 3]/ (popt_single[2] ** 3)) # in km/s

                velocity = c * (1 - popt_single[2] / wave_list[kk + xx - count])  # in km/s
                velocity_error = c * np.sqrt(pcov[3 * xx + 2][3 * xx + 2]) / wave_list[kk + xx - count]  # in km/s

                fitted_df.loc[len(fitted_df)] = [label_list[kk + xx - count], wave_list[kk + xx - count], flux, flux_error, velocity, velocity_error, vel_disp, vel_disp_error] # store measured quantities in df

            if args.testlinefit:
                plt.legend()
                plt.xlabel('Rest wavelength (A)')
                plt.ylabel('Continuum normalised flambda')
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

    if args.testlinefit: print('Deb234:\n', fitted_df)

    return fitted_df, failed_fitting

# -------------------------------------------------------------------------------------------
def fit_group_of_lines(spec, waves_to_fit, args):
    '''
    Function to fit one group of neighbouring emission line
    '''
    n_res_elements_min, n_res_elements_fid, n_res_elements_max = 1, 8, 30 # min and max number of resolution elements to bound the fitted sigma for each line by
    z_allow = 3e-3  # wavelengths are at restframe; assumed error in redshift
    n_res_individual_window = args.nres_elements # 20

    p_init, lbound, ubound = [], [], []
    for xx in range(0, len(waves_to_fit)):
        if args.testlinefit:
            plt.axvline(waves_to_fit[xx], c='b', label='vacuum wavelength' if not xx else None, lw=1, ls='dotted')

        left_edge_single = waves_to_fit[xx] * (1. - n_res_individual_window / args.resolution)
        try: right_edge_single = waves_to_fit[xx+1] * (1. - n_res_individual_window / args.resolution) # try to set the right edge of this line = the left edge of the next line, if at all there si a next line
        except: right_edge_single = waves_to_fit[xx] * (1. + n_res_individual_window / args.resolution)
        spec_single = spec[spec['wave'].between(left_edge_single, right_edge_single)] # extract a small region around a _single_ line, in order to be able to provide _better_ initial guesses to curve_fit(), e.g., the central wavelength to fit for
        if args.debug: print('Deb266:', xx+1, ' out of', len(waves_to_fit), ', between', left_edge_single, right_edge_single, 'spec_single=\n', spec_single) #

        spec_single_sorted = spec_single.sort_values(by='flam_norm', ascending=False) # positions of fluxes arranged in descending order
        amplitude_init = spec_single_sorted['flam_norm'].max() - spec_single_sorted['flam_norm'].min()
        #wave_cen_init = spec_single_sorted['wave'].values[0] # assign position of highest flux as the central wavelength
        wave_cen_init = np.mean(spec_single_sorted['wave'].values[:3]) # assign mean of positions of 3 highest flux peaks as the central wavelength

        p_init = np.append(p_init, [amplitude_init, wave_cen_init, waves_to_fit[xx] * n_res_elements_fid * gf2s / args.resolution])
        lbound = np.append(lbound, [1e-3, waves_to_fit[xx] * (1. - z_allow), waves_to_fit[xx] * n_res_elements_min * gf2s / args.resolution])
        ubound = np.append(ubound, [1e5, waves_to_fit[xx] * (1. + z_allow), waves_to_fit[xx] * n_res_elements_max * gf2s / args.resolution])

    if args.debug: print('Deb269: p_init=', p_init, 'lbound=', lbound, 'ubound=', ubound) #

    # popt is in the format [amplitude_i, wavecen_i, sigma_i....] where i is each emission line in the group
    if spec['flam_u_norm'].any():
        popt, pcov = curve_fit(lambda x, *p: fixcont_gauss(x, 1, len(waves_to_fit), *p), spec['wave'].values, spec['flam_norm'].values, method='trf', p0=p_init, maxfev=10000, bounds=(lbound, ubound), sigma=spec['flam_u_norm'], absolute_sigma=True)
    else:
        popt, pcov = curve_fit(lambda x, *p: fixcont_gauss(x, 1, len(waves_to_fit), *p), spec['wave'], spec['flam_norm'], p0=p_init, max_nfev=10000, bounds=(lbound, ubound))

    if args.testlinefit:
        plt.plot(spec['wave'], fixcont_gauss(spec['wave'], 1, len(waves_to_fit), *p_init), c='b', label='initial guess', lw=1)
        plt.plot(spec['wave'], fixcont_gauss(spec['wave'], 1, len(waves_to_fit), *popt), c='r', label='best fit', lw=2)

    return popt, pcov


# --------------------------------------------------------------------------------
def fit_all_lines_fixgroupz(spec, fits_header, args, which_pixel=None):
    '''
    Fucntion for fitting multiple lines with same redshift (i.e. velocity) for each group of lines
    :param spec:
    :param fits_header:
    :param args:
    :return:
    '''
    if which_pixel is None: which_pixel = args.test_pixel
    wave_list = [float(item) for item in fits_header['lambdas'].split(',')]
    label_list = [item for item in fits_header['labels'].split(',')]

    if 'obs_spec_res(km/s)' in fits_header:
        args.vel_res = fits_header['obs_spec_res(km/s)']  # in km/s
    else:
        args.vel_res = fits_header['base_spec_res(km/s)']
    args.resolution = c / args.vel_res

    # spec = spec[spec['flam'] >= 0]  # keep only those rows (wavelength elements) where flux is non-negative
    if 'cont' not in spec: spec = fitcont(spec, fits_header, args)

    kk, failed_fitting = 1, False
    fitted_df = pd.DataFrame(columns=('label', 'wave', 'flux', 'flux_u', 'vel', 'vel_u', 'vel_disp', 'vel_disp_u'))  # to store the results of the line fits
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
        if center2 * (1. - args.ndlambda_left / args.resolution) > center1 * (1. + args.ndlambda_right / args.resolution):  # if the left-edge of the redder neighbour is beyond (greater than) the right edge of the bluer neighbour, then we can stop adding further neighbours to the current group and fit the current group
            left_edge = first * (1. - args.ndlambda_left / args.resolution)
            right_edge = last * (1. + args.ndlambda_right / args.resolution)
            if args.debug: myprint('Deb131: leftmost edge of next group ' + str(center2 * (1. - args.ndlambda_left / args.resolution)) + ' > rightmost edge of this group ' + str(center1 * (1. + args.ndlambda_right / args.resolution)) + '; therefore ' + str(center2) + ' (' + label2 + ') is beyond neighbourhood of ' + str(center1) + ' (' + label1 + ') which includes ' + str(count) + ' lines from ' + str(first) + ' (' + label_first + ') to ' + str(last) + ' (' + label_last + ') and is bounded by (' + str(left_edge) + ',' + str(right_edge) + ') A', args)

            spec_thisgroup = spec[spec['wave'].between(left_edge, right_edge)]  # sliced only the current group of neighbouring lines

            if args.debug: myprint('Trying to fit ' + str(label_list[kk - count:kk]) + ' line/s at once. Total ' + str(count) + '\n', args)

            if args.testlinefit:
                plt.figure(figsize=(10, 4))
                plt.plot(spec['wave'], spec['flam_norm'], lw=1, c='k', label='flam')
                plt.plot(spec['wave'], spec['flam_u_norm'], lw=1, c='gray', label='flam_u')

            try:
                popt, pcov = fit_fixgroupz_group_of_lines(spec_thisgroup, wave_list[kk - count:kk], args)
                popt, pcov = np.array(popt), np.array(pcov)
                popt = np.concatenate(([1],popt))  # for fitting after continuum normalised, so continuum is fixed=1 and has to be inserted to popt[] by hand after fitting
                pcov = np.hstack((np.zeros((np.shape(pcov)[0] + 1, 1)), np.vstack((np.zeros(np.shape(pcov)[1]),pcov))))  # for fitting after continuum normalised, so error in continuum is fixed=0 and has to be inserted to pcov[] by hand after fitting

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
                pcov = np.zeros((count * 3 + 1,count * 3 + 1))  # if could not fit the line/s fill popt with zeros so flam_array gets zeros
                failed_fitting = True
                if args.debug: myprint('Could not fit lines ' + str(label_list[kk - count:kk]) + ' for pixel ' + str(which_pixel[0]) + ', ' + str(which_pixel[1]) + ' due to \n' + repr(e), args)
                pass

            for xx in range(0, count):
                # in popt for every bunch of lines,
                # elements (2,3) or (4,5) etc. are the net-height(b) and width(d) and element[0] = continuum(a), element[1] = velocity for the whole group
                # therefore for each line mean(c) = obswave[xx] * (1 - velocity/c)
                # so, for each line the elements (a,2,c,3) or (a,4,c,5) etc. make the full suite of (a,b,c,d) gaussian parameters
                # so, for each line, flam f (area under gaussian) = sqrt(2pi)*b*d
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
                cont_at_line = spec[spec['wave'] >= wave_list[kk + xx - count]]['cont'].values[0]
                popt_single = np.concatenate((popt[0:2], popt[2 * xx + 2 : 2 * xx + 4])) # popt_single = [continuum, velocity, amplitude, width] = [a,velocity,b,d]
                this_line_wavecen = wave_list[kk + xx - count] * (1 - popt_single[1]/c)
                this_line_wavecen_u = wave_list[kk + xx - count] * np.sqrt(pcov[1][1])/c

                if args.debug: print('Deb198: linefit param at (', which_pixel[0], ',', which_pixel[1], ') for ',label_list[kk + xx - count], '(ergs/s/pc^2/A) =', popt_single)  #
                if args.testlinefit and count > 1:  # the total fit will already be plotted by fit_group_of_lines() but this is now for plotting the individual lines in a group
                    plt.plot(spec_thisgroup['wave'],fixcont_fixgroupz_gauss(spec_thisgroup['wave'], popt_single[0],  [wave_list[kk + xx - count]], *popt_single[1:]), c='orange', ls='dashed', label='individual line' if not xx else None, lw=1)

                flux = np.sqrt(2 * np.pi) * popt_single[2] * popt_single[3] * cont_at_line  # total flam = integral of guassian fit ; resulting flam in ergs/s/pc^2 units
                flux_error = np.sqrt(2 * np.pi * (popt_single[3] ** 2 * pcov[2 * xx + 2][2 * xx + 2] \
                                                  + popt_single[2] ** 2 * pcov[2 * xx + 3][2 * xx + 3] \
                                                  )) * cont_at_line  # var_f = 3^2(00 + 11) + (1-0)^2*33 - (2)*3^2*10 + (2)*3*(1-0)*(13-03)

                vel_disp = c * popt_single[3] / this_line_wavecen  # in km/s
                vel_disp_error = c * np.sqrt(pcov[2 * xx + 3][2 * xx + 3] / this_line_wavecen**2 + popt_single[3] ** 2 * this_line_wavecen_u ** 2/ this_line_wavecen ** 4)  # in km/s

                velocity = popt_single[1]  # in km/s
                velocity_error = np.sqrt(pcov[1][1]) # in km/s

                fitted_df.loc[len(fitted_df)] = [label_list[kk + xx - count], wave_list[kk + xx - count], flux,
                                                 flux_error, velocity, velocity_error, vel_disp,
                                                 vel_disp_error]  # store measured quantities in df

            if args.testlinefit:
                plt.legend()
                plt.xlabel('Rest wavelength (A)')
                plt.ylabel('Continuum normalised flambda')
                plt.title('Testing line fit for pixel (' + str(which_pixel[0]) + ',' + str(which_pixel[1]) + ')')
                plt.show(block=False)

            first, last = [center2] * 2
            label_first, label_last = [label2] * 2
            count = 1  # initialising count back to 1, for the next potential group of lines
        else:
            last, label_last = center2, label2
            count += 1
            if args.debug: myprint('Deb237: leftmost edge of next group ' + str(
                center2 * (1. - args.ndlambda_left / args.resolution)) + ' <= rightmost edge of this group ' + str(
                center1 * (1. + args.ndlambda_right / args.resolution)) + '; therefore ' + str(
                center2) + ' (' + label2 + ') is included in the current group of lines which now includes ' + str(
                count) + ' lines from ' + str(first) + ' (' + label_first + ') to ' + str(
                last) + ' (' + label_last + ')', args)

        kk += 1

    if args.testlinefit:
        print('Deb234:\n', fitted_df)
        this_photgrid = pd.DataFrame(columns=fitted_df.label, data=dict(zip(fitted_df.label, [[item] for item in fitted_df.flux])))
        Z = get_D16_metallicity(this_photgrid).values[0]
        print('Deb235: derived metallicity from this spectra using D16 would be', Z, 'Z/Zsol = ', np.log10(Z), 'in log scale')

    return fitted_df, failed_fitting

# -------------------------------------------------------------------------------------------
def fit_fixgroupz_group_of_lines(spec, waves_to_fit, args):
    '''
    Function to fit one group of neighbouring emission line with the same redshift (i.e. velicity offset)
    '''
    n_res_elements_min, n_res_elements_fid, n_res_elements_max = 1, 8, 30 # min and max number of resolution elements to bound the fitted sigma for each line by
    threshold, width = 1e-3, 1 # min threshold (veritcal distance from neighbouring pixels i.e. sharpness) and width of the feature detected by find_peaks() in pixels in order to be classified as a "peak"

    p_init, lbound, ubound, velocity_arr = [], [-500], [500], []

    for xx in range(0, len(waves_to_fit)):
        # -----------estimate boundaries to contain individual lines of the group----------
        if xx == 0: left_edge_single = waves_to_fit[xx] * (1. - args.nres_elements / args.resolution) # full group window's left boundary = left edge for first member of group
        else: left_edge_single = right_edge_single # previous member's right edge = this member's left edge

        if xx == len(waves_to_fit) - 1: right_edge_single = waves_to_fit[xx] * (1. + args.nres_elements / args.resolution) # full group window's right boundary = right boundary for last member of group
        else:
            spec_remaining = spec[spec['wave'] > left_edge_single]
            min_sep_between_peaks = len(spec_remaining[spec_remaining['wave'].between(waves_to_fit[xx], waves_to_fit[xx + 1])]) # mean separation between wavelength centroids in pixel units
            peaks_at = []

            while len(peaks_at) < 2: # trying to re-do the peak finding with smaller minimum separation requirement between peaks
                peaks_at = find_peaks(spec['flam_norm'], distance=min_sep_between_peaks, threshold=threshold, width=width)[0]
                peak_waves = spec.iloc[peaks_at]['wave'].values
                if args.debug: print('Deb264:', len(peaks_at), 'peak/s detected at', peak_waves, 'with separation=', min_sep_between_peaks, 'pixels')
                min_sep_between_peaks -= 1
                if len(peaks_at) < 2 and args.debug: print('Deb265: Re-trying to find more peaks with min separation now reduced to', min_sep_between_peaks, 'pixels')

            right_edge_single = np.mean(peak_waves[:2]) # right edge = just between the next two flux peaks; assuming there is two flux peaks including the current one i.e. the current peak and the peak for the next individual line

        spec_single = spec[spec['wave'].between(left_edge_single, right_edge_single)] # extract a small region around a _single_ line, in order to be able to provide _better_ initial guesses to curve_fit(), e.g., the central wavelength to fit for
        if args.debug: print('Deb266:', xx+1, ' out of', len(waves_to_fit), ', between', left_edge_single, right_edge_single, 'spec_single=\n', spec_single) #

        # -----------plotting estimated individual line boundaries----------
        if args.testlinefit:
            plt.axvline(waves_to_fit[xx], c='b', label='vacuum wavelength' if not xx else None, lw=1, ls='dotted')
            plt.axvline(left_edge_single, c='g', label='individual line window - left edge' if not xx else None, lw=1, ls='dotted')
            plt.axvline(right_edge_single, c='g', label='individual line window - right edge' if not xx else None, lw=1, ls='dashed')
            for ind,peak in enumerate(peak_waves): plt.axvline(peak, c='salmon', label='detected peaks' if not xx and not ind else None, lw=1, ls='dotted')

        # -----------estimate initial guesses----------
        spec_single_sorted = spec_single.sort_values(by='flam_norm', ascending=False)  # positions of fluxes arranged in descending order
        wave_cen_init = np.mean(spec_single_sorted['wave'].values[:3])  # assign mean of positions of 3 highest flux peaks as the central wavelength
        velocity_arr = np.append(velocity_arr, [(wave_cen_init / waves_to_fit[xx] - 1.) * c])
        amplitude_init = spec_single_sorted['flam_norm'].max() - spec_single_sorted['flam_norm'].min()

        # -----------setting initial guess and bounds-------------
        p_init = np.append(p_init, [amplitude_init, waves_to_fit[xx] * n_res_elements_fid * gf2s / args.resolution])
        lbound = np.append(lbound, [1e-3, waves_to_fit[xx] * n_res_elements_min * gf2s / args.resolution])
        ubound = np.append(ubound, [1e5, waves_to_fit[xx] * n_res_elements_max * gf2s / args.resolution])

    p_init = np.hstack([[np.mean(velocity_arr)], p_init]) # including the initial guess for group velocity only after examining each line of the group
    if args.debug:
        print('Deb268: velocity_guess_arr=', velocity_arr, 'km/s') #
        print('Deb269: p_init=', p_init, 'lbound=', lbound, 'ubound=', ubound) #

    # popt is in the format [common_velocity, amplitude_i, sigma_i....] where i is each emission line in the group
    if spec['flam_u_norm'].any():
        popt, pcov = curve_fit(lambda x, *p: fixcont_fixgroupz_gauss(x, 1, waves_to_fit, *p), spec['wave'].values, spec['flam_norm'].values, method='trf', p0=p_init, maxfev=10000, bounds=(lbound, ubound), sigma=spec['flam_u_norm'], absolute_sigma=True)
    else:
        popt, pcov = curve_fit(lambda x, *p: fixcont_fixgroupz_gauss(x, 1, waves_to_fit, *p), spec['wave'], spec['flam_norm'], p0=p_init, max_nfev=10000, bounds=(lbound, ubound))

    if args.testlinefit:
        plt.plot(spec['wave'], fixcont_fixgroupz_gauss(spec['wave'], 1, waves_to_fit, *p_init), c='b', label='initial guess', lw=1)
        plt.plot(spec['wave'], fixcont_fixgroupz_gauss(spec['wave'], 1, waves_to_fit, *popt), c='r', label='best fit', lw=2)

    return popt, pcov

# -----------------------------------------------------------------------------
def fit_mock_spectra(args):
    '''
    Function to load mock data cube and fit spectra along each line of sight, and write measured fitted quantities for each emission line into a FITS data cube
    :return: measured_datacube
    '''
    linelist = read_linelist(mappings_lab_dir + 'targetlines.txt')  # list of emission lines
    instrument = telescope(args)  # declare the instrument
    if args.doideal: # for fitting ideal data cube
        file_to_fit = args.idealcube_filename
        if not os.path.exists(file_to_fit):
            myprint('Ideal cube file ' + file_to_fit + ' does not exist, calling make_ideal_datacube.py..', args)
            dummy = get_ideal_datacube(args, linelist)
        measured_cube = idealcube(args, instrument, linelist)  # declare a cube object
    elif args.snr == 0: # for fitting smoothed (but no noise) data cube
        file_to_fit = args.smoothed_cube_filename
        if not os.path.exists(file_to_fit):
            myprint('Mock cube file ' + file_to_fit + ' does not exist, calling make_mock_datacube.py..', args)
            dummy = wrap_get_mock_datacube(args)
        measured_cube = mockcube(args, instrument, linelist)  # declare a cube object
    else: # for fitting mock (smoothed, noisy) data cube
        file_to_fit = args.mockcube_filename
        if not os.path.exists(file_to_fit):
            myprint('Mock cube file ' + file_to_fit + ' does not exist, calling make_mock_datacube.py..', args)
            dummy = wrap_get_mock_datacube(args)
        measured_cube = noisycube(args, instrument, linelist)  # declare the noisy mock datacube object

    measured_cube_filename = get_measured_cube_filename(file_to_fit)
    if os.path.exists(measured_cube_filename) and not (args.clobber or args.testcontfit or args.testlinefit):
        myprint('Reading from already existing file ' + measured_cube_filename + ', use --args.clobber to overwrite', args)
    else:
        myprint('Measured cube file does not exist. Creating now..', args)

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
                                'flam_u': np.array(ifu.error[i, j, :]).byteswap().newbyteorder(), \
                                'linemask': False})
                                # flam_u will have newbyteorder (or not) depending on whether it is a genuine error cube i.e. read in from saved fits file Or just a dummy array of zeros
                                # the above is to avoid "ValueError: Big-endian buffer not supported on little-endian compiler" while trying to slice the spec dataframe
                                # FYI, previous line 332: 'flam_u': np.array(ifu.error[i, j, :]) if (ifu.error == 0).all() else np.array(ifu.error[i, j, :]).byteswap().newbyteorder(), \

            if spec['flam'].any():
                #fitted_df, failed_fitting = fit_all_lines(spec, ifu.header, args, which_pixel=(i, j))
                fitted_df, failed_fitting = fit_all_lines_fixgroupz(spec, ifu.header, args, which_pixel=(i, j))
                if failed_fitting: failed_pixel_count += 1
                measured_cube.data[i, j, :] = fitted_df[measured_quantities].to_numpy().flatten() # store measured quantities for a given pixel in a 1D array
                measured_cube.error[i, j, :] = fitted_df[[item + '_u' for item in measured_quantities]].to_numpy().flatten() # store associated uncertainties
            elif args.debug:
                myprint('Pixel (' + str(i) + ',' + str(j) + '), has no flux at all, hence not bothering to fit spectra', args)

        if failed_pixel_count: myprint(str(failed_pixel_count) + ' out of ' + str(xlen * ylen) + ' pixels, i.e., ' + str(failed_pixel_count * 100. / (xlen * ylen)) + '% pixels failed to fit.', args)
        else: myprint('All pixels fitted successfully', args)

        if not (args.testcontfit or args.testlinefit): write_fitsobj(measured_cube_filename, measured_cube, instrument, args, for_qfits=True, measured_cube=True, measured_quantities=measured_quantities)  # writing into FITS file

    measured_cube = read_measured_cube(measured_cube_filename, args)
    return measured_cube

# -----------------------------------------------------------------------------
if __name__ == '__main__':
    start_time = time.time()

    args = parse_args('8508', 'RD0042')
    if type(args) is tuple: args = args[0] # if the sim has already been loaded in, in order to compute the box center (via utils.pull_halo_center()), then no need to do it again
    if not args.keep: plt.close('all')
    if (args.testlinefit or args.testcontfit) and args.test_pixel is None: raise AssertionError('Cannot test fitting without a specified test_pixel')

    measured_cube = fit_mock_spectra(args)
    print('Complete in %s minutes' % ((time.time() - start_time) / 60))
