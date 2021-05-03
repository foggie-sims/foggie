##!/usr/bin/env python3

"""

    Title :      fit_mock_spectra
    Notes :      Takes a mock IFU datacube and fits the spectra along every pixel
    Output :     FITS cube with each emission line map as a 2D slice
    Author :     Ayan Acharyya
    Started :    March 2021
    Example :    run fit_mock_spectra.py --system ayan_local --halo 8508 --output RD0042 --mergeHII 0.04 --base_spatial_res 0.4 --z 0.25 --obs_wave_range 0.8,0.85 --obs_spatial_res 1 --obs_spec_res 60 --exptime 1200 --snr 5 --debug

"""
from header import *
from util import *
from make_mock_datacube import wrap_get_mock_datacube

# -------------------------------------------------------------------------------------------------
def get_erf(lambda_array, height, centre, width, delta_lambda):
    return np.sqrt(np.pi / 2) * height * width * (
            erf((centre + delta_lambda / 2 - lambda_array) / (np.sqrt(2) * width)) - \
            erf((centre - delta_lambda / 2 - lambda_array) / (np.sqrt(
                2) * width))) / delta_lambda  # https://www.wolframalpha.com/input/?i=integrate+a*exp(-(x-b)%5E2%2F(2*c%5E2))*dx+from+(w-d%2F2)+to+(w%2Bd%2F2)

# -------------------------------------------------------------------------------------------
def fixcont_erf(x, cont, n, *p):
    result = cont
    for xx in range(0, n):
        dw = x[np.where(x >= p[3 * xx + 1])[0][0]] - x[np.where(x >= p[3 * xx + 1])[0][0] - 1]
        result += get_erf(x, p[3 * xx + 0], p[3 * xx + 1], p[3 * xx + 2], dw)
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
        spec.loc[spec['wave'].between(thiswwave - linemask_width, thiswwave + linemask_width), 'badmask'] = True
    masked_spec = spec[~spec['badmask']].reset_index(drop=True).drop('badmask', axis=1)

    wave_bins = np.linspace(rest_wave_range[0], rest_wave_range[1], int(args.nbin_cont/10))
    wave_index = np.digitize(masked_spec['wave'], wave_bins, right=True)

    smoothed_spec = pd.DataFrame()
    for thiscol in masked_spec.columns:
        smoothed_spec[thiscol] = np.array([masked_spec[thiscol].values[np.where(wave_index == ind)].mean() for ind in range(1, len(wave_bins) + 1)])
    smoothed_spec = smoothed_spec.dropna().reset_index(drop=True)  # dropping those fluxes where mean wavelength value in the bin is nan

    if args.testcontfit:
        myprint(smoothed_spec, args)
        fig = plt.figure(figsize=(17, 5))
        fig.subplots_adjust(hspace=0.7, top=0.85, bottom=0.1, left=0.05, right=0.95)

        plt.plot(masked_spec['wave'], masked_spec['flux'], c='r', label='masked flux')
        plt.scatter(masked_spec['wave'], masked_spec['flux'], c='r', lw=0)
        plt.plot(masked_spec['wave'], masked_spec['flux_u'], c='orange', linestyle='dashed', label='masked flux_u')

        plt.plot(smoothed_spec['wave'], smoothed_spec['flux'], c='k', label='smoothly sampled flux')
        plt.scatter(smoothed_spec['wave'], smoothed_spec['flux'], c='k', lw=0)
        plt.plot(smoothed_spec['wave'], smoothed_spec['flux_u'], c='gray', linestyle='dashed', label='smoothly sampled flux_u')

        plt.legend()
        plt.xlabel('Obs wavelength (A)')
        plt.ylabel('flambda (ergs/s/A/pc^2)')
        plt.title('Testing continuum fit for pixel (' + str(args.test_pixel[0]) + ',' + str(args.test_pixel[1]) + ')')
        plt.xlim(rest_wave_range[0], rest_wave_range[1])
        #plt.ylim(-2e-19, 2e-18)
        plt.show(block=False)

    # ----to estimate continuum by interpolation--------
    contfunc = interp1d(smoothed_spec['wave'], smoothed_spec['flux'], kind='cubic', fill_value='extrapolate')
    spec['cont'] = contfunc(spec['wave'])

    # ----to estimate uncertainty in continuum: option 2: measure RMS of fitted cont w.r.t input flux--------
    masked_spec['cont'] = contfunc(masked_spec['wave'])
    spec['cont_u'] = np.ones(len(spec)) * np.sqrt(np.sum((masked_spec['cont'] - masked_spec['flux']) ** 2) / len(spec))

    if args.testcontfit:
        myprint(spec, args)
        plt.plot(spec['wave'], spec['cont'], c='g', label='cont')
        plt.plot(spec['wave'], spec['cont_u'], c='g', label='cont_u', linestyle='dotted')
        plt.legend()
        plt.show(block=False)

    return spec

# -------------Fucntion for fitting multiple lines----------------------------
def fit_all_lines(args, wave, flam, flam_u, cont, pix_i, pix_j, z=0, z_err=0.0001):
    scaling = 1e-19 if args.contsub else 1.  # to make "good" numbers that python can handle
    flam /= scaling
    flam_u /= scaling
    cont /= scaling
    kk, count, flux_array, flux_error_array, vel_disp_array, vel_disp_error_array = 1, 0, [], [], [], []
    ndlambda_left, ndlambda_right = [args.nres] * 2  # how many delta-lambda wide will the window (for line fitting) be on either side of the central wavelength, default 5
    try:
        count = 1
        first, last = [logbook.wlist[0]] * 2
    except IndexError:
        pass
    while kk <= len(logbook.llist):
        center1 = last
        if kk == len(logbook.llist):
            center2 = 1e10  # insanely high number, required to plot last line
        else:
            center2 = logbook.wlist[kk]
        if center2 * (1. - ndlambda_left / logbook.resoln) > center1 * (1. + ndlambda_right / logbook.resoln):
            leftlim = first * (1. - ndlambda_left / logbook.resoln)
            rightlim = last * (1. + ndlambda_right / logbook.resoln)
            wave_short = wave[(leftlim < wave) & (wave < rightlim)]
            flam_short = flam[(leftlim < wave) & (wave < rightlim)]
            flam_u_short = flam_u[(leftlim < wave) & (wave < rightlim)]
            cont_short = cont[(leftlim < wave) & (wave < rightlim)]
            if args.debug: myprint('Trying to fit ' + str(logbook.llist[kk - count:kk]) + ' line/s at once. Total ' + str(count) + '\n',
                args)
            try:
                popt, pcov = fitline(wave_short, flam_short, flam_u_short, logbook.wlist[kk - count:kk], logbook.resoln,
                                     z=z, z_err=z_err, contsub=args.contsub)
                popt, pcov = np.array(popt), np.array(pcov)
                level = 0. if args.contsub else 1.
                popt = np.concatenate(([level],
                                       popt))  # for fitting after continuum normalised (OR subtracted), so continuum is fixed=1 (OR 0) and has to be inserted to popt[] by hand after fitting
                pcov = np.hstack((np.zeros((np.shape(pcov)[0] + 1, 1)), np.vstack((np.zeros(np.shape(pcov)[1]),
                                                                                   pcov))))  # for fitting after continuum normalised (OR subtracted), so error in continuum is fixed=0 and has to be inserted to pcov[] by hand after fitting
                if args.showfit:  #
                    plt.axvline(leftlim, linestyle='--', c='g')
                    plt.axvline(rightlim, linestyle='--', c='g')

                ndlambda_left, ndlambda_right = [args.nres] * 2
                if args.debug: myprint('Done this fitting!' + '\n', args)

            except TypeError:
                if args.debug: myprint('Trying to re-do this fit with broadened wavelength window..\n', args)
                ndlambda_left += 1
                ndlambda_right += 1
                continue
            except (RuntimeError, ValueError):
                level = 0. if args.contsub else 1.
                popt = np.concatenate(([level], np.zeros(
                    count * 3)))  # if could not fit the line/s fill popt with zeros so flux_array gets zeros
                pcov = np.zeros((count * 3 + 1,
                                 count * 3 + 1))  # if could not fit the line/s fill popt with zeros so flux_array gets zeros
                if args.debug: myprint('Could not fit lines ' + str(logbook.llist[kk - count:kk]) + ' for pixel ' + str(
                    pix_i) + ', ' + str(pix_j) + '\n', args)
                pass

            for xx in range(0, count):
                # in popt for every bunch of lines,
                # elements (0,1,2) or (3,4,5) etc. are the height(b), mean(c) and width(d)
                # so, for each line the elements (cont=a,0,1,2) or (cont=a,3,4,5) etc. make the full suite of (a,b,c,d) gaussian parameters
                # so, for each line, flux f (area under gaussian) = sqrt(2pi)*(b-a)*d
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
                # so we also need to include off diagnoal covariance terms while propagating flux errors
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
                cont_at_line = cont[np.where(wave >= logbook.wlist[kk + xx - count])[0][0]]
                if args.debug and args.oneHII is not None: print('Debugging534: linefit param at (', pix_i, ',', pix_j, ') for ', logbook.llist[kk + xx - count], '(ergs/s/pc^2/A) =', popt_single)  #
                flux = np.sqrt(2 * np.pi) * (popt_single[1] - popt_single[0]) * popt_single[3] * scaling  # total flux = integral of guassian fit ; resulting flux in ergs/s/pc^2 units
                if args.debug and not args.contsub: flux *= cont_at_line  # if continuum is normalised (and NOT subtracted) then need to change back to physical units by multiplying continuum at that wavelength
                if args.oneHII is not None: print('Debugging536: lineflux at (', pix_i, ',', pix_j, ') for ', logbook.llist[kk + xx - count], '(ergs/s/pc^2/A) =', flux)  #
                flux_error = np.sqrt(2 * np.pi * (popt_single[3] ** 2 * (pcov[0][0] + pcov[3 * xx + 1][3 * xx + 1]) \
                                                  + (popt_single[1] - popt_single[0]) ** 2 * pcov[3 * (xx + 1)][
                                                      3 * (xx + 1)] \
                                                  - 2 * popt_single[3] ** 2 * pcov[3 * xx + 1][0] \
                                                  + 2 * (popt_single[1] - popt_single[0]) * popt_single[3] * (
                                                          pcov[3 * xx + 1][3 * (xx + 1)] - pcov[0][3 * (xx + 1)]) \
                                                  )) * scaling  # var_f = 3^2(00 + 11) + (1-0)^2*33 - (2)*3^2*10 + (2)*3*(1-0)*(13-03)
                vel_disp = c * popt_single[3]/popt_single[2] # in km/s
                vel_disp_error = c * np.sqrt(pcov[3 * xx + 3][3 * xx + 3]/(popt_single[2] ** 2) + popt_single[3]**2 * pcov[3 * xx + 2][3 * xx + 2] / (popt_single[2] ** 4) + 2 * popt_single[3] * pcov[3 * xx + 2][3 * xx + 3]/ (popt_single[2] ** 3)) # in km/s
                if not args.contsub: flux_error *= cont_at_line  # if continuum is normalised (and NOT subtracted) then need to change back to physical units by multiplying continuum at that wavelength

                flux_array.append(flux)
                flux_error_array.append(flux_error)
                vel_disp_array.append(vel_disp)
                vel_disp_error_array.append(vel_disp_error)
                if args.showfit:
                    leftlim = popt_single[2] * (1. - args.nres / logbook.resoln)
                    rightlim = popt_single[2] * (1. + args.nres / logbook.resoln)
                    wave_short_single = wave[(leftlim < wave) & (wave < rightlim)]
                    cont_short_single = cont[(leftlim < wave) & (wave < rightlim)]
                    if args.contsub: plt.plot(wave_short_single, (su.gaus(wave_short_single,1, *popt_single) + cont_short_single)*scaling,lw=1, c='r') # adding back the continuum just for plotting purpose
                    else: plt.plot(wave_short_single, su.gaus(wave_short_single,1, *popt_single)*cont_short_single,lw=1, c='r')
                    count = 1
            if args.showfit:
                if count > 1:
                    plt.plot(wave_short, undo_contnorm(su.gaus(wave_short, count, *popt), cont_short, contsub=args.contsub), lw=2, c='brown')
                plt.draw()

            first, last = [center2] * 2
        else:
            last = center2
            count += 1
        kk += 1
    # -------------------------------------------------------------------------------------------
    flux_array = np.array(flux_array)
    flux_error_array = np.array(flux_error_array)
    vel_disp_array = np.array(vel_disp_array)
    vel_disp_error_array = np.array(vel_disp_error_array)
    return flux_array, flux_error_array, vel_disp_array, vel_disp_error_array

# -------------------------------------------------------------------------------------------
def fitline(spec, waves_to_fit, fits_header, args):
    '''
    Function to fit one group of neighbouring emission line
    :param spec:
    :param waves_to_fit:
    :param fits_header:
    :param args:
    :return:
    '''
    if 'obs_spec_res(km/s)' in fits_header: vel_res = fits_header['obs_spec_res(km/s)']  # 10*vres in km/s
    else: vel_res = fits_header['base_spec_res(km/s)']
    v_maxwidth = 10 * vel_res
    R = c / vel_res
    z_allow = 3e-3  # wavelengths are at restframe; assumed error in redshift

    p_init, lbound, ubound = [], [], []
    for xx in range(0, len(waves_to_fit)):
        p_init = np.append(p_init, [np.max(spec['flux']) - spec['flux'].values[0], waves_to_fit[xx], waves_to_fit[xx] * 2. * gf2s / R])
        lbound = np.append(lbound, [0., waves_to_fit[xx] * (1. - z_allow), waves_to_fit[xx] * 1. * gf2s / R])
        ubound = np.append(ubound, [np.inf, waves_to_fit[xx] * (1. + z_allow), waves_to_fit[xx] * v_maxwidth * gf2s / c])
    level = 0. if args.contsub else 1.

    if spec['flux_u'].any():
        popt, pcov = curve_fit(lambda x, *p: fixcont_erf(x, level, len(waves_to_fit), *p), spec['wave'], spec['flux'], p0=p_init, maxfev=10000, bounds=(lbound, ubound), sigma=spec['flux_u'], absolute_sigma=True)
    else:
        popt, pcov = curve_fit(lambda x, *p: fixcont_erf(x, level, len(waves_to_fit), *p), spec['wave'], spec['flux'], p0=p_init, max_nfev=10000, bounds=(lbound, ubound))

    if args.testlinefit:
        plt.figure()
        plt.plot(spec['wave'], spec['flux'], c='k', label='flam')
        plt.plot(spec['wave'], spec['flux_u'], c='gray', label='flam_u')
        plt.plot(spec['wave'], spec['cont'], c='g', label='cont')

        plt.plot(spec['wave'], fixcont_erf(spec['wave'], level, len(waves_to_fit), *p_init), c='b', label='initial guess')
        plt.plot(spec['wave'], fixcont_erf(spec['wave'], level, len(waves_to_fit), *popt), c='r', label='best fit')

        #plt.ylim(-0.2e-18, 1.2e-18)
        plt.legend()
        plt.xlabel('Rest wavelength (A)')
        plt.ylabel('flambda (ergs/s/A/pc^2)')
        plt.title('Testing line fit for pixel (' + str(args.test_pixel[0]) + ',' + str(args.test_pixel[1]) + ')')
        plt.show(block=False)

    return popt, pcov

# -----------------------------------------------------------------------------
if __name__ == '__main__':
    start_time = time.time()

    args = parse_args('8508', 'RD0042')
    args.diag = args.diag_arr[0]
    args.Om = args.Om_arr[0]
    if not args.keep: plt.close('all')

    cube_output_path = get_cube_output_path(args)
    args.idealcube_filename = cube_output_path + 'ideal_ifu' + args.mergeHII_text + '.fits'

    ifu = readcube(args.idealcube_filename, args)
    spec = pd.DataFrame({'flux': ifu.data[args.test_pixel[0], args.test_pixel[1], :], \
                         'wave': ifu.wavelength, \
                         'flux_u': np.zeros(len(ifu.wavelength)), \
                         'badmask': False})

    spec = fitcont(spec, ifu.header, args)

    print('Complete in %s minutes' % ((time.time() - start_time) / 60))
