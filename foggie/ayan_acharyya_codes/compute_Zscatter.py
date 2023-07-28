#!/usr/bin/env python3

"""

    Title :      compute_Zscatter
    Notes :      Compute scatter in metallicity distribution for a given FOGGIE galaxy
    Output :     txt file storing all the scatter parameters & mass plus, optionally, Z distribution plots
    Author :     Ayan Acharyya
    Started :    Aug 2022
    Examples :   run compute_Zscatter.py --system ayan_local --halo 8508 --output RD0042 --upto_re 3 --res 0.1 --nbins 100 --keep --weight mass
                 run compute_Zscatter.py --system ayan_local --halo 8508 --output RD0042 --upto_kpc 10 --nbins 100 --weight mass --docomoving --fit_multiple --hide_multiplefit --forproposal
                 run compute_Zscatter.py --system ayan_local --halo 8508 --output RD0030 --upto_kpc 10 --nbins 100 --weight mass --forpaper
                 run compute_Zscatter.py --system ayan_pleiades --halo 8508 --upto_kpc 10 --res 0.1 --nbins 100 --xmax 4 --do_all_sims --weight mass --write_file --use_gasre --noplot

"""
from header import *
from util import *
from datashader_movie import *
from compute_MZgrad import *
from uncertainties import ufloat, unumpy
from yt.utilities.physical_ratios import metallicity_sun
from lmfit.models import GaussianModel, SkewedGaussianModel
from pygini import gini
start_time = time.time()

# ----------------------------------------------------------------------------
# Following function is adapted from https://stackoverflow.com/questions/21844024/weighted-percentile-using-numpy
def weighted_quantile(values, quantiles, sample_weight=None):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of initial array
    :param old_style: if True, will correct output to be consistent with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    if sample_weight is None: sample_weight = np.ones(len(values))
    values = np.array(values)
    quantiles = np.array(quantiles)
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), 'quantiles should be in [0, 1]'

    sorter = np.argsort(values)
    values = values[sorter]
    sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)

# -----------------------------------------------------------
def plot_distribution(Zarr, args, weights=None, fit=None, percentiles=None):
    '''
    Function to plot the metallicity distribution, along with the fitted skewed gaussian distribution if provided
    Saves plot as .png
    '''
    fit, percentiles = None, None ##
    if args.forproposal and args.output == 'RD0042': plt.rcParams['axes.linewidth'] = 1
    weightby_text = '' if args.weight is None else '_wtby_' + args.weight
    fitmultiple_text = '_fitmultiple' if args.fit_multiple else ''
    density_cut_text = '_wdencut' if args.use_density_cut else ''
    islog_text = '_islog' if args.islog else ''
    if args.upto_kpc is not None:
        upto_text = '_upto%.1Fckpchinv' % args.upto_kpc if args.docomoving else '_upto%.1Fkpc' % args.upto_kpc
    else:
        upto_text = '_upto%.1FRe' % args.upto_re
    outfile_rootname = '%s_log_metal_distribution%s%s%s%s%s.png' % (args.output, upto_text, weightby_text, fitmultiple_text, density_cut_text, islog_text)
    if args.do_all_sims: outfile_rootname = 'z=*_' + outfile_rootname[len(args.output)+1:]
    filename = args.fig_dir + outfile_rootname.replace('*', '%.5F' % (args.current_redshift))

    # ---------plotting histogram, and if provided, the fit---------
    if args.forproposal:
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.subplots_adjust(right=0.95, top=0.9, bottom=0.2, left=0.2)
    else:
        fig, ax = plt.subplots(figsize=(8, 8))
        fig.subplots_adjust(hspace=0.05, wspace=0.05, right=0.95, top=0.95, bottom=0.1, left=0.15)

    color = 'cornflowerblue' if args.fortalk else 'salmon'
    if args.weight is None: p = plt.hist(Zarr.flatten(), bins=args.nbins, histtype='step', lw=2, ec=color, density=True)
    else: p = plt.hist(Zarr.flatten(), bins=args.nbins, histtype='step', lw=2, density=True, ec=color, weights=weights.flatten())

    if fit is not None and not (args.forproposal and args.output != 'RD0042'):
        fit_color = 'darkorange' if args.fortalk else 'k'
        xvals = p[1][:-1] + np.diff(p[1])
        ax.plot(xvals, fit.best_fit, c=fit_color, lw=2, label=None if args.forproposal else 'Best fit')
        #ax.plot(xvals, fit.init_fit, c='b', lw=2, label='Initial guess') #
        if not args.hide_multiplefit:
            ax.plot(xvals, GaussianModel().eval(x=xvals, amplitude=fit.best_values['g_amplitude'], center=fit.best_values['g_center'], sigma=fit.best_values['g_sigma']), c=fit_color, lw=2, ls='--', label='Regular Gaussian')
            ax.plot(xvals, SkewedGaussianModel().eval(x=xvals, amplitude=fit.best_values['sg_amplitude'], center=fit.best_values['sg_center'], sigma=fit.best_values['sg_sigma'], gamma=fit.best_values['sg_gamma']), c=fit_color, lw=2, ls='dotted', label='Skewed Gaussian')

    # ----------adding vertical lines-------------
    if fit is not None and not (args.forproposal and args.output != 'RD0042'):
        ax.axvline(fit.best_values['sg_center'], lw=2, ls='dotted', color=fit_color)
        if args.fit_multiple: ax.axvline(fit.best_values['g_center'], lw=2, ls='dashed', color=fit_color)

    if percentiles is not None and not args.fortalk:
        for thisper in np.atleast_1d(percentiles): ax.axvline(thisper, lw=2.5, ls='solid', color='crimson')

    # ----------adding arrows--------------
    if args.annotate_profile and not args.notextonplot:
        ax.annotate('Low metallicity outer disk,\nfitted with a regular Gaussian', xy=(0.2, 1.5), xytext=(0.8, 1.2), arrowprops=dict(color='gray', lw=2, arrowstyle='->'), ha='left', fontsize=args.fontsize/1.2, bbox=dict(facecolor='gray', alpha=0.3, edgecolor='k'))
        ax.annotate('Higher metallicity inner disk,\nfitted with a skewed Gaussian', xy=(1.7, 0.5), xytext=(1.5, 0.75), arrowprops=dict(color='gray', lw=2, arrowstyle='->'), ha='left', fontsize=args.fontsize/1.2, bbox=dict(facecolor='gray', alpha=0.3, edgecolor='k'))

    # ----------tidy up figure-------------
    if not (args.annotate_profile or args.notextonplot): plt.legend(loc='upper right', bbox_to_anchor=(1, 0.75), fontsize=args.fontsize)
    ax.set_xlim(args.xmin, args.xmax)
    ax.set_ylim(0, args.ymax)

    ax.set_xticks(np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 5))
    ax.set_yticks(np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 4 if args.forproposal else 6))

    ax.set_xlabel(r'Log Metallicity (Z$_{\odot}$)' if args.islog else r'Metallicity (Z$_{\odot}$)', fontsize=args.fontsize)
    ax.set_ylabel('Normalised distribution', fontsize=args.fontsize/1.2 if args.forproposal else args.fontsize)
    ax.set_xticklabels(['%.1F' % item for item in ax.get_xticks()], fontsize=args.fontsize)
    ax.set_yticklabels(['%.2F' % item for item in ax.get_yticks()], fontsize=args.fontsize)

    if args.fortalk:
        #mplcyberpunk.add_glow_effects()
        try: mplcyberpunk.make_lines_glow()
        except: pass
        try: mplcyberpunk.make_scatter_glow()
        except: pass

    # ---------annotate and save the figure----------------------
    if not args.notextonplot:
        plt.text(0.97, 0.95, 'z = %.2F' % args.current_redshift, ha='right', transform=ax.transAxes, fontsize=args.fontsize)
        plt.text(0.97, 0.9, 't = %.1F Gyr' % args.current_time, ha='right', transform=ax.transAxes, fontsize=args.fontsize)

        if fit is not None:
            plt.text(0.97, 0.8, r'Mean = %.2F Z$\odot$' % fit.best_values['sg_center'], ha='right', transform=ax.transAxes, fontsize=args.fontsize)
            plt.text(0.97, 0.75, r'Sigma = %.2F Z$\odot$' % fit.best_values['sg_sigma'], ha='right', transform=ax.transAxes, fontsize=args.fontsize)
    plt.savefig(filename, transparent=args.fortalk)
    myprint('Saved figure ' + filename, args)
    if not args.makemovie: plt.show(block=False)

    return fig

# -------------------------------
def fit_distribution(Zarr, args, weights=None):
    '''
    Function to fit the (log) metallicity distribution out to certain Re, given a dataframe containing metallicity
    Returns the fitted parameters for a skewed Gaussian
    '''
    Zarr = Zarr.flatten()
    if weights is not None: weights = weights.flatten()

    y, x = np.histogram(Zarr, bins=args.nbins, density=True, weights=weights)
    x = x[:-1] + np.diff(x)/2

    model = SkewedGaussianModel(prefix='sg_')
    if args.islog: params = model.make_params(sg_amplitude=2.0, sg_center=0.1, sg_sigma=0.5, sg_gamma=-3)
    else: params = model.make_params(sg_amplitude=0.5, sg_center=0.5, sg_sigma=0.5, sg_gamma=0)

    if args.fit_multiple: # fitting a skewed gaussian + another regular gaussian using curve_fit()
        print('Fitting with one skewed gaussian + one regular guassian...')
        g_model = GaussianModel(prefix='g_')
        if args.islog: params.update(g_model.make_params(g_amplitude=2, g_center=-0.9, g_sigma=0.05))
        else: params.update(g_model.make_params(g_amplitude=2, g_center=-0.9, g_sigma=0.05))
        model = model + g_model
    else: # fitting a single skewed gaussian with SkewedGaussianModel()
        print('Fitting with one skewed gaussian...')

    result = model.fit(y, params, x=x)
    print('Fitted parameters:\n', result.best_values)

    return result

# -----main code-----------------
if __name__ == '__main__':
    dummy_args_tuple = parse_args('8508', 'RD0042')  # default simulation to work upon when comand line args not provided
    if type(dummy_args_tuple) is tuple: dummy_args = dummy_args_tuple[0] # if the sim has already been loaded in, in order to compute the box center (via utils.pull_halo_center()), then no need to do it again
    else: dummy_args = dummy_args_tuple
    if not dummy_args.keep: plt.close('all')

    if dummy_args.do_all_sims: list_of_sims = get_all_sims_for_this_halo(dummy_args) # all snapshots of this particular halo
    else: list_of_sims = list(itertools.product([dummy_args.halo], dummy_args.output_arr))
    total_snaps = len(list_of_sims)

    if dummy_args.forpaper:
        dummy_args.res = 0.1  # kpc
        dummy_args.docomoving = True
        dummy_args.islog = True
        dummy_args.use_density_cut = True
        dummy_args.fit_multiple = True

    # -------set up dataframe and filename to store/write gradients in to--------
    cols_in_df = ['output', 'redshift', 'time', 're', 'mass', 'res', 'Zpeak', 'Zpeak_u', 'Z25', 'Z25_u', 'Z50', 'Z50_u', 'Z75', 'Z75_u', 'Zgini', 'Zmean', 'Zmean_u', 'Zvar', 'Zvar_u', 'Zskew', 'Zskew_u', 'Ztotal', 'gauss_amp', 'gauss_amp_u', 'gauss_mean', 'gauss_mean_u', 'gauss_sigma', 'gauss_sigma_u']

    df_grad = pd.DataFrame(columns=cols_in_df)
    weightby_text = '' if dummy_args.weight is None else '_wtby_' + dummy_args.weight
    fitmultiple_text = '_fitmultiple' if dummy_args.fit_multiple else ''
    density_cut_text = '_wdencut' if dummy_args.use_density_cut else ''
    islog_text = '_islog' if dummy_args.islog else ''
    if dummy_args.upto_kpc is not None:
        upto_text = '_upto%.1Fckpchinv' % dummy_args.upto_kpc if dummy_args.docomoving else '_upto%.1Fkpc' % dummy_args.upto_kpc
    else:
        upto_text = '_upto%.1FRe' % dummy_args.upto_re
    grad_filename = dummy_args.output_dir + 'txtfiles/' + dummy_args.halo + '_MZscat%s%s%s%s%s.txt' % (upto_text, weightby_text, fitmultiple_text, density_cut_text, islog_text)
    if dummy_args.write_file and dummy_args.clobber and os.path.isfile(grad_filename): subprocess.call(['rm ' + grad_filename], shell=True)

    if dummy_args.dryrun:
        print('List of the total ' + str(total_snaps) + ' sims =', list_of_sims)
        sys.exit('Exiting dryrun..')
    # parse column names, in case log

    # --------------read in the cold gas profile file ONCE for a given halo-------------
    if (dummy_args.write_file or dummy_args.upto_kpc is None) and dummy_args.use_gasre:
        foggie_dir, output_dir, run_dir, code_path, trackname, haloname, spectra_dir, infofile = get_run_loc_etc(dummy_args)
        gasfilename = '/'.join(output_dir.split('/')[:-2]) + '/' + 'mass_profiles/' + dummy_args.run + '/all_rprof_' + dummy_args.halo + '.npy'

        if os.path.exists(gasfilename):
            print('Reading in cold gas profile from', gasfilename)
        else:
            print('Did not find', gasfilename)
            gasfilename = gasfilename.replace(dummy_args.run, dummy_args.run[:14])
            print('Instead, reading in cold gas profile from', gasfilename)
        gasprofile = np.load(gasfilename, allow_pickle=True)[()]
    else:
        print('Not reading in cold gas profile because any re calculation is not needed')
        gasprofile = None

    # --------domain decomposition; for mpi parallelisation-------------
    comm = MPI.COMM_WORLD
    ncores = comm.size
    rank = comm.rank
    print_master('Total number of MPI ranks = ' + str(ncores) + '. Starting at: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()), dummy_args)
    comm.Barrier() # wait till all cores reached here and then resume

    split_at_cpu = total_snaps - ncores * int(total_snaps/ncores)
    nper_cpu1 = int(total_snaps / ncores)
    nper_cpu2 = nper_cpu1 + 1
    if rank < split_at_cpu:
        core_start = rank * nper_cpu2
        core_end = (rank+1) * nper_cpu2 - 1
    else:
        core_start = split_at_cpu * nper_cpu2 + (rank - split_at_cpu) * nper_cpu1
        core_end = split_at_cpu * nper_cpu2 + (rank - split_at_cpu + 1) * nper_cpu1 - 1

    # --------------------------------------------------------------
    print_mpi('Operating on snapshots ' + str(core_start + 1) + ' to ' + str(core_end + 1) + ', i.e., ' + str(core_end - core_start + 1) + ' out of ' + str(total_snaps) + ' snapshots', dummy_args)

    for index in range(core_start + dummy_args.start_index, core_end + 1):
        start_time_this_snapshot = time.time()
        this_sim = list_of_sims[index]
        this_df_grad = pd.DataFrame(columns=cols_in_df)
        print_mpi('Doing snapshot ' + this_sim[1] + ' of halo ' + this_sim[0] + ' which is ' + str(index + 1 - core_start) + ' out of the total ' + str(core_end - core_start + 1) + ' snapshots...', dummy_args)
        halos_df_name = dummy_args.code_path + 'halo_infos/00' + this_sim[0] + '/' + dummy_args.run + '/' + 'halo_cen_smoothed'
        try:
            if len(list_of_sims) == 1 and not dummy_args.do_all_sims: args = dummy_args_tuple # since parse_args() has already been called and evaluated once, no need to repeat it
            else: args = parse_args(this_sim[0], this_sim[1])

            if type(args) is tuple:
                args, ds, refine_box = args  # if the sim has already been loaded in, in order to compute the box center (via utils.pull_halo_center()), then no need to do it again
                print_mpi('ds ' + str(ds) + ' for halo ' + str(this_sim[0]) + ' was already loaded at some point by utils; using that loaded ds henceforth', args)
            else:
                isdisk_required = np.array(['disk' in item for item in [args.xcol, args.ycol] + args.colorcol]).any()
                ds, refine_box = load_sim(args, region='refine_box', do_filter_particles=True, disk_relative=False, halo_c_v_name=halos_df_name)
        except Exception as e:
            print_mpi('Skipping ' + this_sim[1] + ' because ' + str(e), dummy_args)
            continue

        # parse paths and filenames
        args.fig_dir = args.output_dir + 'figs/' if args.do_all_sims else args.output_dir + 'figs/' + args.output + '/'
        Path(args.fig_dir).mkdir(parents=True, exist_ok=True)
        if args.fortalk:
            setup_plots_for_talks()
            args.forpaper = True
            args.notextonplot = True
        if args.forpaper:
            args.res = 0.1  # kpc
            if not args.fortalk: args.docomoving = True
            args.islog = True
            args.use_density_cut = True
            args.fit_multiple = True
        elif args.forproposal:
            args.res = 0.1  # kpc
            args.notextonplot = True


        args.current_redshift = ds.current_redshift
        args.current_time = ds.current_time.in_units('Gyr').v
        if args.xmin is None:
            args.xmin = -1.5 if args.islog else 0
        if args.xmax is None:
            args.xmax = 2 if args.forproposal else 0.6 if args.islog else 4
        if args.ymax is None:
            args.ymax = 1.5 if args.forproposal else 2.0 if args.use_density_cut else 2.5 if args.islog else 2.5

        if args.write_file or args.upto_kpc is None:
            args.re = get_re_from_coldgas(gasprofile, args) if args.use_gasre else get_re_from_stars(ds, args)
        else:
            args.re = np.nan
        thisrow = [args.output, args.current_redshift, args.current_time, args.re] # row corresponding to this snapshot to append to df

        if args.upto_kpc is not None:
            if args.docomoving: args.galrad = args.upto_kpc / (1 + args.current_redshift) / 0.695  # fit within a fixed comoving kpc h^-1, 0.695 is Hubble constant
            else: args.galrad = args.upto_kpc  # fit within a fixed physical kpc
        else:
            args.galrad = args.re * args.upto_re  # kpc

        if args.galrad > 0:
            # extract the required box
            box_center = ds.arr(args.halo_center, kpc)
            box_width = args.galrad * 2  # in kpc
            box_width_kpc = ds.arr(box_width, 'kpc')
            mstar = get_disk_stellar_mass(args)  # Msun

            # ----------------native res--------------------------
            if args.get_native_res:
                box = ds.r[box_center[0] - box_width_kpc / 2.: box_center[0] + box_width_kpc / 2., box_center[1] - box_width_kpc / 2.: box_center[1] + box_width_kpc / 2., box_center[2] - box_width_kpc / 2.: box_center[2] + box_width_kpc / 2., ]
                if args.use_density_cut:
                    rho_cut = get_density_cut(args.current_time)  # based on Cassi's CGM-ISM density cut-off
                    box = box.cut_region(['obj["gas", "density"] > %.1E' % rho_cut])
                    print('Imposing a density criteria to get ISM above density', rho_cut, 'g/cm^3')

                Znative = box[('gas', 'metallicity')].in_units('Zsun').ndarray_view()
                mnative = box[('gas', 'mass')].in_units('Msun').ndarray_view()
                print('native no. of cells =', np.shape(Znative))

            # ----------------binned res-----------------------------
            for index, res in enumerate(args.res_arr):
                ncells = int(box_width / res)
                box = ds.arbitrary_grid(left_edge=[box_center[0] - box_width_kpc / 2., box_center[1] - box_width_kpc / 2., box_center[2] - box_width_kpc / 2.], \
                                        right_edge=[box_center[0] + box_width_kpc / 2., box_center[1] + box_width_kpc / 2., box_center[2] + box_width_kpc / 2.], \
                                        dims=[ncells, ncells, ncells])
                print('res =', res, 'kpc; box shape=', np.shape(box)) #
                if args.use_density_cut:
                    rho_cut = get_density_cut(args.current_time)  # based on Cassi's CGM-ISM density cut-off
                    box = box.cut_region(['obj["gas", "density"] > %.1E' % rho_cut])
                    print('Imposing a density criteria to get ISM above density', rho_cut, 'g/cm^3')

                Zres = box['gas', 'metallicity'].in_units('Zsun').ndarray_view()
                mres = box['gas', 'mass'].in_units('Msun').ndarray_view()
                if args.weight is not None: wres = box['gas', args.weight].in_units(unit_dict[args.weight]).ndarray_view()
                else: wres = None
                Ztotal = np.sum(Zres * mres) / np.sum(mres) # in Zsun
                if args.islog:
                    Zres = np.log10(Zres) # all operations will be done in log
                    wres = np.ma.compressed(np.ma.array(wres, mask=np.ma.masked_where(wres, Zres < -4))) # discarding wild values of metallicity
                    Zres = np.ma.compressed(np.ma.array(Zres, mask=np.ma.masked_where(Zres, Zres < -4)))

                if args.Zcut is not None: # testing if can completely chop-off low-gaussian component
                    print('Chopping off histogram at a fixed %.1F, therefore NOT fiting multiple components' % args.Zcut)
                    wres = np.ma.compressed(np.ma.array(wres, mask=np.ma.masked_where(wres, Zres < args.Zcut)))
                    Zres = np.ma.compressed(np.ma.array(Zres, mask=np.ma.masked_where(Zres, Zres < args.Zcut)))
                    args.fit_multiple = False

                result = fit_distribution(Zres, args, weights=wres)

                # -------computing quantities to save in file-------------------------
                print('Computing stats...')
                percentiles = np.percentile(Zres, [25, 50, 75])
                #percentiles = weighted_quantile(Zres, [0.25, 0.50, 0.75], sample_weight=wres)
                Zgini = gini(Zres)

                thisrow += [mstar, res, result.best_values['sg_amplitude'], result.params['sg_amplitude'].stderr, \
                            percentiles[0], 0, percentiles[1], 0, percentiles[2], 0, Zgini, \
                            result.best_values['sg_center'], result.params['sg_center'].stderr, \
                            result.best_values['sg_sigma'], result.params['sg_sigma'].stderr, \
                            result.best_values['sg_gamma'], result.params['sg_gamma'].stderr, Ztotal, \
                            result.best_values['g_amplitude'], result.params['g_amplitude'].stderr, \
                            result.best_values['g_center'], result.params['g_center'].stderr, \
                            result.best_values['g_sigma'], result.params['g_sigma'].stderr]
                if not args.noplot: fig = plot_distribution(Zres, args, weights=wres, fit=result, percentiles=percentiles) # plotting the Z profile, with fit

        else:
            thisrow += (np.ones(23)*np.nan).tolist()

        this_df_grad.loc[len(this_df_grad)] = thisrow
        df_grad = pd.concat([df_grad, this_df_grad])
        if args.write_file:
            if not os.path.isfile(grad_filename):
                this_df_grad.to_csv(grad_filename, sep='\t', index=None, header='column_names')
                print('Wrote to gradient file', grad_filename)
            else:
                this_df_grad.to_csv(grad_filename, sep='\t', index=None, mode='a', header=False)
                print('Appended to gradient file', grad_filename)

        print_mpi('This snapshots completed in %s mins' % ((time.time() - start_time_this_snapshot) / 60), dummy_args)


    if ncores > 1: print_master('Parallely: time taken for ' + str(total_snaps) + ' snapshots with ' + str(ncores) + ' cores was %s mins' % ((time.time() - start_time) / 60), dummy_args)
    else: print_master('Serially: time taken for ' + str(total_snaps) + ' snapshots with ' + str(ncores) + ' core was %s mins' % ((time.time() - start_time) / 60), dummy_args)
