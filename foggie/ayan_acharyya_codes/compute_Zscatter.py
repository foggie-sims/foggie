#!/usr/bin/env python3

"""

    Title :      compute_Zscatter
    Notes :      Compute scatter in metallicity distribution for a given FOGGIE galaxy
    Output :     txt file storing all the scatter parameters & mass plus, optionally, Z distribution plots
    Author :     Ayan Acharyya
    Started :    Aug 2022
    Examples :   run compute_Zscatter.py --system ayan_local --halo 8508 --output RD0042 --upto_re 3 --res 0.1 --nbins 100 --keep --weight mass
                 run compute_Zscatter.py --system ayan_local --halo 8508 --output RD0042 --upto_kpc 10 --res 0.1 --nbins 100 --weight mass
                 run compute_Zscatter.py --system ayan_pleiades --halo 8508 --upto_kpc 10 --res 0.1 --nbins 100 --xmax 4 --do_all_sims --weight mass --write_file --use_gasre --noplot

"""
from header import *
from util import *
from datashader_movie import *
from compute_MZgrad import *
from uncertainties import ufloat, unumpy
from yt.utilities.physical_ratios import metallicity_sun
from lmfit.models import SkewedGaussianModel
start_time = time.time()

# -----------------------------------------------------------
def plot_distribution(Zarr, args, weights=None, fit=None):
    '''
    Function to plot the metallicity distribution, along with the fitted skewed gaussian distribution if provided
    Saves plot as .png
    '''
    weightby_text = '' if args.weight is None else '_wtby_' + args.weight
    if args.upto_kpc is not None:
        upto_text = '_upto%.1Fckpchinv' % args.upto_kpc if args.docomoving else '_upto%.1Fkpc' % args.upto_kpc
    else:
        upto_text = '_upto%.1FRe' % args.upto_re
    outfile_rootname = 'log_metal_distribution%s%s.png' % (upto_text, weightby_text)
    if args.do_all_sims: outfile_rootname = 'z=*_' + outfile_rootname
    filename = args.fig_dir + outfile_rootname.replace('*', '%.5F' % (args.current_redshift))

    # ---------plotting histogram, and if provided, the fit---------
    fig, ax = plt.subplots(figsize=(8,8))
    fig.subplots_adjust(hspace=0.05, wspace=0.05, right=0.95, top=0.95, bottom=0.1, left=0.1)

    if args.weight is None: p = plt.hist(Zarr.flatten(), bins=args.nbins, histtype='step', lw=2, ec='salmon', density=True, label='Z')
    else: p = plt.hist(Zarr.flatten(), bins=args.nbins, histtype='step', lw=2, density=True, range=(0, args.xmax), ec='salmon', weights=weights.flatten(), label=args.weight + ' weighted Z')

    if fit is not None:
        xvals = p[1][:-1] + np.diff(p[1])
        ax.plot(xvals, fit.best_fit, c='k', lw=2, label='fit')

    # ----------tidy up figure-------------
    plt.legend(loc='lower right', fontsize=args.fontsize)
    ax.set_xlim(0, args.xmax)
    ax.set_ylim(0, 2.5)

    ax.set_xlabel(r'Z/Z$_{\odot}$', fontsize=args.fontsize)
    ax.set_ylabel('Normalised distribution', fontsize=args.fontsize)
    ax.set_xticklabels(['%.1F' % item for item in ax.get_xticks()], fontsize=args.fontsize)
    ax.set_yticklabels(['%.2F' % item for item in ax.get_yticks()], fontsize=args.fontsize)

    # ---------annotate and save the figure----------------------
    plt.text(0.95, 0.95, 'z = %.4F' % args.current_redshift, ha='right', transform=ax.transAxes, fontsize=args.fontsize)
    plt.text(0.95, 0.9, 't = %.3F Gyr' % args.current_time, ha='right', transform=ax.transAxes, fontsize=args.fontsize)
    plt.text(0.95, 0.85, args.output, ha='right', transform=ax.transAxes, fontsize=args.fontsize)
    plt.savefig(filename, transparent=False)
    myprint('Saved figure ' + filename, args)
    if not args.makemovie: plt.show(block=False)

    return fig

# -------------------------------
def fit_distribution(Zarr, args, weights=None):
    '''
    Function to fit the (log) metallicity distribution out to certain Re, given a dataframe containing metallicity
    Returns the fitted parameters for a skewed Gaussian
    '''
    print('Computing stats...')
    Zarr = Zarr.flatten()
    if weights is not None: weights = weights.flatten()

    Z25 = ufloat(np.percentile(Zarr, 25), 0)
    Z50 = ufloat(np.percentile(Zarr, 50), 0)
    Z75 = ufloat(np.percentile(Zarr, 75), 0)

    model = SkewedGaussianModel()
    params = model.make_params(amplitude=1, center=1, sigma=1, gamma=0)

    y, x = np.histogram(Zarr, bins=args.nbins, density=True, weights=weights, range=(0, args.xmax))
    x = x[:-1] + np.diff(x)/2
    result = model.fit(y, params, x=x)

    Zpeak = ufloat(y[x >= result.params['center'].value][0], 0)
    try:
        Zmean = ufloat(result.params['center'].value, result.params['center'].stderr)
        Zvar = ufloat(result.params['sigma'].value, result.params['sigma'].stderr)
        Zskew = ufloat(result.params['gamma'].value, result.params['gamma'].stderr)
        # Zkurt = ufloat(result.params['amplitude'].value, result.params['amplitude'].stderr)
    except AttributeError as e:
        print('The fit went wrong, returning NaNs')
        Zmean, Zvar, Zskew = ufloat(np.nan, np.nan) * np.ones(3)
        pass

    return result, Zpeak, Z25, Z50, Z75, Zmean, Zvar, Zskew

# -----main code-----------------
if __name__ == '__main__':
    dummy_args_tuple = parse_args('8508', 'RD0042')  # default simulation to work upon when comand line args not provided
    if type(dummy_args_tuple) is tuple: dummy_args = dummy_args_tuple[0] # if the sim has already been loaded in, in order to compute the box center (via utils.pull_halo_center()), then no need to do it again
    else: dummy_args = dummy_args_tuple
    if not dummy_args.keep: plt.close('all')

    if dummy_args.do_all_sims: list_of_sims = get_all_sims_for_this_halo(dummy_args) # all snapshots of this particular halo
    else: list_of_sims = list(itertools.product([dummy_args.halo], dummy_args.output_arr))
    total_snaps = len(list_of_sims)

    # -------set up dataframe and filename to store/write gradients in to--------
    cols_in_df = ['output', 'redshift', 'time', 're', 'mass', 'res', 'Zpeak', 'Zpeak_u', 'Z25', 'Z25_u', 'Z50', 'Z50_u', 'Z75', 'Z75_u', 'Zmean', 'Zmean_u', 'Zvar', 'Zvar_u', 'Zskew', 'Zskew_u', 'Ztotal']

    df_grad = pd.DataFrame(columns=cols_in_df)
    weightby_text = '' if dummy_args.weight is None else '_wtby_' + dummy_args.weight
    if dummy_args.upto_kpc is not None:
        upto_text = '_upto%.1Fckpchinv' % dummy_args.upto_kpc if dummy_args.docomoving else '_upto%.1Fkpc' % dummy_args.upto_kpc
    else:
        upto_text = '_upto%.1FRe' % dummy_args.upto_re
    grad_filename = dummy_args.output_dir + 'txtfiles/' + dummy_args.halo + '_MZscat%s%s.txt' % (upto_text, weightby_text)
    if dummy_args.write_file and dummy_args.clobber and os.path.isfile(grad_filename): subprocess.call(['rm ' + grad_filename], shell=True)

    if dummy_args.dryrun:
        print('List of the total ' + str(total_snaps) + ' sims =', list_of_sims)
        sys.exit('Exiting dryrun..')
    # parse column names, in case log

    # --------------read in the cold gas profile file ONCE for a given halo-------------
    if dummy_args.write_file or dummy_args.upto_kpc is None:
        foggie_dir, output_dir, run_dir, code_path, trackname, haloname, spectra_dir, infofile = get_run_loc_etc(dummy_args)
        gasfilename = '/'.join(output_dir.split('/')[:-2]) + '/' + 'mass_profiles/' + dummy_args.run + '/all_rprof_' + dummy_args.halo + '.npy'
        print('Reading in cold gas profile from', gasfilename)
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
        try:
            if len(list_of_sims) == 1 and not dummy_args.do_all_sims: args = dummy_args_tuple # since parse_args() has already been called and evaluated once, no need to repeat it
            else: args = parse_args(this_sim[0], this_sim[1])

            if type(args) is tuple:
                args, ds, refine_box = args  # if the sim has already been loaded in, in order to compute the box center (via utils.pull_halo_center()), then no need to do it again
                print_mpi('ds ' + str(ds) + ' for halo ' + str(this_sim[0]) + ' was already loaded at some point by utils; using that loaded ds henceforth', args)
            else:
                isdisk_required = np.array(['disk' in item for item in [args.xcol, args.ycol] + args.colorcol]).any()
                ds, refine_box = load_sim(args, region='refine_box', do_filter_particles=True, disk_relative=False)
        except Exception as e:
            print_mpi('Skipping ' + this_sim[1] + ' because ' + str(e), dummy_args)
            continue

        # parse paths and filenames
        args.fig_dir = args.output_dir + 'figs/' if args.do_all_sims else args.output_dir + 'figs/' + args.output + '/'
        Path(args.fig_dir).mkdir(parents=True, exist_ok=True)

        args.current_redshift = ds.current_redshift
        args.current_time = ds.current_time.in_units('Gyr').v
        if args.xmax is None: args.xmax = 4

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

            # ----------------native res--------------------------
            box = ds.r[box_center[0] - box_width_kpc / 2.: box_center[0] + box_width_kpc / 2., box_center[1] - box_width_kpc / 2.: box_center[1] + box_width_kpc / 2., box_center[2] - box_width_kpc / 2.: box_center[2] + box_width_kpc / 2., ]
            Znative = box[('gas', 'metallicity')].in_units('Zsun').ndarray_view()
            mnative = box[('gas', 'mass')].in_units('Msun').ndarray_view()
            mstar = get_disk_stellar_mass(args)  # Msun
            print('native no. of cells =', np.shape(Znative)) #
            '''
            fig, ax = plt.subplots(figsize=(8, 8))
            fig.subplots_adjust(hspace=0.05, wspace=0.05, right=0.95, top=0.95, bottom=0.1, left=0.12)
            col_arr = ['saddlebrown', 'crimson', 'darkolivegreen', 'salmon', 'cornflowerblue', 'burlywood', 'darkturquoise']
            
            pn = plt.hist(Znative, bins=args.nbins, histtype='step', lw=2, density=True, ec=col_arr[index], label='native')
            if args.weight is not None: pnw = plt.hist(Znative, bins=args.nbins, histtype='step', lw=2, density=True, ec=col_arr[index], ls='--', weights=mnative, label='native mass weighted')
            '''
            for index, res in enumerate(args.res_arr):
                ncells = int(box_width / res)
                box = ds.arbitrary_grid(left_edge=[box_center[0] - box_width_kpc / 2., box_center[1] - box_width_kpc / 2., box_center[2] - box_width_kpc / 2.], \
                                        right_edge=[box_center[0] + box_width_kpc / 2., box_center[1] + box_width_kpc / 2., box_center[2] + box_width_kpc / 2.], \
                                        dims=[ncells, ncells, ncells])
                print('res =', res, 'kpc; box shape=', np.shape(box)) #

                Zres = box['gas', 'metallicity'].in_units('Zsun').ndarray_view()
                mres = box['gas', 'mass'].in_units('Msun').ndarray_view()
                if args.weight is not None: wres = box['gas', args.weight].in_units(unit_dict[args.weight]).ndarray_view()
                else: wres = None
                Ztotal = Zres.sum() / mres.sum() / metallicity_sun # in Zsun

                result, Zpeak, Z25, Z50, Z75, Zmean, Zvar, Zskew = fit_distribution(Zres, args, weights=wres)
                if not args.noplot: fig = plot_distribution(Zres, args, weights=wres, fit=result) # plotting the Z profile, with fit

                #pr = plt.hist(Zres.flatten(), bins=args.nbins, histtype='step', lw=2, ec=col_arr[index+1], density=True, label='Z res=%.2F kpc' % (res))
                #if args.weight is not None: prw = plt.hist(Zres.flatten(), bins=args.nbins, histtype='step', lw=2, density=True, weights=wres.flatten(), ls='--', ec=col_arr[index+1], label=args.weight + ' weighted Z res=%.2Fkpc' % (res))
                thisrow += [mstar, res, Zpeak.n, Zpeak.s, Z25.n, Z25.s, Z50.n, Z50.s, Z75.n, Z75.s, Zmean.n, Zmean.s, Zvar.n, Zvar.s, Zskew.n, Zskew.s, Ztotal]
            '''
            ax.set_xlim(-0.5, 6)
            plt.legend(loc='lower right', fontsize=args.fontsize)
            ax.set_xlabel(r'$\log{(\mathrm{Z/Z}_{\odot})}$', fontsize=args.fontsize)
            ax.set_ylabel('Normalised distribution', fontsize=args.fontsize)
            ax.set_xticklabels(['%.1F' % item for item in ax.get_xticks()], fontsize=args.fontsize)
            ax.set_yticklabels(['%.2F' % item for item in ax.get_yticks()], fontsize=args.fontsize)

            filename = args.fig_dir + 'log_metal_dsitribution_%s%s.png' % (upto_text, weightby_text)
            plt.text(0.95, 0.95, 'z = %.4F' % args.current_redshift, ha='right', transform=ax.transAxes, fontsize=args.fontsize)
            plt.text(0.95, 0.9, 't = %.3F Gyr' % args.current_time, ha='right', transform=ax.transAxes, fontsize=args.fontsize)
            plt.savefig(filename, transparent=False)
            myprint('Saved figure ' + filename, args)

            plt.show(block=False)
            '''
        else:
            thisrow += (np.ones(17)*np.nan).tolist()

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
