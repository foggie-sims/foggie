#!/usr/bin/env python3

"""

    Title :      plot_allZ_movie
    Notes :      Plot time evolution of metallicity gradient, metallicity distribution and metallicity profile ALL in one plot
    Output :     Combined plots as png files plus, optionally, these files stitched into a movie
    Author :     Ayan Acharyya
    Started :    Dec 2022
    Examples :   run plot_allZ_movie.py --system ayan_pleiades --halo 8508 --Zgrad_den kpc --upto_kpc 10 --docomoving --res 0.1 --weight mass --zhighlight --overplot_smoothed 1000 --plot_timefraction --Zgrad_allowance 0.05 --upto_z 2 --do_all_sims
                 run plot_allZ_movie.py --system ayan_local --halo 8508 --Zgrad_den kpc --upto_kpc 10 --docomoving --res 0.1 --weight mass --zhighlight --overplot_smoothed 1000 --output RD0030
"""
from header import *
from util import *
from lmfit.models import GaussianModel, SkewedGaussianModel
from compute_MZgrad import get_df_from_ds
from plot_MZgrad import plot_zhighlight
from plot_MZscatter import load_df
from datashader_movie import unit_dict
from uncertainties import ufloat

start_time = time.time()
plt.rcParams["axes.linewidth"] = 1

# ----------------------------------------------------
def plot_full_evolution(df, axes, args, col_arr=['green', 'darkorgange']):
    '''
    Function to plot the time evolution of Zgrad
    '''
    # -------------which quantities to plot in which axis-----------------
    groups = pd.DataFrame({'quantities': [['Zgrad', 'Zgrad_binned'], ['Z50', 'ZIQR'], ['Zmean', 'Zvar']], \
                           'legend': [['Fit to all cells', 'Fit to radial bins'], ['Median Z', 'Inter-quartile range'], ['Fitted mean', 'Fitted width']], \
                           'label': np.hstack([r'$\nabla Z$ (dex/kpc)', np.tile([r'Z/Z$_\odot$'], 2)]), \
                           'limits': [(-0.5, 0.1), (1e-3, 8), (1e-4, 2)]})
    groups = groups[groups.index.isin([0, 2])]
    xlimits = [0, 14] # Gyr
    args.xcol = 'time'

    # --------loop over different Zgrad measurements-------------
    for j in range(len(groups)):
        thisgroup = groups.iloc[j]
        ax = axes[j]
        for i, args.ycol in enumerate(thisgroup.quantities):
            # -----plot line with color gradient--------
            ax.plot(df[args.xcol], df[args.ycol], c=col_arr[i], lw=1, label=thisgroup.legend[i])
            if args.zhighlight: ax = plot_zhighlight(df, ax, col_arr[i], args, ycol=args.ycol)

            # ------- overplotting a boxcar smoothed version of the MZGR------------
            if args.overplot_smoothed:
                mean_dt = (df[args.xcol].max() - df[args.xcol].min())*1000/len(df) # Myr
                npoints = int(np.round(args.overplot_smoothed/mean_dt))
                if npoints % 2 == 0: npoints += 1
                box = np.ones(npoints) / npoints
                df[args.ycol + '_smoothed'] = np.convolve(df[args.ycol], box, mode='same')
                ax.plot(df[args.xcol], df[args.ycol + '_smoothed'], c=col_arr[i], lw=0.5)
                print('Boxcar-smoothed plot for halo', args.halo, 'with', npoints, 'points, =', npoints * mean_dt, 'Myr')

            # ------- overplotting a lower cadence version of the MZGR------------
            elif args.overplot_cadence:
                mean_dt = (df[args.xcol].max() - df[args.xcol].min())*1000/len(df) # Myr
                npoints = int(np.round(args.overplot_cadence/mean_dt))
                df_short = df.iloc[::npoints, :]
                print('Overplot for halo', args.halo, 'only every', npoints, 'th data point, i.e. cadence of', npoints * mean_dt, 'Myr')
                #if 'line' in locals(): line.set_alpha(0.7) # make the actual wiggly line fainter

                yfunc = interp1d(df_short[args.xcol], df_short[args.ycol], fill_value='extrapolate') # interpolating the low-cadence data
                cfunc = interp1d(df_short[args.xcol], df_short[args.colorcol], fill_value='extrapolate')
                df[args.ycol + '_interp'] = yfunc(df[args.xcol])
                df[args.colorcol + '_interp'] = cfunc(df[args.xcol])
                ax.plot(df[args.xcol], df[args.ycol + '_interp'], c=col_arr[i], lw=0.5)

        ax.legend(loc='upper right' if j == 1 else 'lower right', fontsize=args.fontsize)
        ax.set_ylabel(thisgroup.label, fontsize=args.fontsize)
        ax.set_ylim(thisgroup.limits)
        ax.tick_params(axis='y', labelsize=args.fontsize)

        ax.set_xlim(xlimits)
        ax.tick_params(axis='x', labelsize=args.fontsize)
        if j == len(groups) - 1: # last panel
            ax.set_xlabel('Time (Gyr)', fontsize=args.fontsize)

    return axes

# -----------------------------------------------------------
def get_box_from_ds(ds, args):
    '''
    Function to extract a small box of given size, based on args.galrad, from the original yt dataset
    Returns yt dataset type
    '''
    # extract the required box
    box_center = ds.halo_center_kpc
    box_width = args.galrad * 2  # in kpc
    box_width_kpc = ds.arr(box_width, 'kpc')
    box = ds.r[box_center[0] - box_width_kpc / 2.: box_center[0] + box_width_kpc / 2.,
          box_center[1] - box_width_kpc / 2.: box_center[1] + box_width_kpc / 2.,
          box_center[2] - box_width_kpc / 2.: box_center[2] + box_width_kpc / 2., ]

    return box

# -----------------------------------------------------------
def plot_profile(df, ax, args, col_arr=['green', 'darkorgange']):
    '''
    Function to plot the metallicity profile, along with the fitted gradient
    '''
    args.ycol = 'log_metal'
    args.ylim = [-2.2, 1.2]  # [-3, 1]
    args.xlim = [0, args.upto_kpc]

    # ------plot the datashader background---------------------
    artist = dsshow(df, dsh.Point(args.xcol, args.ycol), dsh.count(), norm='linear', x_range=(0, args.galrad / args.re if 're' in args.xcol else args.galrad), y_range=(args.ylim[0], args.ylim[1]), aspect = 'auto', ax=ax, cmap='Greys_r')#, shade_hook=partial(dstf.spread, px=1, shape='square')) # the 40 in alpha_range and `square` in shade_hook are to reproduce original-looking plots as if made with make_datashader_plot()

    # --------bin the metallicity profile and plot the binned profile-----------
    args.bin_edges = np.linspace(0, args.galrad / args.re if 're' in args.xcol else args.galrad, 10)
    linefit_binned, ax = fit_binned(df, ax, args, color=col_arr[1])
    linefit_cells, ax = fit_gradient(df, ax, args, color=col_arr[0])

    # ----------tidy up figure-------------
    ax.set_xlabel('Radius (kpc)', fontsize=args.fontsize)
    ax.set_ylabel(r'$\log{(\mathrm{Z/Z}_\odot})}$', fontsize=args.fontsize)

    ax.set_xlim(args.xlim)
    ax.set_ylim(args.ylim)

    ax.locator_params(axis='both', nbins=4)
    ax.tick_params(axis='both', labelsize=args.fontsize)

    # ---------annotate and save the figure----------------------
    ax.text(0.05, 0.2, 'z = %.2F' % args.current_redshift, transform=ax.transAxes, fontsize=args.fontsize/args.fontfactor, color='k', bbox=dict(facecolor='white', alpha=0.6, edgecolor='k'))
    ax.text(0.05, 0.1, 't = %.1F Gyr' % args.current_time, transform=ax.transAxes, fontsize=args.fontsize/args.fontfactor, color='k', bbox=dict(facecolor='white', alpha=0.6, edgecolor='k'))

    return linefit_binned, linefit_cells, ax

# ---------------------------------------------------------------------------------
def fit_binned(df, ax, args, color='darkorange'):
    '''
    Function to overplot binned data and the fit to it on existing plot
    '''
    df['binned_cat'] = pd.cut(df[args.xcol], args.bin_edges)

    agg_func = lambda x: np.sum(x * df.loc[x.index, args.weight]) / np.sum(df.loc[x.index, args.weight])  # function to get weighted mean
    agg_u_func = lambda x: np.sqrt(((np.sum(df.loc[x.index, args.weight] * x ** 2) / np.sum(df.loc[x.index, args.weight])) - (np.sum(x * df.loc[x.index, args.weight]) / np.sum(df.loc[x.index, args.weight])) ** 2) * (np.sum(df.loc[x.index, args.weight] ** 2)) / (np.sum(df.loc[x.index, args.weight]) ** 2 - np.sum(df.loc[x.index, args.weight] ** 2)))  # eq 6 of http://seismo.berkeley.edu/~kirchner/Toolkits/Toolkit_12.pdf

    y_binned = df.groupby('binned_cat', as_index=False).agg([(args.ycol, agg_func)])[args.ycol].values
    y_u_binned = df.groupby('binned_cat', as_index=False).agg([(args.ycol, agg_u_func)])[args.ycol].values

    # ----------to plot mean binned y vs x profile--------------
    x_bin_centers = args.bin_edges[:-1] + np.diff(args.bin_edges) / 2
    linefit, linecov = np.polyfit(x_bin_centers, y_binned.flatten(), 1, cov=True)#, w=1/(y_u_binned.flatten())**2)

    Zgrad = ufloat(linefit[0], np.sqrt(linecov[0][0]))
    Zcen = ufloat(linefit[1], np.sqrt(linecov[1][1]))
    print('Upon radially binning: Inferred slope for halo ' + args.halo + ' output ' + args.output + ' is', Zgrad, 'dex/re' if 're' in args.xcol else 'dex/kpc')

    ax.errorbar(x_bin_centers, y_binned, c=color, yerr=y_u_binned, lw=1, ls='none')
    ax.scatter(x_bin_centers, y_binned, c=color, s=50, lw=1, ec='black')
    ax.plot(x_bin_centers, np.poly1d(linefit)(x_bin_centers), color=color, lw=1, ls='dashed')
    units = 'dex/re' if 're' in args.xcol else 'dex/kpc'
    ax.text(0.95, 0.9, 'Slope = %.2F ' % linefit[0] + units, color=color, transform=ax.transAxes, fontsize=args.fontsize/args.fontfactor, ha='right', va='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='k'))
    return [Zgrad, Zcen], ax

# -------------------------------
def fit_gradient(df, ax, args, color='limegreen'):
    '''
    Function to linearly fit the (log) metallicity profile out to certain Re, given a dataframe containing metallicity profile
    Returns the fitted gradient with uncertainty
    '''
    linefit, linecov = np.polyfit(df[args.xcol], df[args.ycol], 1, cov=True, w=df[args.weight])
    Zgrad = ufloat(linefit[0], np.sqrt(linecov[0][0]))
    Zcen = ufloat(linefit[1], np.sqrt(linecov[1][1]))

    # ----------plot the fitted metallicity profile---------------
    fitted_y = np.poly1d(linefit)(args.bin_edges)
    ax.plot(args.bin_edges, fitted_y, color=color, lw=1, ls='solid')
    units = 'dex/re' if 're' in args.xcol else 'dex/kpc'
    ax.text(0.95, 0.8, 'Slope = %.2F ' % linefit[0] + units, color=color, transform=ax.transAxes, ha='right', va='center', fontsize=args.fontsize/args.fontfactor, bbox=dict(facecolor='white', alpha=0.8, edgecolor='k'))

    print('Inferred slope for halo ' + args.halo + ' output ' + args.output + ' is', Zgrad, 'dex/re' if 're' in args.xcol else 'dex/kpc')

    return [Zgrad, Zcen], ax

# -----------------------------------------------------------
def get_Zarr_from_box(ds, args):
    '''
    Function to extract the array of metallicities in a given box around the center, after binning the box at a given resolution
    Returns 1D array of metallicities
    '''
    # ---------extract the required box-----------------
    box_center = ds.arr(args.halo_center, kpc)
    box_width = args.galrad * 2  # in kpc
    box_width_kpc = ds.arr(box_width, 'kpc')

    # ----------------bin down to res--------------------------
    res = args.res_arr[0]
    ncells = int(box_width / res)
    box = ds.arbitrary_grid(left_edge=[box_center[0] - box_width_kpc / 2., box_center[1] - box_width_kpc / 2., box_center[2] - box_width_kpc / 2.], \
                            right_edge=[box_center[0] + box_width_kpc / 2., box_center[1] + box_width_kpc / 2., box_center[2] + box_width_kpc / 2.], \
                            dims=[ncells, ncells, ncells])
    print('res =', res, 'kpc; box shape=', np.shape(box))  #

    Zres = box['gas', 'metallicity'].in_units('Zsun').ndarray_view()
    wres = box['gas', args.weight].in_units(unit_dict[args.weight]).ndarray_view()

    return Zres.flatten(), wres.flatten()

# -----------------------------------------------------------
def plot_distribution(Zarr, weights, ax, args, col_arr=['green', 'darkorgange']):
    '''
    Function to plot the metallicity distribution, along with the fitted skewed gaussian distribution if provided
    Saves plot as .png
    '''
    args.xlim, args.ylim = [0, 4], [0, 2.5]
    p = plt.hist(Zarr, bins=args.nbins, histtype='step', lw=2, density=True, range=args.xlim, ec=col_arr[1], weights=weights)

    fit, ax = fit_distribution(Zarr, weights, ax, args, color=col_arr[0], range=args.xlim)

    # ----------tidy up figure-------------
    plt.legend(loc='upper right', bbox_to_anchor=(1, 0.75), fontsize=args.fontsize)
    ax.set_xlim(args.xlim)
    ax.set_ylim(args.ylim)

    ax.set_xlabel(r'Z/Z$_{\odot}$', fontsize=args.fontsize)
    ax.set_ylabel('PDF', fontsize=args.fontsize)
    ax.set_xticklabels(['%.1F' % item for item in ax.get_xticks()], fontsize=args.fontsize)
    ax.set_yticklabels(['%.2F' % item for item in ax.get_yticks()], fontsize=args.fontsize)

    # ---------annotate and save the figure----------------------
    ax.text(0.97, 0.9, 'z = %.2F' % args.current_redshift, ha='right', transform=ax.transAxes, fontsize=args.fontsize/args.fontfactor)
    ax.text(0.97, 0.8, 't = %.1F Gyr' % args.current_time, ha='right', transform=ax.transAxes, fontsize=args.fontsize/args.fontfactor)

    return fit, ax

# -------------------------------
def fit_distribution(Zarr, weights, ax, args, color='k', range=(0, 4)):
    '''
    Function to fit the (log) metallicity distribution out to certain Re, given a dataframe containing metallicity
    Returns the fitted parameters for a skewed Gaussian
    '''
    y, x = np.histogram(Zarr, bins=args.nbins, density=True, weights=weights, range=range)
    x = x[:-1] + np.diff(x)/2

    # ----------fitting distribution with multiple components------------
    model = SkewedGaussianModel(prefix='sg_')
    params = model.make_params(sg_amplitude=0.5, sg_center=0.5, sg_sigma=0.5, sg_gamma=0)
    print('Fitting with one skewed gaussian + one regular guassian...')
    g_model = GaussianModel(prefix='g_')
    if args.islog: params.update(g_model.make_params(g_amplitude=2, g_center=-0.9, g_sigma=0.05))
    else: params.update(g_model.make_params(g_amplitude=2, g_center=-0.9, g_sigma=0.05))
    model = model + g_model

    result = model.fit(y, params, x=x)

    # --------overplotting the fit-----------------
    ax.plot(x, result.best_fit, c=color, lw=1)
    ax.plot(x, GaussianModel().eval(x=x, amplitude=result.best_values['g_amplitude'], center=result.best_values['g_center'], sigma=result.best_values['g_sigma']), c=color, lw=1, ls='--')
    ax.plot(x, SkewedGaussianModel().eval(x=x, amplitude=result.best_values['sg_amplitude'], center=result.best_values['sg_center'], sigma=result.best_values['sg_sigma'], gamma=result.best_values['sg_gamma']), c=color, lw=1, ls='dotted')

    ax.axvline(result.best_values['sg_center'], lw=1, ls='dotted', color=color)
    ax.axvline(result.best_values['g_center'], lw=1, ls='dashed', color=color)

    plt.text(0.97, 0.7, r'Mean = %.2F Z$\odot$' % result.best_values['sg_center'], ha='right', transform=ax.transAxes, fontsize=args.fontsize/args.fontfactor)
    plt.text(0.97, 0.6, r'Sigma = %.2F Z$\odot$' % result.best_values['sg_sigma'], ha='right', transform=ax.transAxes, fontsize=args.fontsize/args.fontfactor)

    return result, ax


# -----main code-----------------
if __name__ == '__main__':
    args_tuple = parse_args('8508', 'RD0042')  # default simulation to work upon when comand line args not provided
    if type(args_tuple) is tuple: args, ds, refine_box = args_tuple # if the sim has already been loaded in, in order to compute the box center (via utils.pull_halo_center()), then no need to do it again
    else: args = args_tuple
    if not args.keep: plt.close('all')
    col_arr = ['saddlebrown', 'royalblue'] #['fuchsia', 'darkturquoise']

    # --------domain decomposition; for mpi parallelisation-------------
    if args.do_all_sims: list_of_sims = get_all_sims_for_this_halo(args) # all snapshots of this particular halo
    else: list_of_sims = list(itertools.product([args.halo], args.output_arr))
    total_snaps = len(list_of_sims)

    comm = MPI.COMM_WORLD
    ncores = comm.size
    rank = comm.rank
    print_master('Total number of MPI ranks = ' + str(ncores) + '. Starting at: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()), args)
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

    # -------------loop over snapshots-----------------
    print_mpi('Operating on snapshots ' + str(core_start + 1) + ' to ' + str(core_end + 1) + ', i.e., ' + str(core_end - core_start + 1) + ' out of ' + str(total_snaps) + ' snapshots', args)

    for index in range(core_start + args.start_index, core_end + 1):
        start_time_this_snapshot = time.time()
        this_sim = list_of_sims[index]
        print_mpi('Doing snapshot ' + this_sim[1] + ' of halo ' + this_sim[0] + ' which is ' + str(index + 1 - core_start) + ' out of the total ' + str(core_end - core_start + 1) + ' snapshots...', args)
        halos_df_name = args.code_path + 'halo_infos/00' + this_sim[0] + '/' + args.run + '/'
        halos_df_name += 'halo_cen_smoothed' if args.use_cen_smoothed else 'halo_c_v'

        # -------loading in snapshot-------------------
        try:
            if len(list_of_sims) > 1 or args.do_all_sims: args = parse_args(this_sim[0], this_sim[1])
            if type(args) is tuple: args, ds, refine_box = args  # if the sim has already been loaded in, in order to compute the box center (via utils.pull_halo_center()), then no need to do it again
            else: ds, refine_box = load_sim(args, region='refine_box', do_filter_particles=True, disk_relative=False, halo_c_v_name=halos_df_name)
        except Exception as e:
            print_mpi('Skipping ' + this_sim[1] + ' because ' + str(e), args)
            continue

        args.current_redshift = ds.current_redshift
        args.current_time = ds.current_time.in_units('Gyr').v
        args.fontsize = 15

        # --------loading df for full time evolution-------------
        args.weightby_text = '_wtby_' + args.weight
        args.fitmultiple_text = '_fitmultiple'
        args.density_cut_text = '_wdencut' if args.use_density_cut else ''
        args.islog_text = '_islog' if args.islog else ''
        df = load_df(args)  # loading dataframe (includes both gradinets and distribution measurements

        # -------setting up fig--------------
        nrow, ncol1, ncol2 = 2, 4, 2 #  the overall figure is nrow x (ncol1 + ncol2)
        fig = plt.figure(figsize=(12,6))
        ax_grad_ev = plt.subplot2grid(shape=(nrow, ncol1 + ncol2), loc=(0, 0), colspan=ncol1)
        ax_dist_ev = plt.subplot2grid(shape=(nrow, ncol1 + ncol2), loc=(1, 0), colspan=ncol1, sharex=ax_grad_ev)
        ax_prof_snap = plt.subplot2grid(shape=(nrow, ncol1 + ncol2), loc=(0, ncol1), colspan=ncol2)
        ax_dist_snap = plt.subplot2grid(shape=(nrow, ncol1 + ncol2), loc=(1, ncol1), colspan=ncol2)
        fig.tight_layout()
        fig.subplots_adjust(top=0.98, bottom=0.1, left=0.07, right=0.98, wspace=0.7, hspace=0.25)

        # ------plotting full time evolution---------------
        ax_grad_ev, ax_dist_ev = plot_full_evolution(df, [ax_grad_ev, ax_dist_ev], args, col_arr=col_arr)

        # ------prepping args for individual snapshot analysis--------
        if args.upto_kpc is not None: args.re = np.nan
        else: args.re = args.upto_re
        args.xcol = 'rad'
        args.fontfactor = 1.3 # factor by which to reduce args.fontsize for text _inside_ plots as opposed to axis text, which will still be at args.fontsize

        if args.upto_kpc is not None:
            if args.docomoving: args.galrad = args.upto_kpc / (1 + args.current_redshift) / 0.695  # fit within a fixed comoving kpc h^-1, 0.695 is Hubble constant
            else: args.galrad = args.upto_kpc  # fit within a fixed physical kpc
        else:
            args.galrad = args.re * args.upto_re  # kpc

        # ------plotting individual snapshot: radial profile--------

        box = get_box_from_ds(ds, args)
        df_snap = get_df_from_ds(box, args)
        linefit_binned, linefit_cells, ax_prof_snap = plot_profile(df_snap, ax_prof_snap, args, col_arr=col_arr)

        # ------plotting individual snapshot: histogram--------
        Zarr, weights = get_Zarr_from_box(ds, args)
        result, ax_dist_snap = plot_distribution(Zarr, weights, ax_dist_snap, args, col_arr=col_arr)

        # ------plotting individual snapshots: corresponding vertical lines on time-evolution plot------
        color = 'k'
        ax_grad_ev.axvline(args.current_time, lw=1, c=color)
        ax_dist_ev.axvline(args.current_time, lw=1, c=color)

        # ------saving fig------------------
        if args.upto_kpc is not None: upto_text = '_upto%.1Fckpchinv' % args.upto_kpc if args.docomoving else '_upto%.1Fkpc' % args.upto_kpc
        else: upto_text = '_upto%.1FRe' % args.upto_re

        args.fig_dir = args.output_dir + 'figs/'
        if not args.do_all_sims: args.fig_dir += args.output + '/'
        Path(args.fig_dir).mkdir(parents=True, exist_ok=True)

        outfile_rootname = '%s_%s_allZ_Zgrad_den_%s%s%s.png' % (args.output, args.halo, args.Zgrad_den, upto_text, args.weightby_text)
        if args.do_all_sims: outfile_rootname = 'z=*_' + outfile_rootname[len(args.output) + 1:]
        figname = args.fig_dir + outfile_rootname.replace('*', '%.5F' % (args.current_redshift))

        fig.savefig(figname)
        print('Saved plot as', figname)

        plt.show(block=False)
        print_mpi('This snapshots completed in %s mins' % ((time.time() - start_time_this_snapshot) / 60), args)

    if ncores > 1: print_master('Parallely: time taken for ' + str(total_snaps) + ' snapshots with ' + str(ncores) + ' cores was %s mins' % ((time.time() - start_time) / 60), args)
    else: print_master('Serially: time taken for ' + str(total_snaps) + ' snapshots with ' + str(ncores) + ' core was %s mins' % ((time.time() - start_time) / 60), args)