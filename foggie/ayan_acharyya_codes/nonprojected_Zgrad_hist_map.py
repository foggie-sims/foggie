#!/usr/bin/env python3

"""

    Title :      nonprojected_Zgrad_hist_map
    Notes :      Plot time evolution of unprojected 3D metallicity gradient, distribution and projections ALL in one plot
    Output :     Combined plots as png files plus, optionally, these files stitched into a movie
    Author :     Ayan Acharyya
    Started :    Aug 2023
    Examples :   run nonprojected_Zgrad_hist_map.py --system ayan_pleiades --halo 2392 --upto_kpc 10 --forpaper --do_all_sims
                 run nonprojected_Zgrad_hist_map.py --system ayan_hd --halo 2392 --upto_kpc 10 --forpaper --output DD0417 --vcol vlos
"""
from header import *
from util import *
from datashader_movie import field_dict, unit_dict, get_correct_tablename
from compute_Zscatter import fit_distribution
from compute_MZgrad import get_density_cut, get_re_from_coldgas, get_re_from_stars
from uncertainties import ufloat, unumpy
from lmfit.models import GaussianModel, SkewedGaussianModel
from mpl_toolkits.axes_grid1 import make_axes_locatable

start_time = time.time()
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['axes.edgecolor'] = 'k'

# ----------------------------------------------------------------
def make_df_from_box(box, args):
    '''
    Function to convert a given dataset to pandas dataframe

    :return: modified dataframe
    '''
    myprint('Now making dataframe from box..', args)

    #df_snap_filename = args.output_dir + '/txtfiles/' + args.output + '_df_boxrad_%.2Fkpc_nonprojectedZ.txt' % (args.galrad)
    df_snap_filename = get_correct_tablename(args)

    if not os.path.exists(df_snap_filename) or args.clobber:
        myprint(df_snap_filename + ' does not exist. Creating afresh..', args)
        df = pd.DataFrame()
        fields = ['rad', 'metal'] # only the relevant properties
        if args.weight is not None: fields += [args.weight]

        for index, field in enumerate(fields):
            myprint('Doing property: ' + field + ', which is ' + str(index + 1) + ' of the ' + str(len(fields)) + ' fields..', args)
            df[field] = box[field_dict[field]].in_units(unit_dict[field]).ndarray_view()

        df.to_csv(df_snap_filename, sep='\t', index=None)
    else:
        myprint('Reading from existing file ' + df_snap_filename, args)
        df = pd.read_table(df_snap_filename, delim_whitespace=True, comment='#')

    df['log_metal'] = np.log10(df['metal'])
    df = df.dropna()

    return df

# -----------------------------------------------------------------------
def plot_projection(quantity, box, box_center, box_width, axes, args, unit=None, clim=None, cmap=None, ncells=None):
    '''
    Function to plot the 2D map of various velocity quantities, at the given resolution of the FRB
    :return: axis handle
    '''
    myprint('Now making projection plots..', args)

    weight_field = ('gas', args.weight) if args.weight is not None else None

    for index, projection in enumerate(['x', 'y', 'z']):
        field_dict.update({'vdisp_los': 'velocity_dispersion_' + projection, 'vlos': 'v' + projection + '_corrected'})
        field = field_dict[quantity]
        prj = yt.ProjectionPlot(box.ds, projection, field, center=box_center, data_source=box, width=box_width, weight_field=weight_field, fontsize=args.fontsize, buff_size=(ncells, ncells) if ncells is not None else (800, 800))
        if unit is not None: prj.set_unit(field, unit)
        if cmap is not None: prj.set_cmap(field, cmap)
        if clim is not None: prj.set_zlim(field, zmin=clim[0], zmax=clim[1])

        # ------plotting onto a matplotlib figure--------------
        ax = axes[index]
        position = ax.get_position()
        prj.plots[field].axes = ax
        divider = make_axes_locatable(ax)
        prj._setup_plots()
        prj.plots[field].axes.set_position(position)  # in order to resize the axis back to where it should be

        ax.tick_params(axis='both', labelsize=args.fontsize)
        ax.set_xlabel(ax.get_xlabel(), fontsize=args.fontsize)
        if index == 0: ax.set_ylabel(ax.get_ylabel(), fontsize=args.fontsize)
        else: ax.set_ylabel('')

    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = ax.figure.colorbar(prj.plots[field].cb.mappable, orientation='vertical', cax=cax)
    cbar.ax.tick_params(labelsize=args.fontsize)
    cbar.set_label(prj.plots[field].cax.get_ylabel(), fontsize=args.fontsize / args.fontfactor if quantity != 'metal' else args.fontsize)

    return axes

# ----------------------------------------------------------------
def plot_Zprof_snap(df, ax, args):
    '''
    Function to plot the radial metallicity profile (from input dataframe) as seen from all three projections, on to the given axis
    Also computes the projected metallicity gradient along each projection
    :return: fitted gradient across each projection, and the axis handle
    '''
    plt.style.use('seaborn-whitegrid')
    myprint('Now making the radial profile plot for ' + args.output + '..', args)
    x_bins = np.linspace(0, args.galrad / args.re if 're' in args.xcol else args.galrad, 10)

    weightcol = args.weight
    ycol = 'metal'
    color = 'salmon'

    df['weighted_metal'] = len(df) * df[ycol] * df[args.weight] / np.sum(df[args.weight])
    df['log_' + ycol] = np.log10(df[ycol])
    if not args.plot_onlybinned:
        if len(df) < 1000: ax.scatter(df['rad'], df['log_metal'], c='cornflowerblue', s=5, lw=0, alpha=0.8) # make scatter plots for smaller datasets, otherwise points on plot are too small with datashader
        else: artist = dsshow(df, dsh.Point('rad', 'log_' + ycol), dsh.count(), norm='linear', x_range=(0, args.galrad * np.sqrt(2)), y_range=(args.Zlim[0], args.Zlim[1]), aspect='auto', ax=ax, cmap='Blues_r')

    df['binned_cat'] = pd.cut(df['rad'], x_bins)

    if args.weight is not None:
        agg_func = lambda x: np.sum(x * df.loc[x.index, weightcol]) / np.sum(df.loc[x.index, weightcol]) # function to get weighted mean
        agg_u_func = lambda x: np.sqrt(((np.sum(df.loc[x.index, weightcol] * x**2) / np.sum(df.loc[x.index, weightcol])) - (np.sum(x * df.loc[x.index, weightcol]) / np.sum(df.loc[x.index, weightcol]))**2) * (np.sum(df.loc[x.index, weightcol]**2)) / (np.sum(df.loc[x.index, weightcol])**2 - np.sum(df.loc[x.index, weightcol]**2))) # eq 6 of http://seismo.berkeley.edu/~kirchner/Toolkits/Toolkit_12.pdf
    else:
        agg_func, agg_u_func = np.mean, np.std

    y_binned = df.groupby('binned_cat', as_index=False).agg([(ycol, agg_func)])[ycol].values.flatten()
    y_u_binned = df.groupby('binned_cat', as_index=False).agg([(ycol, agg_u_func)])[ycol].values.flatten()
    x_bin_centers = x_bins[:-1] + np.diff(x_bins) / 2

    quant = unumpy.log10(unumpy.uarray(y_binned, y_u_binned)) # for correct propagation of errors
    y_binned, y_u_binned = unumpy.nominal_values(quant), unumpy.std_devs(quant) # in logspace

    # getting rid of potential nan values
    indices = np.array(np.logical_not(np.logical_or(np.isnan(x_bin_centers), np.isnan(y_binned))))
    x_bin_centers = x_bin_centers[indices]
    y_binned = y_binned[indices]
    y_u_binned = y_u_binned[indices]

    # ----------to plot mean binned y vs x profile--------------
    linefit, linecov = np.polyfit(x_bin_centers, y_binned, 1, cov=True)#, w=1. / (y_u_binned) ** 2) # linear fitting done in logspace
    y_fitted = np.poly1d(linefit)(x_bin_centers) # in logspace

    Zgrad = ufloat(linefit[0], np.sqrt(linecov[0][0]))

    print('Upon radially binning: Inferred slope for halo ' + args.halo + ' output ' + args.output + ' is', Zgrad, 'dex/kpc')

    ax.errorbar(x_bin_centers, y_binned, c=color, yerr=y_u_binned, lw=2, ls='none', zorder=5)
    ax.scatter(x_bin_centers, y_binned, c=color, s=50, lw=1, ec='black', zorder=10)
    ax.plot(x_bin_centers, y_fitted, color=color, lw=2.5, ls='dashed')
    ax.text(0.97, 0.95, 'Slope = %.2F ' % linefit[0] + 'dex/kpc', color=color, transform=ax.transAxes, fontsize=args.fontsize / args.fontfactor, va='top', ha='right')

    ax.set_xlabel('Radius (kpc)', fontsize=args.fontsize / args.fontfactor)
    ax.set_ylabel(r'log Metallicity (Z$_{\odot}$)', fontsize=args.fontsize / args.fontfactor)
    ax.set_xlim(0, args.xmax if args.xmax is not None else np.ceil(args.galrad) if args.forpaper else np.ceil(args.upto_kpc / 0.695)) # kpc
    ax.set_ylim(args.Zlim[0], args.Zlim[1]) # log limits
    ax.set_xticklabels(['%.1F' % item for item in ax.get_xticks()], fontsize=args.fontsize / args.fontfactor)
    ax.set_yticklabels(['%.1F' % item for item in ax.get_yticks()], fontsize=args.fontsize / args.fontfactor)
    #ax.text(0.03, 0.03, args.output, color='k', transform=ax.transAxes, fontsize=args.fontsize / args.fontfactor, va='bottom', ha='left')

    return Zgrad, ax

# ----------------------------------------------------------------
def plot_Zdist_snap(df, ax, args):
    '''
    Function to plot the metallicity histogram (from input dataframe) as seen from all three projections, on to the given axis
    Also fits the histogram of projected metallicity along each projection
    :return: fitted histogram parameters across each projection, and the axis handle
    '''
    myprint('Now making the histogram plot for ' + args.output + '..', args)

    Zarr = df['metal']
    weights = df[args.weight].values if args.weight is not None else None
    color = 'salmon'

    if args.islog: Zarr = np.log10(Zarr)  # all operations will be done in log

    p = ax.hist(Zarr, bins=args.nbins, histtype='step', lw=1, ls='dashed', density=True, ec=color, weights=weights)

    if args.nofit:
        Zdist = np.nan
    else:
        fit, other_result = fit_distribution(Zarr.values, args, weights=weights)
        xvals = p[1][:-1] + np.diff(p[1])
        ax.plot(xvals, fit.eval(x=np.array(xvals)), c=color, lw=1)
        if not args.hide_multiplefit:
            ax.plot(xvals, SkewedGaussianModel().eval(x=xvals, amplitude=fit.best_values['g1_amplitude'], center=fit.best_values['g1_center'], sigma=fit.best_values['g1_sigma'], gamma=fit.best_values['g1_gamma']), c='k', lw=0.5, ls='dotted', label='High-Z component')
            if 'g2_amplitude' in fit.best_values: ax.plot(xvals, SkewedGaussianModel().eval(x=xvals, amplitude=fit.best_values['g2_amplitude'], center=fit.best_values['g2_center'], sigma=fit.best_values['g2_sigma'], gamma=fit.best_values['g2_gamma']), c='k', lw=0.5, ls='--', label='Low-Z component')

        Zdist = [fit.best_values['g1_sigma'], fit.best_values['g1_center']]
        ax.text(0.03 if args.islog else 0.97, 0.75, 'Center = %.2F\nWidth = %.2F' % (fit.best_values['g1_center'], 2.355 * fit.best_values['g1_sigma']), color=color, transform=ax.transAxes, fontsize=args.fontsize / args.fontfactor, va='top', ha='left' if args.islog else 'right')

    ax.set_xlabel(r'log Metallicity (Z$_{\odot}$)' if args.islog else r'Metallicity (Z$_{\odot}$)', fontsize=args.fontsize / args.fontfactor)
    ax.set_ylabel('Normalised distribution', fontsize=args.fontsize / args.fontfactor)
    ax.set_xlim(-2 if args.islog else 0, 1 if args.islog else 3) # Zsun
    ax.set_ylim(0, 3)
    ax.set_xticklabels(['%.1F' % item for item in ax.get_xticks()], fontsize=args.fontsize / args.fontfactor)
    ax.set_yticklabels(['%.1F' % item for item in ax.get_yticks()], fontsize=args.fontsize / args.fontfactor)

    ax.text(0.03 if args.islog else 0.97, 0.95, 'z = %.2F' % args.current_redshift, ha='left' if args.islog else 'right', va='top', transform=ax.transAxes, fontsize=args.fontsize / args.fontfactor)
    ax.text(0.03 if args.islog else 0.97, 0.85, 't = %.1F Gyr' % args.current_time, ha='left' if args.islog else 'right', va='top', transform=ax.transAxes, fontsize=args.fontsize / args.fontfactor)

    return Zdist, ax

field_dict = {'rad':'radius_corrected', 'mass':'mass', 'vrad': 'radial_velocity_corrected', 'vdisp_3d': 'velocity_dispersion_3d', 'vtan': 'tangential_velocity_corrected', \
              'vphi': 'phi_velocity_corrected', 'vtheta': 'theta_velocity_corrected', 'metal':'metallicity'}
label_dict = {'vrad': r'$v_{\rm radial}$', 'vdisp_3d': r'3D $\sigma_v$', 'vdisp_los': r'LoS $\sigma_v$', 'vtan': r'$v_{\rm tangential}$', \
              'vphi': r'$v_{\phi}$', 'vtheta': r'$v_{\theta}$', 'vlos': r'LoS velocity'}

# -----main code-----------------
if __name__ == '__main__':
    args_tuple = parse_args('8508', 'RD0042')  # default simulation to work upon when comand line args not provided
    if type(args_tuple) is tuple: args, ds, refine_box = args_tuple # if the sim has already been loaded in, in order to compute the box center (via utils.pull_halo_center()), then no need to do it again
    else: args = args_tuple
    if not args.keep: plt.close('all')

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
        if len(list_of_sims) > 1 or args.do_all_sims: args = parse_args(this_sim[0], this_sim[1])
        if type(args) is tuple: args, ds, refine_box = args  # if the sim has already been loaded in, in order to compute the box center (via utils.pull_halo_center()), then no need to do it again
        else: ds, refine_box = load_sim(args, region='refine_box', do_filter_particles=True, disk_relative=False, halo_c_v_name=halos_df_name)
        ds.add_field(('gas', 'velocity_dispersion_3d'), function=get_velocity_dispersion_3d, units='km/s', take_log=False, sampling_type='cell')
        ds.add_field(('gas', 'velocity_dispersion_x'), function=get_velocity_dispersion_x, units='km/s', take_log=False, sampling_type='cell')
        ds.add_field(('gas', 'velocity_dispersion_y'), function=get_velocity_dispersion_y, units='km/s', take_log=False, sampling_type='cell')
        ds.add_field(('gas', 'velocity_dispersion_z'), function=get_velocity_dispersion_z, units='km/s', take_log=False, sampling_type='cell')

        # --------assigning additional keyword args-------------
        if args.forpaper:
            args.use_density_cut = True
            args.docomoving = True
            args.islog = True # for the Z distribution panel
            args.weight = 'mass'
            args.fontsize = 15
            args.nbins = 30 if args.nbins == 200 else args.nbins # for the Z distribution panel
            #args.fit_multiple = True # True # for the Z distribution panel
            args.nofit = True #
            args.hide_multiplefit = True

        args.current_redshift = ds.current_redshift
        args.current_time = ds.current_time.in_units('Gyr').tolist()
        args.fontfactor = 1.2
        args.Zlim = [-1.5, 0.5] if args.forproposal else [-2, 1]# log Zsun units

        # --------determining corresponding text suffixes-------------
        args.weightby_text = '_wtby_' + args.weight if args.weight is not None else ''
        args.fitmultiple_text = '_fitmultiple' if args.fit_multiple else ''
        args.density_cut_text = '_wdencut' if args.use_density_cut else ''
        args.islog_text = '_islog' if args.islog else ''
        if args.upto_kpc is not None: args.upto_text = '_upto%.1Fckpchinv' % args.upto_kpc if args.docomoving else '_upto%.1Fkpc' % args.upto_kpc
        else: args.upto_text = '_upto%.1FRe' % args.upto_re

        # ---------to determine filenames, suffixes, etc.----------------
        args.fig_dir = args.output_dir + 'figs/'
        Path(args.fig_dir).mkdir(parents=True, exist_ok=True)
        outfile_rootname = '%s_%s_nonprojectedZ_prof_hist_map_%s%s%s.png' % (args.output, args.halo, args.vcol, args.upto_text, args.weightby_text)
        if args.do_all_sims: outfile_rootname = 'z=*_' + outfile_rootname[len(args.output) + 1:]
        figname = args.fig_dir + outfile_rootname.replace('*', '%.5F' % (args.current_redshift))

        if not os.path.exists(figname) or args.clobber_plot:
            #try:
            # -------setting up fig--------------
            nrow, ncol1, ncol2 = 3, 2, 3
            ncol = ncol1 * ncol2 # the overall figure is nrow x (ncol1 + ncol2)
            fig = plt.figure(figsize=(8, 8))
            axes_met_proj = [plt.subplot2grid(shape=(nrow, ncol), loc=(0, int(item)), colspan=ncol1) for item in np.linspace(0, ncol1 * 2, 3)]
            axes_vel_proj = [plt.subplot2grid(shape=(nrow, ncol), loc=(1, int(item)), colspan=ncol1) for item in np.linspace(0, ncol1 * 2, 3)]
            ax_prof_snap = plt.subplot2grid(shape=(nrow, ncol), loc=(2, 0), colspan=ncol2)
            ax_dist_snap = plt.subplot2grid(shape=(nrow, ncol), loc=(2, ncol2), colspan=ncol - ncol2)
            fig.tight_layout()
            fig.subplots_adjust(top=0.98, bottom=0.07, left=0.1, right=0.87, wspace=2.0, hspace=0.35)

            # ------tailoring the simulation box for individual snapshot analysis--------
            if args.upto_kpc is not None: args.re = np.nan
            else: args.re = get_re_from_coldgas(args) if args.use_gasre else get_re_from_stars(ds, args)

            if args.upto_kpc is not None:
                if args.docomoving: args.galrad = args.upto_kpc / (1 + args.current_redshift) / 0.695  # fit within a fixed comoving kpc h^-1, 0.695 is Hubble constant
                else: args.galrad = args.upto_kpc  # fit within a fixed physical kpc
            else:
                args.galrad = args.re * args.upto_re  # kpc

            # extract the required box
            box_center = ds.halo_center_kpc
            box = ds.sphere(box_center, ds.arr(args.galrad, 'kpc'))
            box_width = 2 * args.galrad * kpc

            if args.use_density_cut:
                rho_cut = get_density_cut(
                    ds.current_time.in_units('Gyr'))  # based on Cassi's CGM-ISM density cut-off
                box = box.cut_region(['obj["gas", "density"] > %.1E' % rho_cut])
                print('Imposing a density criteria to get ISM above density', rho_cut, 'g/cm^3')

            # ------plotting projected metallcity snapshots---------------
            axes_met_proj = plot_projection('metal', box, box_center, box_width, axes_met_proj, args, clim=[10 ** -1.5, 10 ** 0] if args.forproposal else None, cmap=old_metal_color_map)

            # ------plotting projected velocity quantity snapshots---------------
            axes_vel_proj = plot_projection(args.vcol, box, box_center, box_width, axes_vel_proj, args, clim=[-150, 150] if args.vcol == 'vrad' or args.vcol == 'vphi' or args.vcol == 'vlos' else [0, 150] if args.forproposal else None, cmap='PRGn' if args.vcol == 'vrad' or args.vcol == 'vlos' or args.vcol == 'vphi' else 'viridis')

            # ------plotting nonprojected metallicity profiles---------------
            df_snap = make_df_from_box(box,  args)
            Zgrad, ax_prof_snap = plot_Zprof_snap(df_snap, ax_prof_snap, args)

            # ------plotting nonprojected metallicity histograms---------------
            Zdist, ax_dist_snap = plot_Zdist_snap(df_snap, ax_dist_snap, args)

            # ------saving fig------------------
            fig.savefig(figname)
            myprint('Saved plot as ' + figname, args)

            plt.show(block=False)
            print_mpi('This snapshots completed in %s mins' % ((time.time() - start_time_this_snapshot) / 60), args)
            # except Exception as e:
            #     print_mpi('Skipping ' + this_sim[1] + ' because ' + str(e), args)
            #     continue
        else:
            print('Skipping snapshot %s as %s already exists. Use --clobber_plot to remake figure.' %(args.output, figname))
            continue

    if ncores > 1: print_master('Parallely: time taken for ' + str(total_snaps) + ' snapshots with ' + str(ncores) + ' cores was %s mins' % ((time.time() - start_time) / 60), args)
    else: print_master('Serially: time taken for ' + str(total_snaps) + ' snapshots with ' + str(ncores) + ' core was %s mins' % ((time.time() - start_time) / 60), args)