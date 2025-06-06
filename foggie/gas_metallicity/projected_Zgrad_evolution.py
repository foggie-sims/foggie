#!/usr/bin/env python3

"""

    Title :      projected_Zgrad_evolution
    Notes :      Plot time evolution of PROJECTED metallicity gradient, distribution and projections ALL in one plot, for ALL 3 projections
    Output :     Combined plots as png files plus, optionally, these files stitched into a movie
    Author :     Ayan Acharyya
    Started :    Aug 2023
    Examples :   run projected_Zgrad_evolution.py --system ayan_pleiades --halo 8508 --Zgrad_den kpc --upto_kpc 10 --docomoving --res 0.2 --weight mass --zhighlight --overplot_smoothed 1000 --plot_timefraction --Zgrad_allowance 0.05 --upto_z 2 --do_all_sims
                 run projected_Zgrad_evolution.py --system ayan_local --halo 8508 --Zgrad_den kpc --upto_kpc 10 --res 0.2 --forpaper --output RD0030
"""
from header import *
from util import *
from foggie.gas_metallicity.plot_MZgrad import plot_zhighlight
from foggie.gas_metallicity.compute_Zscatter import fit_distribution
from foggie.gas_metallicity.compute_MZgrad import get_re_from_coldgas, get_re_from_stars
from uncertainties import ufloat, unumpy
from lmfit.models import GaussianModel, SkewedGaussianModel

start_time = time.time()

# -----------------------------------------------------
def get_dist_map(args):
    '''
    Function to get a map of distance of each from the center
    '''
    kpc_per_pix = 2 * args.galrad / args.ncells
    center_pix = (args.ncells - 1)/2.
    map_dist = np.array([[np.sqrt((i - center_pix)**2 + (j - center_pix)**2) for j in range(args.ncells)] for i in range(args.ncells)]) * kpc_per_pix # kpc

    return map_dist

# ----------------------------------------------------------------
def make_frb_from_box(box, box_center, box_width, projection, args):
    '''
    Function to convert a given dataset to Fixed Resolution Buffer (FRB) for a given angle of projection and resolution (args.res)
    :return: FRB (2D numpy array)
    '''
    myprint('Now making the FRB for ' + projection + '..', args)
    if args.use_density_cut:
        rho_cut = get_density_cut(args.current_time)  # based on Cassi's CGM-ISM density cut-off
        box = box.cut_region(['obj["gas", "density"] > %.1E' % rho_cut])
        print('Imposing a density criteria to get ISM above density', rho_cut, 'g/cm^3')

    dummy_field = ('gas', 'density')  # dummy field just to create the FRB; any field can be extracted from the FRB thereafter
    dummy_proj = box.ds.proj(dummy_field, projection, center=box_center, data_source=box)
    frb = dummy_proj.to_frb(box.ds.arr(box_width, 'kpc'), args.ncells, center=box_center)

    return frb

# ----------------------------------------------------------------
def make_df_from_frb(frb, df, projection, args):
    '''
    Function to convert a given FRB array to pandas dataframe
    Requires an existing dataframe as input, so that it can add only the columns corresponding to the FRB's projection axis to the dataframe

    :return: modified dataframe
    '''
    myprint('Now making dataframe from FRB for ' + projection + '..', args)
    map_gas_mass = frb['gas', 'mass']
    map_metal_mass = frb['gas', 'metal_mass']
    map_Z = np.array((map_metal_mass / map_gas_mass).in_units('Zsun')) # now in Zsun units
    df['metal_' + projection] = map_Z.flatten()

    if args.weight is not None:
        map_weights = np.array(frb['gas', args.weight])
        weighted_map_Z = len(map_weights) ** 2 * map_Z * map_weights / np.sum(map_weights)
        df['weights_' + projection] = map_weights.flatten()

    return df, weighted_map_Z

# ----------------------------------------------------------------
def plot_projectedZ_snap(map, projection, ax, args, clim=None, cmap='viridis', color='k'):
    '''
    Function to plot a given projected metallicity map on to a given axis
    :return: axis handle
    '''
    plt.style.use('seaborn-white')
    myprint('Now making projection plot for ' + projection + '..', args)
    #sns.set_style('ticks')  # instead of darkgrid, so that there are no grids overlaid on the projections

    delta = 0.3 # the small offset between the actual limits and intended tick labels is to ensure that tick labels do not reach the very edge of the plot
    proj = ax.imshow(map, cmap=cmap, extent=[-args.galrad - delta, args.galrad + delta, -args.galrad - delta, args.galrad + delta], vmin=clim[0] if clim is not None else None, vmax=clim[1] if clim is not None else None)

    # -----------making the axis labels etc--------------
    ax.set_xticks(np.linspace(-int(args.galrad), int(args.galrad), 5))
    ax.set_yticks(np.linspace(-int(args.galrad), int(args.galrad), 5))
    ax.set_xlabel('Offset (kpc)', fontsize=args.fontsize / args.fontfactor)
    ax.set_ylabel('Offset (kpc)', fontsize=args.fontsize / args.fontfactor)
    ax.set_xticklabels(['%.1F' % item for item in ax.get_xticks()], fontsize=args.fontsize / args.fontfactor)
    ax.set_yticklabels(['%.1F' % item for item in ax.get_yticks()], fontsize=args.fontsize / args.fontfactor)

    ax.text(0.9 * args.galrad, 0.9 * args.galrad, projection, ha='right', va='top', c=color, fontsize=args.fontsize * 1.2, weight='bold')#, bbox=dict(facecolor='k', alpha=0.99, edgecolor='k'))

    # ---------making the colorbar axis once, that will correspond to all projections--------------
    if projection == 'x':
        cax_xpos, cax_ypos, cax_width, cax_height = 0.1, 0.93, 0.8, 0.02
        fig = ax.figure
        cax = fig.add_axes([cax_xpos, cax_ypos, cax_width, cax_height])
        plt.colorbar(proj, cax=cax, orientation='horizontal')

        cax.set_xticklabels(['%.1F' % index for index in cax.get_xticks()], fontsize=args.fontsize / args.fontfactor)
        fig.text(cax_xpos + cax_width / 2, cax_ypos + cax_height + 0.005, r'log Metallicity (Z$_{\odot}$)', ha='center', va='bottom', fontsize=args.fontsize)

    return ax

# ----------------------------------------------------------------
def plot_Zprof_snap(df, ax, args, hidex=False, hidey=False):
    '''
    Function to plot the radial metallicity profile (from input dataframe) as seen from all three projections, on to the given axis
    Also computes the projected metallicity gradient along each projection
    :return: fitted gradient across each projection, and the axis handle
    '''
    plt.style.use('seaborn-whitegrid')
    myprint('Now making the radial profile plot for ' + args.output + '..', args)
    x_bins = np.linspace(0, args.galrad / args.re if 're' in args.xcol else args.galrad, 10)
    Zgrad_arr = []
    darker_color_dict = {'salmon':'maroon', 'seagreen':'darkgreen', 'cornflowerblue':'midnightblue'}

    for index,thisproj in enumerate(args.projections):
        weightcol = 'weights_' + thisproj
        ycol = 'metal_' + thisproj
        color = args.col_arr[index]

        if args.weight is not None: df['weighted_' + ycol] = len(df) * df[ycol] * df['weights_' + thisproj] / np.sum(df['weights_' + thisproj])
        df['log_' + ycol] = np.log10(df[ycol])
        if not args.plot_onlybinned: ax.scatter(df['rad'], df['log_' + ycol], c=args.col_arr[index], s=1, lw=0, alpha=0.3)

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
        linefit, linecov = np.polyfit(x_bin_centers, y_binned, 1, cov=True, w=None if args.noweight_forfit else 1. / (y_u_binned) ** 2)  # linear fitting done in logspace
        y_fitted = np.poly1d(linefit)(x_bin_centers) # in logspace

        Zgrad = ufloat(linefit[0], np.sqrt(linecov[0][0]))
        Zgrad_arr.append(Zgrad)

        print('Upon radially binning: Inferred slope for halo ' + args.halo + ' output ' + args.output + ' projection ' + thisproj + ' is', Zgrad, 'dex/kpc')

        ax.errorbar(x_bin_centers, y_binned, c=color, yerr=y_u_binned, lw=2, ls='none', zorder=5)
        ax.scatter(x_bin_centers, y_binned, c=color, s=50, lw=1, ec='black', zorder=10)
        ax.plot(x_bin_centers, y_fitted, color=darker_color_dict[color], lw=2.5, ls='dashed')
        ax.text(0.97, 0.95 - index * 0.1, thisproj + r': Slope = %.2F $\pm$ %.2F ' % (Zgrad.n, Zgrad.s) + 'dex/kpc', color=color, transform=ax.transAxes, fontsize=args.fontsize / args.fontfactor / 1.2, va='top', ha='right')

    ax.set_xlim(0, np.ceil(args.upto_kpc / 0.695)) # kpc
    ax.set_ylim(args.Zlim[0] - 0.1, args.Zlim[1]) # log limits
    if hidex:
        ax.set_xticklabels([])
    else:
        ax.set_xticklabels(['%.1F' % item for item in ax.get_xticks()], fontsize=args.fontsize / args.fontfactor)
        ax.set_xlabel('Radius (kpc)', fontsize=args.fontsize / args.fontfactor)
    if hidey:
        ax.set_yticklabels([])
    else:
        ax.set_yticklabels(['%.1F' % item for item in ax.get_yticks()], fontsize=args.fontsize / args.fontfactor)
        ax.set_ylabel(r'log Metallicity (Z$_{\odot}$)', fontsize=args.fontsize / args.fontfactor)

    if not args.forpaper: ax.text(0.03, 0.03, args.output, color='k', transform=ax.transAxes, fontsize=args.fontsize / args.fontfactor, va='bottom', ha='left')

    return Zgrad_arr, ax

# ----------------------------------------------------------------
def plot_Zdist_snap(df, ax, args, hidex=False, hidey=False):
    '''
    Function to plot the metallicity histogram (from input dataframe) as seen from all three projections, on to the given axis
    Also fits the histogram of projected metallicity along each projection
    :return: fitted histogram parameters across each projection, and the axis handle
    '''
    myprint('Now making the histogram plot for ' + args.output + '..', args)
    Zdist_arr = []

    for index,thisproj in enumerate(args.projections):
        Zarr = df['metal_' + thisproj].values
        weights = df['weights_' + thisproj].values if args.weight is not None else None
        color = args.col_arr[index]

        if args.islog: Zarr = np.log10(Zarr)  # all operations will be done in log

        fit, __ = fit_distribution(Zarr, args, weights=weights)

        p = ax.hist(Zarr, bins=args.nbins, histtype='step', lw=1, ls='dashed', density=True, ec=color, weights=weights)

        xvals = p[1][:-1] + np.diff(p[1])
        #ax.plot(xvals, fit.init_fit, c=color, lw=1, ls='--') # for plotting the initial guess
        ax.plot(xvals, fit.best_fit, c=color, lw=1)
        if not args.hide_multiplefit:
            ax.plot(xvals, SkewedGaussianModel().eval(x=xvals, amplitude=fit.best_values['g1_amplitude'], center=fit.best_values['g1_center'], sigma=fit.best_values['g1_sigma'], gamma=fit.best_values['g1_gamma']), c=color, lw=2, ls='dotted', label='High-Z component')
            if 'g2_amplitude' in fit.best_values: ax.plot(xvals, SkewedGaussianModel().eval(x=xvals, amplitude=fit.best_values['g2_amplitude'], center=fit.best_values['g2_center'], sigma=fit.best_values['g2_sigma'], gamma=fit.best_values['g2_gamma']), c=color, lw=2, ls='--', label='Low-Z component')

        Zdist_arr.append([fit.best_values['g1_sigma'], fit.best_values['g1_center']])
        ax.text(0.03 if args.islog else 0.97, 0.95 - index * 0.21, '%s: Center = %.2F\n%s: Width = %.2F' % (thisproj, fit.best_values['g1_center'], thisproj, 2.355 * fit.best_values['g1_sigma']), color=color, transform=ax.transAxes, fontsize=args.fontsize / args.fontfactor, va='top', ha='left' if args.islog else 'right')

    ax.set_xlim(-2 if args.islog else 0, 1 if args.islog else 3) # Zsun
    ax.set_ylim(0, 3)
    if hidex:
        ax.set_xticklabels([])
    else:
        ax.set_xticklabels(['%.1F' % item for item in ax.get_xticks()], fontsize=args.fontsize / args.fontfactor)
        ax.set_xlabel(r'log Metallicity (Z$_{\odot}$)' if args.islog else r'Metallicity (Z$_{\odot}$)', fontsize=args.fontsize / args.fontfactor)
    if hidey:
        ax.set_yticklabels([])
    else:
        ax.set_yticklabels(['%.1F' % item for item in ax.get_yticks()], fontsize=args.fontsize / args.fontfactor)
        ax.set_ylabel('Normalised distribution', fontsize=args.fontsize / args.fontfactor)

    ax.text(0.03 if args.islog else 0.97, 0.3, 'z = %.2F' % args.current_redshift, ha='left' if args.islog else 'right', va='top', transform=ax.transAxes, fontsize=args.fontsize / args.fontfactor)
    ax.text(0.03 if args.islog else 0.97, 0.2, 't = %.1F Gyr' % args.current_time, ha='left' if args.islog else 'right', va='top', transform=ax.transAxes, fontsize=args.fontsize / args.fontfactor)

    return Zdist_arr, ax

# ----------------------------------------------------------------
def plot_Zgrad_evolution(df, ax, args):
    '''
    Function to plot the full time evolution of projected metallicity gradient (from input dataframe) as seen from all three projections, on to the given axis
    :return: axis handle
    '''
    myprint('Now making the time evolution plot for Z gradient fits..', args)
    xcol = 'time'
    for index,thisproj in enumerate(args.projections):
        ax.plot(df[xcol], df['Zgrad_' + thisproj], c=args.col_arr[index], lw=1, ls='solid')

    ax.axvline(args.current_time, lw=1, ls='--', c='k')
    ax.set_xlim(0, 14) # Gyr
    ax.set_xticks(ax.get_xticks())
    ax.get_xaxis().set_visible(False)

    ax.set_ylabel(r'$\nabla Z$ (dex/kpc)', fontsize=args.fontsize)
    ax.set_ylim(-0.5, 0.2) # Zsun
    ax.set_yticklabels(['%.1F' % item for item in ax.get_yticks()], fontsize=args.fontsize)

    return ax

# ----------------------------------------------------------------
def plot_Zdist_evolution(df, ax, args):
    '''
    Function to plot the full time evolution of projected metallicity histogram (from input dataframe) as seen from all three projections, on to the given axis
    :return: axis handle
    '''
    myprint('Now making the time evolution plot for histogram fits..', args)
    xcol = 'time'
    for index,thisproj in enumerate(args.projections):
        ax.plot(df[xcol], df['Zpeak_' + thisproj], c=args.col_arr[index], lw=1, ls='solid', label=None if index else 'Peak')
        ax.plot(df[xcol], df['Zwidth_' + thisproj], c=args.col_arr[index], lw=1, ls='dashed', label=None if index else 'Width')

    ax.legend(fontsize=args.fontsize / args.fontfactor)
    ax.axvline(args.current_time, lw=1, ls='--', c='k')
    ax.set_xlabel('Time (Gyr)', fontsize=args.fontsize)
    ax.set_ylabel(r'Z/Z$_{\odot}$', fontsize=args.fontsize)

    ax.set_xlim(0, 14) # Gyr
    ax.set_ylim(1e-4, 2) # Zsun

    ax.set_xticklabels(['%.1F' % item for item in ax.get_xticks()], fontsize=args.fontsize)
    ax.set_yticklabels(['%.1F' % item for item in ax.get_yticks()], fontsize=args.fontsize)

    return ax

# -----main code-----------------
if __name__ == '__main__':
    args_tuple = parse_args('8508', 'RD0042')  # default simulation to work upon when comand line args not provided
    if type(args_tuple) is tuple: args, ds, refine_box = args_tuple # if the sim has already been loaded in, in order to compute the box center (via utils.pull_halo_center()), then no need to do it again
    else: args = args_tuple
    if not args.keep: plt.close('all')

    # --------make new dataframe to store all results-----------------
    columns = ['output', 'redshift', 'time', 'Zgrad_x', 'Zgrad_x_u', 'Zgrad_y', 'Zgrad_y_u', 'Zgrad_z', 'Zgrad_z_u', 'Zwidth_x', 'Zpeak_x', 'Zwidth_y', 'Zpeak_y', 'Zwidth_z', 'Zpeak_z']
    df_full = pd.DataFrame(columns=columns)
    outfilename = args.output_dir + '/txtfiles/' + args.halo + '_projectedZ_evolution.txt'
    if not os.path.exists(outfilename) or args.clobber: df_full.to_csv(outfilename, sep='\t', index=None) # writing to file, so that invidual processors can read in and append

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

        # -------loading in snapshot-------------------
        halos_df_name = args.code_path + 'halo_infos/00' + this_sim[0] + '/' + args.run + '/'
        halos_df_name += 'halo_cen_smoothed' if args.use_cen_smoothed else 'halo_c_v'
        if len(list_of_sims) > 1 or args.do_all_sims: args = parse_args(this_sim[0], this_sim[1])
        if type(args) is tuple: args, ds, refine_box = args  # if the sim has already been loaded in, in order to compute the box center (via utils.pull_halo_center()), then no need to do it again
        else: ds, refine_box = load_sim(args, region='refine_box', do_filter_particles=True, disk_relative=False, halo_c_v_name=halos_df_name)

        # --------assigning additional keyword args-------------
        if args.forpaper:
            args.use_density_cut = True
            args.docomoving = True
            args.fit_multiple = True # True # for the Z distribution panel
            args.islog = True # for the Z distribution panel
            args.nbins = 100 # for the Z distribution panel
            args.weight = 'mass'

        args.current_redshift = ds.current_redshift
        args.current_time = ds.current_time.in_units('Gyr').tolist()

        args.projections = ['x', 'y', 'z']
        args.col_arr = ['salmon', 'seagreen', 'cornflowerblue']  # colors corresponding to different projections
        args.Zlim = [-2, 2]  # log Zsun units
        args.res = args.res_arr[0]
        if args.docomoving: args.res = args.res / (1 + args.current_redshift) / 0.695  # converting from comoving kcp h^-1 to physical kpc
        args.fontsize = 15
        args.fontfactor = 1.5

        # --------determining corresponding text suffixes and figname-------------
        args.weightby_text = '_wtby_' + args.weight
        args.fitmultiple_text = '_fitmultiple' if args.fit_multiple else ''
        args.density_cut_text = '_wdencut' if args.use_density_cut else ''
        args.islog_text = '_islog' if args.islog else ''
        if args.upto_kpc is not None:
            upto_text = '_upto%.1Fckpchinv' % args.upto_kpc if args.docomoving else '_upto%.1Fkpc' % args.upto_kpc
        else:
            upto_text = '_upto%.1FRe' % args.upto_re

        args.fig_dir = args.output_dir + 'figs/'
        if not args.do_all_sims: args.fig_dir += args.output + '/'
        Path(args.fig_dir).mkdir(parents=True, exist_ok=True)

        outfile_rootname = '%s_%s_projectedZ_Zgrad_den_%s%s%s.png' % (
        args.output, args.halo, args.Zgrad_den, upto_text, args.weightby_text)
        if args.do_all_sims: outfile_rootname = 'z=*_' + outfile_rootname[len(args.output) + 1:]
        figname = args.fig_dir + outfile_rootname.replace('*', '%.5F' % (args.current_redshift))

        if not os.path.exists(figname) or args.clobber_plot:
            try:
                # -------setting up fig--------------
                nrow, ncol1, ncol2 = 4, 2, 3
                ncol = ncol1 * ncol2 # the overall figure is nrow x (ncol1 + ncol2)
                fig = plt.figure(figsize=(8, 8))
                axes_proj_snap = [plt.subplot2grid(shape=(nrow, ncol), loc=(0, int(item)), colspan=ncol1) for item in np.linspace(0, ncol1 * 2, 3)]
                ax_prof_snap = plt.subplot2grid(shape=(nrow, ncol), loc=(1, 0), colspan=ncol2)
                ax_dist_snap = plt.subplot2grid(shape=(nrow, ncol), loc=(1, ncol2), colspan=ncol - ncol2)
                ax_grad_ev = plt.subplot2grid(shape=(nrow, ncol), loc=(2, 0), colspan=ncol)
                ax_dist_ev = plt.subplot2grid(shape=(nrow, ncol), loc=(3, 0), colspan=ncol, sharex=ax_grad_ev)
                fig.tight_layout()
                fig.subplots_adjust(top=0.9, bottom=0.07, left=0.1, right=0.95, wspace=0.8, hspace=0.35)

                # ------tailoring the simulation box for individual snapshot analysis--------
                if args.upto_kpc is not None: args.re = np.nan
                else: args.re = get_re_from_coldgas(args) if args.use_gasre else get_re_from_stars(ds, args)


                if args.upto_kpc is not None:
                    if args.docomoving: args.galrad = args.upto_kpc / (1 + args.current_redshift) / 0.695  # fit within a fixed comoving kpc h^-1, 0.695 is Hubble constant
                    else: args.galrad = args.upto_kpc  # fit within a fixed physical kpc
                else:
                    args.galrad = args.re * args.upto_re  # kpc
                args.ncells = int(2 * args.galrad / args.res)

                # extract the required box
                box_center = ds.halo_center_kpc
                box = ds.sphere(box_center, ds.arr(args.galrad, 'kpc'))

                # ------plotting projected metallcity snapshots---------------
                df_snap_filename = args.output_dir + '/txtfiles/' + args.output + '_df_boxrad_%.2Fkpc_projectedZ.txt'%(args.galrad)

                if not os.path.exists(df_snap_filename) or args.clobber:
                    myprint(df_snap_filename + 'not found, creating afresh..', args)
                    df_snap = pd.DataFrame()
                    map_dist = get_dist_map(args)
                    df_snap['rad'] = map_dist.flatten()

                    for index, thisproj in enumerate(args.projections):
                        frb = make_frb_from_box(box, box_center, 2 * args.galrad, thisproj, args)
                        df_snap, weighted_map_Z = make_df_from_frb(frb, df_snap, thisproj,  args)
                        axes_proj_snap[index] = plot_projectedZ_snap(np.log10(weighted_map_Z), thisproj, axes_proj_snap[index], args, clim=args.Zlim, cmap=old_metal_color_map, color=args.col_arr[index])

                    df_snap.to_csv(df_snap_filename, sep='\t', index=None)
                    myprint('Saved file ' + df_snap_filename, args)
                else:
                    myprint('Reading in existing ' + df_snap_filename, args)
                    df_snap = pd.read_table(df_snap_filename, delim_whitespace=True, comment='#')
                    for index, thisproj in enumerate(args.projections):
                        weighted_Z = len(df_snap) * df_snap['metal_' + thisproj] * df_snap['weights_' + thisproj] / np.sum(df_snap['weights_' + thisproj])
                        weighted_map_Z = weighted_Z.values.reshape((args.ncells, args.ncells))
                        axes_proj_snap[index] = plot_projectedZ_snap(np.log10(weighted_map_Z), thisproj, axes_proj_snap[index], args, clim=args.Zlim, cmap=old_metal_color_map, color=args.col_arr[index])

                df_snap = df_snap.dropna()
                # ------plotting projected metallicity profiles---------------
                Zgrad_arr, ax_prof_snap = plot_Zprof_snap(df_snap, ax_prof_snap, args)

                # ------plotting projected metallicity histograms---------------
                Zdist_arr, ax_dist_snap = plot_Zdist_snap(df_snap, ax_dist_snap, args)

                # ------update full dataframe and read it from file-----------
                df_full_row = np.hstack(([args.output, args.current_redshift, args.current_time], np.hstack([[Zgrad_arr[i].n, Zgrad_arr[i].s] for i in range(len(args.projections))]), np.hstack([[Zdist_arr[i][0], Zdist_arr[i][1]] for i in range(len(args.projections))])))
                df_full.loc[0] = df_full_row
                df_full.to_csv(outfilename, mode='a', sep='\t', header=False, index=None)
                df_full = pd.read_table(outfilename, delim_whitespace=True)
                df_full = df_full.drop_duplicates(subset='output', keep='last')
                df_full = df_full.sort_values(by='time')

                # ------plotting full time evolution of projected metallicity gradient---------------
                ax_grad_ev = plot_Zgrad_evolution(df_full, ax_grad_ev, args)

                # ------plotting full time evolution of projected metallicity distribution---------------
                ax_dist_ev = plot_Zdist_evolution(df_full, ax_dist_ev, args)

                # ------saving fig------------------
                fig.savefig(figname)
                myprint('Saved plot as ' + figname, args)

                plt.show(block=False)
                print_mpi('This snapshots completed in %s mins' % ((time.time() - start_time_this_snapshot) / 60), args)

            except Exception as e:
                print_mpi('Skipping ' + this_sim[1] + ' because ' + str(e), args)
                continue
        else:
            print('Skipping snapshot %s as %s already exists. Use --clobber_plot to remake figure.' %(args.output, figname))
            continue

    if ncores > 1: print_master('Parallely: time taken for ' + str(total_snaps) + ' snapshots with ' + str(ncores) + ' cores was %s mins' % ((time.time() - start_time) / 60), args)
    else: print_master('Serially: time taken for ' + str(total_snaps) + ' snapshots with ' + str(ncores) + ' core was %s mins' % ((time.time() - start_time) / 60), args)