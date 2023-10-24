#!/usr/bin/env python3

"""

    Title :      projected_Zgrad_hist_map
    Notes :      Plot PROJECTED metallicity gradient, distribution and projections ALL in one plot, for a given line of sight
    Output :     Combined plots as png files plus
    Author :     Ayan Acharyya
    Started :    Oct 2023
    Examples :   run projected_Zgrad_hist_map.py --system ayan_pleiades --halo 8508 --Zgrad_den kpc --upto_kpc 10 --docomoving --res_arc 0.1 --weight mass --output RD0030
                 run projected_Zgrad_hist_map.py --system ayan_local --halo 8508 --Zgrad_den kpc --upto_kpc 10 --res_arc 0.1 --forpaper --output RD0030
"""
from header import *
from util import *
from plot_MZgrad import plot_zhighlight
from compute_Zscatter import fit_distribution
from uncertainties import ufloat, unumpy
from lmfit.models import GaussianModel, SkewedGaussianModel
from mpl_toolkits.axes_grid1 import make_axes_locatable

start_time = time.time()
plt.rcParams['axes.linewidth'] = 1

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
    dummy_field = ('gas', 'density')  # dummy field just to create the FRB; any field can be extracted from the FRB thereafter
    dummy_proj = box.ds.proj(dummy_field, projection, center=box_center, data_source=box)
    frb = dummy_proj.to_frb(box.ds.arr(box_width, 'kpc'), args.ncells, center=box_center)

    return frb

# -----------------------------------------------------------------
def make_its_own_figure(ax, label, args):
    '''
    Function to take an already filled axis handle and turn it into its stand-alone figure
    Output: saved png
    '''
    import pickle
    import io
    buf = io.BytesIO()
    pickle.dump(ax.figure, buf)
    buf.seek(0)
    fig = pickle.load(buf)

    outfile_rootname = '%s_%s_%s%s%s%s%s.png' % (label, args.output, args.halo, args.Zgrad_den, args.upto_text, args.weightby_text, args.res_text)
    if args.do_all_sims: outfile_rootname = 'z=*_' + outfile_rootname[len(args.output) + 1:]
    figname = args.fig_dir + outfile_rootname.replace('*', '%.5F' % (args.current_redshift))
    fig.savefig(figname)
    myprint('Saved plot as ' + figname, args)

    return

# --------------------------------------------------------------------------------------------------------------
def get_los_velocity_dispersion(field, data, args):
    '''
    Function to compute gas velocity dispersion, given a YT box object
    This function is based on Cassi's script vdisp_vs_mass_res() in foggie/turbulence/turbulence.py
    '''
    pix_res = float(np.min(data['dx'].in_units('kpc')))  # at level 11
    cooling_level = int(re.search('nref(.*)c', args.run).group(1))
    string_to_skip = '%dc' % cooling_level
    forced_level = int(re.search('nref(.*)f', args.run[args.run.find(string_to_skip) + len(string_to_skip):]).group(1))
    lvl1_res = pix_res * 2. ** cooling_level
    level = forced_level
    dx = lvl1_res / (2. ** level)
    smooth_scale = int(25. / dx) / 6.
    myprint('Smoothing velocity field at %.2f kpc to compute velocity dispersion..'%smooth_scale, args)

    vlos = data['v' + args.projection + '_corrected'].in_units('km/s').v
    smooth_vlos = gaussian_filter(vlos, smooth_scale)
    vdisp_los = vlos - smooth_vlos

    return vdisp_los

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

    map_vdisp_los = frb['gas', 'velocity_dispersion_' + projection] # cm*km/s
    map_vdisp_los = (map_vdisp_los / (2 * YTArray(args.galrad, 'kpc'))).in_units('km/s') # km/s
    df['vdisp_' + projection] = map_vdisp_los.flatten()

    if args.weight is not None:
        map_weights = np.array(frb['gas', args.weight])
        weighted_map_Z = len(map_weights) ** 2 * map_Z * map_weights / np.sum(map_weights)
        df['weights_' + projection] = map_weights.flatten()

    return df, weighted_map_Z

# -----------------------------------------------------------------------
def plot_projected_vdisp(map, ax, args, clim=None, cmap='viridis'):
    '''
    Function to plot the 2D LoS velocity dispersion map, at the given resolution of the FRB
    :return: axis handle
    '''
    myprint('Now making projected vel disp plot..', args)
    plt.style.use('seaborn-white')

    proj = ax.imshow(map, cmap=cmap, extent=[-args.galrad, args.galrad, -args.galrad, args.galrad], vmin=clim[0] if clim is not None else None, vmax=clim[1] if clim is not None else None)

    # -----------making the axis labels etc--------------
    ax.set_xticks(np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 5))
    ax.set_yticks(np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 5))
    ax.set_xlabel('Offset (kpc)', fontsize=args.fontsize / args.fontfactor)
    ax.set_ylabel('Offset (kpc)', fontsize=args.fontsize / args.fontfactor)
    ax.set_xticklabels(['%.1F' % item for item in ax.get_xticks()], fontsize=args.fontsize / args.fontfactor)
    ax.set_yticklabels(['%.1F' % item for item in ax.get_yticks()], fontsize=args.fontsize / args.fontfactor)

    # ---------making the colorbar axis once, that will correspond to all projections--------------
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(proj, cax=cax, orientation='vertical')

    cax.set_xticklabels(['%.1F' % index for index in cax.get_xticks()], fontsize=args.fontsize / args.fontfactor)
    cax.set_ylabel(r'LoS $\sigma_v$ (km/s)', fontsize=args.fontsize / args.fontfactor)

    return ax

# ----------------------------------------------------------------
def plot_projectedZ_snap(map, ax, args, clim=None, cmap='viridis'):
    '''
    Function to plot a given projected metallicity map on to a given axis
    :return: axis handle
    '''
    myprint('Now making projection plot..', args)
    plt.style.use('seaborn-white')
    #sns.set_style('ticks')  # instead of darkgrid, so that there are no grids overlaid on the projections

    proj = ax.imshow(map, cmap=cmap, extent=[-args.galrad, args.galrad, -args.galrad, args.galrad], vmin=clim[0] if clim is not None else None, vmax=clim[1] if clim is not None else None)

    # -----------making the axis labels etc--------------
    ax.set_xticks(np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 5))
    ax.set_yticks(np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 5))
    ax.set_xlabel('Offset (kpc)', fontsize=args.fontsize / args.fontfactor)
    ax.set_ylabel('Offset (kpc)', fontsize=args.fontsize / args.fontfactor)
    ax.set_xticklabels(['%.1F' % item for item in ax.get_xticks()], fontsize=args.fontsize / args.fontfactor)
    ax.set_yticklabels(['%.1F' % item for item in ax.get_yticks()], fontsize=args.fontsize / args.fontfactor)

    # ---------making the colorbar axis once, that will correspond to all projections--------------
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(proj, cax=cax, orientation='vertical')

    cax.set_xticklabels(['%.1F' % index for index in cax.get_xticks()], fontsize=args.fontsize / args.fontfactor)
    cax.set_ylabel(r'log Metallicity (Z$_{\odot}$)', fontsize=args.fontsize / args.fontfactor)

    #make_its_own_figure(ax, 'projectedZ', args) #to save a copy of figure as its own separate fig too

    return ax

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
    Zgrad_arr = []

    weightcol = 'weights'
    ycol = 'metal'
    color = 'blue'

    df['weighted_metal'] = len(df) * df[ycol] * df[weightcol] / np.sum(df[weightcol])
    df['log_metal'] = np.log10(df[ycol])
    if not args.plot_onlybinned: ax.scatter(df['rad'], df['log_metal'], c='cornflowerblue', s=5, lw=0, alpha=0.7)

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
    linefit, linecov = np.polyfit(x_bin_centers, y_binned, 1, cov=True)#, w=1. / (y_u_binned) ** 2)  # linear fitting done in logspace
    y_fitted = np.poly1d(linefit)(x_bin_centers) # in logspace

    Zgrad = ufloat(linefit[0], np.sqrt(linecov[0][0]))
    Zgrad_arr.append(Zgrad)

    print('Upon radially binning: Inferred slope for halo ' + args.halo + ' output ' + args.output + ' projection ' + thisproj + ' is', Zgrad, 'dex/kpc')

    ax.errorbar(x_bin_centers, y_binned, c=color, yerr=y_u_binned, lw=2, ls='none', zorder=5)
    ax.scatter(x_bin_centers, y_binned, c=color, s=30, lw=1, ec='black', zorder=10)
    ax.plot(x_bin_centers, y_fitted, color=color, lw=2.5, ls='dashed')

    ax.set_xlabel('Radius (kpc)', fontsize=args.fontsize / args.fontfactor)
    ax.set_ylabel(r'log Metallicity (Z$_{\odot}$)', fontsize=args.fontsize / args.fontfactor)
    ax.set_xlim(0, np.ceil(args.upto_kpc / 0.695)) # kpc
    ax.set_ylim(args.Zlim[0], args.Zlim[1]) # log limits
    ax.set_xticklabels(['%.1F' % item for item in ax.get_xticks()], fontsize=args.fontsize / args.fontfactor)
    ax.set_yticklabels(['%.1F' % item for item in ax.get_yticks()], fontsize=args.fontsize / args.fontfactor)
    ax.text(ax.get_xlim()[1]*0.9, ax.get_ylim()[1]*0.9, 'Slope = %.2F ' % linefit[0] + 'dex/kpc', color='k', transform=ax.transAxes, fontsize=args.fontsize / args.fontfactor, va='top', ha='right')

    return Zgrad_arr, ax

# ----------------------------------------------------------------
def plot_Zdist_snap(df, ax, args):
    '''
    Function to plot the metallicity histogram (from input dataframe) as seen from all three projections, on to the given axis
    Also fits the histogram of projected metallicity along each projection
    :return: fitted histogram parameters across each projection, and the axis handle
    '''
    myprint('Now making the histogram plot for ' + args.output + '..', args)
    Zdist_arr = []

    Zarr = df['metal'].values
    weights = df['weights'].values if args.weight is not None else None
    color = 'blue'

    if args.islog: Zarr = np.log10(Zarr)  # all operations will be done in log

    fit = fit_distribution(Zarr, args, weights=weights)

    p = ax.hist(Zarr, bins=args.nbins, histtype='step', lw=1, density=True, ec='cornflowerblue', weights=weights)

    xvals = p[1][:-1] + np.diff(p[1])
    #ax.plot(xvals, fit.init_fit, c=color, lw=1, ls='--') # for plotting the initial guess
    ax.plot(xvals, fit.best_fit, c=color, lw=1)
    if not args.hide_multiplefit:
        ax.plot(xvals, GaussianModel().eval(x=xvals, amplitude=fit.best_values['g_amplitude'], center=fit.best_values['g_center'], sigma=fit.best_values['g_sigma']), c=color, lw=1, ls='--', label='Regular Gaussian')
        ax.plot(xvals, SkewedGaussianModel().eval(x=xvals, amplitude=fit.best_values['sg_amplitude'], center=fit.best_values['sg_center'], sigma=fit.best_values['sg_sigma'], gamma=fit.best_values['sg_gamma']), c=color, lw=1, ls='dotted', label='Skewed Gaussian')

    Zdist_arr.append([fit.best_values['sg_sigma'], fit.best_values['sg_center']])

    ax.set_xlabel(r'log Metallicity (Z$_{\odot}$)' if args.islog else r'Metallicity (Z$_{\odot}$)', fontsize=args.fontsize / args.fontfactor)
    ax.set_ylabel('Normalised distribution', fontsize=args.fontsize / args.fontfactor)
    ax.set_xlim(args.Zlim[0], args.Zlim[1]) # Zsun
    ax.set_ylim(0, 3)
    ax.set_xticklabels(['%.1F' % item for item in ax.get_xticks()], fontsize=args.fontsize / args.fontfactor)
    ax.set_yticklabels(['%.1F' % item for item in ax.get_yticks()], fontsize=args.fontsize / args.fontfactor)
    ax.text(ax.get_xlim()[1]*0.9, ax.get_ylim()[1]*0.9, 'Center = %.2F\nWidth = %.2F' % (fit.best_values['sg_center'], 2.355 * fit.best_values['sg_sigma']), color='k', transform=ax.transAxes, fontsize=args.fontsize / args.fontfactor, va='top', ha='left' if args.islog else 'right')

    #ax.text(0.03 if args.islog else 0.97, 0.3, 'z = %.2F' % args.current_redshift, ha='left' if args.islog else 'right', va='top', transform=ax.transAxes, fontsize=args.fontsize / args.fontfactor)
    #ax.text(0.03 if args.islog else 0.97, 0.2, 't = %.1F Gyr' % args.current_time, ha='left' if args.islog else 'right', va='top', transform=ax.transAxes, fontsize=args.fontsize / args.fontfactor)

    return Zdist_arr, ax

# -----main code-----------------
if __name__ == '__main__':
    args_tuple = parse_args('8508', 'RD0042')  # default simulation to work upon when comand line args not provided
    if type(args_tuple) is tuple: args, ds, refine_box = args_tuple # if the sim has already been loaded in, in order to compute the box center (via utils.pull_halo_center()), then no need to do it again
    else: args = args_tuple
    if not args.keep: plt.close('all')

    # ---------to determine filenames, suffixes, etc.----------------
    args.fig_dir = args.output_dir + 'figs/'
    if not args.do_all_sims: args.fig_dir += args.output + '/'
    Path(args.fig_dir).mkdir(parents=True, exist_ok=True)

    if args.upto_kpc is not None: args.upto_text = '_upto%.1Fckpchinv' % args.upto_kpc if args.docomoving else '_upto%.1Fkpc' % args.upto_kpc
    else: args.upto_text = '_upto%.1FRe' % args.upto_re

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
            args.use_density_cut = False
            args.docomoving = True
            args.fit_multiple = True # True # for the Z distribution panel
            args.islog = True # for the Z distribution panel
            args.nbins = 30 # for the Z distribution panel
            args.weight = 'mass'
            args.hide_multiplefit = True

        args.current_redshift = ds.current_redshift
        args.current_time = ds.current_time.in_units('Gyr').tolist()

        args.projections = [item for item in args.projection.split(',')]
        args.Zlim = [-2, 1] # log Zsun units
        if args.res_arc is not None:
            args.res = get_kpc_from_arc_at_redshift(float(args.res_arc), args.current_redshift)
            native_res_at_z = 0.27 / (1 + args.current_redshift) # converting from comoving kpc to physical kpc
            if args.res < native_res_at_z:
                print('Computed resolution %.2f kpc is below native FOGGIE res at z=%.2f, so we set resolution to the native res = %.2f kpc.'%(args.res, args.current_redshift, native_res_at_z))
                args.res = native_res_at_z # kpc
        else:
            args.res = args.res_arr[0]
            if args.docomoving: args.res = args.res / (1 + args.current_redshift) / 0.695 # converting from comoving kcp h^-1 to physical kpc
        args.res_text = '_res%.1fkpc' % float(args.res)
        args.fontsize = 15
        args.fontfactor = 1.5

        # --------determining corresponding text suffixes-------------
        args.weightby_text = '_wtby_' + args.weight
        args.fitmultiple_text = '_fitmultiple' if args.fit_multiple else ''
        args.density_cut_text = '_wdencut' if args.use_density_cut else ''
        args.islog_text = '_islog' if args.islog else ''

        # -------setting up fig--------------
        nrow, ncol = len(args.projections), 4
        fig, axes = plt.subplots(nrow, ncol, figsize=(15, 8))
        fig.tight_layout()
        fig.subplots_adjust(top=0.95, bottom=0.07, left=0.05, right=0.93, wspace=0.3, hspace=0.2)

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

        # -----------reading in the snapshot's projected metallicity dataframe----------
        df_snap_filename = args.output_dir + 'txtfiles/' + args.output + '_df_boxrad_%.2Fkpc_projectedZ%s.txt' % (args.galrad, args.res_text)

        if not os.path.exists(df_snap_filename) or args.clobber:
            myprint(df_snap_filename + 'not found, creating afresh..', args)
            df_snap = pd.DataFrame()
            map_dist = get_dist_map(args)
            df_snap['rad'] = map_dist.flatten()

            for thisproj in ['x', 'y', 'z']:
                frb = make_frb_from_box(box, box_center, 2 * args.galrad, thisproj, args)
                df_snap, weighted_map_Z = make_df_from_frb(frb, df_snap, thisproj, args)

            df_snap.to_csv(df_snap_filename, sep='\t', index=None)
            myprint('Saved file ' + df_snap_filename, args)
        else:
            myprint('Reading in existing ' + df_snap_filename, args)
            df_snap = pd.read_table(df_snap_filename, delim_whitespace=True, comment='#')

        # -------loop over lines of sight---------------------
        for index, thisproj in enumerate(args.projections):
            args.projection = thisproj
            ax_row = axes if len(args.projections) == 1 else axes[index]
            df_thisproj = df_snap[['rad', 'metal_' + thisproj, 'weights_' + thisproj, 'vdisp_' + thisproj]]
            df_thisproj.rename(columns={'metal_' + thisproj: 'metal', 'weights_' + thisproj: 'weights', 'vdisp_' + thisproj: 'vdisp_los'}, inplace=True)

            # ------plotting projected metallcity snapshots---------------
            vdisp_map = df_thisproj['vdisp_los'].values.reshape((args.ncells, args.ncells))
            ax_row[3] = plot_projected_vdisp(vdisp_map, ax_row[3], args, clim=[0, 200])

            # ------plotting projected metallcity snapshots---------------
            weighted_Z = len(df_thisproj) * df_thisproj['metal'] * df_thisproj['weights'] / np.sum(df_thisproj['weights'])
            weighted_map_Z = weighted_Z.values.reshape((args.ncells, args.ncells))
            ax_row[2] = plot_projectedZ_snap(np.log10(weighted_map_Z), ax_row[2], args, clim=args.Zlim, cmap=old_metal_color_map)
            df_thisproj = df_thisproj.replace([0, np.inf, -np.inf], np.nan).dropna(subset=['metal', 'weights'], axis=0)

            # ------plotting projected metallicity profiles---------------
            Zgrad_arr, ax_row[0] = plot_Zprof_snap(df_thisproj, ax_row[0], args)

            # ------plotting projected metallicity histograms---------------
            Zdist_arr, ax_row[1] = plot_Zdist_snap(df_thisproj, ax_row[1], args)

        # ------saving fig------------------
        outfile_rootname = '%s_%s_projectedZ_prof_hist_map_%s%s%s%s.png' % (args.output, args.halo, args.Zgrad_den, args.upto_text, args.weightby_text, args.res_text)
        if args.do_all_sims: outfile_rootname = 'z=*_' + outfile_rootname[len(args.output) + 1:]
        figname = args.fig_dir + outfile_rootname.replace('*', '%.5F' % (args.current_redshift))

        fig.savefig(figname)
        myprint('Saved plot as ' + figname, args)

        plt.show(block=False)
        print_mpi('This snapshots completed in %s mins' % ((time.time() - start_time_this_snapshot) / 60), args)

    if ncores > 1: print_master('Parallely: time taken for ' + str(total_snaps) + ' snapshots with ' + str(ncores) + ' cores was %s mins' % ((time.time() - start_time) / 60), args)
    else: print_master('Serially: time taken for ' + str(total_snaps) + ' snapshots with ' + str(ncores) + ' core was %s mins' % ((time.time() - start_time) / 60), args)