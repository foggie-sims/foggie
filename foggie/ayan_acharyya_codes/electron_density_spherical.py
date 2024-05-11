#!/usr/bin/env python3

"""

    Title :      electron_density_spherical
    Notes :      Plot SPHERICAL gas density, electron density and profile ALL in one plot
    Output :     Combined plots as png files plus, optionally, these files stitched into a movie
    Author :     Ayan Acharyya
    Started :    May 2024
    Examples :   run electron_density_spherical.py --system ayan_pleiades --halo 8508 --upto_kpc 50 --docomoving --do_all_sims --write_file
                 run electron_density_spherical.py --system ayan_local --halo 4123 --upto_kpc 10 --output RD0038 --docomoving --nbins 100 --clobber_plot
"""
from header import *
from util import *
plt.rcParams['axes.linewidth'] = 1
from datashader_movie import field_dict, unit_dict, islog_dict, get_correct_tablename
from uncertainties import ufloat, unumpy
from datetime import datetime, timedelta
from mpl_toolkits.axes_grid1 import make_axes_locatable
from plot_MZgrad import load_df
from compute_MZgrad import get_disk_stellar_mass

start_time = datetime.now()

# -------------------------------------------------------------------------------
def get_df_from_ds(box, args, outfilename=None):
    '''
    Function to make a pandas dataframe from the yt dataset, including only the electron density and gas density profile,
    then writes dataframe to file for faster access in future
    This function is somewhat based on foggie.utils.prep_dataframe.prep_dataframe()
    :return: dataframe
    '''
    # -------------read/write pandas df file with ALL fields-------------------
    Path(args.output_dir + 'txtfiles/').mkdir(parents=True, exist_ok=True)  # creating the directory structure, if doesn't exist already
    if outfilename is None: outfilename = get_correct_tablename(args)

    if not os.path.exists(outfilename) or args.clobber:
        myprint(outfilename + ' does not exist. Creating afresh..', args)

        df = pd.DataFrame()
        fields = ['rad', 'density', 'el_density'] # only the relevant properties
        if args.weight is not None: fields += [args.weight]

        for index, field in enumerate(fields):
            myprint('Doing property: ' + field + ', which is ' + str(index + 1) + ' of the ' + str(len(fields)) + ' fields..', args)
            df[field] = box[field_dict[field]].in_units(unit_dict[field]).ndarray_view()

        df.to_csv(outfilename, sep='\t', index=None)
    else:
        myprint('Reading from existing file ' + outfilename, args)
        try:
            df = pd.read_table(outfilename, delim_whitespace=True, comment='#')
        except pd.errors.EmptyDataError:
            print('File existed, but it was empty, so making new file afresh..')
            dummy_args = copy.deepcopy(args)
            dummy_args.clobber = True
            df = get_df_from_ds(box, dummy_args, outfilename=outfilename)

    df = df[df['rad'].between(0, args.galrad)]  # in case this dataframe has been read in from a file corresponding to a larger chunk of the box
    cols_to_extract = ['rad', 'density', 'el_density']
    if args.weight is not None: cols_to_extract += [args.weight]
    df = df[cols_to_extract] # only keeping the columns that are needed to get Z gradient

    return df

# ----------------------------------------------------------------
def plot_projected_map(ds, box, quantity, unit, ax, args, clim=None, cmap='viridis', projection='x'):
    '''
    Function to plot a given projected metallicity map on to a given axis
    :return: axis handle
    '''
    plt.style.use('seaborn-white')
    plt.rcParams['axes.linewidth'] = 1
    field = field_dict[quantity]

    prj = yt.ProjectionPlot(ds, projection, field, center=ds.halo_center_kpc, data_source=box, width=2 * args.galrad * kpc, weight_field=field_dict[args.weight] if args.weight is not None else None, fontsize=args.fontsize / args.fontfactor)

    prj.set_log(field, islog_dict[quantity])
    prj.set_unit(field, unit)
    prj.set_zlim(field, zmin=clim[0], zmax=clim[1])
    cmap.set_bad('k')
    prj.set_cmap(field, cmap)

    # ------plotting onto a matplotlib figure--------------
    position = ax.get_position()
    prj.plots[field].axes = ax
    divider = make_axes_locatable(ax)
    prj._setup_plots()
    prj.plots[field].axes.set_position(position)  # in order to resize the axis back to where it should be

    # -----------making the axis labels etc--------------
    ax.set_xticks(np.linspace(-int(args.galrad), int(args.galrad), 5))
    ax.set_xticklabels(['%.1F' % item for item in ax.get_xticks()], fontsize=args.fontsize / args.fontfactor)
    ax.set_xlabel('Offset (kpc)', fontsize=args.fontsize / args.fontfactor)

    ax.set_yticks(np.linspace(-int(args.galrad), int(args.galrad), 5))
    ax.set_yticklabels(['%.1F' % item for item in ax.get_yticks()], fontsize=args.fontsize / args.fontfactor)
    ax.set_ylabel('Offset (kpc)', fontsize=args.fontsize / args.fontfactor)

    # ---------making the colorbar axis once, that will correspond to all projections--------------
    fig = ax.figure
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(prj.plots[field].cb.mappable, orientation='vertical', cax=cax)
    cbar.ax.tick_params(labelsize=args.fontsize / args.fontfactor, width=2.5, length=5)
    cbar.set_label(prj.plots[field].cax.get_ylabel(), fontsize=args.fontsize / args.fontfactor)

    return ax

# ---------------------------------------------------------------------------------
def piecewise_linear(x, central, alpha, break_rad, beta):
    '''
    Piecewise_linear function, for fitting broken power law in log-space
    '''
    return np.piecewise(x, [x < break_rad], [lambda x: central + alpha * x, lambda x: central + alpha * break_rad + beta * (x - break_rad)])


# ---------------------------------------------------------------------------------
def fit_binned(df, xcol, ycol, x_extent, ax=None, weightcol=None, color='darkorange'):
    '''
    Function to bin data, fit the binned data, and overplot binned data and fit on existing plot
    Returns the fitted parameters and axis handle
    '''
    x_bins = np.arange(0, x_extent, 1) # 1 physical kpc bin size
    df['binned_cat'] = pd.cut(df[xcol], x_bins)

    if weightcol is not None:
        agg_func = lambda x: np.sum(x * df.loc[x.index, weightcol]) / np.sum(df.loc[x.index, weightcol]) # function to get weighted mean
        agg_u_func = lambda x: np.sqrt(((np.sum(df.loc[x.index, weightcol] * x**2) / np.sum(df.loc[x.index, weightcol])) - (np.sum(x * df.loc[x.index, weightcol]) / np.sum(df.loc[x.index, weightcol]))**2) * (np.sum(df.loc[x.index, weightcol]**2)) / (np.sum(df.loc[x.index, weightcol])**2 - np.sum(df.loc[x.index, weightcol]**2))) # eq 6 of http://seismo.berkeley.edu/~kirchner/Toolkits/Toolkit_12.pdf
    else:
        agg_func, agg_u_func = np.mean, np.std

    y_binned = df.groupby('binned_cat', as_index=False).agg([(ycol, agg_func)])[ycol].values.flatten()
    y_u_binned = df.groupby('binned_cat', as_index=False).agg([(ycol, agg_u_func)])[ycol].values.flatten()
    x_bin_centers = x_bins[:-1] + np.diff(x_bins) / 2

    # getting rid of potential nan values
    indices = np.array(np.logical_not(np.logical_or(np.isnan(x_bin_centers), np.isnan(y_binned))))
    x_bin_centers = x_bin_centers[indices]
    y_binned = y_binned[indices]
    y_u_binned = y_u_binned[indices]

    # ----------to plot mean binned y vs x profile--------------
    popt, pcov = curve_fit(piecewise_linear, x_bin_centers, y_binned, p0 = [20., -1, 5., -0.5]) # popt = [central (log cm^-3), alpha (dimensionless), break_rad (kpc), beta (dimensionless)]
    y_fitted = piecewise_linear(x_bin_centers, *popt)

    fit_result = [ufloat(popt[ind], np.sqrt(pcov[ind][ind])) for ind in np.arange(len(popt))]

    print('Upon radially binning, inferred fit parameters are: central = %.2f, alpha = %.1f, break_rad = %.1f, beta = %.1f' % (popt[0], popt[1], popt[2], popt[3]))

    if ax is not None:
        ax.errorbar(x_bin_centers, y_binned, c=color, yerr=y_u_binned, lw=0.5 if len(x_bins) > 10 else 1.0, ls='none', zorder=1)
        ax.scatter(x_bin_centers, y_binned, c=color, s=10 if len(x_bins) > 10 else 20, lw=0.5, ec='black', zorder=1)
        ax.plot(x_bin_centers, y_fitted, color='k', lw=1, ls='solid', zorder=5)
        if not args.notextonplot: ax.text(0.033, 0.05, 'central = %.2f \nalpha = %.2f\nbreak = %.2f kpc \nbeta = %.2f' % (popt[0], popt[1], popt[2], popt[3]), color='k', transform=ax.transAxes, fontsize=args.fontsize/args.fontfactor, ha='left', va='bottom', bbox=dict(facecolor='white', alpha=0.7, edgecolor='white'))
        return fit_result, ax
    else:
        return fit_result

# ----------------------------------------------------------------
def plot_profile(df, ax, args):
    '''
    Function to plot the radial metallicity profile (from input dataframe) as seen from all three projections, on to the given axis
    Also computes the projected metallicity gradient along each projection
    :return: fitted gradient across each projection, and the axis handle
    '''
    plt.style.use('seaborn-whitegrid')
    myprint('Now making the radial profile plot for ' + args.output + '..', args)
    ycol = 'el_density'

    df = df[df[ycol] > 0]

    if args.weight is not None:
        df['weighted_' + ycol] = len(df) * df[ycol] * df[args.weight] / np.sum(df[args.weight])
        ycol = 'weighted_' + ycol

    df['log_' + ycol] = np.log10(df[ycol]) # taking log AFTER the weighting
    ycol = 'log_' + ycol

    # ----------to plot the profile with all cells--------------df
    artist = dsshow(df, dsh.Point('rad', ycol), dsh.count(), norm='linear', x_range=(0, args.galrad), y_range=(args.ed_lim[0], args.ed_lim[1]), aspect = 'auto', ax=ax, cmap='Blues_r')

    # ----------to radially bin and fit the radial bins--------------
    fit_result, ax = fit_binned(df, 'rad', ycol, args.galrad, ax=ax, weightcol=args.weight)

    # ----------to annotate plot axes etc--------------
    ax.set_xlim(0, np.ceil(args.upto_kpc / 0.695) if args.docomoving else args.upto_kpc) # kpc
    ax.set_ylim(args.ed_lim[0], args.ed_lim[1]) # log limits

    ax.set_xticklabels(['%.1F' % item for item in ax.get_xticks()], fontsize=args.fontsize / args.fontfactor)
    ax.set_xlabel('Radius (kpc)', fontsize=args.fontsize / args.fontfactor)

    ax.set_yticklabels(['%.1F' % item for item in ax.get_yticks()], fontsize=args.fontsize / args.fontfactor)
    ax.set_ylabel(r'log El density (1/cm$^3$)', fontsize=args.fontsize / args.fontfactor)

    return fit_result, ax

# -----main code-----------------
if __name__ == '__main__':
    args_tuple = parse_args('8508', 'RD0042')  # default simulation to work upon when comand line args not provided
    if type(args_tuple) is tuple: args, ds, refine_box = args_tuple # if the sim has already been loaded in, in order to compute the box center (via utils.pull_halo_center()), then no need to do it again
    else: args = args_tuple
    if not args.keep: plt.close('all')

    # --------make new dataframe to store all results-----------------
    columns = ['output', 'redshift', 'time', 'sfr', 'log_mstar', 'central', 'central_u', 'alpha', 'alpha_u', 'break_rad', 'break_rad_u', 'beta', 'beta_u']
    df_full = pd.DataFrame(columns=columns)

    args.weightby_text = '_wtby_' + args.weight if args.weight is not None else ''
    args.upto_text = '_upto%.1Fckpchinv' % args.upto_kpc if args.docomoving else '_upto%.1Fkpc' % args.upto_kpc
    args.nbins_text = '_nbins%d' % args.nbins
    outfilename = args.output_dir + 'txtfiles/' + args.halo + '_spherical_el_density_evolution%s%s%s.txt' % (args.upto_text, args.nbins_text, args.weightby_text)

    Path(args.output_dir + 'txtfiles/').mkdir(parents=True, exist_ok=True)
    Path(args.output_dir + 'figs/').mkdir(parents=True, exist_ok=True)
    if not os.path.exists(outfilename) or args.clobber: df_full.to_csv(outfilename, sep='\t', index=None) # writing to file, so that invidual processors can read in and append

    # -------- reading in SFR info-------
    sfr_filename = args.code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/sfr'
    if os.path.exists(sfr_filename):
        print('Reading SFR history from', sfr_filename)
        sfr_df = pd.read_table(sfr_filename, names=('output', 'redshift', 'sfr'), comment='#', delim_whitespace=True)
    else:
        print('Did not find', sfr_filename, ', therefore will not include SFR')
        sfr_df = pd.DataFrame()

    # -------- reading in stellar mass info-------
    try:
        dummy_args = copy.deepcopy(args)
        dummy_args.weight = 'mass'
        dummy_args.weightby_text = '' if dummy_args.weight is None else '_wtby_' + dummy_args.weight
        dummy_args.Zgrad_den = 'kpc'
        dummy_args.use_density_cut = True
        mass_df = load_df(dummy_args)
        mass_df = mass_df[['output', 'log_mass']]
    except Exception:
        mass_df = pd.DataFrame()
        pass

    # --------domain decomposition; for mpi parallelisation-------------
    if args.do_all_sims: list_of_sims = get_all_sims_for_this_halo(args) # all snapshots of this particular halo
    else: list_of_sims = list(itertools.product([args.halo], args.output_arr))
    total_snaps = len(list_of_sims)

    comm = MPI.COMM_WORLD
    ncores = comm.size
    rank = comm.rank
    print_master('Total number of MPI ranks = ' + str(ncores) + '. Starting at: {:%Y-%m-%d %H:%M:%S}'.format(datetime.now()), args)
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
        args.weightby_text = '_wtby_' + args.weight if args.weight is not None else ''
        args.upto_text = '_upto%.1Fckpchinv' % args.upto_kpc if args.docomoving else '_upto%.1Fkpc' % args.upto_kpc
        args.nbins_text = '_nbins%d' % args.nbins

        args.current_redshift = ds.current_redshift
        args.current_time = ds.current_time.in_units('Gyr').tolist()

        args.color = 'cornflowerblue' # colors for the scatter plot and histogram
        args.gd_lim = [-2.5, 2.5]  # log Msun/pc^2 units
        args.ed_lim = [-11, 2] if args.weight is not None else [-6, -1]  # log cm^-2 units
        args.fontsize = 15
        args.fontfactor = 1.5

        # --------determining corresponding text suffixes and figname-------------
        args.fig_dir = args.output_dir + 'figs/'
        if not args.do_all_sims: args.fig_dir += args.output + '/'
        Path(args.fig_dir).mkdir(parents=True, exist_ok=True)

        outfile_rootname = '%s_%s_spherical_el_density%s%s%s.png' % (args.output, args.halo, args.upto_text, args.nbins_text, args.weightby_text)
        if args.do_all_sims: outfile_rootname = 'z=*_' + outfile_rootname[len(args.output) + 1:]
        figname = args.fig_dir + outfile_rootname.replace('*', '%.5F' % (args.current_redshift))

        if not os.path.exists(figname) or args.clobber_plot or args.write_file:
            try:
                # -------setting up fig--------------
                fig, [axes_proj_den, axes_proj_el_den, axes_prof] = plt.subplots(1, 3, figsize=(10, 3))
                fig.subplots_adjust(top=0.95, bottom=0.15, left=0.07, right=0.98, wspace=0.6)

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

                # ----------plotting projection plots of densities----------------------------
                axes_proj_den = plot_projected_map(ds, box, 'density', 'Msun/pc**2', axes_proj_den, args, clim=[1e-3, 1e3],  cmap=density_color_map,)
                axes_proj_el_den = plot_projected_map(ds, box, 'el_density', 'cm**-2', axes_proj_el_den, args, clim=[1e17, 1e21], cmap=e_color_map)

                # ------calculating projected electron density---------------
                df_snap_filename = args.output_dir + '/txtfiles/%s_spherical_el_density%s%s%s.txt' % (args.output, args.upto_text, args.nbins_text, args.weightby_text)

                if not os.path.exists(df_snap_filename) or args.clobber:
                    myprint(df_snap_filename + 'not found, creating afresh..', args)
                    df_snap = get_df_from_ds(box, args)
                    fit_result, axes_prof = plot_profile(df_snap, axes_prof, args)

                    df_snap.to_csv(df_snap_filename, sep='\t', index=None)
                    myprint('Saved file ' + df_snap_filename, args)
                else:
                    myprint('Reading in existing ' + df_snap_filename, args)
                    df_snap = pd.read_table(df_snap_filename, delim_whitespace=True, comment='#')

                    fit_result, axes_prof = plot_profile(df_snap, axes_prof, args)

                fit_result = np.array(fit_result)
                df_snap = df_snap.dropna()
                try: sfr = sfr_df[sfr_df['output'] == args.output]['sfr'].values[0]
                except Exception: sfr = -99

                try: log_mstar = mass_df[mass_df['output'] == args.output]['log_mass'].values[0]
                except Exception: log_mstar = np.log10(get_disk_stellar_mass(args))

                # ------update full dataframe and read it from file-----------
                df_full_row = np.hstack(([args.output, args.current_redshift, args.current_time, sfr, log_mstar], np.hstack([[item.n, item.s] for item in fit_result])))
                temp_df = pd.DataFrame(dict(zip(columns, df_full_row)), index=[0])
                temp_df.to_csv(outfilename, mode='a', sep='\t', header=False, index=None)

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

    # -------reading the full dataframe-----------------------
    df_full = pd.read_table(outfilename, delim_whitespace=True)
    df_full = df_full.drop_duplicates(subset='output', keep='last')
    df_full = df_full.sort_values(by='time')

    if ncores > 1: print_master('Parallely: time taken for ' + str(total_snaps) + ' snapshots with ' + str(ncores) + ' cores was %s' % timedelta(seconds=(datetime.now() - start_time).seconds), args)
    else: print_master('Serially: time taken for ' + str(total_snaps) + ' snapshots with ' + str(ncores) + ' core was %s' % timedelta(seconds=(datetime.now() - start_time).seconds), args)
