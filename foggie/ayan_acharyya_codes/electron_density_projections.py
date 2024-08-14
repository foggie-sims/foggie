#!/usr/bin/env python3

"""

    Title :      electron_density_projections
    Notes :      Plot PROJECTED gas density, electron density, as well as electron density profile ALL in one plot, for ALL 3 projections
    Output :     Combined plots as png files plus, optionally, these files stitched into a movie
    Author :     Ayan Acharyya
    Started :    May 2024
    Examples :   run electron_density_projections.py --system ayan_pleiades --halo 8508 --upto_kpc 50 --res 0.2 --docomoving --do_all_sims --write_file
                 run electron_density_projections.py --system ayan_hd --halo 4123 --upto_kpc 50 --res 0.2 --output RD0038 --docomoving --nbins 100 --clobber_plot
"""
from header import *
from util import *
plt.rcParams['axes.linewidth'] = 1
from projected_Zgrad_evolution import get_dist_map, make_frb_from_box
from plot_MZgrad import load_df
from compute_MZgrad import get_disk_stellar_mass

start_time = datetime.now()

# ----------------------------------------------------------------
def plot_projected_map(map, projection, ax, args, clim=None, cmap='viridis', quantity=None, hidex=False, hidey=False):
    '''
    Function to plot a given projected metallicity map on to a given axis
    :return: axis handle
    '''
    plt.style.use('seaborn-white')
    plt.rcParams['axes.linewidth'] = 1
    myprint('Now making projection plot for ' + projection + '..', args)
    #sns.set_style('ticks')  # instead of darkgrid, so that there are no grids overlaid on the projections

    delta = 0.3 # the small offset between the actual limits and intended tick labels is to ensure that tick labels do not reach the very edge of the plot
    proj = ax.imshow(map, cmap=cmap, extent=[-args.galrad - delta, args.galrad + delta, -args.galrad - delta, args.galrad + delta], vmin=clim[0] if clim is not None else None, vmax=clim[1] if clim is not None else None)

    # -----------making the axis labels etc--------------
    ax.set_xticks(np.linspace(-int(args.galrad), int(args.galrad), 5))
    if hidex:
        ax.set_xticklabels([])
    else:
        ax.set_xticklabels(['%.1F' % item for item in ax.get_xticks()], fontsize=args.fontsize / args.fontfactor)
        ax.set_xlabel('Offset (kpc)', fontsize=args.fontsize / args.fontfactor)

    ax.set_yticks(np.linspace(-int(args.galrad), int(args.galrad), 5))
    if hidey:
        ax.set_yticklabels([])
    else:
        ax.set_yticklabels(['%.1F' % item for item in ax.get_yticks()], fontsize=args.fontsize / args.fontfactor)
        ax.set_ylabel('Offset (kpc)', fontsize=args.fontsize / args.fontfactor)

    ax.text(0.9 * args.galrad, 0.9 * args.galrad, projection, ha='right', va='top', c='k', fontsize=args.fontsize)

    # ---------making the colorbar axis once, that will correspond to all projections--------------
    if projection == 'x':
        cbar_width = 0.18
        if quantity == 'gas':
            cax_xpos, cax_ypos, cax_width, cax_height = 0.1, 0.93, cbar_width, 0.02
            label = r'log Gas density (M$_{\odot}$/pc$^2$)'

            ax.text(0.03, 0.95, args.output, color='k', transform=ax.transAxes, fontsize=args.fontsize / args.fontfactor, va='top', ha='left', bbox=dict(facecolor='white', alpha=0.8, edgecolor='k'))
            ax.text(0.03, 0.2, 'z = %.2F\nt = %.1F Gyr' % (args.current_redshift, args.current_time), ha='left', va='top', transform=ax.transAxes, fontsize=args.fontsize / args.fontfactor, bbox=dict(facecolor='white', alpha=0.8, edgecolor='k'))

        elif quantity == 'el':
            cax_xpos, cax_ypos, cax_width, cax_height = 0.1 + cbar_width + 0.15, 0.93, cbar_width, 0.02
            label = r'log Electron density (1/cm$^2$)'
        else:
            cax_xpos, cax_ypos, cax_width, cax_height = 0.1, 0.93, 0.8, 0.02
            label = ''

        fig = ax.figure
        cax = fig.add_axes([cax_xpos, cax_ypos, cax_width, cax_height])
        plt.colorbar(proj, cax=cax, orientation='horizontal')

        cax.set_xticklabels(['%.1F' % index for index in cax.get_xticks()], fontsize=args.fontsize / args.fontfactor)
        fig.text(cax_xpos + cax_width / 2, cax_ypos + cax_height + 0.005, label, ha='center', va='bottom', fontsize=args.fontsize / args.fontfactor)

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
    x_bins = np.linspace(0, x_extent, 10)
    df['binned_cat'] = pd.cut(df[xcol], x_bins)

    if weightcol is not None:
        agg_func = lambda x: np.sum(x * df.loc[x.index, weightcol]) / np.sum(df.loc[x.index, weightcol]) # function to get weighted mean
        agg_u_func = lambda x: np.sqrt(((np.sum(df.loc[x.index, weightcol] * x**2) / np.sum(df.loc[x.index, weightcol])) - (np.sum(x * df.loc[x.index, weightcol]) / np.sum(df.loc[x.index, weightcol]))**2) * (np.sum(df.loc[x.index, weightcol]**2)) / (np.sum(df.loc[x.index, weightcol])**2 - np.sum(df.loc[x.index, weightcol]**2))) # eq 6 of http://seismo.berkeley.edu/~kirchner/Toolkits/Toolkit_12.pdf
    else:
        agg_func, agg_u_func = np.mean, np.std

    y_binned = df.groupby('binned_cat', as_index=False).agg([(ycol, agg_func)])[ycol].values.flatten()
    y_u_binned = df.groupby('binned_cat', as_index=False).agg([(ycol, agg_u_func)])[ycol].values.flatten()
    x_bin_centers = x_bins[:-1] + np.diff(x_bins) / 2

    # ----------to plot mean binned y vs x profile--------------
    popt, pcov = curve_fit(piecewise_linear, x_bin_centers, y_binned, p0 = [20., -1, 5., -0.5]) # popt = [central (log cm^-3), alpha (dimensionless), break_rad (kpc), beta (dimensionless)]
    y_fitted = piecewise_linear(x_bin_centers, *popt)

    print('Upon radially binning, inferred fit parameters are: central = %.2f, alpha = %.1f, break_rad = %.1f, beta = %.1f' % (popt[0], popt[1], popt[2], popt[3]))

    if ax is not None:
        ax.errorbar(x_bin_centers, y_binned, c=color, yerr=y_u_binned, lw=2, ls='none', zorder=1)
        ax.scatter(x_bin_centers, y_binned, c=color, s=20, lw=0.5, ec='black', zorder=1)
        ax.plot(x_bin_centers, y_fitted, color='k', lw=1, ls='solid', zorder=5)
        if not args.notextonplot: ax.text(0.033, 0.05, 'central = %.2f \nalpha = %.2f\nbreak = %.2f kpc \nbeta = %.2f' % (popt[0], popt[1], popt[2], popt[3]), color='k', transform=ax.transAxes, fontsize=args.fontsize/args.fontfactor, ha='left', va='bottom')#, bbox=dict(facecolor='white', alpha=0.8, edgecolor='white'))
        return popt, ax
    else:
        return popt

# ----------------------------------------------------------------
def plot_profile(df, projection, ax, args, hidex=False, hidey=False):
    '''
    Function to plot the radial metallicity profile (from input dataframe) as seen from all three projections, on to the given axis
    Also computes the projected metallicity gradient along each projection
    :return: fitted gradient across each projection, and the axis handle
    '''
    plt.style.use('seaborn-whitegrid')
    myprint('Now making the radial profile plot for ' + args.output + '..', args)
    ycol = 'el_density_' + projection

    df = df[df[ycol] > 0]

    if args.weight is not None:
        df['weighted_' + ycol] = len(df) * df[ycol] * df['weights_' + projection] / np.sum(df['weights_' + projection])
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

    if hidex:
        ax.set_xticklabels([])
    else:
        ax.set_xticklabels(['%.1F' % item for item in ax.get_xticks()], fontsize=args.fontsize / args.fontfactor)
        ax.set_xlabel('Radius (kpc)', fontsize=args.fontsize / args.fontfactor)

    if hidey:
        ax.set_yticklabels([])
    else:
        ax.set_yticklabels(['%.1F' % item for item in ax.get_yticks()], fontsize=args.fontsize / args.fontfactor)
        ax.set_ylabel(r'log El density (1/cm$^2$)', fontsize=args.fontsize / args.fontfactor)

    return fit_result, ax

# -----main code-----------------
if __name__ == '__main__':
    args_tuple = parse_args('8508', 'RD0042')  # default simulation to work upon when comand line args not provided
    if type(args_tuple) is tuple: args, ds, refine_box = args_tuple # if the sim has already been loaded in, in order to compute the box center (via utils.pull_halo_center()), then no need to do it again
    else: args = args_tuple
    if not args.keep: plt.close('all')

    # --------make new dataframe to store all results-----------------
    columns = ['output', 'redshift', 'time', 'sfr', 'log_mstar', 'central_x', 'alpha_x', 'break_rad_x', 'beta_x', 'central_y', 'alpha_y', 'break_rad_y', 'beta_y', 'central_z', 'alpha_z', 'break_rad_z', 'beta_z']
    df_full = pd.DataFrame(columns=columns)

    args.weightby_text = '_wtby_' + args.weight if args.weight is not None else ''
    args.upto_text = '_upto%.1Fckpchinv' % args.upto_kpc if args.docomoving else '_upto%.1Fkpc' % args.upto_kpc
    args.res_text = '_res%.1Fckpchinv' % float(args.res) if args.docomoving else '_res%.1Fkpc' % float(args.res)
    args.nbins_text = '_nbins%d' % args.nbins
    outfilename = args.output_dir + '/txtfiles/' + args.halo + '_projected_el_density_evolution%s%s%s%s.txt' % (args.upto_text, args.res_text, args.nbins_text, args.weightby_text)
    if not os.path.exists(outfilename): df_full.to_csv(outfilename, sep='\t', index=None) # writing to file, so that invidual processors can read in and append

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
        args.res_text = '_res%.1Fckpchinv' % float(args.res) if args.docomoving else '_res%.1Fkpc' % float(args.res)
        args.nbins_text = '_nbins%d' % args.nbins

        args.current_redshift = ds.current_redshift
        args.current_time = ds.current_time.in_units('Gyr').tolist()

        args.projections = ['x', 'y', 'z']
        args.color = 'cornflowerblue' # colors for the scatter plot and histogram
        args.gd_lim = [-2.5, 2.5]  # log Msun/pc^2 units
        args.ed_lim = [17, 21]  # log cm^-2 units
        args.res = args.res_arr[0]
        if args.docomoving: args.res = args.res / (1 + args.current_redshift) / 0.695  # converting from comoving kcp h^-1 to physical kpc
        args.fontsize = 15
        args.fontfactor = 1.5

        # --------determining corresponding text suffixes and figname-------------
        args.fig_dir = args.output_dir + 'figs/'
        if not args.do_all_sims: args.fig_dir += args.output + '/'
        Path(args.fig_dir).mkdir(parents=True, exist_ok=True)

        outfile_rootname = '%s_%s_projected_el_density%s%s%s%s.png' % (args.output, args.halo, args.upto_text, args.res_text, args.nbins_text, args.weightby_text)
        if args.do_all_sims: outfile_rootname = 'z=*_' + outfile_rootname[len(args.output) + 1:]
        figname = args.fig_dir + outfile_rootname.replace('*', '%.5F' % (args.current_redshift))

        if not os.path.exists(figname) or args.clobber_plot or args.write_file:
            try:
                # -------setting up fig--------------
                nrow, ncol = 3, 3
                fig = plt.figure(figsize=(8, 7))
                axes_proj_den = [plt.subplot2grid(shape=(nrow, ncol), loc=(item, 0), colspan=1) for item in np.arange(3)]
                axes_proj_el_den = [plt.subplot2grid(shape=(nrow, ncol), loc=(item, 1), colspan=1) for item in np.arange(3)]

                axes_prof = [plt.subplot2grid(shape=(nrow, ncol), loc=(item, 2), colspan=1) for item in np.arange(3)]

                fig.tight_layout()
                fig.subplots_adjust(top=0.9, bottom=0.07, left=0.1, right=0.95, wspace=0.8, hspace=0.15)

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

                # ------calculating projected electron density---------------
                df_snap_filename = args.output_dir + 'txtfiles/%s_projected_el_density%s%s%s%s.txt' % (args.output, args.upto_text, args.res_text, args.nbins_text, args.weightby_text)
                fit_result = []

                if not os.path.exists(df_snap_filename) or args.clobber:
                    myprint(df_snap_filename + 'not found, creating afresh..', args)
                    df_snap = pd.DataFrame()
                    map_dist = get_dist_map(args)
                    df_snap['rad'] = map_dist.flatten()

                    for index, thisproj in enumerate(args.projections):
                        print('Doing projection %s, which %d of %d..' %(thisproj, index+1, len(args.projections)))
                        frb = make_frb_from_box(box, box_center, 2 * args.galrad, thisproj, args)
                        gas_density_map = frb['gas', 'density'].in_units('Msun/pc**2')
                        el_density_map = frb['gas', 'El_number_density'].in_units('1/cm**2')
                        df_snap['density_' + thisproj] = gas_density_map.flatten()
                        df_snap['el_density_' + thisproj] = el_density_map.flatten()

                        axes_proj_den[index] = plot_projected_map(np.log10(gas_density_map), thisproj, axes_proj_den[index], args, clim=args.gd_lim, cmap=density_color_map, quantity='gas', hidex=index < len(args.projections) - 1)
                        axes_proj_el_den[index] = plot_projected_map(np.log10(el_density_map), thisproj, axes_proj_el_den[index], args, clim=args.ed_lim, cmap=e_color_map, quantity='el', hidex=index < len(args.projections) - 1)
                        this_fit, axes_prof[index] = plot_profile(df_snap, thisproj, axes_prof[index], args, hidex=index < len(args.projections) - 1)
                        fit_result.append(this_fit)

                    df_snap.to_csv(df_snap_filename, sep='\t', index=None)
                    myprint('Saved file ' + df_snap_filename, args)
                else:
                    myprint('Reading in existing ' + df_snap_filename, args)
                    df_snap = pd.read_table(df_snap_filename, delim_whitespace=True, comment='#')
                    for index, thisproj in enumerate(args.projections):
                        print('Doing projection %s, which %d of %d..' %(thisproj, index+1, len(args.projections)))
                        gas_density_map = df_snap['density_' + thisproj].values.reshape((args.ncells, args.ncells))
                        if args.weight is not None: el_density = len(df_snap) * df_snap['el_density_' + thisproj] * df_snap[args.weight + '_' + thisproj] / np.sum(df_snap[args.weight + '_' + thisproj])
                        else: el_density = df_snap['el_density_' + thisproj]
                        el_density_map = el_density.values.reshape((args.ncells, args.ncells))

                        axes_proj_den[index] = plot_projected_map(np.log10(gas_density_map), thisproj, axes_proj_den[index], args, clim=args.gd_lim, cmap=density_color_map, quantity='gas', hidex=index < len(args.projections) - 1)
                        axes_proj_el_den[index] = plot_projected_map(np.log10(el_density_map), thisproj, axes_proj_el_den[index], args, clim=args.ed_lim, cmap=e_color_map, quantity='el', hidex=index < len(args.projections) - 1)
                        this_fit, axes_prof[index] = plot_profile(df_snap, thisproj, axes_prof[index], args, hidex=index < len(args.projections) - 1)
                        fit_result.append(this_fit)

                fit_result = np.array(fit_result)
                df_snap = df_snap.dropna()
                try: sfr = sfr_df[sfr_df['output'] == args.output]['sfr'].values[0]
                except Exception: sfr = -99

                try: log_mstar = mass_df[mass_df['output'] == args.output]['log_mass'].values[0]
                except Exception: log_mstar = np.log10(get_disk_stellar_mass(args))

                # ------update full dataframe and read it from file-----------
                df_full_row = np.hstack(([args.output, args.current_redshift, args.current_time, sfr, log_mstar], np.hstack([[fit_result[i][0], fit_result[i][1], fit_result[i][2], fit_result[i][2]] for i in range(len(args.projections))])))
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
