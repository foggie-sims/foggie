#!/usr/bin/env python3

"""

    Title :      projected_Zgrad_hist_map
    Notes :      Plot PROJECTED metallicity gradient, distribution and projections and velocity dispersion ALL in one plot, for a given line of sight
    Output :     Combined plots as png files plus
    Author :     Ayan Acharyya
    Started :    Oct 2023
    Examples :   run projected_Zgrad_hist_map.py --system ayan_pleiades --halo 8508 --Zgrad_den kpc --upto_kpc 10 --docomoving --res_arc 0.1 --weight mass --output RD0030
                 run projected_Zgrad_hist_map.py --system ayan_hd --halo 2392 --Zgrad_den kpc --upto_kpc 10 --res_arc 0.1 --forproposal --output DD0417 --proj x,y,z --vcol vtan
"""
from header import *
from util import *
from nonprojected_Zgrad_hist_map import plot_projection, plot_Zprof_snap, plot_Zdist_snap

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
    frb = dummy_proj.to_frb(box_width, args.ncells, center=box_center)

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

    if args.weight is not None:
        map_weights = np.array(frb['gas', args.weight])
        df[args.weight + '_' + projection] = map_weights.flatten()

    return df

field_dict = {'vrad': 'radial_velocity_corrected', 'vdisp_3d': 'velocity_dispersion_3d', 'vtan': 'tangential_velocity_corrected', \
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
        if args.forproposal:
            args.nofit = True
            args.nbins = 20 # for the Z distribution panel
        if args.forpaper or args.forproposal:
            args.use_density_cut = True
            args.docomoving = True
            args.islog = True # for the Z distribution panel
            args.weight = 'mass'
            args.fontsize = 15
        if args.forpaper:
            args.nbins = 30 # for the Z distribution panel
            #args.fit_multiple = True # True # for the Z distribution panel
            args.nofit = True #
            args.hide_multiplefit = True

        args.current_redshift = ds.current_redshift
        args.current_time = ds.current_time.in_units('Gyr').tolist()

        args.Zlim = [-1.5, 0.5] if args.forproposal else [-2, 1]# log Zsun units
        if args.res_arc is not None:
            args.res = get_kpc_from_arc_at_redshift(float(args.res_arc), args.current_redshift)
            native_res_at_z = 0.27 / (1 + args.current_redshift) # converting from comoving kpc to physical kpc
            args.res_text = '_res%.1farc' % float(args.res_arc)
            if args.res < native_res_at_z:
                print('Computed resolution %.2f kpc is below native FOGGIE res at z=%.2f, so we set resolution to the native res = %.2f kpc.'%(args.res, args.current_redshift, native_res_at_z))
                args.res = native_res_at_z # kpc
                args.res_text = '_res%.1fkpc' % float(args.res)
        else:
            args.res = args.res_arr[0]
            if args.docomoving:
                args.res_text = '_res%.1fckpchinv' % float(args.res)
                args.res = args.res / (1 + args.current_redshift) / 0.695 # converting from comoving kcp h^-1 to physical kpc
            else:
                args.res_text = '_res%.1fkpc' % float(args.res)

        args.fontfactor = 1.2

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
        figname = args.fig_dir + '%s_%s_projectedZ_prof_hist_map_%s%s%s%s.png' % (args.output, args.halo, args.vcol, args.upto_text, args.weightby_text, args.res_text)

        if not os.path.exists(figname) or args.clobber_plot:
            # -------setting up fig--------------
            nrow, ncol1, ncol2 = 3, 2, 3
            ncol = ncol1 * ncol2  # the overall figure is nrow x (ncol1 + ncol2)
            fig = plt.figure(figsize=(8, 8))
            axes_met_proj = [plt.subplot2grid(shape=(nrow, ncol), loc=(0, int(item)), colspan=ncol1) for item in np.linspace(0, ncol1 * 2, 3)]
            axes_vel_proj = [plt.subplot2grid(shape=(nrow, ncol), loc=(1, int(item)), colspan=ncol1) for item in np.linspace(0, ncol1 * 2, 3)]
            ax_prof_snap = plt.subplot2grid(shape=(nrow, ncol), loc=(2, 0), colspan=ncol2)
            ax_dist_snap = plt.subplot2grid(shape=(nrow, ncol), loc=(2, ncol2), colspan=ncol - ncol2)
            fig.tight_layout()
            fig.subplots_adjust(top=0.9, bottom=0.07, left=0.1, right=0.87, wspace=2.0, hspace=0.35)

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
            box_width = 2 * args.galrad * kpc

            if args.use_density_cut:
                rho_cut = get_density_cut(ds.current_time.in_units('Gyr'))  # based on Cassi's CGM-ISM density cut-off
                box = box.cut_region(['obj["gas", "density"] > %.1E' % rho_cut])
                print('Imposing a density criteria to get ISM above density', rho_cut, 'g/cm^3')

            # -----------reading in the snapshot's projected metallicity dataframe----------
            df_snap_filename = args.output_dir + 'txtfiles/' + args.output + '_df_boxrad_%.2Fkpc_projectedZ%s%s%s.txt' % (args.galrad, args.res_text, args.weightby_text, args.density_cut_text)

            if not os.path.exists(df_snap_filename) or args.clobber:
                myprint(df_snap_filename + ' not found, creating afresh..', args)
                df_snap = pd.DataFrame()
                map_dist = get_dist_map(args)
                df_snap['rad'] = map_dist.flatten()

                # -------loop over lines of sight---------------------
                for thisproj in ['x', 'y', 'z']:
                    field_dict.update({'vdisp_los':'velocity_dispersion_' + thisproj, 'vlos': 'v' + thisproj + '_corrected'})
                    frb = make_frb_from_box(box, box_center, box_width, thisproj, args)
                    df_snap = make_df_from_frb(frb, df_snap, thisproj, args)

                df_snap.to_csv(df_snap_filename, sep='\t', index=None)
                myprint('Saved file ' + df_snap_filename, args)
            else:
                myprint('Reading in existing ' + df_snap_filename, args)
                df_snap = pd.read_table(df_snap_filename, delim_whitespace=True, comment='#')

            # ------plotting projected metallcity snapshots---------------
            axes_met_proj = plot_projection('metal', box, box_center, box_width, axes_met_proj, args, clim=[10 ** -1.5, 10 ** 0] if args.forproposal else None, cmap=old_metal_color_map, ncells=args.ncells)

            # ------plotting projected velocity quantity snapshots---------------
            axes_vel_proj = plot_projection(args.vcol, box, box_center, box_width, axes_vel_proj, args, clim=[-150, 150] if args.vcol == 'vrad' or args.vcol == 'vphi' or args.vcol == 'vlos' else [0, 150] if args.forproposal else None, cmap='PRGn' if args.vcol == 'vrad' else 'viridis', ncells=args.ncells)

            # -------extracting columns for just the given line of sight---------------------
            columns_to_extract = ['rad', 'metal_' + args.projection]
            if args.weight is not None: columns_to_extract += [args.weight + '_' + args.projection]
            df_thisproj = df_snap[columns_to_extract]
            df_thisproj = df_thisproj.replace([0, np.inf, -np.inf], np.nan).dropna(subset=columns_to_extract, axis=0)
            df_thisproj.columns = df_thisproj.columns.str.replace('_' + args.projection, '')

            # ------plotting projected metallicity profiles---------------
            Zgrad_arr, ax_prof_snap = plot_Zprof_snap(df_thisproj, ax_prof_snap, args)

            # ------plotting projected metallicity histograms---------------
            Zdist_arr, ax_dist_snap = plot_Zdist_snap(df_thisproj, ax_dist_snap, args)

            # ------saving fig------------------
            fig.savefig(figname)
            myprint('Saved plot as ' + figname, args)

            plt.show(block=False)
            print_mpi('This snapshots completed in %s mins' % ((time.time() - start_time_this_snapshot) / 60), args)
        else:
            print('Skipping snapshot %s as %s already exists. Use --clobber_plot to remake figure.' %(args.output, figname))
            continue

    if ncores > 1: print_master('Parallely: time taken for ' + str(total_snaps) + ' snapshots with ' + str(ncores) + ' cores was %s mins' % ((time.time() - start_time) / 60), args)
    else: print_master('Serially: time taken for ' + str(total_snaps) + ' snapshots with ' + str(ncores) + ' core was %s mins' % ((time.time() - start_time) / 60), args)