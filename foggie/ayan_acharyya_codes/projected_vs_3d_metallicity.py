#!/usr/bin/env python3

"""

    Title :      projected_vs_3d_metallicity
    Notes :      Plot projected and unprojected 3D metallicity gradient, distribution and projections ALL in one plot
    Output :     Combined plots as one png file
    Author :     Ayan Acharyya
    Started :    Aug 2023
    Examples :   run projected_vs_3d_metallicity.py --system ayan_local --halo 8508 --Zgrad_den kpc --upto_kpc 10 --forpaper --output RD0030
"""
from header import *
from util import *
import projected_Zgrad_evolution as proj
import nonprojected_Zgrad_evolution as nonproj
from compute_MZgrad import get_re_from_coldgas, get_re_from_stars

start_time = time.time()

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
            args.fit_multiple = True # for the Z distribution panel
            args.hide_multiplefit = True # for the Z distribution panel
            args.islog = True # for the Z distribution panel
            args.nbins = 100 # for the Z distribution panel
            args.weight = 'mass'

        args.current_redshift = ds.current_redshift
        args.current_time = ds.current_time.in_units('Gyr').tolist()

        args.projections = ['x', 'y', 'z']
        args.col_arr = ['salmon', 'seagreen', 'cornflowerblue']  # colors corresponding to different projections
        args.color = 'k' # color for non-projected radial profile fitting
        args.Zlim = [-2, 1]  # log Zsun units
        args.res = args.res_arr[0]
        if args.docomoving: args.res = args.res / (1 + args.current_redshift) / 0.695  # converting from comoving kcp h^-1 to physical kpc
        args.fontsize = 23
        args.fontfactor = 1

        # --------determining corresponding text suffixes and figname-------------
        args.weightby_text = '_wtby_' + args.weight if args.weight is not None else ''
        args.fitmultiple_text = '_fitmultiple' if args.fit_multiple else ''
        args.density_cut_text = '_wdencut' if args.use_density_cut else ''
        args.islog_text = '_islog' if args.islog else ''
        if args.upto_kpc is not None: upto_text = '_upto%.1Fckpchinv' % args.upto_kpc if args.docomoving else '_upto%.1Fkpc' % args.upto_kpc
        else: upto_text = '_upto%.1FRe' % args.upto_re

        args.fig_dir = args.output_dir + 'figs/'
        if not args.do_all_sims: args.fig_dir += args.output + '/'
        Path(args.fig_dir).mkdir(parents=True, exist_ok=True)

        outfile_rootname = '%s_%s_projected_vs_3d_Zgrad_den_%s%s%s.png' % (args.output, args.halo, args.Zgrad_den, upto_text, args.weightby_text)
        if args.do_all_sims: outfile_rootname = 'z=*_' + outfile_rootname[len(args.output) + 1:]
        figname = args.fig_dir + outfile_rootname.replace('*', '%.5F' % (args.current_redshift))

        if not os.path.exists(figname) or args.clobber_plot:
            #try:
            # -------setting up fig--------------
            nrow, ncol1, ncol2 = 2 if args.forpaper else 3, 2, 3
            ncol = ncol1 * ncol2 # the overall figure is nrow x (ncol1 + ncol2)
            if args.forpaper: fig = plt.figure(figsize=(16, 10))
            else: fig = plt.figure(figsize=(16, 12))

            axes_proj_snap = [plt.subplot2grid(shape=(nrow, ncol), loc=(0, int(item)), colspan=ncol1) for item in np.linspace(0, ncol1 * 2, 3)]
            ax_prof_proj = plt.subplot2grid(shape=(nrow, ncol), loc=(1, 0), colspan=ncol2)
            ax_prof_3d = plt.subplot2grid(shape=(nrow, ncol), loc=(1, ncol2), colspan=ncol2)
            if args.forpaper:
                fig.subplots_adjust(top=0.88, bottom=0.1, left=0.08, right=0.98, wspace=0.1, hspace=0.5)
            else:
                ax_dist_proj = plt.subplot2grid(shape=(nrow, ncol), loc=(2, 0), colspan=ncol - ncol2)
                ax_dist_3d = plt.subplot2grid(shape=(nrow, ncol), loc=(2, ncol2), colspan=ncol - ncol2)
                fig.subplots_adjust(top=0.88, bottom=0.07, left=0.1, right=0.95, wspace=0.8, hspace=0.4)

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

            # ------getting the dataframe for projected metallcity and plotting the projections---------------
            df_proj_filename = args.output_dir + 'txtfiles/' + args.output + '_df_boxrad_%.2Fkpc_projectedZ.txt'%(args.galrad)

            if not os.path.exists(df_proj_filename) or args.clobber:
                myprint(df_proj_filename + 'not found, creating afresh..', args)
                df_proj = pd.DataFrame()
                map_dist = proj.get_dist_map(args)
                df_proj['rad'] = map_dist.flatten()

                for index, thisproj in enumerate(args.projections):
                    frb = proj.make_frb_from_box(box, box_center, 2 * args.galrad, thisproj, args)
                    df_proj, weighted_map_Z = proj.make_df_from_frb(frb, df_proj, thisproj,  args)
                    axes_proj_snap[index] = proj.plot_projectedZ_snap(np.log10(weighted_map_Z), thisproj, axes_proj_snap[index], args, clim=args.Zlim, cmap=old_metal_color_map, color=args.col_arr[index])

                df_proj.to_csv(df_proj_filename, sep='\t', index=None)
                myprint('Saved file ' + df_proj_filename, args)
            else:
                myprint('Reading in existing ' + df_proj_filename, args)
                df_proj = pd.read_table(df_proj_filename, delim_whitespace=True, comment='#')
                for index, thisproj in enumerate(args.projections):
                    if args.weight is None: weighted_Z = df_proj['metal_' + thisproj]
                    else: weighted_Z = len(df_proj) * df_proj['metal_' + thisproj] * df_proj['weights_' + thisproj] / np.sum(df_proj['weights_' + thisproj])
                    weighted_map_Z = weighted_Z.values.reshape((args.ncells, args.ncells))
                    axes_proj_snap[index] = proj.plot_projectedZ_snap(np.log10(weighted_map_Z), thisproj, axes_proj_snap[index], args, clim=args.Zlim, cmap=old_metal_color_map, color=args.col_arr[index])

            df_proj = df_proj.dropna()

            # ------getting the dataframe for 3D metallcity---------------
            df_3d = nonproj.make_df_from_box(box, args)

            # ------plotting projected metallicity profiles---------------
            Zgrad_arr, ax_prof_proj = proj.plot_Zprof_snap(df_proj, ax_prof_proj, args)

            # ------plotting nonprojected metallicity profiles---------------
            Zgrad, ax_prof_3d = nonproj.plot_Zprof_snap(df_3d, ax_prof_3d, args, hidey=True)

            if not args.forpaper:
                # ------plotting projected metallicity histograms---------------
                Zdist_arr, ax_dist_proj = proj.plot_Zdist_snap(df_proj, ax_dist_proj, args)

                # ------plotting nonprojected metallicity histograms---------------
                Zdist, ax_dist_3d = nonproj.plot_Zdist_snap(df_3d, ax_dist_3d, args, hidey=True)

            # ------saving fig------------------
            fig.savefig(figname)
            myprint('Saved plot as ' + figname, args)

            plt.show(block=False)
            print_mpi('This snapshots completed in %s mins' % ((time.time() - start_time_this_snapshot) / 60), args)
            '''
            except Exception as e:
                print_mpi('Skipping ' + this_sim[1] + ' because ' + str(e), args)
                continue
            '''
        else:
            print('Skipping snapshot %s as %s already exists. Use --clobber_plot to remake figure.' %(args.output, figname))
            continue

    if ncores > 1: print_master('Parallely: time taken for ' + str(total_snaps) + ' snapshots with ' + str(ncores) + ' cores was %s mins' % ((time.time() - start_time) / 60), args)
    else: print_master('Serially: time taken for ' + str(total_snaps) + ' snapshots with ' + str(ncores) + ' core was %s mins' % ((time.time() - start_time) / 60), args)