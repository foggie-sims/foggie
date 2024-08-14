#!/usr/bin/env python3

"""

    Title :      make_3D_FRB_electron_density
    Notes :      Make a 3D Fixed Resolution Buffer (FRB) for gas density and electron number density and save as a multi-extension fits file, and optionally plot along a chosen line of sight
    Output :     3D data cube as fits file, and optionally png figures
    Author :     Ayan Acharyya
    Started :    Aug 2024
    Examples :   run make_3D_FRB_electron_density.py --system ayan_pleiades --halo 8508 --upto_kpc 50 --docomoving --do_all_sims
                 run make_3D_FRB_electron_density.py --system ayan_hd --halo 4123 --upto_kpc 10 --output RD0038 --docomoving
"""
from header import *
from util import *
plt.rcParams['axes.linewidth'] = 1

start_time = datetime.now()

# --------------------------------------------------------------------------
def plot_3d_frb(data, ax, label=None, unit=None, clim=None,  cmap='viridis'):
    '''
    Function to make a 3D plot given a 3D numpy array in a given axis
    Returns axis handle
    '''
    z, x, y = data.nonzero()
    ax.scatter3D(z, x, y, c=data, cmap=cmap, alpha=0.3)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    return ax

# -----main code-----------------
if __name__ == '__main__':
    args_tuple = parse_args('8508', 'RD0042')  # default simulation to work upon when comand line args not provided
    if type(args_tuple) is tuple: args, ds, refine_box = args_tuple # if the sim has already been loaded in, in order to compute the box center (via utils.pull_halo_center()), then no need to do it again
    else: args = args_tuple
    if not args.keep: plt.close('all')

    quant_dict = {'density':['density', 'Gas density', 'Msun/pc**2', -2.5, 2.5, 'cornflowerblue', density_color_map], 'el_density':['El_number_density', 'Electron density', 'cm**-2', -6, -1, 'cornflowerblue', e_color_map]} # for each quantity: [yt field, label in plots, units, lower limit in log, upper limit in log, color for scatter plot, colormap]
    quant1 = 'el_density'
    quant2 = 'density'

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
        args.upto_text = '_upto%.1Fckpchinv' % args.upto_kpc if args.docomoving else '_upto%.1Fkpc' % args.upto_kpc
        args.current_redshift = ds.current_redshift
        args.current_time = ds.current_time.in_units('Gyr').tolist()

        # --------determining corresponding text suffixes and figname-------------
        args.fig_dir = args.output_dir + 'figs/'
        if not args.do_all_sims: args.fig_dir += args.output + '/'
        Path(args.fig_dir).mkdir(parents=True, exist_ok=True)

        args.fits_dir = args.output_dir + 'txtfiles/'
        Path(args.fits_dir).mkdir(parents=True, exist_ok=True)

        outfile_rootname = '%s_%s_3D_FRB_%s%s%s%s.png' % (args.output, args.halo, quant_dict[quant1][0], args.upto_text, args.nbins_text, args.weightby_text)
        if args.do_all_sims: outfile_rootname = 'z=*_' + outfile_rootname[len(args.output) + 1:]
        figname = args.fig_dir + outfile_rootname.replace('*', '%.5F' % (args.current_redshift))
        fitsname = args.fits_dir + outfile_rootname.replace('*', '%.5F' % (args.current_redshift)).replace('.png', '.fits')

        if not os.path.exists(fitsname) or args.clobber:
            try:
                # ------tailoring the simulation box for individual snapshot analysis--------
                if args.upto_kpc is not None:
                    if args.docomoving: args.galrad = args.upto_kpc / (1 + args.current_redshift) / 0.695  # fit within a fixed comoving kpc h^-1, 0.695 is Hubble constant
                    else: args.galrad = args.upto_kpc  # fit within a fixed physical kpc
                else:
                    args.re = get_re_from_coldgas(args) if args.use_gasre else get_re_from_stars(ds, args)
                    args.galrad = args.re * args.upto_re  # kpc
                args.ncells = int(2 * args.galrad / args.res)
                args.level = 4

                # extract the required box
                box_center = ds.halo_center_kpc
                box = ds.sphere(box_center, ds.arr(args.galrad, 'kpc'))
                sys.exit() ##
                # -------making the 3D FRBs--------------
                all_data = ds.smoothed_covering_grid(level=args.level, left_edge=[0.0, 0.0, 0.0], dims=[args.ncells, args.ncells, args.ncells], data_source=box)
                el_den_FRB = all_data[('gas', quant_dict[quant1][0])].in_units(quant_dict[quant1][2])
                den_FRB = all_data[('gas', quant_dict[quant2][0])].in_units(quant_dict[quant2][2])

                # -------saving the 3D FRBs--------------
                hdulist = fits.HDUList()
                hdulist.append(fits.PrimaryHDU())
                hdulist.append(fits.ImageHDU(data=el_den_FRB))
                hdulist.append(fits.ImageHDU(data=den_FRB))
                hdulist.writeto(fitsname)
                myprint('Saved FRBs as ' + fitsname, args)

                # -------setting up fig--------------
                fig, [ax_el_den, ax_den] = plt.subplots(1, 2, figsize=(6, 6), projection='3d')
                fig.subplots_adjust(top=0.95, bottom=0.15, left=0.07, right=0.98, wspace=0.6)

                # ----------plotting projection plots of densities----------------------------
                ax_el_den = plot_3d_frb(den_FRB, ax_el_den, label=quant_dict[quant1][1], unit=quant_dict[quant1][2], clim=[quant_dict[quant1][3], quant_dict[quant1][4]],  cmap=quant_dict[quant1][6])
                ax_den = plot_3d_frb(el_den_FRB, ax_den, label=quant_dict[quant2][1], unit=quant_dict[quant2][2], clim=[quant_dict[quant2][3], quant_dict[quant2][4]], cmap=quant_dict[quant2][6])

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

    # -----------------------------------------------------------------------------------
    if ncores > 1: print_master('Parallely: time taken for ' + str(total_snaps) + ' snapshots with ' + str(ncores) + ' cores was %s' % timedelta(seconds=(datetime.now() - start_time).seconds), args)
    else: print_master('Serially: time taken for ' + str(total_snaps) + ' snapshots with ' + str(ncores) + ' core was %s' % timedelta(seconds=(datetime.now() - start_time).seconds), args)
