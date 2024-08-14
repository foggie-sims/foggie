#!/usr/bin/env python3

"""

    Title :      make_3D_FRB_electron_density
    Notes :      Make a 3D Fixed Resolution Buffer (FRB) for gas density and electron number density and save as a multi-extension fits file, and optionally plot along a chosen line of sight
    Output :     3D data cube as fits file, and optionally png figures
    Author :     Ayan Acharyya
    Started :    Aug 2024
    Examples :   run make_3D_FRB_electron_density.py --system ayan_pleiades --halo 8508 --res 0.3 --upto_kpc 50 --docomoving --do_all_sims
                 run make_3D_FRB_electron_density.py --system ayan_hd --halo 4123 --res 0.3 --upto_kpc 10 --output RD0038 --docomoving --clobber --plot_3d
"""
from header import *
from util import *
from yt.visualization.fits_image import FITSImageData
plt.rcParams['axes.linewidth'] = 1

start_time = datetime.now()

# --------------------------------------------------------------------------
def plot_3d_frb(data, ax, label=None, unit=None, clim=None,  cmap='viridis'):
    '''
    Function to make a 3D plot given a 3D numpy array in a given axis
    Returns axis handle
    '''
    z, x, y = data.nonzero()
    ax.scatter3D(z, x, y, c=np.log10(data), cmap=cmap, alpha=0.3)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    return ax

# --------------------------------------------------------------------------
def plot_proj_frb(data, ax, label='', unit='', clim=None,  cmap='viridis'):
    '''
    Function to make a 2D projection plot (along one line of sight) given a 3D numpy array, in a given axis
    Returns axis handle
    '''
    data_proj = np.sum(data, axis=2)
    p = ax.imshow(np.log10(data_proj), cmap=cmap)

    ax.set_xlabel('x')
    ax.set_ylabel('y')

    cbar = plt.colorbar(p)
    cbar.set_label(f'log integrated {label} ({unit})')

    return ax

# -----main code-----------------
if __name__ == '__main__':
    args_tuple = parse_args('8508', 'RD0042')  # default simulation to work upon when comand line args not provided
    if type(args_tuple) is tuple: args, ds, refine_box = args_tuple # if the sim has already been loaded in, in order to compute the box center (via utils.pull_halo_center()), then no need to do it again
    else: args = args_tuple
    if not args.keep: plt.close('all')

    quant_dict = {'density':['density', 'Gas density', 'Msun/pc**3', -2.5, 2.5, 'cornflowerblue', density_color_map], 'el_density':['El_number_density', 'Electron density', 'cm**-3', -6, -1, 'cornflowerblue', e_color_map]} # for each quantity: [yt field, label in plots, units, lower limit in log, upper limit in log, color for scatter plot, colormap]
    quant_arr = ['el_density', 'density']

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
        start_time_this_snapshot = datetime.now()
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
        args.res = args.res_arr[0]
        if args.docomoving: args.res = args.res / (1 + args.current_redshift) / 0.695  # converting from comoving kcp h^-1 to physical kpc

        # --------determining corresponding text suffixes and figname-------------
        args.fig_dir = args.output_dir + 'figs/'
        if not args.do_all_sims: args.fig_dir += args.output + '/'
        Path(args.fig_dir).mkdir(parents=True, exist_ok=True)

        args.fits_dir = args.output_dir + 'txtfiles/'
        Path(args.fits_dir).mkdir(parents=True, exist_ok=True)

        outfile_rootname = '%s_%s_%s_FRB_%s%s.png' % (args.output, args.halo, '3D' if args.plot_3d else 'proj', quant_dict[quant_arr[0]][0], args.upto_text)
        if args.do_all_sims: outfile_rootname = 'z=*_' + outfile_rootname[len(args.output) + 1:]
        figname = args.fig_dir + outfile_rootname.replace('*', '%.5F' % (args.current_redshift))
        fitsname = args.fits_dir + outfile_rootname.replace('*', '%.5F' % (args.current_redshift)).replace('.png', '.fits')

        if not os.path.exists(fitsname) or args.clobber:
            #try:
            # ------tailoring the simulation box for individual snapshot analysis--------
            if args.upto_kpc is not None:
                if args.docomoving: args.galrad = args.upto_kpc / (1 + args.current_redshift) / 0.695  # fit within a fixed comoving kpc h^-1, 0.695 is Hubble constant
                else: args.galrad = args.upto_kpc  # fit within a fixed physical kpc
            else:
                args.re = get_re_from_coldgas(args) if args.use_gasre else get_re_from_stars(ds, args)
                args.galrad = args.re * args.upto_re  # kpc

            # extract the required box
            box_width = args.galrad * 2  # in kpc
            box_width_kpc = ds.arr(box_width, 'kpc')
            args.ncells = int(box_width / args.res)
            box_center = ds.halo_center_kpc
            box = ds.r[box_center[0] - box_width_kpc / 2.: box_center[0] + box_width_kpc / 2., box_center[1] - box_width_kpc / 2.: box_center[1] + box_width_kpc / 2., box_center[2] - box_width_kpc / 2.: box_center[2] + box_width_kpc / 2., ]

            # -------setting up fig--------------
            fig = plt.figure(figsize=(10, 5))
            fig.subplots_adjust(top=0.95, bottom=0.15, left=0.07, right=0.98, wspace=0.1, hspace=0.)

            # -------making and plotting the 3D FRBs--------------
            all_data = ds.arbitrary_grid(left_edge=box.left_edge, right_edge=box.right_edge, dims=[args.ncells, args.ncells, args.ncells])
            img_hdu_list = []

            for index, quant in enumerate(quant_arr):
                myprint(f'Making and plotting FRB for {quant} which is {index+1} out of {len(quant_arr)} quantities..', args)
                FRB = all_data[('gas', quant_dict[quant][0])].in_units(quant_dict[quant][2]) # making the 3D FRB
                img_hdu = FITSImageData(FRB, ('gas', quant_dict[quant][1])) # making the FITS ImageHDU
                img_hdu_list.append(img_hdu)
                ax = fig.add_subplot(1, len(quant_arr), index + 1, projection='3d' if args.plot_3d else None)
                if args.plot_3d: ax = plot_3d_frb(FRB, ax, label=quant_dict[quant][1], unit=quant_dict[quant][2], clim=[quant_dict[quant][3], quant_dict[quant][4]], cmap=quant_dict[quant][6]) # making the 3D plot
                else: ax = plot_proj_frb(FRB, ax, label=quant_dict[quant][1], unit=quant_dict[quant][2], clim=[quant_dict[quant][3], quant_dict[quant][4]], cmap=quant_dict[quant][6]) # making the 3D plot

            # ------saving fits file------------------
            combined_img_hdu = FITSImageData.from_images(img_hdu_list)
            combined_img_hdu.writeto(fitsname, overwrite=args.clobber)
            myprint('Saved fits file as ' + fitsname, args)

            # ------saving fig------------------
            fig.savefig(figname)
            myprint('Saved plot as ' + figname, args)

            plt.show(block=False)
            print_mpi('This snapshots completed in %s' % timedelta(seconds=(datetime.now() - start_time_this_snapshot).seconds), args)
            '''
            except Exception as e:
                print_mpi('Skipping ' + this_sim[1] + ' because ' + str(e), args)
                continue
            '''
        else:
            print('Skipping snapshot %s as %s already exists. Use --clobber_plot to remake figure.' %(args.output, fitsname))
            continue

    # -----------------------------------------------------------------------------------
    if ncores > 1: print_master('Parallely: time taken for ' + str(total_snaps) + ' snapshots with ' + str(ncores) + ' cores was %s' % timedelta(seconds=(datetime.now() - start_time).seconds), args)
    else: print_master('Serially: time taken for ' + str(total_snaps) + ' snapshots with ' + str(ncores) + ' core was %s' % timedelta(seconds=(datetime.now() - start_time).seconds), args)
