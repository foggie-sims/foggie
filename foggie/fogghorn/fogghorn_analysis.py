"""

    Filename: fogghorn_analysis.py
    Authors: Cassi, Ayan,
    Created: 06-12-24
    Last modified: 07-22-24 by Cassi

    This "master" script calls the relevant functions to produce a set of basic analysis plots for all outputs in the directory passed to it.
    The user can choose which plots Or groups of plots to make. This script does the book-keeping for existing plots and multiprocessing.
    The actual plotting routines are in XXXX.py

    Plots included so far:
    - Gas density projection
    - New stars density projection
    - Kennicutt-Schmidt relation compared to KMT09 relation

    Example of how to run (in ipython):
    run fogghorn_analysis.py --directory /Users/acharyya/models/simulation_output/foggie/halo_004123/nref11c_nref9f --system ayan_local --halo 4123 --output RD0038 --upto_kpc 10 --docomoving --weight mass --make_plot gas_metallicity_histogram,gas_density_projection --all_sf_plots

"""

from foggie.fogghorn.header import *
from foggie.fogghorn.util import *

start_time = datetime.now()

# --------------------------------------------------------------------------------------------------------------------
def update_table(snap, args):
    '''
    Determines if the halo info table needs to be updated with information from this snapshot
    and returns True if needing an update and False if not.
    '''

    # Load the table if it exists
    if (os.path.exists(args.save_directory + '/halo_data.txt')):
        data = Table.read(args.save_directory + '/halo_data.txt', format='ascii.fixed_width')
    else:
        return True

    # If the table already exists, search for this snapshot in the table
    if (snap in data['snapshot']):
        if not args.silent: print('Halo info for snapshot ' + snap + ' already calculated.', )
        if args.clobber:
            if not args.silent: print(' But we will re-calculate it...')
            calc = True
        else:
            if not args.silent: print(' So we will skip it.')
            calc = False
    else:
        print('Calculating halo info for snapshot ' + snap + '...')
        calc = True

    return calc

# --------------------------------------------------------------------------------------------------------------------
def which_plots_asked_for(args):
    '''
    Determines which plots have been asked for by the user, and then checks, which of them already exists, and
    returns the list of plots that need to be still made
    '''
    plots_asked_for = args.plots_asked_for

    if args.all_plots:
        plots_asked_for += np.hstack([sf_plots, fb_plots, vis_plots, metal_plots, 'info_table']) # these *_plots variables are in header.py
    else:
        if args.all_sf_plots: plots_asked_for += sf_plots
        if args.all_fb_plots: plots_asked_for += fb_plots
        if args.all_vis_plots: plots_asked_for += vis_plots
        if args.all_metal_plots: plots_asked_for += metal_plots
        if args.all_pop_plots: plots_asked_for += ['info_table']

    for p in range(len(plots_asked_for)):
        if plots_asked_for[p] in pop_plots:
            plots_asked_for[p] = 'info_table'

    plots_asked_for = np.unique(plots_asked_for)
    print(plots_asked_for)

    return plots_asked_for

# --------------------------------------------------------------------------------------------------------------------
def make_plots(snap, args, queue):
    '''
    Finds the halo center and other properties of the dataset and then calls the plotting scripts.
    Returns nothing. Saves outputs as multiple png files
    '''

    # ------------- Determine which plots need to be made -----------------------
    plots_asked_for = which_plots_asked_for(args)
    plots_to_make = []

    for thisplot in plots_asked_for:
        output_filename = generate_plot_filename(thisplot, args, snap)
        if need_to_make_this_plot(output_filename, args) or (thisplot=='info_table'):     # Population plots always need to be remade
            plots_to_make += [thisplot]

    need_to_load_snapshot = False
    for plot in plots_to_make:
        if (plot=='info_table'):
            if update_table(snap, args): need_to_load_snapshot = True
        else: need_to_load_snapshot = True

    myprint('Total %d plots asked for, of which %d will be made, others already exist' %(len(plots_asked_for), len(plots_to_make)), args)

    if len(plots_to_make) > 0:
        # ----------------------- Read the snapshot ----------------------
        if need_to_load_snapshot:
            filename = args.directory + '/' + snap + '/' + snap
            if args.trackfile == None:
                halos_df_name = args.code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/'
                halos_df_name += 'halo_cen_smoothed' if args.use_cen_smoothed else 'halo_c_v'
                ds, region = foggie_load(filename, args.trackfile, do_filter_particles=True, disk_relative=args.disk_rel, halo_c_v_name=halos_df_name)
            else:
                ds, region = foggie_load(filename, args.trackfile, do_filter_particles=True, disk_relative=args.disk_rel)

        # ----------------------- Make the plots ---------------------------------------------
        for thisplot in plots_to_make:
            if (thisplot != 'info_table'):
                output_filename = generate_plot_filename(thisplot, args, snap)
                globals()[thisplot](ds, region, args, output_filename) # all plotting functions should preferably have this same argument list in their function definitions
            else:
                if update_table(snap, args):
                    print('Updating table', snap)
                    # Make the table if it does not exist
                    if not (os.path.exists(args.save_directory + '/halo_data.txt')):
                        data = make_table()
                    row = get_halo_info(ds, snap, args)
                    if (args.nproc != 1):
                        print(row)
                        queue.put(row)
                    else:
                        if (os.path.exists(args.save_directory + '/halo_data.txt')): data = Table.read(args.save_directory + '/halo_data.txt', format='ascii.fixed_width')
                        data.add_row(row)
                        data.sort('time')
                        data.write(args.save_directory + '/halo_data.txt', format='ascii.fixed_width', overwrite=True)
                elif (args.nproc != 1):
                    print('Not updating table', snap)
                    queue.put('no entry')
                if (args.nproc==1):
                    for p in pop_plots:
                        output_filename = args.save_directory + '/' + p[5:] + '.png'
                        globals()[p](args, output_filename)

    print('Yayyy you have completed making all plots for this snap ' + snap)

# --------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()

    # ------------------ Figure out directory and outputs -------------------------------------
    if args.save_directory is None:
        args.save_directory = args.directory + '/plots'
        Path(args.save_directory).mkdir(parents=True, exist_ok=True)

    if args.trackfile is None: _, _, _, args.code_path, args.trackfile, _, _, _ = get_run_loc_etc(args) # for FOGGIE production runs it knows which trackfile to grab

    args.table_needed = False
    for plot in pop_plots:
        if (plot in args.make_plots) or (args.all_pop_plots):
            args.table_needed = True

    if args.output is not None: # Running on specific output/s
        outputs = make_output_list(args.output)
    else: # Running on all snapshots in the directory
        outputs = []
        for fname in os.listdir(args.directory):
            folder_path = os.path.join(args.directory, fname)
            if os.path.isdir(folder_path) and ((fname[0:2]=='DD') or (fname[0:2]=='RD')):
                outputs.append(fname)
    print(outputs)

    # ----------------- Add some parameters to args that will be used throughout ----------------------------------
    args.density_cut_text = '_wdencut' if args.use_density_cut else ''
    args.upto_text = '' if args.upto_kpc is None else '_upto%.1Fckpchinv' % args.upto_kpc if args.docomoving else '_upto%.1Fkpc' % args.upto_kpc

    # --------- Loop over outputs, for either single-processor or parallel processor computing ---------------
    if (args.nproc == 1):
        queue = []
        for snap in outputs:
            make_plots(snap, args, queue)
        print('Serially: time taken for ' + str(len(outputs)) + ' snapshots with ' + str(args.nproc) + ' core was %s' % timedelta(seconds=(datetime.now() - start_time).seconds))
    else:
        # ------- Split into a number of groupings equal to the number of processors and run one process per processor ---------
        for i in range(len(outputs)//args.nproc):
            threads = []
            queue = multi.Queue()
            rows = []
            for j in range(args.nproc):
                snap = outputs[args.nproc*i+j]
                print(snap)
                threads.append(multi.Process(target=make_plots, args=[snap, args, queue]))
            for t in threads:
                t.start()
            for t in threads:
                row = queue.get()
                rows.append(row)
            for t in threads:
                t.join()
            for r in rows:
                print(r)
                if (r != 'no entry'):
                    data = Table.read(args.save_directory + '/halo_data.txt', format='ascii.fixed_width')
                    data.add_row(r)
                    data.sort('time')
                    data.write(args.save_directory + '/halo_data.txt', format='ascii.fixed_width', overwrite=True)
        # ----- For any leftover snapshots, run one per processor ------------------
        threads = []
        queue = multi.Queue()
        rows = []
        for j in range(len(outputs) % args.nproc):
            snap = outputs[-(j+1)]
            threads.append(multi.Process(target=make_plots, args=[snap, args, queue]))
        for t in threads:
            t.start()
        for t in threads:
            row = queue.get()
            rows.append(row)
        for t in threads:
            t.join()
        for r in rows:
            print(r)
            if (r != 'no entry'):
                data = Table.read(args.save_directory + '/halo_data.txt', format='ascii.fixed_width')
                data.add_row(r)
                data.sort('time')
                data.write(args.save_directory + '/halo_data.txt', format='ascii.fixed_width', overwrite=True)
        # Remake population plots now that all data has been collected
        plots_asked_for = which_plots_asked_for(args)
        if ('info_table') in plots_asked_for:
            for plot in pop_plots:
                output_filename = args.save_directory + '/' + plot[5:] + '.png'
                globals()[plot](args, output_filename)
        print('Parallely: time taken for ' + str(len(outputs)) + ' snapshots with ' + str(args.nproc) + ' cores was %s' % timedelta(seconds=(datetime.now() - start_time).seconds))

    '''
    # the following only works for running the script with mpirun -n <nproc> python fogghorn_analysis.py --options....
    # --------domain decomposition; for mpi parallelisation-------------
    total_snaps = len(outputs)

    comm = MPI.COMM_WORLD
    args.nproc = comm.size
    rank = comm.rank
    print_master('Total number of MPI ranks = ' + str(args.nproc) + '. Starting at: {:%Y-%m-%d %H:%M:%S}'.format(datetime.now()), args)
    comm.Barrier()  # wait till all cores reached here and then resume

    split_at_cpu = total_snaps - args.nproc * int(total_snaps / args.nproc)
    nper_cpu1 = int(total_snaps / args.nproc)
    nper_cpu2 = nper_cpu1 + 1
    if rank < split_at_cpu:
        core_start = rank * nper_cpu2
        core_end = (rank + 1) * nper_cpu2 - 1
    else:
        core_start = split_at_cpu * nper_cpu2 + (rank - split_at_cpu) * nper_cpu1
        core_end = split_at_cpu * nper_cpu2 + (rank - split_at_cpu + 1) * nper_cpu1 - 1

    # -------------loop over snapshots-----------------
    print_mpi('Operating on snapshots ' + str(core_start + 1) + ' to ' + str(core_end + 1) + ', i.e., ' + str(core_end - core_start + 1) + ' out of ' + str(total_snaps) + ' snapshots', args)

    for index in range(core_start, core_end + 1):
        start_time_this_snapshot = time.time()
        args.snap = outputs[index]
        print_mpi('Doing snapshot ' + args.snap + ' which is ' + str(index + 1 - core_start) + ' out of the total ' + str(core_end - core_start + 1) + ' snapshots...', args)

        make_plots(args.snap, args) # this is the main stuff
        if args.table_needed: update_table(args.snap, args)

    if args.nproc > 1: print_master('Parallely: time taken for ' + str(total_snaps) + ' snapshots with ' + str(args.nproc) + ' cores was %s' % timedelta(seconds=(datetime.now() - start_time).seconds), args)
    else: print_master('Serially: time taken for ' + str(total_snaps) + ' snapshots with ' + str(args.nproc) + ' core was %s' % timedelta(seconds=(datetime.now() - start_time).seconds), args)
'''
