"""

    Filename: fogghorn_analysis.py
    Authors: Cassi, Ayan,
    Created: 06-12-24
    Last modified: 06-12-24 by Ayan

    This "master" script calls the relevant functions to produce a set of basic analysis plots for all outputs in the directory passed to it.
    The user can choose which plots Or groups of plots to make. This script does the book-keeping for existing plots and multiprocessing.
    The actual plotting routines are in XXXX.py

    Plots included so far:
    - Gas density projection
    - New stars density projection
    - Kennicutt-Schmidt relation compared to KMT09 relation

    Example of how to run (in ipython): run fogghorn_analysis.py --directory /Users/acharyya/models/simulation_output/foggie/halo_008508/nref11c_nref9f --system ayan_local --halo 8508 --output RD0030 --upto_kpc 10 --docomoving --weight mass

"""

from header import *
from util import *

start_time = time.time()

# --------------------------------------------------------------------------------------------------------------------
def make_plots(snap, args):
    '''
    Finds the halo center and other properties of the dataset and then calls the plotting scripts.
    Returns nothing. Saves outputs as multiple png files
    '''

    # ----------------------- Read the snapshot ----------------------
    filename = args.directory + '/' + snap + '/' + snap
    ds, region = foggie_load(filename, args.trackfile, disk_relative=True)

    # ----------------- Add some parameters to args that will be used throughout ----------------------------------
    args.snap = snap
    args.projection_axis_dict = {'x': ds.x_unit_disk, 'y': ds.y_unit_disk, 'z': ds.z_unit_disk}
    args.projection_text = '_disk-' + args.projection if args.disk_rel else '_' + args.projection
    args.density_cut_text = '_wdencut' if args.use_density_cut else ''

    # --------- If a upto_kpc is specified, then the analysis 'region' will be restricted up to that value ---------
    if args.upto_kpc is not None:
        if args.docomoving: args.galrad = args.upto_kpc / (1 + ds.current_redshift) / 0.695  # include stuff within a fixed comoving kpc h^-1, 0.695 is Hubble constant
        else: args.galrad = args.upto_kpc  # include stuff within a fixed physical kpc
        region = ds.sphere(ds.halo_center_kpc, ds.arr(args.galrad, 'kpc'))

        args.upto_text = '_upto%.1Fckpchinv' % args.upto_kpc if args.docomoving else '_upto%.1Fkpc' % args.upto_kpc
    else:
        args.galrad = ds.refine_width / 2.
        args.upto_text = ''

    # ----------------------- Make the plots ---------------------------------------------
    gas_density_projection(ds, region, args)
    edge_visualizations(ds, region, args)
    young_stars_density_projection(ds, region, args)
    KS_relation(ds, region, args)
    outflow_rates(ds, region, args)
    gas_metallicity_projection(ds, region, args)
    gas_metallicity_radial_profile(ds, region, args)
    gas_metallicity_histogram(ds, region, args)
    gas_metallicity_resolved_MZR(ds, region, args)
    print_mpi('Yayyy you have completed making all plots for this snap ' + snap, args)

# --------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()

    # ------------------ Figure out directory and outputs -------------------------------------
    if args.save_directory is None:
        args.save_directory = args.directory + '/plots'
        Path(args.save_directory).mkdir(parents=True, exist_ok=True)

    if args.trackfile is None: _, _, _, _, args.trackfile, _, _, _ = get_run_loc_etc(args) # for FOGGIE production runs it knows which trackfile to grab

    if args.output is not None: # Running on specific output/s
        outputs = make_output_list(args.output)
    else: # Running on all snapshots in the directory
        outputs = []
        for fname in os.listdir(args.directory):
            folder_path = os.path.join(args.directory, fname)
            if os.path.isdir(folder_path) and ((fname[0:2]=='DD') or (fname[0:2]=='RD')):
                outputs.append(fname)
    '''
    # --------- Loop over outputs, for either single-processor or parallel processor computing ---------------
    if (args.nproc == 1):
        for snap in outputs:
            make_plots(snap, args)
        print('Serially: time taken for ' + str(len(outputs)) + ' snapshots with ' + str(args.nproc) + ' core was %s mins' % ((time.time() - start_time) / 60))
    else:
        # ------- Split into a number of groupings equal to the number of processors and run one process per processor ---------
        for i in range(len(outputs)//args.nproc):
            threads = []
            for j in range(args.nproc):
                snap = outputs[args.nproc*i+j]
                threads.append(multi.Process(target=make_plots, args=[snap, args]))
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        # ----- For any leftover snapshots, run one per processor ------------------
        threads = []
        for j in range(len(outputs) % args.nproc):
            snap = outputs[-(j+1)]
            threads.append(multi.Process(target=make_plots, args=[snap, args]))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        print('Parallely: time taken for ' + str(len(outputs)) + ' snapshots with ' + str(args.nproc) + ' cores was %s mins' % ((time.time() - start_time) / 60))
    '''
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
    print_mpi('Operating on snapshots ' + str(core_start + 1) + ' to ' + str(core_end + 1) + ', i.e., ' + str(
        core_end - core_start + 1) + ' out of ' + str(total_snaps) + ' snapshots', args)

    for index in range(core_start + args.start_index, core_end + 1):
        start_time_this_snapshot = time.time()
        this_output = outputs[index]
        print_mpi('Doing snapshot ' + this_output + ' of halo ' + args.halo + ' which is ' + str(index + 1 - core_start) + ' out of the total ' + str(core_end - core_start + 1) + ' snapshots...', args)

        ##do stuff
        make_plots(this_output, args)

    if args.nproc > 1: print_master('Parallely: time taken for ' + str(total_snaps) + ' snapshots with ' + str(args.nproc) + ' cores was %s' % timedelta(seconds=(datetime.now() - start_time).seconds), args)
    else: print_master('Serially: time taken for ' + str(total_snaps) + ' snapshots with ' + str(args.nproc) + ' core was %s' % timedelta(seconds=(datetime.now() - start_time).seconds), args)
