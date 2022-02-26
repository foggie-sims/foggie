#!/usr/bin/env python3

"""

    Title :      get_halo_track
    Notes :      Compute the track file for a given FOGGIE halo
    Output :     Two ASCII files: one with the halo centers and one with the halo corners (i.e. track) depending upon the specified refine box size
    Author :     Ayan Acharyya
    Started :    Feb 2022
    Examples :   run get_halo_track.py --system ayan_pleiades --foggie_dir bigbox --run 25Mpc_DM_256-L3-gas --halo 5205 --refsize 200 --reflevel 7

"""
from header import *
from util import *
from foggie.utils.get_halo_center import get_halo_center
from run_foggie_sim import get_shifts

# -----------------------------------------------------------------------------
def make_center_track_file(list_of_sims, center_track_file, args):
    '''
    Function to make the halo center track file
    This is partly based on foggie.halo_analysis.get_center_track.py
    '''
    start_time = time.time()
    total_snaps = len(list_of_sims)
    print('List of the total ' + str(total_snaps) + ' sims =', np.array(list_of_sims)[:,1])

    # --------setup dataframe-----------
    df = pd.DataFrame(columns=['redshift', 'center_x', 'center_y', 'center_z', 'output'])

    # --------domain decomposition; for mpi parallelisation-------------
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

    # --------------------------------------------------------------
    print_mpi('Operating on snapshots ' + str(core_start + 1) + ' to ' + str(core_end + 1) + ', i.e., ' + str(core_end - core_start + 1) + ' out of ' + str(total_snaps) + ' snapshots', args)

    for index in range(core_start + args.start_index, core_end + 1):
        start_time_this_snapshot = time.time()
        this_sim = list_of_sims[index]
        print_mpi('Doing snapshot ' + this_sim[1] + ' of halo ' + this_sim[0] + ' which is ' + str(index + 1 - core_start) + ' out of the total ' + str(core_end - core_start + 1) + ' snapshots...', args)

        args.output = this_sim[1]
        snap_name = args.output_path + args.output + '/' + args.output

        ds = yt.load(snap_name)

        # extract the required quantities
        zz = ds.current_redshift
        new_center, vel_center = get_halo_center(ds, args.center_guess, radius=50) # searches within 50 physical kpc
        df.loc[len(df)] = [zz, new_center[0], new_center[1], new_center[2], args.output]

        print_mpi('This snapshots completed in %s mins' % ((time.time() - start_time_this_snapshot) / 60), args)

    # sorting dataframe
    df = df.sort_values(by='redshift')
    filename_before_interp = center_track_file.replace('_interp', '')
    df.to_csv(filename_before_interp, sep='\t', index=None)
    print_master('Saved file ' + filename_before_interp, args)

    # now interpolate the track to the interval given as a parameter
    df = df[df['redshift'] <= 15]
    n_points = int((np.max(df['redshift']) - np.min(df['redshift']))  / args.z_interval)
    newredshifts = np.min(df['redshift']) + np.arange(n_points + 2) * args.z_interval
    new_center_x = np.interp(newredshifts, df['redshift'], df['center_x'])
    new_center_y = np.interp(newredshifts, df['redshift'], df['center_y'])
    new_center_z = np.interp(newredshifts, df['redshift'], df['center_z'])

    df_interp = pd.DataFrame({'redshift':newredshifts, 'center_x':new_center_x, 'center_y':new_center_y, 'center_z':new_center_z})

    df_interp.to_csv(center_track_file, sep='\t', index=None)
    print_master('Saved file ' + center_track_file, args)

    if ncores > 1: print_master('Parallely: time taken for ' + str(total_snaps) + ' snapshots with ' + str(ncores) + ' cores was %s mins' % ((time.time() - start_time) / 60), args)
    else: print_master('Serially: time taken for ' + str(total_snaps) + ' snapshots with ' + str(ncores) + ' core was %s mins' % ((time.time() - start_time) / 60), args)

# -----main code-----------------
if __name__ == '__main__':
    args = parse_args('8508', 'RD0042')  # default simulation to work upon when comand line args not provided

    if args.center_guess is None:
        # ------------------------get approximate halo center from L0 gas run halo catalogue combined with offsets-------------------
        halos = Table.read('/nobackup/jtumlins/CGM_bigbox/25Mpc_256_shielded-L0/BigBox_z2_rockstar/out_0.list', format='ascii', header_start=0)
        index = [halos['ID'] == int(args.halo[:4])]
        thishalo = halos[index]
        center_L0 = np.array([thishalo['X'][0] / 25., thishalo['Y'][0] / 25., thishalo['Z'][0] / 25.])  # divided by 25 to convert Mpc units to code units

        conf_log_file = args.root_dir + args.foggie_dir + '/' + 'halo_' + args.halo + '/' + args.run + '.conf_log.txt'
        shifts = get_shifts(conf_log_file)
        args.center_guess = center_L0 + np.array(shifts) / 255.  # to convert shifts into code units
        print('Using', args.center_guess, 'as initial guess for halo center at all redshifts')

    # parse paths and filenames
    if args.system == 'ayan_hd' or args.system == 'ayan_local': args.root_dir = '/Users/acharyya/Work/astro/'
    elif args.system == 'ayan_pleiades': args.root_dir = '/nobackup/aachary2/'
    args.output_path = args.root_dir + args.foggie_dir + '/' + 'halo_' + args.halo + '/' + args.run + '/'
    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    center_track_file = args.output_path + 'center_track_interp.dat'

    list_of_sims = get_all_sims_for_this_halo(args.output_path, args) # all snapshots of this particular halo

    if not os.path.exists(center_track_file) or args.clobber:
        print('File does not exist: ' + center_track_file + '; creating afresh..\n')
        make_center_track_file(list_of_sims, center_track_file, args)
    else:
        print('Using existing ' + center_track_file)

    halo_track_file = args.output_path + 'halo_track_%dkpc_nref%d' %(args.refsize, args.reflevel)
    offset = str(0.5 * args.refsize * 1e-3 / 25.) # converting refsize from kpc to physical code units, for the given 25 Mpc box

    if 'pleiades' in args.system: command = "tail -n +2 " + center_track_file + "  | awk '{print $1, $2-" + offset + ", $3-" + offset + ", $4-" + offset + ", $2+" + offset + ", $3+" + offset + ", $4+" + offset + ", " + str(args.reflevel) + "}' | tac > " + halo_track_file
    else: command = "tail -rn +2 " + center_track_file + "  | awk '{print $1, $2-" + offset + ", $3-" + offset + ", $4-" + offset + ", $2+" + offset + ", $3+" + offset + ", $4+" + offset + ", " + str(args.reflevel) + "}' > " + halo_track_file
    # the -n +2 option is to skip the FIRST line of the center track file, which holds the column names, because halo track file cannot take column names

    print('Executing command:', command, '\n')
    ret = subprocess.call(command, shell=True)

    print('Saved ' + halo_track_file)
