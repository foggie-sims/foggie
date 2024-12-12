#!/usr/bin/env python3

"""

    Title :      find_halo_center
    Notes :      Find the halo center (with visuals) for a given FOGGIE snapshot
    Output :     Projection plots
    Author :     Ayan Acharyya
    Started :    Aug 2022
    Examples :   run find_halo_center.py --system ayan_pleiades --foggie_dir bigbox --halo 4348 --run natural_7n/25Mpc_DM_256-L3-gas --output RD0111 --width 50 --search_radius 20 --last_center_guess 0.441452,0.586288,0.531448
                 run find_halo_center.py --system ayan_pleiades --foggie_dir bigbox --halo 4348 --run natural_9n/25Mpc_DM_256-L3-gas --output RD0111 --width 50 --search_radius 20 --last_center_guess 0.441906,0.585674,0.530933
                 run find_halo_center.py --system ayan_pleiades --foggie_dir bigbox --halo 8894 --run natural_7n/25Mpc_DM_256-L3-gas --output RD0111 --width 50 --search_radius 20 --last_center_guess 0.542496,0.458481,0.508194
                 run find_halo_center.py --system ayan_pleiades --foggie_dir bigbox --halo 4348 --run natural_7n/25Mpc_DM_256-L3-gas --output RD0111 --width 50 --search_radius 50

"""
from header import *
from util import *
from get_halo_track import *

# -----main code-----------------
if __name__ == '__main__':
    start_time = time.time()
    args = parse_args('8508', 'RD0042')  # default simulation to work upon when comand line args not provided

    # -------------------------parse paths and filenames-------------------------
    if 'natural' not in args.run: args.run = 'natural_' + str(args.reflevel) + 'n/' + args.run
    if args.system == 'ayan_hd' or args.system == 'ayan_local': args.root_dir = '/Users/acharyya/Work/astro/'
    elif args.system == 'ayan_pleiades': args.root_dir = '/nobackup/aachary2/'
    args.output_path = args.root_dir + args.foggie_dir + '/' + 'halo_' + args.halo + '/' + args.run + '/'
    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    center_track_file = args.output_path + 'center_track_sr' + str(args.search_radius) + 'kpc_interp.dat'

    args.fig_dir = args.root_dir + args.foggie_dir + '/' + 'halo_' + args.halo + '/figs/'
    Path(args.fig_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------get approximate halo center at z=2, from L0 gas run halo catalogue combined with offsets-------------------
    if args.last_center_guess is None:
        halos = Table.read('/nobackup/jtumlins/CGM_bigbox/25Mpc_256_shielded-L0/BigBox_z2_rockstar/out_0.list', format='ascii', header_start=0)
        index = [halos['ID'] == int(args.halo[:4])]
        thishalo = halos[index]
        center_L0 = np.array([thishalo['X'][0], thishalo['Y'][0], thishalo['Z'][0]])/25  # divided by 25 comoving Mpc^-1 to convert comoving Mpc h^-1 units to code units

        conf_log_file = args.root_dir + args.foggie_dir + '/' + 'halo_' + args.halo + '/' + args.run + '.conf_log.txt'
        shifts = get_shifts(conf_log_file)
        args.last_center_guess = center_L0 + np.array(shifts) / 255.  # to convert shifts into code units
    else:
        args.last_center_guess = [float(item) for item in args.last_center_guess.split(',')]

    # ------------------------load ds and find center----------------------------------
    print('Doing snapshot ' + args.output + ' of halo ' + args.halo + '...')
    snap_name = args.output_path + args.output + '/' + args.output
    ds = yt.load(snap_name)

    # extract the required quantities
    zz = ds.current_redshift
    search_radius_physical = args.search_radius / (1 + zz) / ds.hubble_constant  # comoving kpc h^-1 to physical kpc
    print('Searching for DM peak within %.3F physical kpc of guessed center = ' % search_radius_physical, args.last_center_guess)
    new_center, vel_center = get_halo_center(ds, args.last_center_guess, radius=search_radius_physical)  # 'radius' requires physical kpc

    # ------------------------projection plot centered at new_center----------------------------------
    projection_plot(ds, new_center, args.last_center_guess, search_radius_physical, args.projection, args)
    print('This snapshots completed in %s mins' % ((time.time() - start_time) / 60))

