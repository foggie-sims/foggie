##!/usr/bin/env python3

"""

    Title :      filter_star_properties
    Notes :      To extract physial properties of young (< 10Myr) stars e.g., position, velocity, mass etc. and output to an ASCII file
    Output :     One pandas dataframe as a txt file
    Author :     Ayan Acharyya
    Started :    January 2021
    Example :    run filter_star_properties.py --system ayan_local --halo 8508 --output RD0042

"""
from header import *
from util import *
from foggie.galaxy_mocks.mock_ifu.compute_hiir_radii import get_radii_for_df
#from projection_plot import make_projection_plots

# ----------------------------------------------------------------------------------
def get_star_properties(args):
    '''
    Function to filter properties of young stars
    :return: dataframe (and writes dataframe as ASCII file too
    '''
    start_time = time.time()
    outfilename = args.output_dir + 'txtfiles/' + args.output + '_young_star_properties.txt'
    Path(args.output_dir + 'txtfiles/').mkdir(parents=True, exist_ok=True)

    # ----------------------Reading in simulation data-------------------------------------------
    if not os.path.exists(outfilename) or args.clobber:
        if not os.path.exists(outfilename):
            myprint(outfilename + ' does not exist. Creating afresh..', args)
        elif args.clobber:
            myprint(outfilename + ' exists but over-writing..', args)

        ds, refine_box = load_sim(args, region='refine_box')
        ad = ds.all_data()

        if args.plot_proj:
            myprint('Will execute make_projection_plots() for ' + args.output + '...', args)
            prj = make_projection_plots(ds=refine_box.ds, center=ds.halo_center_kpc, \
                                        refine_box=refine_box, x_width=ds.refine_width * kpc, \
                                        fig_dir=args.output_dir + 'figs/', haloname=args.output,
                                        name=halo_dict[args.halo], \
                                        fig_end='projection', do=[ar for ar in args.do.split(',')],
                                        axes=[ar for ar in args.proj.split(',')], is_central=True, add_arrow=False,
                                        add_velocity=False)  # using halo_center_kpc instead of refine_box_center

        xgrid = ad['young_stars', 'particle_position_x']
        myprint('Extracting parameters for ' + str(len(xgrid)) + ' young stars...', args)
        zgrid = ad['young_stars', 'particle_position_z']
        ygrid = ad['young_stars', 'particle_position_y']

        px = xgrid.in_units('kpc')
        py = ygrid.in_units('kpc')
        pz = zgrid.in_units('kpc')

        vx = ad['young_stars', 'particle_velocity_x'].in_units('km/s')
        vy = ad['young_stars', 'particle_velocity_y'].in_units('km/s')
        vz = ad['young_stars', 'particle_velocity_z'].in_units('km/s')

        age = ad['young_stars', 'age'].in_units('Myr')
        mass = ad['young_stars', 'particle_mass'].in_units('Msun')

        coord = np.vstack([xgrid, ygrid, zgrid]).transpose()
        # ambient gas properties only at point where young stars are located:
        pres = ds.find_field_values_at_points([('gas', 'pressure')], coord)
        den = ds.find_field_values_at_points([('gas', 'density')], coord)
        temp = ds.find_field_values_at_points([('gas', 'temperature')], coord)
        Z = ds.find_field_values_at_points([('gas', 'metallicity')], coord)

        # saving the header (with units, etc.) first in a new txt file
        header = 'Units for the following columns: \n\
        pos_x, pos_y, pos_z: kpc \n\
        vel_x, vel_y, vel_z: km/s \n\
        age: Myr \n\
        mass: Msun \n\
        gas_density in a cell: ' + ds.field_info[('gas', 'density')].units + ' \n\
        gas_pressure in a cell: ' + ds.field_info[('gas', 'pressure')].units + ' \n\
        gas_temp in a cell: ' + ds.field_info[('gas', 'temperature')].units + ' \n\
        gas_metal in a cell: ' + ds.field_info[('gas', 'metallicity')].units
        np.savetxt(outfilename, [], header=header, comments='#')

        # creating and saving the dataframe itself to the file which already has the header
        paramlist = pd.DataFrame(
            {'pos_x': px, 'pos_y': py, 'pos_z': pz, 'vel_x': vx, 'vel_y': vy, 'vel_z': vz, 'age': age, 'mass': mass,
             'gas_density': den, 'gas_pressure': pres, 'gas_temp': temp, 'gas_metal': Z})
        paramlist.to_csv(outfilename, sep='\t', mode='a', index=None)
        myprint('Saved file at ' + outfilename, args)
    else:
        myprint('Reading from existing file ' + outfilename, args)
        paramlist = pd.read_table(outfilename, delim_whitespace=True, comment='#')

    myprint(args.output + ' completed in %s minutes' % ((time.time() - start_time) / 60), args)
    if args.automate:
        myprint('Will execute get_radii_for_df() for ' + args.output + '...', args)
        paramlist = get_radii_for_df(paramlist, args)

    return paramlist

# -----------------------------------------------------------------------------------
if __name__ == '__main__':
    start_time = time.time()

    dummy_args_tuple = parse_args('8508', 'RD0042')  # default simulation to work upon when comand line args not provided
    if type(dummy_args_tuple) is tuple: dummy_args = dummy_args_tuple[0] # if the sim has already been loaded in, in order to compute the box center (via utils.pull_halo_center()), then no need to do it again
    else: dummy_args = dummy_args_tuple
    if not dummy_args.keep: plt.close('all')

    if dummy_args.do_all_sims:
        list_of_sims = get_all_sims(dummy_args) # all snapshots of this particular halo
    else:
        if dummy_args.do_all_halos: halos = get_all_halos(dummy_args)
        else: halos = dummy_args.halo_arr
        list_of_sims = list(itertools.product(halos, dummy_args.output_arr))
    total_snaps = len(list_of_sims)

    # --------domain decomposition; for mpi parallelisation-------------
    comm = MPI.COMM_WORLD
    ncores = comm.size
    rank = comm.rank
    print_master('Total number of MPI ranks = ' + str(ncores) + '. Starting at: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()), dummy_args)
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
    print_mpi('Operating on snapshots ' + str(core_start + 1) + ' to ' + str(core_end + 1) + 'i.e., ' + str(core_end - core_start + 1) + ' out of ' + str(total_snaps) + ' snapshots', dummy_args)

    for index in range(core_start, core_end + 1):
        this_sim = list_of_sims[index]
        print_mpi('Doing snapshot ' + this_sim[1] + ' of halo ' + this_sim[0] + ' which is ' + str(index + 1 - core_start) + ' of the ' + str(core_end - core_start + 1) + ' snapshots alloted to this core...', dummy_args)
        try:
            if len(list_of_sims) == 1: args = dummy_args_tuple # since parse_args() has already been called and evaluated once, no need to repeat it
            else: args = parse_args(this_sim[0], this_sim[1])

            if type(args) is tuple:
                args, ds, refine_box = args  # if the sim has already been loaded in, in order to compute the box center (via utils.pull_halo_center()), then no need to do it again
                print_mpi('ds ' + str(ds) + ' for halo ' + str(this_sim[0]) + ' was already loaded at some point by utils; using that loaded ds henceforth', args)

            if args.dryrun: print_mpi('Skipping main computation because this is a dryrun.', args)
            else: paramlist = get_star_properties(args)

        except (FileNotFoundError, PermissionError) as e:
            print_mpi('Skipping ' + this_sim[1] + ' because ' + str(e), dummy_args)
            total_snaps = total_snaps - 1
            continue

    comm.Barrier() # wait till all cores reached here and then resume
    if ncores > 1: print_master('Parallely: time taken for filtering ' + str(total_snaps) + ' snapshots with ' + str(ncores) + ' cores was %s mins' % ((time.time() - start_time) / 60), dummy_args)
    else: print_master('Serially: time taken for filtering ' + str(total_snaps) + ' snapshots with ' + str(ncores) + ' core was %s mins' % ((time.time() - start_time) / 60), dummy_args)
