#!/usr/bin/env python3

"""

    Title :      make_multiple_movies.py
    Notes :      Make multiple movies from existing datashader plots, in parallel (using animate_png.py)
    Output :     time evolution movies as mp4 files
    Author :     Ayan Acharyya
    Started :    August 2021
    Examples :   mpirun -n 10 python make_multiple_movies.py --system ayan_pleiades --halo 4123,2392 --galrad 20 --xcol rad --ycol metal,temp --colorcol vrad,density,temp,metal,phi_L,theta_L --delay 0.05
"""
from header import *
from util import *
start_time = time.time()

# -----main code-----------------
if __name__ == '__main__':
    islog_dict = defaultdict(lambda: False, metal=True, density=True, temp=True)

    args_tuple = parse_args('8508', 'RD0042')  # default simulation to work upon when comand line args not provided
    if type(args_tuple) is tuple: args = args_tuple[0] # if the sim has already been loaded in, in order to compute the box center (via utils.pull_halo_center()), then no need to do it again
    else: args = args_tuple

    if args.do_all_halos: halos_arr = get_all_halos(args)
    else: halos_arr = args.halo_arr

    xcol_arr = [item for item in args.xcol.split(',')]
    ycol_arr = [item for item in args.ycol.split(',')]
    colorcol_arr = [item for item in args.colorcol.split(',')]
    movies = list(itertools.product(xcol_arr, ycol_arr, colorcol_arr, halos_arr))
    list_of_movies = [item for item in movies if len(item) == len(set(item))] # removing cases where x, y or color axes have same quantities
    nmovies = len(list_of_movies)

    # --------domain decomposition; for mpi parallelisation-------------
    comm = MPI.COMM_WORLD
    ncores = comm.size
    rank = comm.rank
    print_master('Total number of MPI ranks = ' + str(ncores) + '. Starting at: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()), args)
    comm.Barrier() # wait till all cores reached here and then resume

    split_at_cpu = nmovies - ncores * int(nmovies/ncores)
    nper_cpu1 = int(nmovies / ncores)
    nper_cpu2 = nper_cpu1 + 1
    if rank < split_at_cpu:
        core_start = rank * nper_cpu2
        core_end = (rank+1) * nper_cpu2 - 1
    else:
        core_start = split_at_cpu * nper_cpu2 + (rank - split_at_cpu) * nper_cpu1
        core_end = split_at_cpu * nper_cpu2 + (rank - split_at_cpu + 1) * nper_cpu1 - 1

    # --------------------------------------------------------------
    print_mpi('Operating on movies ' + str(core_start + 1) + ' to ' + str(core_end + 1) + ', i.e., ' + str(core_end - core_start + 1) + ' out of ' + str(nmovies) + ' movies', args)
    
    for index in range(core_start, core_end + 1):
        start_time_this_snapshot = time.time()
        this_xcol, this_ycol, this_colorcol, this_halo = list_of_movies[index]
        print_mpi('Doing movie ' + this_xcol + 'vs' + this_ycol + 'colored by' + this_colorcol + 'for halo ' + this_halo + ' which is ' + str(index + 1 - core_start) + ' out of the total ' + str(core_end - core_start + 1) + ' movies...', args)

        fig_dir = args.output_dir + 'figs/'
        if islog_dict[this_xcol]: this_xcol = 'log_' + this_xcol
        if islog_dict[this_ycol]: this_ycol = 'log_' + this_ycol
        if islog_dict[this_colorcol]: this_colorcol = 'log_' + this_colorcol

        outfile_rootname = 'z=*_datashader_boxrad_%.2Fkpc_%s_vs_%s_colby_%s.png' % (args.galrad, this_ycol, this_xcol, this_colorcol)
        subprocess.call(['python ' + HOME + '/Work/astro/ayan_codes/animate_png.py --inpath ' + fig_dir + ' --rootname ' + outfile_rootname + ' --delay ' + str(args.delay_frame) + ' --reverse'], shell=True)

    comm.Barrier() # wait till all cores reached here and then resume

    if ncores > 1: print_master('Parallely: time taken for making ' + str(nmovies) + ' movies with ' + str(ncores) + ' cores was %s mins' % ((time.time() - start_time) / 60), args)
    else: print_master('Serially: time taken for making ' + str(nmovies) + ' movies with ' + str(ncores) + ' core was %s mins' % ((time.time() - start_time) / 60), args)

