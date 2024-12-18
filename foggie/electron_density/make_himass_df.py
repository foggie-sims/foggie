#!/usr/bin/env python3

"""

    Title :      make_himass_df
    Notes :      Make a dataframe with HI mass, stellar mass, SFR, etc for a set of given FOGGIE snapshots
    Output :     pandas dataframe as txt file
    Author :     Ayan Acharyya
    Started :    Nov 2024
    Examples :   run make_himass_df.py --system ayan_hd --Zgrad_den kpc --upto_kpc 10 --forpaper --halo 8508,5036,5016,4123,2878,2392 --output RD0030,RD0042 --write_file
                 run make_himass_df.py --system ayan_hd --Zgrad_den kpc --upto_kpc 10 --forpaper --halo 8508 --output RD0042 --write_file
"""
from header import *
from util import *

start_time = time.time()

# -----main code-----------------
if __name__ == '__main__':
    # ----------determining snapshots and file names--------------
    dummy_args_tuple = parse_args('8508', 'RD0042')  # default simulation to work upon when comand line args not provided
    if type(dummy_args_tuple) is tuple: dummy_args = dummy_args_tuple[0] # if the sim has already been loaded in, in order to compute the box center (via utils.pull_halo_center()), then no need to do it again
    else: dummy_args = dummy_args_tuple
    if not dummy_args.keep: plt.close('all')

    if dummy_args.do_all_sims:
        list_of_sims = []
        for dummy_args.halo in dummy_args.halo_arr: list_of_sims.extend(get_all_sims_for_this_halo(dummy_args)) # all snapshots of this particular halo
    else:
        list_of_sims = list(itertools.product(dummy_args.halo_arr, dummy_args.output_arr))
    total_snaps = len(list_of_sims)

    if dummy_args.forpaper:
        dummy_args.docomoving = True
        dummy_args.use_density_cut = True
        dummy_args.weight = 'mass'

    # -------set up dataframe and filename to store/write gradients in to--------
    if dummy_args.write_file:
        cols_in_df = ['halo', 'output', 'redshift', 'time', 'log_sfr', 'log_stellar_mass', 'log_hi_mass', 'log_hii_mass', 'log_h2_mass']
        df = pd.DataFrame(columns=cols_in_df)
        density_cut_text = '_wdencut' if dummy_args.use_density_cut else ''
        if dummy_args.upto_kpc is not None:
            upto_text = '_upto%.1Fckpchinv' % dummy_args.upto_kpc if dummy_args.docomoving else '_upto%.1Fkpc' % dummy_args.upto_kpc
        else:
            upto_text = '_upto%.1FRe' % dummy_args.upto_re
        outfiledir = '/Users/acharyya/Library/CloudStorage/GoogleDrive-ayan.acharyya@inaf.it/My Drive/FOGGIE-Curtin/data/'
        if not os.path.exists(outfiledir): outfiledir = dummy_args.output_dir.replace(dummy_args.halo, '8508') + 'txtfiles/'
        outfilename = outfiledir + 'FOGGIE_baryonic_mass%s%s.txt' % (upto_text, density_cut_text)

        if os.path.isfile(outfilename) and not dummy_args.clobber: # if gradfile already exists
            existing_df = pd.read_table(outfilename)
            existing_halo_outputs = existing_df['halo'].astype(str) + '-' + existing_df['output'].astype(str)

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

    # ---------------starting loop over snapshots-----------------------------------------------
    print_mpi('Operating on snapshots ' + str(core_start + 1) + ' to ' + str(core_end + 1) + ', i.e., ' + str(core_end - core_start + 1) + ' out of ' + str(total_snaps) + ' snapshots', dummy_args)

    for index in range(core_start + dummy_args.start_index, core_end + 1):
        this_sim = list_of_sims[index]
        if 'existing_halo_outputs' in locals() and this_sim[0] + '-' + this_sim[1] in existing_halo_outputs.values:
            print_mpi('Skipping ' + this_sim[0] + '-' + this_sim[1] + ' because it already exists in file', dummy_args)
            continue # skip if this output has already been done and saved on file

        # -------reading in snapshot--------
        start_time_this_snapshot = time.time()
        print_mpi('Doing snapshot ' + this_sim[1] + ' of halo ' + this_sim[0] + ' which is ' + str(index + 1 - core_start) + ' out of the total ' + str(core_end - core_start + 1) + ' snapshots...', dummy_args)
        halos_df_name = dummy_args.code_path + 'halo_infos/00' + this_sim[0] + '/' + dummy_args.run + '/'
        halos_df_name += 'halo_cen_smoothed' if dummy_args.use_cen_smoothed else 'halo_c_v'
        try:
            if len(list_of_sims) == 1 and not dummy_args.do_all_sims: args = dummy_args_tuple # since parse_args() has already been called and evaluated once, no need to repeat it
            else: args = parse_args(this_sim[0], this_sim[1])

            if type(args) is tuple:
                args, ds, refine_box = args  # if the sim has already been loaded in, in order to compute the box center (via utils.pull_halo_center()), then no need to do it again
                print_mpi('ds ' + str(ds) + ' for halo ' + str(this_sim[0]) + ' was already loaded at some point by utils; using that loaded ds henceforth', args)
            else:
                ds, refine_box = load_sim(args, region='refine_box', do_filter_particles=True, disk_relative=False, halo_c_v_name=halos_df_name)
        except Exception as e:
            print_mpi('Skipping ' + this_sim[1] + ' because ' + str(e), dummy_args)
            continue

        # -------determining arg attributes for a snapshot--------
        if args.fortalk:
            setup_plots_for_talks()
            args.forpaper = True
        if args.forpaper:
            args.docomoving = True
            args.use_density_cut = True
            args.weight = 'mass'

        args.current_redshift = ds.current_redshift
        args.current_time = ds.current_time.in_units('Gyr').tolist()

        if args.upto_kpc is not None:
            args.re = np.nan
            if args.docomoving: args.galrad = args.upto_kpc / (1 + args.current_redshift) / 0.695 # fit within a fixed comoving kpc h^-1, 0.695 is Hubble constant
            else: args.galrad = args.upto_kpc # fit within a fixed physical kpc
        else:
            args.re = get_re_from_stars(ds, args)
            args.galrad = args.re * args.upto_re  # kpc

        # ---------extracting the required box--------------------
        box_center = ds.halo_center_kpc
        box = ds.sphere(box_center, ds.arr(args.galrad, 'kpc'))

        # -------read in sfr---------------
        sfr_filename = args.code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/sfr'
        if os.path.exists(sfr_filename):
            print('Reading SFR history from', sfr_filename)
        else:
            print(sfr_filename, 'not found')
            sfr_filename = sfr_filename.replace(args.run, args.run[:14])
            print('Instead, reading SFR history from', sfr_filename)

        sfr_df = pd.read_table(sfr_filename, names=('output', 'redshift', 'sfr'), comment='#', delim_whitespace=True)
        log_sfr = np.log10(sfr_df[sfr_df['output'] == args.output]['sfr'].values[0])

        # --------computing masses----------------------
        stellar_masses = box[('stars', 'particle_mass')].in_units('Msun').ndarray_view()
        hi_masses = box[('gas', 'H_p0_mass')].in_units('Msun').ndarray_view()
        hii_masses = box[('gas', 'H_p1_mass')].in_units('Msun').ndarray_view()
        h2_masses = (box[('gas', 'mean_molecular_weight')] * box[('gas', 'H_p1_number_density')] * box[('gas', 'cell_volume')] * mass_proton).in_units('Msun').ndarray_view()

        log_stellar_mass = np.log10(np.sum(stellar_masses))
        log_hi_mass = np.log10(np.sum(hi_masses))
        log_hii_mass = np.log10(np.sum(hii_masses))
        log_h2_mass = np.log10(np.sum(h2_masses))

        thisrow = [args.halo, args.output, args.current_redshift, args.current_time, log_sfr, log_stellar_mass, log_hi_mass, log_hii_mass, log_h2_mass] # row corresponding to this snapshot to append to df
        this_df = pd.DataFrame({k: [v] for k, v in zip(cols_in_df, thisrow)})

        # ---------writing df to txt file------------------
        if args.write_file:
            if not os.path.isfile(outfilename):
                this_df.to_csv(outfilename, sep='\t', index=None, header='column_names')
                print(f'Written baryon masses df at {outfilename}')
            else:
                this_df.to_csv(outfilename, sep='\t', mode='a', index=False, header=False)
                print('Appended baryon masses to file', outfilename)

        print_mpi('This snapshots completed in %s mins' % ((time.time() - start_time_this_snapshot) / 60), dummy_args)

    df = pd.read_table(outfilename)

    if ncores > 1: print_master('Parallely: time taken for ' + str(total_snaps) + ' snapshots with ' + str(ncores) + ' cores was %s mins' % ((time.time() - start_time) / 60), dummy_args)
    else: print_master('Serially: time taken for ' + str(total_snaps) + ' snapshots with ' + str(ncores) + ' core was %s mins' % ((time.time() - start_time) / 60), dummy_args)
