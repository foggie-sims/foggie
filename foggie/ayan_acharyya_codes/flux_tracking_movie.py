#!/usr/bin/env python3

"""

    Title :      flux_tracking_movie
    Notes :      Run Cassi's foggie/flux_tracking/flux_tracking.calculate_fluxes() code to track metal mass over time and make movies
    Output :     flux profile plots as png files (which can be later converted to a movie via animate_png.py)
    Author :     Ayan Acharyya
    Started :    August 2021
    Examples :   run flux_tracking_movie.py --system ayan_local --halo 8508 --galrad 20 --units_kpc --overplot_stars --output RD0042 --clobber_plot
                 run flux_tracking_movie.py --system ayan_local --halo 8508 --galrad 20 --units_kpc --overplot_stars --do_all_sims --makemovie --delay 0.05
"""
from header import *
from util import *
from flux_tracking_cassi import *
from make_ideal_datacube import shift_ref_frame
from filter_star_properties import get_star_properties
from datashader_movie import get_radial_velocity
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ---------------------------------------------------------------------------------
def overplot_stars(ax, args):
    '''
    Function to overplot young stars on existing plot
    '''
    starlistfile = args.output_dir + 'txtfiles/' + args.output + '_young_star_properties.txt'

    # -------------to read in simulation data------------
    if not os.path.exists(starlistfile):
        print_mpi(starlistfile + 'does not exist. Calling get_star_properties() first..', args)
        dummy = get_star_properties(args)  # this creates the starlistfile
    paramlist = pd.read_table(starlistfile, delim_whitespace=True, comment='#')

    # -------------to prep the simulation data------------
    paramlist = shift_ref_frame(paramlist, args)
    paramlist = paramlist.rename(columns={'gas_metal': 'metal', 'gas_density': 'density', 'gas_pressure': 'pressure', 'gas_temp': 'temp'})
    paramlist = get_radial_velocity(paramlist)
    paramlist = paramlist[paramlist['rad'].between(0, args.galrad)] # to overplot only those young stars that are within the desired radius ('rad' is in kpc)

    # -------------to actually plot the simulation data------------
    im = ax.scatter(paramlist['rad'], paramlist['metal'], c=np.log10(paramlist['mass']), vmin=2.8, vmax=4.0, edgecolors='black', lw=0.2, s=15, cmap='YlGn_r')
    print_mpi('Overplotted ' + str(len(paramlist)) + 'young star particles', args)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    ax.figure.colorbar(im, cax=cax, orientation='vertical')
    cax.set_ylabel(r'Log Mass (M/M$_{\odot}$)', fontsize=args.fontsize)
    cax.set_yticklabels(['%.1F'%item for item in cax.get_yticks()], fontsize=args.fontsize)

    return ax

# -----------------------------------------------------
def make_flux_plot(table_name, fig_name, args):
    '''
    Function to read in hdf5 files created by Cassi's flux_tracking.py and plot the 'metallicity flux' vs radius,
    where 'metallicity flux' is the metal_mass/gas_mass of the inflowing OR outflowing material
    '''
    df = pd.read_hdf(table_name, key='all_data') # read in table

    # ------create new columns for 'metallicity'--------
    df['net_Z_flux'] = df['net_metal_flux'] / df['net_mass_flux']
    df['Z_flux_in'] = df['metal_flux_in'] / df['mass_flux_in']
    df['Z_flux_out'] = df['metal_flux_out'] / df['mass_flux_out']

    quant_arr = ['mass', 'metal', 'Z']
    ylabel_arr = [r'Gas mass flux (M$_{\odot}$/yr)', r'Metal mass flux (M$_{\odot}$/yr)', r'Metallicity flux (Z/Z$_{\odot}$)']
    ylim_arr = [(-25, 110), (-0.5, 2.2), (-0.01, 0.044)]

    # --------plot radial profiles----------------
    fig, axes = plt.subplots(1, 3, figsize=(14,5))
    extra_space = 0.03 if not args.overplot_stars else 0
    plt.subplots_adjust(hspace=0.05, wspace=0.25, right=0.94 + extra_space, top=0.95, bottom=0.12, left=0.07)

    for index, ax in enumerate(axes):
        ax.axhline(0, c='k', ls='--', lw=0.5)
        ax.plot(df['radius'], df[quant_arr[index] + '_flux_in'], c='cornflowerblue', label='Incoming')
        ax.plot(df['radius'], df[quant_arr[index] + '_flux_out'], c='sienna', label='Outgoing')
        ax.plot(df['radius'], df['net_' + quant_arr[index] + '_flux'], c='gray', alpha=0.5, label='Net flux')

        # ----------to overplot young stars----------------
        if args.overplot_stars and quant_arr[index] == 'Z': ax = overplot_stars(ax, args)

        if index == 0: ax.legend(fontsize=args.fontsize)
        ax.set_xlim(0, args.galrad)
        ax.set_xlabel('Radius (kpc)', fontsize=args.fontsize)
        ax.set_xticks(np.linspace(0, args.galrad, 5))
        ax.set_xticklabels(['%.1F'%item for item in ax.get_xticks()], fontsize=args.fontsize)

        ax.set_ylim(ylim_arr[index])
        ax.set_ylabel(ylabel_arr[index], fontsize=args.fontsize)
        ax.set_yticklabels(['%.2F'%item if quant_arr[index] == 'Z' else '%.1F'%item for item in ax.get_yticks()], fontsize=args.fontsize)

    ax.text(0.95, 0.97, 'z = %.4F' % args.current_redshift, transform=ax.transAxes, fontsize=args.fontsize, ha='right', va='top')
    ax.text(0.95, 0.9, 't = %.3F Gyr' % args.current_time, transform=ax.transAxes, fontsize=args.fontsize, ha='right', va='top')

    plt.savefig(fig_name, transparent=False)
    myprint('Saved figure ' + fig_name, args)
    if not args.do_all_sims: plt.show(block=False)

    return df, fig

# -----main code-----------------
if __name__ == '__main__':
    start_time = time.time()
    dummy_args_tuple = parse_args('8508', 'RD0042')  # default simulation to work upon when comand line args not provided
    if type(dummy_args_tuple) is tuple: dummy_args = dummy_args_tuple[0] # if the sim has already been loaded in, in order to compute the box center (via utils.pull_halo_center()), then no need to do it again
    else: dummy_args = dummy_args_tuple
    if not dummy_args.keep: plt.close('all')

    if dummy_args.do_all_sims: list_of_sims = get_all_sims_for_this_halo(dummy_args) # all snapshots of this particular halo
    else: list_of_sims = [(dummy_args.halo, dummy_args.output)]
    total_snaps = len(list_of_sims)

    # parse paths and filenames
    fig_dir = dummy_args.output_dir + 'figs/' if dummy_args.do_all_sims else dummy_args.output_dir + 'figs/' + dummy_args.output + '/'
    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    table_dir = dummy_args.output_dir + 'txtfiles/'
    Path(table_dir).mkdir(parents=True, exist_ok=True)

    dt = 5.38e6

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
    print_mpi('Operating on snapshots ' + str(core_start + 1) + ' to ' + str(core_end + 1) + ', i.e., ' + str(core_end - core_start + 1) + ' out of ' + str(total_snaps) + ' snapshots', dummy_args)

    for index in range(core_start + dummy_args.start_index, core_end + 1):
        start_time_this_snapshot = time.time()
        this_sim = list_of_sims[index]
        print_mpi('Doing snapshot ' + this_sim[1] + ' of halo ' + this_sim[0] + ' which is ' + str(index + 1 - core_start) + ' out of the total ' + str(core_end - core_start + 1) + ' snapshots...', dummy_args)
        try:
            if dummy_args.do_all_sims: args = parse_args(this_sim[0], this_sim[1])
            else: args = dummy_args_tuple # since parse_args() has already been called and evaluated once, no need to repeat it

            if type(args) is tuple:
                args, ds, refine_box = args  # if the sim has already been loaded in, in order to compute the box center (via utils.pull_halo_center()), then no need to do it again
                print_mpi('ds ' + str(ds) + ' for halo ' + str(this_sim[0]) + ' was already loaded at some point by utils; using that loaded ds henceforth', args)
            else:
                ds, refine_box = load_sim(args, region='refine_box', do_filter_particles=False)

        except (FileNotFoundError, PermissionError) as e:
            print_mpi('Skipping ' + this_sim[1] + ' because ' + str(e), dummy_args)
            continue

        refine_width_kpc = ds.arr(ds.refine_width, 'kpc')
        args.current_redshift = ds.current_redshift
        args.current_time = ds.current_time.in_units('Gyr')

        # ----------generating the flux tracking tables----------
        table_root = table_dir + args.output + '_fluxes'
        table_name = table_root + '_mass.hdf5'
        if not os.path.exists(table_name) or args.clobber:
            if not os.path.exists(table_name):
                print_mpi(table_name + ' does not exist. Creating afresh..', args)
            elif args.clobber:
                print_mpi(table_name + ' exists but over-writing..', args)

            message = calc_fluxes(ds, args.output, args.current_redshift, dt, refine_width_kpc, table_root, '', [['sphere', 0.01, args.galrad, 100]], ['mass'], 0, args)
            # inner and outer radii of sphere are by default as a fraction of refine_box_width, unless --units_kpc is specified in user args, in whic case they are in kpc
        else:
            print_mpi('Skipping ' + table_name + ' because file already exists (use --clobber to over-write)', args)

        # ----------plotting the tracked flux----------
        outfile_rootname = 'metal_flux_profile_boxrad_%.2Fkpc.png' % (args.galrad)
        if args.do_all_sims: outfile_rootname = 'z=*_' + outfile_rootname
        fig_name = fig_dir + outfile_rootname.replace('*', '%.5F' % (args.current_redshift))

        if not os.path.exists(fig_name) or args.clobber_plot:
            if not os.path.exists(fig_name):
                print_mpi(fig_name + ' plot does not exist. Creating afresh..', args)
            elif args.clobber_plot:
                print_mpi(fig_name + ' plot exists but over-writing..', args)

            df, fig = make_flux_plot(table_name, fig_name, args)
        else:
            print_mpi('Skipping ' + fig_name + ' because plot already exists (use --clobber_plot to over-write)', args)

        print_mpi('This snapshot ' + this_sim[1] + ' completed in %s' % (datetime.timedelta(seconds=time.time() - start_time_this_snapshot)), args)
    comm.Barrier() # wait till all cores reached here and then resume

    if args.makemovie and args.do_all_sims:
        print_master('Finished creating snapshots, calling animate_png.py to create movie..', args)
        subprocess.call(['python ' + HOME + '/Work/astro/ayan_codes/animate_png.py --inpath ' + fig_dir + ' --rootname ' + outfile_rootname + ' --delay ' + str(args.delay_frame) + ' --reverse'], shell=True)

    if ncores > 1: print_master('Parallely: %d snapshots completed in %s using %d cores' % (total_snaps, datetime.timedelta(seconds=time.time() - start_time), ncores), dummy_args)
    else: print_master('Serially: %d snapshots completed in %s using %d core' % (total_snaps, datetime.timedelta(seconds=time.time() - start_time), ncores), dummy_args)
