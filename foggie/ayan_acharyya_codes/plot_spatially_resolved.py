#!/usr/bin/env python3

"""

    Title :      plot_spatially_resolved.py
    Notes :      Plot spatially resolved relations, profiles at a given resolution, for a given FOGGIE galaxy
    Output :     spatially resolved plots as png
    Author :     Ayan Acharyya
    Started :    Jan 2023
    Examples :   run plot_spatially_resolved.py --system ayan_local --halo 8508 --output RD0030 --upto_kpc 10 --res 0.1 --plot_proj --weight mass --docomoving --proj x
                 run plot_spatially_resolved.py --system ayan_pleiades --halo 8508 --upto_kpc 10 --res 0.1 --do_all_sims --weight mass --use_gasre

"""
from header import *
from util import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
start_time = time.time()

# ----------------------------------------------------
def plot_proj_from_frb(map, args, cmap='viridis', label=None, name='', clim=None):
    '''
    Function to plot projection plot from the given 2D array as input
    '''
    sns.set_style('ticks')  # instead of darkgrid, so that there are no grids overlaid on the projections
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.subplots_adjust(right=0.85, top=0.98, bottom=0.02, left=0.15)

    proj = ax.imshow(map, cmap=cmap, norm=LogNorm(), extent=[-args.galrad, args.galrad, -args.galrad, args.galrad], vmin=clim[0] if clim is not None else None, vmax=clim[1] if clim is not None else None)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(proj, cax=cax)

    ax.set_xlabel(r'x (kpc)', fontsize=args.fontsize)
    ax.set_ylabel(r'y (kpc)', fontsize=args.fontsize)
    ax.set_xticklabels(['%.1F' % item for item in ax.get_xticks()], fontsize=args.fontsize)
    ax.set_yticklabels(['%.1F' % item for item in ax.get_yticks()], fontsize=args.fontsize)

    cbar.ax.tick_params(labelsize=args.fontsize)
    if label is not None: cbar.set_label(label, fontsize=args.fontsize)

    plt.text(0.97, 0.95, 'z = %.2F' % args.current_redshift, ha='right', transform=ax.transAxes, fontsize=args.fontsize)
    plt.text(0.97, 0.9, 't = %.1F Gyr' % args.current_time, ha='right', transform=ax.transAxes, fontsize=args.fontsize)
    plt.show(block=False)

    outfile_rootname = '%s_map_%s%s%s.png' % (args.output, name, args.res_text, args.upto_text)
    if args.do_all_sims: outfile_rootname = 'z=*_' + outfile_rootname[len(args.output)+1:]
    figname = args.fig_dir + outfile_rootname.replace('*', '%.5F' % (args.current_redshift))
    plt.savefig(figname, transparent=False)
    myprint('Saved figure ' + figname, args)

    return fig

# ----------------------------------------------------
def plot_ks_relation(frb, args):
    '''
    Function to plot spatially resolved KS relation
    Requires FRB object as input
    '''
    plt.style.use('seaborn-whitegrid') # instead of ticks, so that grids are overlaid on the plot
    sigma_star_lim = (-1.0, 4.0)
    sigma_gas_lim = (-2.5, 2.5)
    sigma_sfr_lim = (-4.0, 0.5)

    # ----- getting all maps ------------
    map_sigma_star = np.array(frb['deposit', 'stars_density'].in_units('Msun/pc**2')) # stellar mass surface density in Msun/pc^2
    map_sigma_gas = np.array(frb['gas', 'density'].in_units('Msun/pc**2')) # gas mass surface density in Msun/pc^2
    map_sigma_star_young = np.array(frb['deposit', 'young_stars_density'].in_units('Msun/kpc**2')) # young stars mass surface density in Msun/kpc^2
    map_sigma_sfr = map_sigma_star_young / 10e6 # dividing young stars mass by 10 Myr to get SFR surface density in Msun/yr/kpc^2

    # ----- plotting surface maps ----------
    if args.plot_proj:
        cmap = density_color_map
        fig_star = plot_proj_from_frb(map_sigma_star, args, cmap=cmap, label=r'$\Sigma_{\mathrm{star}} (\mathrm{M}_{\odot}/\mathrm{pc}^2)$', name='sigma_star', clim=(10**sigma_star_lim[0], 10**sigma_star_lim[1]))
        fig_gas = plot_proj_from_frb(map_sigma_gas, args, cmap=cmap, label=r'$\Sigma_{\mathrm{gas}} (\mathrm{M}_{\odot}/\mathrm{pc}^2)$', name='sigma_gas', clim=(10**sigma_gas_lim[0], 10**sigma_gas_lim[1]))
        fig_sfr = plot_proj_from_frb(map_sigma_sfr, args, cmap=cmap, label=r'$\Sigma_{\mathrm{SFR}} (\mathrm{M}_{\odot}/\mathrm{yr}/\mathrm{kpc}^2)$', name='sigma_sfr', clim=(10**sigma_sfr_lim[0], 10**sigma_sfr_lim[1]))
    else:
        fig_star, fig_gas, fig_sfr = None, None, None

    # ----- plotting KS relation ------------
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.subplots_adjust(right=0.95, top=0.9, bottom=0.12, left=0.15)

    xdata = np.log10(map_sigma_gas).flatten()
    ydata = np.log10(map_sigma_sfr).flatten()

    xdata = np.ma.compressed(np.ma.masked_array(xdata, ~np.isfinite(ydata)))
    ydata = np.ma.compressed(np.ma.masked_array(ydata, ~np.isfinite(ydata)))
    ax.scatter(xdata, ydata, s=100, lw=0)

    ax.set_xlim(sigma_gas_lim)
    ax.set_ylim(sigma_sfr_lim)

    # ------ fittingthe relation and overplotting ---------
    linefit, linecov = np.polyfit(xdata, ydata, 1, cov=True)
    print('At %.1F kpc resolution, %d out of %d pixels are valid, and KS fit =' % (args.res, len(ydata), len(map_sigma_gas)**2), linefit)
    xarr = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 10)
    ax.plot(xarr, np.poly1d(linefit)(xarr), color='b', lw=2, ls='solid', label=r'Fitted slope = %.1F $\pm$ %.1F' % (linefit[0], np.sqrt(linecov[0][0])))
    ax.plot(xarr, np.poly1d([1.4, -4])(xarr), color='b', lw=2, ls='dashed', label=r'KS relation') # from literature: https://ned.ipac.caltech.edu/level5/March15/Kennicutt/Kennicutt6.html
    ax.legend(loc='lower right', fontsize=args.fontsize)

    ax.set_xlabel(r'$\log{\, \Sigma_{\mathrm{gas}} (\mathrm{M}_{\odot}/\mathrm{pc}^2)}$', fontsize=args.fontsize)
    ax.set_ylabel(r'$\log{\, \Sigma_{\mathrm{SFR}} (\mathrm{M}_{\odot}/\mathrm{yr}/\mathrm{kpc}^2)}$', fontsize=args.fontsize)
    ax.set_xticklabels(['%.1F' % item for item in ax.get_xticks()], fontsize=args.fontsize)
    ax.set_yticklabels(['%.1F' % item for item in ax.get_yticks()], fontsize=args.fontsize)

    plt.text(0.97, 0.35, 'z = %.2F' % args.current_redshift, ha='right', transform=ax.transAxes, fontsize=args.fontsize)
    plt.text(0.97, 0.3, 't = %.1F Gyr' % args.current_time, ha='right', transform=ax.transAxes, fontsize=args.fontsize)
    plt.show(block=False)

    outfile_rootname = '%s_KSrelation%s%s.png' % (args.output, args.res_text, args.upto_text)
    if args.do_all_sims: outfile_rootname = 'z=*_' + outfile_rootname[len(args.output) + 1:]
    figname = args.fig_dir + outfile_rootname.replace('*', '%.5F' % (args.current_redshift))
    plt.savefig(figname, transparent=False)
    myprint('Saved figure ' + figname, args)

    return fig, fig_star, fig_gas, fig_sfr

# -----main code-----------------
if __name__ == '__main__':
    dummy_args_tuple = parse_args('8508', 'RD0042')  # default simulation to work upon when comand line args not provided
    if type(dummy_args_tuple) is tuple:
        dummy_args = dummy_args_tuple[0]  # if the sim has already been loaded in, in order to compute the box center (via utils.pull_halo_center()), then no need to do it again
    else:
        dummy_args = dummy_args_tuple
    if not dummy_args.keep: plt.close('all')

    if dummy_args.do_all_sims:
        list_of_sims = get_all_sims_for_this_halo(dummy_args)  # all snapshots of this particular halo
    else:
        list_of_sims = list(itertools.product([dummy_args.halo], dummy_args.output_arr))
    total_snaps = len(list_of_sims)

    # -------set up dataframe and filename to store/write gradients in to--------
    weightby_text = '' if dummy_args.weight is None else '_wtby_' + dummy_args.weight
    if dummy_args.upto_kpc is not None:
        upto_text = '_upto%.1Fckpchinv' % dummy_args.upto_kpc if dummy_args.docomoving else '_upto%.1Fkpc' % dummy_args.upto_kpc
    else:
        upto_text = '_upto%.1FRe' % dummy_args.upto_re

    # --------domain decomposition; for mpi parallelisation-------------
    comm = MPI.COMM_WORLD
    ncores = comm.size
    rank = comm.rank
    print_master('Total number of MPI ranks = ' + str(ncores) + '. Starting at: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()), dummy_args)
    comm.Barrier()  # wait till all cores reached here and then resume

    split_at_cpu = total_snaps - ncores * int(total_snaps / ncores)
    nper_cpu1 = int(total_snaps / ncores)
    nper_cpu2 = nper_cpu1 + 1
    if rank < split_at_cpu:
        core_start = rank * nper_cpu2
        core_end = (rank + 1) * nper_cpu2 - 1
    else:
        core_start = split_at_cpu * nper_cpu2 + (rank - split_at_cpu) * nper_cpu1
        core_end = split_at_cpu * nper_cpu2 + (rank - split_at_cpu + 1) * nper_cpu1 - 1

    print_mpi('Operating on snapshots ' + str(core_start + 1) + ' to ' + str(core_end + 1) + ', i.e., ' + str(core_end - core_start + 1) + ' out of ' + str(total_snaps) + ' snapshots', dummy_args)

    # ----------------- looping over snapshots ---------------------------------------------
    for index in range(core_start + dummy_args.start_index, core_end + 1):
        start_time_this_snapshot = time.time()
        # ----------- reading in snapshot along with refinebox -------------------
        this_sim = list_of_sims[index]
        print_mpi('Doing snapshot ' + this_sim[1] + ' of halo ' + this_sim[0] + ' which is ' + str(index + 1 - core_start) + ' out of the total ' + str(core_end - core_start + 1) + ' snapshots...', dummy_args)
        halos_df_name = dummy_args.code_path + 'halo_infos/00' + this_sim[0] + '/' + dummy_args.run + '/' + 'halo_cen_smoothed'
        try:
            if len(list_of_sims) == 1 and not dummy_args.do_all_sims:
                args = dummy_args_tuple  # since parse_args() has already been called and evaluated once, no need to repeat it
            else:
                args = parse_args(this_sim[0], this_sim[1])

            if type(args) is tuple:
                args, ds, refine_box = args  # if the sim has already been loaded in, in order to compute the box center (via utils.pull_halo_center()), then no need to do it again
                print_mpi('ds ' + str(ds) + ' for halo ' + str(this_sim[0]) + ' was already loaded at some point by utils; using that loaded ds henceforth', args)
            else:
                ds, refine_box = load_sim(args, region='refine_box', do_filter_particles=True, disk_relative=False, halo_c_v_name=halos_df_name)
        except Exception as e:
            print_mpi('Skipping ' + this_sim[1] + ' because ' + str(e), dummy_args)
            continue

        # ---------- determine certain args parameters ---------------
        args.fig_dir = args.output_dir + 'figs/' if args.do_all_sims else args.output_dir + 'figs/' + args.output + '/'
        Path(args.fig_dir).mkdir(parents=True, exist_ok=True)

        args.current_redshift = ds.current_redshift
        args.current_time = ds.current_time.in_units('Gyr').v

        args.weight_text, args.upto_text = weightby_text, upto_text

        if args.upto_kpc is not None:
            if args.docomoving: args.galrad = args.upto_kpc / (1 + args.current_redshift) / 0.695  # fit within a fixed comoving kpc h^-1, 0.695 is Hubble constant
            else: args.galrad = args.upto_kpc  # fit within a fixed physical kpc
        else:
            args.galrad = args.re * args.upto_re  # kpc

        if args.galrad > 0:
            # extract the required box
            box_center = ds.halo_center_kpc
            box_width_kpc = 2 * args.galrad * kpc  # in kpc
            box = ds.r[box_center[0] - box_width_kpc / 2.: box_center[0] + box_width_kpc / 2., box_center[1] - box_width_kpc / 2.: box_center[1] + box_width_kpc / 2., box_center[2] - box_width_kpc / 2.: box_center[2] + box_width_kpc / 2.]

            dummy_field = ('deposit', 'young_stars_density') # dummy field just to create the FRB; any field can be extracted from the FRB thereafter

            # ---------- creating FRB from refinebox, based on desired resolution ---------------
            for args.res in args.res_arr:
                args.res_text = '_res%.1Fkpc' % args.res
                ncells = int(box_width_kpc / args.res)
                dummy_proj = ds.proj(dummy_field, args.projection, center=box_center, data_source=box)
                frb = dummy_proj.to_frb(box_width_kpc, ncells, center=box_center)

                # ---------- call various plotting routines with the frb ------------
                fig_ks, fig_star, fig_gas, fig_sfr = plot_ks_relation(frb, args)

        print_mpi('This snapshots completed in %s mins' % ((time.time() - start_time_this_snapshot) / 60), dummy_args)


    if ncores > 1: print_master('Parallely: time taken for ' + str(total_snaps) + ' snapshots with ' + str(ncores) + ' cores was %s mins' % ((time.time() - start_time) / 60), dummy_args)
    else: print_master('Serially: time taken for ' + str(total_snaps) + ' snapshots with ' + str(ncores) + ' core was %s mins' % ((time.time() - start_time) / 60), dummy_args)
