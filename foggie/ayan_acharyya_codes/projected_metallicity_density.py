#!/usr/bin/env python3

"""

    Title :      projected_metallicity_density
    Notes :      Plot time evolution of projected density and metallicity along 3 axes
    Output :     Combined plots as png files plus, optionally, these files stitched into a movie
    Author :     Ayan Acharyya
    Started :    Aug 2023
    Examples :   run projected_metallicity_density.py --system ayan_local --halo 8508 --upto_kpc 15 --annotate_box 10 --output DD0238 --docomoving --fontsize 20 --use_cen_smoothed
"""

from header import *
from util import *
from compute_MZgrad import get_density_cut, get_re_from_coldgas, get_re_from_stars
from mpl_toolkits.axes_grid1 import make_axes_locatable
from projection_plot import annotate_box

start_time = time.time()
plt.rcParams['axes.linewidth'] = 1

# ----------------------------------------------------------------
def plot_projected(field, box, box_center, box_width, projection, ax, args, annotate_markers=[]):
    '''
    Function to plot projected field from given dataset box on to a given axis
    :return: handle of the projection plot as well as the axis handle
    '''
    plt.style.use('seaborn-white')
    myprint('Now making ' + field + ' projection plot for ' + projection + '..', args)

    proj = yt.ProjectionPlot(box.ds, projection, field_dict[field], center=box_center, data_source=box, width=box_width * kpc, weight_field=weight_field_dict[field], fontsize=args.fontsize)

    # -----------making the colorbar labels etc--------------
    proj.set_log(field_dict[field], islog_dict[field])
    proj.set_unit(field_dict[field], unit_dict[field])
    proj.set_zlim(field_dict[field], zmin=bounds_dict[field][0], zmax=bounds_dict[field][1])
    cmap = cmap_dict[field]
    cmap.set_bad('k')
    proj.set_cmap(field_dict[field], cmap)

    # --------annotating boxes etc-----------------------
    if args.annotate_box is not None:
        for thisbox in [float(item) for item in args.annotate_box.split(',')]:  # comoving size in kpc
            if args.docomoving: thisbox = thisbox / (1 + ds.current_redshift) / ds.hubble_constant  # physical size at current redshift in kpc
            proj = annotate_box(proj, thisbox, box.ds, args.halo_center, unit='kpc', projection=projection)

    for index,marker in enumerate(annotate_markers):
        print('Marking ' + str(index+1) + ' out of ' + str(len(annotate_markers)) + 'markers:', marker)
        proj.annotate_marker(marker, coord_system='data')#, color=col_arr[index])

    # -----------making the axis labels etc--------------
    position = ax.get_position()
    proj.plots[field_dict[field]].axes = ax
    proj._setup_plots()
    proj.plots[field_dict[field]].axes.set_position(position) # in order to resize the axis back to where it should be
    ax.set_xticks(np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 5))
    ax.set_yticks(np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 5))
    ax.set_xticklabels(['%.1F' % item for item in ax.get_xticks()], fontsize=args.fontsize)
    ax.set_yticklabels(['%.1F' % item for item in ax.get_yticks()], fontsize=args.fontsize)
    ax.text(0.9 * args.galrad, 0.9 * args.galrad, projection, ha='right', va='top', c='w', fontsize=args.fontsize, bbox=dict(facecolor='k', alpha=0.3, edgecolor='k'))

    # ---------making the colorbar axis once, that will correspond to all projections--------------
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(proj.plots[field_dict[field]].cb.mappable, orientation='vertical', cax=cax)
    cbar.ax.tick_params(labelsize=args.fontsize)
    cbar.set_label(proj.plots[field_dict[field]].cax.get_ylabel(), fontsize=args.fontsize)

    return proj, ax

# ----------dicts of colors, limits, etc. --------------------------
field_dict = {'density':('gas', 'density'), 'metal':('gas', 'metallicity')}
unit_dict = {'density': 'Msun/pc**2', 'metal': r'Zsun'}
islog_dict = defaultdict(lambda: False, metal=True, density=True)
weight_field_dict = defaultdict(lambda: None, metal=('gas', 'mass'))
bounds_dict = defaultdict(lambda: (None, None), density=(density_proj_min, density_proj_max), metal=(1e-3, 1e1))
cmap_dict = {'density': density_color_map, 'metal': old_metal_color_map}
col_arr = ['r', 'g', 'b', 'k']

# -----main code-----------------
if __name__ == '__main__':
    args_tuple = parse_args('8508', 'RD0042')  # default simulation to work upon when comand line args not provided
    if type(args_tuple) is tuple: args, ds, refine_box = args_tuple # if the sim has already been loaded in, in order to compute the box center (via utils.pull_halo_center()), then no need to do it again
    else: args = args_tuple
    if not args.keep: plt.close('all')

    fields = ['density', 'metal']
    projections = ['x', 'y', 'z']

    # --------domain decomposition; for mpi parallelisation-------------
    if args.do_all_sims: list_of_sims = get_all_sims_for_this_halo(args) # all snapshots of this particular halo
    else: list_of_sims = list(itertools.product([args.halo], args.output_arr))
    total_snaps = len(list_of_sims)

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

    # -------------loop over snapshots-----------------
    print_mpi('Operating on snapshots ' + str(core_start + 1) + ' to ' + str(core_end + 1) + ', i.e., ' + str(core_end - core_start + 1) + ' out of ' + str(total_snaps) + ' snapshots', args)

    for index in range(core_start + args.start_index, core_end + 1):
        start_time_this_snapshot = time.time()
        this_sim = list_of_sims[index]
        print_mpi('Doing snapshot ' + this_sim[1] + ' of halo ' + this_sim[0] + ' which is ' + str(index + 1 - core_start) + ' out of the total ' + str(core_end - core_start + 1) + ' snapshots...', args)
        halos_df_name = args.code_path + 'halo_infos/00' + this_sim[0] + '/' + args.run + '/'
        halos_df_name += 'halo_cen_smoothed' if args.use_cen_smoothed else 'halo_c_v'

        # -------loading in snapshot-------------------
        try:
            if len(list_of_sims) > 1 or args.do_all_sims: args = parse_args(this_sim[0], this_sim[1])
            if type(args) is tuple: args, ds, refine_box = args  # if the sim has already been loaded in, in order to compute the box center (via utils.pull_halo_center()), then no need to do it again
            else: ds, refine_box = load_sim(args, region='refine_box', do_filter_particles=True, disk_relative=False, halo_c_v_name=halos_df_name)

            args.current_redshift = ds.current_redshift
            args.current_time = ds.current_time.in_units('Gyr').v

            # --------determining corresponding text suffixes-------------
            args.density_cut_text = '_wdencut' if args.use_density_cut else ''

            # -------setting up fig--------------
            nrow, ncol = len(fields), len(projections)
            fig, axes = plt.subplots(nrow, ncol, figsize=(3 + ncol*3, 2 + nrow*2), sharex=True, sharey=True)
            fig.tight_layout()
            fig.subplots_adjust(top=0.97, bottom=0.1, left=0.05, right=0.95, wspace=0.2, hspace=0.05)

            # ------tailoring the simulation box for individual snapshot analysis--------
            if args.upto_kpc is not None: args.re = np.nan
            else: args.re = get_re_from_coldgas(args) if args.use_gasre else get_re_from_stars(ds, args)

            if args.upto_kpc is not None:
                if args.docomoving: args.galrad = args.upto_kpc / (1 + args.current_redshift) / 0.695  # fit within a fixed comoving kpc h^-1, 0.695 is Hubble constant
                else: args.galrad = args.upto_kpc  # fit within a fixed physical kpc
            else:
                args.galrad = args.re * args.upto_re  # kpc

            # extract the required box
            box_center = ds.halo_center_kpc
            box_width = 2 * args.galrad # kpc
            box_width_kpc = ds.arr(box_width, 'kpc')
            box = ds.r[box_center[0] - box_width_kpc / 2.: box_center[0] + box_width_kpc / 2., box_center[1] - box_width_kpc / 2.: box_center[1] + box_width_kpc / 2., box_center[2] - box_width_kpc / 2.: box_center[2] + box_width_kpc / 2., ]

            if args.use_density_cut:
                rho_cut = get_density_cut(ds.current_time.in_units('Gyr'))  # based on Cassi's CGM-ISM density cut-off
                box = box.cut_region(['obj["gas", "density"] > %.1E' % rho_cut])
                print('Imposing a density criteria to get ISM above density', rho_cut, 'g/cm^3')

            # ------plotting projected metallcity and density snapshots---------------
            annotate_markers = [box_center] #
            for i,field in enumerate(fields):
                for j,projection in enumerate(projections):
                    print('Plotting ' + str(i * ncol + j + 1) + ' of ' + str(nrow * ncol) + ' projections..')
                    thisax = axes[i][j] if nrow * ncol > 1 else axes
                    proj, thisax = plot_projected(field, box, ds.halo_center_kpc, box_width, projection, thisax, args, annotate_markers=annotate_markers)
                    thisax.set_xlabel('Offset (kpc)' if i == len(fields) - 1 else '', fontsize=args.fontsize)
                    thisax.set_ylabel('Offset (kpc)' if j == 0 else '', fontsize=args.fontsize)

            # ------saving fig------------------
            if args.upto_kpc is not None: upto_text = '_upto%.1Fckpchinv' % args.upto_kpc if args.docomoving else '_upto%.1Fkpc' % args.upto_kpc
            else: upto_text = '_upto%.1FRe' % args.upto_re

            args.fig_dir = args.output_dir + 'figs/'
            if not args.do_all_sims: args.fig_dir += args.output + '/'
            Path(args.fig_dir).mkdir(parents=True, exist_ok=True)

            outfile_rootname = '%s_%s_projected_met_den_%s.png' % (args.output, args.halo, upto_text)
            if args.do_all_sims: outfile_rootname = 'z=*_' + outfile_rootname[len(args.output) + 1:]
            figname = args.fig_dir + outfile_rootname.replace('*', '%.5F' % (args.current_redshift))

            fig.savefig(figname)
            myprint('Saved plot as ' + figname, args)

            plt.show(block=False)
            print_mpi('This snapshots completed in %s mins' % ((time.time() - start_time_this_snapshot) / 60), args)
        except Exception as e:
            print_mpi('Skipping ' + this_sim[1] + ' because ' + str(e), args)
            continue

    if ncores > 1: print_master('Parallely: time taken for ' + str(total_snaps) + ' snapshots with ' + str(ncores) + ' cores was %s mins' % ((time.time() - start_time) / 60), args)
    else: print_master('Serially: time taken for ' + str(total_snaps) + ' snapshots with ' + str(ncores) + ' core was %s mins' % ((time.time() - start_time) / 60), args)