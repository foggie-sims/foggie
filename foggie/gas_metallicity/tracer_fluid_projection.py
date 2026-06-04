#!/usr/bin/env python3

"""
    Title :      tracer_fluid_projection.py
    Notes :      Plot tracer fluid projections of a given list of filenames (FOGGIE snapshots)
    Output :     .png plots
    Author :     Ayan Acharyya
    Started :    04-06-2026
    Examples :   run tracer_fluid_projection.py --system ayan_local --halo 8508 --output RD0027 --upto_kpc 20
"""
from header import *
from util import *
setup_plot_style()
from projection_plot import make_projection_plots
start_time = datetime.now()
    
# -----main code-----------------
if __name__ == '__main__':
    args = parse_args()  # default simulation to work upon when comand line args not provided
    if not args.keep: plt.close('all')

    # ----------determine list of snapshots--------
    if args.do_all_sims:
        list_of_sims = get_all_sims(args) # all snapshots of this particular halo
    else:
        if args.do_all_halos: halos = get_all_halos(args)
        else: halos = args.halo_arr
        list_of_sims = list(itertools.product(halos, args.output_arr))

    # ------loop over snapshots-------------
    for index, this_sim in enumerate(list_of_sims):
        print('Doing', index + 1, 'out of the total %s sims..' % (len(list_of_sims)))

        halos_df_name = args.code_path + 'halo_infos/00' + this_sim[0] + '/' + args.run + '/'
        halos_df_name += 'halo_cen_smoothed' if args.use_cen_smoothed else 'halo_c_v'
        ds, refine_box = load_sim(args, region='refine_box', do_filter_particles=False, disk_relative=False, halo_c_v_name=halos_df_name)

        # --------------tailoring the extent of the box------------------------
        if args.upto_kpc is not None: args.re = np.nan
        else: args.re = get_re_from_coldgas(args) if args.use_gasre else get_re_from_stars(ds, args)

        if args.upto_kpc is not None:
            if args.docomoving: args.galrad = args.upto_kpc / (1 + ds.current_redshift) / 0.695  # fit within a fixed comoving kpc h^-1, 0.695 is Hubble constant
            else: args.galrad = args.upto_kpc  # fit within a fixed physical kpc
        else:
            args.galrad = args.re * args.upto_re  # kpc

        center = ds.halo_center_kpc

        if args.do_central:
            box = refine_box
        else:
            box_width = 2 * args.galrad * kpc
            box = ds.r[center[0] - box_width / 2.: center[0] + box_width / 2.,
                    center[1] - box_width / 2.: center[1] + box_width / 2.,
                    center[2] - box_width / 2.: center[2] + box_width / 2., ]

        # --------------setting up the projection plot parameters----------------------
        field_arr = [('enzo', f'TracerFluid0{item:0d}') for item in range(1, 4+1)]
        unit = 'Msun/pc**3'
        cmap_arr = ['Reds', 'Blues', 'Greys', 'RdBu']
        fontsize = args.fontsize
        
        fig, axes = plt.subplots(4, 3, figsize=(8, 8))
        fig.subplots_adjust(right=0.85, top=0.95, bottom=0.12, left=0.15)

        # --------------making the projection plot----------------------
        for i, field in enumerate(field_arr):
            for j, proj in enumerate(['x', 'y', 'z']):
                prj = yt.ProjectionPlot(ds, proj, field, center=center, data_source=box, width=2 * args.galrad * kpc, weight_field=None, fontsize=fontsize)

                prj.set_log(field, True)
                #prj.set_unit(field, unit)
                prj.set_cmap(field, cmap_arr[i])
                prj.annotate_timestamp(corner='lower_right', redshift=True, draw_inset_box=True)

                # ------plotting onto a matplotlib figure--------------
                ax = axes[i][j]
                prj.plots[field].axes = ax
                prj._setup_plots()

                annotate_axes(ax, ax.get_xlabel(), ax.get_ylabel(), args=args, 
                            label='', clabel=prj.plots[field].cax.get_ylabel(), p=prj.plots[field].cb.mappable,
                            hide_xaxis=False, hide_yaxis=index, hide_cbar=index < 2, cticks_integer=False)

        save_fig(fig, Path(args.output_dir) / 'figs', f'{args.halo}_{args.output}_{args.do}_projection.png', args=args)

    print('Completed in %s' % timedelta(seconds=(datetime.now() - start_time).seconds))
