##!/usr/bin/env python3

""""

    Title :      projection_plot
    Notes :      Initial attempts to play around with FOGGIE outputs, make projection plots.
    Author:      Ayan Acharyya
    Started  :   January 2021
    Example :    run projection_plot.py --system ayan_local --halo 8508 --output RD0042

"""
from header import *
from collections import defaultdict

# -------------make yt projection plot (adopted from foggie.satellites.for_paper.central_projection_plots) -----------
def do_plot(ds, field, axs, annotate_positions, small_box, center, x_width, cmap, name, unit='Msun/pc**2', \
            zmin=density_proj_min, zmax=density_proj_max, ann_sphere_rad=(1, 'kpc'), weight_field=None):
    prj = yt.ProjectionPlot(ds, axs, field, center=center, data_source=small_box, width=x_width, weight_field=weight_field)

    prj.set_unit(field, unit)
    prj.set_zlim(field, zmin=zmin, zmax=zmax)
    cmap.set_bad('k')
    prj.set_cmap(field, cmap)

    prj.annotate_timestamp(corner='lower_right', redshift=True, draw_inset_box=True)
    prj.annotate_text((0.05, 0.9), name, coord_system='axis', text_args = {'fontsize': 500, 'color': 'white'})#, inset_box_args = {})
    # prj.hide_axes()
    # prj.hide_colorbar()

    for cen in annotate_positions:
        prj.annotate_sphere(cen, radius=ann_sphere_rad, coord_system='data', circle_args={'color': 'white'})

    return prj

# -------------arrange overheads of yt projection plot (adopted from foggie.satellites.for_paper.central_projection_plots) -----------
def make_projection_plots(ds, center, refine_box, x_width, fig_dir, haloname, name, \
                          fig_end='projection', do=['stars', 'gas', 'metal'], axes=['x', 'y', 'z'], annotate_positions=[], \
                          add_velocity=False, is_central=False, add_arrow=False, start_arrow=[], end_arrow=[]):
    if is_central:
        small_box = refine_box
    else:
        small_box = ds.r[center[0] - x_width / 2.: center[0] + x_width / 2.,
                    center[1] - x_width / 2.: center[1] + x_width / 2.,
                    center[2] - x_width / 2.: center[2] + x_width / 2.,]

    metal_color_map = sns.blend_palette(("black", "#5d31c4", "#5d31c4", "#4575b4", "#d73027","darkorange", "#ffe34d"), as_cmap=True)

    # The variables used below come from foggie.utils.consistency.py
    field_dict = {'gas':('gas', 'density'), 'stars':('deposit', 'stars_density'), 'metal':('gas', 'metallicity'), 'temp':('gas', 'temperature'), 'dm':('deposit', 'dm_density'), 'vrad':('gas', 'radial_velocity_corrected')}
    cmap_dict = {'gas':density_color_map, 'stars':plt.cm.Greys_r, 'metal':metal_color_map, 'temp':temperature_color_map, 'dm':plt.cm.gist_heat, 'vrad':velocity_discrete_cmap}
    unit_dict = defaultdict(lambda: 'Msun/pc**2', metal='Zsun', temp='K', vrad='km/s')
    zmin_dict = defaultdict(lambda: density_proj_min, metal=1.e-3, temp=1.e3, vrad=-250)
    zmax_dict = defaultdict(lambda: density_proj_max, metal= metal_max, temp= temperature_max, vrad=250)
    weight_field_dict = defaultdict(lambda: None, metal=('gas', 'density'), temp=('gas', 'density'), vrad=('gas', 'density'))

    for ax in axes:
        for d in do:
            prj = do_plot(ds, field_dict[d], ax, annotate_positions, small_box, center, x_width, cmap_dict[d], name, unit=unit_dict[d], zmin=zmin_dict[d], zmax=zmax_dict[d], weight_field=weight_field_dict[d])

            if add_velocity: prj.annotate_velocity(factor=20)
            if add_arrow:
                if (start_arrow == []) | (end_arrow == []):
                    print('Called add_arrow, but missing start_arrow or end_arrow')
                else:
                    for s_arrow, e_arrow in zip(start_arrow, end_arrow):
                        prj.annotate_arrow(pos=e_arrow, starting_pos=s_arrow, coord_system='data')

            prj.save(name=fig_dir + '%s_%s' % (haloname, d), suffix='png', mpl_kwargs={'dpi': 500})
    return prj

# -----main code-----------------
if __name__ == '__main__':

    loop_over = [('8508', 'RD0042')]#, ('5036', 'RD0039'), ('5016', 'RD0042'), ('4123', 'RD0031'), ('2878', 'RD0020'), ('2392', 'RD0030')]

    for thisloop in loop_over:
        args = parse_args(thisloop[0], thisloop[1])
        ds, refine_box = load_sim(args, region='refine_box')

        print('box center =', ds.refine_box_center, 'box width =', ds.refine_width * kpc)

        foggie_dir, output_dir, run_loc, code_path, trackname, haloname, spectra_dir, infofile = get_run_loc_etc(args)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        prj = make_projection_plots(ds=refine_box.ds, center=ds.refine_box_center, \
                                    refine_box=refine_box, x_width=ds.refine_width * kpc, \
                                    fig_dir=output_dir+'figs/', haloname=args.output, name=halo_dict[args.halo], \
                                    fig_end='projection', do=[ar for ar in args.do.split(',')], axes=[ar for ar in args.proj.split(',')], is_central=True, add_arrow=False, add_velocity=False)
        prj.show()
    print('Completed')
