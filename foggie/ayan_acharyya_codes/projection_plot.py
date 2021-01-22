##!/usr/bin/env python3

""""

    Title :      projection_plot
    Notes :      Initial attempts to play around with FOGGIE outputs, make projection plots.
    Author:      Ayan Acharyya
    Started  :   January 2021

"""
from header import *

# ---------to parse keyword arguments----------
def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='''identify satellites in FOGGIE simulations''')
    parser.add_argument('--system', metavar='system', type=str, action='store', help='Which system are you on? Default is Jase')
    parser.set_defaults(system='ayan_local')

    parser.add_argument('--do', metavar='do', type=str, action='store', help='Which particles do you want to plot? Default is gas')
    parser.set_defaults(do='gas')

    parser.add_argument('--run', metavar='run', type=str, action='store', help='which run? default is natural')
    parser.set_defaults(run="nref11c_nref9f")

    parser.add_argument('--halo', metavar='halo', type=str, action='store', help='which halo? default is 8508 (Tempest)')
    parser.set_defaults(halo='8508')

    parser.add_argument('--proj', metavar='proj', type=str, action='store', help='Which projection do you want to plot? Default is x')
    parser.set_defaults(proj='x')

    parser.add_argument('--pwd', dest='pwd', action='store_true', help='just use the pwd?, default is no')
    parser.set_defaults(pwd=False)

    parser.add_argument('--run_all', dest='run_all', action='store_true', help='just use the pwd?, default is no')
    parser.set_defaults(pwd=False)

    parser.add_argument('--output', metavar='output', type=str, action='store', help='which output? default is RD0020')
    parser.set_defaults(output='RD0042')

    args = parser.parse_args()
    return args

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

    # The variables used below come from
    field_dict = {'gas':('gas', 'density'), 'stars':('deposit', 'stars_density'), 'metal':('gas', 'metallicity'), 'temp':('gas', 'temperature'), 'dm':('deposit', 'dm_density')}
    cmap_dict = {'gas':density_color_map, 'stars':plt.cm.Greys_r, 'metal':metal_color_map, 'temp':temperature_color_map, 'dm':plt.cm.gist_heat}
    unit_dict = {'gas':'Msun/pc**2', 'stars':'Msun/pc**2', 'metal':'Zsun', 'temp':'K', 'dm':'Msun/pc**2'}
    zmin_dict = {'gas':density_proj_min, 'stars':density_proj_min, 'metal':1.e-3, 'temp':1.e3, 'dm':density_proj_min}
    zmax_dict = {'gas':density_proj_max, 'stars':density_proj_max, 'metal':metal_max, 'temp':temperature_max, 'dm':density_proj_max}
    weight_field_dict = {'gas':None, 'stars':None, 'metal':('gas', 'density'), 'temp':('gas', 'density'), 'dm':None}

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

            # prj = eps.single_plot(prj) # failed attempt at saving as eps instead of png
            # prj.save_fig(name=fig_dir + '/%s_%s' % (haloname, d), format='eps', mpl_kwargs={'dpi': 500})
            prj.save(name=fig_dir + '%s_%s' % (haloname, d), suffix='png', mpl_kwargs={'dpi': 500})
    return prj

# -----main code-----------------
if __name__ == '__main__':

    args = parse_args()
    print(args.system) #
    ds, refine_box = load_sim(args, region='refine_box')

    print('box center =', ds.refine_box_center, 'box width =', ds.refine_width * kpc)

    foggie_dir, output_dir, run_loc, code_path, trackname, haloname, spectra_dir, infofile = get_run_loc_etc(args)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    prj = make_projection_plots(ds=refine_box.ds, center=ds.refine_box_center, \
                                refine_box=refine_box, x_width=ds.refine_width * kpc, \
                                fig_dir=output_dir, haloname=args.halo, name=halo_dict[args.halo], \
                                fig_end='projection', do=[ar for ar in args.do.split(',')], axes=[ar for ar in args.proj.split(',')], is_central=True, add_arrow=False, add_velocity=False)
    prj.show()
    print('Completed')
