##!/usr/bin/env python3

"""

    Title :      projection_plot
    Notes :      Initial attempts to play around with FOGGIE outputs, make projection plots.
    Output :     projection plots as png files
    Author :     Ayan Acharyya
    Started :    January 2021
    Example :    run projection_plot.py --system ayan_local --halo 8508 --output RD0042 --do stars

"""
from header import *
from util import *
from collections import defaultdict
start_time = time.time()

# --------------------------------------------------------------------------------
def do_plot(ds, field, axs, annotate_positions, small_box, center, x_width, cmap, name, unit='Msun/pc**2', \
            zmin=density_proj_min, zmax=density_proj_max, ann_sphere_rad=(1, 'kpc'), weight_field=None, particleplot=False):
    '''
    Function to make yt projection plot
    (adopted from foggie.satellites.for_paper.central_projection_plots)
    '''

    if field[1] == 'age': # then do ParticlePlot
        particleplot = True

    if particleplot: # then do ParticlePlot
        prj = yt.ParticlePlot(ds, (field[0], 'particle_position_x'), (field[0], 'particle_position_y'), field, data_source=small_box, width=x_width, weight_field=weight_field)
        #field = field[1]
    else: # else do ProjectionPlot
        prj = yt.ProjectionPlot(ds, axs, field, center=center, data_source=small_box, width=x_width, weight_field=weight_field)

    prj.set_unit(field, unit)
    prj.set_zlim(field, zmin=zmin, zmax=zmax)
    try: cmap.set_bad('k')
    except: pass
    prj.set_cmap(field, cmap)

    prj.annotate_timestamp(corner='lower_right', redshift=True, draw_inset_box=True)
    prj.annotate_text((0.05, 0.9), name, coord_system='axis', text_args = {'fontsize': 500, 'color': 'white'})#, inset_box_args = {})
    # prj.hide_axes()
    # prj.hide_colorbar()

    for cen in annotate_positions:
        prj.annotate_sphere(cen, radius=ann_sphere_rad, coord_system='data', circle_args={'color': 'white'})

    return prj

# --------------------------------------------------------------------------------
def make_projection_plots(ds, center, refine_box, x_width, fig_dir, haloname, name, \
                          fig_end='projection', do=['stars', 'gas', 'metal'], axes=['x', 'y', 'z'], annotate_positions=[], \
                          add_velocity=False, is_central=False, add_arrow=False, start_arrow=[], end_arrow=[]):
    '''
    Function to arrange overheads of yt projection plot
    (adopted from foggie.satellites.for_paper.central_projection_plots)
    '''

    if is_central:
        small_box = refine_box
    else:
        small_box = ds.r[center[0] - x_width / 2.: center[0] + x_width / 2.,
                    center[1] - x_width / 2.: center[1] + x_width / 2.,
                    center[2] - x_width / 2.: center[2] + x_width / 2.,]

    metal_color_map = sns.blend_palette(("black", "#5d31c4", "#5d31c4", "#4575b4", "#d73027","darkorange", "#ffe34d"), as_cmap=True)

    # The variables used below come from foggie.utils.consistency.py
    field_dict = {'gas':('gas', 'density'), 'gas_entropy':('gas', 'entropy'), 'stars':('deposit', 'stars_density'),'ys_density':('deposit', 'young_stars_density'), 'ys_age':('young_stars', 'age'), 'ys_mass':('deposit', 'young_stars_mass'), 'metal':('gas', 'metallicity'), 'temp':('gas', 'temperature'), 'dm':('deposit', 'dm_density'), 'vrad':('gas', 'radial_velocity_corrected')}
    cmap_dict = {'gas':density_color_map, 'gas_entropy':entropy_color_map, 'stars':plt.cm.Greys_r, 'ys_density':density_color_map, 'ys_age':density_color_map, 'ys_mass':density_color_map, 'metal':metal_color_map, 'temp':temperature_color_map, 'dm':plt.cm.gist_heat, 'vrad':velocity_discrete_cmap}
    unit_dict = defaultdict(lambda: 'Msun/pc**2', metal='Zsun', temp='K', vrad='km/s', ys_age='Myr', ys_mass='pc*Msun', gas_entropy='keV*cm**3')
    zmin_dict = defaultdict(lambda: density_proj_min, metal=1.e-3, temp=1.e3, vrad=-250, ys_age=0, ys_mass=1, ys_density=1e-3, gas_entropy=1.6e25)
    zmax_dict = defaultdict(lambda: density_proj_max, metal= metal_max, temp= temperature_max, vrad=250, ys_age=10, ys_mass=2e3, ys_density=1e1, gas_entropy=1.2e27)
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
    dummy_args = parse_args('8508', 'RD0042')
    if dummy_args.do_all_sims: list_of_sims = all_sims
    else: list_of_sims = [('8508', 'RD0042')]  # default simulation to work upon when comand line args not provided

    for this_sim in list_of_sims:
        args = parse_args(this_sim[0], this_sim[1])
        ds, refine_box = load_sim(args, region='refine_box')

        Path(args.output_dir+'figs/').mkdir(parents=True, exist_ok=True)

        prj = make_projection_plots(ds=refine_box.ds, center=ds.halo_center_kpc, \
                                    refine_box=refine_box, x_width=ds.refine_width * kpc, \
                                    fig_dir=args.output_dir+'figs/', haloname=args.output, name=halo_dict[args.halo], \
                                    fig_end='projection', do=[ar for ar in args.do.split(',')], axes=[ar for ar in args.projection.split(',')], is_central=True, add_arrow=False, add_velocity=False) # using halo_center_kpc instead of refine_box_center
        prj.show()
        print('Completed in %s minutes' % ((time.time() - start_time) / 60))
