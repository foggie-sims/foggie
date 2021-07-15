##!/usr/bin/env python3

"""

    Title :      projection_plot
    Notes :      Initial attempts to play around with FOGGIE outputs, make projection plots.
    Output :     projection plots as png files
    Author :     Ayan Acharyya
    Started :    January 2021
    Example :    run projection_plot.py --system ayan_hd --halo 4123 --output RD0038 --do gas --proj x --fullbox --nrot 0 --iscolorlog

"""
from header import *
from util import *
from foggie.utils.get_proper_box_size import get_proper_box_size
start_time = time.time()

# --------------------------------------------------------------------------------
def do_plot(ds, field, axs, annotate_positions, small_box, center, box_width, cmap, name, unit='Msun/pc**2', \
            zmin=density_proj_min, zmax=density_proj_max, ann_sphere_rad=(1, 'kpc'), weight_field=None, \
            normal_vector=None, north_vector=None, hide_axes=False, iscolorlog=False, noweight=False):
    '''
    Function to make yt projection plot
    (adopted from foggie.satellites.for_paper.central_projection_plots)
    '''
    box_width_code = (box_width/kpc) / get_proper_box_size(ds) # converting from kpc to code units
    if noweight: weight_field = None

    start_time2 = time.time()
    if field[1] == 'age': # then do ParticlePlot
        prj = yt.ParticleProjectionPlot(ds, axs, field, center=center, data_source=small_box, width=box_width, weight_field=weight_field)
    elif normal_vector is not None: # do rotated off axis ProjectionPlot
        prj = yt.OffAxisProjectionPlot(ds, normal_vector, field, north_vector=north_vector, center=center, width=box_width_code, data_source=small_box, weight_field=weight_field)
    else: # do ProjectionPlot
        prj = yt.ProjectionPlot(ds, axs, field, center=center, data_source=small_box, width=box_width, weight_field=weight_field)

    print('Just the plotting took %s minutes' % ((time.time() - start_time2) / 60))

    prj.set_log(field, iscolorlog)
    if not noweight:
        prj.set_unit(field, unit)
        prj.set_zlim(field, zmin=zmin, zmax=zmax)
    if field[1] == 'age': prj.set_buff_size((67, 67)) ##
    try: cmap.set_bad('k')
    except: pass
    prj.set_cmap(field, cmap)

    prj.annotate_timestamp(corner='lower_right', redshift=True, draw_inset_box=True)
    prj.annotate_text((0.05, 0.9), name, coord_system='axis', text_args = {'fontsize': 500, 'color': 'white'})#, inset_box_args = {})
    if hide_axes:
        prj.hide_axes()
        prj.annotate_scale(size_bar_args={'color': 'white'}, corner='lower_left')
    # prj.hide_colorbar()

    for cen in annotate_positions:
        prj.annotate_sphere(cen, radius=ann_sphere_rad, coord_system='data', circle_args={'color': 'white'})

    prj.set_figure_size(5)
    return prj

# --------------------------------------------------------------------------------
def make_projection_plots(ds, center, refine_box, box_width, fig_dir, name, \
                          fig_end='projection', do=['stars', 'gas', 'metal'], axes=['x', 'y', 'z'], annotate_positions=[], \
                          is_central=False, add_velocity=False, add_arrow=False, start_arrow=[], end_arrow=[], rot_frame=0, \
                          nframes=200, hide_axes=False, iscolorlog=False, noweight=False):
    '''
    Function to arrange overheads of yt projection plot
    (adopted from foggie.satellites.for_paper.central_projection_plots)
    '''

    if is_central:
        small_box = refine_box
    else:
        small_box = ds.r[center[0] - box_width / 2.: center[0] + box_width / 2.,
                    center[1] - box_width / 2.: center[1] + box_width / 2.,
                    center[2] - box_width / 2.: center[2] + box_width / 2., ]

    # The variables used below come from foggie.utils.consistency.py
    field_dict = {'gas':('gas', 'density'), 'gas_entropy':('gas', 'entropy'), 'stars':('deposit', 'stars_density'),'ys_density':('deposit', 'young_stars_density'), 'ys_age':('my_young_stars', 'age'), 'ys_mass':('deposit', 'young_stars_mass'), 'metal':('gas', 'metallicity'), 'temp':('gas', 'temperature'), 'dm':('deposit', 'dm_density'), 'vrad':('gas', 'radial_velocity_corrected'), 'vlos':('gas', 'v_corrected')}
    cmap_dict = {'gas':density_color_map, 'gas_entropy':entropy_color_map, 'stars':plt.cm.Greys_r, 'ys_density':density_color_map, 'ys_age':density_color_map, 'ys_mass':density_color_map, 'metal':metal_color_map, 'temp':temperature_color_map, 'dm':plt.cm.gist_heat, 'vrad':velocity_discrete_cmap, 'vlos':velocity_discrete_cmap}
    unit_dict = defaultdict(lambda: 'Msun/pc**2', metal='Zsun', temp='K', vrad='km/s', ys_age='Myr', ys_mass='pc*Msun', gas_entropy='keV*cm**3', vlos='km/s')
    zmin_dict = defaultdict(lambda: density_proj_min, metal=2e-2, temp=1.e3, vrad=-50, ys_age=0.1, ys_mass=1, ys_density=1e-3, gas_entropy=1.6e25, vlos=-50)
    zmax_dict = defaultdict(lambda: density_proj_max, metal= 5e0, temp= temperature_max, vrad=50, ys_age=10, ys_mass=2e3, ys_density=1e1, gas_entropy=1.2e27, vlos=50)
    weight_field_dict = defaultdict(lambda: None, metal=('gas', 'density'), temp=('gas', 'density'), vrad=('gas', 'density'), vlos=('gas', 'density'))
    colorlog_dict = defaultdict(lambda: False, metal=True, gas=True, temp=True, gas_entropy=True)
    north_vector_dict = {'x':[0,1,0], 'y':[0,0,1], 'z':[1,0,0]} # north vector = which way is up; therefore, for proj = x, i.e. LoS _along_ x-axis (with 0 rot), up direction is actually y-axis, and for proj=y, z axis is 'up' direction
    normal_vector_dict = {'x': [np.cos(2 * np.pi * rot_frame / nframes), 0, np.sin(2 * np.pi * rot_frame / nframes)], \
                          'y': [np.sin(2 * np.pi * rot_frame / nframes), np.cos(2 * np.pi * rot_frame / nframes), 0], \
                          'z': [0, np.sin(2 * np.pi * rot_frame / nframes), np.cos(2 * np.pi * rot_frame / nframes)]}

    rot_text = '_rot_%03d_outof_%d' % (rot_frame, nframes) if rot_frame else ''

    for ax in axes:
        north_vector = north_vector_dict[ax] if rot_frame else None
        normal_vector = normal_vector_dict[ax] if rot_frame else None

        for d in do:
            zmin = zmin_dict[d] if args.cmin is None else args.cmin
            zmax = zmax_dict[d] if args.cmax is None else args.cmax
            prj = do_plot(ds, 'v' + ax + '_corrected' if d == 'vlos' else field_dict[d], ax, annotate_positions, small_box, center, box_width, cmap_dict[d], name, unit=unit_dict[d], zmin=zmin, zmax=zmax, weight_field=weight_field_dict[d], normal_vector=normal_vector, north_vector=north_vector, hide_axes=hide_axes, iscolorlog=iscolorlog if iscolorlog else colorlog_dict[d], noweight=noweight)

            if add_velocity: prj.annotate_velocity(factor=20)
            if add_arrow:
                if (start_arrow == []) | (end_arrow == []):
                    print('Called add_arrow, but missing start_arrow or end_arrow')
                else:
                    for s_arrow, e_arrow in zip(start_arrow, end_arrow):
                        prj.annotate_arrow(pos=e_arrow, starting_pos=s_arrow, coord_system='data')

            prj.save(fig_dir + '%s' % (d) + '_box=%.2Fkpc' % (box_width) + '_proj_' + ax + rot_text + '_' + fig_end + '.png', mpl_kwargs={'dpi': 500})
    return prj

# -------------------------------------------------------------------
def my_young_stars(pfilter, data):
    '''
    Filter star particles with creation time < threshold Myr ago
    To use: yt.add_particle_filter("young_stars8", function=_young_stars8, filtered_type='all', requires=["creation_time"])
    Based on: foggie.yt_fields._young_stars8()
    '''
    isstar = data[(pfilter.filtered_type, "particle_type")] == 2
    age = data.ds.current_time - data[pfilter.filtered_type, "creation_time"]
    filter = np.logical_and(isstar, age.in_units('Myr') <= args.age_thresh, age >= 0)
    return filter

# -----main code-----------------
if __name__ == '__main__':
    dummy_args_tuple = parse_args('8508', 'RD0042')  # default simulation to work upon when comand line args not provided
    if type(dummy_args_tuple) is tuple: dummy_args = dummy_args_tuple[0] # if the sim has already been loaded in, in order to compute the box center (via utils.pull_halo_center()), then no need to do it again
    else: dummy_args = dummy_args_tuple

    if dummy_args.do_all_halos: list_of_sims = get_all_sims(dummy_args) # all snapshots of all halos
    elif dummy_args.do_all_sims: list_of_sims = get_all_sims_for_this_halo(dummy_args) # all snapshots of this particular halo
    else: list_of_sims = [(dummy_args.halo, dummy_args.output)]

    for this_sim in list_of_sims:
        if dummy_args.do_all_sims: args = parse_args(this_sim[0], this_sim[1])
        else: args = dummy_args_tuple # since parse_args() has already been called and evaluated once, no need to repeat it

        if type(args) is tuple:
            args, ds, refine_box = args  # if the sim has already been loaded in, in order to compute the box center (via utils.pull_halo_center()), then no need to do it again
            myprint('ds ' + str(ds) + ' for halo ' + str(this_sim[0]) + ' was already loaded at some point by utils; using that loaded ds henceforth', args)
        else:
            ds, refine_box = load_sim(args, region='refine_box', do_filter_particles=False)

        fig_dir = args.output_dir + 'figs/' + args.output + '/'
        Path(fig_dir).mkdir(parents=True, exist_ok=True)

        yt.add_particle_filter('my_young_stars', function=my_young_stars, filtered_type='all', requires=['creation_time', 'particle_type'])
        ds.add_particle_filter('my_young_stars')

        if args.fullbox: args.galrad = ds.refine_width / 2 # kpc
        center = refine_box.center.in_units(kpc) if args.do_central else ds.arr(args.halo_center, kpc)

        if not args.noplot:
            if args.makerotmovie:
                min_nrot, max_nrot = 0, args.nframes
            else:
                min_nrot = int(args.nrot * args.nframes) # 0 <= args.nrot <= 1
                max_nrot = min_nrot + 1

            for nrot in range(min_nrot, max_nrot):
                print('Plotting', nrot + 1, 'out of', max_nrot, 'frames..')
                prj = make_projection_plots(ds=refine_box.ds, center=center, \
                                        refine_box=refine_box, box_width=2 * args.galrad * kpc, \
                                        fig_dir=fig_dir, name=halo_dict[args.halo], \
                                        fig_end='projection', do=[ar for ar in args.do.split(',')], axes=[ar for ar in args.projection.split(',')], \
                                        is_central=args.do_central, add_arrow=args.add_arrow, add_velocity=args.add_velocity, rot_frame=nrot, \
                                        nframes=args.nframes, hide_axes=args.hide_axes, iscolorlog=args.iscolorlog, noweight=args.noweight) # using halo_center_kpc instead of refine_box_center
            #prj.show(block=False)
        else:
            print('Skipping plotting step')

        print('Completed in %s minutes' % ((time.time() - start_time) / 60))
