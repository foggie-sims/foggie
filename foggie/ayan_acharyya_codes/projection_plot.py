##!/usr/bin/env python3

"""

    Title :      projection_plot
    Notes :      Initial attempts to play around with FOGGIE outputs, make projection plots.
    Output :     projection plots as png files
    Author :     Ayan Acharyya
    Started :    January 2021
    Example :    run projection_plot.py --system ayan_local --halo 4123 --output RD0038 --do gas --proj x --fullbox --nframes 1 --rot_normal_by -30 --rot_normal_about y --rot_north_by 45 --rot_north_about x --iscolorlog
                 run projection_plot.py --system ayan_local --halo 8508 --upto_kpc 1000 --output RD0042 --do mrp --annotate_grids --annotate_box 200,400
                 run projection_plot.py --system ayan_local --halo 8508 --upto_kpc 10 --output RD0030 --do metal --use_density_cut --proj y --docomoving

"""
from header import *
from util import *
from foggie.utils.get_proper_box_size import get_proper_box_size
from mpl_toolkits.axes_grid1 import make_axes_locatable
from compute_MZgrad import *

start_time = time.time()

# -----------------------------------------------------------
def ptype4(pfilter, data):
    filter = data[(pfilter.filtered_type, "particle_type")] == 4
    return filter

# ---------------------------------------------------------
def annotate_box(p, width, ds, center, unit='kpc', projection='x'):
    '''
    Function to annotate a given yt plot with a box of a given size (width) centered on a given center
    '''
    color, linewidth = 'red', 2
    width_code = ds.arr(width, unit).in_units('code_length').value.tolist()
    center = ds.arr(center, 'kpc').in_units('code_length').value.tolist() # because input center is in kpc
    proj_dict = {'x': 1, 'y': 2, 'z': 0}

    for left_array, right_array in [[np.array([-1, -1, 0]), np.array([-1, +1, 0])], \
                                    [np.array([-1, +1, 0]), np.array([+1, +1, 0])], \
                                    [np.array([+1, +1, 0]), np.array([+1, -1, 0])], \
                                    [np.array([+1, -1, 0]), np.array([-1, -1, 0])]]:
        p.annotate_line(center + np.roll(left_array, proj_dict[projection]) * width_code/2, center + np.roll(right_array, proj_dict[projection]) * width_code/2, coord_system='data', plot_args={'color': color, 'linewidth': linewidth},)

    return p

# --------------------------------------------------------------------------------
def do_plot(ds, field, axs, annotate_positions, small_box, center, box_width, cmap, name, unit='Msun/pc**2', \
            zmin=density_proj_min, zmax=density_proj_max, ann_sphere_rad=(1, 'kpc'), weight_field=None, annotate_markers=[], \
            normal_vector=None, north_vector=None, hide_axes=False, iscolorlog=False, noweight=False, fontsize=20, args=None):
    '''
    Function to make yt projection plot
    (adopted from foggie.satellites.for_paper.central_projection_plots)
    '''
    if noweight: weight_field = None

    start_time2 = time.time()

    if field[1] == 'age': # then do ParticlePlot
        prj = yt.ParticleProjectionPlot(ds, axs, field, center=center, data_source=small_box, width=box_width, weight_field=weight_field)
    elif normal_vector is not None or north_vector is not None: # do rotated off axis ProjectionPlot
        prj = yt.OffAxisProjectionPlot(ds, normal_vector, field, north_vector=north_vector, center=center, width=(box_width.v.tolist(), 'kpc'), data_source=small_box, weight_field=weight_field)
    else:  # do ProjectionPlot
        prj = yt.ProjectionPlot(ds, axs, field, center=center, data_source=small_box, width=box_width, weight_field=weight_field, fontsize=fontsize)

    print('Just the plotting took %s minutes' % ((time.time() - start_time2) / 60))

    prj.set_log(field, iscolorlog)
    if not noweight:
        prj.set_unit(field, unit)
        prj.set_zlim(field, zmin=zmin, zmax=zmax)
    if field[1] == 'age': prj.set_buff_size((67, 67)) ##
    try: cmap.set_bad('k')
    except: pass
    prj.set_cmap(field, cmap)

    if args is not None:
        if args.annotate_grids:
            prj.annotate_grids(min_level=args.min_level)

        if args.annotate_box is not None:
            for thisbox in [float(item) for item in args.annotate_box.split(',')]: # comoving size in kpc
                thisphys = thisbox / (1 + ds.current_redshift) / ds.hubble_constant # physical size at current redshift in kpc
                prj = annotate_box(prj, thisphys, ds, center, unit='kpc', projection=axs)

    if not args.forproposal:
        prj.annotate_timestamp(corner='lower_right', redshift=True, draw_inset_box=True)
        prj.annotate_text((0.05, 0.9), name, coord_system='axis', text_args = {'fontsize': 500, 'color': 'white'})#, inset_box_args = {})
    if hide_axes:
        prj.hide_axes()
        prj.annotate_scale(size_bar_args={'color': 'white'}, corner='lower_left')
    # prj.hide_colorbar()

    for cen in annotate_positions:
        prj.annotate_sphere(cen, radius=ann_sphere_rad, coord_system='data', circle_args={'color': 'white'})

    for cen in annotate_markers:
        prj.annotate_marker(cen, coord_system='data')

    return prj

# --------------------------------------------------------------------------------
def make_projection_plots(ds, center, refine_box, box_width, fig_dir, name, \
                          fig_end='projection', do=['stars', 'gas', 'metal'], projections=['x', 'y', 'z'], annotate_positions=[], annotate_markers=[], \
                          is_central=False, add_velocity=False, add_arrow=False, start_arrow=[], end_arrow=[], total_normal_rot=0, \
                          total_north_rot=0, rot_frame=0, nframes=200, hide_axes=False, iscolorlog=False, noweight=False, \
                          rot_north_about='x', rot_normal_about='y', output='', fontsize=20, cbar_horizontal=False, use_density_cut=False, args=None):
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

    if use_density_cut:
        rho_cut = get_density_cut(ds.current_time.in_units('Gyr'))  # based on Cassi's CGM-ISM density cut-off
        small_box = small_box.cut_region(['obj["gas", "density"] > %.1E' % rho_cut])
        print('Imposing a density criteria to get ISM above density', rho_cut, 'g/cm^3')

    # The variables used below come from foggie.utils.consistency.py
    field_dict = {'gas':('gas', 'density'), 'gas_entropy':('gas', 'entropy'), \
                  'stars':('deposit', 'stars_density'),'ys_density':('deposit', 'young_stars_density'), 'ys_age':('my_young_stars', 'age'), 'ys_mass':('deposit', 'young_stars_mass'), \
                  'metal':('gas', 'metallicity'), 'temp':('gas', 'temperature'), 'dm':('deposit', 'dm_density'), 'vrad':('gas', 'radial_velocity_corrected'), \
                  'grid': ('index', 'grid_level'), 'mrp': ('deposit', 'ptype4_mass'), 'vdisp_3d':('gas', 'velocity_dispersion_3d'), \
                  'vtan':('gas', 'tangential_velocity_corrected'), 'vphi':('gas', 'phi_velocity_corrected'), 'vtheta':('gas', 'theta_velocity_corrected')}
    cmap_dict = {'gas':density_color_map, 'gas_entropy':entropy_color_map, 'stars':plt.cm.Greys_r, 'ys_density':density_color_map, 'ys_age':density_color_map, \
                 'ys_mass':density_color_map, 'metal':old_metal_color_map, 'temp':temperature_color_map, 'dm':plt.cm.gist_heat, 'vrad':velocity_discrete_cmap, \
                 'vlos':velocity_discrete_cmap, 'grid':'viridis', 'mrp':'viridis', 'vdisp_los':'viridis', 'vdisp_3d':'viridis', 'vtan':'viridis', \
                 'vphi':velocity_discrete_cmap, 'vtheta':'viridis'}
    unit_dict = defaultdict(lambda: 'Msun/pc**2', metal='Zsun', temp='K', vrad='km/s', ys_age='Myr', ys_mass='pc*Msun', gas_entropy='keV*cm**3', \
                            vlos='km/s', grid='', mrp='cm*g', vdisp_los='km/s', vdisp_3d='km/s', vtan='km/s', vphi='km/s', vtheta='km/s')
    zmin_dict = defaultdict(lambda: density_proj_min, metal=7e-2 if args.forpaper else 2e-2, temp=1.e3, vrad=-200, ys_age=0.1, ys_mass=1, ys_density=1e-3, \
                            gas_entropy=1.6e25, vlos=-500, grid=1, mrp=1e57, vdisp_los=0, vdisp_3d=0, vtan=0, vphi=-500, vtheta=0)
    zmax_dict = defaultdict(lambda: density_proj_max, metal= 2 if args.forproposal else 4e0 if args.forpaper else 5e0, temp= temperature_max, vrad=200, \
                            ys_age=10, ys_mass=2e3, ys_density=1e1, gas_entropy=1.2e27, vlos=500, grid=11, mrp=1e65, vdisp_los=500, vdisp_3d=500, \
                            vtan=500, vphi=500, vtheta=500)
    weight_field_dict = defaultdict(lambda: None, metal=('gas', 'mass') if args.forpaper else ('gas', 'density'), temp=('gas', 'density'), \
                                    vrad=('gas', 'density'), vlos=('gas', 'density'), vdisp_los=('gas', 'density'), vdisp_3d=('gas', 'density'),\
                                    vtan=('gas', 'density'), vphi=('gas', 'density'), vtheta=('gas', 'density'))
    colorlog_dict = defaultdict(lambda: False, metal=False if args.forproposal else True, gas=True, temp=True, gas_entropy=True, mrp=True)

    # north vector = which way is up; this is set up such that the north vector rotates ABOUT the normal vector
    rot_north_by = (total_north_rot * np.pi / 180) * rot_frame / nframes # to convert total_north_rot from deg to radian
    north_vector_dict = {'x': [0, np.sin(rot_north_by), np.cos(rot_north_by)], \
                         'y': [np.cos(rot_north_by), 0, np.sin(rot_north_by)], \
                         'z': [np.sin(rot_north_by), np.cos(rot_north_by), 0]}

    # normal vector = vector coming out of the plabe of image; this is set up such the normal vector rotates about the north vector
    rot_normal_by = (total_normal_rot * np.pi / 180) * rot_frame / nframes # to convert total_normal_rot from deg to radian
    normal_vector_dict = {'x': [0, np.sin(rot_normal_by), np.cos(rot_normal_by)], \
                          'y': [np.cos(rot_normal_by), 0, np.sin(rot_normal_by)], \
                          'z': [np.sin(rot_normal_by), np.cos(rot_normal_by), 0]}

    rot_text = '_normrotby_%.3F_northrotby_%.3F_frame_%03d_of_%03d' % (total_normal_rot, total_north_rot, rot_frame, nframes) if (total_normal_rot + total_north_rot) != 0 else ''
    density_cut_text = '_wdencut' if use_density_cut else ''

    for thisproj in projections:
        field_dict.update({'vlos':('gas', 'v' + thisproj + '_corrected'), 'vdisp_los':('gas', 'velocity_dispersion_' + thisproj)})
        north_vector = north_vector_dict[rot_north_about] if rot_frame else None
        normal_vector = normal_vector_dict[rot_normal_about] if rot_frame else None

        print('Deb105: north_vector=', north_vector, 'normal_vector=', normal_vector) #

        for d in do:
            zmin = zmin_dict[d] if args.cmin is None else args.cmin
            zmax = zmax_dict[d] if args.cmax is None else args.cmax

            prj = do_plot(ds, field_dict[d], thisproj, annotate_positions, small_box, center, box_width, cmap_dict[d], name, unit=unit_dict[d], zmin=zmin, zmax=zmax, weight_field=weight_field_dict[d], normal_vector=normal_vector, north_vector=north_vector, hide_axes=hide_axes, iscolorlog=iscolorlog if iscolorlog else colorlog_dict[d], noweight=noweight, fontsize=fontsize, annotate_markers=annotate_markers, args=args)

            if add_velocity: prj.annotate_velocity(factor=20)
            if add_arrow:
                if (start_arrow == []) | (end_arrow == []):
                    print('Called add_arrow, but missing start_arrow or end_arrow')
                else:
                    for s_arrow, e_arrow in zip(start_arrow, end_arrow):
                        prj.annotate_arrow(pos=e_arrow, starting_pos=s_arrow, coord_system='data')

            # ------plotting onto a matplotlib figure--------------
            fig, axes = plt.subplots(figsize=(8, 8))

            #prj.plots[thisfield].figure = fig
            prj.plots[field_dict[d]].axes = axes
            divider = make_axes_locatable(axes)
            prj._setup_plots()

            if cbar_horizontal:
                fig.subplots_adjust(right=0.95, top=0.95, bottom=0.12, left=0.05)
                cax = divider.append_axes('bottom', size='5%', pad=1.3)
                cbar = fig.colorbar(prj.plots[field_dict[d]].cb.mappable, orientation='horizontal', cax=cax)
            else:
                fig.subplots_adjust(right=0.85, top=0.95, bottom=0.12, left=0.15)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = fig.colorbar(prj.plots[field_dict[d]].cb.mappable, orientation='vertical', cax=cax)
            cbar.ax.tick_params(labelsize=fontsize, width=2.5, length=5)
            cbar.set_label(prj.plots[field_dict[d]].cax.get_ylabel(), fontsize=fontsize)
            if args.forpaper:
                cbar.set_ticks([1e-1, 5e-1, 2e0])
                cbar.set_ticklabels(['0.1', '0.5', '2.0'])


            axes.xaxis.set_major_locator(plt.MaxNLocator(5))
            axes.yaxis.set_major_locator(plt.MaxNLocator(5))
            axes.set_xticklabels(['%.1F' % item for item in axes.get_xticks()], fontsize=fontsize)
            axes.set_yticklabels(['%.1F' % item for item in axes.get_yticks()], fontsize=fontsize)
            axes.set_xlabel(axes.get_xlabel(), fontsize=fontsize)
            axes.set_ylabel(axes.get_ylabel(), fontsize=fontsize)

            filename = fig_dir + '%s_%s' % (output, d) + '_box=%.2Fkpc' % (box_width) + '_proj_' + thisproj + rot_text + '_' + fig_end + density_cut_text + '.png'
            plt.savefig(filename, transparent=args.fortalk)
            myprint('Saved figure ' + filename, args)
            plt.show()

    return fig, prj

# -------------------------------------------------------------------
def my_young_stars(pfilter, data):
    '''
    Filter star particles with creation time < threshold Myr ago
    To use: yt.add_particle_filter("young_stars8", function=_young_stars8, filtered_type='all', requires=["creation_time"])
    Based on: foggie.yt_fields._young_stars8()
    '''
    print('Creating new particle filter for stars < ' + str(args.age_thresh) + ' Myr..')
    isstar = data[(pfilter.filtered_type, "particle_type")] == 2
    age = data.ds.current_time - data[pfilter.filtered_type, "creation_time"]
    filter = np.logical_and(isstar, age.in_units('Myr') <= args.age_thresh, age >= 0)
    return filter

# -----main code-----------------
if __name__ == '__main__':
    dummy_args_tuple = parse_args('8508', 'RD0042')  # default simulation to work upon when comand line args not provided
    if type(dummy_args_tuple) is tuple: dummy_args = dummy_args_tuple[0] # if the sim has already been loaded in, in order to compute the box center (via utils.pull_halo_center()), then no need to do it again
    else: dummy_args = dummy_args_tuple

    if dummy_args.do_all_sims:
        list_of_sims = get_all_sims(dummy_args) # all snapshots of this particular halo
    else:
        if dummy_args.do_all_halos: halos = get_all_halos(dummy_args)
        else: halos = dummy_args.halo_arr
        list_of_sims = list(itertools.product(halos, dummy_args.output_arr))

    for index, this_sim in enumerate(list_of_sims):
        print('Doing', index + 1, 'out of the total %s sims..' % (len(list_of_sims)))
        if len(list_of_sims) == 1: args = dummy_args_tuple # since parse_args() has already been called and evaluated once, no need to repeat it
        else: args = parse_args(this_sim[0], this_sim[1])

        halos_df_name = dummy_args.code_path + 'halo_infos/00' + this_sim[0] + '/' + dummy_args.run + '/'
        halos_df_name += 'halo_cen_smoothed' if args.use_cen_smoothed else 'halo_c_v'
        if type(args) is tuple:
            args, ds, refine_box = args  # if the sim has already been loaded in, in order to compute the box center (via utils.pull_halo_center()), then no need to do it again
            myprint('ds ' + str(ds) + ' for halo ' + str(this_sim[0]) + ' was already loaded at some point by utils; using that loaded ds henceforth', args)
        else:
            ds, refine_box = load_sim(args, region='refine_box', do_filter_particles=False, halo_c_v_name=halos_df_name)

        ds.add_field(('gas', 'velocity_dispersion_3d'), function=get_velocity_dispersion_3d, units='km/s', take_log=False, sampling_type='cell')
        ds.add_field(('gas', 'velocity_dispersion_x'), function=get_velocity_dispersion_x, units='km/s', take_log=False, sampling_type='cell')
        ds.add_field(('gas', 'velocity_dispersion_y'), function=get_velocity_dispersion_y, units='km/s', take_log=False, sampling_type='cell')
        ds.add_field(('gas', 'velocity_dispersion_z'), function=get_velocity_dispersion_z, units='km/s', take_log=False, sampling_type='cell')

        fig_dir = args.output_dir + 'figs/' + args.output + '/'
        Path(fig_dir).mkdir(parents=True, exist_ok=True)
        if args.fortalk:
            setup_plots_for_talks()
            args.forpaper = True
        if args.forpaper or args.forproposal:
            args.use_density_cut = True
        if args.forpaper:
            args.fontsize = 15
            args.docomoving = True


        yt.add_particle_filter('my_young_stars', function=my_young_stars, filtered_type='all', requires=['creation_time', 'particle_type'])
        ds.add_particle_filter('my_young_stars')

        if 'mrp' in args.do:
            yt.add_particle_filter('ptype4', function=ptype4, requires=['particle_type'])
            ds.add_particle_filter('ptype4')

        # --------------tailoring the extent of the box------------------------
        if args.upto_kpc is not None: args.re = np.nan
        else: args.re = get_re_from_coldgas(args) if args.use_gasre else get_re_from_stars(ds, args)

        if args.upto_kpc is not None:
            if args.docomoving: args.galrad = args.upto_kpc / (1 + ds.current_redshift) / 0.695  # fit within a fixed comoving kpc h^-1, 0.695 is Hubble constant
            else: args.galrad = args.upto_kpc  # fit within a fixed physical kpc
        else:
            args.galrad = args.re * args.upto_re  # kpc

        if args.fullbox: args.galrad = ds.refine_width / 2 # kpc
        center = ds.halo_center_kpc

        annotate_markers = [] ##

        if not args.noplot:
            start_frame = 0 if args.makerotmovie or (args.rot_normal_by == 0 and args.rot_north_by == 0) else 1
            end_frame = args.nframes if args.makerotmovie else start_frame + 1

            for nrot in range(start_frame, end_frame):
                print('Plotting', nrot+1, 'out of', end_frame, 'frames..')
                fig = make_projection_plots(ds=refine_box.ds, center=center, \
                                        refine_box=refine_box, box_width=2 * args.galrad * kpc, \
                                        fig_dir=fig_dir, name=halo_dict[args.halo], output=this_sim[1], fontsize=args.fontsize*1.5, \
                                        fig_end='projection', do=[ar for ar in args.do.split(',')], projections=[ar for ar in args.projection.split(',')], annotate_positions=[], \
                                        is_central=args.do_central, add_arrow=args.add_arrow, add_velocity=args.add_velocity, rot_frame=nrot, annotate_markers=annotate_markers, \
                                        total_normal_rot=args.rot_normal_by, total_north_rot=args.rot_north_by, rot_north_about=args.rot_north_about, rot_normal_about=args.rot_normal_about, \
                                        nframes=(end_frame - start_frame), hide_axes=args.hide_axes, iscolorlog=args.iscolorlog, noweight=args.noweight, cbar_horizontal=False, use_density_cut=args.use_density_cut, args=args) # using halo_center_kpc instead of refine_box_center
        else:
            print('Skipping plotting step')

        print_master('Completed in %s minutes' % ((time.time() - start_time) / 60), args)
