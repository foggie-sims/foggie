'''
    Filename: visualization_plots.py
    Author: Cassi
    Created: 6-12-24
    Last modified: 6-12-24 by Cassi
    This file works with fogghorn_analysis.py to make a set of basic visualization plots.
    If you add a new function to this scripts, then please also add the function name to the appropriate list at the end of fogghorn/header.py
'''

from foggie.fogghorn.header import *
from foggie.fogghorn.util import *

# --------------------------------------------------------------------------------------------------------------------
def gas_density_projection(ds, region, args):
    '''
    Plots a gas density projection of the galaxy disk.
    '''

    if '-disk' in args.projection:
        if 'x' in args.projection:
            p_dir = ds.x_unit_disk
            north_vector = ds.z_unit_disk
        if 'y' in args.projection:
            p_dir = ds.y_unit_disk
            north_vector = ds.z_unit_disk
        if 'z' in args.projection:
            p_dir = ds.z_unit_disk
            north_vector = ds.x_unit_disk
        p = yt.ProjectionPlot(ds, p_dir, 'density', data_source=region, width=(20, 'kpc'), center=ds.halo_center_code, north_vector=north_vector)
    else: p = yt.ProjectionPlot(ds, args.projection, 'density', data_source=region, width=(20, 'kpc'), center=ds.halo_center_code)
    p.set_unit('density','Msun/pc**2')
    p.set_cmap('density', density_color_map)
    p.set_zlim('density',0.01,300)
    p.set_font_size(16)
    p.annotate_timestamp(corner='upper_left', redshift=True, time=True, draw_inset_box=True)
    p.save(args.output_filename)
    print('Saved figure ' + args.output_filename)

# --------------------------------------------------------------------------------------------------------------------
def gas_metallicity_projection(ds, region, args):
    '''
    Plots a gas metallicity projection of the galaxy disk.
    If the --disk_rel argument was used, this function will automatically project w.r.t the disk, instead of the box edges.
    Returns nothing. Saves output as png file
    '''
    if args.projection in args.projection_arr:
        args.output_filename = generate_plot_filename('gas_metallicity_projection', args)
        if 'disk' in args.projection: p = yt.OffAxisProjectionPlot(ds, args.projection_axis_dict[args.projection], 'metallicity', data_source=region, width=(20, 'kpc'), center=ds.halo_center_code, north_vector=args.north_vector_dict[args.projection])
        else: p = yt.ProjectionPlot(ds, args.projection, 'metallicity', data_source=region, width=(20, 'kpc'), center=ds.halo_center_code)
        p.set_unit('metallicity','Zsun*cm') # the length dimension is because this is a projected quantity
        p.set_cmap('metallicity', old_metal_color_map)
        #p.set_zlim('metallicity', 2e-2, 4e0)
        p.save(args.output_filename)
        print('Saved figure ' + args.output_filename)

# --------------------------------------------------------------------------------------------------------------------
def edge_visualizations(ds, region, args):
    '''
    Plot slices & thin projections of galaxy temperature viewed from the disk edge.
    '''

    # Visualize along two perpendicular edge axes
    for label, axis in zip(["disk-x","disk-y"], [ds.x_unit_disk, ds.y_unit_disk]):

        p_filename = args.output_filename.replace('disk-x', label) # incoming output_filename = args.save_directory + '/' + args.snap + '_Projection_disk-x_temperature_density.png'
        s_filename = p_filename.replace('Projection', 'Slice')

        # "Thin" projections (20 kpc deep).
        p = yt.ProjectionPlot(ds, axis, "temperature", weight_field="density",
                            center=ds.halo_center_code, data_source=region,
                            width=(60,"kpc"), depth=(20,"kpc"),
                            north_vector=ds.z_unit_disk)
        p.set_cmap('temperature', sns.blend_palette(('salmon', "#984ea3", "#4daf4a", "#ffe34d", 'darkorange'), as_cmap=True))
        p.set_zlim('temperature', 1e4,1e7)
        p.annotate_timestamp(corner='upper_left', redshift=True, time=True, draw_inset_box=True)
        p.save(p_filename)
        print('Saved figure ' + p_filename)

        # Slices
        s = yt.SlicePlot(ds, axis, "temperature",
                        center=ds.halo_center_code, data_source=region,
                        width=(60,"kpc"), north_vector=ds.z_unit_disk)
        s.set_cmap('temperature', sns.blend_palette(('salmon', "#984ea3", "#4daf4a", "#ffe34d", 'darkorange'), as_cmap=True))
        s.set_zlim('temperature', 1e4,1e7)
        s.annotate_timestamp(corner='upper_left', redshift=True, time=True, draw_inset_box=True)
        s.save(s_filename)
        print('Saved figure ' + s_filename)
