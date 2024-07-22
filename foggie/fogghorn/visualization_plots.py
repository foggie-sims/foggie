'''
    Filename: visualization_plots.py
    Author: Cassi
    Created: 6-12-24
    Last modified: 7-22-24 by Cassi
    This file works with fogghorn_analysis.py to make a set of basic visualization plots.
    If you add a new function to this scripts, then please also add the function name to the appropriate list at the end of fogghorn/header.py
'''

from foggie.fogghorn.header import *
from foggie.fogghorn.util import *

# --------------------------------------------------------------------------------------------------------------------
def gas_density_projection(ds, region, args, output_filename):
    '''
    Plots a gas density projection of the galaxy disk.
    '''
    for projection in args.projection_arr:
        if '-disk' in projection:
            if 'x' in projection:
                p_dir = ds.x_unit_disk
                north_vector = ds.z_unit_disk
            if 'y' in projection:
                p_dir = ds.y_unit_disk
                north_vector = ds.z_unit_disk
            if 'z' in projection:
                p_dir = ds.z_unit_disk
                north_vector = ds.x_unit_disk
            p = yt.ProjectionPlot(ds, p_dir, 'density', data_source=region, width=(20, 'kpc'), center=ds.halo_center_code, north_vector=north_vector)
        else: p = yt.ProjectionPlot(ds, projection, 'density', data_source=region, width=(20, 'kpc'), center=ds.halo_center_code)
        p.set_unit('density','Msun/pc**2')
        p.set_cmap('density', density_color_map)
        p.set_zlim('density',0.01,300)
        p.set_font_size(16)
        p.annotate_timestamp(corner='upper_left', redshift=True, time=True, draw_inset_box=True)
        output_filename = output_filename[:-4] + '_' + projection + '.png'
        p.save(output_filename)
        print('Saved figure ' + output_filename)

# --------------------------------------------------------------------------------------------------------------------
def gas_metallicity_projection(ds, region, args, output_filename):
    '''
    Plots a gas metallicity projection of the galaxy disk.
    If the --disk_rel argument was used, this function will automatically project w.r.t the disk, instead of the box edges.
    Returns nothing. Saves output as png file
    '''
    if args.disk_rel:
        projection_axis_dict = {'x': ds.x_unit_disk, 'y': ds.y_unit_disk, 'z': ds.z_unit_disk}
        north_vector_dict = {'disk-x': ds.z_unit_disk, 'disk-y': ds.z_unit_disk, 'disk-z': ds.x_unit_disk}

    for projection in args.projection_arr:
        if 'disk' in projection: p = yt.OffAxisProjectionPlot(ds, projection_axis_dict[projection], 'metallicity', data_source=region, width=(20, 'kpc'), center=ds.halo_center_code, north_vector=north_vector_dict[projection])
        else: p = yt.ProjectionPlot(ds, projection, 'metallicity', data_source=region, width=(20, 'kpc'), center=ds.halo_center_code)
        p.set_unit('metallicity','Zsun*cm') # the length dimension is because this is a projected quantity
        p.set_cmap('metallicity', old_metal_color_map)
        #p.set_zlim('metallicity', 2e-2, 4e0)
        output_filename = output_filename[:-4] + '_' + projection + '.png'
        p.save(output_filename)
        print('Saved figure ' + output_filename)

# --------------------------------------------------------------------------------------------------------------------
def edge_visualizations(ds, region, args, output_filename):
    '''
    Plot slices & thin projections of galaxy temperature viewed from the disk edge.
    '''

    # Visualize along two perpendicular edge axes
    for label, axis in zip(["disk-x","disk-y"], [ds.x_unit_disk, ds.y_unit_disk]):

        p_filename = output_filename.replace('disk-x', label) # incoming output_filename = args.save_directory + '/' + args.snap + '_Projection_disk-x_temperature_density.png'
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
