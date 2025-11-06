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
import trident 

# --------------------------------------------------------------------------------------------------------------------
def gas_density_projection_x(ds, region, args, output_filename):
    gas_density_projection(ds, region, args, output_filename, 'x')

def gas_density_projection_y(ds, region, args, output_filename):
    gas_density_projection(ds, region, args, output_filename, 'y')

def gas_density_projection_z(ds, region, args, output_filename):
    gas_density_projection(ds, region, args, output_filename, 'z')

def gas_temperature_projection_x(ds, region, args, output_filename):
    gas_temperature_projection(ds, region, args, output_filename, 'x')

def gas_temperature_projection_y(ds, region, args, output_filename):
    gas_temperature_projection(ds, region, args, output_filename, 'y')          

def gas_temperature_projection_z(ds, region, args, output_filename):
    gas_temperature_projection(ds, region, args, output_filename, 'z')

def gas_h1_projection_x(ds, region, args, output_filename):
    gas_h1_projection(ds, region, args, output_filename, 'x')

def gas_h1_projection_y(ds, region, args, output_filename):
    gas_h1_projection(ds, region, args, output_filename, 'y')

def gas_h1_projection_z(ds, region, args, output_filename):
    gas_h1_projection(ds, region, args, output_filename, 'z')


def gas_H2_projection_x(ds, region, args, output_filename):
    gas_H2_projection(ds, region, args, output_filename, 'x')

def gas_H2_projection_y(ds, region, args, output_filename):
    gas_H2_projection(ds, region, args, output_filename, 'y')

def gas_H2_projection_z(ds, region, args, output_filename):
    gas_H2_projection(ds, region, args, output_filename, 'z')


def gas_mg2_projection_x(ds, region, args, output_filename):
    gas_mg2_projection(ds, region, args, output_filename, 'x')

def gas_mg2_projection_y(ds, region, args, output_filename):
    gas_mg2_projection(ds, region, args, output_filename, 'y')

def gas_mg2_projection_z(ds, region, args, output_filename):
    gas_mg2_projection(ds, region, args, output_filename, 'z')

def gas_o6_projection_x(ds, region, args, output_filename):
    gas_o6_projection(ds, region, args, output_filename, 'x')

def gas_o6_projection_y(ds, region, args, output_filename):
    gas_o6_projection(ds, region, args, output_filename, 'y')               

def gas_o6_projection_z(ds, region, args, output_filename):
    gas_o6_projection(ds, region, args, output_filename, 'z')

def gas_density_projection_x_disk(ds, region, args, output_filename):
    gas_density_projection(ds, region, args, output_filename, 'x-disk')

def gas_density_projection_y_disk(ds, region, args, output_filename):
    gas_density_projection(ds, region, args, output_filename, 'y-disk')

def gas_density_projection_z_disk(ds, region, args, output_filename):
    gas_density_projection(ds, region, args, output_filename, 'z-disk')

# --------------------------------------------------------------------------------------------------------------------
def gas_density_projection(ds, region, args, output_filename, projection):
    '''
    Plots a gas density projection of the galaxy disk.
    '''
    run_name = (os.getcwd()).split("/")[-1]

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
        p = yt.ProjectionPlot(ds, p_dir, 'density', weight_field=('gas','density'), width=(args.proj_width, 'kpc'), center=ds.halo_center_code, north_vector=north_vector)
    else: p = yt.ProjectionPlot(ds, projection, 'density', weight_field=('gas','density'), width=(args.proj_width, 'kpc'), center=ds.halo_center_code)
    p.set_unit('density','g/cm**3')
    p.set_cmap('density', density_color_map)
    p.set_zlim('density',1e-30,1e-24)
    p.set_font_size(16)
    p.annotate_timestamp(corner='upper_left', redshift=True, time=True, draw_inset_box=True)
    p.annotate_title(run_name + '   ' + ds.snapname + '   ' + str(ds.current_datetime.date())) 
    p.save(output_filename)
    print('Saved figure ' + output_filename)

# --------------------------------------------------------------------------------------------------------------------
def gas_temperature_projection(ds, region, args, output_filename, projection):
    '''
    Plots a gas density projection of the galaxy disk.
    '''
    run_name = (os.getcwd()).split("/")[-1]

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
        p = yt.ProjectionPlot(ds, p_dir, 'temperature', weight_field=('gas','density'), width=(args.proj_width, 'kpc'), center=ds.halo_center_code, north_vector=north_vector)
    else: p = yt.ProjectionPlot(ds, projection, 'temperature', weight_field=('gas','density'), width=(args.proj_width, 'kpc'), center=ds.halo_center_code)
    p.set_unit('temperature','K')
    p.set_cmap('temperature', temperature_color_map)
    p.set_zlim('temperature', 100, 1e6)
    p.set_font_size(16)
    p.annotate_timestamp(corner='upper_left', redshift=True, time=True, draw_inset_box=True)
    p.annotate_title(run_name + '   ' + ds.snapname + '   ' + str(ds.current_datetime.date())) 
    p.save(output_filename)
    print('Saved figure ' + output_filename)

# --------------------------------------------------------------------------------------------------------------------
def gas_h1_projection(ds, region, args, output_filename, projection):
    '''
    Plots a gas H I projection of the galaxy disk.
    '''
    run_name = (os.getcwd()).split("/")[-1]

    trident.add_ion_fields(ds, ions=['H I']) 

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
        p = yt.ProjectionPlot(ds, p_dir, ('gas', 'H_p0_number_density'), width=(args.proj_width, 'kpc'), center=ds.halo_center_code, north_vector=north_vector)
    else: p = yt.ProjectionPlot(ds, projection, ('gas', 'H_p0_number_density'), width=(args.proj_width, 'kpc'), center=ds.halo_center_code)
    p.set_unit(('gas', 'H_p0_number_density'),'cm**-2')
    p.set_cmap(('gas', 'H_p0_number_density'), h1_color_map)
    p.set_zlim(('gas', 'H_p0_number_density'), h1_proj_min,h1_proj_max) 
    p.set_font_size(16)
    p.annotate_timestamp(corner='upper_left', redshift=True, time=True, draw_inset_box=True)
    p.annotate_title(run_name + '   ' + ds.snapname + '   ' + str(ds.current_datetime.date())) 
    p.save(output_filename)
    print('Saved figure ' + output_filename)

# --------------------------------------------------------------------------------------------------------------------
def gas_mg2_projection(ds, region, args, output_filename, projection):
    '''
    Plots a gas Mg II projection of the galaxy disk.
    '''
    run_name = (os.getcwd()).split("/")[-1]

    trident.add_ion_fields(ds, ions=['Mg II']) 

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
        p = yt.ProjectionPlot(ds, p_dir, ('gas', 'Mg_p1_number_density'), weight_field=('gas','density'), width=(args.proj_width, 'kpc'), center=ds.halo_center_code, north_vector=north_vector)
    else: p = yt.ProjectionPlot(ds, projection, ('gas', 'Mg_p1_number_density'), width=(args.proj_width, 'kpc'), center=ds.halo_center_code)
    p.set_unit(('gas', 'Mg_p1_number_density'),'cm**-2')
    p.set_cmap(('gas', 'Mg_p1_number_density'), mg2_color_map)
    p.set_zlim(('gas', 'Mg_p1_number_density'), mg2_min,mg2_max) 
    p.set_font_size(16)
    p.annotate_timestamp(corner='upper_left', redshift=True, time=True, draw_inset_box=True)
    p.annotate_title(run_name + '   ' + ds.snapname + '   ' + str(ds.current_datetime.date())) 
    p.set_background_color('Mg_p1_number_density', '#0A0780') 
    p.save(output_filename)
    print('Saved figure ' + output_filename)

# --------------------------------------------------------------------------------------------------------------------
def gas_o6_projection(ds, region, args, output_filename, projection):
    '''
    Plots a gas H I projection of the galaxy disk.
    '''
    run_name = (os.getcwd()).split("/")[-1]

    trident.add_ion_fields(ds, ions=['O VI']) 

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
        p = yt.ProjectionPlot(ds, p_dir, ('gas', 'O_p5_number_density'), weight_field=('gas','density'), width=(args.proj_width, 'kpc'), center=ds.halo_center_code, north_vector=north_vector)
    else: p = yt.ProjectionPlot(ds, projection, ('gas', 'O_p5_number_density'), width=(args.proj_width, 'kpc'), center=ds.halo_center_code)
    p.set_unit(('gas', 'O_p5_number_density'),'cm**-2')
    p.set_cmap(('gas', 'O_p5_number_density'), o6_color_map)
    p.set_zlim(('gas', 'O_p5_number_density'), o6_min,o6_max) 
    p.set_font_size(16)
    p.annotate_timestamp(corner='upper_left', redshift=True, time=True, draw_inset_box=True)
    p.annotate_title(run_name + '   ' + ds.snapname + '   ' + str(ds.current_datetime.date())) 
    p.save(output_filename)
    print('Saved figure ' + output_filename)


def gas_H2_projection(ds, region, args, output_filename, projection):
    '''
    Plots a gas H2 projection of the galaxy disk.
    '''
    run_name = (os.getcwd()).split("/")[-1]

    if (ds.parameters['MultiSpecies'] != 2): # we cannot do this if the fields are not present 
        print("Skipping H2 projection -- MultiSpecies != 2")
        return

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
        p = yt.ProjectionPlot(ds, p_dir, ('gas', 'H2_number_density'), width=(args.proj_width, 'kpc'), center=ds.halo_center_code, north_vector=north_vector)
    else: p = yt.ProjectionPlot(ds, projection, ('gas', 'H2_number_density'), width=(args.proj_width, 'kpc'), center=ds.halo_center_code)
    p.set_unit(('gas', 'H2_number_density'),'cm**-2')
    p.set_cmap(('gas', 'H2_number_density'), 'inferno')
    p.set_zlim(('gas', 'H2_number_density'), 1e10, 1e20)
    p.set_font_size(16)
    p.annotate_timestamp(corner='upper_left', redshift=True, time=True, draw_inset_box=True)
    p.annotate_title(run_name + '   ' + ds.snapname + '   ' + str(ds.current_datetime.date())) 
    p.save(output_filename)
    print('Saved figure ' + output_filename)


# --------------------------------------------------------------------------------------------------------------------
def gas_metallicity_projection_x(ds, region, args, output_filename):
    gas_metallicity_projection(ds, region, args, output_filename, 'x')

def gas_metallicity_projection_y(ds, region, args, output_filename):
    gas_metallicity_projection(ds, region, args, output_filename, 'y')
    
def gas_metallicity_projection_z(ds, region, args, output_filename):
    gas_metallicity_projection(ds, region, args, output_filename, 'z')
    
def gas_metallicity_projection_x_disk(ds, region, args, output_filename):
    gas_metallicity_projection(ds, region, args, output_filename, 'x-disk')
    
def gas_metallicity_projection_y_disk(ds, region, args, output_filename):
    gas_metallicity_projection(ds, region, args, output_filename, 'y-disk')
    
def gas_metallicity_projection_z_disk(ds, region, args, output_filename):
    gas_metallicity_projection(ds, region, args, output_filename, 'z-disk')

# --------------------------------------------------------------------------------------------------------------------
def gas_metallicity_projection(ds, region, args, output_filename, projection):
    '''
    Plots a gas metallicity projection of the galaxy disk.
    If the --disk_rel argument was used, this function will automatically project w.r.t the disk, instead of the box edges.
    Returns nothing. Saves output as png file
    '''

    run_name = (os.getcwd()).split("/")[-1]

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
        p = yt.OffAxisProjectionPlot(ds, p_dir, 'metallicity', weight_field=('gas','density'), width=(args.proj_width, 'kpc'), center=ds.halo_center_code, north_vector=north_vector)
    else: p = yt.ProjectionPlot(ds, projection, 'metallicity', weight_field=('gas','density'), width=(args.proj_width, 'kpc'), center=ds.halo_center_code)
    p.set_unit('metallicity','Zsun')
    p.set_cmap('metallicity', metal_color_map)
    p.set_zlim('metallicity', 1e-4, 10)
    p.annotate_timestamp(corner='upper_left', redshift=True, time=True, draw_inset_box=True)
    p.set_font_size(16)
    p.annotate_title(run_name + '   ' + ds.snapname + '   ' + str(ds.current_datetime.date())) 
    p.save(output_filename)
    print('Saved figure ' + output_filename)

# --------------------------------------------------------------------------------------------------------------------
def edge_projection_x_disk(ds, region, args, output_filename):
    edge_projection(ds, region, args, output_filename, 'x-disk')

def edge_projection_y_disk(ds, region, args, output_filename):
    edge_projection(ds, region, args, output_filename, 'y-disk')

# --------------------------------------------------------------------------------------------------------------------
def edge_projection(ds, region, args, output_filename, projection):
    '''
    Plot slices & thin projections of galaxy temperature viewed from the disk edge.
    '''

    if 'x' in projection:
        p_dir = ds.x_unit_disk
        north_vector = ds.z_unit_disk
    if 'y' in projection:
        p_dir = ds.y_unit_disk
        north_vector = ds.z_unit_disk

    # "Thin" projections (10 kpc deep).
    p = yt.ProjectionPlot(ds, p_dir, "temperature", weight_field="density",
                        center=ds.halo_center_code,
                        width=(60,"kpc"), depth=(10,"kpc"),
                        north_vector=north_vector)
    p.set_cmap('temperature', sns.blend_palette(('salmon', "#984ea3", "#4daf4a", "#ffe34d", 'darkorange'), as_cmap=True))
    p.set_zlim('temperature', 1e4,1e7)
    p.annotate_timestamp(corner='upper_left', redshift=True, time=True, draw_inset_box=True)
    p.save(output_filename)
    print('Saved figure ' + output_filename)

# --------------------------------------------------------------------------------------------------------------------
def edge_slice_x_disk(ds, region, args, output_filename):
    edge_slice(ds, region, args, output_filename, 'x-disk')

def edge_slice_y_disk(ds, region, args, output_filename):
    edge_slice(ds, region, args, output_filename, 'y-disk')

# --------------------------------------------------------------------------------------------------------------------
def edge_slice(ds, region, args, output_filename, projection):
    '''
    Plot slices & thin projections of galaxy temperature viewed from the disk edge.
    '''

    if 'x' in projection:
        p_dir = ds.x_unit_disk
        north_vector = ds.z_unit_disk
    if 'y' in projection:
        p_dir = ds.y_unit_disk
        north_vector = ds.z_unit_disk

    p = yt.SlicePlot(ds, p_dir, "temperature",
                        center=ds.halo_center_code,
                        width=(60,"kpc"),
                        north_vector=north_vector)
    p.set_cmap('temperature', sns.blend_palette(('salmon', "#984ea3", "#4daf4a", "#ffe34d", 'darkorange'), as_cmap=True))
    p.set_zlim('temperature', 1e4,1e7)
    p.annotate_timestamp(corner='upper_left', redshift=True, time=True, draw_inset_box=True)
    p.save(output_filename)
    print('Saved figure ' + output_filename)
