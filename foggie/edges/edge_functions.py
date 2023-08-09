# region and edge analysis functions 
# CL and JT 2022 

import numpy as np, pandas as pd, scipy.ndimage as ndimage, yt, cmyt 
import copy 
yt.set_log_level(40)
from foggie.utils.consistency import axes_label_dict, logfields, categorize_by_temp, \
    categorize_by_metals, categorize_by_fraction
from foggie.utils.yt_fields import *
from foggie.utils.foggie_load import *
import seaborn as sns

def function_for_edges(ds, trackfile, refine_box, box_size = 400., sampling_level = 9): 
    
    # this function creates the basic dictionary containting stuff derived from the dataset 
    # and the dictionary we'll use but none of the region screening 

    pix_res = float(np.min(refine_box[('gas','dx')].in_units('kpc')))  # cell size at highest AMR level 

    lvl1_res = pix_res*2.**11.   # convert to cell size at level 1
                             # specify the level of refinement to match in the covering grid 


    dx = lvl1_res/(2.**sampling_level)       # convert cell size to specified level (could just put in a number here instead of 3 previous lines)
    refine_res = int(box_size/dx)           # calculate resolution of FRB based on desired box size and cell size
    # Now define the actual covering grid based on box_size and refine_res, box is a YTCoveringGrid object 

    print("function_for_edges: creating covering grid of dimension ", refine_res, " on a side.")

    box = ds.covering_grid(level=sampling_level, \
                       left_edge=ds.halo_center_kpc-ds.arr([box_size/2.,box_size/2.,box_size/2.],'kpc'), 
                       dims=[refine_res, refine_res,refine_res])
   
    print("function for edges: assembling output dictionary")

    x_cut_string = "(abs(obj[('stream','x')]) < 2.* 3.086e21)"
    y_cut_string = "(abs(obj[('stream','y')]) < 2.* 3.086e21)"
    z_cut_string = "(abs(obj[('stream','z')]) < 2.* 3.086e21)"

    region_dict = {'trackfile':trackfile, 'refine_box':refine_box, 'box_size':box_size, 'box':box, 'sampling_level':sampling_level,  
                  'x_cut_string':x_cut_string, 'y_cut_string':y_cut_string, 'z_cut_string':z_cut_string, 'refine_res':refine_res}
    
    return region_dict 


def apply_region_cuts_to_dict(region_dict, number_of_edge_iterations = 1, region = 'inflow'): 

    box = region_dict["box"]

    region_dict['region'] = region

    # We need temperature and density to do the region cuts 
    temperature = box['temperature'].v
    density = box['density'].in_units('g/cm**3').v

    if ('disk' in region):
        print('Will produce analysis for region = ', region_dict['region'])
        region_mask = ( temperature < 20000.)
        x_cut_string = "(obj['temperature'] < 20000.) & " + region_dict["x_cut_string"]
        y_cut_string = "(obj['temperature'] < 20000.) & " + region_dict["y_cut_string"]
        z_cut_string = "(obj['temperature'] < 20000.) & " + region_dict["z_cut_string"]
        region_name = 'disk'
    if ('inflow' in region):
        print('Will produce analysis for region = ', region_dict['region'])
        radial_velocity = box['radial_velocity_corrected'].in_units('km/s').v
        vff = box['vff'].in_units('km/s').v
        region_mask = (radial_velocity < vff)
        region_name = 'inflow'
        x_cut_string = "(obj[('stream', 'radial_velocity')] < obj['vff']) & " + region_dict["x_cut_string"]
        y_cut_string = "(obj[('stream', 'radial_velocity')] < obj['vff']) & " + region_dict["y_cut_string"]
        z_cut_string = "(obj[('stream', 'radial_velocity')] < obj['vff']) & " + region_dict["z_cut_string"]
    
 
    # This next block describes how to grab a region and then expand it to grab areas just outside the region 
    # disk_mask is a boolean 3D array with 1's for the disk and 0's elsewhere

    print("function for edges: performing binary dilation")

    # struct is needed just to identify how the later dilation is done geometrically (ie in 3D)
    struct = ndimage.generate_binary_structure(3,3)

    # disk_mask_expanded is a boolean 3D array with 1's for the disk AND the region surrounding it
    region_mask_expanded = ndimage.binary_dilation(region_mask, structure=struct, iterations=number_of_edge_iterations)

    # Dilation expands the cells to include the region surrounding the disk, then closing up any holes
    region_mask_expanded = ndimage.binary_closing(region_mask_expanded, structure=struct, iterations=number_of_edge_iterations)

    # disk_edges is a boolean 3D array with 1's for ONLY pixels surrounding ISM regions -- nothing inside ISM regions
    region_edges = region_mask_expanded & ~region_mask
    

    # Set the density and temperature of everything other than the disk edges to minimum value for viz.
    density_region = np.copy(density)
    density_region[~region_mask] = 1e-40

    temperature_region = np.copy(temperature)
    temperature_region[~region_mask] = 10. 

    density_edges = np.copy(density)
    density_edges[~region_edges] = 1e-40

    temperature_edges = np.copy(temperature)
    temperature_edges[~region_edges] = 10. 

    print("apply_region_cuts_to_dict: adding screened fields to region_dict.")

    region_dict["number_of_edge_iterations"] = number_of_edge_iterations
    region_dict['x_cut_string'] = x_cut_string 
    region_dict['y_cut_string'] = y_cut_string 
    region_dict['z_cut_string'] = z_cut_string 
    region_dict['region_mask'] = region_mask
    region_dict['region_mask_expanded'] = region_mask_expanded 
    region_dict['region_edges'] = region_edges
    region_dict['density_region'] = density_region
    region_dict['temperature_region'] = temperature_region
    region_dict['density_edges'] = density_edges
    region_dict['temperature_edges'] = temperature_edges
    region_dict['region_name'] = region_name

    return region_dict



def merge_two_regions(region1, region2): 

    overlap_mask = region1["region_edges"] & region2["region_edges"]

    #copy one of the dictionaries so we can manipulate its content 
    
    #new_dict = copy.deepcopy(region1)
    new_dict = region1.copy()

    new_dict["region"] = 'overlap'
    
    new_dict['x_cut_string'] = "(obj['temperature_region'] > 20.) & (abs(obj[('stream','x')]) < 2.* 3.086e21)"
    new_dict['y_cut_string'] = "(obj['temperature_region'] > 20.) & (abs(obj[('stream','y')]) < 2.* 3.086e21)"
    new_dict['z_cut_string'] = "(obj['temperature_region'] > 20.) & (abs(obj[('stream','z')]) < 2.* 3.086e21)"

    #reset the temperature and density fields for "region" which are the masked fields
    new_dict['temperature_region'] = new_dict['box']['temperature']
    new_dict['density_region'] = new_dict['box']['density']

    new_dict['density_region'][~overlap_mask] = 1e-40
    new_dict['temperature_region'][~overlap_mask] = 10. 

    new_dict['region_mask'] = overlap_mask

    return new_dict


def convert_to_new_dataset(region_dict, box_size):

    box = region_dict["box"]

    data = dict(temperature = (box['temperature'].v, 'K'), 
            temperature_region = (region_dict["temperature_region"], 'K'), \
            temperature_edges = (region_dict["temperature_edges"], 'K'), 
            density_region = (region_dict["density_region"], 'g/cm**3'), \
            density_edges = (region_dict["density_edges"], 'g/cm**3'), \
            density = (box['density'].v, 'g/cm**3'), \
            radial_velocity = (box['radial_velocity_corrected'].in_units('km/s').v, 'km/s'), 
            tangential_velocity = (box['tangential_velocity_corrected'].in_units('km/s').v, 'km/s'), \
            cooling_time = (box['cooling_time'].in_units('yr').v, 'yr'), \
            metallicity = box['cooling_time'], \
            tcool_tff = (box['tcool_tff'].v, 'dimensionless'), \
            radius = (box['radius_corrected'].in_units('kpc').v, 'kpc'), \
            vff = (box['vff'].in_units('km/s').v, 'km/s')) 

    bbox = np.array([[-0.5*box_size, 0.5*box_size], [-0.5*box_size, 0.5*box_size], [-0.5*box_size, 0.5*box_size]])

    grid = yt.load_uniform_grid(data, box['temperature'].shape, length_unit="kpc", bbox=bbox)
 
    return grid



### (ONE) On-axis projections of original dataset 
def plot_one(ds, region_dict, dataset_name): 
    proj = yt.ProjectionPlot(ds, 'x', 'temperature', center=ds.halo_center_kpc, weight_field=('gas', 'density'), width=(region_dict["box_size"],'kpc'))
    proj.set_cmap(('gas','temperature'), sns.blend_palette(('salmon', "#984ea3", "#4daf4a", "#ffe34d", 'darkorange'), as_cmap=True))
    proj.annotate_timestamp(redshift=True)
    proj.set_zlim(('gas','temperature'), 1e4,1e7)
    proj.save(dataset_name)
 
    proj = yt.ProjectionPlot(ds, 'y', 'temperature', center=ds.halo_center_kpc, weight_field=('gas', 'density'), width=(region_dict["box_size"],'kpc'))
    proj.set_cmap(('gas','temperature'), sns.blend_palette(('salmon', "#984ea3", "#4daf4a", "#ffe34d", 'darkorange'), as_cmap=True))
    proj.annotate_timestamp(redshift=True)
    proj.set_zlim(('gas','temperature'), 1e4,1e7)
    proj.save(dataset_name)
 
    proj = yt.ProjectionPlot(ds, 'z', 'temperature', center=ds.halo_center_kpc, weight_field=('gas', 'density'), width=(region_dict["box_size"],'kpc'))
    proj.set_cmap(('gas','temperature'), sns.blend_palette(('salmon', "#984ea3", "#4daf4a", "#ffe34d", 'darkorange'), as_cmap=True))
    proj.annotate_timestamp(redshift=True)
    proj.set_zlim(('gas','temperature'), 1e4,1e7)
    proj.save(dataset_name)


### (TWO) On-axis projection plots of the full covering grid dataset 
def plot_two(ds, region_dict, dataset_name): 

    proj = yt.ProjectionPlot(ds, 'x', ('gas','temperature'), weight_field=('gas', 'density'), width=(region_dict["box_size"],'kpc'))
    proj.set_cmap(('gas','temperature'), sns.blend_palette(('salmon', "#984ea3", "#4daf4a", "#ffe34d", 'darkorange'), as_cmap=True))
    proj.set_zlim(('gas','temperature'), 1e4,1e7)
    proj.save(dataset_name+'_dilated')
 
    proj = yt.ProjectionPlot(ds, 'y', ('gas','temperature'), weight_field=('gas', 'density'), width=(region_dict["box_size"],'kpc'))
    proj.set_cmap(('gas','temperature'), sns.blend_palette(('salmon', "#984ea3", "#4daf4a", "#ffe34d", 'darkorange'), as_cmap=True))
    proj.set_zlim(('gas','temperature'), 1e4,1e7)
    proj.save(dataset_name+'_dilated')
 
    proj = yt.ProjectionPlot(ds, 'z', ('gas','temperature'), weight_field=('gas', 'density'), width=(region_dict["box_size"],'kpc'))
    proj.set_cmap(('gas','temperature'), sns.blend_palette(('salmon', "#984ea3", "#4daf4a", "#ffe34d", 'darkorange'), as_cmap=True))
    proj.set_zlim(('gas','temperature'), 1e4,1e7)
    proj.save(dataset_name+'_dilated')


### (THREE) Quasi-slice of the covering grid 
def plot_three(grid, region_dict, dataset_name): 
    ad = grid.all_data() 
    box_size = region_dict["box_size"] 
    quasi_slice = ad.cut_region("(abs(obj[('stream','x')]) < 2.* 3.086e21)")
    proj = yt.ProjectionPlot(grid, 'x', ('stream','temperature'), data_source=quasi_slice, weight_field=('stream','density'), width=(region_dict["box_size"],'kpc'))
    proj.set_cmap(('stream','temperature'), sns.blend_palette(('salmon', "#984ea3", "#4daf4a", "#ffe34d", 'darkorange'), as_cmap=True))
    proj.set_zlim(('stream','temperature'), 1e4,1e7)
    proj.save(dataset_name+'_dilated_grid')
 
    quasi_slice = ad.cut_region("(abs(obj[('stream','y')]) < 2.* 3.086e21)")
    proj = yt.ProjectionPlot(grid, 'y', ('stream','temperature'), data_source=quasi_slice, weight_field=('stream','density'), width=(region_dict["box_size"],'kpc'))
    proj.set_cmap(('stream','temperature'), sns.blend_palette(('salmon', "#984ea3", "#4daf4a", "#ffe34d", 'darkorange'), as_cmap=True))
    proj.set_zlim(('stream','temperature'), 1e4,1e7)
    proj.save(dataset_name+'_dilated_grid')
 
    quasi_slice = ad.cut_region("(abs(obj[('stream','z')]) < 2.* 3.086e21)")
    proj = yt.ProjectionPlot(grid, 'z', ('stream','temperature'), data_source=quasi_slice, weight_field=('stream','density'), width=(region_dict["box_size"],'kpc'))
    proj.set_cmap(('stream','temperature'), sns.blend_palette(('salmon', "#984ea3", "#4daf4a", "#ffe34d", 'darkorange'), as_cmap=True))
    proj.set_zlim(('stream','temperature'), 1e4,1e7)
    proj.save(dataset_name+'_dilated_grid')


### (FOUR) Quasi-slice plots of the region 
def plot_four(grid, region_dict, dataset_name): 
    ad = grid.all_data() 

    cut_region = ad.cut_region(region_dict['x_cut_string'])
    proj = yt.ProjectionPlot(grid, 'x', ('gas','temperature'), data_source=cut_region, weight_field='density', width=(region_dict["box_size"],'kpc'))
    proj.set_cmap(('gas','temperature'), sns.blend_palette(('salmon', "#984ea3", "#4daf4a", "#ffe34d", 'darkorange'), as_cmap=True))
    proj.set_zlim(('gas','temperature'), 1e4,1e7)
    proj.save(dataset_name+'_x_'+region_dict["region"])
 
    cut_region = ad.cut_region(region_dict['y_cut_string'])
    proj = yt.ProjectionPlot(grid, 'y', ('gas','temperature'), data_source=cut_region, weight_field='density', width=(region_dict["box_size"],'kpc'))
    proj.set_cmap(('gas','temperature'), sns.blend_palette(('salmon', "#984ea3", "#4daf4a", "#ffe34d", 'darkorange'), as_cmap=True))
    proj.set_zlim(('gas','temperature'), 1e4,1e7)
    proj.save(dataset_name+'_y_'+region_dict["region"])
 
    cut_region = ad.cut_region(region_dict['z_cut_string'])
    proj = yt.ProjectionPlot(grid, 'z', ('gas','temperature'), data_source=cut_region, weight_field='density', width=(region_dict["box_size"],'kpc'))
    proj.set_cmap(('gas','temperature'), sns.blend_palette(('salmon', "#984ea3", "#4daf4a", "#ffe34d", 'darkorange'), as_cmap=True))
    proj.set_zlim(('gas','temperature'), 1e4,1e7)
    proj.save(dataset_name+'_z_'+region_dict["region"])    


### (FIVE) on-axis quasi-slice plots of the edges - actually they are thin projections
def plot_five(grid, region_dict, dataset_name): 
    ad = grid.all_data() 

    x_cut_string = "(obj['temperature_edges'] > 20.) & (abs(obj[('stream','x')]) < 2.* 3.086e21)"
    edges_region = ad.cut_region(x_cut_string)
    proj = yt.ProjectionPlot(grid, 'x', ('stream','temperature_edges'), data_source=edges_region, weight_field='density', width=(region_dict["box_size"],'kpc'))
    proj.set_cmap(('stream','temperature_edges'), sns.blend_palette(('salmon', "#984ea3", "#4daf4a", "#ffe34d", 'darkorange'), as_cmap=True))
    proj.set_zlim(('stream','temperature_edges'), 1e4,1e7)
    proj.save(dataset_name+'_'+region_dict["region"]+'_edges')
 
    y_cut_string = "(obj['temperature_edges'] > 20.) & (abs(obj[('stream','y')]) < 2.* 3.086e21)"
    edges_region = ad.cut_region(y_cut_string)
    proj = yt.ProjectionPlot(grid, 'y', ('stream','temperature_edges'), data_source=edges_region, weight_field='density', width=(region_dict["box_size"],'kpc'))
    proj.set_cmap(('stream','temperature_edges'), sns.blend_palette(('salmon', "#984ea3", "#4daf4a", "#ffe34d", 'darkorange'), as_cmap=True))
    proj.set_zlim(('stream','temperature_edges'), 1e4,1e7)
    proj.save(dataset_name+'_'+region_dict["region"]+'_edges')

    z_cut_string = "(obj['temperature_edges'] > 20.) & (abs(obj[('stream','z')]) < 2.* 3.086e21)"
    edges_region = ad.cut_region(z_cut_string)
    proj = yt.ProjectionPlot(grid, 'z', ('stream','temperature_edges'), data_source=edges_region, weight_field='density', width=(region_dict["box_size"],'kpc'))
    proj.set_cmap(('stream','temperature_edges'), sns.blend_palette(('salmon', "#984ea3", "#4daf4a", "#ffe34d", 'darkorange'), as_cmap=True))
    proj.set_zlim(('stream','temperature_edges'), 1e4,1e7)
    proj.save(dataset_name+'_'+region_dict["region"]+'_edges')



def datashades( region_grid, region, dataset_name ): 
    
    # take in region dictionary and its recreated dataset 

    import foggie.render.shade_maps as sm 

    import datashader as dshader
    from datashader.utils import export_image
    import datashader.transfer_functions as tf

    refine_res = region['refine_res']
    #ad = region_grid.all_data() 

    #first we create a dataframe
    d = {'temperature': np.log10(np.reshape(region_grid[("gas", "temperature")], refine_res**3)), 
         'density': np.log10(np.reshape(region_grid[("gas", "density")], refine_res**3)),
        'density_region': np.log10(np.reshape(region_grid['density_region'], refine_res**3)),
        'temperature_region': np.log10(np.reshape(region_grid['temperature_region'], refine_res**3)),
         'temperature_edges': np.log10(np.reshape(region_grid['temperature_edges'], refine_res**3)),
         'density_edges': np.log10(np.reshape(region_grid['density_edges'], refine_res**3)),
         'radial_velocity': np.reshape(region_grid[('stream','radial_velocity')], refine_res**3), 
         'tangential_velocity':np.reshape(region_grid[('stream','tangential_velocity')], refine_res**3),
         'cell_mass':np.log10(np.reshape(region_grid['cell_mass'], refine_res**3)), 
         'metallicity':np.reshape(region_grid['metallicity'], refine_res**3),
         'cooling_time':np.log10(np.reshape(region_grid['cooling_time'], refine_res**3)),
         'radius':np.reshape(region_grid['radius'], refine_res**3), 
         'vff':np.reshape(region_grid['vff'], refine_res**3)} 
    df = pd.DataFrame(data=d)
    df['phase'] = categorize_by_temp(df['temperature'])
    df.phase = df.phase.astype('category')
    df['metal'] = categorize_by_metals(df['metallicity'])
    df.metal = df.metal.astype('category')


    #full phase diagram 
    cvs = dshader.Canvas(plot_width=600, plot_height=600, x_range=[-32,-22], y_range=[1,8])
    agg = cvs.points(df, 'density', 'temperature', dshader.count_cat('phase'))
    img = tf.spread(tf.shade(agg, how='eq_hist', color_key=colormap_dict['phase'], min_alpha=40), shape='square', px=0)
    export_image(img,dataset_name+'_full_phase')

    #full velocities plot 
    cvs = dshader.Canvas(plot_width=600, plot_height=600, x_range=[-500,500], y_range=[-50,550])
    agg = cvs.points(df, 'radial_velocity', 'tangential_velocity', dshader.count_cat('phase'))
    img = tf.spread(tf.shade(agg, how='eq_hist', color_key=colormap_dict['phase'], min_alpha=40), shape='square', px=0)
    export_image(img,dataset_name+'_full_velocities')

    #full velocities plot 
    cvs = dshader.Canvas(plot_width=600, plot_height=600, x_range=[0,400], y_range=[-500,500])
    agg = cvs.points(df, 'radius', 'radial_velocity', dshader.count_cat('phase'))
    img = tf.spread(tf.shade(agg, how='eq_hist', color_key=colormap_dict['phase'], min_alpha=40), shape='square', px=0)
    export_image(img,dataset_name+'_full_rv')


    #full cell mass plot 
    cvs = dshader.Canvas(plot_width=600, plot_height=600, x_range=[1,8], y_range=[0,6])
    agg = cvs.points(df, 'temperature', 'cell_mass', dshader.count_cat('phase'))
    img = tf.spread(tf.shade(agg, how='eq_hist', color_key=colormap_dict['phase'], min_alpha=40), shape='square', px=0)
    export_image(img,dataset_name+'_full_cell_mass')

    #full cell mass plot 
    cvs = dshader.Canvas(plot_width=600, plot_height=600, x_range=[1,8], y_range=[4,12])
    agg = cvs.points(df, 'temperature', 'cooling_time', dshader.count_cat('phase'))
    img = tf.spread(tf.shade(agg, how='eq_hist', color_key=colormap_dict['phase'], min_alpha=40), shape='square', px=0)
    export_image(img,dataset_name+'_full_cooling_time')


    # disk edge phase diagram 
    cvs = dshader.Canvas(plot_width=600, plot_height=600, x_range=[-32,-22], y_range=[1,8])
    agg = cvs.points(df[df.temperature_edges > 1.5], 'density', 'temperature', dshader.count_cat('phase'))
    img = tf.spread(tf.shade(agg, how='eq_hist', color_key=colormap_dict['phase'], min_alpha=40), shape='square', px=0)
    export_image(img,dataset_name+'_'+region['region_name']+'_edge_phase')

    # disk edge velocity plot 
    cvs = dshader.Canvas(plot_width=600, plot_height=600, x_range=[-500.,500.], y_range=[-50,550])
    agg = cvs.points(df[df.temperature_edges > 1.5], 'radial_velocity', 'tangential_velocity', dshader.count_cat('phase'))
    img = tf.spread(tf.shade(agg, how='eq_hist', color_key=colormap_dict['phase'], min_alpha=40), shape='square', px=0)
    export_image(img,dataset_name+'_'+region['region_name']+'_edge_velocities')


    #full velocities plot 
    cvs = dshader.Canvas(plot_width=600, plot_height=600, x_range=[0,400], y_range=[-500,500])
    agg = cvs.points(df[df.temperature_edges > 1.5], 'radius', 'radial_velocity', dshader.count_cat('phase'))
    img = tf.spread(tf.shade(agg, how='eq_hist', color_key=colormap_dict['phase'], min_alpha=40), shape='square', px=0)
    export_image(img,dataset_name+'_'+region['region_name']+'_edge_rv')


    # disk edge cell_mass plot 
    cvs = dshader.Canvas(plot_width=600, plot_height=600, x_range=[1,8], y_range=[0,6])
    agg = cvs.points(df[df.temperature_edges > 1.5], 'temperature', 'cell_mass', dshader.count_cat('phase'))
    img = tf.spread(tf.shade(agg, how='eq_hist', color_key=colormap_dict['phase'], min_alpha=40), shape='square', px=0)
    export_image(img,dataset_name+'_'+region['region_name']+'_edge_cell_mass')

    # disk edge cooling_time plot 
    cvs = dshader.Canvas(plot_width=600, plot_height=600, x_range=[1,8], y_range=[4,12])
    agg = cvs.points(df[df.temperature_edges > 1.5], 'temperature', 'cooling_time', dshader.count_cat('phase'))
    img = tf.spread(tf.shade(agg, how='eq_hist', color_key=colormap_dict['phase'], min_alpha=40), shape='square', px=0)
    export_image(img,dataset_name+'_'+region['region_name']+'_edge_cooling_time')


    #disk only phase diagram 
    cvs = dshader.Canvas(plot_width=600, plot_height=600, x_range=[-32,-22], y_range=[1,8])
    agg = cvs.points(df, 'density_region', 'temperature_region', dshader.count_cat('phase'))
    img = tf.spread(tf.shade(agg, how='eq_hist', color_key=colormap_dict['phase'], min_alpha=40), shape='square', px=0)
    export_image(img,dataset_name+'_disk_phase')


    #disk only velocity plot
    cvs = dshader.Canvas(plot_width=600, plot_height=600, x_range=[-500,500], y_range=[-50,550])
    agg = cvs.points(df[df.temperature_region > 1.5], 'radial_velocity', 'tangential_velocity', dshader.count_cat('phase'))
    img = tf.spread(tf.shade(agg, how='eq_hist', color_key=colormap_dict['phase'], min_alpha=40), shape='square', px=0)
    export_image(img,dataset_name+'_disk_velocities')

    #full velocities plot 
    cvs = dshader.Canvas(plot_width=600, plot_height=600, x_range=[0,400], y_range=[-500,500])
    agg = cvs.points(df[df.temperature_region > 1.5], 'radius', 'radial_velocity', dshader.count_cat('phase'))
    img = tf.spread(tf.shade(agg, how='eq_hist', color_key=colormap_dict['phase'], min_alpha=40), shape='square', px=0)
    export_image(img,dataset_name+'_disk_rv')


    #disk only cell masses
    cvs = dshader.Canvas(plot_width=600, plot_height=600, x_range=[1,8], y_range=[0,6])
    agg = cvs.points(df[df.temperature_region > 1.5], 'temperature', 'cell_mass', dshader.count_cat('phase'))
    img = tf.spread(tf.shade(agg, how='eq_hist', color_key=colormap_dict['phase'], min_alpha=40), shape='square', px=0)
    export_image(img,dataset_name+'_disk_cell_masses')


    #full cell mass plot 
    cvs = dshader.Canvas(plot_width=600, plot_height=600, x_range=[1,8], y_range=[4,12])
    agg = cvs.points(df[df.temperature_region > 1.5], 'temperature', 'cooling_time', dshader.count_cat('phase'))
    img = tf.spread(tf.shade(agg, how='eq_hist', color_key=colormap_dict['phase'], min_alpha=40), shape='square', px=0)
    export_image(img,dataset_name+'_disk_cooling_time')

