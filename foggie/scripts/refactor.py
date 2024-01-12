import yt 
import trident 
import numpy as np 
import foggie.utils.foggie_load as foggie_load
import foggie.utils.get_region as gr 
from foggie.utils.consistency import * 
from astropy.table import Table
from astropy.cosmology import WMAP9 as cosmo
import os, argparse 
from astropy import units as u
import foggie.render.shade_maps as sm
from foggie.utils import prep_dataframe

TRACKFILE = os.getenv('FOGGIE_REPO') + 'halo_tracks/008508/nref11n_selfshield_15/halo_track_200kpc_nref10'
HALO_C_V_FILE = os.getenv('FOGGIE_REPO') + 'halo_infos/008508/nref11c_nref9f/halo_cen_smoothed' 

def ludicrous_views(ds, ds_disk): 

    for ax in ['x', 'y', 'x']: 
        p = yt.ProjectionPlot(ds, ax, 'density', center = ds.halo_center_code, weight_field='density', width=(60, 'kpc'))
        p.set_cmap('density', density_color_map)
        p.set_zlim('density', 1e-30, 1e-22)
        p.save()

        p = yt.ProjectionPlot(ds, ax, 'temperature', center = ds.halo_center_code, weight_field='density', width=(60, 'kpc'))
        p.set_zlim('temperature', 3e3, 1e6)
        p.set_cmap('temperature', 'magma')
        p.save()

        p = yt.ProjectionPlot(ds, ax, 'metallicity', center = ds.halo_center_code, weight_field='density', width=(60, 'kpc'))
        p.set_cmap('metallicity', metal_color_map)
        p.set_zlim('metallicity', 0.003, 3)
        p.save()

        p = yt.ProjectionPlot(ds, ax, 'H_p0_number_density', center = ds.halo_center_code, width=(60, 'kpc'))
        p.set_cmap('H_p0_number_density', h1_color_map)
        p.set_zlim('H_p0_number_density', h1_proj_min, h1_proj_max) 
        p.save()

        p = yt.ProjectionPlot(ds, ax, 'O_p5_number_density', center = ds.halo_center_code, width=(60, 'kpc'))
        p.set_cmap('O_p5_number_density', o6_color_map)
        p.set_zlim('O_p5_number_density', o6_min, o6_max) 
        p.save()

        p = yt.ProjectionPlot(ds, ax, 'Mg_p1_number_density', center = ds.halo_center_code, width=(60, 'kpc'))
        p.set_cmap('Mg_p1_number_density', mg2_color_map) 
        p.set_zlim('Mg_p1_number_density', mg2_min, mg2_max) 
        p.save()

    snapname = ds.filename[-6:]

    # Edge-on "z" Projection
    p = yt.ProjectionPlot(ds_disk, ds_disk.x_unit_disk, 'density', north_vector=ds_disk.z_unit_disk, center = ds_disk.halo_center_code, weight_field='density', width=(60, 'kpc'))
    p.set_cmap('density', density_color_map)
    p.set_zlim('density', dens_phase_min, dens_phase_max)
    p.save(snapname+'_Edge_x')

    p = yt.ProjectionPlot(ds_disk, ds_disk.x_unit_disk, 'temperature', north_vector=ds_disk.z_unit_disk, center = ds_disk.halo_center_code, weight_field='density', width=(60, 'kpc'))
    p.set_cmap('temperature', temperature_color_map)
    p.set_zlim('temperature', temperature_min, temperature_max)
    p.save(snapname+'_Edge_x')

    p = yt.ProjectionPlot(ds_disk, ds_disk.x_unit_disk, 'metallicity', north_vector=ds_disk.z_unit_disk, center = ds_disk.halo_center_code, weight_field='density', width=(60, 'kpc'))
    p.set_cmap('metallicity', metal_color_map)
    p.set_zlim('metallicity', metal_min, metal_max) 
    p.save(snapname+'_Edge_x')

    p = yt.ProjectionPlot(ds_disk, ds_disk.y_unit_disk, 'density', north_vector=ds_disk.z_unit_disk, center = ds_disk.halo_center_code, weight_field='density', width=(60, 'kpc'))
    p.set_cmap('density', density_color_map)
    p.set_zlim('density', dens_phase_min, dens_phase_min)
    p.save(snapname+'_Edge_y')

    p = yt.ProjectionPlot(ds_disk, ds_disk.y_unit_disk, 'temperature', north_vector=ds_disk.z_unit_disk, center = ds_disk.halo_center_code, weight_field='density', width=(60, 'kpc'))
    p.set_zlim('temperature', temperature_min, temperature_max) 
    p.set_cmap('temperature', temperature_color_map) 
    p.save(snapname+'_Edge_y')

    p = yt.ProjectionPlot(ds_disk, ds_disk.y_unit_disk, 'metallicity', north_vector=ds_disk.z_unit_disk, center = ds_disk.halo_center_code, weight_field='density', width=(60, 'kpc'))
    p.set_cmap('metallicity', metal_color_map)
    p.set_zlim('metallicity', metal_min, metal_max) 
    p.save(snapname+'_Edge_y')

    ### Now try an oriented face-on projection
    p = yt.ProjectionPlot(ds_disk, ds_disk.z_unit_disk, 'density', north_vector=ds_disk.x_unit_disk, center = ds_disk.halo_center_code, weight_field='density', width=(60, 'kpc'))
    p.set_cmap('density', density_color_map)
    p.set_zlim('density', dens_phase_min, dens_phase_min)
    p.save(snapname+'_Face_z')

    p = yt.ProjectionPlot(ds_disk, ds_disk.z_unit_disk, 'temperature', north_vector=ds_disk.x_unit_disk, center = ds_disk.halo_center_code, weight_field='density', width=(60, 'kpc'))
    p.set_zlim('temperature', temperature_min, temperature_max) 
    p.set_cmap('temperature', temperature_color_map) 
    p.save(snapname+'_Face_z')

    p = yt.ProjectionPlot(ds_disk, ds_disk.z_unit_disk, 'metallicity', north_vector=ds_disk.x_unit_disk, center = ds_disk.halo_center_code, weight_field='density', width=(60, 'kpc'))
    p.set_cmap('metallicity', metal_color_map)
    p.set_zlim('metallicity', metal_min, metal_max) 
    p.save(snapname+'_Face_z')

    

def get_and_prepare_dataset(ds_name):  
    """ This function obtains the dataset and all the usual cut_regions
        It returns a dictionary, keyed by region name, containing each 
        cut region we use. """
    
    # load the dataset the default way 
    ds, _ = foggie_load.foggie_load(ds_name, TRACKFILE) 

    ds_disk, refine_box = foggie_load.foggie_load(ds_name, TRACKFILE, 
                                 particle_type_for_angmom='gas', disk_relative=True) 
    
    trident.add_ion_fields(ds, ions=['H I','C II', 'C III', 'C IV', 'O I', 'O II', 'O III', 'O IV', 'O V', 'O VI', 'O VII', 'O VIII', 'Mg II']) 
    
    # a blank dictionary to contain all the cut regions we'll use downstream 
    cut_region_dict = {}
        
    
    cut_region_dict['cgm'] = gr.get_region(ds, 'cgm') 
    cut_region_dict['rvir'] = gr.get_region(ds, 'rvir') 
    #cut_region_dict['trackbox'] = gr.get_region(ds, 'trackbox') 

    cut_region_dict['cool_cgm'] = gr.get_region(ds, 'cgm', filter=cool_cgm_filter)
    cut_region_dict['warm_cgm'] = gr.get_region(ds, 'cgm', filter=warm_cgm_filter)

    cut_region_dict['cgm_outflows'] = gr.get_region(ds, 'cgm', filter=cgm_outflow_filter)
    cut_region_dict['cool_outflows'] = gr.get_region(ds, 'cgm', filter=cool_outflow_filter)
    cut_region_dict['warm_outflows'] = gr.get_region(ds, 'cgm', filter=warm_outflow_filter)

    cut_region_dict['cgm_inflows'] = gr.get_region(ds, 'cgm', filter=cgm_inflow_filter)
    cut_region_dict['cool_inflows'] = gr.get_region(ds, 'cgm', filter=cool_inflow_filter)
    cut_region_dict['warm_inflows'] = gr.get_region(ds, 'cgm', filter=warm_inflow_filter)
    
    return ds, ds_disk, cut_region_dict 

def velocities(dataset, region, region_name, prefix): 

    print("running velocities for region ", region )

    field_list = [('gas','x'), ('gas','y'), ('gas','radius_corrected'), 'temperature', \
 			('gas', 'radial_velocity_corrected'), ('gas', 'tangential_velocity_corrected'), ('gas', 'O_p5_ion_fraction')] 

    data_frame = prep_dataframe.prep_dataframe(region, field_list, 'phase')

    print(data_frame)

    #first we do the 'normal' plots that are NOT O VI filtered 
    filename = prefix+'normal/x_y/'+dataset.parameter_filename[-6:]+'_x_y_'+region_name+'_phase' 
    image = sm.render_image(data_frame, 'x', 'y', 'phase', (-200,200),(-200,200), filename) 
    sm.wrap_axes(dataset, image, filename, 'x', 'y', 'phase', ((-200,200),(-200,200)), region_name, filter=None)

    filename = prefix+'normal/r_temp/'+dataset.parameter_filename[-6:]+'_radius_temperature_'+region_name+'_phase' 
    image = sm.render_image(data_frame, 'radius_corrected', 'temperature', 'phase', (0,200),(1, 8), filename)       
    sm.wrap_axes(dataset, image, filename, 'radius_corrected', 'temperature', 'phase', ((0,200),(1,8)), region_name, filter=None)

    filename = prefix+'normal/rv_tv/'+dataset.parameter_filename[-6:]+'_rv_tv_'+region_name+'_phase' 
    image = sm.render_image(data_frame, 'radial_velocity_corrected', 'tangential_velocity_corrected', 'phase', (-500,500),(-50,500), filename)       
    sm.wrap_axes(dataset, image, filename, 'radial_velocity_corrected', 'tangential_velocity_corrected', 'phase', ((-500,500),(-50,500)), region_name, filter=None)

    filename = prefix+'normal/r_tv/'+dataset.parameter_filename[-6:]+'_r_tv_'+region_name+'_phase' 
    image = sm.render_image(data_frame, 'radius_corrected', 'tangential_velocity_corrected', 'phase', (0,200),(-50,500), filename)       
    sm.wrap_axes(dataset, image, filename, 'radius_corrected', 'tangential_velocity_corrected', 'phase', ((0,200),(-50,500)), region_name, filter=None)

    filename = prefix+'normal/r_rv/'+dataset.parameter_filename[-6:]+'_r_rv_'+region_name+'_phase' 
    image = sm.render_image(data_frame, 'radius_corrected', 'radial_velocity_corrected', 'phase', (0,200),(-500,500), filename)       
    sm.wrap_axes(dataset, image, filename, 'radius_corrected', 'radial_velocity_corrected', 'phase', ((0,200),(-500,500)), region_name, filter=None)



    # now do it again but filtered by O VI this time. 
    mask = (data_frame['O_p5_ion_fraction'] > 0.1) & (data_frame['O_p5_ion_fraction'] < 1.)

    filename = prefix+'fOVI/x_y/'+dataset.parameter_filename[-6:]+'_x_y_'+region_name+'_phase_fOVI' 
    image = sm.render_image(data_frame[mask], 'x', 'y', 'phase', (-200,200),(-200,200), filename) 
    sm.wrap_axes(dataset, image, filename, 'x', 'y', 'phase', ((-200,200),(-200,200)), region_name, filter=None)

    filename = prefix+'fOVI/r_temp/'+dataset.parameter_filename[-6:]+'_radius_temperature_'+region_name+'_phase_fOVI' 
    image = sm.render_image(data_frame[mask], 'radius_corrected', 'temperature', 'phase', (0,200),(1, 8), filename)       
    sm.wrap_axes(dataset, image, filename, 'radius_corrected', 'temperature', 'phase', ((0,200),(1,8)), region_name, filter=None)

    filename = prefix+'fOVI/rv_tv/'+dataset.parameter_filename[-6:]+'_rv_tv_'+region_name+'_phase_fOVI' 
    image = sm.render_image(data_frame[mask], 'radial_velocity_corrected', 'tangential_velocity_corrected', 'phase', (-500,500),(-50,500), filename)       
    sm.wrap_axes(dataset, image, filename, 'radial_velocity_corrected', 'tangential_velocity_corrected', 'phase', ((-500,500),(-50,500)), region_name, filter=None)

    filename = prefix+'fOVI/r_tv/'+dataset.parameter_filename[-6:]+'_r_tv_'+region_name+'_phase_fOVI' 
    image = sm.render_image(data_frame[mask], 'radius_corrected', 'tangential_velocity_corrected', 'phase', (0,200),(-50,500), filename)       
    sm.wrap_axes(dataset, image, filename, 'radius_corrected', 'tangential_velocity_corrected', 'phase', ((0,200),(-50,500)), region_name, filter=None)

    filename = prefix+'fOVI/r_rv/'+dataset.parameter_filename[-6:]+'_r_rv_'+region_name+'_phase_fOVI' 
    image = sm.render_image(data_frame[mask], 'radius_corrected', 'radial_velocity_corrected', 'phase', (0,200),(-500,500), filename)       
    sm.wrap_axes(dataset, image, filename, 'radius_corrected', 'radial_velocity_corrected', 'phase', ((0,200),(-500,500)), region_name, filter=None)

def frb_radius(ds, prefix):  
    #radius at this redshift using a slice 
    r = yt.SlicePlot(ds, 'z', ('gas','radius_corrected'), center=ds.halo_center_code, width=(200, 'kpc'))
    radius_frb = r.data_source.to_frb((200., "kpc"), 512)
    radius_frb.save_as_dataset(filename=prefix+'radius/'+ds_name[-6:]+'_radius', fields=[('gas', 'radius_corrected')]) 

def TBDfrb_foviTBD(ds_name): 

    data_set = yt.load(ds_name)
    data_set, refine_box = foggie_load.foggie_load(ds_name, track, disk_relative=True, particle_type_angmom='young_stars')
    trident.add_ion_fields(data_set, ions=['H I','C II', 'C III', 'C IV', 'O I', 'O II', 'O III', 'O IV', 'O V', 'O VI', 'O VII', 'O VIII', 'Mg II'])

    #CGM - all temperatures and velocities, fOVI > 0.1
    cgm = gr.get_region(data_set, 'cgm', filter="obj['O_p5_ion_fraction'] > 0.1") 
    p = yt.ProjectionPlot(data_set, 'z', 'O_p5_number_density', data_source=cgm, center=cgm.base_object.dobj1.center, width=(200, 'kpc')) 
    proj_frb = p.data_source.to_frb((200., "kpc"), 512)
    proj_frb.save_as_dataset(filename='cgm_fOVI/'+ds_name[-6:]+'_cgm_fOVI_frb', fields=['density', 'H_p0_number_density', 'O_p5_number_density'])

    #Cool CGM - all velocities, fOVI > 0.1
    cool_cgm = gr.get_region(data_set, 'cgm', filter=" (obj['O_p5_ion_fraction'] > 0.1) & ((obj['temperature'] > 1.5e4) | (obj['density'] < 2e-26)) & (obj['temperature'] < 1e5)" )
    p = yt.ProjectionPlot(data_set, 'z', 'O_p5_number_density', data_source=cool_cgm, center=cgm.base_object.dobj1.center, width=(200, 'kpc'))
    proj_frb = p.data_source.to_frb((200., "kpc"), 512)
    proj_frb.save_as_dataset(filename='cgm_fOVI/'+ds_name[-6:]+'_cool_cgm_fOVI_frb', fields=['density', 'H_p0_number_density', 'O_p5_number_density'])
    
    #Warm CGM - all velocities, fOVI > 0.1
    warm_cgm = gr.get_region(data_set, 'cgm', filter=" (obj['O_p5_ion_fraction'] > 0.1) & ((obj['temperature'] > 1.5e4) | (obj['density'] < 2e-26)) & (obj['temperature'] > 1e5)" )
    p = yt.ProjectionPlot(data_set, 'z', 'O_p5_number_density', data_source=warm_cgm, center=cgm.base_object.dobj1.center, width=(200, 'kpc'))
    proj_frb = p.data_source.to_frb((200., "kpc"), 512)
    proj_frb.save_as_dataset(filename='cgm_fOVI/'+ds_name[-6:]+'_warm_cgm_fOVI_frb', fields=['density', 'H_p0_number_density', 'O_p5_number_density'])

def regions_to_frbs(ds, region, region_name, prefix):  
    """ This function creates the column density plots and FRBs.
        It's passed a dataset (ds) and the cut_region. It operates
        on only one  cut_region object which comes in as 
        the second argument."""

    p = yt.ProjectionPlot(ds, 'z', 'O_p5_number_density', data_source=region, center=ds.halo_center_code, width=(200, 'kpc')) 
    proj_frb = p.data_source.to_frb((200., "kpc"), 512)
    proj_frb.save_as_dataset(filename=prefix+'cgm/'+ds_name[-6:]+'_'+region_name+'_frb', fields=['density', 'H_p0_number_density', 'O_p5_number_density'])

def frame(ds, axis, region, region_name, prefix): 

    field='density' 
    p = yt.ProjectionPlot(ds, axis, field, data_source=region, center = ds.halo_center_code, width=(ds.refine_width, 'kpc'))
    p.set_zlim(field, proj_min_dict[field], proj_max_dict[field])
    p.set_cmap(field, colormap_dict[field])
    p.set_buff_size(1080)
    p.set_font({'family':'sans-serif', 'style':'normal', 'weight':'bold', 'size':20})
    p.hide_colorbar()
    p.hide_axes()
    p.annotate_timestamp(redshift=True, draw_inset_box=True) 
    p.set_figure_size(10.8)
    p.set_background_color(field, 'black') 
    p.save(prefix+axis+'/density/'+ds.parameter_filename[-6:]+'_'+region_name)

    field='temperature' 
    p = yt.ProjectionPlot(ds, axis, field, data_source=region, weight_field='density', center = ds.halo_center_code, width=(ds.refine_width, 'kpc')) 
    p.set_zlim(field, 3e3, 1e6) 
    p.set_cmap(field, 'magma') 
    p.set_buff_size(1080) 
    p.set_font({'family':'sans-serif', 'style':'normal', 'weight':'bold', 'size':20}) 
    p.hide_colorbar() 
    p.hide_axes() 
    p.set_figure_size(10.8) 
    p.set_background_color(field, 'black')
    p.save(prefix+axis+'/temperature/'+ds.parameter_filename[-6:]+'_'+region_name) 

def flows(ds, axis, region, region_name, prefix): 

    field = 'H_p0_number_density'
    p = yt.ProjectionPlot(ds, axis, field, center=ds.halo_center_code, width=(ds.refine_width, 'kpc'), data_source=region)
    p.set_zlim(field, 1e10, 1e19)
    p.set_cmap(field, colormap_dict[field])
    p.set_background_color(field, color=background_color_dict[field]) 
    p.annotate_timestamp(redshift=True, draw_inset_box=True)
    p.save(prefix+'flows/'+ds.parameter_filename[-6:]+'_HI_'+region_name)

    field = 'O_p5_number_density'
    p = yt.ProjectionPlot(ds, axis, field, center=ds.halo_center_code, width=(ds.refine_width, 'kpc'), data_source=region)
    p.set_zlim(field, 1e9, 1e15)
    p.set_cmap(field, colormap_dict[field])
    p.set_background_color(field, color=background_color_dict[field]) 
    p.annotate_timestamp(redshift=True, draw_inset_box=True)
    p.save(prefix+'flows/'+ds.parameter_filename[-6:]+'_OVI_'+region_name)

    field = 'O_p6_number_density'
    p = yt.ProjectionPlot(ds, axis, field, center=ds.halo_center_code, width=(ds.refine_width, 'kpc'), data_source=region)
    p.set_zlim(field, 1e9, 1e15)
    p.set_cmap(field, 'magma') 
    p.set_background_color(field, color='black') 
    p.annotate_timestamp(redshift=True, draw_inset_box=True)
    p.save(prefix+'flows/'+ds.parameter_filename[-6:]+'_OVII_'+region_name)

    field = 'Mg_p1_number_density'
    p = yt.ProjectionPlot(ds, axis, field, center=ds.halo_center_code, width=(ds.refine_width, 'kpc'), data_source=region)
    p.set_zlim(field, 1e8, 1e14)
    p.set_cmap(field, colormap_dict[field])
    p.set_background_color(field, color=background_color_dict[field]) 
    p.annotate_timestamp(redshift=True, draw_inset_box=True)
    p.save(prefix+'flows/'+ds.parameter_filename[-6:]+'_MgII_'+region_name)

def shades(ds_name): 
    sm.simple_plot(ds_name,TRACKFILE,('gas','x'),('gas','y'), 'phase', ( (-100,100), (-100,100) ), 'outputs/x_y_phase/'+ds_name[-6:]+'_x_y_phase_cgm_fOVI', \
					region='cgm') 
    sm.simple_plot(ds_name,TRACKFILE,('gas', 'radius_corrected'),('gas','temperature'),  'phase', ((0,250), (1,8)), 'outputs/radius/temperature/'+ds_name[-6:]+'_radius_temperature_phase_cgm_fOVI', \
					region='cgm') 
    sm.simple_plot(ds_name,TRACKFILE,('gas', 'radius_corrected'),('gas','metallicity'),  'phase', ((0,250), (-6,2)), 'outputs/radius/metallicity/'+ds_name[-6:]+'_radius_metallicity_phase_cgm_fOVI', \
					region='cgm') 
    sm.simple_plot(ds_name,TRACKFILE,('gas', 'radius_corrected'),('gas','cooling_time'), 'phase', ((0,250), (4,12)), 'outputs/radius/cooling_time/'+ds_name[-6:]+'_radius_cooling_time_phase_cgm_fOVI', \
					region='cgm') 
    sm.simple_plot(ds_name,TRACKFILE,('gas', 'radius_corrected'),('gas','density'),      'phase', ((0,250), (-32,-22)), 'outputs/radius/density/'+ds_name[-6:]+'_radius_density_phase_cgm_fOVI', \
					region='cgm') 
    sm.simple_plot(ds_name,TRACKFILE,('gas', 'radius_corrected'),('gas','O_p5_column_density'), 'phase', ((0,250), (8,14)), 'outputs/radius/NOVI/'+ds_name[-6:]+'_radius_NOVI_phase_cgm_fOVI', \
					region='cgm') 
    sm.simple_plot(ds_name,TRACKFILE,('gas', 'radius_corrected'),('gas', 'H_p0_column_density'), 'phase', ((0,250), (8,14)), 'outputs/radius/NHI/'+ds_name[-6:]+'_radius_NHI_phase_cgm_fOVI', \
					region='cgm') 

#def lum(ds_name, axis, width, prefix):
    # removed - see original frame.py to bring it back 

# def disk(ds_name, axis, width, prefix, region): 
   # removed - see original frame.py to bring it back 

# def zfilter(ds_name, axis, width, prefix, region): 
   # removed - see original frame.py to bring it back 

parser = argparse.ArgumentParser()
parser.add_argument('--snap_number', type=int, required=True)
parser.add_argument('--sims_dir', type=str, default='./') 
args = parser.parse_args()
if (args.snap_number < 10000): snap_string='DD'+str(args.snap_number)
if (args.snap_number < 1000): snap_string='DD0'+str(args.snap_number)
if (args.snap_number < 100): snap_string='DD00'+str(args.snap_number)
if (args.snap_number < 10): snap_string='DD000'+str(args.snap_number)
print('Hello your snap_number is:', args.snap_number, snap_string)

print('Hello your sims_dir is:', args.sims_dir) 

ds_name = args.sims_dir+'/'+snap_string+'/'+snap_string

print(ds_name)

ds, ds_disk, crd = get_and_prepare_dataset(ds_name)

for region in crd.keys(): 
    print("region")
    #velocities(ds, crd[region], region, 'outputs/')
    #shades(ds_name) 
    #regions_to_frbs(ds, crd[region], region, './outputs/')
    #for axis in ['x']: 
    #    frame(ds, axis, crd[region], region, './outputs/')
    #    flows(ds, axis, crd[region], region, './outputs/')
    ludicrous_views(ds, ds_disk) 

#frb_radius(ds, './outputs/') #<--- this is not cut_region dependent 





