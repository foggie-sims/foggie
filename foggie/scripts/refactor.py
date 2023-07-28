import yt 
import trident 
import numpy as np 
import foggie.utils.foggie_load as foggie_load
import foggie.utils.get_halo_center as ghc 
import foggie.utils.get_region as gr 
from foggie.utils.consistency import proj_min_dict, proj_max_dict, \
    colormap_dict, background_color_dict, o6_min, o6_max, o6_color_map, \
    h1_color_map, h1_proj_max, cgm_outflow_filter, cgm_inflow_filter, \
    cool_outflow_filter, cool_inflow_filter, warm_inflow_filter, \
    warm_outflow_filter
from astropy.table import Table
from astropy.cosmology import WMAP9 as cosmo
import argparse 
from astropy import units as u
import foggie.render.shade_maps as sm
from foggie.utils import prep_dataframe

TRACKFILE = '/Users/tumlinson/Dropbox/FOGGIE/foggie/foggie/halo_tracks/005036/nref11n_selfshield_15/halo_track_200kpc_nref10'

def get_and_prepare_dataset(ds_name):  
    """ This function obtains the dataset and all the usual cut_regions"""
    
    ds, _ = foggie_load.foggie_load(ds_name, TRACKFILE)
    
    trident.add_ion_fields(ds, ions=['H I','C II', 'C III', 'C IV', 'O I', 'O II', 'O III', 'O IV', 'O V', 'O VI', 'O VII', 'O VIII', 'Mg II']) 
    
    # a blank dictionary to contain all the cut regions we'll use downstream 
    cut_region_dict = {}
        
    cut_region_dict['cgm'] = gr.get_region(ds, 'cgm') 
    cut_region_dict['rvir'] = gr.get_region(ds, 'rvir') 
    cut_region_dict['trackbox'] = gr.get_region(ds, 'trackbox') 

    cut_region_dict['cgm_outflows'] = gr.get_region(ds, 'cgm', filter=cgm_outflow_filter)
    cut_region_dict['cool_outflows'] = gr.get_region(ds, 'cgm', filter=cool_outflow_filter)
    cut_region_dict['warm_outflows'] = gr.get_region(ds, 'cgm', filter=warm_outflow_filter)

    cut_region_dict['cgm_inflows'] = gr.get_region(ds, 'cgm', filter=cgm_inflow_filter)
    cut_region_dict['cool_inflows'] = gr.get_region(ds, 'cgm', filter=cool_inflow_filter)
    cut_region_dict['warm_inflows'] = gr.get_region(ds, 'cgm', filter=warm_inflow_filter)
    
    return ds, cut_region_dict 

def velocities(ds_name, axis, width, prefix, region): 

    field_list = ['position_x', 'position_y', ('gas','radius_corrected'), 'temperature', \
 			('gas', 'radial_velocity_corrected'), ('gas', 'tangential_velocity_corrected'), 'O_p5_ion_fraction'] 

    dataset, all_data = sm.prep_dataset(ds_name, TRACKFILE, \
                            ion_list=['H I','C II','C III','C IV','Si II','Si III','Si IV',\
                                        'O I','O II','O III','O IV','O V','O VI','O VII','O VIII'], region=region) 

    data_frame = prep_dataframe.prep_dataframe(dataset, all_data, field_list, 'phase', \
                        halo_center = dataset.halo_center_code, halo_vcenter=dataset.halo_velocity_kms)

    #first we do the 'normal' plots that are NOT O VI filtered 
    filename = prefix+'normal/x_y/'+dataset.parameter_filename[-6:]+'_x_y_'+region+'_phase' 
    image = sm.render_image(data_frame, 'position_x', 'position_y', 'phase', (-200,200),(-200,200), filename) 
    sm.wrap_axes(dataset, image, filename, 'position_x', 'position_y', 'phase', ((-200,200),(-200,200)), region, filter=None)

    filename = prefix+'normal/r_temp/'+dataset.parameter_filename[-6:]+'_radius_temperature_'+region+'_phase' 
    image = sm.render_image(data_frame, 'radius_corrected', 'temperature', 'phase', (0,200),(1, 8), filename)       
    sm.wrap_axes(dataset, image, filename, 'radius_corrected', 'temperature', 'phase', ((0,200),(1,8)), region, filter=None)

    filename = prefix+'normal/rv_tv/'+dataset.parameter_filename[-6:]+'_rv_tv_'+region+'_phase' 
    image = sm.render_image(data_frame, 'radial_velocity_corrected', 'tangential_velocity_corrected', 'phase', (-500,500),(-50,500), filename)       
    sm.wrap_axes(dataset, image, filename, 'radial_velocity_corrected', 'tangential_velocity_corrected', 'phase', ((-500,500),(-50,500)), region, filter=None)

    filename = prefix+'normal/r_tv/'+dataset.parameter_filename[-6:]+'_r_tv_'+region+'_phase' 
    image = sm.render_image(data_frame, 'radius_corrected', 'tangential_velocity_corrected', 'phase', (0,200),(-50,500), filename)       
    sm.wrap_axes(dataset, image, filename, 'radius_corrected', 'tangential_velocity_corrected', 'phase', ((0,200),(-50,500)), region, filter=None)

    filename = prefix+'normal/r_rv/'+dataset.parameter_filename[-6:]+'_r_rv_'+region+'_phase' 
    image = sm.render_image(data_frame, 'radius_corrected', 'radial_velocity_corrected', 'phase', (0,200),(-500,500), filename)       
    sm.wrap_axes(dataset, image, filename, 'radius_corrected', 'radial_velocity_corrected', 'phase', ((0,200),(-500,500)), region, filter=None)

    # now do it again but filtered by O VI this time. 
    mask = (data_frame['O_p5_ion_fraction'] > 0.1) & (data_frame['O_p5_ion_fraction'] < 1.)

    filename = prefix+'fOVI/x_y/'+dataset.parameter_filename[-6:]+'_x_y_'+region+'_phase_fOVI' 
    image = sm.render_image(data_frame[mask], 'position_x', 'position_y', 'phase', (-200,200),(-200,200), filename) 
    sm.wrap_axes(dataset, image, filename, 'position_x', 'position_y', 'phase', ((-200,200),(-200,200)), region, filter=None)

    filename = prefix+'fOVI/r_temp/'+dataset.parameter_filename[-6:]+'_radius_temperature_'+region+'_phase_fOVI' 
    image = sm.render_image(data_frame[mask], 'radius_corrected', 'temperature', 'phase', (0,200),(1, 8), filename)       
    sm.wrap_axes(dataset, image, filename, 'radius_corrected', 'temperature', 'phase', ((0,200),(1,8)), region, filter=None)

    filename = prefix+'fOVI/rv_tv/'+dataset.parameter_filename[-6:]+'_rv_tv_'+region+'_phase_fOVI' 
    image = sm.render_image(data_frame[mask], 'radial_velocity_corrected', 'tangential_velocity_corrected', 'phase', (-500,500),(-50,500), filename)       
    sm.wrap_axes(dataset, image, filename, 'radial_velocity_corrected', 'tangential_velocity_corrected', 'phase', ((-500,500),(-50,500)), region, filter=None)

    filename = prefix+'fOVI/r_tv/'+dataset.parameter_filename[-6:]+'_r_tv_'+region+'_phase_fOVI' 
    image = sm.render_image(data_frame[mask], 'radius_corrected', 'tangential_velocity_corrected', 'phase', (0,200),(-50,500), filename)       
    sm.wrap_axes(dataset, image, filename, 'radius_corrected', 'tangential_velocity_corrected', 'phase', ((0,200),(-50,500)), region, filter=None)

    filename = prefix+'fOVI/r_rv/'+dataset.parameter_filename[-6:]+'_r_rv_'+region+'_phase_fOVI' 
    image = sm.render_image(data_frame[mask], 'radius_corrected', 'radial_velocity_corrected', 'phase', (0,200),(-500,500), filename)       
    sm.wrap_axes(dataset, image, filename, 'radius_corrected', 'radial_velocity_corrected', 'phase', ((0,200),(-500,500)), region, filter=None)

def zfilter(ds_name, axis, width, prefix, region): 

    ds, _ = foggie_load.foggie_load(ds_name, TRACKFILE)
    trident.add_ion_fields(ds, ions=['H I','C II', 'C III', 'C IV', 'O I', 'O II', 'O III', 'O IV', 'O V', 'O VI', 'O VII', 'O VIII', 'Mg II']) 

    #halo_center, velocity = ghc.get_halo_center(ds, refine_box_center) 
    cut_region = gr.get_region(ds, region)
    cut_region_lowz = gr.get_region(ds, 'rvir', filter="obj['metallicity'] < 0.01")

    # all HI cells
    p = yt.ProjectionPlot(ds, 'x', "H_p0_number_density", data_source=cut_region, center=ds.halo_center_code, width=(width, 'Mpc'))
    p.set_cmap(field='H_p0_number_density', cmap = h1_color_map)
    p.annotate_timestamp(redshift=True)
    p.set_zlim('H_p0_number_density', 1e8, 1e22)
    p.save(prefix+axis+'/zfilter/'+ds.parameter_filename[-6:]+'_h1_all')

    # HI 'clouds' only
    p = yt.ProjectionPlot(ds, 'x', "H_p0_number_density", data_source=cut_region_lowz, center=ds.halo_center_code, width=(width, 'Mpc'))
    p.set_cmap(field='H_p0_number_density', cmap = h1_color_map)
    p.annotate_timestamp(redshift=True)
    p.set_zlim('H_p0_number_density', 1e8, h1_proj_max)
    p.save(prefix+axis+'/zfilter/'+ds.parameter_filename[-6:]+'_h1_lowz2')

    # all OVI cells
    p = yt.ProjectionPlot(ds, 'x', "O_p5_number_density", data_source=cut_region, center=ds.halo_center_code, width=(width, 'Mpc'))
    p.set_cmap(field='O_p5_number_density', cmap = o6_color_map)
    p.annotate_timestamp(redshift=True)
    p.set_zlim('O_p5_number_density', o6_min, o6_max)
    p.save(prefix+axis+'/zfilter/'+ds.parameter_filename[-6:]+'_o6_all')

    # OVI 'clouds' only
    p = yt.ProjectionPlot(ds, 'x', "O_p5_number_density", data_source=cut_region_lowz, center=ds.halo_center_code, width=(width, 'Mpc'))
    p.set_cmap(field='O_p5_number_density', cmap = o6_color_map)
    p.annotate_timestamp(redshift=True)
    p.set_zlim('O_p5_number_density', o6_min, o6_max)
    p.save(prefix+axis+'/zfilter/'+ds.parameter_filename[-6:]+'_o6_lowz2')
                       
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

def lum(ds_name, axis, width, prefix):
    
    def star_particle_luminosity(field, data):
        s99 = Table.read('s99', format='ascii')
        star_abmag = np.interp(data[('stars', 'age')].in_units('yr'), s99['col1'], s99['col2'])
        star_lum = 10.**(5.-star_abmag) / 1e6 * data[('stars', 'particle_mass')]
        return yt.YTArray(star_lum, 'dimensionless') 
    
    ds, _ = foggie_load.foggie_load(ds_name, TRACKFILE)
    ds.add_field(('stars', 'star_lum'), function=star_particle_luminosity, units='dimensionless', \
                 take_log=True, force_override=True, particle_type=True)
    
    axis1 = {'x':'particle_position_y', 'y':'particle_position_x', 'z':'particle_position_x'} 
    axis2 = {'x':'particle_position_z', 'y':'particle_position_z', 'z':'particle_position_y'} 
    if (width > 0.): 
        print("Inside lum plot width will be: ", width, " in kpc") 
        p = yt.ParticlePlot(ds, axis1[axis], axis2[axis], ('stars','star_lum'), center=ds.halo_center_code, width=(width, 'kpc'))
    else:
        #if we are in this branch of the if stmnt the width given is arcmin which we convert to kpc 
        plotwidth = (-1.*width) * (cosmo.kpc_proper_per_arcmin(ds.current_redshift)).value 
        print("Inside lum plot width will be: ", plotwidth, " from angular size of ", width) 
        p = yt.ParticlePlot(ds, axis1[axis], axis2[axis], ('stars','star_lum'), center=ds.halo_center_code, width=(plotwidth, 'kpc'))
    p.annotate_timestamp(redshift=True, draw_inset_box=True, text_args={'color':'black'}) 
    p.set_cmap(('stars','star_lum'), 'gray')
    p.set_buff_size(1080)
    p.set_font({'family':'sans-serif', 'style':'normal', 'weight':'bold', 'size':20})
    p.hide_colorbar()
    p.hide_axes()
    p.set_figure_size(10.8)
    p.save(prefix+axis+'/lum/'+ds.parameter_filename[-6:]+'_lum')

# def disk(ds_name, axis, width, prefix, region): 
   # removed - see original frame.py to bring it back 


parser = argparse.ArgumentParser()
parser.add_argument('--snap_number', type=int, required=True)
args = parser.parse_args()
if (args.snap_number < 1000): snap_string='RD0'+str(args.snap_number)
if (args.snap_number < 100): snap_string='RD00'+str(args.snap_number)
if (args.snap_number < 10): snap_string='RD000'+str(args.snap_number)
print('Hello your snap_number is:', args.snap_number, snap_string)

ds_name = snap_string+'/'+snap_string
TRACKFILE = '/Users/tumlinson/Dropbox/FOGGIE/foggie/foggie/halo_tracks/005036/nref11n_selfshield_15/halo_track_200kpc_nref10'

print(ds_name)

ds, crd = get_and_prepare_dataset(ds_name)

for region in crd.keys(): 
    print("region")
    shades(ds_name) 
    for axis in ['x']: 
        frame(ds, axis, crd[region], region, './outputs/')
        flows(ds, axis, crd[region], region, './outputs/')