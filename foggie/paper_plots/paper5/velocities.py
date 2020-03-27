import foggie.render.shade_maps as sm
from foggie.utils import prep_dataframe   


fname = 'RD0040/RD0040'  

trackfile = '/Users/tumlinson/Dropbox/FOGGIE/foggie/foggie/halo_tracks/008508/nref11n_selfshield_15/halo_track_200kpc_nref10'


field_list = ['position_x', 'position_y', 'radius_corrected', 'temperature', 'density'] 


dataset, all_data, halo_center  = sm.prep_dataset(fname, trackfile, \
                        ion_list=['H I','C II','C III','C IV','Si II','Si III','Si IV',\
                                    'O I','O II','O III','O IV','O V','O VI','O VII','O VIII'], region='cgm') 

data_frame = prep_dataframe.prep_dataframe(dataset, all_data, field_list, 'phase', \
                        halo_center = dataset.halo_center_code, halo_vcenter=dataset.halo_velocity_kms)


image = sm.render_image(data_frame, 'position_x', 'position_y', 'phase', (-200,200),(-200,200), 'a', pixspread=0) 
sm.wrap_axes(dataset, image, 'a', 'position_x', 'position_y', 'phase', ((-200,200),(-200,200)), 'cgm', filter=None) 

image = sm.render_image(data_frame, 'radius_corrected', 'temperature', 'phase', (0,200),(1, 8), 'b', pixspread=0) 
sm.wrap_axes(dataset, image, 'b', 'radius_corrected', 'temperature', 'phase', ((0,200),(1,8)), 'cgm', filter=None) 
