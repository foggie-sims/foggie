import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'stixgeneral'
import numpy as np
import sys
import seaborn as sns
import collections
import astropy.units as u
from matplotlib.colors import to_hex
from offaxproj_dshader_modules import *
# from mocky_way_modules import data_dir_sys_dir, prepdata
from foggie.utils import consistency

### first, read in dataset
sim_name = 'nref11n_nref10f' # 'nref11c_nref9f_selfshield_z6'
dd_name = 'DD2175' # 'RD0037' 
obs_point =  'halo_center' ## halo_center, observer_location
dshader_field = 'vel_pos' ## radius, velocity, vel_pos, vel_neg

from foggie.mocky_way.core_funcs import prepdata
ds, ds_paras = prepdata(dd_name, sim_name=sim_name)

## decide where the observer is. halo_center == gc, observer_location == off_center
if obs_point == 'halo_center':
    obs_bulkvel = 'disk_bulkvel'
else:
    obs_bulkvel = 'offcenter_bulkvel'
print("I am doing the calculation from the %s......"%(obs_point))

## 2nd, get the source object, and set the field parameters
# obj_dict = obj_source_halo_disk(ds, ds_paras)
# obj_source = obj_dict[obj_tag]
obj_source = ds.sphere(ds_paras[obs_point], (15, 'kpc'))
obj_source.set_field_parameter("observer_location", ds_paras[obs_point])
obj_source.set_field_parameter("observer_bulkvel", ds_paras[obs_bulkvel])
# obj_source.set_field_parameter("L_vec", ds_paras["L_vec"])

### 3rd, now decide which velocity field to run
if dshader_field == 'vel_pos':
    from offaxproj_dshader_modules import prep_dataframe_vel_pos
    dataframe = prep_dataframe_vel_pos(obj_source,
                                       ds_paras,
                                       obs_point=obs_point,
                                       fields=['H_nuclei_density',
                                               'los_velocity_mw'])
elif dshader_field == 'vel_neg':
    from offaxproj_dshader_modules import prep_dataframe_vel_neg
    dataframe = prep_dataframe_vel_neg(obj_source, ds_paras,
                                       obs_point=obs_point,
                                       fields=['H_nuclei_density',
                                               'los_velocity_mw'])
else:
    from offaxproj_dshader_modules import prep_dataframe
    dataframe = prep_dataframe(obj_source,
                               ds_paras,
                               obs_point=obs_point,
                               fields=['H_nuclei_density',
                                       'los_velocity_mw'])

from offaxproj_dshader_modules import get_df_dicts
cat_field, c_label, discrete_cmap, categories = get_df_dicts(dshader_field,
                                                             dd_name,
                                                             obs_point)
pngfile = '%s_%s_offaxproj_%s_%s'%(sim_name, dd_name, obs_point, dshader_field)

#x_range = [-150, 150]
x_range = [-20, 20]
x_field = 'image_x'
x_label = r'x (kpc)' # axes_label_dict[x_field] # r'log n$_H$ (cm$^{-3}$)'

#y_range = [-150, 150]
y_range = [-20, 20]
y_field = 'image_y'
y_label = r'y (kpc)'

export_path = 'figs/offaxproj_dshader/'

# now just make the data shader
print("Making data shader frame...")
img = offaxproj_noaxes(dataframe, x_field=x_field, x_range=x_range,
                           y_field=y_field, y_range=y_range,
                           cat_field=cat_field, export_path=export_path,
                           save_to_file=pngfile)
### now put the axes on with ticks and labels
img_x_width = 1000
img_y_width = 1000
c_ticklabels = [ss.decode('UTF-8').upper() for ss in categories]
filename = '%s/%s.png'%(export_path, pngfile)
print("Putting axes, ticks and labels on the datashader frame...")
wa = wrap_axes(filename, discrete_cmap, x_range=x_range, y_range=y_range,
               x_label=x_label, y_label=y_label,
               img_x_width=img_x_width, img_y_width=img_y_width,
               c_label=c_label, c_ticklabels=c_ticklabels)
