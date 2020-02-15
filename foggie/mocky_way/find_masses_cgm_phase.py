import numpy as np
from foggie.utils import yt_fields
from core_funcs import prepdata

import yt
yt.add_particle_filter("stars",function=yt_fields._stars, filtered_type='all',requires=["particle_type"])
yt.add_particle_filter("dm",function=yt_fields._dm, filtered_type='all',requires=["particle_type"])

dd_name = 'DD2175'
sim_name = 'nref11n_nref10f'
ds, ds_paras = prepdata(dd_name, sim_name=sim_name)

# create new derived particle fields with “stars_mass” and "dm_mass". from Raymond
ds.add_particle_filter('stars')
ds.add_particle_filter('dm')

# get halo, disk, and cgm
# sp = ds.sphere(ds_paras['halo_center'], ds_paras['rvir'])
sp = ds.sphere(ds_paras['halo_center'], (15, 'kpc'))
disk_rs = 3.3 # kpc
disk_zs = 0.5 # kpc, chec disk_scale_length_rs and scale_height_zs
disk_size_r = 6*disk_rs # 4 is decided by eyeballing the size in find_flat_disk_offaxproj
disk_size_z = 4*disk_zs # one side,
disk = ds.disk(ds_paras['halo_center'],
               ds_paras['L_vec'],
               (disk_size_r, 'kpc'),
               (disk_size_z, 'kpc'))
cgm = sp-disk

logT = np.log10(cgm['temperature'])

### this part can be changed depending what you want
cell_mass = cgm['cell_mass']
#from yt import units as u
#nHI = cgm['H_p0_number_density']
#vol = cgm['cell_volume']
#cell_mass = nHI*vol*u.mp

cold_ind = logT<=4
cold_mass = cell_mass[cold_ind].sum().in_units('Msun')
print("cold cgm: %.2e Msun"%(cold_mass))

cool_ind = np.all([logT>4, logT<=5], axis=0)
cool_mass = cell_mass[cool_ind].sum().in_units('Msun')
print("cool cgm: %.2e Msun"%(cool_mass))


warm_ind = np.all([logT>5, logT<=6], axis=0)
warm_mass = cell_mass[warm_ind].sum().in_units('Msun')
print("warm cgm: %.2e Msun"%(warm_mass))


hot_ind = logT>6
hot_mass = cell_mass[hot_ind].sum().in_units("Msun")
print("hot cgm: %.2e Msun"%(hot_mass))

'''
#### PLOT ####
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
mpl.rcParams['font.family'] = 'stixgeneral'

x = ['Cold', 'Cool', 'Warm', 'Hot']
tot_mass = cold_mass + cool_mass + warm_mass + hot_mass
fy = np.array([cold_mass, cool_mass, warm_mass, hot_mass])/tot_mass
y = np.array([cold_mass, cool_mass, warm_mass, hot_mass])/1e9

from foggie.utils import consistency
cmap = consistency.temperature_discrete_cmap
c_cold = cmap(0.05)
c_cool = cmap(0.25)
c_warm = cmap(0.6)
c_hot = cmap(0.9)

fig = plt.figure()
ax = fig.add_subplot(111)
barlist = ax.bar(x, y, color='c')
barlist[0].set_color(c_cold)
barlist[1].set_color(c_cool)
barlist[2].set_color(c_warm)
barlist[3].set_color(c_hot)

ax.tick_params(labelsize=16)
ax.set_xlabel('CGM Phases', fontsize=16)
ax.set_ylabel('Mass (1e9 Msun)', fontsize=16)
fig.savefig('/Users/Yong/Desktop/fig_cgm_mass.pdf')
'''
