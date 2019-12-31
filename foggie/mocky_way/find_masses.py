import numpy as np
from foggie.utils import yt_fields
from core_funcs import prepdata

import yt
yt.add_particle_filter("stars",function=yt_fields._stars, filtered_type='all',requires=["particle_type"])
yt.add_particle_filter("dm",function=yt_fields._dm, filtered_type='all',requires=["particle_type"])

import sys
sim_name = sys.argv[1] # 'nref11n_nref10f'
dd_name = sys.argv[2] # 'RD0039'
ds, ds_paras = prepdata(dd_name, sim_name=sim_name)

# create new derived particle fields with “stars_mass” and "dm_mass". from Raymond
ds.add_particle_filter('stars')
ds.add_particle_filter('dm')

# get halo, disk, and cgm
sp = ds.sphere(ds_paras['halo_center'], ds_paras['rvir'])

from core_funcs import dict_disk_rs_zs
disk_rs, disk_zs = dict_disk_rs_zs(dd_name, sim_name=sim_name) # kpc
disk_size_r = 6*disk_rs # 4 is decided by eyeballing the size in find_flat_disk_offaxproj
disk_size_z = 4*disk_zs # one side,
disk = ds.disk(ds_paras['halo_center'],
               ds_paras['L_vec'],
               (disk_size_r, 'kpc'),
               (disk_size_z, 'kpc'))
cgm = sp-disk

# now get the masses for different parts
print("Calculating mass for %s/%s"%(sim_name, dd_name))
for obj, tag in zip([sp, disk, cgm], ['all', 'disk', 'cgm']):
    # now let's get the mass for the designated part
    star_mass = obj['stars_mass'].in_units('Msun')
    dm_mass = obj['dm_mass'].in_units('Msun')
    gas_mass = obj['cell_mass'].in_units('Msun')

    M_star = star_mass.sum()
    M_dm = dm_mass.sum()
    M_gas = gas_mass.sum()

    print(tag)
    print("   M_star = %.2e Msun"%(M_star))
    print("   M_dm   = %.2e Msun"%(M_dm))
    print("   M_gas  = %.2e Msun\n\n"%(M_gas))
