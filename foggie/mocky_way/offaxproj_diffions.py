# Off axis projection for edge on and face on view of the galaxy, and
# find the right angular_momentum_vector for the flat disk.
# Example:
# python find_flat_disk_offaxproj.py nref11n_nref10f RD0039 HI
# Yong Zheng. 10/09/2019.
import os
import sys
import numpy as np
import yt
from foggie.utils import consistency # for plotting
from core_funcs import find_halo_center_yz
from core_funcs import dict_rvir_proper
from core_funcs import data_dir_sys_dir
from core_funcs import get_sphere_ang_mom_vecs

data_dir, sys_dir = data_dir_sys_dir()

sim_name = 'nref11n_nref10f'
dd_name = 'DD2175'
ion_to_proj = 'CIV' # 'SiII', 'SiIII', 'SiIV', 'CII', 'CIV', 'OVI', 'NV',
                   # 'OVII', 'OVIII', 'NeVII', 'NeVIII'

from foggie.mocky_way.core_funcs import prepdata
ds, ds_paras = prepdata(dd_name)
halo_center = ds_paras['halo_center']
rvir = ds_paras['rvir']
proj_length = 50 # kpc
width = 2*ds.quan(proj_length, 'kpc')

# let's select a region to project based on phase diagram
sp_to_proj = ds.sphere(halo_center, (proj_length, 'kpc'))
# sp_to_proj.set_field_parameter("observer_location", halo_center)

# ok, now make some face on and edge on projection
data_source = sp_to_proj
field_to_proj = consistency.species_dict[ion_to_proj]
cmap = consistency.colormap_dict[field_to_proj]
zmin = consistency.proj_min_dict[field_to_proj]
zmax = consistency.proj_max_dict[field_to_proj]
fig_dir = sys_dir+'/foggie/mocky_way/figs/offaxproj/column_density/'

highlight_disk = False
L_vec = ds_paras['L_vec']
phi_vec = ds_paras['phi_vec']
sun_vec = ds_paras['sun_vec']

for vec_to_proj, north_vector, tag in zip([phi_vec, sun_vec, L_vec],
                                          [L_vec, L_vec, sun_vec],
                                          ['edgeon1', 'edgeon2', 'faceon']):

    pj = yt.OffAxisProjectionPlot(ds, vec_to_proj, field_to_proj,
                                  center=halo_center, width=width,
                                  data_source=data_source,
                                  north_vector=north_vector)
    pj.set_cmap(field_to_proj, cmap=cmap)
    pj.set_zlim(field_to_proj, zmin=zmin, zmax=zmax)
    fig_name = '%s_%s_halo_center_r%d_N%s_%s.pdf'%(sim_name, dd_name,
                                                proj_length, ion_to_proj, tag)
    pj.save('%s/%s'%(fig_dir, fig_name))
    print('Saving... %s/%s'%(fig_dir, fig_name))

    if highlight_disk == True:
        print("High light the disk box boundaries.")
        # add this to show where the disk is after deciding rs and zs
        disk_rs = 3.4 # kpc for DD2175
        disk_zs = 0.5 # kpc
        disk_size_r = disk_rs*6
        disk_size_z = disk_zs*4

        image_width = ds.quan(80, 'kpc')
        rr = disk_size_r/image_width.value
        hh = disk_size_z/image_width.value

        if tag in ['edgeon1', 'edgeon2']:
            pj.annotate_line((0.5-rr, 0.5+hh), (0.5+rr, 0.5+hh),
                             coord_system='axis')
            pj.annotate_line((0.5-rr, 0.5-hh), (0.5+rr, 0.5-hh),
                             coord_system='axis')
            pj.annotate_line((0.5-rr, 0.5-hh), (0.5-rr, 0.5+hh),
                             coord_system='axis')
            pj.annotate_line((0.5+rr, 0.5-hh), (0.5+rr, 0.5+hh),
                             coord_system='axis')
        else: # for faceon scenario
            pj.annotate_sphere(halo_center,
                               radius=(disk_size_r, 'kpc'),
                               circle_args={'color':'w'})

        pj.set_width(image_width)
        fig_name = '%s_%s_halo_center_r%d_N%s_%s_zoom.pdf'%(sim_name, dd_name,
                                                proj_length, tag, ion_to_proj)
        pj.save('%s/%s'%(fig_dir, fig_name))
        print('Saving... %s/%s'%(fig_dir, fig_name))

# break
    # pj.close()
