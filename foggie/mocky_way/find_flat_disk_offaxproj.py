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

#sim_name = sys.argv[1]    # nref11n_nref10f
#dd_name = sys.argv[2]     # RD0039
#ion_to_proj = sys.argv[3] # mainly for HI
sim_name = 'nref11n_nref10f'
dd_name = 'RD0039' # 'DD2175'
ion_to_proj = 'HI'

ds_file = '%s/%s/%s/%s'%(data_dir, sim_name, dd_name, dd_name)
if os.path.isfile(ds_file) == False:
    drive_dir = '/Volumes/Yong4TB/foggie/halo_008508'
    ds_file = '%s/%s/%s/%s'%(drive_dir, sim_name, dd_name, dd_name)

ds = yt.load(ds_file)
zsnap = ds.get_parameter('CosmologyCurrentRedshift')

## find halo center
halo_center = find_halo_center_yz(ds, zsnap, sim_name, data_dir)
rvir = dict_rvir_proper(dd_name, sim_name=sim_name)
field_to_proj = consistency.species_dict[ion_to_proj]

# let's select a region to project based on phase diagram
sp_to_proj = ds.sphere(halo_center, (rvir, 'kpc'))
sp_to_proj.set_field_parameter("observer_location", halo_center)

# set up the directory to save images and other plotting stuff
fig_dir = sys_dir+'/foggie/mocky_way/figs/find_flat_disk'
if os.path.isdir(fig_dir) == False:
    print('Not finding %s, mkdir'%(fig_dir))
    os.mkdir(fig_dir)
cmap = consistency.h1_color_map
zmin = consistency.h1_proj_min_mw
zmax = consistency.h1_proj_max_mw

# ok, now make some face on and edge on projection
width = ds.quan(2*rvir, 'kpc')
data_source = sp_to_proj

from core_funcs import default_random_seed
random_seed = default_random_seed()
# for r_for_L in [5, 10, 15, 20]:
highlight_disk = True
for r_for_L in [5, 10, 15]:
    dict_vecs = get_sphere_ang_mom_vecs(ds, halo_center, r_for_L,
                                        random_seed=random_seed)
    L_vec = dict_vecs['L_vec']
    phi_vec = dict_vecs['phi_vec']
    sun_vec = dict_vecs['sun_vec']

    for vec_to_proj, north_vector, tag in zip([phi_vec, sun_vec, L_vec],
                                              [L_vec, L_vec, sun_vec],
                                              ['edgeon1', 'edgeon2', 'faceon']):

        pj = yt.OffAxisProjectionPlot(ds, vec_to_proj, field_to_proj,
                                      center=halo_center, width=width,
                                      data_source=data_source,
                                      north_vector=north_vector)
        pj.set_cmap(field_to_proj, cmap=cmap)
        pj.set_zlim(field_to_proj, zmin=zmin, zmax=zmax)
        fig_name = '%s_%s_AngMon%skpc_%s.pdf'%(sim_name, dd_name, r_for_L, tag)
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
            fig_name = '%s_%s_AngMon%skpc_%s_zoom_rszsnofixed.pdf'%(sim_name, dd_name, r_for_L, tag)
            pj.save('%s/%s'%(fig_dir, fig_name))
            print('Saving... %s/%s'%(fig_dir, fig_name))

    # break
        # pj.close()
