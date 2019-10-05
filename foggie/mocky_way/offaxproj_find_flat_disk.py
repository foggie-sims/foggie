# from astropy.table import Table
import os
import sys
import numpy as np
import yt
from foggie.utils import consistency # for plotting
from foggie.mocky_way.core_funcs import find_halo_center_yz
from foggie.mocky_way.core_funcs import dict_rvir_proper
from foggie.mocky_way.core_funcs import data_dir_sys_dir
from foggie.mocky_way.core_funcs import get_sphere_ang_mom_vecs

data_dir, sys_dir = data_dir_sys_dir()

sim_name = sys.argv[1]    # nref11n_nref10f
dd_name = sys.argv[2]     # RD0039
ion_to_proj = sys.argv[3] # mainly for HI

ds_file = '%s/%s/%s/%s'%(data_dir, sim_name, dd_name, dd_name)
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
fig_dir = '%s/foggie/mocky_way/figs/offaxproj/%s_%s'%(sys_dir, sim_name, dd_name)
if os.path.isdir(fig_dir) == False:
    print('Not finding %s, mkdir'%(fig_dir))
    os.mkdir(fig_dir)
cmap = consistency.h1_color_map
zmin = consistency.h1_proj_min_mw
zmax = consistency.h1_proj_max_mw

# ok, now make some face on and edge on projection
width = ds.quan(2*rvir, 'kpc')
data_source = sp_to_proj

for r_for_L in [5, 10, 15, 20]:
    dict_vecs = get_sphere_ang_mom_vecs(ds, halo_center, r_for_L,
                                        random_seed=99)
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
        #pj.annotate_text([0.05, 0.9],
        #                       'Edge-on, %s, z=%.2f'%(dd_name, ds_paras['zsnap']),
        #                       coord_system='axis', text_args={'color':'black'})
        fig_name = '%s_%s_AngMon%skpc_%s.pdf'%(sim_name, dd_name, r_for_L, tag)
        pj.save('%s/%s'%(fig_dir, fig_name))
        print('Saving... %s/%s'%(fig_dir, fig_name))

    # break
        # pj.close()
