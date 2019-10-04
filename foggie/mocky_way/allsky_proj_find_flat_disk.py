# History:
# 10/04/2019, YZ.
# Change how to import the funcs and merge into foggie/mocky_way 
#
# 08/07/2019, YZ.
# actually, now I set up bulk velocity for the designated sphere before
# calculating angular momentum. In allsky projection case, both ways result
# in similar projeciton (but in offaxis projeciton ,it is important to make sure to
# always set bulk velocity before calculating angular momentum)
#
# 03/27/2019, YZ.
# Check **/mocky_way/figs/allskyproj/nref11c_nref9f_selfshield_z6/***
# image name of [dd_name]_LNNkpc_n32_x800_R100.0.pdf, where NN=10, 20, 30, 50
# ang mom from 10 kpc sphere from halo center,
# with use_gas=True, use_particle=False
# no pre setup of bulk velocity before calculating angmom, see code line 72, 73.

### point to the right data path
import os
import sys
import numpy as np
import healpy as hp
import astropy.units as u
import matplotlib.pyplot as plt

from foggie.foggie.mocky_way.core_funcs import data_dir_sys_dir
from foggie.foggie.mocky_way.core_funcs import find_halo_center_yz
import foggie.foggie.utils.consistency as consistency

import yt
from yt.utilities.math_utils import ortho_find
from yt.visualization.volume_rendering.healpix_projection import healpix_projection

dd_name = 'RD0037'
sim_name = 'nref11n_nref10f'
data_dir, sys_dir = data_dir_sys_dir()
fig_dir = sys_dir+'%s/foggie/foggie/mocky_way/figs/allsky_proj_find_flat_disk'
os.sys.path.insert(0, sys_dir)

ds_file = '%s/%s/%s/%s'%(data_dir, sim_name, dd_name, dd_name)
ds = yt.load(ds_file)
zsnap = ds.get_parameter('CosmologyCurrentRedshift')
halo_center = find_halo_center_yz(ds, zsnap, sim_name, data_dir)

ion_to_proj = 'HI'
rvir = 160 # kpc, roughly the rvir

# get the angular momentum
nside = 2**5 # 2**10 is probably too big....pix number = 12*nside^2
xsize = 800
gc = plt.cm.Greys(0.8) # gc = gridcolor

#for radius in [5, 10, 20, 30, 50]:
# for radius in [6, 7, 8, 9, 11, 12]:
for r_for_L in [8]:
    sp = ds.h.sphere(halo_center, (r_for_L, 'kpc'))
    # let's set up the bulk velocity before setting up the angular momentum
    disk_bulkvel = sp.quantities.bulk_velocity(use_gas=True, use_particles=False)
    sp.set_field_parameter('bulk_velocity', disk_bulkvel)

    # angular momentum
    spec_L = sp.quantities.angular_momentum_vector(use_gas=True, use_particles=False)
    norm_L = spec_L / np.sqrt((spec_L**2).sum())
    n1_L, n2_sun, n3_phi = ortho_find(norm_L)  # UVW vector
    print("radius = %d kpc"%(radius))
    print(n1_L, n2_sun, n3_phi)

    # make the all sky projection
    field_to_proj = consistency.species_dict[ion_to_proj]
    item = ('gas', field_to_proj)  # NHI across the sky
    im = healpix_projection(ds, halo_center, rvir, nside, item,
                            normal_vector=n2_sun, north_vector=n1_L)

    fig = plt.figure(figsize=(8, 4))
    img_cmap = consistency.colormap_dict[field_to_proj]
    img_min = np.log10(consistency.proj_min_dict[field_to_proj])
    img_max = np.log10(consistency.proj_max_dict[field_to_proj])
    hp.mollview(np.log10(im), cbar=None, cmap=img_cmap,
                xsize=xsize, min=img_min, max=img_max,
                title='%s (z=%.2f)'%(dd_name, zsnap))
    hp.graticule(color=gc)

    fig = plt.gcf()
    ax = plt.gca()
    image = ax.get_images()[0]
    img_label = consistency.axes_label_dict[field_to_proj]
    cbar=fig.colorbar(image, ax=ax, pad=0.02, orientation='horizontal',
                      shrink=0.6, label=img_label)

    plt.savefig('%s/%s_%s_%skpc_for_L.pdf'%(fig_dir, sim_name, dd_name, r_for_L))
    plt.close()
    # break
