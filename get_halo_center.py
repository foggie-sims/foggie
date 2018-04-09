from __future__ import print_function

import yt
import numpy as np

def get_halo_center(ds, center_guess, **kwargs):
    # returns a list of the halo center coordinates
    radius = kwargs.get("radius", 50.)  # search radius in kpc
    vel_radius = kwargs.get('vel_radius', 2.)

    # now determine the location of the highest DM density, which should be the center of the main halo
    ad = ds.sphere(center_guess, (radius, 'kpc')) # extract a sphere centered at the middle of the box
    x,y,z = np.array(ad["x"]), np.array(ad["y"]), np.array(ad["z"])
    dm_density =  ad['Dark_Matter_Density']
    imax = (np.where(dm_density > 0.9999 * np.max(dm_density)))[0]
    halo_center = [x[imax[0]], y[imax[0]], z[imax[0]]]
    print('We have located the main halo at :', halo_center)

    sph = ds.sphere(halo_center, (vel_radius,'kpc'))
    velocity = [np.mean(sph['x-velocity']), np.mean(sph['y-velocity']), np.mean(sph['z-velocity'])]

    return halo_center, velocity
