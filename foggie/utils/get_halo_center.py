"""
Obtains center position for a halo, and the x,y,z velocity components.
"""
from __future__ import print_function
import numpy as np

def get_halo_center(ds, center_guess, **kwargs):
    """
    Inputs are a dataset, and the center_guess.
    Outputs center and velocity tuples composed of x,y,z coordinates.
    """

    radius = kwargs.get('radius', 50.)  # search radius in kpc
    units = kwargs.get('units', 'code')

    length = 'code_length'
    vel = 'code_velocity'

    print('get_halo_center:', length, vel)
    sphere_region = ds.sphere(center_guess, (radius, 'kpc'))
    print("get_halo_center: obtained the spherical region")

    x_pos, y_pos, z_pos = np.array(sphere_region["x"].in_units(length)), \
                          np.array(sphere_region["y"].in_units(length)), \
                          np.array(sphere_region["z"].in_units(length))

    dm_density = sphere_region['Dark_Matter_Density']
    print("get_halo_center: extracted the DM density")

    # now determine the location of the highest DM density, which should be the
    # center of the main halo
    imax = (np.where(dm_density > 0.9999 * np.max(dm_density)))[0]
    halo_center = [x_pos[imax[0]], y_pos[imax[0]], z_pos[imax[0]]]
    print("get_halo_center: we have obtained the preliminary center")

    sph = ds.sphere(halo_center, (5., 'kpc'))
    velocity = [np.mean(sph['x-velocity']),
                np.mean(sph['y-velocity']),
                np.mean(sph['z-velocity'])]
    print("got the velocities")

    if (units == 'physical'): # do it over again but in the physical units
        x_pos, y_pos, z_pos = np.array(sphere_region["x"].in_units('kpc')), \
                              np.array(sphere_region["y"].in_units('kpc')), \
                              np.array(sphere_region["z"].in_units('kpc'))
        halo_center = [x_pos[imax[0]], y_pos[imax[0]], z_pos[imax[0]]]
        velocity = [np.mean(sph['x-velocity'].in_units('km/s')),
                        np.mean(sph['y-velocity'].in_units('km/s')),
                        np.mean(sph['z-velocity'].in_units('km/s'))]

    print('get_halo_center: located the main halo at:', halo_center, velocity)

    return halo_center, velocity
