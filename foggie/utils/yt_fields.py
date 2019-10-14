
# Note: any field definition functions that start with _ cannot be loaded
# into other modules because python makes them private functions.

import numpy as np

def _static_average_rampressure(field, data):
    bulk_velocity = data.get_field_parameter("bulk_velocity").in_units('km/s')
    velx = data['enzo', 'x-velocity'].to('km/s') - bulk_velocity[0]
    vely = data['enzo', 'y-velocity'].to('km/s') - bulk_velocity[1]
    velz = data['enzo', 'z-velocity'].to('km/s') - bulk_velocity[2]
    vel = np.sqrt(velx**2. + vely**2. + velz**2.)/np.sqrt(3)
    rp = data['density'] * vel**2.
    return np.log10(rp.to('dyne/cm**2').value)

def _static_radial_rampressure(field, data):
    vel = data['gas', 'radial_velocity']
    vel[vel<0] = 0.
    rp = data['density'] * vel**2.
    return np.log10(rp.to('dyne/cm**2').value)


def _radial_rampressure(field, data):

    vel = data['gas', 'circular_velocity'] + data['gas', 'radial_velocity']
    vel[vel<0] = 0.
    rp = data['density'] * vel**2.
    return np.log10(rp.to('dyne/cm**2').value)


### Filter Particles ###
def _stars(pfilter, data):
    """Filter star particles
    To use: yt.add_particle_filter("stars",function=_stars, filtered_type='all',requires=["particle_type"])"""

    return data[(pfilter.filtered_type, "particle_type")] == 2


def _dm(pfilter, data):
    """Filter dark matter particles
    To use: yt.add_particle_filter("darkmatter",function=_darkmatter, filtered_type='all',requires=["particle_type"])"""
    return data[(pfilter.filtered_type, "particle_type")] == 4



def _cooling_criteria(field,data):
    """adds cooling criteria field
    to use: yt.add_field(("gas","cooling_criteria"),function=_cooling_criteria,units=None)"""
    return -1*data['cooling_time'] / ((data['dx']/data['sound_speed']).in_units('s'))

def vx_corrected(field, data):
    """Corrects the x-velocity for bulk motion of the halo. Requires 'halo_velocity_kms', which
    is the halo velocity with yt units of km/s, to be defined."""
    halo_velocity_kms = data.ds.halo_velocity_kms
    return data['gas','velocity_x'].in_units('km/s') - halo_velocity_kms[0]

def vy_corrected(field, data):
    """Corrects the y-velocity for bulk motion of the halo. Requires 'halo_velocity_kms', which
    is the halo velocity with yt units of km/s, to be defined."""
    halo_velocity_kms = data.ds.halo_velocity_kms
    return data['gas','velocity_y'].in_units('km/s') - halo_velocity_kms[1]

def vz_corrected(field, data):
    """Corrects the z-velocity for bulk motion of the halo. Requires 'halo_velocity_kms', which
    is the halo velocity with yt units of km/s, to be defined."""
    halo_velocity_kms = data.ds.halo_velocity_kms
    return data['gas','velocity_z'].in_units('km/s') - halo_velocity_kms[2]

def radial_velocity_corrected(field, data):
    """Corrects the radial velocity for bulk motion of the halo and the halo center.
    Requires 'halo_center_kpc', which is the halo center with yt units of kpc, to be defined.
    Requires the other fields of _vx_corrected, _vy_corrected, and _vz_corrected."""
    halo_center_kpc = data.ds.halo_center_kpc
    x_hat = data['x'].in_units('kpc') - halo_center_kpc[0]
    y_hat = data['y'].in_units('kpc') - halo_center_kpc[1]
    z_hat = data['z'].in_units('kpc') - halo_center_kpc[2]
    r = np.sqrt(x_hat*x_hat + y_hat*y_hat + z_hat*z_hat)
    x_hat /= r
    y_hat /= r
    z_hat /= r
    return data['vx_corrected']*x_hat + data['vy_corrected']*y_hat + data['vz_corrected']*z_hat

def radius_corrected(field, data):
    """Corrects the radius for the center of the halo. Requires 'halo_center_kpc', which is the halo
    center with yt units of kpc, to be defined."""
    halo_center_kpc = data.ds.halo_center_kpc
    x_hat = data['x'].in_units('kpc') - halo_center_kpc[0]
    y_hat = data['y'].in_units('kpc') - halo_center_kpc[1]
    z_hat = data['z'].in_units('kpc') - halo_center_kpc[2]
    r = np.sqrt(x_hat*x_hat + y_hat*y_hat + z_hat*z_hat)
    return r

def theta_pos(field, data):
    """Calculates the azimuthal position of cells for conversions to spherical coordinates.
    Requires 'halo_center_kpc', which is the halo center with yt units of kpc, to be defined."""
    halo_center_kpc = data.ds.halo_center_kpc
    x_hat = data['x'].in_units('kpc') - halo_center_kpc[0]
    y_hat = data['y'].in_units('kpc') - halo_center_kpc[1]
    z_hat = data['z'].in_units('kpc') - halo_center_kpc[2]
    r = np.sqrt(x_hat*x_hat + y_hat*y_hat + z_hat*z_hat)
    return np.arccos(z_hat/r)

def phi_pos(field, data):
    """Calculates the angular position of cells for conversions to spherical coordinates.
    Requires 'halo_center_kpc', which is the halo center with yt units of kpc, to be defined."""
    halo_center_kpc = data.ds.halo_center_kpc
    x_hat = data['x'].in_units('kpc') - halo_center_kpc[0]
    y_hat = data['y'].in_units('kpc') - halo_center_kpc[1]
    z_hat = data['z'].in_units('kpc') - halo_center_kpc[2]
    r = np.sqrt(x_hat*x_hat + y_hat*y_hat + z_hat*z_hat)
    return np.arctan2(y_hat, x_hat)

def kinetic_energy_corrected(field, data):
    """Calculates the kinetic energy of cells relative to the center of the halo and corrected
    for the halo velocity. Requires 'halo_velociy_kms', which is the halo velocity with yt
    units of km/s, to be defined."""
    return 0.5 * data['cell_mass'] * data['radial_velocity_corrected']**2.

def _no6(field,data):  
    return data["dx"] * data['O_p5_number_density']  
    
def _nh1(field,data):  
    return data["dx"] * data['H_p0_number_density']  

def _no5(field,data):  
    return data["dx"] * data['O_p4_number_density']  

def _c4(field,data):  
    return data["dx"] * data['C_p3_number_density']  