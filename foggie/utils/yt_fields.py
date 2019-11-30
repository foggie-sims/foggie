
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

def _young_stars(pfilter, data):
    """Filter star particles with creation time < 10 Myr ago
    To use: yt.add_particle_filter("young_stars", function=_young_stars, filtered_type='all', requires=["creation_time"])"""

    age = data.ds.current_time - data[pfilter.filtered_type, "creation_time"]
    filter = np.logical_and(age.in_units('Myr') <= 10, age >= 0)
    return filter

def _old_stars(pfilter, data):
    """Filter star particles with creation time > 10 Myr ago
    To use: yt.add_particle_filter("young_stars", function=_old_stars, filtered_type='all', requires=["creation_time"])"""
    age = data.ds.current_time - data[pfilter.filtered_type, "creation_time"]
    filter = np.logical_or(age.in_units('Myr') >= 10, age < 0)
    return filter

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
    is the halo velocity with yt units of km/s, to be defined. -Cassi"""
    halo_velocity_kms = data.ds.halo_velocity_kms
    return data['gas','velocity_x'].in_units('km/s') - halo_velocity_kms[0]

def vy_corrected(field, data):
    """Corrects the y-velocity for bulk motion of the halo. Requires 'halo_velocity_kms', which
    is the halo velocity with yt units of km/s, to be defined. -Cassi"""
    halo_velocity_kms = data.ds.halo_velocity_kms
    return data['gas','velocity_y'].in_units('km/s') - halo_velocity_kms[1]

def vz_corrected(field, data):
    """Corrects the z-velocity for bulk motion of the halo. Requires 'halo_velocity_kms', which
    is the halo velocity with yt units of km/s, to be defined. -Cassi"""
    halo_velocity_kms = data.ds.halo_velocity_kms
    return data['gas','velocity_z'].in_units('km/s') - halo_velocity_kms[2]

def radial_velocity_corrected(field, data):
    """Corrects the radial velocity for bulk motion of the halo and the halo center.
    Requires 'halo_center_kpc', which is the halo center with yt units of kpc, to be defined.
    Requires the other fields of _vx_corrected, _vy_corrected, and _vz_corrected. -Cassi"""
    halo_center_kpc = data.ds.halo_center_kpc
    x_hat = data['x'].in_units('kpc') - halo_center_kpc[0]
    y_hat = data['y'].in_units('kpc') - halo_center_kpc[1]
    z_hat = data['z'].in_units('kpc') - halo_center_kpc[2]
    r = np.sqrt(x_hat*x_hat + y_hat*y_hat + z_hat*z_hat)
    x_hat /= r
    y_hat /= r
    z_hat /= r
    return data['vx_corrected']*x_hat + data['vy_corrected']*y_hat + data['vz_corrected']*z_hat

def theta_velocity_corrected(field, data):
    """Corrects the theta direction of the spherical coordinate velocity for the bulk motion of the
    halo and the halo center. Requires 'halo_center_kpc', which is the halo center with yt units
    of kpc, to be defined. Requires the other fields of vx_corrected, vy_corrected, and vz_corrected.
    -Cassi"""
    xv = data['vx_corrected']
    yv = data['vy_corrected']
    zv = data['vz_corrected']
    center = data.ds.halo_center_kpc
    x_hat = data['x'].in_units('kpc') - center[0]
    y_hat = data['y'].in_units('kpc') - center[1]
    z_hat = data['z'].in_units('kpc') - center[2]
    r = np.sqrt(x_hat*x_hat + y_hat*y_hat + z_hat*z_hat)
    rxy = np.sqrt(x_hat*x_hat + y_hat*y_hat)
    theta_v = (xv*y_hat - x_hat*yv)/(x_hat*x_hat + y_hat*y_hat)*rxy
    return theta_v

def phi_velocity_corrected(field, data):
    """Corrects the phi direction of the spherical coordinate velocity for the bulk motion of the
    halo and the halo center. Requires 'halo_center_kpc', which is the halo center with yt units
    of kpc, to be defined. Requires the other fields of vx_corrected, vy_corrected, and vz_corrected.
    -Cassi"""
    xv = data['vx_corrected']
    yv = data['vy_corrected']
    zv = data['vz_corrected']
    center = data.ds.halo_center_kpc
    x_hat = data['x'].in_units('kpc') - center[0]
    y_hat = data['y'].in_units('kpc') - center[1]
    z_hat = data['z'].in_units('kpc') - center[2]
    r = np.sqrt(x_hat*x_hat + y_hat*y_hat + z_hat*z_hat)
    rxy = np.sqrt(x_hat*x_hat + y_hat*y_hat)
    phi_v = (z_hat*(x_hat*xv + y_hat*yv)-zv*(x_hat*x_hat + y_hat*y_hat))/(r*r*rxy)*r
    return phi_v

def tangential_velocity_corrected(field, data):
    """Returns sqrt(v_theta**2+v_phi**2), corrected for bulk flows. -Cassi"""

    return np.sqrt(data['theta_velocity_corrected']**2. + data['phi_velocity_corrected']**2.)

def radius_corrected(field, data):
    """Corrects the radius for the center of the halo. Requires 'halo_center_kpc', which is the halo
    center with yt units of kpc, to be defined. -Cassi"""
    halo_center_kpc = data.ds.halo_center_kpc
    x_hat = data['x'].in_units('kpc') - halo_center_kpc[0]
    y_hat = data['y'].in_units('kpc') - halo_center_kpc[1]
    z_hat = data['z'].in_units('kpc') - halo_center_kpc[2]
    r = np.sqrt(x_hat*x_hat + y_hat*y_hat + z_hat*z_hat)
    return r

def theta_pos(field, data):
    """Calculates the azimuthal position of cells for conversions to spherical coordinates.
    Requires 'halo_center_kpc', which is the halo center with yt units of kpc, to be defined. -Cassi"""
    halo_center_kpc = data.ds.halo_center_kpc
    x_hat = data['x'].in_units('kpc') - halo_center_kpc[0]
    y_hat = data['y'].in_units('kpc') - halo_center_kpc[1]
    z_hat = data['z'].in_units('kpc') - halo_center_kpc[2]
    r = np.sqrt(x_hat*x_hat + y_hat*y_hat + z_hat*z_hat)
    return np.arccos(z_hat/r)

def phi_pos(field, data):
    """Calculates the angular position of cells for conversions to spherical coordinates.
    Requires 'halo_center_kpc', which is the halo center with yt units of kpc, to be defined. -Cassi"""
    halo_center_kpc = data.ds.halo_center_kpc
    x_hat = data['x'].in_units('kpc') - halo_center_kpc[0]
    y_hat = data['y'].in_units('kpc') - halo_center_kpc[1]
    z_hat = data['z'].in_units('kpc') - halo_center_kpc[2]
    r = np.sqrt(x_hat*x_hat + y_hat*y_hat + z_hat*z_hat)
    return np.arctan2(y_hat, x_hat)

def kinetic_energy_corrected(field, data):
    """Calculates the kinetic energy of cells relative to the center of the halo and corrected
    for the halo velocity. Requires 'halo_velociy_kms', which is the halo velocity with yt
    units of km/s, to be defined. -Cassi"""
    return 0.5 * data['cell_mass'] * data['radial_velocity_corrected']**2.

def _no6(field,data):
    return data["dx"] * data['O_p5_number_density']

def _nh1(field,data):
    return data["dx"] * data['H_p0_number_density']

def _no5(field,data):
    return data["dx"] * data['O_p4_number_density']

def _c4(field,data):
    return data["dx"] * data['C_p3_number_density']


def x_diskrel(field, data):
    '''Returns the x-position (in kpc) in a new coordinate system aligned with the disk.
    Requires ds.disk_rot_arr to be defined as the rotation array into a coordinate system
    defined by the disk.
    Requires ds.halo_center_kpc to be defined as the center of the halo in kpc. -Cassi'''

    halo_center_kpc = data.ds.halo_center_kpc
    oldx = data['x'].in_units('kpc') - halo_center_kpc[0]
    oldy = data['y'].in_units('kpc') - halo_center_kpc[1]
    oldz = data['z'].in_units('kpc') - halo_center_kpc[2]
    newx = data.ds.disk_rot_arr[0][0]*oldx+data.ds.disk_rot_arr[0][1]*oldy+data.ds.disk_rot_arr[0][2]*oldz

    return newx.in_units('kpc')

def y_diskrel(field, data):
    '''Returns the y-position (in kpc) in a new coordinate system aligned with the disk.
    Requires ds.disk_rot_arr to be defined as the rotation array into a coordinate system
    defined by the disk.
    Requires ds.halo_center_kpc to be defined as the center of the halo in kpc. -Cassi'''

    halo_center_kpc = data.ds.halo_center_kpc
    oldx = data['x'].in_units('kpc') - halo_center_kpc[0]
    oldy = data['y'].in_units('kpc') - halo_center_kpc[1]
    oldz = data['z'].in_units('kpc') - halo_center_kpc[2]
    newy = data.ds.disk_rot_arr[1][0]*oldx+data.ds.disk_rot_arr[1][1]*oldy+data.ds.disk_rot_arr[1][2]*oldz

    return newy.in_units('kpc')

def z_diskrel(field, data):
    '''Returns the z-position (in kpc) in a new coordinate system aligned with the disk.
    Requires ds.disk_rot_arr to be defined as the rotation array into a coordinate system
    defined by the disk.
    Requires ds.halo_center_kpc to be defined as the center of the halo in kpc. -Cassi'''

    halo_center_kpc = data.ds.halo_center_kpc
    oldx = data['x'].in_units('kpc') - halo_center_kpc[0]
    oldy = data['y'].in_units('kpc') - halo_center_kpc[1]
    oldz = data['z'].in_units('kpc') - halo_center_kpc[2]
    newz = data.ds.disk_rot_arr[2][0]*oldx+data.ds.disk_rot_arr[2][1]*oldy+data.ds.disk_rot_arr[2][2]*oldz

    return newz.in_units('kpc')

def vx_diskrel(field, data):
    """Converts the x-velocity into a coordinate system defined by the disk. Requires ds.disk_rot_arr,
    which is the rotation array for the coordinate system shift, to be defined. -Cassi"""

    old_vx = data['vx_corrected']
    old_vy = data['vy_corrected']
    old_vz = data['vz_corrected']
    new_vx = data.ds.disk_rot_arr[0][0]*old_vx+data.ds.disk_rot_arr[0][1]*old_vy+data.ds.disk_rot_arr[0][2]*old_vz

    return new_vx.in_units('km/s')

def vy_diskrel(field, data):
    """Converts the y-velocity into a coordinate system defined by the disk. Requires ds.disk_rot_arr,
    which is the rotation array for the coordinate system shift, to be defined. -Cassi"""

    old_vx = data['vx_corrected']
    old_vy = data['vy_corrected']
    old_vz = data['vz_corrected']
    new_vy = data.ds.disk_rot_arr[1][0]*old_vx+data.ds.disk_rot_arr[1][1]*old_vy+data.ds.disk_rot_arr[1][2]*old_vz

    return new_vy.in_units('km/s')

def vz_diskrel(field, data):
    """Converts the z-velocity into a coordinate system defined by the disk. Requires ds.disk_rot_arr,
    which is the rotation array for the coordinate system shift, to be defined. -Cassi"""

    old_vx = data['vx_corrected']
    old_vy = data['vy_corrected']
    old_vz = data['vz_corrected']
    new_vz = data.ds.disk_rot_arr[2][0]*old_vx+data.ds.disk_rot_arr[2][1]*old_vy+data.ds.disk_rot_arr[2][2]*old_vz

    return new_vz.in_units('km/s')

def phi_pos_diskrel(field, data):
    """Calculates the azimuthal position of cells for conversions to spherical coordinates, in a
    coordinate system defined by the angular momentum vector of the disk. -Cassi"""

    x_hat = data['x_disk']
    y_hat = data['y_disk']
    z_hat = data['z_disk']
    r = np.sqrt(x_hat*x_hat + y_hat*y_hat + z_hat*z_hat)

    return np.arccos(z_hat/r)

def theta_pos_diskrel(field, data):
    """Calculates the angular position of cells for conversions to spherical coordinates, in a
    coordinate system defined by the angular momentum vector of the disk. -Cassi"""

    x_hat = data['x_disk']
    y_hat = data['y_disk']
    z_hat = data['z_disk']
    r = np.sqrt(x_hat*x_hat + y_hat*y_hat + z_hat*z_hat)

    return np.arctan2(y_hat, x_hat)

def theta_velocity_diskrel(field, data):
    """Converts disk-relative velocities into spherical velocities. theta is the direction around
    in the plane of the disk. -Cassi"""
    xv = data['vx_disk']
    yv = data['vy_disk']
    zv = data['vz_disk']
    x_hat = data['x_disk']
    y_hat = data['y_disk']
    z_hat = data['z_disk']
    r = np.sqrt(x_hat*x_hat + y_hat*y_hat + z_hat*z_hat)
    rxy = np.sqrt(x_hat*x_hat + y_hat*y_hat)
    theta_v = (xv*y_hat - x_hat*yv)/(x_hat*x_hat + y_hat*y_hat)*rxy
    return theta_v

def phi_velocity_diskrel(field, data):
    """Converts disk-relative velocities into spherical velocities. phi is the direction above and
    below the disk. -Cassi"""
    xv = data['vx_disk']
    yv = data['vy_disk']
    zv = data['vz_disk']
    x_hat = data['x_disk']
    y_hat = data['y_disk']
    z_hat = data['z_disk']
    r = np.sqrt(x_hat*x_hat + y_hat*y_hat + z_hat*z_hat)
    rxy = np.sqrt(x_hat*x_hat + y_hat*y_hat)
    phi_v = (z_hat*(x_hat*xv + y_hat*yv)-zv*(x_hat*x_hat + y_hat*y_hat))/(r*r*rxy)*r
    return phi_v

def tangential_velocity_diskrel(field, data):
    """Returns sqrt(v_theta^2 + v_phi^2), where v_theta and v_phi are oriented with the disk. -Cassi"""

    return np.sqrt(data['vtheta_disk']**2. + data['vphi_disk']**2.)
