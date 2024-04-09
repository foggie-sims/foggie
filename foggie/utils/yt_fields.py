# Note: any field definition functions that start with _ cannot be loaded
# into other modules because python makes them private functions.

import numpy as np
import yt
yt_ver = yt.__version__
if (yt_ver[0]=='3'):
    from yt.units import *
if (yt_ver[0]=='4'):
    from unyt import *

def _static_average_rampressure(field, data):
    bulk_velocity = data.get_field_parameter("bulk_velocity").in_units('km/s')
    velx = data['enzo', 'x-velocity'].in_units('km/s') - bulk_velocity[0]
    vely = data['enzo', 'y-velocity'].in_units('km/s') - bulk_velocity[1]
    velz = data['enzo', 'z-velocity'].in_units('km/s') - bulk_velocity[2]
    vel = np.sqrt(velx**2. + vely**2. + velz**2.)/np.sqrt(3)
    rp = data['density'] * vel**2.
    return np.log10(rp.in_units('dyne/cm**2').value)

def _static_radial_rampressure(field, data):
    vel = data['gas', 'radial_velocity']
    vel[vel<0] = 0.
    rp = data['gas','density'] * vel**2.
    return np.log10(rp.in_units('dyne/cm**2').value)

def _radial_rampressure(field, data):

    vel = data['gas', 'circular_velocity'] + data['gas', 'radial_velocity']
    vel[vel<0] = 0.
    rp = data['gas','density'] * vel**2.
    return np.log10(rp.in_units('dyne/cm**2').value)


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

def _young_stars3(pfilter, data):
    """Filter star particles with creation time < 3 Myr ago
    To use: yt.add_particle_filter("young_stars3", function=_young_stars, filtered_type='all', requires=["creation_time"])"""

    age = data.ds.current_time - data[pfilter.filtered_type, "creation_time"]
    filter = np.logical_and(age.in_units('Myr') <= 3, age >= 0)
    return filter

def _young_stars7(pfilter, data):
    """Filter star particles with creation time < 10 Myr ago
    To use: yt.add_particle_filter("young_stars7", function=_young_stars7, filtered_type='all', requires=["creation_time"])"""

    age = data.ds.current_time - data[pfilter.filtered_type, "creation_time"]
    filter = np.logical_and(age.in_units('Myr') <= 10, age >= 0)
    return filter

def _young_stars8(pfilter, data):
    """Filter star particles with creation time < 100 Myr ago
    To use: yt.add_particle_filter("young_stars8", function=_young_stars8, filtered_type='all', requires=["creation_time"])"""

    age = data.ds.current_time - data[pfilter.filtered_type, "creation_time"]
    filter = np.logical_and(age.in_units('Myr') <= 100, age >= 0)
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
    return -1*data['gas','cooling_time'] / ((data['gas','dx']/data['gas','sound_speed']).in_units('s'))




def get_particle_relative_specific_angular_momentum(ptype):
    def particle_relative_specific_angular_momentum(field, data):
        """
        Added by Raymond. Native YT "particle angular momentum" re-defines Z wrt the given normal direction.
        These definitions define it wrt to the cartesian grid, to be consistent with gas.
        Calculate the angular of a particle velocity.

        Returns a vector for each particle.
        """
        #ptype = field[0]
        #print (field)
        pos = data.ds.arr([data[ptype, f"relative_particle_position_%s" % ax] for ax in "xyz"]).T
        vel = data.ds.arr([data[ptype, f"relative_particle_velocity_%s" % ax] for ax in "xyz"]).T
        return ucross(pos, vel, registry=data.ds.unit_registry)
    return particle_relative_specific_angular_momentum


def get_particle_relative_specific_angular_momentum_x(ptype):
    def particle_relative_specific_angular_momentum_x(field, data):
        return data[ptype, "particle_relative_specific_angular_momentum"][:, 0]
    return particle_relative_specific_angular_momentum_x


def get_particle_relative_specific_angular_momentum_y(ptype):
    def particle_relative_specific_angular_momentum_y(field, data):
        return data[ptype, "particle_relative_specific_angular_momentum"][:, 1]
    return particle_relative_specific_angular_momentum_y

def get_particle_relative_specific_angular_momentum_z(ptype):
    def particle_relative_specific_angular_momentum_z(field, data):
        return data[ptype, "particle_relative_specific_angular_momentum"][:, 2]
    return particle_relative_specific_angular_momentum_z

def get_particle_relative_angular_momentum_x(ptype):
    def particle_relative_angular_momentum_x(field, data):
        return data[ptype, "particle_mass"] * data[ptype, f"particle_relative_specific_angular_momentum_x"]
    return particle_relative_angular_momentum_x

def get_particle_relative_angular_momentum_y(ptype):
    def particle_relative_angular_momentum_y(field, data):
        return data[ptype, "particle_mass"] * data[ptype, f"particle_relative_specific_angular_momentum_y"]
    return particle_relative_angular_momentum_y

def get_particle_relative_angular_momentum_z(ptype):
    def particle_relative_angular_momentum_z(field, data):
        return data[ptype, "particle_mass"] * data[ptype, f"particle_relative_specific_angular_momentum_z"]
    return particle_relative_angular_momentum_z

def phi_angular_momentum(field, data):
    '''
    Function to compute the phi direction of the angular momentum vector
    Added here by Ayan
    Based onRaymond's foggie.angular_momentum.lasso_data_selection.ipynb since the function cannot be imported from notebook
    '''
    if ('dm' in field.name[0]) | ('stars' in field.name[0]):
        name = 'particle_angular_momentum'
    else:
        name = 'angular_momentum'
    Lx = data['%s_x' % name]
    Ly = data['%s_y' % name]
    Lz = data['%s_z' % name]
    L_tot = np.sqrt(Lx ** 2. + Ly ** 2. + Lz ** 2.)
    phi_L = np.arccos(Lz / L_tot) * 180. / np.pi
    phi_L[np.isnan(phi_L)] = 0.

    try: phi_L = unyt_array(phi_L, 'deg')
    except NameError: phi_L = yt.YTArray(phi_L, 'deg')
    return phi_L

def theta_angular_momentum(field, data):
    '''
    Function to compute the theta direction of the angular momentum vector
    Added here by Ayan
    Basically copied form Raymond's foggie.angular_momentum.lasso_data_selection.ipynb since the function cannot be imported from notebook
    '''
    if ('dm' in field.name[0]) | ('stars' in field.name[0]):
        name = 'particle_angular_momentum'
    else:
        name = 'angular_momentum'

    Lx = data['%s_x' % name]
    Ly = data['%s_y' % name]

    theta_L = np.arctan2(Ly, Lx) * 180. / np.pi
    theta_L[np.isnan(theta_L)] = 0.

    try: theta_L = unyt_array(theta_L, 'deg')
    except NameError: theta_L = yt.YTArray(theta_L, 'deg')
    return theta_L

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

def vel_mag_corrected(field, data):
    """Corrects the velocity magnitude for bulk motion of the halo. Requires 'halo_velocity_kms',
    which is the halo velocity with yt units of km/s, to be defined. -Cassi"""

    return np.sqrt(data['gas','vx_corrected']**2. + data['gas','vy_corrected']**2. + data['gas','vz_corrected']**2.)

def radial_velocity_corrected(field, data):
    """Corrects the radial velocity for bulk motion of the halo and the halo center.
    Requires 'halo_center_kpc', which is the halo center with yt units of kpc, to be defined.
    Requires the other fields of _vx_corrected, _vy_corrected, and _vz_corrected. -Cassi"""
    halo_center_kpc = data.ds.halo_center_kpc
    x_hat = data['gas','x'].in_units('kpc') - halo_center_kpc[0]
    y_hat = data['gas','y'].in_units('kpc') - halo_center_kpc[1]
    z_hat = data['gas','z'].in_units('kpc') - halo_center_kpc[2]
    r = np.sqrt(x_hat*x_hat + y_hat*y_hat + z_hat*z_hat)
    x_hat /= r
    y_hat /= r
    z_hat /= r
    vr = data['gas','vx_corrected']*x_hat + data['gas','vy_corrected']*y_hat + data['gas','vz_corrected']*z_hat
    vr[np.isnan(vr)] = 0.
    return vr

def radial_velocity_corrected_dm(field, data):
    """Corrects the radial velocity for bulk motion of the halo and the halo center.
    Requires 'halo_center_kpc', which is the halo center with yt units of kpc, to be defined.
    Requires the other fields of _vx_corrected, _vy_corrected, and _vz_corrected. -Cassi"""
    halo_center_kpc = data.ds.halo_center_kpc
    halo_velocity_kms = data.ds.halo_velocity_kms
    x_hat = data['dm','particle_position_x'].in_units('kpc') - halo_center_kpc[0]
    y_hat = data['dm','particle_position_y'].in_units('kpc') - halo_center_kpc[1]
    z_hat = data['dm','particle_position_z'].in_units('kpc') - halo_center_kpc[2]
    r = np.sqrt(x_hat*x_hat + y_hat*y_hat + z_hat*z_hat)
    x_hat /= r
    y_hat /= r
    z_hat /= r
    vx = data['dm','particle_velocity_x'].in_units('km/s') - halo_velocity_kms[0]
    vy = data['dm','particle_velocity_y'].in_units('km/s') - halo_velocity_kms[1]
    vz = data['dm','particle_velocity_z'].in_units('km/s') - halo_velocity_kms[2]
    vr = vx*x_hat + vy*y_hat + vz*z_hat
    vr[np.isnan(vr)] = 0.
    return vr

def theta_velocity_corrected(field, data):
    """Corrects the theta direction of the spherical coordinate velocity for the bulk motion of the
    halo and the halo center. Requires 'halo_center_kpc', which is the halo center with yt units
    of kpc, to be defined. Requires the other fields of vx_corrected, vy_corrected, and vz_corrected.
    -Cassi"""
    xv = data['gas','vx_corrected']
    yv = data['gas','vy_corrected']
    zv = data['gas','vz_corrected']
    center = data.ds.halo_center_kpc
    x_hat = data['gas','x'].in_units('kpc') - center[0]
    y_hat = data['gas','y'].in_units('kpc') - center[1]
    z_hat = data['gas','z'].in_units('kpc') - center[2]
    r = np.sqrt(x_hat*x_hat + y_hat*y_hat + z_hat*z_hat)
    rxy = np.sqrt(x_hat*x_hat + y_hat*y_hat)
    theta_v = (xv*y_hat - x_hat*yv)/(x_hat*x_hat + y_hat*y_hat)*rxy
    theta_v[np.isnan(theta_v)] = 0.
    return theta_v

def phi_velocity_corrected(field, data):
    """Corrects the phi direction of the spherical coordinate velocity for the bulk motion of the
    halo and the halo center. Requires 'halo_center_kpc', which is the halo center with yt units
    of kpc, to be defined. Requires the other fields of vx_corrected, vy_corrected, and vz_corrected.
    -Cassi"""
    xv = data['gas','vx_corrected']
    yv = data['gas','vy_corrected']
    zv = data['gas','vz_corrected']
    center = data.ds.halo_center_kpc
    x_hat = data['gas','x'].in_units('kpc') - center[0]
    y_hat = data['gas','y'].in_units('kpc') - center[1]
    z_hat = data['gas','z'].in_units('kpc') - center[2]
    r = np.sqrt(x_hat*x_hat + y_hat*y_hat + z_hat*z_hat)
    rxy = np.sqrt(x_hat*x_hat + y_hat*y_hat)
    phi_v = (z_hat*(x_hat*xv + y_hat*yv)-zv*(x_hat*x_hat + y_hat*y_hat))/(r*r*rxy)*r
    phi_v[np.isnan(phi_v)] = 0.
    return phi_v

def tangential_velocity_corrected(field, data):
    """Returns sqrt(v_theta**2+v_phi**2), corrected for bulk flows. -Cassi"""

    return np.sqrt(data['gas','theta_velocity_corrected']**2. + data['gas','phi_velocity_corrected']**2.)

def radius_corrected(field, data):
    """Corrects the radius for the center of the halo. Requires 'halo_center_kpc', which is the halo
    center with yt units of kpc, to be defined. -Cassi"""
    halo_center_kpc = data.ds.halo_center_kpc
    x_hat = data['gas','x'].in_units('kpc') - halo_center_kpc[0]
    y_hat = data['gas','y'].in_units('kpc') - halo_center_kpc[1]
    z_hat = data['gas','z'].in_units('kpc') - halo_center_kpc[2]
    r = np.sqrt(x_hat*x_hat + y_hat*y_hat + z_hat*z_hat)
    return r

def radius_corrected_stars(field, data):
    """Corrects the radius for star particles for the center of the halo. Requires 'halo_center_kpc', which is the halo
    center with yt units of kpc, to be defined. -Cassi"""
    halo_center_kpc = data.ds.halo_center_kpc
    x_hat = data['stars','particle_position_x'].in_units('kpc') - halo_center_kpc[0]
    y_hat = data['stars','particle_position_y'].in_units('kpc') - halo_center_kpc[1]
    z_hat = data['stars','particle_position_z'].in_units('kpc') - halo_center_kpc[2]
    r = np.sqrt(x_hat*x_hat + y_hat*y_hat + z_hat*z_hat)
    return r

def radius_corrected_young_stars(field, data):
    """Corrects the radius for star particles for the center of the halo. Requires 'halo_center_kpc', which is the halo
    center with yt units of kpc, to be defined. -Cassi"""
    halo_center_kpc = data.ds.halo_center_kpc
    x_hat = data['young_stars','particle_position_x'].in_units('kpc') - halo_center_kpc[0]
    y_hat = data['young_stars','particle_position_y'].in_units('kpc') - halo_center_kpc[1]
    z_hat = data['young_stars','particle_position_z'].in_units('kpc') - halo_center_kpc[2]
    r = np.sqrt(x_hat*x_hat + y_hat*y_hat + z_hat*z_hat)
    return r

def radius_corrected_young_stars8(field, data):
    """Corrects the radius for star particles for the center of the halo. Requires 'halo_center_kpc', which is the halo
    center with yt units of kpc, to be defined. -Cassi"""
    halo_center_kpc = data.ds.halo_center_kpc
    x_hat = data['young_stars8','particle_position_x'].in_units('kpc') - halo_center_kpc[0]
    y_hat = data['young_stars8','particle_position_y'].in_units('kpc') - halo_center_kpc[1]
    z_hat = data['young_stars8','particle_position_z'].in_units('kpc') - halo_center_kpc[2]
    r = np.sqrt(x_hat*x_hat + y_hat*y_hat + z_hat*z_hat)
    return r

def radius_corrected_old_stars(field, data):
    """Corrects the radius for star particles for the center of the halo. Requires 'halo_center_kpc', which is the halo
    center with yt units of kpc, to be defined. -Cassi"""
    halo_center_kpc = data.ds.halo_center_kpc
    x_hat = data['old_stars','particle_position_x'].in_units('kpc') - halo_center_kpc[0]
    y_hat = data['old_stars','particle_position_y'].in_units('kpc') - halo_center_kpc[1]
    z_hat = data['old_stars','particle_position_z'].in_units('kpc') - halo_center_kpc[2]
    r = np.sqrt(x_hat*x_hat + y_hat*y_hat + z_hat*z_hat)
    return r

def radius_corrected_dm(field, data):
    """Corrects the radius for DM particles for the center of the halo. Requires 'halo_center_kpc', which is the halo
    center with yt units of kpc, to be defined. -Cassi"""
    halo_center_kpc = data.ds.halo_center_kpc
    x_hat = data['dm','particle_position_x'].in_units('kpc') - halo_center_kpc[0]
    y_hat = data['dm','particle_position_y'].in_units('kpc') - halo_center_kpc[1]
    z_hat = data['dm','particle_position_z'].in_units('kpc') - halo_center_kpc[2]
    r = np.sqrt(x_hat*x_hat + y_hat*y_hat + z_hat*z_hat)
    return r

def theta_pos(field, data):
    """Calculates the azimuthal position of cells for conversions to spherical coordinates.
    Requires 'halo_center_kpc', which is the halo center with yt units of kpc, to be defined. -Cassi"""
    halo_center_kpc = data.ds.halo_center_kpc
    x_hat = data['gas','x'].in_units('kpc') - halo_center_kpc[0]
    y_hat = data['gas','y'].in_units('kpc') - halo_center_kpc[1]
    z_hat = data['gas','z'].in_units('kpc') - halo_center_kpc[2]
    r = np.sqrt(x_hat*x_hat + y_hat*y_hat + z_hat*z_hat)
    return np.arccos(z_hat/r)

def phi_pos(field, data):
    """Calculates the angular position of cells for conversions to spherical coordinates.
    Requires 'halo_center_kpc', which is the halo center with yt units of kpc, to be defined. -Cassi"""
    halo_center_kpc = data.ds.halo_center_kpc
    x_hat = data['gas','x'].in_units('kpc') - halo_center_kpc[0]
    y_hat = data['gas','y'].in_units('kpc') - halo_center_kpc[1]
    z_hat = data['gas','z'].in_units('kpc') - halo_center_kpc[2]
    r = np.sqrt(x_hat*x_hat + y_hat*y_hat + z_hat*z_hat)
    return np.arctan2(y_hat, x_hat)

def kinetic_energy_corrected(field, data):
    """Calculates the kinetic energy of cells corrected
    for the halo velocity. Requires 'halo_velociy_kms', which is the halo velocity with yt
    units of km/s, to be defined. -Cassi"""
    return 0.5 * data['gas','cell_mass'] * data['gas','vel_mag_corrected']**2.

def radial_kinetic_energy(field, data):
    """Calculates the radial kinetic energy of cells corrected
    for the halo velocity. Requires 'halo_velociy_kms', which is the halo velocity with yt
    units of km/s, to be defined. -Cassi"""
    return 0.5 * data['gas','cell_mass'] * data['gas','radial_velocity_corrected']**2.

def tangential_kinetic_energy(field, data):
    """Calculates the tangential kinetic energy of cells corrected
    for the halo velocity. Requires 'halo_velociy_kms', which is the halo velocity with yt
    units of km/s, to be defined. -Cassi"""
    return 0.5 * data['gas','cell_mass'] * data['gas','tangential_velocity_corrected']**2.

def get_cell_ids(field, data): 
    """Assigns each cell a unique integer ID for indexing and tracking. -JT"""
    return np.arange(np.size(data[('gas', 'density')]), dtype=np.int_)

def _no6(field,data):
    return data['gas',"dx"] * data['gas','O_p5_number_density']

def _nh1(field,data):
    return data['gas',"dx"] * data['gas','H_p0_number_density']

def _no5(field,data):
    return data['gas',"dx"] * data['gas','O_p4_number_density']

def _c4(field,data):
    return data['gas',"dx"] * data['gas','C_p3_number_density']


def x_diskrel(field, data):
    '''Returns the x-position (in kpc) in a new coordinate system aligned with the disk.
    Requires ds.disk_rot_arr to be defined as the rotation array into a coordinate system
    defined by the disk.
    Requires ds.halo_center_kpc to be defined as the center of the halo in kpc. -Cassi'''

    halo_center_kpc = data.ds.halo_center_kpc
    oldx = data['gas','x'].in_units('kpc') - halo_center_kpc[0]
    oldy = data['gas','y'].in_units('kpc') - halo_center_kpc[1]
    oldz = data['gas','z'].in_units('kpc') - halo_center_kpc[2]
    newx = data.ds.disk_rot_arr[0][0]*oldx+data.ds.disk_rot_arr[0][1]*oldy+data.ds.disk_rot_arr[0][2]*oldz

    return newx.in_units('kpc')

def y_diskrel(field, data):
    '''Returns the y-position (in kpc) in a new coordinate system aligned with the disk.
    Requires ds.disk_rot_arr to be defined as the rotation array into a coordinate system
    defined by the disk.
    Requires ds.halo_center_kpc to be defined as the center of the halo in kpc. -Cassi'''

    halo_center_kpc = data.ds.halo_center_kpc
    oldx = data['gas','x'].in_units('kpc') - halo_center_kpc[0]
    oldy = data['gas','y'].in_units('kpc') - halo_center_kpc[1]
    oldz = data['gas','z'].in_units('kpc') - halo_center_kpc[2]
    newy = data.ds.disk_rot_arr[1][0]*oldx+data.ds.disk_rot_arr[1][1]*oldy+data.ds.disk_rot_arr[1][2]*oldz

    return newy.in_units('kpc')

def z_diskrel(field, data):
    '''Returns the z-position (in kpc) in a new coordinate system aligned with the disk.
    Requires ds.disk_rot_arr to be defined as the rotation array into a coordinate system
    defined by the disk.
    Requires ds.halo_center_kpc to be defined as the center of the halo in kpc. -Cassi'''

    halo_center_kpc = data.ds.halo_center_kpc
    oldx = data['gas','x'].in_units('kpc') - halo_center_kpc[0]
    oldy = data['gas','y'].in_units('kpc') - halo_center_kpc[1]
    oldz = data['gas','z'].in_units('kpc') - halo_center_kpc[2]
    newz = data.ds.disk_rot_arr[2][0]*oldx+data.ds.disk_rot_arr[2][1]*oldy+data.ds.disk_rot_arr[2][2]*oldz

    return newz.in_units('kpc')

def x_diskrel_dm(field, data):
    '''Returns the x-position (in kpc) in a new coordinate system aligned with the disk.
    Requires ds.disk_rot_arr to be defined as the rotation array into a coordinate system
    defined by the disk.
    Requires ds.halo_center_kpc to be defined as the center of the halo in kpc. -Cassi'''

    halo_center_kpc = data.ds.halo_center_kpc
    oldx = data['dm','particle_position_x'].in_units('kpc') - halo_center_kpc[0]
    oldy = data['dm','particle_position_y'].in_units('kpc') - halo_center_kpc[1]
    oldz = data['dm','particle_position_z'].in_units('kpc') - halo_center_kpc[2]
    newx = data.ds.disk_rot_arr[0][0]*oldx+data.ds.disk_rot_arr[0][1]*oldy+data.ds.disk_rot_arr[0][2]*oldz

    return newx.in_units('kpc')

def y_diskrel_dm(field, data):
    '''Returns the y-position (in kpc) in a new coordinate system aligned with the disk.
    Requires ds.disk_rot_arr to be defined as the rotation array into a coordinate system
    defined by the disk.
    Requires ds.halo_center_kpc to be defined as the center of the halo in kpc. -Cassi'''

    halo_center_kpc = data.ds.halo_center_kpc
    oldx = data['dm','particle_position_x'].in_units('kpc') - halo_center_kpc[0]
    oldy = data['dm','particle_position_y'].in_units('kpc') - halo_center_kpc[1]
    oldz = data['dm','particle_position_z'].in_units('kpc') - halo_center_kpc[2]
    newy = data.ds.disk_rot_arr[1][0]*oldx+data.ds.disk_rot_arr[1][1]*oldy+data.ds.disk_rot_arr[1][2]*oldz

    return newy.in_units('kpc')

def z_diskrel_dm(field, data):
    '''Returns the z-position (in kpc) in a new coordinate system aligned with the disk.
    Requires ds.disk_rot_arr to be defined as the rotation array into a coordinate system
    defined by the disk.
    Requires ds.halo_center_kpc to be defined as the center of the halo in kpc. -Cassi'''

    halo_center_kpc = data.ds.halo_center_kpc
    oldx = data['dm','particle_position_x'].in_units('kpc') - halo_center_kpc[0]
    oldy = data['dm','particle_position_y'].in_units('kpc') - halo_center_kpc[1]
    oldz = data['dm','particle_position_z'].in_units('kpc') - halo_center_kpc[2]
    newz = data.ds.disk_rot_arr[2][0]*oldx+data.ds.disk_rot_arr[2][1]*oldy+data.ds.disk_rot_arr[2][2]*oldz

    return newz.in_units('kpc')

def x_diskrel_stars(field, data):
    '''Returns the x-position (in kpc) in a new coordinate system aligned with the disk.
    Requires ds.disk_rot_arr to be defined as the rotation array into a coordinate system
    defined by the disk.
    Requires ds.halo_center_kpc to be defined as the center of the halo in kpc. -Cassi'''

    halo_center_kpc = data.ds.halo_center_kpc
    oldx = data['stars','particle_position_x'].in_units('kpc') - halo_center_kpc[0]
    oldy = data['stars','particle_position_y'].in_units('kpc') - halo_center_kpc[1]
    oldz = data['stars','particle_position_z'].in_units('kpc') - halo_center_kpc[2]
    newx = data.ds.disk_rot_arr[0][0]*oldx+data.ds.disk_rot_arr[0][1]*oldy+data.ds.disk_rot_arr[0][2]*oldz

    return newx.in_units('kpc')

def y_diskrel_stars(field, data):
    '''Returns the y-position (in kpc) in a new coordinate system aligned with the disk.
    Requires ds.disk_rot_arr to be defined as the rotation array into a coordinate system
    defined by the disk.
    Requires ds.halo_center_kpc to be defined as the center of the halo in kpc. -Cassi'''

    halo_center_kpc = data.ds.halo_center_kpc
    oldx = data['stars','particle_position_x'].in_units('kpc') - halo_center_kpc[0]
    oldy = data['stars','particle_position_y'].in_units('kpc') - halo_center_kpc[1]
    oldz = data['stars','particle_position_z'].in_units('kpc') - halo_center_kpc[2]
    newy = data.ds.disk_rot_arr[1][0]*oldx+data.ds.disk_rot_arr[1][1]*oldy+data.ds.disk_rot_arr[1][2]*oldz

    return newy.in_units('kpc')

def z_diskrel_stars(field, data):
    '''Returns the z-position (in kpc) in a new coordinate system aligned with the disk.
    Requires ds.disk_rot_arr to be defined as the rotation array into a coordinate system
    defined by the disk.
    Requires ds.halo_center_kpc to be defined as the center of the halo in kpc. -Cassi'''

    halo_center_kpc = data.ds.halo_center_kpc
    oldx = data['stars','particle_position_x'].in_units('kpc') - halo_center_kpc[0]
    oldy = data['stars','particle_position_y'].in_units('kpc') - halo_center_kpc[1]
    oldz = data['stars','particle_position_z'].in_units('kpc') - halo_center_kpc[2]
    newz = data.ds.disk_rot_arr[2][0]*oldx+data.ds.disk_rot_arr[2][1]*oldy+data.ds.disk_rot_arr[2][2]*oldz

    return newz.in_units('kpc')

def x_diskrel_young_stars8(field, data):
    '''Returns the x-position (in kpc) in a new coordinate system aligned with the disk.
    Requires ds.disk_rot_arr to be defined as the rotation array into a coordinate system
    defined by the disk.
    Requires ds.halo_center_kpc to be defined as the center of the halo in kpc. -Cassi'''

    halo_center_kpc = data.ds.halo_center_kpc
    oldx = data['young_stars8','particle_position_x'].in_units('kpc') - halo_center_kpc[0]
    oldy = data['young_stars8','particle_position_y'].in_units('kpc') - halo_center_kpc[1]
    oldz = data['young_stars8','particle_position_z'].in_units('kpc') - halo_center_kpc[2]
    newx = data.ds.disk_rot_arr[0][0]*oldx+data.ds.disk_rot_arr[0][1]*oldy+data.ds.disk_rot_arr[0][2]*oldz

    return newx.in_units('kpc')

def y_diskrel_young_stars8(field, data):
    '''Returns the y-position (in kpc) in a new coordinate system aligned with the disk.
    Requires ds.disk_rot_arr to be defined as the rotation array into a coordinate system
    defined by the disk.
    Requires ds.halo_center_kpc to be defined as the center of the halo in kpc. -Cassi'''

    halo_center_kpc = data.ds.halo_center_kpc
    oldx = data['young_stars8','particle_position_x'].in_units('kpc') - halo_center_kpc[0]
    oldy = data['young_stars8','particle_position_y'].in_units('kpc') - halo_center_kpc[1]
    oldz = data['young_stars8','particle_position_z'].in_units('kpc') - halo_center_kpc[2]
    newy = data.ds.disk_rot_arr[1][0]*oldx+data.ds.disk_rot_arr[1][1]*oldy+data.ds.disk_rot_arr[1][2]*oldz

    return newy.in_units('kpc')

def z_diskrel_young_stars8(field, data):
    '''Returns the z-position (in kpc) in a new coordinate system aligned with the disk.
    Requires ds.disk_rot_arr to be defined as the rotation array into a coordinate system
    defined by the disk.
    Requires ds.halo_center_kpc to be defined as the center of the halo in kpc. -Cassi'''

    halo_center_kpc = data.ds.halo_center_kpc
    oldx = data['young_stars8','particle_position_x'].in_units('kpc') - halo_center_kpc[0]
    oldy = data['young_stars8','particle_position_y'].in_units('kpc') - halo_center_kpc[1]
    oldz = data['young_stars8','particle_position_z'].in_units('kpc') - halo_center_kpc[2]
    newz = data.ds.disk_rot_arr[2][0]*oldx+data.ds.disk_rot_arr[2][1]*oldy+data.ds.disk_rot_arr[2][2]*oldz

    return newz.in_units('kpc')

def vx_diskrel(field, data):
    """Converts the x-velocity into a coordinate system defined by the disk. Requires ds.disk_rot_arr,
    which is the rotation array for the coordinate system shift, to be defined. -Cassi"""

    old_vx = data['gas','vx_corrected']
    old_vy = data['gas','vy_corrected']
    old_vz = data['gas','vz_corrected']
    new_vx = data.ds.disk_rot_arr[0][0]*old_vx+data.ds.disk_rot_arr[0][1]*old_vy+data.ds.disk_rot_arr[0][2]*old_vz

    return new_vx.in_units('km/s')

def vy_diskrel(field, data):
    """Converts the y-velocity into a coordinate system defined by the disk. Requires ds.disk_rot_arr,
    which is the rotation array for the coordinate system shift, to be defined. -Cassi"""

    old_vx = data['gas','vx_corrected']
    old_vy = data['gas','vy_corrected']
    old_vz = data['gas','vz_corrected']
    new_vy = data.ds.disk_rot_arr[1][0]*old_vx+data.ds.disk_rot_arr[1][1]*old_vy+data.ds.disk_rot_arr[1][2]*old_vz

    return new_vy.in_units('km/s')

def vz_diskrel(field, data):
    """Converts the z-velocity into a coordinate system defined by the disk. Requires ds.disk_rot_arr,
    which is the rotation array for the coordinate system shift, to be defined. -Cassi"""

    old_vx = data['gas','vx_corrected']
    old_vy = data['gas','vy_corrected']
    old_vz = data['gas','vz_corrected']
    new_vz = data.ds.disk_rot_arr[2][0]*old_vx+data.ds.disk_rot_arr[2][1]*old_vy+data.ds.disk_rot_arr[2][2]*old_vz

    return new_vz.in_units('km/s')

def phi_pos_diskrel(field, data):
    """Calculates the azimuthal position of cells for conversions to spherical coordinates, in a
    coordinate system defined by the angular momentum vector of the disk. -Cassi"""

    x_hat = data['gas','x_disk']
    y_hat = data['gas','y_disk']
    z_hat = data['gas','z_disk']
    r = np.sqrt(x_hat*x_hat + y_hat*y_hat + z_hat*z_hat)

    return np.arccos(z_hat/r)

def theta_pos_diskrel(field, data):
    """Calculates the angular position of cells for conversions to spherical coordinates, in a
    coordinate system defined by the angular momentum vector of the disk. -Cassi"""

    x_hat = data['gas','x_disk']
    y_hat = data['gas','y_disk']
    z_hat = data['gas','z_disk']
    r = np.sqrt(x_hat*x_hat + y_hat*y_hat + z_hat*z_hat)

    return np.arctan2(y_hat, x_hat)

def phi_pos_diskrel_dm(field, data):
    """Calculates the azimuthal position of cells for conversions to spherical coordinates, in a
    coordinate system defined by the angular momentum vector of the disk. -Cassi"""

    x_hat = data['dm','x_disk']
    y_hat = data['dm','y_disk']
    z_hat = data['dm','z_disk']
    r = np.sqrt(x_hat*x_hat + y_hat*y_hat + z_hat*z_hat)

    return np.arccos(z_hat/r)

def theta_pos_diskrel_dm(field, data):
    """Calculates the angular position of cells for conversions to spherical coordinates, in a
    coordinate system defined by the angular momentum vector of the disk. -Cassi"""

    x_hat = data['dm','x_disk']
    y_hat = data['dm','y_disk']
    z_hat = data['dm','z_disk']
    r = np.sqrt(x_hat*x_hat + y_hat*y_hat + z_hat*z_hat)

    return np.arctan2(y_hat, x_hat)

def theta_velocity_diskrel(field, data):
    """Converts disk-relative velocities into spherical velocities. theta is the direction around
    in the plane of the disk. -Cassi"""
    xv = data['gas','vx_disk']
    yv = data['gas','vy_disk']
    zv = data['gas','vz_disk']
    x_hat = data['gas','x_disk']
    y_hat = data['gas','y_disk']
    z_hat = data['gas','z_disk']
    r = np.sqrt(x_hat*x_hat + y_hat*y_hat + z_hat*z_hat)
    rxy = np.sqrt(x_hat*x_hat + y_hat*y_hat)
    theta_v = (xv*y_hat - x_hat*yv)/(x_hat*x_hat + y_hat*y_hat)*rxy
    theta_v[np.isnan(theta_v)] = 0.
    return theta_v

def phi_velocity_diskrel(field, data):
    """Converts disk-relative velocities into spherical velocities. phi is the direction above and
    below the disk. -Cassi"""
    xv = data['gas','vx_disk']
    yv = data['gas','vy_disk']
    zv = data['gas','vz_disk']
    x_hat = data['gas','x_disk']
    y_hat = data['gas','y_disk']
    z_hat = data['gas','z_disk']
    r = np.sqrt(x_hat*x_hat + y_hat*y_hat + z_hat*z_hat)
    rxy = np.sqrt(x_hat*x_hat + y_hat*y_hat)
    phi_v = (z_hat*(x_hat*xv + y_hat*yv)-zv*(x_hat*x_hat + y_hat*y_hat))/(r*r*rxy)*r
    phi_v[np.isnan(phi_v)] = 0.
    return phi_v

def tangential_velocity_diskrel(field, data):
    """Returns sqrt(v_theta^2 + v_phi^2), where v_theta and v_phi are oriented with the disk. -Cassi"""

    return np.sqrt(data['gas','vtheta_disk']**2. + data['gas','vphi_disk']**2.)

def tangential_kinetic_energy_diskrel(field, data):
    """Calculates the tangential kinetic energy of cells corrected
    for the halo velocity, relative to the disk. Requires 'halo_velociy_kms', which is the halo velocity with yt
    units of km/s, to be defined. -Cassi"""
    return 0.5 * data['gas','cell_mass'] * data['gas','vtan_disk']**2.

def t_ff(field, data):
    """Returns the free-fall time of the gas. Note tff is an interpolated function of radius so
    this value will be the same for all cells with the same radius."""

    rho = data.ds.Menc_profile(data['gas','radius_corrected'])*Msun/(data['gas','radius_corrected']**3.) * 3./(4.*np.pi)
    return np.sqrt(3.*np.pi/(32.*G*rho))

def v_ff(field, data):
    """Returns the free-fall velocity of the gas. Note vff is an interpolated function of radius so
    this value will be the same for all cells with the same radius."""

    return -np.sqrt((2.*G*data.ds.Menc_profile(data['gas','radius_corrected'])*Msun)/(data['gas','radius_corrected']))

def v_ff_dm(field, data):
    """Returns the free-fall velocity of the dark matter particles. Note vff is an interpolated function of radius so
    this value will be the same for all cells with the same radius."""

    return -np.sqrt((2.*G*data.ds.Menc_profile(data['dm','radius_corrected'])*Msun)/(data['dm','radius_corrected']))

def v_esc(field, data):
    """Returns the escape velocity of the gas. Note vesc is an interpolated function of radius so this
    value will be the same for all cells with the same radius."""

    vesc = np.sqrt(2.*G*data.ds.Menc_profile(data['gas','radius_corrected'])*Msun/(data['gas','radius_corrected']))
    return vesc

def tcool_tff_ratio(field, data):
    """Returns the ratio of cooling time to free-fall time of the gas. Note tff is an interpolated
    function of radius based on the enclosed mass profiles."""

    return data['gas','cooling_time']/data['tff']

def cell_mass_msun(field, data):
    """Returns the cell mass in units of Msun rather than the default of grams."""

    return data['gas','cell_mass'].in_units('Msun')

def grav_pot(field, data):
    Menc = data.ds.Menc_profile(data['gas','radius_corrected'])*Msun
    return G.in_units('cm**3/g/s**2')*Menc.in_units('g')/data['gas','radius_corrected'].in_units('cm')

def hse_ratio(field, data):
    center = data.ds.halo_center_kpc
    x_hat = data['gas',"x"].in_units('kpc') - center[0]
    y_hat = data['gas',"y"].in_units('kpc') - center[1]
    z_hat = data['gas',"z"].in_units('kpc') - center[2]
    r = np.sqrt(x_hat*x_hat+y_hat*y_hat+z_hat*z_hat)
    gx = -data['gas',"density"] * data['gas',"grav_pot_gradient_x"]
    gy = -data['gas',"density"] * data['gas',"grav_pot_gradient_y"]
    gz = -data['gas',"density"] * data['gas',"grav_pot_gradient_z"]
    x_hat /= r
    y_hat /= r
    z_hat /= r
    gr = gx*x_hat + gy*y_hat + gz*z_hat
    pr = data['gas','pressure_gradient_x']*x_hat + data['gas','pressure_gradient_y']*y_hat + data['gas','pressure_gradient_z']*z_hat
    return np.sqrt(pr**2./gr**2.)
