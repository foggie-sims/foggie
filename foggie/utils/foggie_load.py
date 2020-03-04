import numpy as np
import yt
from yt.units import *
from yt import YTArray
from astropy.table import Table

from foggie.utils.consistency import *
from foggie.utils.get_halo_center import get_halo_center
from foggie.utils.get_proper_box_size import get_proper_box_size
from foggie.utils.yt_fields import *
from foggie.utils.foggie_utils import filter_particles
import foggie.utils as futils
import foggie.utils.get_refine_box as grb

def load(snap, trackfile, **kwargs):
    """This function loads a specified snapshot named by 'snap', the halo track "trackfile'
    Based off of a helper function to flux_tracking written by Cassi, adapted for utils by JT."""
    use_halo_c_v = kwargs.get('use_halo_c_v', False)
    halo_c_v_name = kwargs.get('halo_c_v_name', 'halo_c_v')
    disk_relative = kwargs.get('disk_relative', False)
    particle_type_for_angmom = kwargs.get('particle_type_for_angmom', 'young_stars')
    do_filter_particles = kwargs.get('do_filter_particles', True)

    print ('Opening snapshot ' + snap)
    ds = yt.load(snap)

    track = Table.read(trackfile, format='ascii')
    track.sort('col1')

    # Get the refined box in physical units
    zsnap = ds.get_parameter('CosmologyCurrentRedshift')
    proper_box_size = get_proper_box_size(ds)
    refine_box, refine_box_center, refine_width_code = grb.get_refine_box(ds, zsnap, track)
    refine_width = refine_width_code * proper_box_size
    refine_width_kpc = YTArray([refine_width], 'kpc')

    # Get halo center
    if (use_halo_c_v):
        halo_c_v = Table.read(halo_c_v_name, format='ascii')
        halo_ind = np.where(halo_c_v['col3']==snap[-6:])[0][0]
        halo_center_kpc = ds.arr([float(halo_c_v['col4'][halo_ind]), \
                                  float(halo_c_v['col5'][halo_ind]), \
                                  float(halo_c_v['col6'][halo_ind])], 'kpc')
        halo_velocity_kms = ds.arr([float(halo_c_v['col7'][halo_ind]), \
                                    float(halo_c_v['col8'][halo_ind]), \
                                    float(halo_c_v['col9'][halo_ind])], 'km/s')
        ds.halo_center_kpc = halo_center_kpc
        ds.halo_velocity_kms = halo_velocity_kms
    else:
        halo_center, halo_velocity = get_halo_center(ds, refine_box_center)
        # Define the halo center in kpc and the halo velocity in km/s
        halo_center_kpc = ds.arr(np.array(halo_center)*proper_box_size, 'kpc')
        sphere_region = ds.sphere(halo_center_kpc, (10., 'kpc') )
        bulk_velocity = sphere_region.quantities['BulkVelocity']().in_units('km/s')
        ds.halo_center_kpc = halo_center_kpc
        ds.halo_velocity_kms = bulk_velocity

    ds.track = track

    ds.add_field(('gas','vx_corrected'), function=vx_corrected, units='km/s', take_log=False, \
                 sampling_type='cell')
    ds.add_field(('gas', 'vy_corrected'), function=vy_corrected, units='km/s', take_log=False, \
                 sampling_type='cell')
    ds.add_field(('gas', 'vz_corrected'), function=vz_corrected, units='km/s', take_log=False, \
                 sampling_type='cell')
    ds.add_field(('gas', 'vel_mag_corrected'), function=vel_mag_corrected, units='km/s', take_log=False, \
                 sampling_type='cell')
    ds.add_field(('gas', 'radius_corrected'), function=radius_corrected, units='kpc', \
                 take_log=False, force_override=True, sampling_type='cell')
    ds.add_field(('gas', 'theta_pos'), function=theta_pos, units=None, take_log=False, \
                 sampling_type='cell')
    ds.add_field(('gas', 'phi_pos'), function=phi_pos, units=None, take_log=False, \
                 sampling_type='cell')
    ds.add_field(('gas', 'radial_velocity_corrected'), function=radial_velocity_corrected, \
                 units='km/s', take_log=False, force_override=True, sampling_type='cell')
    ds.add_field(('gas', 'theta_velocity_corrected'), function=theta_velocity_corrected, \
                 units='km/s', take_log=False, force_override=True, sampling_type='cell')
    ds.add_field(('gas', 'phi_velocity_corrected'), function=phi_velocity_corrected, \
                 units='km/s', take_log=False, force_override=True, sampling_type='cell')
    ds.add_field(('gas', 'tangential_velocity_corrected'), function=tangential_velocity_corrected, \
                 units='km/s', take_log=False, force_override=True, sampling_type='cell')
    ds.add_field(('gas', 'kinetic_energy_corrected'), function=kinetic_energy_corrected, \
                 units='erg', take_log=True, force_override=True, sampling_type='cell')

    # filter particles into star and dm
    # JT moved this to before "disk_relative" so that the if statement can use the filtered particle fields
    if (do_filter_particles): filter_particles(refine_box, filter_particle_types = ['young_stars', 'old_stars', 'young_stars7', 'young_stars8', 'stars', 'dm'])

    ds.add_field(('stars', 'radius_corrected'), function=radius_corrected_stars, units='kpc', \
                 take_log=False, force_override=True, sampling_type='particle')
    ds.add_field(('dm', 'radius_corrected'), function=radius_corrected_dm, units='kpc', \
                 take_log=False, force_override=True, sampling_type='particle')

    # Option to define velocities and coordinates relative to the angular momentum vector of the disk
    if (disk_relative):
        # Calculate angular momentum vector using sphere centered on halo center
        sphere = ds.sphere(ds.halo_center_kpc, (15., 'kpc'))
        print('using particle type ', particle_type_for_angmom, ' to derive angular momentum')
        L = sphere.quantities.angular_momentum_vector(use_gas=False, use_particles=True, particle_type=particle_type_for_angmom)
        print('found angular momentum vector')
        norm_L = L / np.sqrt((L**2).sum())
        # Define other unit vectors orthagonal to the angular momentum vector
        np.random.seed(99)
        x = np.random.randn(3)            # take a random vector
        x -= x.dot(norm_L) * norm_L       # make it orthogonal to L
        x /= np.linalg.norm(x)            # normalize it
        y = np.cross(norm_L, x)           # cross product with L
        x_vec = ds.arr(x)
        y_vec = ds.arr(y)
        L_vec = ds.arr(norm_L)
        ds.x_unit_disk = x_vec
        ds.y_unit_disk = y_vec
        ds.z_unit_disk = L_vec
        # Calculate the rotation matrix for converting from original coordinate system
        # into this new basis
        xhat = np.array([1,0,0])
        yhat = np.array([0,1,0])
        zhat = np.array([0,0,1])
        transArr0 = np.array([[xhat.dot(ds.x_unit_disk), xhat.dot(ds.y_unit_disk), xhat.dot(ds.z_unit_disk)],
                             [yhat.dot(ds.x_unit_disk), yhat.dot(ds.y_unit_disk), yhat.dot(ds.z_unit_disk)],
                             [zhat.dot(ds.x_unit_disk), zhat.dot(ds.y_unit_disk), zhat.dot(ds.z_unit_disk)]])
        rotationArr = np.linalg.inv(transArr0)
        ds.disk_rot_arr = rotationArr

        # Add the new fields
        ds.add_field(('gas', 'x_disk'), function=x_diskrel, units='kpc', take_log=False, \
                     force_override=True, sampling_type='cell')
        ds.add_field(('gas', 'y_disk'), function=y_diskrel, units='kpc', take_log=False, \
                     force_override=True, sampling_type='cell')
        ds.add_field(('gas', 'z_disk'), function=z_diskrel, units='kpc', take_log=False, \
                     force_override=True, sampling_type='cell')
        ds.add_field(('gas', 'vx_disk'), function=vx_diskrel, units='km/s', take_log=False, \
                     force_override=True, sampling_type='cell')
        ds.add_field(('gas', 'vy_disk'), function=vy_diskrel, units='km/s', take_log=False, \
                     force_override=True, sampling_type='cell')
        ds.add_field(('gas', 'vz_disk'), function=vz_diskrel, units='km/s', take_log=False, \
                     force_override=True, sampling_type='cell')
        ds.add_field(('gas', 'phi_pos_disk'), function=phi_pos_diskrel, units=None, take_log=False, \
                     force_override=True, sampling_type='cell')
        ds.add_field(('gas', 'theta_pos_disk'), function=theta_pos_diskrel, units=None, take_log=False, \
                     force_override=True, sampling_type='cell')
        ds.add_field(('gas', 'vphi_disk'), function=phi_velocity_diskrel, units='km/s', take_log=False, \
                     force_override=True, sampling_type='cell')
        ds.add_field(('gas', 'vtheta_disk'), function=theta_velocity_diskrel, units='km/s', take_log=False, \
                     force_override=True, sampling_type='cell')
        ds.add_field(('gas', 'vtan_disk'), function=tangential_velocity_diskrel, units='km/s', take_log=False, \
                     force_override=True, sampling_type='cell')

    return ds, refine_box, refine_box_center, refine_width
