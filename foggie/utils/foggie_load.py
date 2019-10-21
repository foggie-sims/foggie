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
        halo_center_kpc = YTArray([float(halo_c_v['col4'][halo_ind]), \
                                  float(halo_c_v['col5'][halo_ind]), \
                                  float(halo_c_v['col6'][halo_ind])], 'kpc')
        halo_velocity_kms = YTArray([float(halo_c_v['col7'][halo_ind]), \
                                    float(halo_c_v['col8'][halo_ind]), \
                                    float(halo_c_v['col9'][halo_ind])], 'km/s')
    else:
        halo_center, halo_velocity = get_halo_center(ds, refine_box_center)
        # Define the halo center in kpc and the halo velocity in km/s
        halo_center_kpc = YTArray(np.array(halo_center)*proper_box_size, 'kpc')
        halo_velocity_kms = YTArray(halo_velocity).in_units('km/s')

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
    ds.add_field(('gas', 'kinetic_energy_corrected'), function=kinetic_energy_corrected, \
                 units='erg', take_log=True, force_override=True, sampling_type='cell')

    # filter particles into star and dm
    filter_particles(refine_box)

    return ds, refine_box, refine_box_center, refine_width
