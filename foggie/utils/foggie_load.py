import numpy as np
import yt
from yt.units import *
from yt import YTArray
from astropy.table import Table
import os
from scipy.interpolate import InterpolatedUnivariateSpline as IUS

from foggie.utils.consistency import *
from foggie.utils.get_halo_center import get_halo_center
from foggie.utils.get_proper_box_size import get_proper_box_size
from foggie.utils.get_run_loc_etc import get_run_loc_etc
from foggie.utils.yt_fields import *
from foggie.utils.foggie_utils import filter_particles
import foggie.utils as futils
import foggie.utils.get_refine_box as grb

def load_sim(args, **kwargs):
    '''Loads the specified simulation dataset, where the required arguments are:
    system -- what system are you running on?
    halo -- which halo do you want?
    run -- which run of that halo do you want?
    output -- which output?
    Available optional arguments are:
    use_halo_c_v -- do you want to use the halo_c_v catalog file? Default is yes
    find_halo_center -- do you want to calculate the halo center at all? Default is yes
    disk_relative -- do you want to calculate the angular momentum of the disk and define disk-relative positions? Default is no
    particle_type_for_angmom -- what particles to use for angular momentum? Default is young_stars, only need this if disk_relative=True
    filter_partiles -- do you want to filter particles? Default is yes
    region -- what region do you want to return?
    '''
    use_halo_c_v = kwargs.get('use_halo_c_v', True)
    disk_relative = kwargs.get('disk_relative', False)
    particle_type_for_angmom = kwargs.get('particle_type_for_angmom', 'young_stars')
    do_filter_particles = kwargs.get('do_filter_particles', True)
    find_halo_center = kwargs.get('find_halo_center', True)
    region = kwargs.get('region', 'refine_box')

    foggie_dir, output_dir, run_loc, code_path, trackname, haloname, spectra_dir, infofile = get_run_loc_etc(args)
    snap_name = foggie_dir + run_loc + args.output + '/' + args.output
    catalog_dir = code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/'
    halo_c_v_name = catalog_dir + 'halo_c_v'
    ds, region = foggie_load(snap_name, trackname, find_halo_center=find_halo_center, halo_c_v_name=halo_c_v_name, disk_relative=disk_relative, \
                            particle_type_for_angmom=particle_type_for_angmom, do_filter_particles=do_filter_particles, \
                            region=region)

    return ds, region

def foggie_load(snap, trackfile, **kwargs):
    """This function loads a specified snapshot named by 'snap', the halo track "trackfile'
    Based off of a helper function to flux_tracking written by Cassi, adapted for utils by JT."""
    find_halo_center = kwargs.get('find_halo_center', True)
    halo_c_v_name = kwargs.get('halo_c_v_name', 'halo_c_v')
    disk_relative = kwargs.get('disk_relative', False)
    particle_type_for_angmom = kwargs.get('particle_type_for_angmom', 'young_stars')
    do_filter_particles = kwargs.get('do_filter_particles', True)
    region = kwargs.get('region', 'refine_box')
    gravity = kwargs.get('gravity', False)
    masses_dir = kwargs.get('masses_dir', '')
    correct_bulk_velocity = kwargs.get('correct_bulk_velocity', False)

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
    if (find_halo_center):
        if (os.path.exists(halo_c_v_name)):
            halo_c_v = Table.read(halo_c_v_name, format='ascii')
            if ('smoothed' in halo_c_v_name):
                if (snap[-6:] in halo_c_v['col2']):
                    print('Pulling halo center from smoothed path catalog file')
                    halo_ind = np.where(halo_c_v['col2']==snap[-6:])[0][0]
                    halo_center_kpc = ds.arr([float(halo_c_v['col5'][halo_ind]), \
                                              float(halo_c_v['col6'][halo_ind]), \
                                              float(halo_c_v['col7'][halo_ind])], 'kpc')
                    ds.halo_center_kpc = halo_center_kpc
                    ds.halo_center_code = halo_center_kpc.in_units('code_length')
                    sp = ds.sphere(ds.halo_center_kpc, (5., 'kpc'))
                    bulk_vel = sp.quantities.bulk_velocity(use_gas=False,use_particles=True,particle_type='all').to('km/s')
                    ds.halo_velocity_kms = bulk_vel
                    calc_hc = False
                else:
                    print('This snapshot is not in the halo_cen_smoothed file, calculating halo center (which will NOT be smoothed)...')
                    calc_hc = True
            else:
                if (snap[-6:] in halo_c_v['col3']):
                    print("Pulling halo center from catalog file")
                    halo_ind = np.where(halo_c_v['col3']==snap[-6:])[0][0]
                    halo_center_kpc = ds.arr([float(halo_c_v['col4'][halo_ind]), \
                                              float(halo_c_v['col5'][halo_ind]), \
                                              float(halo_c_v['col6'][halo_ind])], 'kpc')
                    halo_velocity_kms = ds.arr([float(halo_c_v['col7'][halo_ind]), \
                                                float(halo_c_v['col8'][halo_ind]), \
                                                float(halo_c_v['col9'][halo_ind])], 'km/s')
                    ds.halo_center_kpc = halo_center_kpc
                    ds.halo_center_code = halo_center_kpc.in_units('code_length')
                    ds.halo_velocity_kms = halo_velocity_kms
                    calc_hc = False
                else:
                    print('This snapshot is not in the halo_c_v file, calculating halo center...')
                    calc_hc = True
        else:
            print("This halo_c_v file doesn't exist, calculating halo center...")
            calc_hc = True
        if (calc_hc):
            halo_center, halo_velocity = get_halo_center(ds, refine_box_center)
            # Define the halo center in kpc and the halo velocity in km/s
            halo_center_kpc = ds.arr(np.array(halo_center)*proper_box_size, 'kpc')
            sphere_region = ds.sphere(halo_center_kpc, (10., 'kpc') )
            bulk_velocity = sphere_region.quantities['BulkVelocity']().in_units('km/s')
            ds.halo_center_code = halo_center
            ds.halo_center_kpc = halo_center_kpc
            ds.halo_velocity_kms = bulk_velocity
        if (correct_bulk_velocity):
            sp = ds.sphere(ds.halo_center_kpc, (3., 'kpc'))
            bulk_vel = sp.quantities.bulk_velocity(use_gas=False,use_particles=True,particle_type='all').to('km/s')
            ds.halo_velocity_kms = bulk_vel
    else:
        print("Not finding halo center")
        ds.halo_center_kpc = ds.arr([np.nan, np.nan, np.nan], 'kpc')
        ds.halo_center_code = ds.arr([np.nan, np.nan, np.nan], 'code_length')
        ds.halo_velocity_kms = ds.arr([np.nan, np.nan, np.nan], 'km/s')

    ds.track = track
    ds.refine_box_center = refine_box_center
    ds.refine_width = refine_width

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
                 units='km/s', take_log=False, force_override=True, sampling_type='cell', display_name='Radial Velocity')
    ds.add_field(('gas', 'theta_velocity_corrected'), function=theta_velocity_corrected, \
                 units='km/s', take_log=False, force_override=True, sampling_type='cell')
    ds.add_field(('gas', 'phi_velocity_corrected'), function=phi_velocity_corrected, \
                 units='km/s', take_log=False, force_override=True, sampling_type='cell')
    ds.add_field(('gas', 'tangential_velocity_corrected'), function=tangential_velocity_corrected, \
                 units='km/s', take_log=False, force_override=True, sampling_type='cell', display_name='Tangential Velocity')
    ds.add_field(('gas', 'kinetic_energy_corrected'), function=kinetic_energy_corrected, \
                 units='erg', take_log=True, force_override=True, sampling_type='cell')
    ds.add_field(('gas', 'radial_kinetic_energy'), function=radial_kinetic_energy, \
                 units='erg', take_log=True, force_override=True, sampling_type='cell')
    ds.add_field(('gas', 'tangential_kinetic_energy'), function=tangential_kinetic_energy, \
                 units='erg', take_log=True, force_override=True, sampling_type='cell')
    ds.add_field(('gas', 'cell_mass_msun'), function=cell_mass_msun, units='Msun', take_log=True, \
                 force_override=True, sampling_type='cell')

    # filter particles into star and dm
    # JT moved this to before "disk_relative" so that the if statement can use the filtered particle fields
    if (do_filter_particles):
        filter_particles(refine_box, filter_particle_types = ['young_stars', 'old_stars', 'stars', 'dm'])

        ds.add_field(('stars', 'radius_corrected'), function=radius_corrected_stars, units='kpc', \
                     take_log=False, force_override=True, sampling_type='particle')
        ds.add_field(('young_stars', 'radius_corrected'), function=radius_corrected_young_stars, units='kpc', \
                     take_log=False, force_override=True, sampling_type='particle')
        ds.add_field(('old_stars', 'radius_corrected'), function=radius_corrected_old_stars, units='kpc', \
                     take_log=False, force_override=True, sampling_type='particle')
        ds.add_field(('dm', 'radius_corrected'), function=radius_corrected_dm, units='kpc', \
                     take_log=False, force_override=True, sampling_type='particle')


        sam_un = ds.unit_system["specific_angular_momentum"]
        am_un  = ds.unit_system["angular_momentum"]
        for ptype in ['stars', 'young_stars', 'old_stars', 'dm']:
            ds.add_field((ptype, "particle_relative_specific_angular_momentum"), sampling_type="particle",
                         function=get_particle_relative_specific_angular_momentum(ptype), units=sam_un)

            ds.add_field((ptype, "particle_relative_specific_angular_momentum_x"),sampling_type="particle",
                        function= get_particle_relative_specific_angular_momentum_x(ptype), units=sam_un)
            ds.add_field((ptype, "particle_relative_specific_angular_momentum_y"),sampling_type="particle",
                        function= get_particle_relative_specific_angular_momentum_y(ptype), units=sam_un)
            ds.add_field((ptype, "particle_relative_specific_angular_momentum_z"),sampling_type="particle",
                        function= get_particle_relative_specific_angular_momentum_z(ptype), units=sam_un)

            ds.add_field((ptype, "particle_relative_angular_momentum_x"),sampling_type="particle",
                        function= get_particle_relative_angular_momentum_x(ptype), units=am_un)
            ds.add_field((ptype, "particle_relative_angular_momentum_y"),sampling_type="particle",
                        function= get_particle_relative_angular_momentum_y(ptype), units=am_un)
            ds.add_field((ptype, "particle_relative_angular_momentum_z"),sampling_type="particle",
                        function= get_particle_relative_angular_momentum_z(ptype), units=am_un)

            '''
            ds.add_field((ptype, "particle_relative_specific_angular_momentum"),
              sampling_type="particle",
              function=_particle_relative_specific_angular_momentum,
              units=ds.unit_system["specific_angular_momentum"]
              )


            for axi, ax in enumerate("xyz"):
                f, v = _get_spec_ang_mom_comp(axi, ax, ptype)

            ds.add_field(
                (ptype, f"particle_relative_angular_momentum_{ax}"),
                sampling_type="particle",
                function=v,
                units=ds.unit_system["angular_momentum"]
            )
            '''


            '''

            def _particle_relative_specific_angular_momentum(field, data):
                """Calculate the angular of a particle velocity.

                Returns a vector for each particle.
                """
                pos = data.ds.arr([data[ptype, f"relative_particle_position_%s" % ax] for ax in "xyz"]).T
                vel = data.ds.arr([data[ptype, f"relative_particle_velocity_%s" % ax] for ax in "xyz"]).T
                return ucross(pos, vel, registry=data.ds.unit_registry)

            def _get_spec_ang_mom_comp(axi, ax, _ptype):
                def _particle_specific_angular_momentum_component(field, data):
                    return data[_ptype, "particle_relative_specific_angular_momentum"][:, axi]

                def _particle_angular_momentum_component(field, data):
                    return (
                        data[_ptype, "particle_mass"]
                        * data[ptype, f"particle_relative_specific_angular_momentum_{ax}"]
                    )

                return (
                    _particle_specific_angular_momentum_component,
                    _particle_angular_momentum_component,
                )
            '''



    # Option to define velocities and coordinates relative to the angular momentum vector of the disk
    if (disk_relative):
        # Calculate angular momentum vector using sphere centered on halo center
        sphere = ds.sphere(ds.halo_center_kpc, (15., 'kpc'))
        print('using particle type ', particle_type_for_angmom, ' to derive angular momentum')
        if (particle_type_for_angmom=='gas'):
            sphere = sphere.include_below(('gas','temperature'), 1e4)
            sphere.set_field_parameter('bulk_velocity', ds.halo_velocity_kms)
            L = sphere.quantities.angular_momentum_vector(use_gas=True, use_particles=False)
        else:
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
        ds.add_field(('dm', 'x_disk'), function=x_diskrel_dm, units='kpc', take_log=False, \
                     force_override=True, sampling_type='particle')
        ds.add_field(('dm', 'y_disk'), function=y_diskrel_dm, units='kpc', take_log=False, \
                     force_override=True, sampling_type='particle')
        ds.add_field(('dm', 'z_disk'), function=z_diskrel_dm, units='kpc', take_log=False, \
                     force_override=True, sampling_type='particle')
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
        ds.add_field(('dm', 'phi_pos_disk'), function=phi_pos_diskrel_dm, units=None, take_log=False, \
                     force_override=True, sampling_type='particle')
        ds.add_field(('dm', 'theta_pos_disk'), function=theta_pos_diskrel_dm, units=None, take_log=False, \
                     force_override=True, sampling_type='particle')
        ds.add_field(('gas', 'vphi_disk'), function=phi_velocity_diskrel, units='km/s', take_log=False, \
                     force_override=True, sampling_type='cell')
        ds.add_field(('gas', 'vtheta_disk'), function=theta_velocity_diskrel, units='km/s', take_log=False, \
                     force_override=True, sampling_type='cell')
        ds.add_field(('gas', 'vtan_disk'), function=tangential_velocity_diskrel, units='km/s', take_log=False, \
                     force_override=True, sampling_type='cell')
        ds.add_field(('gas', 'tangential_kinetic_energy_disk'), function=tangential_kinetic_energy_diskrel, \
                     units='erg', take_log=True, force_override=True, sampling_type='cell')

    if (gravity):
        # Interpolate enclosed mass function to get tff
        if (zsnap > 2.):
            masses = Table.read(masses_dir + 'masses_z-gtr-2.hdf5', path='all_data')
        else:
            masses = Table.read(masses_dir + 'masses_z-less-2.hdf5', path='all_data')
        snap_ind = masses['snapshot']==snap[-6:]
        ds.Menc_profile = IUS(np.concatenate(([0],masses['radius'][snap_ind])), np.concatenate(([0],masses['total_mass'][snap_ind])))
        ds.add_field(('gas', 'tff'), function=t_ff, units='yr', display_name='Free fall time', take_log=True, \
                    force_override=True, sampling_type='cell')
        ds.add_field(('gas', 'vff'), function=v_ff, units='km/s', display_name='Free fall velocity', take_log=False, \
                    force_override=True, sampling_type='cell')
        ds.add_field(('gas', 'vesc'), function=v_esc, units='km/s', display_name='Escape velocity', take_log=False, \
                    force_override=True, sampling_type='cell')
        ds.add_field(('gas', 'tcool_tff'), function=tcool_tff_ratio, units=None, display_name='t_{cool}/t_{ff}', take_log=True, \
                    force_override=True, sampling_type='cell')
        ds.add_field(('gas','grav_pot'), function=grav_pot, units='cm**2/s**2', force_override=True, sampling_type='cell', \
                    display_name = 'Gravitational Potential')
        grad_fields_grav = ds.add_gradient_fields(('gas','grav_pot'))
        ds.add_field(('gas','HSE'), function=hse_ratio, units='', \
                     display_name='HSE Parameter', force_override=True, sampling_type='cell')

    if (region=='refine_box'):
        region = refine_box
    elif (region=='cgm'):
        cen_sphere = ds.sphere(ds.halo_center_kpc, (cgm_inner_radius, "kpc"))
        rvir_sphere = ds.sphere(ds.halo_center_kpc, (cgm_outer_radius, 'kpc'))
        cgm = rvir_sphere - cen_sphere
        cgm_filtered = cgm.cut_region(cgm_field_filter)
        region = cgm_filtered

    return ds, region
