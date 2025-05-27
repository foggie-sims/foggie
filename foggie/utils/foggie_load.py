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
    default_halo_c_v_name = catalog_dir + 'halo_c_v'
    halo_c_v_name = kwargs.get('halo_c_v_name', default_halo_c_v_name) # added by Ayan, so that halo_c_v_name can be flexibly passed on to load_sim()

    ds, region = foggie_load(snap_name, trackname, find_halo_center=find_halo_center, halo_c_v_name=halo_c_v_name, disk_relative=disk_relative, \
                            particle_type_for_angmom=particle_type_for_angmom, do_filter_particles=do_filter_particles, \
                            region=region)

    return ds, region

def foggie_load(snap, trackfile, **kwargs):
    """This function loads a specified snapshot named by 'snap', the halo track "trackfile'
    Based off of a helper function to flux_tracking written by Cassi, adapted for utils by JT."""
    find_halo_center = kwargs.get('find_halo_center', True)
    halo_c_v_name = kwargs.get('halo_c_v_name', 'none')
    disk_relative = kwargs.get('disk_relative', False)
    particle_type_for_angmom = kwargs.get('particle_type_for_angmom', 'young_stars')
    do_filter_particles = kwargs.get('do_filter_particles', True)
    region = kwargs.get('region', 'refine_box')
    gravity = kwargs.get('gravity', False)
    masses_dir = kwargs.get('masses_dir', '')
    smooth_AM_name = kwargs.get('smooth_AM_name', False)

    print ('Opening snapshot ' + snap)
    ds = yt.load(snap)
    ad = ds.all_data()

    track = Table.read(trackfile, format='ascii') #read the track file, which must be included as an argument
    track.sort('col1')

    zsnap = ds.get_parameter('CosmologyCurrentRedshift') # Get the refined box in physical units from the track file 
    proper_box_size = get_proper_box_size(ds)
    refine_box, refine_box_center, refine_width_code = grb.get_refine_box(ds, zsnap, track)
    refine_width = refine_width_code * proper_box_size

    center_style = "calculate" # this will be the default if no halo_c_v file is found

    does_the_halo_c_v_exist = os.path.exists(halo_c_v_name)
    print('Does the halo_c_v catalog exist? ', does_the_halo_c_v_exist)
    if (does_the_halo_c_v_exist): 
        halo_c_v = Table.read(halo_c_v_name, format='ascii', header_start=0, delimiter='|')
        if ('smooth' in halo_c_v_name): 
            center_style = 'smoothed'
            is_the_snapshot_in_halo_c_v = snap[-6:] in halo_c_v['snap']
        if ('halo_c_v' in halo_c_v_name): 
            center_style = 'catalog'
            is_the_snapshot_in_halo_c_v = snap[-6:] in halo_c_v['name']
        if ('root' in halo_c_v_name): 
            center_style = 'root_index'
            is_the_snapshot_in_halo_c_v = True 

    print('Is the snapshot in the halo_c_v catalog? ', is_the_snapshot_in_halo_c_v)

    print('Center style will be:  ', center_style) 
    print('Halo_c_v name is: ', halo_c_v_name)


    if (('smoothed' in center_style) & os.path.exists(halo_c_v_name)):
        halo_c_v = Table.read(halo_c_v_name, format='ascii', header_start=0, delimiter='|')
        if (snap[-6:] in halo_c_v['snap']):
            print('Pulling halo center from smoothed catalog file :', halo_c_v_name)
            halo_ind = np.where(halo_c_v['snap']==snap[-6:])[0][0]
            halo_center_kpc = ds.arr([float(halo_c_v[halo_ind]['xc']), \
                                      float(halo_c_v[halo_ind]['yc']), \
                                      float(halo_c_v[halo_ind]['zc'])], 'kpc')
            ds.halo_center_kpc = halo_center_kpc
            ds.halo_center_code = halo_center_kpc.in_units('code_length')
            sp = ds.sphere(ds.halo_center_kpc, (3., 'kpc'))
            bulk_vel = sp.quantities.bulk_velocity(use_gas=False,use_particles=True,particle_type='all').to('km/s')
            ds.halo_velocity_kms = bulk_vel
        else: 
            print("Your snapshot was not found in the halo_c_v catalog file: ", halo_c_v_name)

    elif (('catalog' in center_style) & os.path.exists(halo_c_v_name)):
        print('Using halo_c_v catalog file: ', halo_c_v_name, ' for center style ', center_style)
        halo_c_v = Table.read(halo_c_v_name, format='ascii', header_start=0, delimiter='|')
        if (snap[-6:] in halo_c_v['name']):
            print("Pulling halo center from catalog file") 
            halo_ind = np.where(halo_c_v['name']==snap[-6:])[0][0]
            halo_center_kpc = ds.arr([float(halo_c_v[halo_ind]['x_c']), \
                                      float(halo_c_v[halo_ind]['y_c']), \
                                      float(halo_c_v[halo_ind]['z_c'])], 'kpc')
            halo_velocity_kms = ds.arr([float(halo_c_v[halo_ind]['v_x']), \
                                        float(halo_c_v[halo_ind]['v_y']), \
                                        float(halo_c_v[halo_ind]['v_z'])], 'km/s')
            ds.halo_center_kpc = halo_center_kpc
            ds.halo_center_code = halo_center_kpc.in_units('code_length')
            ds.halo_velocity_kms = halo_velocity_kms
            print('halo center in kpc: ', ds.halo_center_kpc)
            print('halo velocity in km/s: ', ds.halo_velocity_kms)
        else: 
            print("Your snapshot was not found in the halo_c_v catalog file: ", halo_c_v_name)
    
    elif ('root_index' in center_style): 

        print('Using root catalog file: ', halo_c_v_name, ' for center style ', center_style)
        root_particles = Table.read(halo_c_v_name, format='ascii', header_start=0, delimiter='|')
        halo0 = root_particles['root_index']
        
        x_particles = ad['particle_position_x']
        y_particles = ad['particle_position_y']
        z_particles = ad['particle_position_z']
        
        root_indices = halo0
        now_indices = ad['particle_index']
        indices = np.where(np.isin(now_indices, root_indices))[0]
        
        center_x  = float(np.mean(x_particles[indices].in_units('code_length'))) 
        center_y  = float(np.mean(y_particles[indices].in_units('code_length'))) 
        center_z  = float(np.mean(z_particles[indices].in_units('code_length'))) 
        
        halo_center = [center_x, center_y, center_z]
        
        halo_center_kpc = ds.arr(np.array(halo_center)*proper_box_size, 'kpc')
        sp = ds.sphere(halo_center_kpc, (3., 'kpc'))
        bulk_vel = sp.quantities.bulk_velocity(use_gas=False,use_particles=True,particle_type='all').to('km/s')
        ds.halo_center_code = ds.arr(np.array(halo_center), 'code_length')
        ds.halo_center_kpc = halo_center_kpc
        ds.halo_velocity_kms = bulk_vel
        print(ds.halo_velocity_kms)


    ds.track = track
    ds.refine_box_center = refine_box_center
    ds.refine_width = refine_width
    ds.omega_baryon = 0.0461
    # Note that if you want to use the ('gas', 'baryon_overdensity') field, you must include this line after you've defined some data object from ds:
    # > obj.set_field_parameter('omega_baryon', ds.omega_baryon)
    # foggie_load returns a 'region' data object given by the 'region' keyword, and the 'omega_baryon' parameter is already set for that.

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
    ds.add_field(('index', 'cell_id'), function=get_cell_ids, sampling_type='cell')
    
    # filter particles into star and dm
    # JT moved this to before "disk_relative" so that the if statement can use the filtered particle fields
    if (do_filter_particles):
        filter_particles(refine_box, filter_particle_types = ['young_stars', 'young_stars3', 'young_stars8', 'old_stars', 'stars', 'dm'])

        ds.add_field(('stars', 'radius_corrected'), function=radius_corrected_stars, units='kpc', \
                     take_log=False, force_override=True, sampling_type='particle')
        ds.add_field(('young_stars', 'radius_corrected'), function=radius_corrected_young_stars, units='kpc', \
                     take_log=False, force_override=True, sampling_type='particle')
        ds.add_field(('young_stars3', 'radius_corrected'), function=radius_corrected_young_stars8, units='kpc', \
                     take_log=False, force_override=True, sampling_type='particle')
        ds.add_field(('young_stars8', 'radius_corrected'), function=radius_corrected_young_stars8, units='kpc', \
                     take_log=False, force_override=True, sampling_type='particle')
        ds.add_field(('old_stars', 'radius_corrected'), function=radius_corrected_old_stars, units='kpc', \
                     take_log=False, force_override=True, sampling_type='particle')
        ds.add_field(('dm', 'radius_corrected'), function=radius_corrected_dm, units='kpc', \
                     take_log=False, force_override=True, sampling_type='particle')
        ds.add_field(('dm', 'radial_velocity_corrected'), function=radial_velocity_corrected_dm, units='km/s', \
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
        if (smooth_AM_name):
            smooth_am = Table.read(smooth_AM_name, format='ascii')
            ind = np.where(smooth_am['col2']==snap[-6:])[0][0]
            L = np.array([float(smooth_am['col5'][ind]), float(smooth_am['col6'][ind]), float(smooth_am['col7'][ind])])
        else:
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
        ds.add_field(('stars', 'x_disk'), function=x_diskrel_stars, units='kpc', take_log=False, \
                     force_override=True, sampling_type='particle')
        ds.add_field(('stars', 'y_disk'), function=y_diskrel_stars, units='kpc', take_log=False, \
                     force_override=True, sampling_type='particle')
        ds.add_field(('stars', 'z_disk'), function=z_diskrel_stars, units='kpc', take_log=False, \
                     force_override=True, sampling_type='particle')
        ds.add_field(('young_stars8', 'x_disk'), function=x_diskrel_young_stars8, units='kpc', take_log=False, \
                     force_override=True, sampling_type='particle')
        ds.add_field(('young_stars8', 'y_disk'), function=y_diskrel_young_stars8, units='kpc', take_log=False, \
                     force_override=True, sampling_type='particle')
        ds.add_field(('young_stars8', 'z_disk'), function=z_diskrel_young_stars8, units='kpc', take_log=False, \
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
        if (do_filter_particles):
            ds.add_field(('dm', 'vff'), function=v_ff_dm, units='km/s', \
                     take_log=False, force_override=True, sampling_type='particle')

    if (region=='refine_box'):
        region = refine_box
    elif (region=='cgm'):
        cen_sphere = ds.sphere(ds.halo_center_kpc, (cgm_inner_radius, "kpc"))
        rvir_sphere = ds.sphere(ds.halo_center_kpc, (cgm_outer_radius, 'kpc'))
        cgm = rvir_sphere - cen_sphere
        cgm_filtered = cgm.cut_region(cgm_field_filter)
        region = cgm_filtered

    region.set_field_parameter('omega_baryon', ds.omega_baryon)

    return ds, region
