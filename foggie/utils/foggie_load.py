import yt, numpy as np, os
from yt.units import *
from yt import YTArray
from astropy.table import Table
from scipy.interpolate import InterpolatedUnivariateSpline as IUS

from foggie.utils.consistency import *
from foggie.utils.get_halo_center import get_halo_center
from foggie.utils.get_proper_box_size import get_proper_box_size
from foggie.utils.get_run_loc_etc import get_run_loc_etc
from foggie.utils.yt_fields import *
from foggie.utils.foggie_utils import filter_particles
import foggie.utils.get_refine_box as grb
from datetime import datetime

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

    ds, region = foggie_load(snap_name, trackfile_name=trackname, find_halo_center=find_halo_center, halo_c_v_name=halo_c_v_name, disk_relative=disk_relative, \
                            particle_type_for_angmom=particle_type_for_angmom, do_filter_particles=do_filter_particles, \
                            region=region)

    return ds, region

def get_center_from_catalog(ds, halo_c_v_name, snap, trackfile_name):
    """This function is a helper function to get the halo center from the halo_c_v catalog file.
    It is used in foggie_load() to determine the halo center."""
    print('GET_CENTER_FROM_CATALOG: Using halo_c_v catalog file: ', halo_c_v_name)
    halo_c_v = Table.read(halo_c_v_name, format='ascii', header_start=0, delimiter='|')
    if (snap[-6:] in halo_c_v['name']):
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
        # If track file given, return the refine box as the region, else return full dataset
        if (trackfile_name != None):
            track = Table.read(trackfile_name, format='ascii') # read the track file
            track.sort('col1')
            print('GET_CENTER_FROM_CATALOG: Read track file:', trackfile_name)

            proper_box_size = get_proper_box_size(ds) # get the proper size of the computational domain (NOT the refine region)
            region, refine_box_center, refine_width_code = grb.get_refine_box(ds, ds.get_parameter('CosmologyCurrentRedshift') , track)
            ds.refine_width = refine_width_code * proper_box_size
        else:
            region = ds.all_data()
    else:
        print('GET_CENTER_FROM_CATALOG: That snapshot is not in the catalog! Attempting to calculate from track file...')
        if (trackfile_name != None):
            ds, region = get_center_from_calculated(ds, trackfile_name)
        else:
            raise ValueError('GET_CENTER_FROM_CATALOG: Snapshot is not in catalog and no trackfile_name given to calculate center! Exiting.')
    
    return ds, region

def get_center_from_DM_region(ds):
    """This function is a helper function to get the halo center when there is no track.
    The use case is for non-central halos without a track file (new ICs, etc.)"""

    ad = ds.all_data()
    ptype = ad['particle_type'] 
    x_dm = ad['all', 'particle_position_x'][ptype == 4]
    y_dm = ad['all', 'particle_position_y'][ptype == 4]
    z_dm = ad['all', 'particle_position_z'][ptype == 4]

    center_x = 0.5 * (x_dm.max() + x_dm.min())
    center_y = 0.5 * (y_dm.max() + y_dm.min())
    center_z = 0.5 * (z_dm.max() + z_dm.min())
    halo_center_kpc = [center_x.in_units('kpc'), center_y.in_units('kpc'), center_z.in_units('kpc')]

    ds.halo_center_kpc = halo_center_kpc
    ds.halo_center_code = ds.arr([center_x, center_y, center_z]) 
    ds.halo_velocity_kms = [np.nan, np.nan, np.nan] 
    
    subregion = ds.r[x_dm.min():x_dm.max(), y_dm.min():y_dm.max(), z_dm.min():z_dm.max()]
    return ds, subregion 

def get_center_from_smoothed_catalog(ds, halo_c_v_name, snap, trackfile_name):
    """This function is a helper function to get the halo center from the smoothed halo_c_v catalog file.
    It is used in foggie_load() to determine the halo center."""
    print('CENTER_FROM_SMOOTHED_CATALOG: Reading catalog file: ', halo_c_v_name)
    halo_c_v = Table.read(halo_c_v_name, format='ascii', header_start=0, delimiter='|')
    if (snap[-6:] in halo_c_v['snap']):
        halo_ind = np.where(halo_c_v['snap']==snap[-6:])[0][0]
        halo_center_kpc = ds.arr([float(halo_c_v[halo_ind]['xc']), \
                                  float(halo_c_v[halo_ind]['yc']), \
                                  float(halo_c_v[halo_ind]['zc'])], 'kpc')
        ds.halo_center_kpc = halo_center_kpc
        ds.halo_center_code = halo_center_kpc.in_units('code_length')
        sp = ds.sphere(ds.halo_center_kpc, (3., 'kpc'))
        bulk_vel = sp.quantities.bulk_velocity(use_gas=False,use_particles=True,particle_type='all').to('km/s')
        ds.halo_velocity_kms = bulk_vel
        if (trackfile_name != None):
            print('CENTER_FROM_SMOOTHED_CATALOG: You gave a track file but track does not align with smoothed catalogs, so returning full dataset as region')
        region = ds.all_data()
    else:
        raise ValueError('CENTER_FROM_SMOOTHED_CATALOG: That snapshot is not in the catalog! Exiting.')
    
    return ds, region

def get_center_from_root_catalog(ds, root_catalog_name):

    """This function is a helper function to get the halo center from the root catalog file.
    It is used in foggie_load() to determine the halo center."""
    print('CENTER_FROM_ROOT_CATALOG: Reading catalog file: ', root_catalog_name)
    root_particles = Table.read(root_catalog_name, format='ascii', header_start=0, delimiter='|')
    halo0 = root_particles['root_index']

    ad = ds.all_data() 
        
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
    
    proper_box_size = get_proper_box_size(ds) # get the proper size of the computational domain (NOT the refine region)
    halo_center_kpc = ds.arr(np.array(halo_center)*proper_box_size, 'kpc')
    sp = ds.sphere(halo_center_kpc, (3., 'kpc'))
    bulk_vel = sp.quantities.bulk_velocity(use_gas=False,use_particles=True,particle_type='all').to('km/s')
    ds.halo_center_code = ds.arr(np.array(halo_center), 'code_length')
    ds.halo_center_kpc = halo_center_kpc
    ds.halo_velocity_kms = bulk_vel

    return ds, ds.all_data()

def get_center_from_calculated(ds, trackfile_name):
    """This function is a helper function to calculate the halo center.
    It uses get_halo_center to set the center as the density peak of the dark matter
    in a 50 kpc sphere. The track-defined refine_box_center is used as the initial guess"""
    
    track = Table.read(trackfile_name, format='ascii') # read the track file
    track.sort('col1')
    print('CALCULATE_CENTER_FROM_TRACK: Read track file:', trackfile_name)

    proper_box_size = get_proper_box_size(ds) # get the proper size of the computational domain (NOT the refine region)
    refine_box, refine_box_center, refine_width_code = grb.get_refine_box(ds, ds.get_parameter('CosmologyCurrentRedshift') , track)
    ds.refine_width = refine_width_code * proper_box_size
    
    halo_center, _ = get_halo_center(ds, refine_box_center)
    # Define the halo center in kpc and the halo velocity in km/s
    halo_center_kpc = ds.arr(np.array(halo_center)*proper_box_size, 'kpc')
    sp = ds.sphere(halo_center_kpc, (3., 'kpc'))
    ds.halo_center_code = ds.arr(np.array(halo_center), 'code_length')
    ds.halo_center_kpc = halo_center_kpc
    ds.halo_velocity_kms = sp.quantities.bulk_velocity(use_gas=False,use_particles=True,particle_type='all').to('km/s')

    return ds, refine_box

def get_center_from_track(ds, trackfile_name):
    """This function is a helper function to obtain the halo center.
    It adopts the center from the track file without checking or modifying it.  
    The refine box center is by definition the center given in the track file. 
    The halo velocity is calculated as the bulk velocity of the entire track box."""
    
    track = Table.read(trackfile_name, format='ascii') # read the track file
    track.sort('col1')
    print('USE_TRACK_CENTER: Read track file:', trackfile_name)

    proper_box_size = get_proper_box_size(ds) # get the proper size of the computational domain (NOT the refine region)
    refine_box, refine_box_center, refine_width_code = grb.get_refine_box(ds, ds.get_parameter('CosmologyCurrentRedshift'), track)
    refine_width = refine_width_code * proper_box_size
    
    # Define the halo center in kpc and the halo velocity in km/s
    halo_center_kpc = ds.arr(np.array(refine_box_center)*proper_box_size, 'kpc')
    bulk_vel = refine_box.quantities.bulk_velocity(use_gas=False,use_particles=True,particle_type='all').to('km/s')
    ds.halo_center_code = ds.arr(np.array(halo_center_kpc/proper_box_size), 'code_length')
    ds.halo_center_kpc = halo_center_kpc
    ds.halo_velocity_kms = bulk_vel
    ds.refine_width = refine_width

    return ds, refine_box

def foggie_load(snap, **kwargs):
        
    """Load a foggie snapshot and attach useful derived fields.

    Parameters
    - snap (str): Path to the snapshot file to load (passed to `yt.load`).
    - kwargs: Optional keyword arguments (defaults shown below):
        - central_halo (bool, default False): Treat this halo as the central/main halo.
        - trackfile_name (str or None, default None): Path to a track file used to set/compute the refine-box center.
        - halo_c_v_name (str or None, default None): Path to a halo catalog (halo_c_v). If the filename contains
             'smoothed' a smoothed center is used.
        - do_filter_particles (bool, default True): Whether to run particle filtering and add particle fields.
        - disk_relative (bool, default False): If True, compute disk-aligned coordinates/velocities from angular momentum.
        - smooth_AM_name (str or False, default False): Path to a table with a precomputed smoothed angular-momentum vector
            (optional, only used when `disk_relative` is True).
        - particle_type_for_angmom (str, default 'young_stars'): Particle type used when computing angular momentum.
        - gravity (bool, default False): If True, load precomputed enclosed-mass profiles and add gravity-related fields
            (`tff`, `vff`, `vesc`, etc.). Requires `masses_dir`. Do not use unless using a halo_c_v file.
        - masses_dir (str, default ''): Directory containing `masses_*.hdf5` tables used when `gravity` is True.

    Returns
        - ds: A `yt` Dataset with added attributes and fields (e.g., `halo_center_kpc`, `halo_velocity_kms`, disk fields).
        - region: A `yt` data object defining the selected region (refine box or a sub-region determined by track/catalog/DM).
    Notes
        - The function sets `ds.halo_center_code`, `ds.halo_center_kpc`, and `ds.halo_velocity_kms` when a meaningful
            center/velocity is found. Derived fields (corrected positions/velocities, particle angular-momentum fields,
            disk-aligned fields, gravity fields) are only added when the required inputs are present.
        - Other internal keyword arguments (e.g., `region`) are honored where used in the function.
    """

    #get all the keywords and process them 
    central_halo = kwargs.get('central_halo', True) # if this is set, we assume we are working with the central halo 
    trackfile_name = kwargs.get('trackfile_name', None)
    halo_c_v_name = kwargs.get('halo_c_v_name', None)
    root_catalog_name = kwargs.get('root_catalog_name', None)
    do_filter_particles = kwargs.get('do_filter_particles', True)

    #open the snapshot and get the all_data object
    ds = yt.load(snap)
    ds.current_datetime = datetime.now() # add to the dataset the time that we opened it
    ds.snapname = snap[-6:]

    #set up some dataset properties that will be useful elsewhere, and nonsense defaults for others.
    zsnap = ds.get_parameter('CosmologyCurrentRedshift')
    ds.omega_baryon = ds.parameters['CosmologyOmegaMatterNow'] - ds.parameters['CosmologyOmegaDarkMatterNow']
    ds.halo_center_code = [np.nan, np.nan, np.nan] # this is the default halo_center_code until its overwritten by center-finding 
    ds.halo_velocity_kms = [np.nan, np.nan, np.nan] # this is the default halo_velocity_kms until its overwritten by center-finding 
    region = ds.r[0:1, 0:1, 0:1] # this is the default region 

    # now we start the logic for what to do about the halo center and refine box
    if (central_halo == True): # branch 1 of the flowchart 
        if (halo_c_v_name != None): # branch 3 of the flowchart
            print('FOGGIE_LOAD: Central halo, halo_c_v_name given' )
            if ('smoothed' in halo_c_v_name): # branch 3.2 of the flowchart
                ds, region = get_center_from_smoothed_catalog(ds, halo_c_v_name, snap, trackfile_name)
            else: # regular *un-smoothed* center, branch 3.1 of the flowchart
                ds, region = get_center_from_catalog(ds, halo_c_v_name, snap, trackfile_name) # this will take the center from the catalog file without modification
        elif (halo_c_v_name == None): # branch 4 of the flowchart
            print("FOGGIE_LOAD: Central halo, no halo_c_v_name given")
            if (trackfile_name != None): # branch 4.1 of the flowchart
                print("FOGGIE_LOAD: We will define the center from the track file and refine it")
                ds, region = get_center_from_calculated(ds, trackfile_name)
            elif (trackfile_name == None): 
                raise ValueError("FOGGIE_LOAD: You have specified a central halo but provided no track, which can't work. Exiting.")
    elif (central_halo == False): # branch 2 of the flowchart
        if (trackfile_name != None): # branch 5 of the flowchart
            print("FOGGIE_LOAD: No central halo, using center of track file")
            ds, region = get_center_from_track(ds, trackfile_name)
        elif (root_catalog_name != None): # branch 7 of the flowchart
            print("FOGGIE_LOAD: No central halo, using center of mass of root particles")
            ds, region = get_center_from_root_catalog(ds, root_catalog_name)
        else: #branch 6 of the flowchart
            print("FOGGIE_LOAD: No central halo, no trackfile or root catalog, define the region as the smallest DM box and center as the center of that box")
            ds, region = get_center_from_DM_region(ds)
    else: 
        print("FOGGIE_LOAD: central_halo keyword = ", central_halo)
        raise ValueError("FOGGIE_LOAD: You don't want to be here... something is wrong with your central_halo keyword")

    # Note that if you want to use the ('gas', 'baryon_overdensity') field, you must include this line after you've defined some data object from ds:
    # > obj.set_field_parameter('omega_baryon', ds.omega_baryon)
    # foggie_load returns a 'region' data object given by the 'region' keyword, and the 'omega_baryon' parameter is already set for that.

    if (np.isnan(ds.halo_center_code).any() == False):  # we will only add these gas cell coordinates in kpc if there is a meaningful halo center
        ds.add_field(('gas', 'x_kpc'), function=x_kpc, sampling_type='cell', units='kpc')
        ds.add_field(('gas', 'y_kpc'), function=y_kpc, sampling_type='cell', units='kpc')
        ds.add_field(('gas', 'z_kpc'), function=z_kpc, sampling_type='cell', units='kpc')

    if (np.isnan(ds.halo_velocity_kms).any() == False):  # we will only add these velocity and energy fields if there is a meaningful halo bulk velocity
        print('ds.halo_velocity_kms = ', ds.halo_velocity_kms, ' so we can add the centered velocity and energy fields')
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

    # add cell IDs 
    ds.add_field(('index', 'cell_id'), function=get_cell_ids, sampling_type='cell')
    
    # filter particles into star and dm
    if (do_filter_particles == True):
        filter_particles(region, filter_particle_types = ['young_stars', 'young_stars3', 'young_stars8', 'old_stars', 'stars', 'dm'])

    # create radius and angular momentum fields for the filtered particles 
    if (do_filter_particles & (np.isnan(ds.halo_velocity_kms).any() == False)): # filter particles into star and dm

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

    # Option to define velocities and coordinates relative to the angular momentum vector of the disk
    smooth_AM_name = kwargs.get('smooth_AM_name', False)
    disk_relative = kwargs.get('disk_relative', False)
    particle_type_for_angmom = kwargs.get('particle_type_for_angmom', 'young_stars')
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

    gravity = kwargs.get('gravity', False)
    if (gravity):
        masses_dir = kwargs.get('masses_dir', '')
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

    region.set_field_parameter('omega_baryon', ds.omega_baryon)

    return ds, region
