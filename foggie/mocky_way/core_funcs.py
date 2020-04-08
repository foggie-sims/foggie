"""
This file contains most of funcs used for mocky_way. Was origially
in yzhenggit/mocky_way repo, now slowly merge everything into foggie.

Here are the list the functions that have been added to this module:

History:
Nov 2018, Yong Zheng @ UCB
10/04/2019, Yong Zheng, merge mocky_way into foggie/mocky_way

"""

##################################################################
import yt
import numpy as np
# import foggie
from foggie.utils import consistency # for plotting
# from foggie.foggie.mocky_way import derived_fields_mw # some particular fields needed
                                        # for mocky_way, like l, b
#import pandas
#import warnings
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# mpl.rcParams['font.family'] = 'stixgeneral'

#import datetime
#now = datetime.datetime.now().strftime("%Y-%m-%d")

# this is important to run before setting up other packages
# Tell the code where to look for data and packages if on different sys
import os
import sys
def data_dir_sys_dir():
    test_dir = os.environ['PWD']
    if test_dir.split('/')[1] == 'home5':
        sys_dir = '/home5/yzheng7/foggie'
        data_dir = '/nobackup/yzheng7/halo_008508'
    elif test_dir.split('/')[1] == 'Users':
        # sys_dir = '/Users/Yong/Dropbox/GitRepo'
        sys_dir = '/Users/Yong/ForkedRepo/foggie'
        data_dir = '/Users/Yong/YongData/foggie/halo_008508'
    else:
        print('Do not recognize the system path, not in Yong or Pleiades.')
        sys.exit(1)
    return data_dir, sys_dir
# sys_dir = data_dir_sys_dir()[1]
# os.sys.path.insert(0, sys_dir)

def default_random_seed():
    random_seed = 99
    return random_seed

def prepdata(dd_name, sim_name='nref11n_nref10f', robs2rs=2,
             shift_obs_location=False, shift_n45=1):
    """
    This is a generalized function setup to process a simulation output.
    It returns information of {} which will be used routinely and consistently
    by other functions.

    If you are running this for the first time for any new simulation output,
    you need to look at funcs disk_scale_length() and sphere_for_galaxy_ang_mom()
    first to decide the corresponding disk scale length and the size of the sphere
    used to calculate the disk's angular momentum.

    Input:
    dd_name: the output name, could be RD**** or DD****
    sim_name: used to default to nref11c_nref9f_selfshield_z6,
              now default to nref11n_nref10f
    robs2rs: default to put mock observer at twice the disk scale length rs,
             couldchange accordingly.
    shift_obs_location: boolean value, design to shift the observer to another
             7 offcenter location of the disk (shfit_n45 from 1 to 7)
    shift_n45: values of 1 to 7, when equal to 1, it means the observer is
               45 (nx45) degree from the original/fiducial location,

    Hitory:
    05/04/2019, Created, Yong Zheng, UCB
    08/06/2019, Merging prepdata_dd0946 and prepdata_rd0037 to
                a universal one. Yong Zheng.
    10/04/2019, update for nref11n_nref10f/RD0039 (used to be nref11c_nref9f),
                + Merging mocky_way to foggie/mocky_way. Yong Zheng.
    12/18/2019, now the code can find dataset from my hard drive, to check other
                simulation output. Yong Zheng.
    """

    # my rule: vector should have no unit, just numpy array
    # the point location have unit of code_length
    ds_paras = {} # this is the full parameter we will derive

    #### first, need to know where the data sit
    from foggie.mocky_way.core_funcs import data_dir_sys_dir
    data_dir, sys_dir = data_dir_sys_dir()
    ds_file = '%s/%s/%s/%s'%(data_dir, sim_name, dd_name, dd_name)

    import os
    if os.path.isfile(ds_file) == False:
        drive_dir = '/Volumes/Yong4TB/foggie/halo_008508'
        ds_file = '%s/%s/%s/%s'%(drive_dir, sim_name, dd_name, dd_name)

    ds_paras['dd_name'] = dd_name
    ds_paras['sim_name'] = sim_name
    ds_paras['data_path'] = ds_file
    print("Working on %s/%s"%(sim_name, dd_name))

    #### loading simulation and get redshift
    ds = yt.load(ds_file)
    zsnap = ds.get_parameter('CosmologyCurrentRedshift')
    from astropy.cosmology import Planck15 as cosmo
    age = cosmo.age(zsnap).value
    ds_paras['zsnap'] = zsnap
    print('CosmologyCurrentRedshift: %s, %.3f, %.2f Gyrs'%(dd_name, zsnap, age))

    #### post-processing, add ion fields in addition to (H, He) to ds
    # ionbg = 'sfcr'
    # iontb = '/Users/Yong/.trident/hm2012_hr_sh_cr.h5'
    import trident
    ion_list = ['Si II', 'Si III', 'Si IV', 'C II', 'C IV', 'O VI', 'N V',
                'O VII', 'O VIII', 'Ne VII', 'Ne VIII']

    print("Adding ion fields: ", ion_list)
    trident.add_ion_fields(ds, ftype="gas",
                           ions=ion_list) # ,
                           # force_override=True)

    #### add line of sight velocity field to the dataset
    from yt.fields.api import ValidateParameter
    from foggie.mocky_way.mocky_way_fields import _los_velocity_mw
    ds.add_field(("gas", "los_velocity_mw"),
                 function=_los_velocity_mw, units="km/s",
                 validators=[ValidateParameter("observer_location"),
                             ValidateParameter("observer_bulkvel")])

    #### add line of sight longitude and latitude for each cell
    from foggie.mocky_way.mocky_way_fields import _l_Galactic_Longitude, _b_Galactic_Latitude
    ds.add_field(("gas", "l"), function=_l_Galactic_Longitude,
                 units="degree", take_log=False,
                 validators=[ValidateParameter("observer_location"),
                             ValidateParameter("L_vec"),    # disk ang mom vector
                             ValidateParameter("sun_vec")]) # GC/sun direction
    ds.add_field(("gas", "b"), function=_b_Galactic_Latitude,
                 units="degree", take_log=False,
                 validators=[ValidateParameter("observer_location"), # loc of observer
                             ValidateParameter("L_vec")]) # disk ang mom vector

    #### find halo center and bulk velocity
    from foggie.mocky_way.core_funcs import find_halo_center_yz
    halo_center = find_halo_center_yz(ds, zsnap, sim_name, data_dir)
    ds_paras['halo_center'] = halo_center

    #### Get r200
    from foggie.mocky_way.core_funcs import dict_rvir_proper
    rvir_proper = dict_rvir_proper(dd_name, sim_name=sim_name)
    ds_paras['rvir'] = ds.quan(rvir_proper, 'kpc')
    print(ds_paras['rvir'])

    #### decide the angular momentum of the disk, and the three vectors
    ## L_vec: angular momentum vector
    ## sun_vec: vector from galaxy center to observer at sun
    ## phi_vec: vec perpendicular to L and sun_vec following right handed rule
    from foggie.mocky_way.core_funcs import dict_sphere_for_gal_ang_mom
    from foggie.mocky_way.core_funcs import get_sphere_ang_mom_vecs
    r_for_L = dict_sphere_for_gal_ang_mom(dd_name, sim_name=sim_name)

    from foggie.mocky_way.core_funcs import default_random_seed
    random_seed = default_random_seed()
    dict_vecs = get_sphere_ang_mom_vecs(ds, halo_center, r_for_L,
                                        random_seed=random_seed)
    ds_paras['L_vec'] = dict_vecs['L_vec']
    ds_paras['sun_vec'] = dict_vecs['sun_vec']
    ds_paras['phi_vec'] = dict_vecs['phi_vec']
    ds_paras['disk_bulkvel'] = dict_vecs['disk_bulkvel']

    # only do this if we want to shift the observer to other location in the
    # in the galactic disk
    if shift_obs_location == True:
        from foggie.mocky_way.core_funcs import shift_obs_location_func
        # n45 varies from 1 to 7 (0 equal to current position)
        ds_paras = shift_obs_location_func(ds_paras, shift_n45=shift_n45)

    ############### best to normal here #####################
    #### add disk scale height and scale length to the dict
    from foggie.mocky_way.core_funcs import dict_disk_rs_zs
    disk_rs, disk_zs = dict_disk_rs_zs(dd_name, sim_name=sim_name)
    ds_paras['disk_rs'] = disk_rs
    ds_paras['disk_zs'] = disk_zs

    #### decide off-center observer location ###
    from foggie.mocky_way.core_funcs import locate_offcenter_observer
    ds_paras = locate_offcenter_observer(ds, ds_paras, robs2rs=robs2rs)

    return ds, ds_paras

def shift_obs_location_func(ds_paras, shift_n45=1):
    """
    for the exisiting [L_vec, sun_vec, phi_vec] coordinate system, shift the
    sun_vec and phi_vec by nx45 degrees, which will change the location of
    the mock observer. The code updates the sun_vec and phi_vec vectors in
    ds_paras.

    input:
    ds_paras: the overall parameter
    shift_n45: 1, 2, 3, ..., 7, corresponding the other 7 locations
          on the disk plane to put mock observer on.

    Hisotry:
    12/18/2019, first created, Yong Zheng. UCB.
    """

    ori_L_vec = ds_paras['L_vec']
    ori_sun_vec = ds_paras['sun_vec']
    ori_phi_vec = ds_paras['phi_vec']

    # first, find the new observer location vector
    other_7_obs_locations = {'n45=1': ori_sun_vec+ori_phi_vec,
                             'n45=2': ori_phi_vec,
                             'n45=3': -ori_sun_vec+ori_phi_vec,
                             'n45=4': -ori_sun_vec,
                             'n45=5': -ori_sun_vec-ori_phi_vec,
                             'n45=6': -ori_phi_vec,
                             'n45=7': -ori_phi_vec+ori_sun_vec}
    ntag = 'n45=%d'%(shift_n45)
    obs_vec = other_7_obs_locations[ntag]

    # now get the unique vector
    obs_vec = obs_vec/np.sqrt(np.sum(obs_vec**2))
    new_sun_vec = obs_vec
    new_phi_vec = np.cross(obs_vec, ori_L_vec)
    new_phi_vec = new_phi_vec/np.sqrt(np.sum(new_phi_vec**2))

    ds_paras['sun_vec'] = new_sun_vec
    ds_paras['phi_vec'] = new_phi_vec

    return ds_paras

def locate_offcenter_observer(ds, ds_paras, robs2rs=2):
    """
    Shift the observer from galactic center to an off-center location

    Input:
    robs2rs: the offcenter location is default to twice the scale length of
             the galaxy disk.

    Return:
    ds_paras: add two other keys "offcenter_location" and "offcenter_bulkvel"
              to ds_paras

    History:
    10/08/2019, Yong Zheng. UCB.
    """

    #### Now locate the observer to 2Rs, similar to MW disk
    disk_rs = ds_paras['disk_rs']
    halo_center = ds_paras['halo_center']
    obs_vec = ds_paras['sun_vec']

    obs_dist = ds.quan(robs2rs*disk_rs, "kpc").in_units("code_length")
    offcenter_location = halo_center + obs_vec*obs_dist # observer location

    # set the bulk velocity of the observer, taken to be gas within 1 kpc
    # note that this obs_bv is in the simulation's rest frame.
    obs_sp = ds.sphere(offcenter_location, (1, "kpc"))
    obs_bv = obs_sp.quantities.bulk_velocity(use_gas=True, use_particles=True)
    offcenter_bulkvel = obs_bv.in_units("km/s")

    ds_paras['offcenter_location'] = offcenter_location
    ds_paras['offcenter_bulkvel'] = offcenter_bulkvel

    return ds_paras

def calc_r200_proper(ds, halo_center,
                     start_rad = 5,
                     rad_units = 'kpc',
                     delta_rad_coarse = 30,
                     delta_rad_fine = 5,
                     delta_rad_tiny = 0.5):
    """
    Provide R200 wrt p_cirical in proper unit

    Code adapted from Raymond.

    """
    vir_check = 0
    r_start = ds.quan(start_rad, rad_units)

    from astropy.cosmology import Planck15 as cosmo
    rho_crit = cosmo.critical_density(ds.current_redshift)   #is in g/cm^3
    rho_crit = rho_crit.to("g/cm**3")

    start_ratio = 10000
    r_progressing = r_start
    from foggie.mocky_way.core_funcs import mean_rho
    for delta_r_step in [delta_rad_coarse, delta_rad_fine, delta_rad_tiny]:
        rho_ratio = start_ratio
        print("- Now in step of %s kpc..."%(delta_r_step))
        while rho_ratio > 200:
            r_previous = r_progressing
            r_progressing = r_progressing + ds.quan(delta_r_step, rad_units)
            rho_internal = mean_rho(ds, halo_center, r_progressing)
            rho_ratio = rho_internal/rho_crit
            print('- Refine mean rho at r=%d kpc, rho_mean/rho_crit=%.1f'%(r_progressing, rho_ratio))
        r_progressing = r_previous

    r200 = r_progressing
    rho_ratio = mean_rho(ds, halo_center, r200)/rho_crit
    print('- Phew! Find r200 (proper) =%.1f kpc, ratio=%.1f'%(r200, rho_ratio))

    return r200.in_units('kpc')

def find_halo_center_yz(ds, zsnap, sim_name, data_dir):
    """
    Kinda just quick to use the foggie funcs to find halo center by reading
    in halo track information

    History: 10/04/2019, Yong Zheng.
    """

    from foggie.utils.get_halo_center import get_halo_center
    from foggie.utils.get_refine_box import get_refine_box
    from astropy.table import Table

    halo_track = '%s/%s/halo_track'%(data_dir, sim_name)
    track = Table.read(halo_track, format='ascii')
    track.sort('col1')
    box_paras = get_refine_box(ds, zsnap, track)
    refine_box = box_paras[0]
    refine_box_center = box_paras[1]
    refine_width_code = box_paras[2]
    halo_center, halo_velocity = get_halo_center(ds, refine_box_center)
    halo_center = ds.arr(halo_center, 'code_length')

    return halo_center

def mean_rho(ds, center, r):
    """
    Mean density within a sphere of radius r

    History:
    10/04/2019, Yong Zheng wrote at some time, now merge into foggie/mocky_way

    """

    sp = ds.sphere(center, r)
    # dark matter and gas mass within r, g
    dm_mass = sp['particle_mass'].sum()
    gas_mass = sp[('gas', 'cell_mass')].sum()
    mr = (dm_mass + gas_mass).in_units('g')

    # volume, proper units, cm3
    vr = 4*np.pi/3 * (r.in_units('cm'))**3

    # mean density within r, g/cm3
    rho_internal = mr/vr

    return rho_internal

def ortho_find_yz(z, random_seed=99):
    """
    Realize the yt version of ortho_find do not have the flexiblity to change
    the different x, y vectors, so decide to write one myself. It's basically
    identical to yt.utilities.math_utils.ortho_find, with an additional para
    of random_seed so that we can change the x, y vectors if needed.

    Return: unit vector, L_vec, sun_vec, phi_vec, follow right hand rule.

    10/13/2019, YZ
    """
    import numpy as np
    np.random.seed(random_seed)
    z /= np.linalg.norm(z)  # normalize it

    x = np.random.randn(3)  # take a random vector
    x -= x.dot(z) * z       # make it orthogonal to z
    x /= np.linalg.norm(x)  # normalize it

    y = np.cross(z, x)      # cross product with z
    y /= np.linalg.norm(y)  # normalize it

    sun_vec = yt.YTArray(x)
    phi_vec = yt.YTArray(y)
    L_vec = yt.YTArray(z)

    return L_vec, sun_vec, phi_vec

def get_sphere_ang_mom_vecs(ds, sp_center, r_for_L=20,
                            use_gas=True, use_particles=False,
                            random_seed=99):
    """
    Calculate the angular momentum vector for gas within a sphere of radius r.
    IMPORTANT: do not change use_gas, use_particles, random_seed, unless you
               you testing something new.

    Return:
    Dict of 3 vectors, L_vec, sun_vec, phi_vec

    History:

    10/04/2019, Yong Zheng., UCB
    """

    sp = ds.h.sphere(sp_center, (r_for_L, 'kpc'))
    # let's set up the bulk velocity before setting up the angular momentum
    # this setup is very important
    sp_bulkvel = sp.quantities.bulk_velocity(use_gas=use_gas,
                                             use_particles=use_particles)
    sp.set_field_parameter('bulk_velocity', sp_bulkvel)

    # angular momentum
    spec_L = sp.quantities.angular_momentum_vector(use_gas=use_gas,
                                                   use_particles=use_particles)
    norm_L = spec_L / np.sqrt((spec_L**2).sum())

    from foggie.mocky_way.core_funcs import ortho_find_yz
    n1_L, n2_sun, n3_phi = ortho_find_yz(norm_L, random_seed=random_seed)  # UVW vector
    dict_vecs = {'L_vec': n1_L,
                 'sun_vec': n2_sun,
                 'phi_vec': n3_phi,
                 'disk_bulkvel': sp_bulkvel.in_units('km/s')}

    return dict_vecs

def dict_rvir_proper(dd_name, sim_name='nref11n_nref10f'):
    """
    Since it takes a while to run to get r200, let's make pre-designed library
    for those sim output that has r200 calculated.

    History:
    08/06/2019, Yong Zheng.
    10/04/2019, added sim_name since now I've worked on two different sims,
                to avoid confusion. Was originally dict_rvir_proper, now
                dict_rvir_proper. Yong Zheng.
    """

    all_rvir_proper = {'nref11c_nref9f_selfshield_z6/RD0035': 144.0, # kpc
                        'nref11c_nref9f_selfshield_z6/RD0036': 147.0, # kpc
                        'nref11c_nref9f_selfshield_z6/RD0037': 150.5,
                        'nref11c_nref9f_selfshield_z6/DD0946': 98.0,
                        'nref11n_nref10f/RD0039': 157.5,
                        'nref11n_nref10f/DD2175': 161.0,
                        'nref11n_nref10f/RD0041': 165.5,
                        'nref11n_nref10f/RD0042': 170.0
                        }

    output_string = '%s/%s'%(sim_name, dd_name)
    try:
        rvir_proper = all_rvir_proper[output_string]
    except:
        print("I do not find %s/%s"%(sim_name, dd_name))
        print("Go Run > python find_r200.py sim_name dd_name")
        sys.exit(0)
    return rvir_proper

def dict_sphere_for_gal_ang_mom(dd_name, sim_name='nref11n_nref10f'):
    """
    This is a hand-coded function, the radius of the sphere is pre-decided by
    running the allsky_proj_GCview.py function to check the allsky projection
    of the galaxy from the point of galactic center, and pick a sphere radius
    which results in a flat disk. Check allsky_proj_GCview.py for more info

    If allsky_proj_GCview.py has been run, then record the max sphere radius
    in a dict; if not, then assume the sphere to be 10 kpc. But note that,
    in this case, the all sky projection may not result in a flat disk.

    History:
    09/06/2019, Yong Zheng, UCB,
    10/04/2019, was originally named as sphere_for_galaxy_ang_mom,
                now dict_sphere_for_gal_ang_mom. Yong Zheng.
    """

    # kpc, looks good from the allsky projection from GC.
                 # see RD0037_L08kpc_n32_x800_R100.0_final.pdf
    dict_sphere_L_rr = {'nref11c_nref9f_selfshield_z6/RD0035': 8, # unit of kpc
                        'nref11c_nref9f_selfshield_z6/RD0036': 7,
                        'nref11c_nref9f_selfshield_z6/RD0037': 8,
                        'nref11c_nref9f_selfshield_z6/DD0946': 10,
                        'nref11n_nref10f/RD0039': 20,
                        'nref11n_nref10f/DD2175': 5,
                        'nref11n_nref10f/RD0041': 15,
                        'nref11n_nref10f/RD0042': 10}

    output_string = '%s/%s'%(sim_name, dd_name)
    if output_string in dict_sphere_L_rr:
        this_sphere_L_rr = dict_sphere_L_rr[output_string]
    else:
        import sys
        print('******** Exciting! First time running this output, right? ')
        print("The sphere for which angular momentum vecotr has not been decided yet. ")
        print('You need to run find_flat_disk_offaxproj and xxx_allskyproj first, ')
        print('then add info to foggie.mocky_way.core_funcs.dict_sphere_for_gal_ang_mom')
        print('before you can proceed :)')
        sys.exit(0)
    return this_sphere_L_rr

def dict_disk_rs_zs(dd_name, sim_name='nref11n_nref10f'):
    """
    This is a hand-coded function to record the disk scale length and height
    of input runs. Need to run disk_scale_length_rs.py and disk_scale_height_zs.py
                then record numbers here.

    History:
    10/07/2019, Yong Zheng, UCB.
    """

    # kpc, looks good from the allsky projection from GC.
                 # see RD0037_L08kpc_n32_x800_R100.0_final.pdf
    dict_rs = {'nref11c_nref9f_selfshield_z6/RD0037': 3.9,
               'nref11n_nref10f/RD0039': 3.3,
               'nref11n_nref10f/DD2175': 3.4,
               'nref11n_nref10f/RD0041': 3.9,
               'nref11n_nref10f/RD0042': 4.4,
               }

    dict_zs = {'nref11c_nref9f_selfshield_z6/RD0037': 1.4,
               'nref11n_nref10f/RD0039': 0.4,
               'nref11n_nref10f/DD2175': 0.5,
               'nref11n_nref10f/RD0041': 0.3,
               'nref11n_nref10f/RD0042': 0.6}

    output_string = '%s/%s'%(sim_name, dd_name)
    if output_string in dict_rs:
        this_rs = dict_rs[output_string]
        this_zs = dict_zs[output_string]
    else:
        import sys
        print('******** Exciting! First time running this output, right? ')
        print("Do not have rs or zs recorded, go run disk_scale_length_rs.py, ")
        print('and disk_scale_height_zs.py first')
        print('then add info to foggie.mocky_way.core_funcs.dict_disk_rs_zs')
        print('before you can proceed :)')
        sys.exit(0)
    return this_rs, this_zs

def obj_source_shell(ds, ds_paras, shell_rin, shell_rout):
    """
    Take a shell out of a halo.

    shell_rin: inner radius of a shell
    shell_rout: outer radius of a shell.

    History:
    10/15/2019, Created, Yong Zheng. UCB.
    """

    sp_in = ds.sphere(ds_paras['halo_center'], (shell_rin, 'kpc'))
    sp_out = ds.sphere(ds_paras['halo_center'], (shell_rout, 'kpc'))
    shell = sp_out - sp_in

    return shell


def obj_source_all_disk_cgm(ds, ds_paras, obj_tag, test=False):
    """
    This is to cut the simulation into halo-only, disk, and both halo and disk
    objects for further data analyses. And I'll use this for other functions
    as well.

    obj_tag: 'all', 'disk', 'cgm'

    test: if True, then do a small sphere of 20 kpc to test code.

    09/26/19, Yong Zheng, UCB.
    10/09/2019, Yong Zheng, was obj_source_halo_disk, now merging into foggie.mocky_way
    10/09/2019, Yong Zheng, now need to specify which part of the galaxy you want to process
    10/11/2019, realizing the rvir of DD2175 is 160, which is beyond the refine
                box (+/-130 kpc), so I'm doing the sphere of 120 kpc from now on.
                Yong Zheng. UCB.
    10/15/2019, add test para to speed up code testing.
    """

    # halo_radius = ds_paras['rvir']
    nrs = 6 # size of the disk in r direction
    nzs = 4 # size of the disk in z direction, +/-nzs * zs

    if test == True:
        halo_radius = ds.quan(20, 'kpc')
        sp = ds.sphere(ds_paras['halo_center'], (20, 'kpc'))
        obj = sp

    elif obj_tag == 'disk':
        disk_size_r = nrs * ds_paras['disk_rs']
        disk_size_z = nzs * ds_paras['disk_zs']
        disk = ds.disk(ds_paras['halo_center'],
                       ds_paras['L_vec'],
                       (disk_size_r, 'kpc'),
                       (disk_size_z, 'kpc'))
        obj = disk

    elif obj_tag == 'all-rvir':
        sp = ds.sphere(ds_paras['halo_center'], ds_paras['rvir'])
        obj = sp

    elif obj_tag == 'all-refined':
        sp = ds.sphere(ds_paras['halo_center'], (120, 'kpc'))
        obj = sp

    elif obj_tag == 'cgm-rvir':
        sp = ds.sphere(ds_paras['halo_center'], ds_paras['rvir'])
        disk_size_r = nrs * ds_paras['disk_rs']
        disk_size_z = nzs * ds_paras['disk_zs']
        disk = ds.disk(ds_paras['halo_center'],
                       ds_paras['L_vec'],
                       (disk_size_r, 'kpc'),
                       (disk_size_z, 'kpc'))
        cgm = sp-disk
        obj = cgm

    elif obj_tag == 'cgm-refined':
        sp = ds.sphere(ds_paras['halo_center'], (120, 'kpc'))
        disk_size_r = nrs * ds_paras['disk_rs']
        disk_size_z = nzs * ds_paras['disk_zs']
        disk = ds.disk(ds_paras['halo_center'],
                       ds_paras['L_vec'],
                       (disk_size_r, 'kpc'),
                       (disk_size_z, 'kpc'))
        cgm = sp-disk
        obj = cgm

    elif obj_tag == 'cgm-15kpc':
        sp = ds.sphere(ds_paras['halo_center'], (15, 'kpc'))
        disk_size_r = nrs * ds_paras['disk_rs'] # 4 is decided by eyeballing the size in find_flat_disk_offaxproj
        disk_size_z = nzs * ds_paras['disk_zs'] # one side,
        disk = ds.disk(ds_paras['halo_center'],
                       ds_paras['L_vec'],
                       (disk_size_r, 'kpc'),
                       (disk_size_z, 'kpc'))
        cgm_15kpc = sp-disk
        obj = cgm_15kpc

    elif obj_tag == 'cgm-20kpc':
        sp = ds.sphere(ds_paras['halo_center'], (20, 'kpc'))
        disk_size_r = nrs * ds_paras['disk_rs'] # 4 is decided by eyeballing the size in find_flat_disk_offaxproj
        disk_size_z = nzs * ds_paras['disk_zs'] # one side,
        disk = ds.disk(ds_paras['halo_center'],
                       ds_paras['L_vec'],
                       (disk_size_r, 'kpc'),
                       (disk_size_z, 'kpc'))
        cgm_20kpc = sp-disk
        obj = cgm_20kpc

    else:
        print("I have no idea what you want, please put in all, disk, or cgm.")
        import sys
        sys.exit()

    return obj

def temperature_category():
    """
    This func setup the temperature ranges that will be consisitently used
    through this work.

    History:
    Created on 08/14/2019, YZ, UCB.
    """

    temp_dict = {'cold': [-np.inf, 1e4],
                 'cool': [1e4, 1e5],
                 'warm': [1e5, 1e6],
                 'hot': [1e6, +np.inf]}
    return temp_dict

def calc_mean_median_3sig_2sig_1sig(data):
    """
    Sort the data from small to large, then find the mean,
    median (50%), 3sig, 2sig, and 1 sig boundaries of the data,
    where 3/2/1sig means the range which enclose 99.7%, 95%,
    and 68% of the data.

    Return: a dict called data_stat, with keys of mean, median,
    3sig_up, 3sig_low,  2sig_up, 2sig_low, 1sig_up, 1sig_low.

    History:
    10/26/2019, Yong Zheng, UCB.
    11/02/2019, update comments. UCB.
    """
    import numpy as np
    data_stat = {}

    data = data[np.argsort(data)]
    all_index = np.arange(data.size)+1
    cum_frac = all_index/data.size

    # mean value
    data_stat['mean'] = np.mean(data)

    # median value
    indmed = np.argmin(np.abs(cum_frac-0.5))
    data_stat['median'] = data[indmed]

    # the boundaries which enclose 99.7% of the data
    threesig = 0.9973
    indup = np.argmin(np.abs(cum_frac-(0.5+threesig/2.)))
    indlow = np.argmin(np.abs(cum_frac-(0.5-threesig/2.)))
    data_stat['3sig_up'] = data[indup]   # upper 3 sigma limit
    data_stat['3sig_low'] = data[indlow] # lower 3 sigmma limit

    # the boundaries which enclose 95% of the data
    twosig = 0.95
    indup = np.argmin(np.abs(cum_frac-(0.5+twosig/2.)))
    indlow = np.argmin(np.abs(cum_frac-(0.5-twosig/2.)))
    data_stat['2sig_up'] = data[indup]  # upper 2 sigma limit
    data_stat['2sig_low'] = data[indlow] # lower 2 sigmma limit

    # the boundaries which enclose 68% of the data
    onesig = 0.68
    indup = np.argmin(np.abs(cum_frac-(0.5+onesig/2.)))
    indlow = np.argmin(np.abs(cum_frac-(0.5-onesig/2.)))
    data_stat['1sig_up'] = data[indup]    # upper 1 sigma limit
    data_stat['1sig_low'] = data[indlow]  # lower 1 sigmma limit

    return data_stat

##### this is to get the ion information along a designated line of sight
def ray_info_at_l_b(ds, ds_paras, los_l_deg, los_b_deg,
                    los_ray_start, ray_length_kpc,
                    ion_fields=['H_p0_number_density',
                                'temperature',
                                'metallicity']):
    """
    Based on old code logN_per_los.py, use to find the ion information
    along a particular line of sight.

    Input:
    los_l_deg: Galactic longitude, see derived_fields_mw.py
    los_b_deg: Galactic latitude, see derived_fields_mw.py
    los_ray_start: ray starting point, code unit   # ray end is called los_ray_end
    ray_length_kpc: total ray length, starting at los_ray_start,
                pointing toward direction of (los_l_deg, los_b_deg)
    ion_fields: the ion field that we will get number density for
                currrently the prepdata function has setup the trident
                for a list of ions: SiII, SiIII, SiIV, CII, CIII, CIV, OVI
    other_field_to_add: in case in the future we want to explore other properties
                for each cell, let's save some more information

    Output:
    output_ray_info:
        "dr_cm": the ray length (cm) in each cell intercepted by the line of sight
        "vr_los_kms": the line of sight velocity of each cell seen by observer
                   obsever information is from prepdata(dd_name)
        "ion_num_den": a dictionary with number density per cell along ray los
                    for each entry in the ion_fields
        other fields could be temperature, metallicity, etc.

    History:
    04/30/2019, created, Yong Zheng, UCB
    08/09/2019, merged into mocky_way_modules, Yong Zheng, UCB
    10/26/2019, copy/paste to foggie.mocky_way.core_func. Served as a record
                for potential future fuse, have not been tested since copied.
                Yong Zheng. UCB.
    11/01/2019, split this function into calc_ray_ion_column_density and
                calc_ray_end. funcs can still be run. But better to use a
                combination of calc_ray_end and calc_ray_ion_column_density
                for code clarity. Yong Zheng. UCB.
    """

    # from mocky_way_modules import calc_ray_end
    from foggie.mocky_way.core_funcs import calc_ray_end
    ray_length = ds.quan(ray_length_kpc, "kpc").in_units("code_length")
    los_ray_end, los_unit_vec = calc_ray_end(ds, ds_paras, los_l_deg, los_b_deg,
                                             los_ray_start, ray_length_kpc)

    # let's build a library that contains information for ray along los
    output_ray_info = {}

    # use trident to get SiIV , CIV, and OVI density fields
    import trident
    save_temp_ray = "temp/ray.h5"
    td_ray = trident.make_simple_ray(ds,
                                     start_position=los_ray_start.copy(),
                                     end_position=los_ray_end.copy(),
                                     data_filename=save_temp_ray,
                                     # lines=line_list,
                                     # fields=[ion_field],
                                     fields=ion_fields.copy(),
                                     ftype="gas")
    # sort tri_ray according to the distance of each from ray start
    td_x = td_ray.r["gas", "x"].in_units("code_length") - los_ray_start[0]
    td_y = td_ray.r["gas", "y"].in_units("code_length") - los_ray_start[1]
    td_z = td_ray.r["gas", "z"].in_units("code_length") - los_ray_start[2]
    td_r = np.sqrt(td_x**2 + td_y**2 + td_z**2).in_units("kpc")
    td_sort = np.argsort(td_r)

    #### parameter 1: ray path r, and interval dr in unit of cm
    td_r_cm = td_r[td_sort].in_units("cm")
    td_dr_cm = td_r_cm[1:]-td_r_cm[:-1]
    output_ray_info['dr_cm'] = td_dr_cm

    #### parameter 2: los velocity with respect to observer
    # trident vel wrt observer, this is the same as my own yt code,
    # check test_triray_coords_vlsr.ipynb
    # sphere_radius = 1 # kpc
    # obs_sp = ds.sphere(ds_paras["observer_location"], (sphere_radius, "kpc"))
    # obs_bv = obs_sp.quantities.bulk_velocity(use_gas=True, use_particles=False)
    # obs_bv = obs_bv.in_units("km/s")

    ## YZ comment this out on 11/1/2019, not necessary now, should add later.
    ## obs_bv = ds_paras['observer_bulkvel']
    ## costheta_bv_los = np.dot(los_unit_vec, (obs_bv/np.linalg.norm(obs_bv)).value)
    ## obs_bv_rproj = np.linalg.norm(obs_bv)*costheta_bv_los
    ## td_vr = td_ray.r["gas", "velocity_los"].in_units("km/s")
    ## td_vr_los_kms = -td_vr - ds.quan(obs_bv_rproj, 'km/s')
    ## td_vr_los_kms = td_vr_los_kms[td_sort][:-1]
    ## output_ray_info['vr_los_kms'] = td_vr_los_kms

    # parameter 3:**: ion number densities and other parameters along the ray
    # td_nion = td_ray.r["gas", ion_field][td_sort][:-1]
    for ion_field in ion_fields:
        aa = td_ray.r["gas", ion_field][td_sort][:-1]
        output_ray_info[ion_field] = td_ray.r["gas", ion_field][td_sort][:-1]

    # let's do a print to show what information is recorded
    print("Hello, I'm returning these information along the ray: ")
    print(output_ray_info.keys())

    # column density, full velocity range
    # logN = np.log10((td_nion * td_dr_cm).sum())

    # column density, integrate over [vmin, vmax]
    # indv = np.all([td_vr_los>=vmin, td_vr_los<=vmax], axis=0)
    # logN_low = np.log10((td_nion[indv]*td_dr_cm[indv]).sum())

    # return logN, logN_low

    return output_ray_info

def calc_ray_end(ds, ds_paras, los_l_deg, los_b_deg,
                 los_ray_start, ray_length_kpc):
    """
    Calculate the ray ending pointing in the Galactic coordinates using
    (los_l_deg, los_b_deg, ray_length). Old code can be found in
    old_codes/old_calc_ray_end.py los_ray_start is also in unit of code_length

    Input: ds, ds_paras, los_l_deg, los_b_deg, los_ray_start, ray_length_kpc

    Return: ray_end, los_unit_vector

    History:
    04/30/2019, created, Yong Zheng, UCB
    08/09/2019, merged into mocky_way_modules, Yong Zheng, UCB
    10/26/2019, copy/paste to foggie.mocky_way.core_func. Served as a record
                for potential future fuse, have not been tested since copied.
                Yong Zheng. UCB.
    11/1/2019, tested for compatibility with foggie. all set. Yong Zheng.
    """
    import numpy as np

    ray_length = ds.quan(ray_length_kpc, "kpc").in_units("code_length")
    los_l_rad = los_l_deg/180.*np.pi
    los_b_rad = los_b_deg/180.*np.pi

    vec_uv_plane = -np.cos(los_l_rad)*ds_paras['sun_vec'] - \
                    -np.sin(los_l_rad)*ds_paras['phi_vec']
    #vec_uv_plane = -np.cos(np.radians(los_l))*ds_paras['sun_vec'] - \
    #                -np.sin(np.radians(los_l))*ds_paras['phi_vec']
    vec_uv_plane_unit = vec_uv_plane/np.linalg.norm(vec_uv_plane)

    # note: here uvw, I mean in the galactic plane,
    # W is the L direction, u is run, and v is phi

    los_vector_uv = vec_uv_plane_unit*(ray_length*np.cos(los_b_rad)).value
    los_vector_w = ds_paras["L_vec"]*(ray_length*np.sin(los_b_rad)).value

    #los_vector_uv = vec_uv_plane_unit*(ray_length*np.cos(np.radians(los_b))).value
    #los_vector_w = ds_paras["L_vec"]*(ray_length*np.sin(np.radians(los_b))).value
    los_vector =  los_vector_uv + los_vector_w
    los_length = ds.quan(np.linalg.norm(los_vector), 'code_length')
    # the length should be the same as los_r
    if (los_length-ray_length).in_units('kpc').value > 0.01:
        print('vector length not equal to los length, wrong!')

    ray_end = ds.arr(los_vector, "code_length") + los_ray_start

    los_unit_vector = los_vector/np.linalg.norm(los_vector)

    return ray_end, los_unit_vector

def calc_ray_ion_column_density(ds, ion, los_ray_start, los_ray_end,
                                rayfilename = "ray.h5"):
    """
    Calculate the column density of ion along the line of sight

    Return: ion column density, and other information along the ray

    History:
    10/26/2019, created, Yong Zheng, UCB.
    """
    import numpy as np
    import trident
    from foggie.utils import consistency
    ion_fields = [consistency.species_dict[ion]]
    save_temp_ray = './%s'%(rayfilename)
    td_ray = trident.make_simple_ray(ds,
                                     start_position=los_ray_start.copy(),
                                     end_position=los_ray_end.copy(),
                                     data_filename=save_temp_ray,
                                     # lines=line_list,
                                     # fields=[ion_field],
                                     fields=ion_fields.copy(),
                                     ftype="gas")
    # sort tri_ray according to the distance of each from ray start
    td_x = td_ray.r["gas", "x"].in_units("code_length") - los_ray_start[0]
    td_y = td_ray.r["gas", "y"].in_units("code_length") - los_ray_start[1]
    td_z = td_ray.r["gas", "z"].in_units("code_length") - los_ray_start[2]
    td_r = np.sqrt(td_x**2 + td_y**2 + td_z**2).in_units("kpc")
    td_sort = np.argsort(td_r)

    output_ray_info = {}
    output_ray_info['r_kpc'] = td_r[td_sort].in_units("kpc")[:-1]
    #### parameter 1: ray path r, and interval dr in unit of cm and kpc
    td_r_cm = td_r[td_sort].in_units("cm")
    td_dr_cm = td_r_cm[1:]-td_r_cm[:-1]
    output_ray_info['dr_cm'] = td_dr_cm

    td_r_kpc = td_r[td_sort].in_units("kpc")
    td_dr_kpc = td_r_kpc[1:]-td_r_kpc[:-1]
    output_ray_info['dr_kpc'] = td_dr_kpc

    # parameter 2:**: ion number densities and other parameters along the ray
    # td_nion = td_ray.r["gas", ion_field][td_sort][:-1]
    ion_field = consistency.species_dict[ion]
    aa = td_ray.r["gas", ion_field][td_sort][:-1]
    nion = td_ray.r["gas", ion_field][td_sort][:-1]
    output_ray_info['n_%s'%(ion)] = nion

    # get the column density
    column_density = (nion*td_dr_cm).sum()
    ray_Nion = column_density.value

    return ray_Nion, output_ray_info

def los_r(ds, data, observer_location):
    """
    Calculate the distance between each cell in data and the observer
    at either galaxy center or at an offcenter location

    observer_location: halo_center, or offcenter_location
    data: a sphere, disk ,or anything

    Return:
    los_r_kpc: distance between gas and obsever in unit of kpc

    History:
    10/29/2019, created to calculate mass flux rate, Yong Zheng, UCB.

    """
    # calculate the distance from observer to each gas cell
    # position and position vector of each cell

    import numpy as np

    x = data["gas", "x"].in_units("code_length").flatten()
    y = data["gas", "y"].in_units("code_length").flatten()
    z = data["gas", "z"].in_units("code_length").flatten()
    los_x = x - observer_location[0] # shape of (N, )
    los_y = y - observer_location[1]
    los_z = z - observer_location[2]

    los_xyz = np.array([los_x, los_y, los_z]) # shape of (3, N)
    los_r_codelength = np.sqrt(los_x**2 + los_y**2 + los_z**2) # shape of (N, )
    los_r_kpc = los_r_codelength.in_units('kpc')

    return los_r_kpc

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.
    This return the biased weighted std, see:
    https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_variance

    values, weights -- Numpy ndarrays with the same shape.
    """
    import numpy as np
    import math
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (average, math.sqrt(variance))

def get_line_info(line):
    ## e.g., lines = ['SiIV1394', 'CIV1548']
    from yztools.line_wave_fval import line_wave_fval
    from foggie.utils import consistency
    import re

    line_info = line_wave_fval(line)
    line_wave = line_info['wave']

    ion = re.split('(\d+)', line)[0]
    ion_field  = consistency.species_dict[ion]

    ele = ion_field.split('_')[0]
    stat = ion[len(ele):]
    input_wave = re.split('(\d+)', line)[1]

    tr_line_format = '%s %s %s'%(ele, stat, input_wave)

    return line_wave, ion_field, tr_line_format
