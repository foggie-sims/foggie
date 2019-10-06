"""
This file contains most of funcs used for mocky_way. Was origially
in yzhenggit/mocky_way repo, now slowly merge everything into foggie.

Here are the list the functions that have been added to this module:

History:
Nov 2018, YZ @ UCB
10/04/2019, YZ, merge mocky_way into foggie/mocky_way

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
        sys_dir = '/home5/yzheng7'
        data_dir = '/nobackup/yzheng7/halo_008508'
    elif test_dir.split('/')[1] == 'Users':
        # sys_dir = '/Users/Yong/Dropbox/GitRepo'
        sys_dir = '/Users/Yong/ForkedRepo/foggie'
        data_dir = '/Users/Yong/YongData/foggie/halo_008508'
    else:
        print('Do not recognize the system path, not in Yong or Pleiades.')
        sys.exit(1)
    return data_dir, sys_dir

# data_dir, sys_dir = data_dir_sys_dir()
# os.sys.path.insert(0, sys_dir)

def prepdata(dd_name, sim_name='nref11n_nref10f', robs2rs=2):
    """
    This is a generalized function setup to process a simulation output.
    It returns information of [] which will be used routinely and consistently
    by other functions.

    If you are running this for the first time for any new simulation output,
    you need to look at funcs disk_scale_length() and sphere_for_galaxy_ang_mom()
    first to decide the corresponding disk scale length and the size of the sphere
    used to calculate the disk's angular momentum. The scale length is defaulted
    to 2 kpc, and ang_sphere_rr defaulted to 10 kpc if first time running it.

    Input:
    dd_name: the output name, could be RD**** or DD****
    sim_name: used to default to nref11c_nref9f_selfshield_z6,
              now default to nref11n_nref10f
    robs2rs: default to put mock observer at twice the disk scale length rs,
             couldchange accordingly.

    Hitory:
    05/04/2019, Created, YZ, UCB
    08/06/2019, Merging prepdata_dd0946 and prepdata_rd0037 to a universal one. YZ.
    10/04/2019, update for nref11n_nref10f/RD0039 (used to be nref11c_nref9f),
                + Merging mocky_way to foggie/mocky_way. YZ.
    """

    # my rule: vector should have no unit, just numpy array
    # the point location have unit of code_length
    ds_paras = {} # this is the full parameter we will derive

    #### first, need to know where the data sit
    from foggie.foggie.mocky_way.core_funcs import data_dir_sys_dir
    data_dir, sys_dir = data_dir_sys_dir()
    ds_file = '%s/%s/%s/%s'%(data_dir, sim_name, dd_name, dd_name)
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
    ion_list = ['Si II', 'Si III', 'Si IV', 'C II', 'C III', 'C IV', 'O VI']
    print("Adding ion fields: ", ion_list)
    trident.add_ion_fields(ds, ftype="gas", ions=ion_list, force_override=True)

    #### find halo center and bulk velocity
    from foggie.mocky_way.core_funcs import find_halo_center_yz
    halo_center = find_halo_center_yz(ds, zsnap, sim_name, data_dir)
    ds_paras['halo_center'] = halo_center

    #### find r200.
    #### --> For new halos not been processed before, this function
    ####     will call foggie.mocky_way.core_funcs.calc_r200_proper
    #### --> For halos have been processed, this func will read from
    ####     a pre-defined dict.
    from foggie.mocky_way.core_funcs import dict_rvir_proper
    rvir_proper = dict_rvir_proper(dd_name, sim_name=sim_name)
    ds_paras['rvir'] = ds.quan(rvir_proper, 'kpc')

    #### decide the angular momentum of the disk, and the three vectors
    ## L_vec: angular momentum vector
    ## sun_vec: vector from galaxy center to observer at sun
    ## phi_vec: vec perpendicular to L and sun_vec following right handed rule
    from foggie.mocky_way.core_funcs import find_galaxy_L_sun_phi_bulkvel
    L_vec, sun_vec, phi_vec, disk_bulkvel = find_galaxy_L_sun_phi_bulkvel()
    ds_paras['L_vec'] = L_vec
    ds_paras['sun_vec'] = sun_vec
    ds_paras['phi_vec'] = phi_vec
    ds_paras['disk_bulkvel'] = disk_bulkvel

    #### observer location: fit the gas disk with exp profile, find rscale,
    ## check disk_radial_scale.ipynb
    ## observer to be at twice scale length, similar to MW
    ## check disk_scale_length for the value
    from foggie.mocky_way.core_funcs import locate_offcenter_observer
    ## 'disk_scale_length_kpc': disk_scale_length_kpc
    offcenter_vars = locate_offcenter_observer()
    ## offceter observer default at 2Rs from Galactic center in the disk
    ds_paras['offcenter_location'] = offcenter_vars[0]
    ## peculair motion of the observer, taken to be gas within 1 Kpc
    ds_paras['offcenter_bulkvel'] = offcenter_vars[1]

    return ds, ds_paras

def dict_rvir_proper(dd_name, sim_name='nref11n_nref10f'):
    """
    Since it takes a while to run to get r200, let's make pre-designed library
    for those sim output that has r200 calculated.

    History:
    08/06/2019, YZ.
    10/04/2019, added sim_name since now I've worked on two different sims,
                to avoid confusion. Was originally dict_rvir_proper, now
                dict_rvir_proper. YZ.
    """

    all_rvir_proper = {'nref11c_nref9f_selfshield_z6/RD0035': 144.0, # kpc
                        'nref11c_nref9f_selfshield_z6/RD0036': 147.0, # kpc
                        'nref11c_nref9f_selfshield_z6/RD0037': 150.5,
                        'nref11c_nref9f_selfshield_z6/DD0946': 98.0,
                        'nref11n_nref10f/RD0039': 157.5
                        }

    output_string = '%s/%s'%(sim_name, dd_name)
    try:
        rvir_proper = all_rvir_proper[output_string]
    except:
        print("I do not find %s/%s"%(sim_name, dd_name))
        print("Go Run > python find_r200.py sim_name dd_name")
        sys.exit(0)
    return rvir_proper

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
            print('- Refine mean rho at r=%d kpc, rho/rho_200=%.1f'%(r_progressing, rho_ratio))
        r_progressing = r_previous

    r200 = r_progressing
    rho_ratio = mean_rho(ds, halo_center, r200)/rho_crit
    print('- Phew! Find r200 (proper) =%.1f kpc, ratio=%.1f'%(r200, rho_ratio))

    return r200.in_units('kpc')

def find_galaxy_L_sun_phi_bulkvel(ds, halo_center):

    """
    Decide the angular momentum of the galaxy, and find the three unit vectors
    L_vec: angular momentum vector
    sun_vec: vector from galaxy center to observer at sun
    phi_vec: vec perpendicular to L and sun_vec following right handed rule

    History:
    10/04/2019: this chunk of code was originally inside prepdata, now separte it
                out to make code cleaner. YZ.

    """
    # tested with all-sky projection, pick the sphere radius that makes a flat projected plane
    # note that this need to be checked by eye to finally decide the value
    # go to the corresponding function to see the relevant values.
    from foggie.mocky_way.core_funcs import dict_sphere_for_gal_ang_mom
    ang_sphere_rr = sphere_for_galaxy_ang_mom(dd_name)
    # sp will be used to calculate the angular momentum of the disk
    sp = ds.sphere(halo_center, (ang_sphere_rr, 'kpc'))

    # Find the velocity of the galactic center, here defined it within a sphere
    # of ang_sphere_rr from halo_center.
    # Note that it's important to set the bulk velocity before calculating
    # angular momentum of the galaxy, not sure why, but only this way makes
    # the edge-on and face-on projection looks right.
    disk_bulkvel = sp.quantities.bulk_velocity(use_gas=True, use_particles=False)
    sp.set_field_parameter('bulk_velocity', disk_bulkvel)

    # angular momention of the disk
    spec_ang_mom = sp.quantities.angular_momentum_vector(use_gas=True, \
                                                         use_particles=False)
    ang_mom_vector = spec_ang_mom/np.sqrt(np.sum(spec_ang_mom**2))

    from yt.utilities.math_utils import ortho_find
    np.random.seed(99) ## to make sure we get the same thing everytime
    L_vec, sun_vec, phi_vec = ortho_find(ang_mom_vector) # no unit

    print("Find...")
    print("L_vec: ", L_vec)
    print("sun_vec: ", sun_vec)
    print("phi_vec: ", phi_vec)

    return L_vec, sun_vec, phi_vec, disk_bulkvel

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
    09/06/2019, YZ, UCB,
    10/04/2019, was originally named as sphere_for_galaxy_ang_mom,
                now dict_sphere_for_gal_ang_mom. YZ.
    """

    # kpc, looks good from the allsky projection from GC.
                 # see RD0037_L08kpc_n32_x800_R100.0_final.pdf
    dict_sphere_L_rr = {'nref11c_nref9f_selfshield_z6/RD0035': 8, # unit of kpc
                        'nref11c_nref9f_selfshield_z6/RD0036': 7,
                        'nref11c_nref9f_selfshield_z6/RD0037': 8,
                        'nref11c_nref9f_selfshield_z6/DD0946': 10}

    output_string = '%s/%s'%(sim_name, dd_name)
    if output_string in dict_sphere_L_rr:
        this_sphere_L_rr = dict_sphere_L_rr[output_string]
    else:
        import sys
        print('******** Exciting! First time running this output, right? ')
        print("The sphere for which angular momentum vecotr has not been decided yet. ")
        print('You need to run xx code first, then add info to foggie.mocky_way.core_funcs.dict_sphere_for_gal_ang_mom')
        print('before you can proceed :)')
        sys.exit(0)
    return this_sphere_L_rr


def locate_offcenter_observer(ds):
    disk_scale_length_kpc = disk_scale_length(dd_name) # kpc
    obs_dist = ds.quan(robs2rs*disk_scale_length_kpc, "kpc").in_units("code_length")

    ### decice at which point of the circle we want to put the observer on
    # obs_loc_vectors = [sun_vec, sun_vec+phi_vec, phi_vec, -sun_vec+phi_vec,
    #                    -sun_vec, -sun_vec-phi_vec, -phi_vec, -phi_vec+sun_vec]
    # obs_vec = obs_loc_vectors[n_vec]
    obs_vec = sun_vec+phi_vec # othis is tested by putting observor at eight different locations
    obs_vec = obs_vec/np.sqrt(np.sum(obs_vec**2))

    offcenter_location = halo_center + obs_vec*obs_dist # observer location

    # we also need to change the sun vector with the location of the observer now
    new_sun_vec = obs_vec
    new_phi_vec = -sun_vec+phi_vec
    new_phi_vec = new_phi_vec/np.sqrt(np.sum(new_phi_vec**2))

    # set the bulk velocity of the observer, taken to be gas within 1 kpc
    obs_sp = ds.sphere(offcenter_location, (1, "kpc"))
    obs_bv = obs_sp.quantities.bulk_velocity(use_gas=True, use_particles=True)
    obs_bv = obs_bv.in_units("km/s")
    offcenter_bulkvel = obs_bv
    return offcenter_location, offcenter_bulkvel

def find_halo_center_yz(ds, zsnap, sim_name, data_dir):
    """
    Kinda just quick to use the foggie funcs to find halo center by reading
    in halo track information

    History: 10/04/2019, YZ.
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
    10/04/2019, YZ wrote at some time, now merge into foggie/mocky_way

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

def get_sphere_ang_mom_vecs(ds, sp_center, r_for_L=10,
                            use_gas=True, use_particles=False,
                            random_seed=99):
    """
    Calculate the angular momentum vector for gas within a sphere of radius r.
    IMPORTANT: do not change use_gas, use_particles, random_seed, unless you
               you testing something new.

    Return:
    Dict of 3 vectors, L_vec, sun_vec, phi_vec

    History:

    10/04/2019, YZ., UCB
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

    from yt.utilities.math_utils import ortho_find
    np.random.seed(random_seed) ## to make sure we get the same thing everytime
    n1_L, n2_sun, n3_phi = ortho_find(norm_L)  # UVW vector
    dict_vecs = {'L_vec': n1_L,
                 'sun_vec': n2_sun,
                 'phi_vec': n3_phi}
    return dict_vecs
