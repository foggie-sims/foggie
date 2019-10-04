"""
This file contains most of funcs used for mocky_way. Was origially
in yzhenggit/mocky_way repo, now slowly merge everything into foggie.

Here are the list the functions that have been added to this module:

History:
Nov 2018, YZ @ UCB
10/04/2019, YZ, merge mocky_way into foggie/mocky_way

"""

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
        sys_dir = '/Users/Yong/Dropbox/GitRepo'
        data_dir = '/Users/Yong/YongData/foggie/halo_008508'
    else:
        print('Do not recognize the system path, not in Yong or Pleiades.')
        sys.exit(1)
    return data_dir, sys_dir

# data_dir, sys_dir = data_dir_sys_dir()
# os.sys.path.insert(0, sys_dir)

##################################################################
import yt
import numpy as np
# import foggie
from foggie.utils import consistency # for plotting
from mocky_way import derived_fields_mw # some particular fields needed
                                        # for mocky_way, like l, b
#import pandas
#import warnings
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# mpl.rcParams['font.family'] = 'stixgeneral'

#import datetime
#now = datetime.datetime.now().strftime("%Y-%m-%d")

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

    #### first, need to know where the data sit
    from foggie.mocky_way.core_funcs import data_dir_sys_dir
    data_dir, sys_dir = data_dir_sys_dir()
    halo_track = '%s/%s/halo_track'%(data_dir, sim_name)
    ds_file = '%s/%s/%s/%s'%(data_dir, sim_name, dd_name, dd_name)
    print("Working on %s/%s"%(sim_name, dd_name))

    #### loading simulation and get redshift
    ds = yt.load(ds_file)
    zsnap = ds.get_parameter('CosmologyCurrentRedshift')
    from astropy.cosmology import Planck15 as cosmo
    age = cosmo.age(zsnap).value
    print('CosmologyCurrentRedshift: %s, %.3f, %.2f Gyrs'%(dd_name, zsnap, age))

    #### post-processing, add ion fields in addition to (H, He) to ds
    # ionbg = 'sfcr'
    # iontb = '/Users/Yong/.trident/hm2012_hr_sh_cr.h5'
    import trident
    ion_list = ['Si II', 'Si III', 'Si IV', 'C II', 'C III', 'C IV', 'O VI']
    print("Adding ion fields: ", ion_list)
    trident.add_ion_fields(ds, ftype="gas", ions=ion_list, force_override=True)

    #### find halo center and bulk velocity
    from foggie.utils.get_halo_center import get_halo_center
    from foggie.utils.get_refine_box import get_refine_box
    from astropy.table import Table
    track = Table.read(halo_track, format='ascii')
    track.sort('col1')
    box_paras = get_refine_box(ds, zsnap, track)
    refine_box = box_paras[0]
    refine_box_center = box_paras[1]
    refine_width_code = box_paras[2]
    halo_center, halo_velocity = get_halo_center(ds, refine_box_center)
    halo_center = ds.arr(halo_center, 'code_length')

    #### find r200.
    #### --> For new halos not been processed before, this function
    ####     will call foggie.mocky_way.core_funcs.calc_r200_proper
    #### --> For halos have been processed, this func will read from
    ####     a pre-defined dict.
    from foggie.mocky_way.core_funcs import dict_rvir_proper
    rvir_proper = dict_rvir_proper(ds, dd_name, sim_name=sim_name,
                                   halo_center=halo_center)

    #### decide the angular momentum of the disk, and the three vectors
    #### L_vec: angular momentum vector
    #### sun_vec: vector from galaxy center to observer at sun
    #### phi_vec: vec perpendicular to L and sun_vec following right handed rule
    L_vec, sun_vec, phi_vec = find_galaxy_L_sun_phi_vecs()

    # observer location: fit the gas disk with exp profile, find rscale,
    # check disk_radial_scale.ipynb
    # observer to be at twice scale length, similar to MW
    # check disk_scale_length for the value
    disk_scale_length_kpc = disk_scale_length(dd_name) # kpc
    obs_dist = ds.quan(robs2rs*disk_scale_length_kpc, "kpc").in_units("code_length")

    ### decice at which point of the circle we want to put the observer on
    # obs_loc_vectors = [sun_vec, sun_vec+phi_vec, phi_vec, -sun_vec+phi_vec,
    #                    -sun_vec, -sun_vec-phi_vec, -phi_vec, -phi_vec+sun_vec]
    # obs_vec = obs_loc_vectors[n_vec]
    obs_vec = sun_vec+phi_vec # othis is tested by putting observor at eight different locations
    obs_vec = obs_vec/np.sqrt(np.sum(obs_vec**2))

    observer_location = halo_center + obs_vec*obs_dist # observer location

    # we also need to change the sun vector with the location of the observer now
    new_sun_vec = obs_vec
    new_phi_vec = -sun_vec+phi_vec
    new_phi_vec = new_phi_vec/np.sqrt(np.sum(new_phi_vec**2))

    # set the bulk velocity of the observer, taken to be gas within 1 kpc
    obs_sp = ds.sphere(observer_location, (1, "kpc"))
    obs_bv = obs_sp.quantities.bulk_velocity(use_gas=True, use_particles=True)
    obs_bv = obs_bv.in_units("km/s")

    # my rule: vector should have no unit, just numpy array
    # the point location have unit of code_length
    ds_paras = {'halo_center': halo_center, \
                'halo_velocity': halo_velocity, \
                # paras for the Zoom refine region
                'refine_box': refine_box, \
                'refine_box_center': refine_box_center, \
                'refine_width_code': refine_width_code, \
                # observer at 2Rs from Galactic center in the disk
                'observer_location': observer_location, \
                # peculair motion of the observer, taken to be gas within 1 Kpc
                'observer_bulkvel': obs_bv,
                # angular momentum vector
                'L_vec': L_vec, \
                # from GC to observer direction
                'sun_vec': new_sun_vec, \
                # perpendicular to sun_vec
                'phi_vec': new_phi_vec, \
                # in unit of kpc
                'rvir': rvir_proper,
                # disk scale length,
                'disk_scale_length_kpc': disk_scale_length_kpc, \
                # bulkvel of the disk, taken to be gas bulk vel within 10 kpc.
                'disk_bulkvel': disk_bulkvel.in_units('km/s'), \
                # simulation info
                'dd_name': dd_name,
                'sim_name': sim_name,
                'data_path': ds_file,
                'trackfile': halo_track,
                'zsnap': zsnap,
                'ang_sphere_rr': ang_sphere_rr
                }
    return ds, ds_paras


def dict_rvir_proper(ds, dd_name, sim_name='nref11n_nref10f', halo_center=[0., 0., 0.]):
    """
    Since it takes a while to run to get r200, let's make pre-designed library
    for those sim output that has r200 calculated, and only call calc_r200_proper
    when the rvir value is missing.

    History:
    08/06/2019, YZ.
    10/04/2019, added sim_name since now I've worked on two different sims,
                to avoid confusion. Was originally dict_rvir_proper, now
                dict_rvir_proper. YZ.
    """

    rvir_unit = 'kpc'
    all_rvir_proper = {'nref11c_nref9f_selfshield_z6/RD0035': ds.quan(144.0, rvir_unit),
                        'nref11c_nref9f_selfshield_z6/RD0036': ds.quan(147.0, rvir_unit),
                        'nref11c_nref9f_selfshield_z6/RD0037': ds.quan(150.5, rvir_unit),
                        'nref11c_nref9f_selfshield_z6/DD0946': ds.quan(98.0, rvir_unit)
                        }

    output_string = '%s/%s'%(sim_name, dd_name)
    if output_string in all_rvir_proper:
        rvir_proper = all_rvir_proper[output_string]
        print('rvir_proper exists already for %s'%(output_string))
    else:
        print("Do not have rvir info, now calculating...")
        from foggie.mocky_way.core_funcs import calc_r200_proper
        rvir_proper = calc_r200_proper(ds, halo_center,
                                       start_rad=50,  # in unit of 50 kpc
                                       delta_rad_coarse=30,
                                       delta_rad_fine=5,
                                       delta_rad_tiny=0.5)
        import sys
        print('NEW!! rvir_proper=%.1f kpc for %s'%(output_string))
        print('******** Exciting! First time running this output, right? ')
        print('You need to add info to all_rivir_proper to ')
        print('foggie.mocky_way.core_funcs.dict_rvir_proper before you can proceed :)')
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
    for delta_r_step in [delta_rad_coarse, delta_rad_fine, delta_rad_tiny]:
        rho_ratio = start_ratio
        print("Now in step of %s kpc..."%(delta_r_step))
        while rho_ratio > 200:
            r_previous = r_progressing
            r_progressing = r_progressing + ds.quan(delta_r_step, rad_units)
            rho_internal = mean_rho(ds, halo_center, r_progressing)
            rho_ratio = rho_internal/rho_crit
            print('Refine mean rho at r=%d kpc, rho/rho_200=%.1f'%(r_progressing, rho_ratio))
        r_progressing = r_previous

    r200 = r_progressing
    rho_ratio = mean_rho(ds, halo_center, r200)/rho_crit
    print('Find r200 (proper) =%.1f kpc, ratio=%.1f'%(r200, rho_ratio))

    return r200.in_units('kpc')

def find_galaxy_L_sun_phi_vecs(ds, halo_center):

    """
    Decide the angular momentum of the galaxy, and find the three unit vectors
    L_vec: angular momentum vector
    sun_vec: vector from galaxy center to observer at sun
    phi_vec: vec perpendicular to L and sun_vec following right handed rule

    History:
    10/04/2019: this chunk of code was originally inside prepdat, now separte it
                out to make code cleaner. YZ.

    """
    # tested with all-sky projection, pick the sphere radius that makes a flat projected plane
    # note that this need to be checked by eye to finally decide the value
    # go to the corresponding function to see the relevant values.
    from foggie.mocky_way.core_funcs import dict_sphere_for_gal_ang_mom(dd_name)
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

    return L_vec, sun_vec, phi_vec

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
    sphere_L_rr = {'nref11c_nref9f_selfshield_z6/RD0035': 8, # unit of kpc
                   'nref11c_nref9f_selfshield_z6/RD0036': 7,
                   'nref11c_nref9f_selfshield_z6/RD0037': 8,
                   'nref11c_nref9f_selfshield_z6/DD0946': 10}

    output_string = '%s/%s'%(sim_name, dd_name)
    if output_string in sphere_L_rr:
        this_sphere_L_rr = sphere_L_rr[output_string]
    else:
        print("The sphere r for %s ang mom has not been finallized yet, set it to 10 kpc for now."%(dd_name))
        this_sphere_L_rr = 10
    return this_sphere_L_rr
