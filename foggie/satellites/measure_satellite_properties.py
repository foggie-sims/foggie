# Written by Raymond Simons, last updated 10/8/2019
# tools to identify satellites (clusters of stars) in the FOGGIE refine box
import foggie
from foggie.utils.get_run_loc_etc import get_run_loc_etc
from foggie.utils.get_refine_box import get_refine_box
from foggie.utils.get_halo_center import get_halo_center
from foggie.utils.get_proper_box_size import get_proper_box_size
import os
import argparse
import numpy as np
from astropy.table import Table, Column
import matplotlib.pyplot as plt
from foggie.utils.consistency import *
#from foggie.utils.foggie_load import *
from foggie.utils.foggie_load import *
from foggie.utils import yt_fields
from scipy.signal import find_peaks  
import yt
from numpy import *
from photutils.segmentation import detect_sources
from astropy.io import ascii
import time

plt.ioff()
def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='''identify satellites in FOGGIE simulations''')
    parser.add_argument('-system', '--system', metavar='system', type=str, action='store', \
                        help='Which system are you on? Default is Jase')
    parser.set_defaults(system="jase")

    parser.add_argument('--run', metavar='run', type=str, action='store',
                        help='which run? default is natural')
    parser.set_defaults(run="nref11c_nref9f")

    parser.add_argument('--halo', metavar='halo', type=str, action='store',
                        help='which halo? default is 8508 (Tempest)')
    parser.set_defaults(halo="8508")

    parser.add_argument('--pwd', dest='pwd', action='store_true',
                        help='just use the pwd?, default is no')
    parser.set_defaults(pwd=False)

    parser.add_argument('--run_all', dest='run_all', action='store_true',
                        help='just use the pwd?, default is no')
    parser.set_defaults(pwd=False)

    parser.add_argument('--do_sat_proj_plots', dest='do_sat_proj_plots', action='store_true',
                        help='just use the pwd?, default is no')
    parser.set_defaults(pwd=False)

    parser.add_argument('--do_proj_plots', dest='do_proj_plots', action='store_true',
                        help='just use the pwd?, default is no')
    parser.set_defaults(pwd=False)

    parser.add_argument('--do_sat_profiles', dest='do_sat_profiles', action='store_true',
                        help='just use the pwd?, default is no')
    parser.set_defaults(pwd=False)

    parser.add_argument('--do_measure_props', dest='do_measure_props', action='store_true',
                        help='just use the pwd?, default is no')
    parser.set_defaults(pwd=False)


    parser.add_argument('--output', metavar='output', type=str, action='store',
                        help='which output? default is RD0020')
    parser.set_defaults(output="RD0020")

    parser.add_argument('--save_dir', metavar='save_dir', type=str, action='store',
                        help='directory to save products')
    parser.set_defaults(save_dir="~/Desktop")

    parser.add_argument('--use_halo_c_v', dest='use_halo_c_v', action='store_true',
                        help='just use the pwd?, default is no')
    parser.set_defaults(use_halo_c_v=False)




    args = parser.parse_args()
    return args



def save_profs(ds, args, sat_cat, profile_name, n_bins = 100):
    profs = {}

    for sat in sat_cat:   
        if (sat['halo'] == int(args.halo))\
            & (sat['run'] == args.run) \
            &  (sat['output'] == args.output):

            satx = sat['x']
            saty = sat['y']
            satz = sat['z']

            sat_center = ds.arr([satx, saty, satz], 'kpc')
            if args.do_sat_profiles:
                from yt.units import kpc
                sp = ds.sphere(center = sat_center, radius = 5*kpc)
                sp_cold = sp.cut_region(["(obj['temperature'] < {} )".format(1.5e4)])


                grid_prof_fields = [('gas', 'cell_mass'), \
                                    ('deposit', 'stars_mass'), \
                                    ('deposit', 'dm_mass')]
                vel_grid_prof_fields = [('gas', 'velocity_x'),
                                        ('gas', 'velocity_y'),
                                        ('gas', 'velocity_z')]

                prof = yt.create_profile(sp, ['radius'], fields = grid_prof_fields, n_bins = n_bins, weight_field = None, accumulation = True)
                prof_cold = yt.create_profile(sp_cold, ['radius'], fields = [('gas', 'cell_mass')], n_bins = n_bins, weight_field = None, accumulation = True)

                vel_prof = yt.create_profile(sp, ['radius'], fields = vel_grid_prof_fields, n_bins = n_bins, weight_field = ('gas', 'cell_mass'), accumulation = False)
                vel_prof_cold = yt.create_profile(sp_cold, ['radius'], fields = vel_grid_prof_fields, n_bins = n_bins, weight_field = ('gas', 'cell_mass'), accumulation = False)

                profs[sat['id']] = {}
                profs[sat['id']]['selectid'] = sat['selectid']                
                profs[sat['id']]['radius'] = prof.x.to('kpc')
                profs[sat['id']]['radius_cold'] = prof_cold.x.to('kpc')
                profs[sat['id']]['gas_mass'] = prof.field_data[('gas', 'cell_mass')].to('Msun')
                profs[sat['id']]['cold_gas_mass'] = prof_cold.field_data[('gas', 'cell_mass')].to('Msun')
                profs[sat['id']]['dm_mass'] = prof.field_data[('deposit', 'dm_mass')].to('Msun')
                profs[sat['id']]['stars_mass'] = prof.field_data[('deposit', 'stars_mass')].to('Msun')

                for orient in ['x', 'y', 'z']:
                    profs[sat['id']]['gas_v%s'%orient] = vel_prof.field_data[('gas', 'velocity_%s'%orient)].to('km/s')
                    profs[sat['id']]['cold_gas_v%s'%orient] = vel_prof_cold.field_data[('gas', 'velocity_%s'%orient)].to('km/s')



                sp_stars = ds.sphere(center = sat_center, radius = 0.5*kpc)


                stars_vx = sp_stars.quantities.weighted_average_quantity(('stars', 'particle_velocity_x'), ('stars', 'particle_mass')).to('km/s')
                stars_vy = sp_stars.quantities.weighted_average_quantity(('stars', 'particle_velocity_y'), ('stars', 'particle_mass')).to('km/s')
                stars_vz = sp_stars.quantities.weighted_average_quantity(('stars', 'particle_velocity_z'), ('stars', 'particle_mass')).to('km/s')

                profs[sat['id']]['stars_vx'] = stars_vx
                profs[sat['id']]['stars_vy'] = stars_vy
                profs[sat['id']]['stars_vz'] = stars_vz



                print (sat['id'])
                print ('\tgas_mass(Msun):', '%.2f'%log10(profs[sat['id']]['gas_mass'][-1]))
                print ('\tcold_gas_mass(Msun):', '%.2f'%log10(profs[sat['id']]['cold_gas_mass'][-1]))
                print ('\tdark_mass(Msun):', '%.2f'%log10(profs[sat['id']]['dm_mass'][-1]))
                print ('\tstars_mass(Msun):', '%.2f'%(profs[sat['id']]['stars_mass'][-1]/(1.e10)))
                print ('\tstars_velx(km/s):', '%.2f'%(stars_vx))
                print ('\tgas_velx(km/s):', '%.2f'%(profs[sat['id']]['cold_gas_vx'][0]))


    np.save(profile_name, profs)

    return profs







if __name__ == '__main__':

    args = parse_args()
    print (args.system)
    #Run this in series on all of the halos
    if args.run_all:
        inputs = [('2392', 'DD0581'),
                  ('2878', 'DD0581'), 
                  ('4123', 'DD0581'),
                  ('5016', 'DD0581'), 
                  ('5036', 'DD0581'),
                  ('8508', 'DD0487')]
    else:
        inputs = [(args.halo, args.output),]


    # we want to load in the catalogs directory before reading in any specific halo
    # need to manually inset the directory in here for the time-being
    


    sat_cat = ascii.read(args.save_dir + '/%s_%s_%s_satellite_locations.cat'%(args.run, args.halo, args.output))
    if args.do_measure_props:
        sat_prop_cat = sat_cat.copy()

        sat_prop_cat.add_column(Column(name = 're_stars', data = np.zeros(len(sat_prop_cat))*np.nan))
        sat_prop_cat.add_column(Column(name = 're_gas', data = np.zeros(len(sat_prop_cat))*np.nan))
        sat_prop_cat.add_column(Column(name = 're_cold_gas', data = np.zeros(len(sat_prop_cat))*np.nan))

        components = ['stars', 'gas', 'cold_gas', 'dm']
        radii = ['2kpc', 're_s', '2re_s', 're_g', '2re_g', 're_cg', '2re_cg']

        radii_v = radii.copy()
        radii_v[radii == '2kpc'] = 'cen'


        for comp in components:
            for rad in radii:
                sat_prop_cat.add_column(Column(name = 'm_%s_%s'%(comp, rad), data = np.zeros(len(sat_prop_cat))*np.nan))

            if ((comp == 'dm') | (comp == 'stars')): continue                
            for rad in radii_v:
                sat_prop_cat.add_column(Column(name = 'v_%s_%s'%(comp, rad), data = np.zeros(len(sat_prop_cat))*np.nan))



    for args.halo, args.output in inputs:
        ds, refine_box = load_sim(args)



        profile_name = '{}/{}_{}_{}_sat_mass_profiles.npy'.format(args.save_dir, args.run, args.halo, args.output)

        if args.do_sat_profiles: profs = save_profs(ds, args, sat_cat, profile_name)


        fig_dir = args.save_dir

        if args.do_measure_props:
            if not os.path.isfile(profile_name):
                print ('%s not found, aborting...'%profile_name)
                break
            else:
                print ('loading %s'%profile_name)

            profs = np.load(profile_name, allow_pickle = True)[()]
            for sat in sat_prop_cat:  

                if (sat['halo'] == int(args.halo))\
                    & (sat['run'] == args.run) \
                    &  (sat['output'] == args.output):


                    prof = profs[sat['id']]

                    prof_stars_vx = prof['stars_vx']
                    prof_stars_vy = prof['stars_vy']
                    prof_stars_vz = prof['stars_vz']

                    r = prof['radius']
                    r_cold =  prof['radius_cold']
                    m_stars = prof['stars_mass']
                    m_gas = prof['gas_mass']
                    m_cold_gas = prof['cold_gas_mass']

                    


                    if sat['id'] is not '0':
                        gd = where(r < 2)[0]
                    else: gd = where(r<np.inf)[0]

                    

                    diff_s = abs(m_stars - m_stars[gd][-1]/2.)
                    diff_g = abs(m_gas - m_gas[gd][-1]/2.)
                    diff_cg = abs(m_cold_gas - m_cold_gas[gd][-1]/2.)

                    re_s_index = argmin(diff_s)
                    re_g_index = argmin(diff_g)
                    re_cg_index = argmin(diff_g)

                    re_stars = r[re_s_index]
                    re_gas = r[re_g_index]
                    re_cold_gas = r_cold[re_cg_index]

                    twore_s_index = argmin(abs(r - 2*re_stars))
                    twore_g_index = argmin(abs(r - 2*re_gas))
                    twore_cg_index = argmin(abs(r_cold - 2*re_cold_gas))




                    sat['re_stars'] = round(float(re_stars.value), 3)
                    sat['re_gas'] = round(float(re_gas.value), 3)
                    sat['re_cold_gas'] = round(float(re_cold_gas.value), 3)



                    for comp in components:

                        mass_prof = prof['%s_mass'%comp]

                        for rad in radii:
                            if rad == '2kpc': index = [gd[-1]]
                            elif rad == 're_s': index = re_s_index
                            elif rad == '2re_s': index = twore_s_index
                            elif rad == 're_g': index = re_g_index
                            elif rad == '2re_g': index = twore_g_index
                            elif rad == 're_cg': index = re_cg_index
                            elif rad == '2re_cg': index = twore_cg_index

                            sat['m_%s_%s'%(comp, rad)] =   round(float(mass_prof[index].value) * 1.e-8, 5)

                    for comp in components:
                        if ((comp == 'dm') | (comp == 'stars')): continue

                        velx_prof = prof['%s_vx'%comp]
                        vely_prof = prof['%s_vy'%comp]
                        velz_prof = prof['%s_vz'%comp]



                        diff_v =     sqrt((prof_stars_vx - velx_prof)**2. \
                                        + (prof_stars_vy - vely_prof)**2. \
                                        + (prof_stars_vz - velz_prof)**2.)

                        for rad in radii_v:
                            if rad == 'center': index = [0]
                            elif rad == 're_s': index = re_s_index
                            elif rad == '2re_s': index = twore_s_index
                            elif rad == 're_g': index = re_g_index
                            elif rad == '2re_g': index = twore_g_index
                            elif rad == 're_cg': index = re_cg_index
                            elif rad == '2re_cg': index = twore_cg_index

                            sat['v_%s_%s'%(comp, rad)] = round(float(diff_v[index].value), 2)




    if args.do_measure_props:
        sat_prop_cat.meta['comments'].append('re_stars: effective radius using stars (kpc)')
        sat_prop_cat.meta['comments'].append('re_gas: effective radius using gas (kpc)')
        sat_prop_cat.meta['comments'].append('m_stars_2kpc: "total" stellar mass, mass inside 2kpc (1.e8 Msun)')
        sat_prop_cat.meta['comments'].append('m_stars_re_s: stellar mass inside re_stars (1.e8 Msun)')
        sat_prop_cat.meta['comments'].append('m_stars_re_g: stellar mass inside re_gas (1.e8 Msun)')
        sat_prop_cat.meta['comments'].append('m_stars_2re_s: stellar mass inside 2 x re_stars (1.e8 Msun)')
        sat_prop_cat.meta['comments'].append('m_stars_2re_g: stellar mass inside 2 x re_gas (1.e8 Msun)')
        sat_prop_cat.meta['comments'].append('etc. for rest of mass components')
        sat_prop_cat.meta['comments'].append('v_gas_center: difference between gas and star velocities at the center (km/s)')
        sat_prop_cat.meta['comments'].append('v_gas_re_s: difference between gas and star velocities at re_stars (km/s)')
        sat_prop_cat.meta['comments'].append('etc. for rest of velocity components')


        ascii.write(sat_prop_cat, args.save_dir + '/%s_%s_%s_satellite_properties.cat'%(args.run, args.halo, args.output), format = 'commented_header', overwrite = True)

















