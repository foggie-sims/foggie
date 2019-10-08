# Written by Raymond Simons, last updated 10/3/2019
# tools to identify satellites (clusters of stars) in the FOGGIE refine box
import foggie
from foggie.utils.get_run_loc_etc import get_run_loc_etc
from foggie.utils.get_refine_box import get_refine_box
from foggie.utils.get_halo_center import get_halo_center
from foggie.utils.get_proper_box_size import get_proper_box_size
import os
import argparse
import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
from foggie.utils.consistency import *
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


    args = parser.parse_args()
    return args



def save_profs(ds, sats_halo, profile_name):
    profs = {}

    for sat in sats_halo:   
        satx = sat['x']
        saty = sat['y']
        satz = sat['z']

        sat_center = ds.arr([satx, saty, satz], 'kpc')
        if args.do_sat_profiles:
            from yt.units import kpc
            sp = ds.sphere(center = sat_center, radius = 2*kpc)
            sp_cold = sp.cut_region(["(obj['temperature'] < {} )".format(1.5e4)])


            grid_prof_fields = [('gas', 'cell_mass'), \
                                ('deposit', 'stars_mass'), \
                                ('deposit', 'dm_mass')]
            vel_grid_prof_fields = [('gas', 'velocity_x'),
                                    ('gas', 'velocity_y'),
                                    ('gas', 'velocity_z')]


            prof = yt.create_profile(sp, ['radius'], fields = grid_prof_fields, weight_field = None, accumulation = True)
            prof_cold = yt.create_profile(sp_cold, ['radius'], fields = [('gas', 'cell_mass')], weight_field = None, accumulation = True)

            vel_prof = yt.create_profile(sp, ['radius'], fields = vel_grid_prof_fields, weight_field = ('gas', 'cell_mass'), accumulation = False)
            vel_prof_cold = yt.create_profile(sp_cold, ['radius'], fields = vel_grid_prof_fields, weight_field = ('gas', 'cell_mass'), accumulation = False)

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





            stars_vx = sp.quantities.weighted_average_quantity(('stars', 'particle_velocity_x'), ('stars', 'particle_mass')).to('km/s')
            stars_vy = sp.quantities.weighted_average_quantity(('stars', 'particle_velocity_y'), ('stars', 'particle_mass')).to('km/s')
            stars_vz = sp.quantities.weighted_average_quantity(('stars', 'particle_velocity_z'), ('stars', 'particle_mass')).to('km/s')

            profs[sat['id']]['stars_vx'] = stars_vx
            profs[sat['id']]['stars_vy'] = stars_vy
            profs[sat['id']]['stars_vz'] = stars_vz



            print (sat['id'])
            print ('\tgas_mass(Msun):', '%.2f'%log10(profs[sat['id']]['gas_mass'][-1]))
            print ('\tcold_gas_mass(Msun):', '%.2f'%log10(profs[sat['id']]['cold_gas_mass'][-1]))
            print ('\tdark_mass(Msun):', '%.2f'%log10(profs[sat['id']]['dm_mass'][-1]))
            print ('\tstars_mass(Msun):', '%.2f'%log10(profs[sat['id']]['stars_mass'][-1]))
            print ('\tstars_velx(km/s):', '%.2f'%(stars_vx))
            print ('\tgas_velx(km/s):', '%.2f'%(profs[sat['id']]['cold_gas_vx'][0]))


    np.save(profile_name, profs)

    return profs


if __name__ == '__main__':

    args = parse_args()
    print (args.system)
    #Run this in series on all of the halos
    if args.run_all:
        inputs = [('2878', 'DD0581'), 
                  ('5016', 'DD0581'), 
                  ('5036', 'DD0581'),
                  ('2392', 'DD0581'),
                  ('4123', 'DD0581'),
                  ('8508', 'DD0487')]
    else:
        inputs = [(args.halo, args.output),]


    for args.halo, args.output in inputs:

        foggie_dir, output_dir, run_loc, trackname, haloname, spectra_dir = get_run_loc_etc(args)

        save_dir = foggie_dir.replace('sims', 'outputs/identify_satellites')

        print (foggie_dir + run_loc)

        run_dir = foggie_dir + run_loc

        ds_loc = run_dir + args.output + "/" + args.output
        ds = yt.load(ds_loc)
        yt.add_particle_filter("stars",function=yt_fields._stars, filtered_type='all',requires=["particle_type"])
        yt.add_particle_filter("dm",function=yt_fields._dm, filtered_type='all',requires=["particle_type"])
        ds.add_particle_filter('stars')
        ds.add_particle_filter('dm')

        track = Table.read(trackname, format='ascii')
        track.sort('col1')
        zsnap = ds.get_parameter('CosmologyCurrentRedshift')

        refine_box, refine_box_center, x_width = get_refine_box(ds, zsnap, track)

        sat_cat = ascii.read(save_dir + '/satellite_locations.cat')
        sat_prop_cat = sat_cat.copy()

        sats_halo = sat_cat[(sat_cat['halo'] == int(args.halo)) & (sat_cat['run'] == args.run) &  (sat_cat['output'] == args.output)]


        profile_name = '{}/{}_sat_mass_profiles.npy'.format(save_dir, haloname)

        if args.do_sat_profiles: profs = save_profs(ds, sats_halo, profile_name)


        fig_dir = foggie_dir.replace('sims', 'figures/identify_satellites') 


        if args.do_measure_props:
            if not os.path.isfile(profile_name):
                print ('%s not found, aborting...'%profile_name)
                break
            else:
                print ('loading %s'%profile_name)

            profs = np.load(profile_name, allow_pickle = True)[()]
            plt.close('all')
            fig, ax = plt.subplots(1,1, figsize = (6,6))
            for sat in sats_halo:   
                prof = profs[sat['id']]
                r = profs[sat['id']]['radius']
                m = profs[sat['id']]['stars_mass']
                gd = where(r < 2)[0]

                diff = abs(m - m[gd][-1]/2.)
                re = r[argmin(diff)]
                print (haloname, sat['id'], sat['selectid'], '%.3f %.3f'%(log10(m[gd][-1]), re))


                ax.plot(log10(m[gd][-1]), re, 'k.')


            ax.set_xlabel(r'M$_*$ (M$_{\odot}$)')
            ax.set_ylabel(r'r$_e$ (kpc)')
            ax.set_yscale('log')
            ax.set_ylim(5e-2, 10)
            fig.savefig(fig_dir + '%s_mass_size.png'%haloname, dpi = 300)























