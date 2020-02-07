'''
Filename: get_rvir.py
Author: Raymond
Created: 01-16-19
Last modified:  01-16-19

This file calculates the fraction of the refine box volume that has its cooling length resolved.
'''
import glob
from glob import glob
import yt
from astropy.io import ascii
from yt.units import kpc
from foggie.utils.get_run_loc_etc import get_run_loc_etc
import foggie
from foggie.utils.foggie_utils import filter_particles
import matplotlib.pyplot as plt
from astropy.cosmology import Planck15 as cosmo
import numpy as np
from numpy import *
import argparse
from foggie.utils.foggie_load import *
from foggie.utils.get_refine_box import get_refine_box
from foggie.utils.get_region import get_region
from foggie.utils.consistency import cgm_inner_radius, cgm_outer_radius, cgm_field_filter, ism_field_filter



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

    parser.add_argument('--output', metavar='output', type=str, action='store',
                        help='which output? default is RD0020')
    parser.set_defaults(output="RD0020")


    args = parser.parse_args()
    return args


if __name__ == '__main__':


  args = parse_args()
  inputs = [('2392', 'DD0581'),
            ('2878', 'DD0581'), 
            ('4123', 'DD0581'),
            ('5016', 'DD0581'), 
            ('5036', 'DD0581'),
            ('8508', 'DD0487')]

  for (args.halo, args.output) in inputs[:]:


    foggie_dir, output_dir, run_loc, trackname, haloname, spectra_dir, infofile = get_run_loc_etc(args)
    #code_path = trackname.split('halo_tracks')[0]  
    #track_dir = code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/'
    #halo_c_v_name = track_dir + 'halo_c_v'

    #snap_name = foggie_dir + run_loc + args.output + '/' + args.output
    #ds, refine_box, refine_box_center, refine_width = load(snap_name, trackname, use_halo_c_v=args.use_halo_c_v, halo_c_v_name=halo_c_v_name)
    run_loc = run_loc.replace('nref11n', 'natural')
    run_dir = foggie_dir + run_loc

    ds_loc = run_dir + args.output + "/" + args.output
    ds = yt.load(ds_loc)
    track = Table.read(trackname, format='ascii')
    track.sort('col1')
    zsnap = ds.get_parameter('CosmologyCurrentRedshift')

    refine_box, refine_box_center, x_width = get_refine_box(ds, zsnap, track)
    #refine_box = get_region(ds, 'cgm')

    sound_speed = refine_box['gas', 'sound_speed']
    cooling_time = refine_box['gas', 'cooling_time']
    #cell_index = refine_box['index', 'grid_level']
    cell_volume = refine_box['index', 'cell_volume']
    cell_mass = refine_box['gas', 'cell_mass']

    cell_size = cell_volume.to('cm**3')**(1/3.)
    cooling_length = (sound_speed * cooling_time).to('cm')


    total_volume = sum(cell_volume).to('kpc**3.')    
    total_mass   = sum(cell_mass).to('Msun')    
    
    resolved = where(cell_size/cooling_length < 1.)[0]
    unresolved = where(cell_size/cooling_length > 1.)[0]

    resolved_volume = sum(cell_volume[resolved]).to('kpc**3.')
    unresolved_volume = sum(cell_volume[unresolved]).to('kpc**3.')


    resolved_mass  = sum(cell_mass[resolved]).to('Msun')
    unresolved_mass = sum(cell_mass[unresolved]).to('Msun')

    f_vresolved = resolved_volume/total_volume
    f_mresolved = resolved_mass/total_mass

    print(args.halo, '%.2f %.2f %.2f %.5f %.5f'%(unresolved_volume, resolved_volume, total_volume,f_vresolved, f_mresolved))




    cen_sphere = ds.sphere(refine_box_center, (cgm_inner_radius, "kpc"))  #<--using box center from the trackfile above 
    rvir_sphere = ds.sphere(refine_box_center, (cgm_outer_radius, 'kpc')) 
    cgm = refine_box - cen_sphere
    refine_box = cgm.cut_region(cgm_field_filter)   #<---- cgm_field_filter is from consistency.py 


    sound_speed = refine_box['gas', 'sound_speed']
    cooling_time = refine_box['gas', 'cooling_time']
    #cell_index = refine_box['index', 'grid_level']
    cell_volume = refine_box['index', 'cell_volume']
    cell_mass = refine_box['gas', 'cell_mass']

    cell_size = cell_volume.to('cm**3')**(1/3.)
    cooling_length = (sound_speed * cooling_time).to('cm')


    total_volume = sum(cell_volume).to('kpc**3.')    
    total_mass   = sum(cell_mass).to('Msun')    
    
    resolved = where(cell_size/cooling_length < 1.)[0]
    unresolved = where(cell_size/cooling_length > 1.)[0]

    resolved_volume = sum(cell_volume[resolved]).to('kpc**3.')
    unresolved_volume = sum(cell_volume[unresolved]).to('kpc**3.')


    resolved_mass  = sum(cell_mass[resolved]).to('Msun')
    unresolved_mass = sum(cell_mass[unresolved]).to('Msun')

    f_vresolved = resolved_volume/total_volume
    f_mresolved = resolved_mass/total_mass

    print(args.halo, '%.2f %.2f %.2f %.3f %.3f'%(unresolved_volume, resolved_volume, total_volume,f_vresolved, f_mresolved))





























