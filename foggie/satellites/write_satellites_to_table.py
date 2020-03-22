# Written by Raymond Simons, last updated 10/4/2019
# write satellites output to table
import foggie
from foggie.utils.get_run_loc_etc import get_run_loc_etc
import argparse
import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
from foggie.utils.consistency import *
from foggie.utils import yt_fields
from scipy.signal import find_peaks  
from foggie.utils.foggie_load import *

import yt
from numpy import *
import glob
from glob import glob
from astropy.table import Table
from foggie.utils.foggie_load import *

from astropy.io import ascii
import os
import string
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

    parser.add_argument('--save_dir', metavar='save_dir', type=str, action='store',
                        help='directory to save products')
    parser.set_defaults(save_dir="~/")

    parser.add_argument('--run_all', dest='run_all', action='store_true',
                        help='just use the pwd?, default is no')
    parser.set_defaults(pwd=False)

    parser.add_argument('--use_halo_c_v', dest='use_halo_c_v', action='store_true',
                        help='just use the pwd?, default is no')
    parser.set_defaults(use_halo_c_v=False)

    args = parser.parse_args()
    return args




def load_sim(args):
    foggie_dir, output_dir, run_loc, trackname, haloname, spectra_dir, infofile = get_run_loc_etc(args)
    track_dir =  trackname.split('halo_tracks')[0]   + 'halo_infos/00' + args.halo + '/' + args.run + '/'
    snap_name = foggie_dir + run_loc + args.output + '/' + args.output
    ds, refine_box, refine_box_center, refine_width = load(snap = snap_name, 
                                                           trackfile = trackname, 
                                                           use_halo_c_v=args.use_halo_c_v, 
                                                           halo_c_v_name=track_dir + 'halo_c_v')

    refine_box.set_field_parameter('center', ds.arr(ds.halo_center_kpc, 'kpc'))
    bulk_vel = refine_box.quantities.bulk_velocity()
    refine_box.set_field_parameter("bulk_velocity", bulk_vel)

    return ds, refine_box, refine_box_center, refine_width







if __name__ == '__main__':

  args = parse_args()
  print (args.system)

  if args.run_all:

    inputs = [('2878', 'DD0581'), 
              ('5016', 'DD0581'), 
              ('5036', 'DD0581'),
              ('2392', 'DD0581'),
              ('4123', 'DD0581'),
              ('8508', 'DD0487')]

  else:
    inputs = [(args.halo, args.output),]

  all_halos= []
  all_runs = []
  all_outputs = []
  all_x     = []
  all_y = []
  all_z = []

  all_com_x     = []
  all_com_y = []
  all_com_z = []


  all_id = []
  all_names = []
  all_dist_center = []
  ignore_list = np.loadtxt('ignore_list', dtype = 'str')
  abc_list = list(string.ascii_lowercase) 



  for args.halo, args.output in inputs:


      halos = []
      runs = []
      outputs = []
      ids = []
      x = []
      y = []
      z = []
      com_x = []
      com_y = []
      com_z = []
      names = []
      dist_center = []

      ds, refine_box, refine_box_center, refine_width = load_sim(args)



      sat_catalog = args.save_dir + '/%s_%s_%s_satellite_selection.npy'%(args.run, args.halo, args.output)




      cat = np.load(sat_catalog, allow_pickle = True)[()]
      id_sat_counter = 0
      for cc, c in enumerate(cat):
        print (args.halo, c['selectid'])
        sat_center = ds.arr([c['x'], c['y'], c['z']])
        diff = sqrt(sum((sat_center - ds.halo_center_kpc)**2.)) 

        if diff < 0.3: 
          print (args.halo, diff)
          id_sat = -1
          halos.insert(0, '00%s'%args.halo)
          runs.insert(0, args.run)
          outputs.insert(0, args.output)
          ids.insert(0, '0')
          x.insert(0, float(c['x'].value))
          y.insert(0, float(c['y'].value))
          z.insert(0, float(c['z'].value))

          com_x.insert(0, float(c['x'].value))
          com_y.insert(0, float(c['y'].value))
          com_z.insert(0, float(c['z'].value))


          names.insert(0, c['selectid'])
          dist_center.insert(0, diff)

        elif ('{}_{}'.format(args.halo,c['selectid']) not in ignore_list):
          id_sat = id_sat_counter
          id_sat_counter+=1

          halos.append('00%s'%args.halo)
          runs.append(args.run)
          outputs.append(args.output)
          if id_sat_counter > len(abc_list):
            ids.append('a'+abc_list[id_sat])
  
          else:              
            ids.append(abc_list[id_sat])
          x_select = float(c['x'].value)
          y_select = float(c['y'].value)
          z_select = float(c['z'].value)
          x.append(x_select)
          y.append(y_select)
          z.append(z_select)
          names.append(c['selectid'])
          dist_center.append(diff)
          from yt.units import kpc

          sat_center = ds.arr([x_select, y_select, z_select], 'kpc')
          sp = ds.sphere(center = sat_center, radius = 1*kpc)

          com = sp.quantities.center_of_mass(use_gas=False, use_particles=True, particle_type = 'stars').to('kpc')
          com_x.append(round(float(com[0].value), 3))
          com_y.append(round(float(com[1].value), 3))
          com_z.append(round(float(com[2].value), 3))

        else:
          print ('\t', args.halo, c['selectid'])





      all_halos += halos
      all_runs += runs
      all_outputs += outputs
      all_id += ids
      all_x += x
      all_y += y
      all_z +=z
      all_com_x += com_x
      all_com_y += com_y
      all_com_z +=com_z

      all_names+=names
      all_dist_center+=dist_center






  t = Table([all_halos,all_runs, all_outputs, all_id, all_x, all_y, all_z, all_com_x, all_com_y, all_com_z, all_names, all_dist_center], \
            names = ['halo', 'run', 'output', 'id', 'x_select', 'y_select', 'z_select', 'x', 'y', 'z', 'selectid', 'distance_halo'])

  t.meta['comments'] = ['halo: halo ID']
  t.meta['comments'].append('run: foggie run type')
  t.meta['comments'].append('output: DD output name')
  t.meta['comments'].append('x_select: satellite x position, from initial selection (kpc)')
  t.meta['comments'].append('y_select: satellite y position, from initial selection (kpc)')
  t.meta['comments'].append('z_select: satellite z position, from initial selection (kpc)')
  t.meta['comments'].append('x: center-of-mass x position (kpc)')
  t.meta['comments'].append('y: center-of-mass y position (kpc)')
  t.meta['comments'].append('z: center-of-mass z position (kpc)')
  t.meta['comments'].append('selectid: identifier in raw seg catalogs')
  t.meta['comments'].append('distance_halo: distance from center of halo')

  ascii.write(t, args.save_dir + '/%s_%s_%s_satellite_locations.cat'%(args.run, args.halo, args.output), format = 'commented_header', overwrite = True)





















































