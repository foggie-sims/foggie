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
import yt
from numpy import *
import glob
from glob import glob
from astropy.table import Table
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


    args = parser.parse_args()
    return args





if __name__ == '__main__':

  args = parse_args()
  print (args.system)


  inputs = [('2878', 'DD0581'), 
            ('5016', 'DD0581'), 
            ('5036', 'DD0581'),
            ('2392', 'DD0581'),
            ('4123', 'DD0581'),
            ('8508', 'DD0487')]


  all_halos= []
  all_runs = []
  all_outputs = []
  all_x     = []
  all_y = []
  all_z = []
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
      names = []
      dist_center = []


      foggie_dir, output_dir, run_loc, trackname, haloname, spectra_dir = get_run_loc_etc(args)
      save_dir = foggie_dir.replace('sims', 'outputs/identify_satellites')

      sat_catalog = glob(save_dir + '/satellite_selection_%s.npy'%haloname)[0]

      center_file = foggie_dir.replace('/sims/', '/outputs/centers/{}_{}.npy'.format(haloname, args.output))
      run_dir = foggie_dir + run_loc

      ds_loc = run_dir + args.output + "/" + args.output
      ds = yt.load(ds_loc)

      halo_center = np.load(center_file)
      halo_center_kpc = ds.arr(halo_center, 'code_length').to('kpc')

      cat = np.load(sat_catalog, allow_pickle = True)[()]
      id_sat_counter = 0
      for cc, c in enumerate(cat):
        print (args.halo, c['selectid'])
        sat_center = ds.arr([c['x'], c['y'], c['z']])
        diff = sqrt(sum((sat_center - halo_center_kpc)**2.)) 

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
          x.append(float(c['x'].value))
          y.append(float(c['y'].value))
          z.append(float(c['z'].value))
          names.append(c['selectid'])
          dist_center.append(diff)

        else:
          print ('\t', args.halo, c['selectid'])





      all_halos += halos
      all_runs += runs
      all_outputs += outputs
      all_id += ids
      all_x += x
      all_y += y
      all_z +=z
      all_names+=names
      all_dist_center+=dist_center


  t = Table([all_halos,all_runs, all_outputs, all_id, all_x, all_y, all_z, all_names, all_dist_center], names = ['halo', 'run', 'output', 'id', 'x_select', 'y_select', 'z_select', 'selectid', 'distance_halo'])

  t.meta['comments'] = ['halo: halo ID']
  t.meta['comments'].append('run: foggie run type')
  t.meta['comments'].append('output: DD output name')
  t.meta['comments'].append('x_select: satellite x position, from initial selection (kpc)')
  t.meta['comments'].append('y_select: satellite y position, from initial selection (kpc)')
  t.meta['comments'].append('z_select: satellite z position, from initial selection (kpc)')
  t.meta['comments'].append('selectid: identifier in raw seg catalogs')
  t.meta['comments'].append('distance_halo: distance from center of halo')

  ascii.write(t, save_dir + '/satellite_locations.cat', format = 'commented_header', overwrite = True)





















































