#!/u/rcsimons/miniconda3/bin/python3.7
import astropy
from astropy.io import fits
from astropy.table import Table, Column
import numpy as np
from numpy import *
import math
from joblib import Parallel, delayed
import os, sys, argparse
import yt
from numpy import rec
from astropy.io import ascii
import foggie
from foggie.utils.foggie_utils import filter_particles
from foggie.utils.get_run_loc_etc import get_run_loc_etc
from foggie.utils.get_refine_box import get_refine_box
from foggie.utils.get_halo_center import get_halo_center
from foggie.utils.get_proper_box_size import get_proper_box_size
from foggie.utils import yt_fields


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='''identify satellites in FOGGIE simulations''')
    parser.add_argument('-system', '--system', metavar='system', type=str, action='store', \
                        help='Which system are you on? Default is Jase')
    parser.set_defaults(system="pleiades_raymond")

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
    parser.set_defaults(output="DD0487")


    args = parser.parse_args()
    return args




def run_tracker(args, anchor_ids, sat, temp_outdir, id_all, x_all, y_all, z_all):
    print ('tracking %s %s'%(args.halo, sat))
    gd_indices = []
    print ('\t', args.halo, sat, 'finding anchor stars..')
    for a, anchor_id in enumerate(anchor_ids[:1000]):
      match = where(id_all == anchor_id)[0]
      if len(match) > 0: gd_indices.append(int(match))

    x_anchors =  x_all[gd_indices]
    y_anchors =  y_all[gd_indices]
    z_anchors =  z_all[gd_indices]


    x_sat = nanmedian(x_anchors)
    y_sat = nanmedian(y_anchors)
    z_sat = nanmedian(z_anchors)
    print ('\t found location for %s %s: (%.3f, %.3f, %.3f)'%(args.halo, sat, x_sat, y_sat, z_sat))

    np.save(temp_outdir + '/' + args.halo + '_' + args.output + '_' + sat + '.npy', np.array([x_sat, y_sat, z_sat]))

    return 

if __name__ == '__main__':
    args = parse_args()


    foggie_dir, output_dir, run_loc, trackname, haloname, spectra_dir = get_run_loc_etc(args)

    output_path = output_dir[0:output_dir.find('foggie')] + 'foggie'
    cat_dir = output_path + '/catalogs'
    fig_dir = output_path +  '/figures/track_satellites'
    temp_outdir = cat_dir + '/sat_track_locations/temp'


    if not os.path.isdir(cat_dir): os.system('mkdir ' + cat_dir) 
    if not os.path.isdir(fig_dir): os.system('mkdir ' + fig_dir) 
    if not os.path.isdir(cat_dir + '/sat_track_locations'): os.system('mkdir ' + cat_dir + '/sat_track_locations')
    if not os.path.isdir(cat_dir + '/sat_track_locations/temp'): os.system('mkdir ' + cat_dir + '/sat_track_locations/temp')


    sat_cat = ascii.read(cat_dir + '/satellite_properties.cat')
    anchors = np.load(cat_dir + '/anchors.npy', allow_pickle = True)[()]
    anchors = anchors[args.halo]

    run_dir = foggie_dir + run_loc

    ds_loc = run_dir + args.output + "/" + args.output
    ds = yt.load(ds_loc)

    track = Table.read(trackname, format='ascii')
    track.sort('col1')
    zsnap = ds.get_parameter('CosmologyCurrentRedshift')

    refine_box, refine_box_center, x_width = get_refine_box(ds, zsnap, track)

    load_particle_fields = ['particle_index',\
                            'particle_position_x', 'particle_position_y', 'particle_position_z']
    filter_particles(refine_box, load_particles = True, load_particle_types = ['stars'], load_particle_fields = load_particle_fields)
    print ('particles loaded')




    # Need to pass in these arrays, can't pass in refine box
    id_all = refine_box['stars', 'particle_index']
    x_all = refine_box['stars', 'particle_position_x'].to('kpc')
    y_all = refine_box['stars', 'particle_position_y'].to('kpc')
    z_all = refine_box['stars', 'particle_position_z'].to('kpc')

    Parallel(n_jobs = -1)(delayed(run_tracker)(args, anchors[sat]['ids'], sat, temp_outdir, id_all, x_all, y_all, z_all) for sat in anchors.keys())



    #Collect outputs      
    output = {}
    annotate_others = []

    for sat in anchors.keys():

      temp = np.load(temp_outdir + '/' + args.halo + '_' + args.output + '_' + sat + '.npy')
      output[sat] = {}

      from yt.units import kpc
      sp = ds.sphere(center = ds.arr(temp, 'kpc'), radius = 1*kpc)
      com = sp.quantities.center_of_mass(use_gas=False, use_particles=True, particle_type = 'stars').to('kpc')
      output[sat]['x'] = round(float(com[0].value), 3)
      output[sat]['y'] = round(float(com[1].value), 3)
      output[sat]['z'] = round(float(com[2].value), 3)

      # Make individual satellite projection plots

      from foggie.satellites.make_satellite_projections import make_projection_plots
      sat_center = ds.arr([output[sat]['x'], output[sat]['y'], output[sat]['z']], 'kpc')

      if sat == '0': halo_center = sat_center.copy()

      fig_width = 10 * kpc

      make_projection_plots(refine_box.ds, sat_center, refine_box, fig_width, fig_dir, haloname, \
                          fig_end = 'satellite_{}_{}_{}'.format(args.halo, args.output, sat), \
                          do = ['gas', 'stars'], axes = ['x'],  annotate_center = True, annotate_others = [],\
                          add_velocity = False)        


    for sat in anchors.keys(): annotate_others.append(ds.arr([output[sat]['x'], output[sat]['y'], output[sat]['z']], 'kpc'))

    make_projection_plots(refine_box.ds, halo_center, refine_box, x_width, fig_dir, haloname,\
                          fig_end = 'central',\
                          do = ['gas', 'stars'], axes = ['x'],\
                          annotate_center = True, annotate_others = annotate_others, is_central = True)


    np.save('%s/%s_%s.npy'%(temp_outdir.replace('/temp', ''), args.halo, args.output), output)

    os.system('rm %s/%s_%s_*.npy'%(temp_outdir, args.halo, args.output))


























