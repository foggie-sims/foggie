# Written by Raymond Simons, last updated 3/23/2020
import yt
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
from foggie.utils.foggie_utils import filter_particles
from foggie.utils.consistency import *
from foggie.utils import yt_fields
from scipy.signal import find_peaks  
from numpy import *
from photutils.segmentation import detect_sources
from astropy.io import ascii
from foggie.utils.foggie_utils import filter_particles
import PIL
from PIL import Image
from glob import glob
from joblib import Parallel, delayed
import matplotlib.colors as colors
from yt.units import kpc
import matplotlib
import multiprocessing as mp
import os
from foggie.utils.foggie_load import *
from foggie.utils.get_run_loc_etc import get_run_loc_etc

os.environ['KMP_DUPLICATE_LIB_OK']='True'
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
    parser.set_defaults(run_all=False)

    parser.add_argument('--do_central', dest='do_central', action='store_true',
                        help='just use the pwd?, default is no')
    parser.set_defaults(do_central=False)

    parser.add_argument('--output', metavar='output', type=str, action='store',
                        help='which output? default is RD0020')
    parser.set_defaults(output="RD0020")

    parser.add_argument('--rot_n', metavar='rot_n', type=int, action='store',
                        help='which output? default is RD0020')
    parser.set_defaults(rot_n=np.nan)


    args = parser.parse_args()
    return args


def do_plot(ds, field, normal_vector, north_vector, annotate_positions, \
                box_proj, center, \
                cmap, unit = None, \
                 ann_sphere_rad = (1, 'kpc'), weight_field = None, zmin = None, zmax = None, proj_width = 0):
    print (proj_width)
    prj = yt.OffAxisProjectionPlot(ds, normal_vector, field, north_vector = north_vector, center = center, width = proj_width, data_source = box_proj, weight_field = weight_field)


    prj.set_unit(field, unit)
    prj.set_zlim(field, zmin = zmin, zmax =  zmax)
    prj.set_cmap(field, cmap)


    return prj





def make_off_axis_projection_plots(ds, center, box_proj, fig_dir, haloname, normal_vector,north_vector,\
                         fig_end = 'projection',  do = ['stars', 'gas', 'dm'],\
                         axes = ['x', 'y', 'z'], annotate_positions = [],annotate_center = False, \
                          add_velocity = False,  add_arrow = False, start_arrow = [], end_arrow = [], proj_width = 0, hide_colorbar = False, add_timesamp = False):
    print (center,proj_width)
    for axs in axes:
        for d in do:
            if d == 'gas':
                field = ('gas', 'density')
                cmap = density_color_map
                cmap.set_bad('k')
                weight_field = None
                zmin = density_proj_min
                zmax = density_proj_max
                unit = 'Msun/pc**2'
            if d == 'stars':
                field = ('deposit', 'stars_density')
                cmap =  plt.cm.Greys_r
                cmap.set_bad('k')
                weight_field = None
                zmin = density_proj_min
                zmax = density_proj_max
                unit = 'Msun/pc**2'

            if d == 'dm':
                field = ('deposit', 'dm_density')
                cmap =  plt.cm.gist_heat
                cmap.set_bad('k')
                weight_field = None
                zmin = density_proj_min
                zmax = density_proj_max
                unit = 'Msun/pc**2'

            if d == 'temp':
                field = ('gas', 'temperature')
                cmap =  temperature_color_map
                cmap.set_bad('k')
                weight_field = ('gas', 'density')
                zmin = 1.e3
                zmax = temperature_max
                unit = 'K'


            if d == 'metal':
                field = ('gas', 'metallicity')
                metal_color_map = sns.blend_palette(
                    ("black", "#5d31c4", "#5d31c4","#4575b4", "#d73027",
                     "darkorange", "#ffe34d"), as_cmap=True)
                cmap =  metal_color_map
                cmap.set_bad('k')
                weight_field = ('gas', 'density')

                metal_min = 1.e-3
                zmin = metal_min
                zmax = metal_max
                unit = 'Zsun'
            prj = do_plot(ds = ds, field = field, normal_vector = normal_vector,  \
                          north_vector = north_vector, annotate_positions = annotate_positions, \
                          box_proj = box_proj, center = center, \
                          cmap = cmap, unit = unit, weight_field = weight_field, zmin = zmin, zmax = zmax, proj_width = proj_width)

            if add_velocity: prj.annotate_velocity(factor=20)
            if add_arrow: 
                if (start_arrow == []) | (end_arrow == []):
                    print ('Called add_arrow, but missing start_arrow or end_arrow')
                else:
                    for aa, (s_arrow, e_arrow) in enumerate(zip(start_arrow, end_arrow)):
                        if aa == 0:
                            prj.annotate_arrow(pos = e_arrow, starting_pos = s_arrow, coord_system = 'data')
                        else:
                            prj.annotate_arrow(pos = e_arrow, starting_pos = s_arrow, coord_system = 'data', plot_args={'color':'blue'})                            
            #prj.annotate_timestamp(corner='upper_left', redshift=True, draw_inset_box=True)            

            prj.hide_axes()
            prj.annotate_scale(size_bar_args={'color':'white'})
            prj.save(fig_dir + '/%s_%s_%s_%s.png'%(haloname, axs, d, fig_end))



    return prj


def load_sim(args):
    '''
    foggie_dir, output_dir, run_loc, trackname, haloname, spectra_dir, infofile = get_run_loc_etc(args)
    track_dir =  trackname.split('halo_tracks')[0]   + 'halo_infos/00' + args.halo + '/' + args.run + '/'
    snap_name = foggie_dir + run_loc + args.output + '/' + args.output
    ds, refine_box, refine_box_center, refine_width = load(snap = snap_name, 
                                                           trackfile = trackname, 
                                                           use_halo_c_v=False, 
                                                           halo_c_v_name=track_dir + 'halo_c_v')
    '''
    foggie_dir, output_dir, run_loc, trackname, haloname, spectra_dir, infofile = get_run_loc_etc(args)
    run_dir = foggie_dir + run_loc

    ds_loc = run_dir + args.output + "/" + args.output
    ds = yt.load(ds_loc)
    track = Table.read(trackname, format='ascii')
    track.sort('col1')
    zsnap = ds.get_parameter('CosmologyCurrentRedshift')

    refine_box, refine_box_center, x_width = get_refine_box(ds, zsnap, track)


    return ds, refine_box, x_width


if __name__ == '__main__':

    args = parse_args()
    inputs = [(args.halo, args.output),]

    fig_dir = '/nobackupp2/rcsimons/foggie/figures/off_axis_satellite_projections'

    ds, refine_box, refine_width = load_sim(args)

    filter_particles(refine_box, filter_particle_types = ['young_stars', 'old_stars', 'stars', 'dm', 'young_stars7', 'young_stars8'])

    sat_center = ds.halo_center_kpc 
    box_proj = refine_box
    box_width = refine_width
    print (sat_center)
    if np.isnan(args.rot_n):
        min_nrot = 0
        max_nrot = 200
    else:
        min_nrot = args.rot_n
        max_nrot = args.rot_n + 1
    start_arrows = []
    end_arrows = []
    #dos = ['stars', 'gas',  'dm', 'temp', 'metal']
    dos = ['young_stars7', 'young_stars', 'young_stars8', 'old_stars']
    

    if not os.path.exists(fig_dir + '/' + args.halo + '/' + args.output):
        os.system('mkdir ' + fig_dir + '/' + args.halo + '/' + args.output)

    for nrot in np.arange(min_nrot, max_nrot):
        normal_vector = [0, np.cos(2*pi * (1.*nrot)/200), np.sin(2*pi * (1.*nrot)/200)]

        prj = make_off_axis_projection_plots(ds = ds, center = sat_center, box_proj = box_proj, fig_dir = fig_dir + '/' + args.halo + '/' + args.output,
                                            haloname = args.halo, normal_vector = normal_vector, north_vector = [1,0,0], \
                                            fig_end = 'central_%.3i'%(nrot), \
                                            do = dos, axes = ['off_axis'], annotate_center = not args.do_central,
                                            add_velocity = False, add_arrow = not args.do_central, start_arrow = start_arrows, end_arrow = end_arrows, proj_width = box_width)  










