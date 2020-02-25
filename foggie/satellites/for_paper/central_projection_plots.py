import yt
import numpy as np
from numpy import *
from astropy.table import Table, Column
from foggie.utils.foggie_utils import filter_particles
from foggie.utils.get_run_loc_etc import get_run_loc_etc
from foggie.utils.get_refine_box import get_refine_box
from foggie.utils.get_halo_center import get_halo_center
from foggie.utils.get_proper_box_size import get_proper_box_size
from foggie.utils import yt_fields
import os, sys, argparse
from foggie.utils.consistency import *
import matplotlib.pyplot as plt




def parse_args(haloname, DDname):
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
    parser.set_defaults(halo=haloname.strip('halo_00'))

    parser.add_argument('--pwd', dest='pwd', action='store_true',
                        help='just use the pwd?, default is no')
    parser.set_defaults(pwd=False)

    parser.add_argument('--run_all', dest='run_all', action='store_true',
                        help='just use the pwd?, default is no')
    parser.set_defaults(pwd=False)


    parser.add_argument('--output', metavar='output', type=str, action='store',
                        help='which output? default is RD0020')
    parser.set_defaults(output=DDname)


    args = parser.parse_args()
    return args


halonames = array([('halo_002392', 'Hurricane', 'DD0581'), 
                   ('halo_002878', 'Cyclone',  'DD0581'), 
                   ('halo_004123', 'Blizzard',  'DD0581'), 
                   ('halo_005016', 'Squall',  'DD0581'), 
                   ('halo_005036', 'Maelstrom',  'DD0581'), 
                   ('halo_008508', 'Tempest',  'DD0487')])


def do_plot(ds, field, axs, annotate_positions, \
                small_box, center, x_width, \
                cmap, name, unit = 'Msun/pc**2', zmin = density_proj_min, zmax = density_proj_max,\
                 ann_sphere_rad = (1, 'kpc')):

    prj = yt.ProjectionPlot(ds, axs, field, center = center, data_source = small_box, width=x_width)

    prj.set_unit(field, unit)
    prj.set_zlim(field, zmin = zmin, zmax =  zmax)
    prj.set_cmap(field, cmap)

    #prj.annotate_scale(size_bar_args={'color':'white'})

    #prj.annotate_timestamp(corner='upper_right', redshift=True, draw_inset_box=True)
    '''
    prj.annotate_text((0.05, 0.9), name, coord_system='axis', \
                      text_args = {'fontsize': 500, 'color': 'white'},\
                      inset_box_args={'boxstyle':'square,pad=2.0',
                                 'facecolor':'black',
                                 'linewidth':3,
                                 'edgecolor':'white', \
                                 'alpha':0.5})
    '''
    #prj.annotate_text((0.05, 0.9), name, coord_system='axis', \
    #                  text_args = {'fontsize': 500, 'color': 'white'}, inset_box_args = {})

    prj.hide_axes()
    prj.hide_colorbar()


    for cen in annotate_positions:
        prj.annotate_sphere(cen, radius = ann_sphere_rad, coord_system='data', circle_args={'color':'white'})                        
        #prj.annotate_marker(cen, coord_system='data')
    

    return prj





def make_projection_plots(ds, center, refine_box, x_width,fig_dir, haloname, name, \
                         fig_end = 'projection',  do = ['stars', 'gas', 'dm'],\
                         axes = ['x', 'y', 'z'], annotate_positions = [],\
                          add_velocity = False, is_central = False, add_arrow = False, start_arrow = [], end_arrow = []):

    if not is_central:
        small_box = ds.r[center[0] - x_width/2.: center[0] + x_width/2.,
                     center[1] - x_width/2.: center[1] + x_width/2.,
                     center[2] - x_width/2.: center[2] + x_width/2.,
                    ]
    else:
        small_box = refine_box

    for axs in axes:
        for d in do:
            if d == 'gas':
                field = ('gas', 'density')
                cmap = density_color_map
                cmap.set_bad('k')
            if d == 'stars':
                field = ('deposit', 'stars_density')
                cmap =  plt.cm.Greys_r
                cmap.set_bad('k')

            if d == 'dm':
                field = ('deposit', 'dm_density')
                cmap =  plt.cm.gist_heat
                cmap.set_bad('k')

            prj = do_plot(ds, field, axs, annotate_positions, \
                          small_box, center, x_width,\
                          cmap, name)

            if add_velocity: prj.annotate_velocity(factor=20)
            if add_arrow: 
                if (start_arrow == []) | (end_arrow == []):
                    print ('Called add_arrow, but missing start_arrow or end_arrow')
                else:
                    for s_arrow, e_arrow in zip(start_arrow, end_arrow):
                        prj.annotate_arrow(pos = e_arrow, starting_pos = s_arrow, coord_system = 'data')

            prj.save(fig_dir + '/%s_%s_%s_%s_nolabels.png'%(haloname, axs, d, fig_end), mpl_kwargs = {'dpi': 500})
    return prj




halonames = halonames[4:]
for (haloname, name, DDname) in halonames:

    args = parse_args(haloname, DDname)



    #ds = yt.load(flname)
    #center_dic =  np.load('/Users/rsimons/Desktop/foggie/outputs/centers/%s_nref11c_nref9f_%s.npy'%(haloname,DDname), allow_pickle = True)[()]
    #center = ds.arr(center_dic, 'code_length').to('kpc')


    foggie_dir, output_dir, run_loc, trackname, haloname, spectra_dir, infofile = get_run_loc_etc(args)


    run_dir = foggie_dir + run_loc

    ds_loc = run_dir + args.output + "/" + args.output
    ds = yt.load(ds_loc)

    track = Table.read(trackname, format='ascii')
    track.sort('col1')
    zsnap = ds.get_parameter('CosmologyCurrentRedshift')

    refine_box, refine_box_center, x_width = get_refine_box(ds, zsnap, track)
    filter_particles(refine_box)

    fig_dir = '/Users/rsimons/Dropbox/foggie/figures/for_paper/central_projections'
    prj = make_projection_plots(refine_box.ds, ds.arr(refine_box_center, 'code_length').to('kpc'),\
                        refine_box, x_width, fig_dir, haloname, name, \
                        fig_end = 'projection',\
                        do = ['stars'], axes = ['x'], is_central = True, add_arrow = False)











