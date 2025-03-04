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
from foggie.utils.foggie_load import *
from yt.units import *




def parse_args(haloname, DDname):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='''identify satellites in FOGGIE simulations''')
    parser.add_argument('-system', '--system', metavar='system', type=str, action='store', \
                        help='Which system are you on? Default is Jase')
    parser.set_defaults(system="jase")

    parser.add_argument('-do', '--do', metavar='do', type=str, action='store', \
                        help='Which system are you on? Default is Jase')
    parser.set_defaults(system="none")

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



def do_plot(ds, field, axs, annotate_positions, \
                small_box, center, x_width, \
                cmap, name, unit = 'Msun/pc**2', zmin = density_proj_min, zmax = density_proj_max,\
                 ann_sphere_rad = (1, 'kpc'), weight_field = None):
    print (x_width)
    prj = yt.ProjectionPlot(ds, axs, field, center = center, data_source = small_box,\
                            width=x_width, weight_field = weight_field, buff_size = (250,250))

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

    #prj.hide_axes()
    prj.hide_colorbar()


    for cen in annotate_positions:
        prj.annotate_sphere(cen, radius = ann_sphere_rad, coord_system='data', circle_args={'color':'white'})                        
        #prj.annotate_marker(cen, coord_system='data')
    

    return prj





def make_projection_plots(ds, center, refine_box, x_width,fig_dir, haloname, name, \
                         fig_end = 'projection',  do = ['stars', 'gas', 'dm'],\
                         axes = ['x', 'y', 'z'], annotate_positions = [],\
                          add_velocity = False, is_central = False, add_arrow = False, start_arrow = [], end_arrow = []):
    if is_central:
        small_box = refine_box

    else:
        small_box = ds.r[center[0] - x_width/2.: center[0] + x_width/2.,
                     center[1] - x_width/2.: center[1] + x_width/2.,
                     center[2] - x_width/2.: center[2] + x_width/2.,
                    ]
        
    for axs in axes:
        for d in do:
            if (d == 'gas'):
                field = ('gas', 'density')
                cmap = density_color_map
                cmap.set_bad('k')
                unit = 'Msun/pc**2'
                zmin = density_proj_min 
                zmax = density_proj_max
                weight_field = None

            if (d == 'gas2'):
                field = ('gas', 'density')
                cmap = density_color_map
                cmap.set_bad('k')
                unit = 'Msun/pc**3'
                zmin = density_proj_min*1e-5
                zmax = density_proj_max*1e-6
                weight_field = ('gas', 'density')

            if d == 'stars':
                field = ('deposit', 'stars_density')
                cmap =  plt.cm.Greys_r
                cmap.set_bad('k')
                unit = 'Msun/pc**2'
                zmin = density_proj_min 
                zmax = density_proj_max
                weight_field = None

            if d == 'dm':
                field = ('deposit', 'dm_density')
                cmap =  plt.cm.gist_heat
                cmap.set_bad('k')
                unit = 'Msun/pc**2'
                zmin = density_proj_min
                zmax = density_proj_max
                weight_field = None

            if d == 'dm2':
                field = ('deposit', 'dm_density')
                cmap =  plt.cm.Greys_r
                cmap.set_bad('k')
                unit = 'Msun/pc**3'
                zmin = density_proj_min*1e-5
                zmax = density_proj_max*1e-3
                weight_field = ('deposit', 'dm_density')


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

            if d == 'vrad':
                field = ('gas', 'metallicity')
                cmap =  velocity_colors
                cmap.set_bad('k')
                weight_field = ('gas', 'density')
                zmin = -250
                zmax = 250
                unit = 'km/s'

            prj = do_plot(ds, field, axs, annotate_positions, \
                          small_box, center, x_width,\
                          cmap, name, unit = unit, zmin = zmin, zmax = zmax, weight_field = weight_field)

            if add_velocity: prj.annotate_velocity(factor=20)
            if add_arrow: 
                if (start_arrow == []) | (end_arrow == []):
                    print ('Called add_arrow, but missing start_arrow or end_arrow')
                else:
                    for s_arrow, e_arrow in zip(start_arrow, end_arrow):
                        prj.annotate_arrow(pos = e_arrow, starting_pos = s_arrow, coord_system = 'data')

            prj.save(fig_dir + '/%s_%s_%s_%s_nolabels.png'%(haloname, axs, d, fig_end), mpl_kwargs = {'dpi': 500})
    return prj


if __name__ == '__main__':
    halonames = array([('halo_002392', 'Hurricane', 'DD0581'), 
                       ('halo_002878', 'Cyclone',  'DD0581'), 
                       ('halo_004123', 'Blizzard',  'DD0581'), 
                       ('halo_005016', 'Squall',  'DD0581'), 
                       ('halo_005036', 'Maelstrom',  'DD0581'), 
                       ('halo_008508', 'Tempest',  'DD0487'),
                       ('halo_002878', 'Tempest',  'DD0581')])


    halonames = halonames[6:7]

    for (haloname, name, DDname) in halonames:
        #ds = yt.load(flname)
        #center_dic =  np.load('/Users/rsimons/Desktop/foggie/outputs/centers/%s_nref11c_nref9f_%s.npy'%(haloname,DDname), allow_pickle = True)[()]
        #center = ds.arr(center_dic, 'code_length').to('kpc')

        args = parse_args(haloname, DDname)
        ds, refine_box = load_sim(args)
        if args.do == 'dm2':
            def dm_density_sqrd(field, data):
                return (data['deposit','dm_density'])**2.
            yt.add_field(('dm', 'dm_density2'), function=dm_density_sqrd, units='Msun**2/pc**4', \
                         take_log=False, force_override=True, sampling_type='particle')

        #fig_dir = '/Users/rsimons/Dropbox/foggie/figures/for_paper/central_projections'
        fig_dir = '.'
        print (ds.refine_box_center, ds.refine_width*kpc)

        use_box = ds.all_data()
        use_box.set_field_parameter('bulk_velocity', ds.halo_velocity_kms)
        '''
        prj = make_projection_plots(ds = refine_box.ds, center = ds.refine_box_center,\
                                    refine_box = refine_box, x_width = 3*ds.refine_width*kpc,\
                                    fig_dir = fig_dir, haloname = haloname, name = name, \
                                    fig_end = 'projection',\
                                    do = [args.do], axes = ['x'], is_central = True, add_arrow = False)
        '''

        prj = make_projection_plots(ds = use_box.ds, center = ds.refine_box_center,\
                                refine_box = use_box, x_width = 3*ds.refine_width*kpc,\
                                fig_dir = fig_dir, haloname = haloname, name = name, \
                                fig_end = 'projection',\
                                do = [args.do], axes = ['x'], is_central = True, add_arrow = False)










