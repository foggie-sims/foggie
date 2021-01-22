#Last modified 01/22/21, Raymond Simons
# An example script to create projection plots.
# A number of fields are demonstrated: 
# 'gas density', 'stars density', 'dm density', 
# 'temperature', 'metallicity', 'radial velocity'
# 


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




def parse_args(haloname, output):
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
    parser.set_defaults(output=output)

    parser.add_argument('--orient', metavar='output', type=str, action='store',
                        help='which output? default is RD0020')
    parser.set_defaults(output='x')

    args = parser.parse_args()
    return args

def retrieve_projection_parameters(args):
    # this function draws the arguments that are passed in
    # to the projection map generator
    if args.do == 'gas':
        field = ('gas', 'density')
        cmap = density_color_map
        cmap.set_bad('k')
        unit = 'Msun/pc**2'
        zmin = density_proj_min 
        zmax = density_proj_max
        weight_field = None

    if args.do == 'stars':
        field = ('deposit', 'stars_density')
        cmap =  plt.cm.Greys_r
        cmap.set_bad('k')
        unit = 'Msun/pc**2'
        zmin = density_proj_min 
        zmax = density_proj_max
        weight_field = None

    if args.do == 'dm':
        field = ('deposit', 'dm_density')
        cmap =  plt.cm.gist_heat
        cmap.set_bad('k')
        unit = 'Msun/pc**2'
        zmin = density_proj_min 
        zmax = density_proj_max
        weight_field = None

    if args.do == 'temp':
        field = ('gas', 'temperature')
        cmap =  temperature_color_map
        cmap.set_bad('k')
        weight_field = ('gas', 'density')
        zmin = 1.e3
        zmax = temperature_max
        unit = 'K'

    if args.do == 'metal':
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

    if args.do == 'vrad':
        field = ('gas', 'metallicity')
        cmap =  velocity_colors
        cmap.set_bad('k')
        weight_field = ('gas', 'density')
        zmin = -250
        zmax = 250
        unit = 'km/s'

    return field, cmap, weight_field, zmin, zmax, unit

#if you pass these into the command line parser, it will override these
haloname, output = ('halo_008508', 'DD1000')

#Read in command-line arguments
args = parse_args(haloname, output)
ds, refine_box = load_sim(args)
field, cmap, weight_field, zmin, zmax, unit = retrieve_projection_parameters(args)
prj = yt.ProjectionPlot(ds, axs, field, center = center, data_source = refine_box,\
                        width=x_width, weight_field = weight_field)

prj.set_unit(field, unit)
prj.set_zlim(field, zmin = zmin, zmax =  zmax)
prj.set_cmap(field, cmap)

if False:
    #These are all optional
    prj.annotate_scale(size_bar_args={'color':'white'})
    prj.annotate_timestamp(corner='upper_right', redshift=True, draw_inset_box=True)
    prj.annotate_text((0.05, 0.9), haloname, coord_system='axis', \
                  text_args = {'fontsize': 500, 'color': 'white'}, inset_box_args = {})
    prj.hide_axes()
    prj.hide_colorbar()    
    if False:
        #if you want to add velocity arrows 
        prj.annotate_velocity(factor=20)

prj.save('./%s_%s_%s_%s.png'%(haloname, axs, do, fig_end), mpl_kwargs = {'dpi': 500})






























