# Written by Raymond Simons, last updated 10/3/2019
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
from yt.units import kpc
import multiprocessing as mp
import os
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
    prj = yt.OffAxisProjectionPlot(ds, normal_vector, field, north_vector = north_vector, center = center, width = proj_width, data_source = box_proj, weight_field = weight_field)


    prj.set_unit(field, unit)
    prj.set_zlim(field, zmin = zmin, zmax =  zmax)
    prj.set_cmap(field, cmap)


    return prj





def make_off_axis_projection_plots(ds, center, box_proj, fig_dir, haloname, normal_vector,north_vector,\
                         fig_end = 'projection',  do = ['stars', 'gas', 'dm'],\
                         axes = ['x', 'y', 'z'], annotate_positions = [],annotate_center = False, \
                          add_velocity = False,  add_arrow = False, start_arrow = [], end_arrow = [], proj_width = 0):
    print (center,print (proj_width))
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
            prj.annotate_timestamp(corner='upper_left', redshift=True, draw_inset_box=True)            
            if annotate_center: prj.annotate_marker((0.5, 0.5), coord_system='axis')
            prj.hide_axes()
            prj.annotate_scale()

            prj.save(fig_dir + '/%s_%s_%s_%s.png'%(haloname, axs, d, fig_end))



    return prj



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
        inputs = inputs
    else:
        inputs = [(args.halo, args.output),]

    if args.system == 'jase':
        sat_cat = ascii.read('/Users/rsimons/Dropbox/foggie/catalogs/satellite_properties.cat')        
        fig_dir = '/Users/rsimons/Dropbox/foggie/figures/off_axis_satellite_projections'
    else:
        sat_cat = ascii.read('/nobackupp2/rcsimons/foggie/catalogs/satellite_properties.cat')        
        fig_dir = '/nobackupp2/rcsimons/foggie/figures/off_axis_satellite_projections'

    for args.halo, args.output in inputs:
        sat_cat_halo = sat_cat[sat_cat['halo'] == int(args.halo)]


        foggie_dir, output_dir, run_loc, trackname, haloname, spectra_dir, blek = get_run_loc_etc(args)
        run_dir = foggie_dir + run_loc

        ds_loc = run_dir + args.output + "/" + args.output
        ds = yt.load(ds_loc)
        track = Table.read(trackname, format='ascii')
        track.sort('col1')
        zsnap = ds.get_parameter('CosmologyCurrentRedshift')

        refine_box, refine_box_center, x_width = get_refine_box(ds, zsnap, track)
        filter_particles(refine_box)


        small_sp = ds.sphere(refine_box_center, (20, 'kpc'))
        refine_box_bulk_vel = small_sp.quantities.bulk_velocity()

        #def run_sat(ds, sat, refine_box, refine_box_center, args): 
        for sat in sat_cat_halo:
            movie_fname = fig_dir + '/movies/%s_%s.mp4'%(haloname, sat['id'])
            if os.path.exists(movie_fname): continue
            if args.do_central:
                if sat['id'] != '0': continue
                sat_center = ds.arr(refine_box_center, 'code_length').to('kpc') 
                box_proj = refine_box

            else:
                if sat['id'] == '0': continue
                box_width = 10 * kpc
                sat_center = ds.arr([sat['x'], sat['y'], sat['z']], 'kpc')     
                #ensures that a 45 degree tilted box stays in the frame
                box_proj = ds.r[sat_center[0]     - box_width/np.sqrt(2): sat_center[0] + box_width/np.sqrt(2),
                                sat_center[1]     - box_width/np.sqrt(2): sat_center[1] + box_width/np.sqrt(2),
                                sat_center[2]     - box_width/np.sqrt(2): sat_center[2] + box_width/np.sqrt(2),
                            ]


            if os.path.isdir(fig_dir + '/movies/%s_%s'%(args.halo, sat['id'])):
                os.system('mkdir ' + fig_dir + '/movies/%s_%s'%(args.halo, sat['id']))


            if not args.do_central:
                sp = ds.sphere(center = ds.arr(sat_center, 'kpc'), radius = 1*kpc)

                #sp.set_field_parameter('bulk_velocity', refine_box_bulk_vel)

                stars_vx = sp.quantities.weighted_average_quantity(('stars', 'particle_velocity_x'), ('stars', 'particle_mass')).to('km/s') -  refine_box_bulk_vel[0]
                stars_vy = sp.quantities.weighted_average_quantity(('stars', 'particle_velocity_y'), ('stars', 'particle_mass')).to('km/s') -  refine_box_bulk_vel[1]
                stars_vz = sp.quantities.weighted_average_quantity(('stars', 'particle_velocity_z'), ('stars', 'particle_mass')).to('km/s') -  refine_box_bulk_vel[2]
                
                print (stars_vx)


                np.random.seed(1)

                sat_velocity = ds.arr([stars_vx, stars_vy, stars_vz])
                sat_velocity_norm = sat_velocity/np.linalg.norm(sat_velocity)
                start_arrow = sat_center
                end_arrow = sat_center + sat_velocity_norm.value *  kpc
                main_gal = sat_cat_halo[sat_cat_halo['id'] == '0']

                to_center_vec = ds.arr([float(main_gal['x']), float(main_gal['y']), float(main_gal['z'])], 'kpc') -   sat_center  
                end_arrow_2 = sat_center  + to_center_vec.value/np.linalg.norm(to_center_vec.value) * 1. * kpc



            fname = fig_dir + '/combined/%s_%s.png'%(haloname, sat['id'])

            x = np.random.randn(3)
            x -= x.dot(sat_velocity_norm.value) * sat_velocity_norm.value
            normal_vector_1 = x/np.linalg.norm(x) 


            norm_vector = normal_vector_1
            prj = make_off_axis_projection_plots(ds, sat_center, box_proj, fig_dir + '/' + args.halo, haloname,norm_vector, north_vector = sat_velocity_norm.value, \
                                fig_end = '%s'%(sat['id']), \
                                do = ['gas', 'stars', 'dm', 'temp', 'metal'], axes = ['off_axis'],  annotate_center = True,
                                add_velocity = False, add_arrow = True, start_arrow = [start_arrow], end_arrow = [end_arrow], proj_width = box_width)  

            fl_dm = fig_dir + '/%s/%s_%s_dm_%s.png'%(args.halo, haloname, 'off_axis', sat['id'])
            fl_stars = fig_dir + '/%s/%s_%s_stars_%s.png'%(args.halo, haloname, 'off_axis', sat['id'])
            fl_gas = fig_dir + '/%s/%s_%s_gas_%s.png'%(args.halo, haloname, 'off_axis', sat['id'])
            fl_temp = fig_dir + '/%s/%s_%s_temp_%s.png'%(args.halo, haloname, 'off_axis', sat['id'])
            fl_metal = fig_dir + '/%s/%s_%s_metal_%s.png'%(args.halo, haloname, 'off_axis', sat['id'])


            fls = [fl_dm,  fl_stars, fl_gas, fl_temp, fl_metal]
            imgs = [PIL.Image.open(fl) for fl in fls]


            min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
            imgs_comb = np.hstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )


            imgs_comb = PIL.Image.fromarray( imgs_comb)
            imgs_comb.save(fname)

            if True:
                if np.isnan(args.rot_n):
                    min_nrot = 0
                    max_nrot = 200
                else:
                    min_nrot = args.rot_n
                    max_nrot = args.rot_n+1


                if args.do_central:
                        #preload these, i assume it makes it faster (instead of having to do it over and over again inside the function)
                        print ('pre-loading grid data')
                        grid_fields = [('deposit', 'stars_density'), ('gas', 'density'),\
                                       ('deposit', 'dm_density'), ('gas', 'temperature'),\
                                       ('gas', 'metallicity')]
                        for (n1, n2) in grid_fields:
                            print ('\t loading (%s, %s)'%(n1, n2))
                            temp_variable = box_proj[n1, n2]

                        start_arrows = []
                        end_arrows = []
                else:
                    start_arrows = [start_arrow, start_arrow]
                    end_arrows   = [end_arrow, end_arrow_2]

                for nrot in np.arange(min_nrot, max_nrot):
                    fname = fig_dir + '/combined/%s_%s_%.3i.png'%(haloname, sat['id'], nrot)

                    normal_vector = [0, np.cos(2*pi * (1.*nrot)/max_nrot), np.sin(2*pi * (1.*nrot)/max_nrot)]

                    prj = make_off_axis_projection_plots(ds = ds, center = sat_center, box_proj = box_proj, fig_dir = fig_dir + '/' + args.halo,
                                                        haloname = haloname, normal_vector = normal_vector, north_vector = [1,0,0], \
                                                        fig_end = '%s_%.3i'%(sat['id'], nrot), \
                                                        do = ['gas', 'stars', 'dm', 'temp', 'metal'], axes = ['off_axis'], annotate_center = not args.do_central,
                                                        add_velocity = False, add_arrow = not args.do_central, start_arrow = start_arrows, end_arrow = end_arrows, proj_width = box_width)  

                    fl_dm = fig_dir + '/%s/%s_%s_dm_%s_%.3i.png'%(args.halo, haloname, 'off_axis', sat['id'], nrot)
                    fl_stars = fig_dir + '/%s/%s_%s_stars_%s_%.3i.png'%(args.halo, haloname, 'off_axis', sat['id'], nrot)
                    fl_gas = fig_dir + '/%s/%s_%s_gas_%s_%.3i.png'%(args.halo, haloname, 'off_axis', sat['id'], nrot)
                    fl_temp = fig_dir + '/%s/%s_%s_temp_%s_%.3i.png'%(args.halo, haloname, 'off_axis', sat['id'], nrot)
                    fl_metal = fig_dir + '/%s/%s_%s_metal_%s_%.3i.png'%(args.halo, haloname, 'off_axis', sat['id'], nrot)



                    fls = [fl_dm,  fl_stars, fl_gas, fl_temp, fl_metal]
                    imgs = [PIL.Image.open(fl) for fl in fls]


                    min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
                    imgs_comb = np.hstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )


                    imgs_comb = PIL.Image.fromarray( imgs_comb)
                    imgs_comb.save(fname)


                if ('jase' in args.system):
                    os.system('ffmpeg -y -r 24 -f image2 -start_number 0 -i ' + fig_dir + '/combined/%s_%s_'%(haloname, sat['id']) + r'%03d.png ' +'-vframes 1000 -vcodec libx264 -crf 25  -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p ' + fig_dir + '/movies/%s_%s.mp4'%(haloname, sat['id']))













