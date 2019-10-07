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

    parser.add_argument('--output', metavar='output', type=str, action='store',
                        help='which output? default is RD0020')
    parser.set_defaults(output="RD0020")


    args = parser.parse_args()
    return args


def do_plot(field, axs, annotate_others, annotate_center, \
                small_box, halo_center, x_width, \
                cmap, unit = 'Msun/pc**2', zmin = density_proj_min, zmax = density_proj_max,\
                 ann_sphere_rad = (1, 'kpc')):

    prj = yt.ProjectionPlot(ds, axs, field, center = halo_center, data_source = small_box, width=x_width)

    prj.set_unit(field, unit)
    prj.set_zlim(field, zmin = zmin, zmax =  zmax)
    prj.set_cmap(field, cmap)


    prj.annotate_timestamp(corner='upper_left', redshift=True, draw_inset_box=True)
    for cen in annotate_others:
        prj.annotate_sphere(cen, radius = ann_sphere_rad, coord_system='data', circle_args={'color':'darkred'})                        
    if annotate_center:
        prj.annotate_sphere((0.5, 0.5), radius = ann_sphere_rad, coord_system='axis', circle_args={'color':'red'})

    return prj




def make_projection_plots(ds, halo_center, refine_box, x_width,fig_dir, \
                         fig_end = 'projection',  do = ['stars', 'gas', 'dm'],\
                         axes = ['x', 'y', 'z'], annotate_center = False, annotate_others = [],\
                          add_velocity = False, is_central = False):

    if not is_central:
        small_box = ds.r[halo_center[0] - x_width/2.: halo_center[0] + x_width/2.,
                     halo_center[1] - x_width/2.: halo_center[1] + x_width/2.,
                     halo_center[2] - x_width/2.: halo_center[2] + x_width/2.,
                    ]

        stars_vx = small_box.quantities.weighted_average_quantity(('stars', 'particle_velocity_x'), ('stars', 'particle_mass')).to('cm/s')
        stars_vy = small_box.quantities.weighted_average_quantity(('stars', 'particle_velocity_y'), ('stars', 'particle_mass')).to('cm/s')
        stars_vz = small_box.quantities.weighted_average_quantity(('stars', 'particle_velocity_z'), ('stars', 'particle_mass')).to('cm/s')

        vel_dir = ds.arr([stars_vx, stars_vy, stars_vz]).to('km/s')
        vel_mag = np.sqrt(vel_dir**2.)
        vel_dir/=vel_mag
        from yt.units import kpc
        ray = ds.ray((halo_center), (halo_center + vel_dir * vel_mag/(ds.arr(100, 'km/s'))*kpc))
        start_arrow = halo_center
        end_arrow = halo_center + vel_dir * vel_mag/(ds.arr(100, 'km/s'))*kpc

        bulk_vel = ds.arr([stars_vx, stars_vy, stars_vz])
        small_box.set_field_parameter("bulk_velocity", bulk_vel * x_width.to('cm'))
        print (small_box.get_field_parameter("bulk_velocity"))
    else:
        small_box = refine_box

    for axs in axes:
        for d in do:
            if d == 'gas':
                field = ('gas', 'density')
                cmap = density_color_map

            if d == 'stars':
                field = ('deposit', 'stars_density')
                cmap =  plt.cm.Greys_r
                cmap.set_bad('k')

            if d == 'dm':
                field = ('deposit', 'dm_density')
                cmap =  plt.cm.gist_heat
                cmap.set_bad('k')

            prj = do_plot(field, axs, annotate_others, annotate_center, \
                          small_box, halo_center, x_width,\
                          cmap)
            prj.save(fig_dir + '/bare/%s_%s_%s_%s.png'%(haloname, axs, d, fig_end))
            if not is_central:
                prj.annotate_arrow(pos = end_arrow, starting_pos = start_arrow, coord_system = 'data')
                prj.save(fig_dir + '/with_single_arrow/%s_%s_%s_%s.png'%(haloname, axs, d, fig_end))
                prj.annotate_velocity(factor=20)
                prj.save(fig_dir + '/with_all_arrows/%s_%s_%s_%s.png'%(haloname, axs, d, fig_end))

    return



def load_particle_data(refine_box):
    print ('loading star particle data...')
    print ('\t particle masses')
    mass_stars = refine_box['stars', 'particle_mass'].to('Msun')
    print ('\t particle positions')
    x_stars = refine_box['stars', 'particle_position_x'].to('kpc')
    y_stars = refine_box['stars', 'particle_position_y'].to('kpc')
    z_stars = refine_box['stars', 'particle_position_z'].to('kpc')
    print ('\t particle ids')
    particle_ids = refine_box['stars', 'particle_index']

    return mass_stars, x_stars, y_stars, z_stars, particle_ids

def make_segmentation_figure(sm_im, seg_im, figname):
    # Aligning the image with YT.ProjectionPlot
    plt.close('all')


    cmap = plt.cm.viridis
    cmap.set_bad('k')


    to_show = np.rot90(sm_im)
    seg_to_show = np.rot90(seg_im)

    if orient == 'y':
        to_show = np.flipud(sm_im)
        seg_to_show = np.flipud(seg_im)

    fig, axes = plt.subplots(1,2, figsize = (10,5))
    axes[0].imshow(to_show, cmap = cmap, origin = 'upper')
    axes[1].imshow(seg_to_show, cmap = cmap, origin = 'upper')
    axes[0].set_title('highest density regions')
    axes[1].set_title('segmentation map')
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.savefig(figname)
    plt.close('all')
    return

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


        center_file = foggie_dir.replace('/sims/', '/outputs/centers/{}_{}.npy'.format(haloname, args.output))
        if os.path.isfile(center_file): 
            # Load the halo center
            halo_center = np.load(center_file)
        else:
            # Calculate the center of the halo and save for future
            halo_center = get_halo_center(ds, refine_box_center)[0]
            np.save(center_file, halo_center)

        fig_dir = foggie_dir.replace('sims/', 'figures/identify_satellites') 

        sat_cat = ascii.read(save_dir + '/satellite_locations.cat')

        sats_halo = sat_cat[(sat_cat['halo'] == int(args.halo)) & (sat_cat['run'] == args.run) &  (sat_cat['output'] == args.output)]




        
        annotate_others = []
        for sat in sats_halo: annotate_others.append(ds.arr([sat['x'], sat['y'], sat['z']], 'kpc'))
        if False:
            for sat in sats_halo[:]: 
                from yt.units import kpc
                fig_width = 10 * kpc
                
                # Make individual satellite projection plots
                sat_center = ds.arr([sat['x'], sat['y'], sat['z']], 'kpc')     


                make_projection_plots(ds, sat_center, refine_box, fig_width, fig_dir, \
                                    fig_end = 'satellite_{}'.format(sat['id']), \
                                    do = ['gas', 'stars', 'dm'], axes = ['y'],  annotate_center = False, annotate_others = annotate_others,\
                                    add_velocity = False)        

        # Show satellites on a figure of the central
        if True:
            make_projection_plots(ds, ds.arr(halo_center, 'code_length').to('kpc'), refine_box, x_width, fig_dir, \
                                  fig_end = 'central',\
                                  do = ['stars'], axes = ['y'],\
                                  annotate_center = True, annotate_others = annotate_others, is_central = True)














