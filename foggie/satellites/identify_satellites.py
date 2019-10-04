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






def make_projection_plots(ds, halo_center, refine_box, x_width,fig_dir, fig_end = 'projection',  do = ['stars', 'gas', 'dm'], axes = ['x', 'y', 'z'], annotate_sphere = False):
    for axs in axes:
        if 'gas' in do:
            prj = yt.ProjectionPlot(ds, axs, 'density', center = halo_center, data_source = refine_box, width=x_width)
            prj.set_unit(('gas','density'), 'Msun/pc**2')
            prj.set_zlim(('gas', 'density'), zmin = density_proj_min, zmax =  density_proj_max)
            prj.set_cmap(('gas', 'density'), density_color_map)
            prj.annotate_timestamp(corner='upper_left', redshift=True, draw_inset_box=True)
            if annotate_sphere:
                prj.annotate_sphere((0.5, 0.5), radius = (2, 'kpc'), coord_system='axis', circle_args={'color':'red'})
            prj.save(fig_dir + '/%s_%s_gas_%s.png'%(haloname, axs, fig_end))
    
        if 'stars' in do:
            prj = yt.ParticleProjectionPlot(ds, axs, ('stars', 'particle_mass'), center = halo_center, data_source=refine_box, width = x_width)   
            cmp = plt.cm.Greys_r
            cmp.set_bad('k')
            prj.set_cmap(field = ('stars','particle_mass'), cmap = cmp)
            prj.set_zlim(field = ('stars','particle_mass'), zmin = 1.e37, zmax = 1.e42)
            prj.annotate_timestamp(corner='upper_left', redshift=True, draw_inset_box=True)
            if annotate_sphere:
                prj.annotate_sphere((0.5, 0.5), radius = (2, 'kpc'), coord_system='axis', circle_args={'color':'red'})
            prj.save(fig_dir + '/%s_%s_star_%s.png'%(haloname, axs, fig_end))

        if 'dm' in do:
            prj = yt.ParticleProjectionPlot(ds, axs, ('dm', 'particle_mass'), center = halo_center, data_source=refine_box, width = x_width)   
            cmp = plt.cm.gist_heat
            cmp.set_bad('k')
            prj.set_cmap(field = ('dm','particle_mass'), cmap = cmp)
            prj.set_zlim(field = ('dm','particle_mass'), zmin = 1.e37, zmax = 1.e42)
            prj.annotate_timestamp(corner='upper_left', redshift=True, draw_inset_box=True)
            if annotate_sphere:
                prj.annotate_sphere((0.5, 0.5), radius = (2, 'kpc'), coord_system='axis', circle_args={'color':'red'})
            prj.save(fig_dir + '/%s_%s_dm_%s.png'%(haloname, axs, fig_end))

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

        fig_dir = foggie_dir.replace('sims', 'figures/identify_satellites') 
        if args.do_proj_plots:
            make_projection_plots(ds, halo_center, refine_box, x_width, fig_dir, do = ['gas', 'dm', 'stars'], axes = ['x','y','z'])



        # Set the defined center coordinate of the box at the halo center

        refine_box.set_field_parameter('center', ds.arr(halo_center, 'code_length'))
        bulk_vel = refine_box.quantities.bulk_velocity()
        refine_box.set_field_parameter("bulk_velocity", bulk_vel)

        mass_stars = refine_box['stars', 'particle_mass'].to('Msun')
        x_stars = refine_box['stars', 'particle_position_x'].to('kpc')
        y_stars = refine_box['stars', 'particle_position_y'].to('kpc')
        z_stars = refine_box['stars', 'particle_position_z'].to('kpc')
        particle_ids = refine_box['stars', 'particle_index']

        halo_center_kpc = ds.arr(halo_center, 'code_length').to('kpc').value

        width = ds.arr(x_width, 'code_length').to('kpc').value      

        bin_size = 0.25
        xbin = np.arange(halo_center_kpc[0] - width/2., halo_center_kpc[0] + width/2. + bin_size, bin_size)
        ybin = np.arange(halo_center_kpc[1] - width/2., halo_center_kpc[1] + width/2. + bin_size, bin_size)
        zbin = np.arange(halo_center_kpc[2] - width/2., halo_center_kpc[2] + width/2. + bin_size, bin_size)

        p = np.histogramdd((x_stars, y_stars, z_stars), \
                            weights = mass_stars, \
                            bins = (xbin, ybin, zbin))



        pp = p[0]
        pp[p[0] < 1.e6] = np.nan
        cmap = plt.cm.viridis

        cmap.set_bad('k')

        plt.close('all')


        #for i, orient in enumerate(['x', 'y', 'z']):
        sat_coords = []

        all_stars = [x_stars, y_stars, z_stars]
        ortho_orients = [[1,2], [0,2], [0,1]]


        class Satellite:
            def __init__(self, name, x, y, z, ids):

                self.name = name
                self.x = x
                self.y = y
                self.z = z
                self.ids = particle_ids




        satellites = []
        for (i, orient) in array([(0, 'x'), (1, 'y'), (2, 'z')]):
            i = int(i)
            sm_im = np.log10(np.nansum(pp, axis = i))
            seg_im = detect_sources(sm_im, threshold = 0, npixels = 1, connectivity = 8)

            # Alinging the image with YT.ProjectionPlot
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

            fig.savefig(fig_dir + '/%s_%s_satellite_selection.png'%(haloname, orient))


            for label in seg_im.labels:

                edges1 = p[1][ortho_orients[i][0]]
                edges2 = p[1][ortho_orients[i][1]]

                gd = where(seg_im.data == label)

                all_ids = array([])
                for gd1, gd2 in zip(gd[0], gd[1]):
                    coord1_min, coord1_max = edges1[gd1], edges1[gd1+1]
                    coord2_min, coord2_max = edges2[gd2], edges2[gd2+1]

                    coords1 = all_stars[ortho_orients[i][0]]
                    coords2 = all_stars[ortho_orients[i][1]]
                    gd_ids = where((coords1 > coord1_min) & (coords1 < coord1_max) &\
                                   (coords2 > coord2_min) & (coords2 < coord2_max))[0]
                    all_ids = concatenate((all_ids, gd_ids), axis = None)
                mn_x = np.median(x_stars[all_ids.astype('int')])
                mn_y = np.median(y_stars[all_ids.astype('int')])
                mn_z = np.median(z_stars[all_ids.astype('int')])
                ids = particle_ids[all_ids.astype('int')]

                tag = 0
                print (label, mn_x, mn_y, mn_z)
                for sat in satellites:
                    diff = np.sqrt((mn_x - sat['x'])**2. + (mn_y - sat['y'])**2. + (mn_z - sat['z'])**2.)
                    if diff.value < 1.: 
                        print ('\t', 'match', diff.value, sat['name'])
                        tag = 1
                        break

                if tag == 0: 
                    new_satellite_dic = {}
                    new_satellite_dic['name'] = '%s_%i'%(orient, label)
                    new_satellite_dic['x'] = mn_x
                    new_satellite_dic['y'] = mn_y
                    new_satellite_dic['z'] = mn_z                    
                    new_satellite_dic['ids'] = ids



                    satellites.append(new_satellite_dic)
            print ('\n\n')

        save_dir = foggie_dir.replace('sims', 'outputs/identify_satellites')
        np.save(save_dir + '/satellites_%s.npy'%haloname, satellites)
        if args.do_sat_proj_plots:
            for sat in satellites:
                satx = float(sat['x'].value)
                saty = float(sat['y'].value)
                satz = float(sat['z'].value)
                sat_center = ds.arr([satx, saty, satz], 'kpc')
                from yt.units import kpc
                fig_width = 20 * kpc
                make_projection_plots(ds, sat_center, refine_box, fig_width, fig_dir, \
                                    fig_end = 'satellite_{}'.format(sat['name']), \
                                    do = ['stars', 'dm', 'gas'], axes = ['x', 'y','z'],  annotate_sphere = True)




























