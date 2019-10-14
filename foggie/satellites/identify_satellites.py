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
from yt.units import kpc
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

    parser.add_argument('--do_identify_satellites', dest='do_identify_satellites', action='store_true',
                        help='just use the pwd?, default is no')
    parser.set_defaults(pwd=False)


    parser.add_argument('--do_record_anchor_particles', dest='do_record_anchor_particles', action='store_true',
                        help='just use the pwd?, default is no')
    parser.set_defaults(pwd=False)

    parser.add_argument('--output', metavar='output', type=str, action='store',
                        help='which output? default is RD0020')
    parser.set_defaults(output="RD0020")



    args = parser.parse_args()
    return args






def make_projection_plots(ds, halo_center, refine_box, x_width,fig_dir, fig_end = 'projection',  do = ['stars', 'gas', 'dm'], axes = ['x', 'y', 'z'], annotate_center = False, annotate_others = [], ann_sphere_rad = (1, 'kpc')):
    for axs in axes:
        if 'gas' in do:
            prj = yt.ProjectionPlot(ds, axs, 'density', center = halo_center, data_source = refine_box, width=x_width)
            prj.set_unit(('gas','density'), 'Msun/pc**2')
            prj.set_zlim(('gas', 'density'), zmin = density_proj_min, zmax =  density_proj_max)
            prj.set_cmap(('gas', 'density'), density_color_map)
            prj.annotate_timestamp(corner='upper_left', redshift=True, draw_inset_box=True)
            for cen in annotate_others:
                prj.annotate_sphere(cen, radius = ann_sphere_rad, coord_system='data', circle_args={'color':'darkred'})                        
            if annotate_center:
                prj.annotate_sphere((0.5, 0.5), radius = ann_sphere_rad, coord_system='axis', circle_args={'color':'red'})



            prj.save(fig_dir + '/%s_%s_gas_%s.png'%(haloname, axs, fig_end))
    


        if 'stars' in do:
            prj = yt.ProjectionPlot(ds, axs, ('deposit', 'stars_density'), center = halo_center, data_source = refine_box, width=x_width)

            cmp = plt.cm.Greys_r
            cmp.set_bad('k')
            prj.set_unit(('deposit', 'stars_density'), 'Msun/pc**2')
            prj.set_cmap(field = ('deposit', 'stars_density'), cmap = cmp)
            prj.set_zlim(field = ('deposit', 'stars_density'), zmin = density_proj_min, zmax =  density_proj_max)
            prj.annotate_timestamp(corner='upper_left', redshift=True, draw_inset_box=True)

            for cen in annotate_others:
                prj.annotate_sphere(cen, radius = ann_sphere_rad, coord_system='data', circle_args={'color':'darkred'})                        

            if annotate_center:
                prj.annotate_sphere((0.5, 0.5), radius = ann_sphere_rad, coord_system='axis', circle_args={'color':'red'})


            prj.save(fig_dir + '/%s_%s_star_%s.png'%(haloname, axs, fig_end))



        if 'dm' in do:
            prj = yt.ProjectionPlot(ds, axs, ('deposit', 'dm_density'), center = halo_center, data_source = refine_box, width=x_width)

            cmp = plt.cm.gist_heat
            cmp.set_bad('k')
            prj.set_unit(('deposit', 'dm_density'), 'Msun/pc**2')
            prj.set_cmap(field = ('deposit', 'dm_density'), cmap = cmp)
            prj.set_zlim(field = ('deposit', 'dm_density'), zmin = density_proj_min, zmax =  density_proj_max)
            prj.annotate_timestamp(corner='upper_left', redshift=True, draw_inset_box=True)
            for cen in annotate_others:
                prj.annotate_sphere(cen, radius = ann_sphere_rad, coord_system='data', circle_args={'color':'darkred'})                        

            if annotate_center:
                prj.annotate_sphere((0.5, 0.5), radius = ann_sphere_rad, coord_system='axis', circle_args={'color':'red'})

            prj.save(fig_dir + '/%s_%s_dm_%s.png'%(haloname, axs, fig_end))






        if 'stars_particle' in do:
            prj = yt.ParticleProjectionPlot(ds, axs, ('stars', 'particle_mass'), center = halo_center, data_source=refine_box, width = x_width)   
            cmp = plt.cm.Greys_r
            cmp.set_bad('k')

            prj.set_cmap(field = ('stars','particle_mass'), cmap = cmp)
            prj.set_zlim(field = ('stars','particle_mass'), zmin = 1.e37, zmax = 1.e42)
            prj.annotate_timestamp(corner='upper_left', redshift=True, draw_inset_box=True)
            if annotate_center:
                prj.annotate_sphere((0.5, 0.5), radius = ann_sphere_rad, coord_system='axis', circle_args={'color':'red'})

            for cen in annotate_others:
                prj.annotate_sphere(cen, radius = ann_sphere_rad, coord_system='data', circle_args={'color':'blue'})                        

            prj.save(fig_dir + '/star_particle_projection/%s_%s_star_particleproj_%s.png'%(haloname, axs, fig_end))

        if 'dm_particle' in do:
            prj = yt.ParticleProjectionPlot(ds, axs, ('dm', 'particle_mass'), center = halo_center, data_source=refine_box, width = x_width)   
            cmp = plt.cm.gist_heat
            cmp.set_bad('k')
            prj.set_cmap(field = ('dm','particle_mass'), cmap = cmp)
            prj.set_zlim(field = ('dm','particle_mass'), zmin = 1.e37, zmax = 1.e42)
            prj.annotate_timestamp(corner='upper_left', redshift=True, draw_inset_box=True)
            if annotate_sphere:
                prj.annotate_sphere((0.5, 0.5), radius = (2, 'kpc'), coord_system='axis', circle_args={'color':'red'})
            prj.save(fig_dir + '/star_particle_projection/%s_%s_dm_particleproj_%s.png'%(haloname, axs, fig_end))

    return


def identify_satellites(sat_file, halo_center_kpc, haloname, foggie_dir, i_orients = array([(0, 'x'), (1, 'y'), (2, 'z')]), selection_props = [(0.5, 5.e5), (1.0, 1.e6)]):

    satellites = []
    sat_count = 0

    mass_stars, x_stars, y_stars, z_stars, particle_ids = load_particle_data(refine_box)
    all_stars = [x_stars, y_stars, z_stars]
    ortho_orients = [[1,2], [0,2], [0,1]]

    for (bin_size, mass_limit) in selection_props:


        xbin = np.arange(halo_center_kpc[0] - width/2., halo_center_kpc[0] + width/2. + bin_size, bin_size)
        ybin = np.arange(halo_center_kpc[1] - width/2., halo_center_kpc[1] + width/2. + bin_size, bin_size)
        zbin = np.arange(halo_center_kpc[2] - width/2., halo_center_kpc[2] + width/2. + bin_size, bin_size)

        p = np.histogramdd((x_stars, y_stars, z_stars), \
                            weights = mass_stars, \
                            bins = (xbin, ybin, zbin))

        pp = p[0]
        pp[p[0] < mass_limit] = np.nan

        for (i, orient) in i_orients:
        
            i = int(i)
            sm_im = np.log10(np.nansum(pp, axis = i))
            seg_im = detect_sources(sm_im, threshold = 0, npixels = 1, connectivity = 8)
            make_segmentation_figure(sm_im, seg_im, figname = fig_dir + '/%s_%s_satellite_selection_%.1f_%.1f.png'%(haloname, orient, bin_size, mass_limit * 1.e-6))

            for label in seg_im.labels:
                edges1 = p[1][ortho_orients[i][0]]
                edges2 = p[1][ortho_orients[i][1]]

                gd = where(seg_im.data == label)[0:10]
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

                print (label, mn_x, mn_y, mn_z)
                already_in_catalog = False
                for sat in satellites:
                    diff = np.sqrt((mn_x - sat['x'])**2. + (mn_y - sat['y'])**2. + (mn_z - sat['z'])**2.)
                    if diff.value < 1.: 
                        #print ('\t', 'match', diff.value, sat['selectid'])
                        already_in_catalog = True
                        break
                if not already_in_catalog:
                    new_satellite_dic = {}
                    new_satellite_dic['selectid'] = '%i'%(sat_count)
                    new_satellite_dic['x'] = mn_x
                    new_satellite_dic['y'] = mn_y
                    new_satellite_dic['z'] = mn_z                    
                    new_satellite_dic['ids'] = ids
                    sat_count+=1

                    satellites.append(new_satellite_dic)
            print ('\n\n')

    save_dir = foggie_dir.replace('sims', 'outputs/identify_satellites')
    np.save(sat_file, satellites)


def load_particle_data(refine_box, only_ids_ages = False):
    print ('loading star particle data...')
    print ('\t particle ids')
    particle_ids = refine_box['stars', 'particle_index']
    particle_ages = refine_box['stars', 'age'].in_units('Gyr')
    print ('\t particle masses')    
    mass_stars = refine_box['stars', 'particle_mass'].to('Msun')
    print ('\t particle positions')
    x_stars = refine_box['stars', 'particle_position_x'].to('kpc')
    y_stars = refine_box['stars', 'particle_position_y'].to('kpc')
    z_stars = refine_box['stars', 'particle_position_z'].to('kpc')

    return mass_stars, x_stars, y_stars, z_stars, particle_ids, particle_ages

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
    if args.run_all:
        #Run this in series on all of the halos
        inputs = [('2878', 'DD0581'), 
                  ('5016', 'DD0581'), 
                  ('5036', 'DD0581'),
                  ('2392', 'DD0581'),
                  ('4123', 'DD0581'),
                  ('8508', 'DD0487')]
    else:
        inputs = [(args.halo, args.output),]

    anchors = {}
    for args.halo, args.output in inputs:
        anchors[args.halo] = {}
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

        halo_center_kpc = ds.arr(halo_center, 'code_length').to('kpc').value
        width = ds.arr(x_width, 'code_length').to('kpc').value      

        fig_dir = foggie_dir.replace('sims', 'figures/identify_satellites') 
        if args.do_proj_plots:
            make_projection_plots(ds, halo_center, refine_box, x_width, fig_dir, do = ['gas', 'dm', 'stars'], axes = ['x','y','z'])

        sat_file = save_dir + '/satellite_selection_%s.npy'%(haloname)

        # Set the defined center coordinate of the box at the halo center
        refine_box.set_field_parameter('center', ds.arr(halo_center, 'code_length'))
        bulk_vel = refine_box.quantities.bulk_velocity()
        refine_box.set_field_parameter("bulk_velocity", bulk_vel)



        if args.do_identify_satellites: identify_satellites(sat_file, halo_center_kpc, haloname, foggie_dir)



        if args.do_record_anchor_particles:
            sat_cat = ascii.read(foggie_dir.replace('/sims/', '/outputs/identify_satellites/satellite_locations_wcom.cat'))
            sat_cat = sat_cat[sat_cat['halo'] == int(args.halo)]
            for s, sat in enumerate(sat_cat):
                anchors[args.halo][sat['id']] = {}
                satx = sat['x']
                saty = sat['y']
                satz = sat['z']

                sat_center = ds.arr([satx, saty, satz], 'kpc')
                sp = ds.sphere(center = sat_center, radius = 0.5*kpc)

                mass_stars, x_stars, y_stars, z_stars, particle_ids, particle_ages = load_particle_data(sp)

                srt = argsort(particle_ages)[::-1][:1000]

                anchors[args.halo][sat['id']]['ids']     = particle_ids[srt]
                anchors[args.halo][sat['id']]['masses']  = mass_stars[srt]
                anchors[args.halo][sat['id']]['ages']    = particle_ages[srt]




    if args.do_record_anchor_particles: 

        anchor_save = foggie_dir.replace('/sims/', '/outputs/identify_satellites/anchors.npy')
        np.save(anchor_save, anchors)















