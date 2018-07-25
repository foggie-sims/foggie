"""
creates "core sample" velocity plots
JT 070318
"""
import datashader as dshader
import datashader.transfer_functions as tf
from datashader import reductions
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'stixgeneral'
from matplotlib.gridspec import GridSpec
from consistency import *

import copy

import argparse

import trident
import yt
from astropy.io import fits
from astropy.table import Table

CORE_WIDTH = 20.

def get_fion_threshold(ion_to_use, coldens_fraction):
    cut = 0.999
    total = np.sum(ion_to_use)
    ratio = 0.01
    while ratio < coldens_fraction:
        part = np.sum(ion_to_use[ion_to_use > cut * np.max(ion_to_use)])
        ratio = part / total
        cut = cut - 0.01

    threshold = cut * np.max(ion_to_use)
    number_of_cells_above_threshold = np.size(np.where(ion_to_use > threshold))

    return threshold, number_of_cells_above_threshold




def get_sizes(x, ion_to_use, coldens_threshold):

    threshold, number_of_cells = get_fion_threshold(ion_to_use, coldens_threshold)

    sizes = []
    indices = []
    xs = []
    for m in np.arange(100):
        i = np.squeeze(np.where(np.array(ion_to_use) > threshold))
        if np.size(i) >= 1:
            startindex = np.min(i)
            f = ion_to_use[startindex]
            index = startindex
            ion_to_use[startindex] = 0.0
            count = 0
            while f > threshold:
                count += 1
                index += 1
                if index == np.size(x): # this means we're at the edge
                    index = np.size(x)-1
                    f = 0.0
                else:
                    f = ion_to_use[index]
                    ion_to_use[index] = 0.0
                if ((count % 10) == 0): print("count",count)

            sizes.append(x[startindex]-x[index])
            indices.append(index)
            xs.append(x[index])

    return xs, indices, sizes, number_of_cells



def show_velphase(ds, ray_df, ray_start, ray_end, hdulist, fileroot):
    """ the docstring is missing, is it??? """

    ray_s = ray_start.ndarray_view()
    ray_e = ray_end.ndarray_view()

    # take out a "core sample" along the ray with a width given by core_width
    proper_box_size = ds.get_parameter('CosmologyComovingBoxSize') \
        / ds.get_parameter('CosmologyHubbleConstantNow') * 1000. # in kpc
    print("PROPER BOX SIZE : ", proper_box_size)
    current_redshift = ds.get_parameter('CosmologyCurrentRedshift')
    print("Current Redshift = ", current_redshift)
    all_data = ds.r[ray_s[0]:ray_e[0],
                    ray_s[1]-0.5*CORE_WIDTH/proper_box_size:ray_s[1]+
                    0.5*CORE_WIDTH/proper_box_size,
                    ray_s[2]-0.5*CORE_WIDTH/proper_box_size:ray_s[2]+
                    0.5*CORE_WIDTH/proper_box_size]

    dens = np.log10(all_data['density'].ndarray_view())
    temp = np.log10(all_data['temperature'].ndarray_view())
    metallicity = all_data['metallicity'].ndarray_view()

    phase_label = categorize_by_temp(temp)
    metal_label = categorize_by_metallicity(metallicity)

    df = pd.DataFrame({'x':all_data['x'].ndarray_view() * proper_box_size,
                       'y':all_data['y'].ndarray_view() * proper_box_size,
                       'z':all_data['z'].ndarray_view() * proper_box_size,
                       'vx':all_data["x-velocity"].in_units('km/s'),
                       'vy':all_data["y-velocity"].in_units('km/s'),
                       'vz':all_data["z-velocity"].in_units('km/s'),
                       'temp':temp, 'dens':dens, 'phase_label':phase_label,
                       'metal_label':metal_label})
    df.phase_label = df.phase_label.astype('category')
    df.metal_label = df.metal_label.astype('category')

    #establish the grid of plots and obtain the axis objects
    fig = plt.figure(figsize=(8,6))
    gs = GridSpec(2, 5, width_ratios=[1, 1, 5, 5, 5], height_ratios=[4, 1])
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[2])
    ax2.set_title('HI Lya')
    ax3 = plt.subplot(gs[3])
    ax3.set_title('Si II 1260')
    ax4 = plt.subplot(gs[4])
    ax4.set_title('O VI 1032')
    ax5 = plt.subplot(gs[5])
    ax5.spines["top"].set_color('white')
    ax5.spines["bottom"].set_color('white')
    ax5.spines["left"].set_color('white')
    ax5.spines["right"].set_color('white')
    ax6 = plt.subplot(gs[6])
    ax6.spines["top"].set_color('white')
    ax6.spines["bottom"].set_color('white')
    ax6.spines["left"].set_color('white')
    ax6.spines["right"].set_color('white')
    ax7 = plt.subplot(gs[7])
    ax7.set_ylabel('Flux')
    ax7.set_xlabel(' ')
    ax8 = plt.subplot(gs[8])
    ax8.set_xlabel('Velocity [km / s]')
    ax9 = plt.subplot(gs[9])
    ax9.set_xlabel(' ')

    # this one makes the datashaded "core sample" with phase coloring
    cvs = dshader.Canvas(plot_width=800, plot_height=200,
                         x_range=(np.min(df['x']), np.max(df['x'])),
                         y_range=(np.mean(df['y'])-CORE_WIDTH/0.695,
                                  np.mean(df['y'])+CORE_WIDTH/0.695))
    agg = cvs.points(df, 'x', 'y', dshader.count_cat('phase_label'))
    img = tf.shade(agg, color_key=phase_color_key)
    x_y_phase = tf.spread(img, px=2, shape='square')

    cvs = dshader.Canvas(plot_width=800, plot_height=200,
                         x_range=(np.min(df['x']), np.max(df['x'])),
                         y_range=(np.mean(df['y'])-CORE_WIDTH/0.695,
                                  np.mean(df['y'])+CORE_WIDTH/0.695))
    agg = cvs.points(df, 'x', 'y', dshader.count_cat('metal_label'))
    img = tf.shade(agg, cmap = metal_color_map, how='log')
    x_y_metal = tf.spread(img, px=2, shape='square')

    ax0.imshow(np.rot90(x_y_phase.to_pil()))

    ytext = ax0.set_ylabel('x [comoving kpc]', fontname='Arial', fontsize=10)
    ax0.set_yticks([0, 200, 400, 600, 800])
    ax0.set_yticklabels([ str(int(s)) for s in [0, 50, 100, 150, 200]],
                        fontname='Arial', fontsize=8)
    ax0.set_xticks([0, 100, 200])
    ax0.set_xticklabels([ str(s) for s in [-50, 0, 50]], fontname='Arial',
                        fontsize=8)

    ax1.set_yticklabels([' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '])
    ax1.set_xticks([0, 100, 200])
    ax1.set_xticklabels([ str(s) for s in [-50, 0, 50]], fontname='Arial',
                        fontsize=8)

    ax1.imshow(np.rot90(x_y_metal.to_pil()))

    # render x vs. vx but don't show it yet.
    cvs = dshader.Canvas(plot_width=800, plot_height=300,
                         x_range=(np.min(df['x']), np.max(df['x'])),
                         y_range=(-300, 300)) # < ----- what units?
    agg = cvs.points(df, 'x', 'vx', dshader.count_cat('phase_label'))
    x_vx_phase = tf.spread(tf.shade(agg, color_key=phase_color_key), shape='square')


    #now iterate over the species to get the ion fraction plots
    for species, ax, lax in zip(['HI', 'SiII', 'OVI'], [ax2, ax3, ax4], [ax7, ax8, ax9]):

        print("Current species: ", species)
        cvs = dshader.Canvas(plot_width=800, plot_height=300,
                             x_range=(ray_s[0], ray_e[0]),
                             y_range=(-300,300))
        vx_render = tf.shade(cvs.points(ray_df, 'x', 'x-velocity',
                                        agg=reductions.mean(species_dict[species])),
                                        how='log')
        ray_vx = tf.spread(vx_render, px=2, shape='square')

        ax.imshow(np.rot90(x_vx_phase.to_pil()))
        ax.imshow(np.rot90(ray_vx.to_pil()))

        ax.set_xlim(0,300)
        ax.set_ylim(0,800)


    x_ray = ds.get_parameter('CosmologyHubbleConstantNow') * 8 * 0.001 * \
                proper_box_size * (ray_df['x']-ray_s[0]) / (ray_e[0] - ray_s[0])
    print('x_ray: ', np.min(x_ray), np.max(x_ray))

    comoving_x = (ray_df['x']-ray_s[0]) * proper_box_size * ds.get_parameter('CosmologyHubbleConstantNow')
    print('comoving_x: ', np.min(comoving_x), np.max(comoving_x))
    x_ray = comoving_x

    h1 = 40. * ray_df['H_p0_number_density']/np.max(ray_df['H_p0_number_density'])
    si2 = 40. * ray_df['Si_p1_number_density']/np.max(ray_df['Si_p1_number_density'])
    o6 = 40. * ray_df['O_p5_number_density']/np.max(ray_df['O_p5_number_density'])
    ax2.step(h1[np.argsort(x_ray)], 800. - 4. * x_ray[np.argsort(x_ray)], linewidth=0.5)
    ax3.step(si2[np.argsort(x_ray)], 800. - 4. * x_ray[np.argsort(x_ray)], linewidth=0.5)
    ax4.step(o6[np.argsort(x_ray)], 800. - 4. * x_ray[np.argsort(x_ray)], linewidth=0.5)

    ######
    x = np.array(200. - x_ray[np.argsort(x_ray)])
    h1  = np.array(ray_df['H_p0_number_density'][np.argsort(x_ray)])
    si2 = np.array(ray_df['Si_p1_number_density'][np.argsort(x_ray)])
    o6  = np.array(ray_df['O_p5_number_density'][np.argsort(x_ray)])

    #get the cloud sizes for the top 80% of the column density
    print("About to attempt size stuff:")
    size_dict = {'coldens_threshold':0.8}
    xs, indices, sizes, n_cells = get_sizes(x, h1, 0.8)
    size_dict['h1_xs'] = xs
    size_dict['h1_indices'] = indices
    size_dict['h1_sizes'] = sizes
    size_dict['h1_n_cells'] = n_cells
    print('H I cloud positions: ', xs)
    print('H I cloud sizes: ', sizes)
    print("Number of cells contributing to 80 percent column density: ", n_cells)
    for xx, ss in zip(xs, sizes):
        ax2.plot([50.,50.], [4. * xx, 4. * (xx+ss)], '-')

    xs, indices, sizes, n_cells = get_sizes(x, o6, 0.8)
    size_dict['o6_xs'] = xs
    size_dict['o6_indices'] = indices
    size_dict['o6_sizes'] = sizes
    size_dict['o6_n_cells'] = n_cells
    print('O VI cloud positions: ', xs)
    print('O VI cloud sizes: ', sizes)
    for xx, ss in zip(xs, sizes):
        ax4.plot([50.,50.], [4. * xx, 4. * (xx+ss)], '-')
    print("Number of cells contributing to 80 percent column density: ", n_cells)

    pickle.dump( size_dict, open( fileroot+"_sizes.pkl", "wb" ) )

    fion = Table([x, h1, si2, o6], names=('x','h1','si2','o6') )
    fion.write(fileroot+'_fion.fits', overwrite=True)
    ######

    restwave = hdulist['H I 1216'].header['RESTWAVE']
    vel = (hdulist['H I 1216'].data['wavelength']/(1.+current_redshift) - restwave) / restwave * c_kms
    ax7.step(vel, hdulist['H I 1216'].data['flux'])
    ax7.set_xlim(-300,300)
    ax7.set_ylim(0,1)
    ax7.set_yticklabels(['0','0.5',''])

    restwave = hdulist['Si II 1260'].header['RESTWAVE']
    vel = (hdulist['Si II 1260'].data['wavelength']/(1.+current_redshift) - restwave) / restwave * c_kms
    ax8.step(vel, hdulist['Si II 1260'].data['flux'])
    ax8.set_xlim(-300,300)
    ax8.set_ylim(0,1)
    ax8.set_yticklabels([' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '])

    restwave = hdulist['O VI 1032'].header['RESTWAVE']
    vel = (hdulist['O VI 1032'].data['wavelength']/(1.+current_redshift) - restwave) / restwave * c_kms
    ax9.step(vel, hdulist['O VI 1032'].data['flux'])
    ax9.set_xlim(-300,300)
    ax9.set_ylim(0,1)
    ax9.set_yticklabels([' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '])


    ax0.set_xticklabels([' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '])
    ax1.set_xticklabels([' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '])
    ax2.axes.get_xaxis().set_ticks([])
    ax2.axes.get_yaxis().set_ticks([])
    ax3.axes.get_xaxis().set_ticks([])
    ax3.axes.get_yaxis().set_ticks([])
    ax4.axes.get_xaxis().set_ticks([])
    ax4.axes.get_yaxis().set_ticks([])
    ax5.axes.get_xaxis().set_ticks([])
    ax5.axes.get_yaxis().set_ticks([])
    ax6.axes.get_xaxis().set_ticks([])
    ax6.axes.get_yaxis().set_ticks([])

    gs.update(hspace=0.0, wspace=0.1)
    plt.savefig(fileroot+'_velphase.png', dpi=300)
    plt.close(fig)


def parse_args():
    '''
    Parse command line arguments.  Returns args object.
    '''
    parser = argparse.ArgumentParser(description="extracts spectra from refined region")

    parser.add_argument('--run', metavar='run', type=str, action='store',
                        help='which run? default is nref9f')
    parser.set_defaults(run="nref9f")

    parser.add_argument('--output', metavar='output', type=str, action='store',
                        help='which output? default is RD0020')
    parser.set_defaults(output="RD0020")

    parser.add_argument('--system', metavar='system', type=str, action='store',
                        help='which system are you on? default is oak')
    parser.set_defaults(system="oak")

    parser.add_argument('--fitsfile', metavar='fitsfile', type=str, action='store',
                        help='what fitsfile would you like to read in? this does not work yet')


    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()
    if args.system == "oak":
        ds_base = "/astro/simulations/FOGGIE/"
        output_path = "/Users/molly/Dropbox/foggie-collab/"
    elif args.system == "dhumuha" or args.system == "palmetto":
        ds_base = "/Users/molly/foggie/"
        output_path = "/Users/molly/Dropbox/foggie-collab/"
    elif args.system == "harddrive":
        ds_base = "/Volumes/foggie/"
        output_path = "/Users/molly/Dropbox/foggie-collab/"
    elif args.system == "pancho":
        ds_base = "/Users/tumlinson/Dropbox/foggie-test/"
        output_path = "/Users/tumlinson/Dropbox/foggie-test/"
    elif args.system == "lefty":
        ds_base = "/Users/tumlinson/Dropbox/foggie-test/"
        output_path = "/Users/tumlinson/Dropbox/foggie-test/"

    if args.run == "natural":
        ds_loc = ds_base + "halo_008508/nref11n/natural/" + args.output + "/" + args.output
        output_dir = output_path + "plots_halo_008508/nref11n/natural/spectra/"
        haloname = "halo008508_nref11n"
    elif args.run == "nref10f":
        ds_loc =  ds_base + "halo_008508/nref11n/nref11n_nref10f_refine200kpc/" + args.output + "/" + args.output
        output_dir = output_path + "plots_halo_008508/nref11n/nref11n_nref10f_refine200kpc/spectra/"
        haloname = "halo008508_nref11n_nref10f"
    elif args.run == "nref9f":
        path_part = "halo_008508/nref11n/nref11n_"+args.run+"_refine200kpc/"
        ds_loc =  ds_base + path_part + args.output + "/" + args.output
        output_dir = output_path + "plots_"+path_part+"spectra/"
        haloname = "halo008508_nref11n_nref9f"
    elif args.run == "nref11f":
        ds_loc =  ds_base + "halo_008508/nref11n/nref11f_refine200kpc/" + args.output + "/" + args.output
        output_dir = output_path + "plots_halo_008508/nref11n/nref11f_refine200kpc/spectra/"
        haloname = "halo008508_nref11f"

    ds = yt.load(ds_loc)
    trident.add_ion_fields(ds, ions=['Si II', 'Si III', 'Si IV', 'C II',
                    'C III', 'C IV', 'O VI', 'Mg II', 'Ne VIII'])

    dataset_list = ['hlsp_misty_foggie_halo008508_nref11n_nref10f_rd0018_axx_i010.4-a2.25_v5_los.fits.gz']

    for filename in dataset_list:
        print("opening ", filename)
        hdulist = fits.open(filename)
        ray_start_str, ray_end_str = hdulist[0].header['RAYSTART'], hdulist[0].header['RAYEND']
        ray_start = [float(ray_start_str.split(",")[0].strip('unitary')), \
               float(ray_start_str.split(",")[1].strip('unitary')), \
               float(ray_start_str.split(",")[2].strip('unitary'))]
        ray_end = [float(ray_end_str.split(",")[0].strip('unitary')), \
               float(ray_end_str.split(",")[1].strip('unitary')), \
               float(ray_end_str.split(",")[2].strip('unitary'))]
        rs, re = np.array(ray_start), np.array(ray_end)
        rs = ds.arr(rs, "code_length")
        re = ds.arr(re, "code_length")
        ray = ds.ray(rs, re)
        ray['x-velocity'] = ray['x-velocity'].convert_to_units('km/s')
        ray['y-velocity'] = ray['y-velocity'].convert_to_units('km/s')
        ray['z-velocity'] = ray['z-velocity'].convert_to_units('km/s')
        ray_df = ray.to_dataframe(["x", "y", "z", "density", "temperature",
                                "metallicity", "HI_Density",
                                "x-velocity", "y-velocity", "z-velocity",
                                "C_p2_number_density", "C_p3_number_density",
                                "H_p0_number_density",
                                "Mg_p1_number_density", "O_p5_number_density",
                                "Si_p2_number_density",
                                "Si_p1_number_density", "Si_p3_number_density",
                                "Ne_p7_number_density"])
        fileroot = filename.strip('_los.fits.gz')
        show_velphase(ds, ray_df, rs, re, hdulist, fileroot)
        hdulist.close()
