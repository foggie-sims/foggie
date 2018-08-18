"""
creates "core sample" velocity plots - JT 070318
"""

import datashader as dshader
import datashader.transfer_functions as tf
from datashader import reductions
import numpy as np
import pandas as pd
import pickle
import glob
import os
from consistency import *
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'stixgeneral'
from matplotlib.gridspec import GridSpec
os.sys.path.insert(0, os.environ['FOGGIE_REPO'])

import copy
import argparse

import foggie_utils as futils

import trident
import yt
from astropy.io import fits
from astropy.table import Table

#import shade_maps as sm
import foggie.shade_maps as sm

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

def get_sizes(ray_df, species, x, ion_to_use, cell_mass, coldens_threshold):

    print("RAY_DF INSIDE GET_SIZES", ray_df)

    threshold, number_of_cells = get_fion_threshold(ion_to_use, coldens_threshold)

    sizes = []
    masses = []
    indices = []
    xs = []
    for m in np.arange(100): # can find up to 100 peaks
        i = np.squeeze(np.where(np.array(ion_to_use) > threshold))
        if np.size(i) >= 1:
            startindex = np.min(i)
            f = ion_to_use[startindex]
            index = startindex
            ion_to_use[startindex] = 0.0
            sum_mass = cell_mass[startindex]
            count = 0
            while (f > threshold) and (index < np.size(x)-1):
                count += 1
                if (count > 10000): sys.exit('stuck in the size finding loop')
                index += 1
                if index == np.size(x): # this means we're at the edge
                    index = np.size(x)-1
                    f = 0.0
                else:
                    f = ion_to_use[index]
                    ion_to_use[index] = 0.0
                    sum_mass = sum_mass + cell_mass[index]
                if ((count % 10) == 0): print("count",count)

            sizes.append(x[startindex]-x[index])
            masses.append(sum_mass)
            indices.append(index)
            xs.append(x[index])

    size_dict = {'coldens_threshold':coldens_threshold}
    size_dict[species+'_xs'] = xs
    size_dict[species+'_indices'] = indices
    size_dict[species+'_sizes'] = sizes
    size_dict[species+'_n_cells'] = number_of_cells
    size_dict[species+'_cell_masses'] = masses
    return size_dict

def show_velphase(ds, ray_df, ray_start, ray_end, hdulist, fileroot):
    """ oh, the docstring is missing, is it??? """

    df = futils.ds_to_df(ds, ray_start, ray_end)

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
    #ax7.set_ylabel(' ')
    #ax7.set_xlabel(' ')
    ax8 = plt.subplot(gs[8])
    ax8.set_xlabel('Velocity [km / s]')
    ax9 = plt.subplot(gs[9])
    #ax9.set_xlabel(' ')
    ax7.set_yticklabels([' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '])
    ax8.set_yticklabels([' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '])
    ax9.set_yticklabels([' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '])

    # this one makes the datashaded "core sample" with phase coloring
    cvs = dshader.Canvas(plot_width=800, plot_height=200,
                         x_range=(np.min(df['x']), np.max(df['x'])),
                         y_range=(np.mean(df['y'])-CORE_WIDTH/0.695,
                                  np.mean(df['y'])+CORE_WIDTH/0.695))
    agg = cvs.points(df, 'x', 'y', dshader.count_cat('phase_label'))
    img = tf.shade(agg, color_key=new_phase_color_key)
    x_y_phase = tf.spread(img, px=2, shape='square')
    ax0.imshow(np.rot90(x_y_phase.to_pil()))

    cvs = dshader.Canvas(plot_width=800, plot_height=200,
                         x_range=(np.min(df['x']), np.max(df['x'])),
                         y_range=(np.mean(df['y'])-CORE_WIDTH/0.695,
                                  np.mean(df['y'])+CORE_WIDTH/0.695))
    agg = cvs.points(df, 'x', 'y', dshader.count_cat('metal_label'))
    img = tf.shade(agg, cmap=metal_color_map, how='log')
    x_y_metal = tf.spread(img, px=2, shape='square')
    ax1.imshow(np.rot90(x_y_metal.to_pil()))

    ytext = ax0.set_ylabel('x [comoving kpc]', fontname='Arial', fontsize=10)
    ax0.set_yticks([0, 200, 400, 600, 800])
    ax0.set_yticklabels([ str(int(s)) for s in [0, 50, 100, 150, 200]],
                        fontname='Arial', fontsize=8)

    # render x vs. vx but don't show it yet.
    cvs = dshader.Canvas(plot_width=800, plot_height=300,
                         x_range=(np.min(df['x']), np.max(df['x'])),
                         y_range=(-350, 350)) # < ----- what units?
    agg = cvs.points(df, 'x', 'vx', dshader.count_cat('phase_label'))
    x_vx_phase = tf.spread(tf.shade(agg, color_key=new_phase_color_key), shape='square')

    #now iterate over the species to get the ion fraction plots
    for species, ax in zip( ['HI', 'SiII', 'OVI'], [ax2, ax3, ax4] ):

        print("Current species: ", species)
        cvs = dshader.Canvas(plot_width=800, plot_height=300, y_range=(-350,350),
                             x_range=(ray_start[0], ray_end[0]))
        vx_render = tf.shade(cvs.points(ray_df, 'x', 'x-velocity',
                                        agg=reductions.mean(species_dict[species])),
                                        how='log')
        ray_vx = tf.spread(vx_render, px=2, shape='square')

        ax.imshow(np.rot90(x_vx_phase.to_pil()))
        ax.imshow(np.rot90(ray_vx.to_pil()))

        ax.set_xlim(0,300)
        ax.set_ylim(0,800)

    nh1 = np.sum(np.array(ray_df['dx'] * ray_df['H_p0_number_density']))
    nsi2 = np.sum(np.array(ray_df['dx'] * ray_df['Si_p1_number_density']))
    no6 = np.sum(np.array(ray_df['dx'] * ray_df['O_p5_number_density']))

    comoving_box_size = ds.get_parameter('CosmologyComovingBoxSize') \
        / ds.get_parameter('CosmologyHubbleConstantNow') * 1000.

    x_ray = (ray_df['x']-ray_start[0]) * comoving_box_size * \
                ds.get_parameter('CosmologyHubbleConstantNow') # comoving kpc
    ray_df['x_ray'] = x_ray[np.argsort(x_ray)]
    #can add stuff to the ray df instead of computing new variables

    ray_length = ds.get_parameter('CosmologyHubbleConstantNow') * \
                    comoving_box_size * (ray_end[0] - ray_start[0])

    # Add the ionization fraction traces to the datashaded velocity vs. x plots
    h1 = 50. * ray_df['H_p0_number_density']/np.max(ray_df['H_p0_number_density'])
    si2 = 50. * ray_df['Si_p1_number_density']/np.max(ray_df['Si_p1_number_density'])
    o6 = 50. * ray_df['O_p5_number_density']/np.max(ray_df['O_p5_number_density'])
    ax2.step(h1, 800. - 4. * x_ray, linewidth=0.5)
    ax3.step(si2, 800. - 4. * x_ray, linewidth=0.5)
    ax4.step(o6, 800. - 4. * x_ray, linewidth=0.5)

    vx = 300. - 300.*((ray_df['x-velocity'] + 350.) / 700.)
    ax2.step(vx[np.argsort(vx)], h1[np.argsort(vx)], linewidth=0.5, color='darkblue')
    ax3.step(vx[np.argsort(vx)], si2[np.argsort(vx)], linewidth=0.5, color='darkblue')
    ax4.step(vx[np.argsort(vx)], o6[np.argsort(vx)], linewidth=0.5, color='darkblue')

    x = np.array(ray_length - x_ray)
    cell_mass = np.array(ray_df['cell_mass'])

    h1_size_dict = get_sizes(ray_df, 'h1', x, np.array(ray_df['H_p0_number_density']), cell_mass, 0.8)
    for xx, ss in zip(h1_size_dict['h1_xs'], h1_size_dict['h1_sizes']):
        ax2.plot([50.,50.], [4. * xx, 4. * (xx+ss)], '-')
    h1_size_dict['nh1'] = nh1

    si2_size_dict = get_sizes(ray_df, 'si2', x, np.array(ray_df['Si_p1_number_density']), cell_mass, 0.8)
    for xx, ss in zip(si2_size_dict['si2_xs'], si2_size_dict['si2_sizes']):
        ax3.plot([50.,50.], [4. * xx, 4. * (xx+ss)], '-')
    si2_size_dict['nsi2'] = nsi2

    o6_size_dict = get_sizes(ray_df, 'o6', x, np.array(ray_df['O_p5_number_density']), cell_mass, 0.8)
    for xx, ss in zip(o6_size_dict['o6_xs'], o6_size_dict['o6_sizes']):
        ax4.plot([50.,50.], [4. * xx, 4. * (xx+ss)], '-')
    o6_size_dict['no6'] = no6

    # concatenate the dictionaries for the various species
    size_dict = {**h1_size_dict, **si2_size_dict, **o6_size_dict}
    pickle.dump( size_dict, open( fileroot+"sizes.pkl", "wb" ) )

    for ax, key in zip([ax7, ax8, ax9], ['H I 1216', 'Si II 1260', 'O VI 1032']):
        ax.set_xlim(-350,350)
        ax.set_ylim(0,1)
        ax.set_yticklabels([' ',' ',' '])
        if (hdulist.__contains__(key)):
            restwave = hdulist[key].header['RESTWAVE']
            vel = (hdulist[key].data['wavelength']/(1.+\
                ds.get_parameter('CosmologyCurrentRedshift')) - restwave) / restwave * c_kms
            ax.step(vel, hdulist[key].data['flux'])

    ax0.set_xticklabels([' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '])
    ax1.set_xticklabels([' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '])

    for ax in [ax2, ax3, ax4, ax5, ax6]:
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])

    ax1.set_yticks([])
    ax0.set_xticks([])
    ax1.set_xticks([])
    ax0.set_xlim(60,140)
    ax1.set_xlim(60,140)

    gs.update(hspace=0.0, wspace=0.1)
    plt.savefig(fileroot+'velphase.png', dpi=300)
    plt.close(fig)

def grab_ray_file(ds, filename):
    """
    opens a fits file containing a FOGGIE / Trident ray and returns a dataframe
    with useful quantities along the ray
    """
    print("grab_ray_file is opening: ", filename)
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
    rs = rs.ndarray_view()
    re = re.ndarray_view()
    ray['x-velocity'] = ray['x-velocity'].convert_to_units('km/s')
    ray['y-velocity'] = ray['y-velocity'].convert_to_units('km/s')
    ray['z-velocity'] = ray['z-velocity'].convert_to_units('km/s')
    ray['dx'] = ray['dx'].convert_to_units('cm')

    ray_df = ray.to_dataframe(["x", "y", "z", "density", "temperature",
                            "metallicity", "HI_Density", "cell_mass", "dx",
                            "x-velocity", "y-velocity", "z-velocity",
                            "C_p2_number_density", "C_p3_number_density",
                            "H_p0_number_density", "Mg_p1_number_density",
                            "O_p5_number_density", "Si_p2_number_density",
                            "Si_p1_number_density", "Si_p3_number_density",
                            "Ne_p7_number_density"])

    ray_df = ray_df.sort_values(by=['x'])

    return ray_df, rs, re, hdulist

def loop_over_rays(ds, dataset_list):
    for filename in dataset_list:
        ray_df, rs, re, hdulist = grab_ray_file(ds, filename)
        show_velphase(ds, ray_df, rs, re, hdulist, filename.strip('los.fits.gz'))

def drive_velphase(ds_name, wildcard):
    """
    for running as imported module.
    """
    ds = yt.load(ds_name)
    trident.add_ion_fields(ds, ions=['Si II', 'Si III', 'Si IV', 'C II',
                    'C III', 'C IV', 'O VI', 'Mg II', 'Ne VIII'])

    dataset_list = glob.glob(os.path.join(os.getcwd(), wildcard))
    loop_over_rays(ds, dataset_list)

if __name__ == "__main__":
    """
    for running at the command line.
    """
    args = futils.parse_args()
    ds_loc, output_path, output_dir, haloname = futils.get_path_info(args)

    dataset_list = glob.glob(os.path.join(output_dir, '*rd0020*v5_los*fits.gz'))

    ds = yt.load(ds_loc)
    trident.add_ion_fields(ds, ions=['Si II', 'Si III', 'Si IV', 'C II',
                    'C III', 'C IV', 'O VI', 'Mg II', 'Ne VIII'])

    loop_over_rays(ds, dataset_list)
