""" a module for datashader renders of phase diagrams"""
from functools import partial
import datashader as dshader
from datashader.utils import export_image
import datashader.transfer_functions as tf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import foggie.utils.prep_dataframe as prep_dataframe 
import matplotlib as mpl
mpl.use('agg')
mpl.rcParams['font.family'] = 'stixgeneral'
mpl.rcParams.update({'font.size': 14})

import yt
import trident
import numpy as np
from astropy.table import Table

import os
os.sys.path.insert(0, os.environ['FOGGIE_REPO'])
import cmap_utils as cmaps

import foggie.utils.get_refine_box as grb
from foggie.get_halo_center import get_halo_center
from consistency import colormap_dict, axes_label_dict


def prep_dataset(fname, trackfile, ion_list=['H I'], region='trackbox'):
    """prepares the dataset for rendering by extracting box or sphere"""
    data_set = yt.load(fname)

    trident.add_ion_fields(data_set, ions=ion_list)
    for ion in ion_list:
        print("prep_dataset: Added ion "+ion+" into the dataset.")

    if ('domain' in trackfile): 
        print("prep_dataset will set the subregion to be the domain")
        refine_box = data_set.r[0:1, 0:1, 0.48:0.52]
        refine_box_center = [0.5, 0.5, 0.5]
    else: 
        track = Table.read(trackfile, format='ascii')
        track.sort('col1')
        refine_box, refine_box_center, refine_width= \
               grb.get_refine_box(data_set, data_set.current_redshift, track)

    print('prep_dataset: Refine box corners: ', refine_box)
    print('prep_dataset:             center: ', refine_box_center)

    if region == 'trackbox':
        all_data = refine_box
    elif region == 'sphere':
        sph = data_set.sphere(center=refine_box_center, radius=(500, 'kpc'))
        all_data = sph
    elif region == 'domain': 
        print("your region is the entire domain, prepare to wait")
    else:
        print("prep_dataset: your region is invalid!")

    #halo_center, halo_vcenter = get_halo_center(data_set, refine_box_center, \
    #                                    units = 'physical')

    filter = "obj['temperature'] < 1e9"
    print("Will now apply filter ", filter)
    cut_region_all_data = all_data.cut_region([filter])

    halo_center, halo_vcenter = 0., 0. 

    return cut_region_all_data, refine_box, halo_center, halo_vcenter

def wrap_axes(filename, field1, field2, colorcode, ranges):
    """intended to be run after render_image, take the image and wraps it in
        axes using matplotlib and so offering full customization."""

    img = mpimg.imread(filename+'.png')
    fig = plt.figure(figsize=(8,8),dpi=300)
    
    ax1 = fig.add_axes([0.1, 0.1, 0.85, 0.85])
    ax1.imshow(np.flip(img[:,:,0:3],0), alpha=1.)

    xstep = 1
    x_max = ranges[0][1]
    x_min = ranges[0][0]
    if (x_max > 10.): xstep = 10
    if (x_max > 100.): xstep = 100
    xtext = ax1.set_xlabel(axes_label_dict[field1], fontsize=30)
    ax1.set_xticks(np.arange((x_max - x_min) + 1., step=xstep) * 1000. / (x_max - x_min))
    ax1.set_xticklabels([ str(int(s)) for s in \
        np.arange((x_max - x_min) + 1., step=xstep) +  x_min ], fontsize=22)

    ystep = 1
    y_max = ranges[1][1]
    y_min = ranges[1][0]
    if (y_max > 10.): ystep = 10
    if (y_max > 100.): ystep = 100
    ytext = ax1.set_ylabel(axes_label_dict[field2], fontsize=30)
    ax1.set_yticks(np.arange((y_max - y_min) + 1., step=ystep) * 1000. / (y_max - y_min))
    ax1.set_yticklabels([ str(int(s)) for s in \
        np.arange((y_max - y_min) + 1., step=ystep) + y_min], fontsize=22)

    for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
                ax1.get_xticklabels() + ax1.get_yticklabels()):
                item.set_fontsize(18)

    x0,x1 = ax1.get_xlim()
    y0,y1 = ax1.get_ylim()
    ax1.set_aspect(abs(x1-x0)/abs(y1-y0))

    ax2 = fig.add_axes([0.7, 0.93, 0.25, 0.06])
    
    phase_cmap, metal_cmap = cmaps.create_foggie_cmap()

    if 'phase' in colorcode:
        ax2.imshow(np.flip(phase_cmap.to_pil(), 1))
        ax2.set_xticks([50,300,550])
        ax2.set_xticklabels(['4','5','6',' '],fontsize=11)
        ax2.text(230, 120, 'log T [K]',fontsize=13)
    elif 'metal' in colorcode:
        ax2.imshow(np.flip(metal_cmap.to_pil(), 1))
        ax2.set_xticks([36, 161, 287, 412, 537, 663])
        ax2.set_xticklabels(['-4', '-3', '-2', '-1', '0', '1'],fontsize=11)
        ax2.text(230, 120, 'log Z',fontsize=13)

    ax2.spines["top"].set_color('white')
    ax2.spines["bottom"].set_color('white')
    ax2.spines["left"].set_color('white')
    ax2.spines["right"].set_color('white')
    ax2.set_ylim(60, 180)
    ax2.set_xlim(-10, 750)
    ax2.set_yticklabels([])
    ax2.set_yticks([])

    plt.savefig(filename)




def render_image(frame, field1, field2, count_cat, x_range, y_range, filename):
    """ renders density and temperature 'Phase' with linear aggregation"""

    export = partial(export_image, background='white', export_path="./")
    cvs = dshader.Canvas(plot_width=1000, plot_height=1000,
                         x_range=x_range, y_range=y_range)

    agg = cvs.points(frame, field1, field2, dshader.count_cat(count_cat))

    img = tf.shade(agg, color_key=colormap_dict[count_cat], how='linear',min_alpha=250)

    export(img, filename)
    export(img, 'image.png')

    return img


def simple_plot(fname, trackfile, field1, field2, colorcode, ranges, outfile):
    """This function makes a simple plot with two dataset fields plotted against
        one another. The color coding is given by variable 'colorcode'
        which can be phase, metal, or an ionization fraction"""

    all_data, refine_box, halo_center, halo_vcenter = \
        prep_dataset(fname, trackfile, ion_list=['H I', 'C IV', 'Si IV', 'O VI'], region='trackbox')

    data_frame = prep_dataframe.prep_dataframe(all_data, field1, field2, colorcode, \
                        halo_center = halo_center, halo_vcenter=halo_vcenter)

    image = render_image(data_frame, field1, field2, colorcode, *ranges, outfile)

    wrap_axes(outfile, field1, field2, colorcode, ranges)
    
    return data_frame




















# code below here is in storage for later use in building rorating movies.

def cart2pol(x, y):
    return np.sqrt(x**2 + y**2), np.arctan2(y, x)

def pol2cart(rho, phi):
    return rho * np.cos(phi), rho * np.sin(phi)

def rotate_box(fname, trackfile, x1, y1, x2, y2):
    """ not yet functional"""

    print("NEED TO DO VARIABLE NORMALIZATION HERE SINCE IT IS NOT DONE ANYWEHRE ELSE NOW")
    all_data, refine_box, refine_width = \
        prep_dataset(fname, trackfile, ion_list=['H I', 'C IV', 'Si IV', 'O VI'],
                     region='sphere')

    data_frame = prep_dataframe(all_data, refine_box, refine_width)

    phase = ((-1.1, 1.1), (-1.1, 1.1))
    proj = ((-3.1, 3.1), (-3.1, 3.1))

    # this function rotates from x/y plane to density / y
    for ii in np.arange(100):
        x_center, d_center = 0.5, 0.5
        rr, phi = cart2pol(data_frame['x'] - x_center, data_frame['dens'] - d_center)
        xxxx, yyyy = pol2cart(rr, phi - np.pi / 2. / 100.)
        data_frame.x = xxxx+x_center
        data_frame.dens = yyyy+d_center
        render_image(data_frame, 'x', 'y', 'phase', *phase, 'RD0020_phase'+str(1000+ii))
        print(ii)

    # now start with dens / y and gradually turn y into temperature
    for ii in np.arange(100):
        y_center, t_center = 0.5, 0.5
        rr, phi = cart2pol(data_frame['y'] - y_center, data_frame['temperature'] - t_center)
        xxxx, yyyy = pol2cart(rr, phi - np.pi / 2. / 100.)
        data_frame.y = xxxx+y_center
        data_frame.temperature = yyyy+t_center
        render_image(data_frame, 'x', 'y', 'phase', *phase, 'RD0020_phase'+str(2000+ii))
        print(ii)
