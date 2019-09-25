""" a module for datashader renders of phase diagrams"""
import datashader as dshader
from datashader.utils import export_image
import datashader.transfer_functions as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import foggie.utils.prep_dataframe as prep_dataframe 
import foggie.utils.get_region as gr 
import matplotlib as mpl
mpl.use('agg')
mpl.rcParams['font.family'] = 'stixgeneral'
mpl.rcParams.update({'font.size': 14})

import yt
from yt.units import dimensions 
import trident
import numpy as np
from astropy.table import Table

import os
os.sys.path.insert(0, os.environ['FOGGIE_REPO'])
import foggie.utils as futils
import foggie.cmap_utils as cmaps
from foggie.get_halo_center import get_halo_center
import foggie.utils.get_refine_box as grb
from foggie.consistency import *

def _no6(field,data):  
    print("INSIDE THE HELPER FUNCTION THE FIELD NAME IS: ", field)
    return data["dx"] * data['O_p5_number_density']  
    
def _nh1(field,data):  
    print("INSIDE THE HELPER FUNCTION THE FIELD NAME IS: ", field)
    return data["dx"] * data['H_p0_number_density']  

def _no5(field,data):  
    print("INSIDE THE HELPER FUNCTION THE FIELD NAME IS: ", field)
    return data["dx"] * data['O_p4_number_density']  

def _c4(field,data):  
    print("INSIDE THE HELPER FUNCTION THE FIELD NAME IS: ", field)
    return data["dx"] * data['C_p3_number_density']  

def prep_dataset(fname, trackfile, ion_list=['H I'], filter="obj['temperature'] < 1e9", region='trackbox'):
    """prepares the dataset for rendering by extracting box or sphere"""
    data_set = yt.load(fname)

    trident.add_ion_fields(data_set, ions=ion_list)
                   
    data_set.add_field(("gas", "C_p3_column_density"), function=_c4, units='cm**(-2)', dimensions=dimensions.length**(-2))
    data_set.add_field(("gas", "O_p5_column_density"), function=_no6, units='cm**(-2)', dimensions=dimensions.length**(-2))
    data_set.add_field(("gas", "H_p0_column_density"), function=_nh1, units='cm**(-2)', dimensions=dimensions.length**(-2))

    track = Table.read(trackfile, format='ascii')
    track.sort('col1')
    refine_box, refine_box_center, _ = \
            grb.get_refine_box(data_set, data_set.current_redshift, track)

    print('prep_dataset: Refine box corners: ', refine_box)
    print('prep_dataset:             center: ', refine_box_center)

    if region == 'trackbox':
        print("prep_dataset: your region is the refine box")
        all_data = refine_box
    else:
        all_data = gr.get_region(data_set, region)

    print("prep_dataset: will now apply filter ", filter)
    cut_region_all_data = all_data.cut_region([filter])

    halo_vcenter = 0. 

    return data_set, cut_region_all_data, refine_box_center, halo_vcenter

def wrap_axes(dataset, img, filename, field1, field2, colorcode, ranges, region, filter):
    """intended to be run after render_image, take the image and wraps it in
        axes using matplotlib and so offering full customization."""

    img = mpimg.imread(filename+'.png')
    fig = plt.figure(figsize=(8,8),dpi=300)
    
    ax1 = fig.add_axes([0.1, 0.1, 0.85, 0.85])
    ax1.imshow(np.flip(img[:,:,0:4],0), alpha=1.)

    xstep = 1
    x_max = ranges[0][1]
    x_min = ranges[0][0]
    if (x_max > 30.): xstep = 10
    if (x_max > 100.): xstep = 100
    ax1.set_xlabel(axes_label_dict[field1], fontsize=30)
    ax1.set_xticks(np.arange((x_max - x_min) + 1., step=xstep) * 1000. / (x_max - x_min))
    ax1.set_xticklabels([ str(int(s)) for s in \
        np.arange((x_max - x_min) + 1., step=xstep) +  x_min ], fontsize=22)

    ystep = 1
    y_max = ranges[1][1]
    y_min = ranges[1][0]
    if (y_max > 30.): ystep = 10
    if (y_max > 100.): ystep = 100
    ax1.set_ylabel(axes_label_dict[field2], fontsize=30)
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

    plt.text(0.033, 0.965, 'region = '+region, transform=ax1.transAxes)
    plt.text(0.033, 0.93, 'z = '+str(np.round(dataset.current_redshift * 100.) / 100.), transform=ax1.transAxes) 

    plt.savefig(filename)

    return fig

def render_image(frame, field1, field2, count_cat, x_range, y_range, filename):
    """ renders density and temperature 'Phase' with linear aggregation"""

    cvs = dshader.Canvas(plot_width=1000, plot_height=1000, x_range=x_range, y_range=y_range)
    agg = cvs.points(frame, field1, field2, dshader.count_cat(count_cat))
    img = tf.spread(tf.shade(agg, color_key=colormap_dict[count_cat], how='eq_hist',min_alpha=50), px=0) 
    export_image(img, filename)
    return img

def summed_image(frame, field1, field2, count_cat, x_range, y_range, filename):
    """ renders density and temperature 'Phase' with linear aggregation
        CURRENTLY DOES NOT WORK. DON'T USE IT. """

    cvs = dshader.Canvas(plot_width=1000, plot_height=1000, x_range=x_range, y_range=y_range)
    #this call to 'agg' needs to change if we want a summed map, like projected column density 
    agg = cvs.points(frame, field1, field2, dshader.sum(count_cat))
    img = tf.spread(tf.shade(agg, color_key="magma", how='log', min_alpha=50), px=0) 
    export_image(img, filename)
    return img

def simple_plot(fname, trackfile, field1, field2, colorcode, ranges, outfile, region='trackbox',
                filter="obj['temperature'] < 1e9", screenfield='none', screenrange=[-99,99]):
    """This function makes a simple plot with two dataset fields plotted against
        one another. The color coding is given by variable 'colorcode'
        which can be phase, metal, or an ionization fraction"""

    dataset, all_data, halo_center, halo_vcenter = prep_dataset(fname, trackfile, \
                        ion_list=['H I','C II','C IV','Si IV','O I','O II','O III','O IV','O V','O VI','O VII','O VIII'], 
                        filter=filter, region=region)

    print("simple_plot: ", halo_center)

    if ('none' not in screenfield): 
        field_list = [field1, field2, screenfield]
    else: 
        field_list = [field1, field2]    

    data_frame = prep_dataframe.prep_dataframe(dataset, all_data, field_list, colorcode, \
                        halo_center = halo_center, halo_vcenter=halo_vcenter)

    print(data_frame.head()) 
    if ('ion_fraction' in colorcode): colorcode = 'frac'
    image = render_image(data_frame, field1, field2, colorcode, *ranges, outfile)

    # if there is to be screening of the df, it should happen here. 
    print('Within simple_plot, the screen is: ', screenfield)
    if ('none' not in screenfield): 
        mask = (data_frame[screenfield] > screenrange[0]) & (data_frame[screenfield] < screenrange[1])
        print(mask)
        image = render_image(data_frame[mask], field1, field2, colorcode, *ranges, outfile)

    wrap_axes(dataset, image, outfile, field1, field2, colorcode, ranges, region, filter)
    
    return data_frame, image, dataset

def sightline_plot(wildcards, field1, field2, colorcode, ranges, outfile):
    """ an attempt at a general facility for datashading the physical 
        varibles in our FOGGIE spectra. JT August 2019""" 

    all_sightlines = prep_dataframe.rays_to_dataframe(wildcards[0], wildcards[1], wildcards[2])
    all_sightlines = prep_dataframe.check_dataframe(all_sightlines, field1, field2, colorcode) 
    all_sightlines = prep_dataframe.check_dataframe(all_sightlines, 'metallicity', 'temperature', colorcode) 

    h1_clouds_only = all_sightlines[all_sightlines["h1_cloud_flag"] > 0]
    o6_clouds_only = all_sightlines[all_sightlines["o6_cloud_flag"] > 0]

    img = render_image(all_sightlines, field1, field2, colorcode, *ranges, outfile) 
    wrap_axes(img, outfile, field1, field2, colorcode, ranges )  

    img = render_image(h1_clouds_only, field1, field2, colorcode, *ranges, outfile+'_HI_clouds_only') 
    wrap_axes(img, outfile+'_HI_clouds_only', field1, field2, colorcode, ranges )  

    img = render_image(o6_clouds_only, field1, field2, colorcode, *ranges, outfile+'_OVI_clouds_only') 
    wrap_axes(img, outfile+'_OVI_clouds_only', field1, field2, colorcode, ranges )  


