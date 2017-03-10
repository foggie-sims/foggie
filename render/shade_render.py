import datashader as dshader
import datashader.transfer_functions as tf
import pandas as pd
import yt
import matplotlib.pyplot as plt 
import numpy as np 
from functools import partial
from datashader.utils import export_image
from datashader.colors import colormap_select, Greys9, Hot, viridis, inferno
from IPython.core.display import HTML, display


def shade_render(ds, region, halo_center, fileroot): 

    #need to give this an extacted, already centered halo sphere 

    proper_box_size = ds.get_parameter('CosmologyComovingBoxSize') / ds.get_parameter('CosmologyHubbleConstantNow') * 1000. # in kpc 

    ad = region 

    cell_vol = ad["cell_volume"]
    cell_size = np.array(cell_vol)**(1./3.)*proper_box_size 

    x_particles = ad['x'].ndarray_view() * proper_box_size + cell_size * (np.random.rand(np.size(cell_vol)) * 2. - 1. ) 
    y_particles = ad['y'].ndarray_view() * proper_box_size + cell_size * (np.random.rand(np.size(cell_vol)) * 2. - 1. ) 
    z_particles = ad['z'].ndarray_view() * proper_box_size + cell_size * (np.random.rand(np.size(cell_vol)) * 2. - 1. ) 
    dens = np.log10(ad['density'].ndarray_view()) 
    temp = np.log10(ad['temperature'].ndarray_view()) 
    ent = np.log10(ad['entropy'].ndarray_view()) 
    metal = ad['metallicity'].ndarray_view()  

    #categorize by phase 
    phase = np.chararray(np.size(dens), 4)
    phase[temp < 9.] = 'hot'
    phase[temp < 6.] = 'warm'
    phase[temp < 5.] = 'cool'
    phase[temp < 4.] = 'cold'
    phase_color_key = {'hot':'yellow', 'warm':'#4daf4a', 'cool':'#984ea3', 'cold':'salmon'}
    
    #categorize by metallicity 
    metalcode = np.chararray(np.size(metal), 4)
    metalcode[metal < 90.] = 'high' 
    metalcode[metal < 1.] = 'medi'
    metalcode[metal < 0.1] = 'low'
    metalcode[metal < 0.01] = 'vlow'
    metal_color_key = {'high':'red', 'medi':'green', 'low':'blue', 'vlow':'purple'}
    
    df = pd.DataFrame({'x':x_particles, 'y':y_particles, 'z':z_particles, \
         'temp':temp, 'dens':dens, 'phase':phase, 'entropy':ent, 'metallicity':metal, 'metalcode':metalcode}) 
    df.phase = df.phase.astype('category')
    df.metalcode = df.metalcode.astype('category')
    
    #galaxy =    ((70200.0,70800.0), (67500,68100.0)) # xy 
    galaxy =    ((halo_center[0]*proper_box_size-250.,halo_center[0]*proper_box_size+250.), (halo_center[2]*proper_box_size-250., halo_center[2]*proper_box_size+250.)) # xz
    print 'halo center ', halo_center 
    print 'ranges      ', galaxy 
    #phase = ((-32,-23),(-5,5)) 
    phase = ((-32,-23),(2,8)) # rho-T 
    
    background = "white"
    plot_width  = int(1080)
    plot_height = int(1080)
    
    # define the functions that create the images, then create them 
    def phase_image(x_range, y_range, w=plot_width, h=plot_height):
        cvs = dshader.Canvas(plot_width=w, plot_height=h, x_range=x_range, y_range=y_range)
        agg = cvs.points(df, 'dens', 'temp', dshader.count_cat('phase'))
        img = tf.colorize(agg, phase_color_key, how='eq_hist')
        return img # tf.dynspread(img, threshold=0.3, max_px=1)             
    
    def galaxy_image(x_range, y_range, w=plot_width, h=plot_height):
        cvs = dshader.Canvas(plot_width=w, plot_height=h, x_range=x_range, y_range=y_range)
        agg = cvs.points(df, 'x', 'z', dshader.count_cat('phase'))
        img = tf.colorize(agg, phase_color_key, how='eq_hist')
        return img # tf.dynspread(img, threshold=0.3, max_px=1)             
    
    export = partial(export_image, background = background)
    cm = partial(colormap_select, reverse=(background!="black"))
    export(galaxy_image(*galaxy),"Galaxy_"+fileroot) 
    export(phase_image(*phase),"Phase_"+fileroot) 

