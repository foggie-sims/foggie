import datashader as dshader
from datashader.utils import export_image
import datashader.transfer_functions as tf
import pandas as pd
import yt
import numpy as np
from astropy.table import Table
from functools import partial
from datashader.utils import export_image
from datashader.colors import colormap_select, Greys9, Hot, viridis, inferno
from get_refine_box import get_refine_box as grb

# this will show phase diagrams with the category coding done by ion 
# fraction instead of "temperature" or whatever 


from consistency import *

def plot(fname, trackfile, ion=['O VI']):

    background = 'white'
    plot_width  = int(1080)
    plot_height = int(1080)

    export = partial(export_image, background = background, export_path="export")
    cm = partial(colormap_select, reverse=(background!="black"))

    # preliminaries - get the snapshot, and its refine box, and then "all_data"
    ds=yt.load(fname)

    trident.add_ion_fields(ds, ions=ion_list) # ask trident to add fields for the ions you want 
    #lines = ['O_p5_number_density'] 

    track = Table.read(trackfile, format='ascii')
    track.sort('col1')
    refine_box, refine_box_center, refine_width = grb(ds, ds.current_redshift, track)
    print('Refine box : ', refine_box) 
    print('Refine box center: ', refine_box_center) 
    ad = refine_box

    cell_vol = ad["cell_volume"]
    cell_mass = ad["cell_volume"].in_units('kpc**3') * ad["density"].in_units('Msun / kpc**3')
    cell_size = np.array(cell_vol)**(1./3.)

    x_particles = ad['x'].ndarray_view() + cell_size * (np.random.rand(np.size(cell_vol)) * 2. - 1. )
    y_particles = ad['y'].ndarray_view() + cell_size * (np.random.rand(np.size(cell_vol)) * 2. - 1. )
    z_particles = ad['z'].ndarray_view() + cell_size * (np.random.rand(np.size(cell_vol)) * 2. - 1. )
    x_particles = (x_particles - refine_box.center[0].ndarray_view()) / (refine_width/2.)
    y_particles = (y_particles - refine_box.center[1].ndarray_view()) / (refine_width/2.)
    z_particles = (z_particles - refine_box.center[2].ndarray_view()) / (refine_width/2.)
    dens = np.log10(ad['density'])
    temp = np.log10(ad['temperature'])
    mass = np.log10(cell_mass) # log of the cell mass in Msun

    if (False):
        #categorize by metallicity'
        metallicity = ad['PartType0', 'Metallicity']
        metal_label = np.chararray(np.size(dens), 5)
        metal_label[metallicity < 10.] = 'high'
        metal_label[metallicity < 0.005] = 'solar'
        metal_label[metallicity < 0.000001] = 'low'
        metal_label[metallicity < 0.0000001] = 'poor'
        metal_color_key = {'high':'yellow', 'solar':'green', 'low':'purple', 'poor':'salmon'}

    #categorize by phase
    phase = np.chararray(np.size(dens), 4)
    phase[temp < 9.] = 'hot'
    phase[temp < 6.] = 'warm'
    phase[temp < 5.] = 'cool'
    phase[temp < 4.] = 'cold'
    dens = (dens + 24.) / 6.
    temp = (temp - 6.5) / 3.
    mass = (mass - 3.0) / 5.

    df = pd.DataFrame({'x':x_particles, 'y':y_particles, 'z':z_particles, \
         'temp':temp, 'dens':dens, 'mass': mass, 'phase':phase})
    df.phase = df.phase.astype('category')
    #df.metal_label = df.metal_label.astype('category')

    galaxy = ((-1.1,1.1),(-1.0,1.0))

    def create_image(frame, x_range, y_range, w=plot_width, h=plot_height):
        cvs = dshader.Canvas(plot_width=w, plot_height=h, x_range=x_range, y_range=y_range)
        agg = cvs.points(frame, 'x', 'y', dshader.count_cat('phase'))
        img = tf.shade(agg, color_key=phase_color_key, how='eq_hist')
        return img

    def create_image2(frame, x_range, y_range, w=plot_width, h=plot_height):
        cvs = dshader.Canvas(plot_width=w, plot_height=h, x_range=x_range, y_range=y_range)
        agg = cvs.points(frame, 'x', 'mass', dshader.count_cat('phase'))
        raw = tf.shade(agg, color_key=phase_color_key, how='eq_hist')
        img = tf.spread(raw, px=1, shape='square')
        return img

    export(create_image2(df, *galaxy),"Phase_Alls0")

    frame = df[df['phase'] == b'hot']
    export(create_image2(frame, *galaxy),"Phase_Hot")

    frame = df[df['phase'] == b'warm']
    export(create_image2(frame, *galaxy),"Phase_Warm")

    frame = df[df['phase'] == b'cool']
    export(create_image2(frame, *galaxy),"Phase_Cool")

    frame = df[df['phase'] == b'cold']
    export(create_image2(frame, *galaxy),"Phase_Cold")