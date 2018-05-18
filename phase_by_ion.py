import datashader as dshader
from datashader.utils import export_image
import datashader.transfer_functions as tf
import pandas as pd
import yt
import trident 
import numpy as np
from astropy.table import Table
from functools import partial
from datashader.utils import export_image
from datashader.colors import colormap_select, Greys9, Hot, viridis, inferno
from get_refine_box import get_refine_box as grb

# this will show phase diagrams with the category coding done by ion 
# fraction instead of "temperature" or whatever 

from consistency import *

def plot(fname, trackfile, ion_list=['O VI'], radius=5000.):

    background = 'white'
    plot_width  = int(1080)
    plot_height = int(1080)

    export = partial(export_image, background = background, export_path="export")
    cm = partial(colormap_select, reverse=(background != "black"))

    # preliminaries - get the snapshot, add ion field, obtain the refine box, and then "all_data"
    ds=yt.load(fname)

    trident.add_ion_fields(ds, ions=ion_list) # ask trident to add fields for the ions you want 

    track = Table.read(trackfile, format='ascii')
    track.sort('col1')
    refine_box, refine_box_center, refine_width = grb(ds, ds.current_redshift, track)
    print('Refine box : ', refine_box) 
    print('Refine box center: ', refine_box_center) 
  
    #replace with sphere 
    print('Extracting sphere of radius: ', radius) 
    sph = ds.sphere(center=refine_box_center, radius=(5000, 'kpc'))
    ad = sph 

    cell_vol = ad["cell_volume"]
    cell_mass = ad["cell_volume"].in_units('kpc**3') * ad["density"].in_units('Msun / kpc**3')
    cell_size = np.array(cell_vol)**(1./3.)
    print(cell_vol) 

    x_particles = ad['x'].ndarray_view() + cell_size * (np.random.rand(np.size(cell_vol)) * 2. - 1. )
    y_particles = ad['y'].ndarray_view() + cell_size * (np.random.rand(np.size(cell_vol)) * 2. - 1. )
    z_particles = ad['z'].ndarray_view() + cell_size * (np.random.rand(np.size(cell_vol)) * 2. - 1. )
    x_particles = (x_particles - refine_box.center[0].ndarray_view()) / (refine_width/2.)
    y_particles = (y_particles - refine_box.center[1].ndarray_view()) / (refine_width/2.)
    z_particles = (z_particles - refine_box.center[2].ndarray_view()) / (refine_width/2.)
    dens = np.log10(ad['density'])
    temp = np.log10(ad['temperature'])
    mass = np.log10(cell_mass) # log of the cell mass in Msun
    f_o6 = ad['O_p5_ion_fraction'] 
    print('X min max', np.min(x_particles), np.max(x_particles)) 
    print('Y min max', np.min(y_particles), np.max(y_particles)) 
    print('Z min max', np.min(z_particles), np.max(z_particles)) 

    #categorize by ion fraction 
    frac = np.chararray(np.size(dens), 4)
    frac[f_o6 > -10.] = 'all'
    frac[f_o6 > 0.01] = 'low' # yellow 
    frac[f_o6 > 0.1] = 'med'  # orange 
    frac[f_o6 > 0.2] = 'high' # red 
    dens = (dens + 25.) / 6.
    temp = (temp - 5.0) / 3.
    mass = (mass - 3.0) / 5.
    print(np.min(f_o6), np.max(f_o6)) 

    frac_color_key = {b'all':'black',
                      b'high':'red',
                      b'med':'yellow',
                      b'low':'green'}

    df = pd.DataFrame({'x':x_particles, 'y':y_particles, 'z':z_particles, \
         'temp':temp, 'dens':dens, 'mass': mass, 'frac':frac})
    df.frac = df.frac.astype('category')

    phase  = ((-1.1,1.1),(-1.0,1.0))
    galaxy = ((-3.1,3.1),(-3.1,3.1))

    def create_image(frame, x_range, y_range, w=plot_width, h=plot_height):
        cvs = dshader.Canvas(plot_width=w, plot_height=h, x_range=x_range, y_range=y_range)
        agg = cvs.points(frame, 'dens', 'temp', dshader.count_cat('frac'))
        img = tf.shade(agg, color_key=frac_color_key, how='linear')
        return img

    def create_image2(frame, x_range, y_range, w=plot_width, h=plot_height):
        cvs = dshader.Canvas(plot_width=w, plot_height=h, x_range=x_range, y_range=y_range)
        agg = cvs.points(frame, 'x', 'y', dshader.count_cat('frac'))
        img = tf.shade(agg, color_key=frac_color_key, how='linear')
        return img

    # all cells 
    export(create_image(df, *phase),fname[0:6]+"_O6frac_Phase_All")
    export(create_image2(df, *galaxy),fname[0:6]+"_O6frac_Proj_All")

    frame = df[df['frac'] == b'high']
    export(create_image(frame, *phase),fname[0:6]+"_O6frac_Phase_High")
    export(create_image2(frame, *galaxy),fname[0:6]+"_O6frac_Proj_High")

    frame = df[df['frac'] == b'med']
    export(create_image(frame, *phase),fname[0:6]+"_O6frac_Phase_Med")
    export(create_image2(frame, *galaxy),fname[0:6]+"_O6frac_Proj_Med")

    frame = df[df['frac'] == b'low']
    export(create_image(frame, *phase),fname[0:6]+"_O6frac_Phase_Low")
    export(create_image2(frame, *galaxy),fname[0:6]+"_O6frac_Proj_Low")


