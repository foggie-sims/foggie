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

from consistency import *

def plot(fname, trackfile):

    background = 'white'
    export = partial(export_image, background = background, export_path="export")
    cm = partial(colormap_select, reverse=(background!="black"))

    # preliminaries - get the snapshot, and its refine box, and then "all_data"
    ds=yt.load(fname)
    track = Table.read(trackfile, format='ascii')
    track.sort('col1')
    refine_box, refine_box_center, refine_width = grb(ds, ds.current_redshift, track)
    ad = refine_box

    cell_vol = ad["cell_volume"]
    cell_mass = ad["cell_volume"].in_units('kpc**3') * ad["density"].in_units('Msun / kpc**3')
    cell_size = np.array(cell_vol)**(1./3.)
 
    def scale_lvec(lvec): 
        lvec[lvec > 0.] = (np.log10(lvec[lvec > 0.]) - 25.) / 8.  
        lvec[lvec < 0.] = (-1. * np.log10(-1.*lvec[lvec < 0.]) + 25.) / 8. 
        return lvec 

    x_cells = ad['x'].ndarray_view() + cell_size * (np.random.rand(np.size(cell_vol)) * 2. - 1. )
    y_cells = ad['y'].ndarray_view() + cell_size * (np.random.rand(np.size(cell_vol)) * 2. - 1. )
    z_cells = ad['z'].ndarray_view() + cell_size * (np.random.rand(np.size(cell_vol)) * 2. - 1. )
    x_cells = (x_cells - refine_box.center[0].ndarray_view()) / (refine_width/2.)
    y_cells = (y_cells - refine_box.center[1].ndarray_view()) / (refine_width/2.)
    z_cells = (z_cells - refine_box.center[2].ndarray_view()) / (refine_width/2.)
    # obtain and manipulate gas cell angular momentum fields 
    lx = scale_lvec(ad['specific_angular_momentum_x']) 
    ly = scale_lvec(ad['specific_angular_momentum_y']) 
    lz = scale_lvec(ad['specific_angular_momentum_z']) 
    dens = np.log10(ad['density'])
    temp = np.log10(ad['temperature'])
    mass = np.log10(cell_mass) # log of the cell mass in Msun
    #categorize by phase
    phase = np.chararray(np.size(dens), 4)
    phase[temp < 9.] = 'hot'
    phase[temp < 6.] = 'warm'
    phase[temp < 5.] = 'cool'
    phase[temp < 4.] = 'cold'
    # munge the physical variables into more readily plottable values 
    dens = (dens + 24.) / 6.
    temp = (temp - 6.5) / 3.
    mass = (mass - 3.0) / 5.

    df = pd.DataFrame({'x':x_cells, 'y':y_cells, 'z':z_cells, 'lx':lx, 'ly':ly, 'lz':lz, \
         'temp':temp, 'dens':dens, 'mass': mass, 'phase':phase})
    df.phase = df.phase.astype('category')


    particle_type = ad['particle_type'] 
    dm_x = (ad['particle_position_x'].ndarray_view() - refine_box.center[0].ndarray_view()) / (refine_width/2.)
    dm_y = (ad['particle_position_y'].ndarray_view() - refine_box.center[0].ndarray_view()) / (refine_width/2.)
    dm_z = (ad['particle_position_z'].ndarray_view() - refine_box.center[0].ndarray_view()) / (refine_width/2.)
    dm_lx = ad['particle_specific_angular_momentum_x']  
    dm_ly = ad['particle_specific_angular_momentum_y']  
    dm_lz = ad['particle_specific_angular_momentum_z']  
    dm_specific_lx = scale_lvec(dm_lx)
    dm_specific_ly = scale_lvec(dm_ly)
    dm_specific_lz = scale_lvec(dm_lz)
    dm = pd.DataFrame({'x':dm_x, 'y':dm_y, 'z':dm_z, \
             'lx':dm_specific_lx, 'ly':dm_specific_ly, 'lz':dm_specific_lz, 'ptype':particle_type}) 
    dm.ptype = dm.ptype.astype('category')
    ptype_color_key = {2:'red',4:'black'} 

    limits =    ( (-1.1,1.1), (-1.0,1.0))

    plot_width  = int(1080)
    plot_height = int(1080)

    def create_image(frame, axis, x_range, y_range, w=plot_width, h=plot_height):
        cvs = dshader.Canvas(plot_width=w, plot_height=h, x_range=x_range, y_range=y_range)
        agg = cvs.points(frame, 'x', axis, dshader.count_cat('phase'))
        img = tf.shade(agg, color_key=phase_color_key, how='eq_hist')
        return img

    def create_image_dm(frame, axis, x_range, y_range, w=plot_width, h=plot_height):
        cvs = dshader.Canvas(plot_width=w, plot_height=h, x_range=x_range, y_range=y_range)
        agg = cvs.points(frame, 'x', axis, dshader.count_cat('ptype'))
        img = tf.shade(agg, color_key=ptype_color_key, how='eq_hist')
        return img

    for axis in ['lx','ly','lz']: 
        export(create_image_dm(dm, 'lx', *limits), axis+"_particles")

    frame = dm[dm['ptype'] == 4.]
    export(create_image_dm(frame, 'lx', *limits), "lx_DM")
    export(create_image_dm(frame, 'ly', *limits), "ly_DM")
    export(create_image_dm(frame, 'lz', *limits), "lz_DM")
    frame = dm[dm['ptype'] == 2.]
    export(create_image_dm(frame, 'lx', *limits), "lx_Star")
    export(create_image_dm(frame, 'ly', *limits), "ly_Star")
    export(create_image_dm(frame, 'lz', *limits), "lz_Star")

    export(create_image1(df, 'lx', *limits), "lx")
    export(create_image2(df, 'ly', *limits), "ly")
    export(create_image3(df, 'lz', *limits), "lz")

    frame = df[df['phase'] == b'hot']
    export(create_image(frame,'lx',*limits),"lx_hot")
    export(create_image(frame,'ly',*limits),"ly_hot")
    export(create_image(frame,'lz',*limits),"lz_hot")

    frame = df[df['phase'] == b'warm']
    export(create_image(frame,'lx',*limits),"lx_warm")
    export(create_image(frame,'ly',*limits),"ly_warm")
    export(create_image(frame,'lz',*limits),"lz_warm")

    frame = df[df['phase'] == b'cool']
    export(create_image(frame,'lx',*limits),"lx_cool")
    export(create_image(frame,'ly',*limits),"ly_cool")
    export(create_image(frame,'lz',*limits),"lz_cool")

    frame = df[df['phase'] == b'cold']
    export(create_image(frame,'lx',*limits),"lx_cold")
    export(create_image(frame,'ly',*limits),"ly_cold")
    export(create_image(frame,'lz',*limits),"lz_cold")


