"""
helper functions for creating colormaps for FOGGIE shade_maps - JT 090618
    functions: 
    - create_foggie_cmap creates colormaps for "phase" and "metal" colorcodes
                    with handmade dataframes with the correct ranges. 
                    add new colormaps to this if they are needed. 
    - grab_cmap is a helper to create_foggie_cmap that actually does the shading
"""
import os
import numpy as np
try: os.sys.path.insert(0, os.environ['FOGGIE_REPO']) # exception catching added by Ayan on July 14, 2021
except: pass
import yt
import datashader as dshader, pandas as pd
import datashader.transfer_functions as tf
import matplotlib as mpl
from foggie.utils.consistency import metal_min,  metal_max

from foggie.utils.consistency import new_phase_color_key, new_metals_color_key, categorize_by_temp, categorize_by_metals

def create_foggie_cmap():
    """This function creates colormaps for FOGGIE shade maps for 'phase', 'metal', and 'cell_mass' colorcodes."""
    x = np.random.rand(100000)
    y = np.random.rand(100000)
    temp = np.random.rand(100000) * 3.2 + 3.8 # log values of temperature range from 3.8 to 7 
    metallicity = np.random.rand(100000) * 0.5 - 2. 
    
    df = pd.DataFrame({})
    df['x'] = x
    df['y'] = y
    df['temperature'] = temp
    df['metallicity'] = metallicity

    df['phase'] = categorize_by_temp(df['temperature'])
    df.phase = df.phase.astype('category')

    df['metal'] = categorize_by_metals(df['metallicity'])
    df.metal = df.metal.astype('category')

    cell_mass = np.random.rand(100000) * 8. - 2.
    df['cell_mass'] = cell_mass
    df['cell_mass'] = df['cell_mass'].astype('float')

    print(df) 

    phase_img = grab_cmap(df, 'temperature', 'y', 'phase', new_phase_color_key)
    metal_img = grab_cmap(df, 'metallicity', 'y', 'metal', new_metals_color_key)
    cell_mass_img = grab_cmap(df, 'cell_mass', 'y', 'cell_mass', new_metals_color_key)

    return phase_img, metal_img, cell_mass_img


def grab_cmap(df, axis_to_use, second_axis, labels_to_use, color_key):
    """ takes in a dataframe and some other info and returns the colormap image
    JT 090618
    """
    
    cvs = dshader.Canvas(plot_width=750, plot_height=100,
                         x_range=(np.min(df[axis_to_use]),
                                  np.max(df[axis_to_use])),
                         y_range=(np.min(df[second_axis]),
                                  np.max(df[second_axis])))
    if (labels_to_use == 'cell_mass'):
        agg = cvs.points(df, axis_to_use, second_axis, dshader.mean(labels_to_use))
        cmap = tf.spread(tf.shade(agg, cmap=mpl.cm.get_cmap('icefire'), how='eq_hist',min_alpha=40), shape='square', px=2) 
    else: 
        agg = cvs.points(df, axis_to_use, second_axis, dshader.count_cat(labels_to_use))
        cmap = tf.spread(tf.shade(agg, color_key=color_key), px=2, shape='square')

    return cmap
