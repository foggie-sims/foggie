"""
helper functions for creating colormaps for FOGGIE shade_maps - JT 090618
    functions: 
    - create_foggie_cmap creates colormaps for "phase" and "metal" colorcodes
                    with handmade dataframes with the correct ranges. 
                    add new colormaps to this if they are needed. 
    - grab_cmap is a helper to create_foggie_cmap that actually does the shading
    - create_foggie_cmap_deprecated should not be used for obvious reasons.  
        - this was the original colormap maker that used an actual dataset, 
        which was very inefficient. Candidate for deletion. 

"""
import os
import numpy as np
try: os.sys.path.insert(0, os.environ['FOGGIE_REPO']) # exception catching added by Ayan on July 14, 2021
except: pass
import yt
import datashader as dshader
import datashader.transfer_functions as tf
from foggie.utils import foggie_utils as futils
import pandas as pd
import foggie.utils.prep_dataframe as prep_dataframe

from foggie.utils.consistency import new_phase_color_key, new_metals_color_key, categorize_by_temp, categorize_by_metals

def create_foggie_cmap():

    x = np.random.rand(100000)
    y = np.random.rand(100000)
    temp = np.random.rand(100000) * 3.2 + 3.8 # log values of temperature range from 3.8 to 7 

    df = pd.DataFrame({})
    df['x'] = x
    df['y'] = y
    df['temperature'] = temp

    print("inside create_foggie_cmap")
    df.head()

    df['phase'] = categorize_by_temp(df['temperature'])
    df.phase = df.phase.astype('category')

    df['metallicity'] = 10.**(np.random.rand(100000) * 8. - 7.)
    df['metal'] = categorize_by_metals(df['metallicity'])
    df.metal = df.metal.astype('category')

    phase_img = grab_cmap(df, 'temperature', 'y', 'phase', new_phase_color_key)
    metal_img = grab_cmap(df, 'x', 'y', 'metal', new_metals_color_key)

    return phase_img, metal_img


def grab_cmap(df, axis_to_use, second_axis, labels_to_use, color_key):
    """
    takes in a dataframe and some other info and returns the colormap image
    JT 090618
    """
    n_labels = np.size(list(color_key))
    sightline_length = np.max(df[axis_to_use]) - np.min(df[axis_to_use])

    value = np.max(df[axis_to_use])
    #for index in np.flip(np.arange(n_labels), 0):
    #    df[labels_to_use][df[axis_to_use] > value
    #                      -sightline_length*
    #                      (1.*index+1)/n_labels] = list(color_key)[index]
    print(df) 
    print('axis_to_use', axis_to_use) 
    print('second_axis', second_axis) 
    print('labels_to_use', labels_to_use) 
    cvs = dshader.Canvas(plot_width=750, plot_height=100,
                         x_range=(np.min(df[axis_to_use]),
                                  np.max(df[axis_to_use])),
                         y_range=(np.min(df[second_axis]),
                                  np.max(df[second_axis])))
    agg = cvs.points(df, axis_to_use, second_axis,dshader.count_cat(labels_to_use))
    cmap = tf.spread(tf.shade(agg, color_key=color_key), px=2, shape='square')
    return cmap




