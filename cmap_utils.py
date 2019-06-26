"""
helper functions for FOGGIE colormap creation - JT 090618
"""
import os
import numpy as np
os.sys.path.insert(0, os.environ['FOGGIE_REPO'])
import yt
import datashader as dshader
import datashader.transfer_functions as tf
os.sys.path.insert(0, os.environ['FOGGIE_REPO'])
from foggie.utils import foggie_utils as futils
import pandas as pd
import foggie.manage_path_names as pathnames
import foggie.utils.prep_dataframe as prep_dataframe 

from foggie.consistency import new_phase_color_key, new_metals_color_key, categorize_by_temp, categorize_by_metals

def create_foggie_cmap_deprecated(**kwargs):
    """ returns two colormaps"""
    
    foggie_dir, output_dir = pathnames.get_path_names() 

    ds = yt.load(foggie_dir + 'halo_008508/nref11n/natural/RD0020/RD0020')

    ray_start = np.array([0.49441502, 0.488119, 0.50229639])
    ray_end = np.array([0.49341502, 0.490119, 0.50229639])

    df = futils.ds_to_df(ds, ray_start, ray_end)

    phase_img = grab_cmap(df, 'y', 'z', 'phase_label', new_phase_color_key)
    metal_img = grab_cmap(df, 'y', 'z', 'metal_label', new_metals_color_key)

    return phase_img, metal_img

def create_foggie_cmap(): 

        x = np.random.rand(100000) 
        y = np.random.rand(100000)  
        temp = np.random.rand(100000) * 8. + 1. # log values of temperature range from 1 to 9 

        df = pd.DataFrame({})
        df['x'] = x 
        df['y'] = y 
        df['temp'] = temp 

        df['phase'] = categorize_by_temp(df['temp'])
        df.phase = df.phase.astype('category')

        df['metallicity'] = 10.**(np.random.rand(100000) * 8. - 7.) 
        df['metal'] = categorize_by_metals(df['metallicity'])
        df.metal = df.metal.astype('category')

        print(df)

        phase_img = grab_cmap(df, 'x', 'y', 'phase', new_phase_color_key) 
        metal_img = grab_cmap(df, 'x', 'y', 'metal', new_metals_color_key) 

        return phase_img, metal_img


def grab_cmap(df, axis_to_use, second_axis, labels_to_use, color_key):
    """
    takes in a dataframe and some other info and returns the colormap image
    JT 090618
    """
    n_labels = np.size(list(color_key))
    sightline_length = np.max(df[axis_to_use]) - np.min(df[axis_to_use])
    print("SIGHTLINE TO  USE:", sightline_length)

    value = np.max(df[axis_to_use])
    for index in np.flip(np.arange(n_labels), 0):
        df[labels_to_use][df[axis_to_use] > value
                          -sightline_length*
                          (1.*index+1)/n_labels] = list(color_key)[index]
    cvs = dshader.Canvas(plot_width=750, plot_height=100,
                         x_range=(np.min(df[axis_to_use]),
                                  np.max(df[axis_to_use])),
                         y_range=(np.min(df[second_axis]),
                                  np.max(df[second_axis])))
    agg = cvs.points(df, axis_to_use, second_axis,
                     dshader.count_cat(labels_to_use))
    cmap = tf.spread(tf.shade(agg, color_key=color_key), px=2, shape='square')
    return cmap
