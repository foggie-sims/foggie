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
import foggie_utils as futils
from consistency import new_phase_color_key, new_metals_color_key

def create_foggie_cmap():
    """ returns two colormaps"""
    ds = yt.load("""/Users/tumlinson/Dropbox/FOGGIE/outputs/halo_008508/nref11n/nref11n_nref10f_refine200kpc/RD0020/RD0020""")

    ray_start = np.array([0.49441502, 0.488119, 0.50229639])
    ray_end = np.array([0.49441502, 0.490119, 0.50229639])

    df = futils.ds_to_df(ds, ray_start, ray_end)

    phase_img = grab_cmap(df, 'y', 'z', 'phase_label', new_phase_color_key)
    metal_img = grab_cmap(df, 'y', 'z', 'metal_label', new_metals_color_key)

    return phase_img, metal_img

def grab_cmap(df, axis_to_use, second_axis, labels_to_use, color_key):
    """
    takes in a dataframe and some other info and returns the colormap image
    JT 090618
    """
    n_labels = np.size(list(color_key))
    sightline_length = np.max(df[axis_to_use]) - np.min(df[axis_to_use])

    value = np.max(df[axis_to_use])
    for index in np.flip(np.arange(n_labels), 0):
        df[labels_to_use][df[axis_to_use] > value
                          -sightline_length*
                          (1.*index+1)/n_labels] = list(color_key)[index]
    cvs = dshader.Canvas(plot_width=800, plot_height=200,
                         x_range=(np.min(df[axis_to_use]),
                                  np.max(df[axis_to_use])),
                         y_range=(np.mean(df[second_axis])-20/0.695,
                                  np.mean(df[second_axis])+20/0.695))
    agg = cvs.points(df, axis_to_use, second_axis,
                     dshader.count_cat(labels_to_use))
    cmap = tf.spread(tf.shade(agg, color_key=color_key), px=2, shape='square')
    return cmap
