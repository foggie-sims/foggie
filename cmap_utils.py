"""
helper functions for FOGGIE colormap creation - JT 090618
"""
import datashader as dshader
import numpy as np
import datashader.transfer_functions as tf

# create a function that makes a colormap dataframe here

def create_foggie_cmap(df, axis_to_use, second_axis, labels_to_use, color_key_to_use):

    n_labels = np.size(list(color_key_to_use))
    sightline_length = np.max(df[axis_to_use]) - np.min(df[axis_to_use])

    value = np.max(df[axis_to_use])
    for index in np.flip(np.arange(n_labels),0):
        df[labels_to_use][df[axis_to_use] > value-sightline_length*(1.*index+1)/n_labels] = list(color_key_to_use)[index]
    cvs = dshader.Canvas(plot_width=800, plot_height=200, x_range=(np.min(df[axis_to_use]), np.max(df[axis_to_use])),
                         y_range=(np.mean(df[second_axis])-20/0.695, np.mean(df[second_axis])+20/0.695))
    agg = cvs.points(df, axis_to_use, second_axis, dshader.count_cat(labels_to_use))
    im = tf.shade(agg, color_key=color_key_to_use)
    img = tf.spread(im, px=2, shape='square')
    return(img)