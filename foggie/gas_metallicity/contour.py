#!/usr/bin/env python3

"""

    Title :      contour
    Notes :      Pick out a contour from a 2D array based on a threshold, and overplot that contour on a different plot
    Output :     Plots as png files
    Author :     Ayan Acharyya
    Started :    Mar 2024
    Examples :   run contour.py

"""
import os
HOME = os.getenv('HOME')
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure

# ------------generating the random shape (10x10)------------------------
random_shape = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 3, 2, 0, 0, 0, 0],
                         [0, 0, 0, 0, 5, 2, 0, 0, 0, 0],
                         [0, 0, 0, 0, 4, 1, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]) # one can randomise this by changing location/number of non-zero elements
random_shape = np.ma.masked_where(random_shape == 0, random_shape) # masking zero values so that they don't appear in the plot

# ------------generating the background shape (10x10)------------------------
x, y = np.meshgrid(np.linspace(-3, 3, 10), np.linspace(-3, 3, 10)) # getting x,y coordinates
background_shape = 1./np.sqrt(x**2 + y**2)  # making a smooth circular shape, with higher values in the center
background_shape = np.ma.masked_where(background_shape == 0, background_shape) # masking zero values so that they don't appear in the plot

# --------- making the figure---------------------
cmap, clim = 'rainbow', [0, 5]
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(4, 6), sharex=True)
fig.subplots_adjust(top=0.98, bottom=0.08, left=0.17, right=0.8, hspace=0.05, wspace=0.05)

im1 = axes[0].imshow(random_shape, cmap=cmap, vmin=clim[0], vmax=clim[1])
im2 = axes[1].imshow(background_shape, cmap=cmap, vmin=clim[0], vmax=clim[1])

# ------------picking the contour around the random shape------------------------
contour = measure.find_contours(random_shape, level = 0, fully_connected='high', positive_orientation='high')[0]
axes[0].plot(contour[:, 1], contour[:, 0], linewidth=2, c='k') # sanity check to see if contours based on random_shape are reasonable
axes[1].plot(contour[:, 1], contour[:, 0], linewidth=2, c='k') # using the same contours to overplot on background_shape

# ------------figure asthetics------------------------
cbar_ax = fig.add_axes([0.82, 0.09, 0.05, 0.88])
cbar = fig.colorbar(im2, cax=cbar_ax)

axes[0].set_yticklabels(axes[0].get_yticks() - np.shape(background_shape)[0]/2 + 1)
axes[1].set_yticklabels(axes[1].get_yticks() - np.shape(background_shape)[0]/2 + 1)
axes[1].set_xticklabels(axes[0].get_xticks() - np.shape(background_shape)[0]/2 + 1)

axes[0].set_ylabel('Offset')
axes[1].set_ylabel('Offset')
axes[1].set_xlabel('Offset')
cbar.set_label('Quantity')

fig.savefig(HOME + '/Downloads/contour_tests.png')
plt.show(block=False)
