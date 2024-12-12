#!/usr/bin/env python3

"""

    Title :      binned_profile.py
    Notes :      script to test some utility functions related to binned data
    Author :     Ayan Acharyya
    Started :    Sep 2023
"""
from scipy import stats
from matplotlib import pyplot as plt
import numpy as np

# ---------------------------------------------------------
def remove_nans(list1, list2):
    '''
    Function to remove NaNs from all lists provided
    Returns shortened lists
    '''
    good_indices = np.array(np.logical_not(np.logical_or(np.isnan(list1), np.isnan(list2))))
    list1 = list1[good_indices]
    list2 = list2[good_indices]
    return list1, list2

# -----------------------------------------------------
def make_profile_from_map(map, physical_width):
    '''
    Function to get a radial profile based on a given 2D map; assumes map is square
    Returns 2 lists: quantities (y) and distance from the center (x) so that y can be plotted vs x (externally) to get a radial profile
    '''
    ncells = np.shape(map)[0] # number of cells
    kpc_per_pix = physical_width / ncells # width in kpc
    center_pix = ncells/2
    distance_map = np.array([[np.sqrt((i - center_pix)**2 + (j - center_pix)**2) for j in range(ncells)] for i in range(ncells)]) * kpc_per_pix # kpc

    distance_list = distance_map.flatten()
    map_list = map.flatten()

    sorted_group = np.array(sorted(zip(distance_list, map_list))) # sorting according to distance
    distance_list, map_list = sorted_group[:,0], sorted_group[:,1]

    distance_list, map_list = remove_nans(distance_list, map_list) # removing nans

    return distance_list, map_list

# -------------------------------------------------------
def make_binned_profile(distance_list, quant_list, bin_resolution):
    '''
    Function to make binned radial profile from a given full radial profile
    Outputs: 2 lists -- one for centers of the radial bins and the other for corresponding average data
    '''
    max_distance = np.max(distance_list)
    nbins = int(max_distance / bin_resolution)
    bin_edges = np.linspace(0, max_distance, nbins + 1)
    bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2

    binned_quant = stats.binned_statistic(distance_list, quant_list, statistic='mean', bins=bin_edges).statistic
    bin_centers, binned_quant = remove_nans(bin_centers, binned_quant)

    return bin_centers, binned_quant

# -----main code-----------------
if __name__ == '__main__':

    map = np.load('/Users/acharyya/Downloads/bla.npy') # loading the 2D data
    map = np.nan_to_num(map, nan=0) # replacing nans with zeros
    box_width, bin_width = 150, 1 # kpc

    distance_list, map_list = make_profile_from_map(map, box_width)
    bin_centers, binned_quant = make_binned_profile(distance_list, map_list, bin_width)

    plt.scatter(bin_centers, binned_quant)
    plt.show(block=False)
