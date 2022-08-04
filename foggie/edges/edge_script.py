import numpy as np, pandas as pd, scipy.ndimage as ndimage, yt, cmyt
yt.set_log_level(40)
from foggie.utils.consistency import axes_label_dict, logfields, categorize_by_temp,     categorize_by_metals, categorize_by_fraction
from foggie.utils.yt_fields import *
from foggie.utils.foggie_load import *
from foggie.edges.edge_functions import *

# ### Open dataset and obtain the disk orientation parameters

def edge_script(dataset_name):

    plots_to_make = (True,True,True,True,True)

    trackname = '/nobackup/jtumlins/foggie/foggie/halo_tracks/004123/nref11n_selfshield_15/halo_track_200kpc_nref10'
    snapshot_dir='/nobackup/mpeeples/halo_004123/nref11c_nref9f/'
    print('I will now open: ', snapshot_dir+dataset_name+'/'+dataset_name)

    box_size = 400.
    sampling_level = 9 # specify the level of refinement to match in the covering grid
    number_of_edge_iterations = 3

    ds, refine_box = foggie_load(snapshot_dir+dataset_name+'/'+dataset_name, trackname, \
                        disk_relative=True, gravity=True, \
                        masses_dir='/nobackup/jtumlins/foggie/foggie/halo_infos/004123/nref11c_nref9f/')
    ad = ds.all_data()

    # these are the region_dicts returned by function_for_edges
    disk = function_for_edges(ds, trackname, refine_box)
    disk = apply_region_cuts_to_dict(disk, number_of_edge_iterations = number_of_edge_iterations, region = 'disk')
    disk_grid = convert_to_new_dataset(disk, box_size)

    inflow = function_for_edges(ds, trackname, refine_box)
    inflow = apply_region_cuts_to_dict(inflow, number_of_edge_iterations = number_of_edge_iterations, region = 'inflow')
    inflow_grid = convert_to_new_dataset(inflow ,box_size)

    plot_four(disk_grid, disk, dataset_name)
    plot_five(disk_grid, disk, dataset_name)

    plot_four(inflow_grid, inflow, dataset_name)
    plot_five(inflow_grid, inflow, dataset_name)

    new_dict = merge_two_regions(disk, inflow)

    #convert to new dataset
    overlap_grid = convert_to_new_dataset(new_dict, box_size)
    ad_grid = overlap_grid.all_data()

    plot_four(overlap_grid, new_dict, dataset_name)
    plot_five(overlap_grid, new_dict, dataset_name)


