
"""
This file contains useful helper functions for FOGGIE
use it as:
import foggie.utils as futils

JT 081318
"""

import yt
import pandas as pd
import argparse
from consistency import *

CORE_WIDTH = 20.


def get_ray_axis(ray_start, ray_end):
    """ takes in ray and returns an integer, 0, 1, 2 for x, y, z, orients"""

    axes_labels = ['x','y','z']
    second_axes = {'x':'y', 'y':'z', 'z':'x'}

    ray_length = ray_end-ray_start
    if (ray_length[0] > 0.):
        ray_index = 0
        first_axis = axes_labels[ray_index]
        second_axis = second_axes[first_axis]
        return ray_index, first_axis, second_axis
    elif (ray_length[1] > 0.):
        ray_index = 1
        first_axis = axes_labels[ray_index]
        second_axis = second_axes[first_axis]
        return ray_index, first_axis, second_axis
    elif (ray_length[2] > 0.):
        ray_index = 2
        first_axis = axes_labels[ray_index]
        second_axis = second_axes[first_axis]
        return ray_index, first_axis, second_axis
    else:
        print('Your ray is bogus, try again!')
        return False


def parse_args():
    '''
    Parse command line arguments.  Returns args object.
    '''
    parser = argparse.ArgumentParser(description="extracts spectra from refined region")

    parser.add_argument('--run', metavar='run', type=str, action='store',
                        help='which run? default is nref9f')
    parser.set_defaults(run="nref9f")

    parser.add_argument('--output', metavar='output', type=str, action='store',
                        help='which output? default is RD0020')
    parser.set_defaults(output="RD0020")

    parser.add_argument('--system', metavar='system', type=str, action='store',
                        help='which system are you on? default is oak')
    parser.set_defaults(system="oak")

    parser.add_argument('--fitsfile', metavar='fitsfile', type=str, action='store',
                        help='what fitsfile would you like to read in? this does not work yet')

    args = parser.parse_args()
    return args

def get_path_info(args):

    args = parse_args()
    if args.system == "oak":
        ds_base = "/astro/simulations/FOGGIE/"
        output_path = "/Users/molly/Dropbox/foggie-collab/"
    elif args.system == "dhumuha" or args.system == "palmetto":
        ds_base = "/Users/molly/foggie/"
        output_path = "/Users/molly/Dropbox/foggie-collab/"
    elif args.system == "harddrive":
        ds_base = "/Volumes/foggie/"
        output_path = "/Users/molly/Dropbox/foggie-collab/"
    elif args.system == "townes":
        print("SYSTEM = ", args.system)
        ds_base = "/Users/tumlinson/Dropbox/FOGGIE/outputs/"
        output_path = "/Users/tumlinson/Dropbox/foggie/collab/"
        print(ds_base, output_path)
    elif args.system == "lefty":
        print("SYSTEM = ", args.system)
        ds_base = "/Users/tumlinson/Dropbox/FOGGIE/outputs/"
        output_path = "/Users/tumlinson/Dropbox/FOGGIE/collab/"

    if args.run == "natural":
        ds_loc = ds_base + "halo_008508/nref11n/natural/" + args.output + "/" + args.output
        output_dir = output_path + "plots_halo_008508/nref11n/natural/spectra/"
        haloname = "halo008508_nref11n"
    elif args.run == "nref10f":
        ds_loc =  ds_base + "halo_008508/nref11n/nref11n_nref10f_refine200kpc/" + args.output + "/" + args.output
        output_dir = output_path + "plots_halo_008508/nref11n/nref11n_nref10f_refine200kpc/spectra/"
        haloname = "halo008508_nref11n_nref10f"
        trackfile = ds_base + "halo_008508/nref11n/nref11n_nref10f_refine200kpc/halo_track"
    elif args.run == "nref9f":
        path_part = "halo_008508/nref11n/nref11n_"+args.run+"_refine200kpc/"
        ds_loc =  ds_base + path_part + args.output + "/" + args.output
        output_dir = output_path + "plots_halo_008508/nref11n/nref11n_nref9f_refine200kpc/spectra/"
        haloname = "halo008508_nref11n_nref9f"
    elif args.run == "nref11f":
        ds_loc =  ds_base + "halo_008508/nref11n/nref11f_refine200kpc/" + args.output + "/" + args.output
        output_dir = output_path + "plots_halo_008508/nref11n/nref11f_refine200kpc/spectra/"
        haloname = "halo008508_nref11f"

    return ds_loc, output_path, output_dir, haloname



def ds_to_df(ds, ray_start, ray_end):
    """
    this is a utility function that accepts a yt dataset and the start and end
    points of a ray and returns a pandas dataframe that is useful for shading
    and other analysis.
    """
    current_redshift = ds.get_parameter('CosmologyCurrentRedshift')
    proper_box_size = ds.get_parameter('CosmologyComovingBoxSize') \
        / ds.get_parameter('CosmologyHubbleConstantNow') * 1000.

    ray_index, first_axis, second_axis = get_ray_axis(ray_start, ray_end)
    if (ray_index == 0):
        all_data = ds.r[ray_start[0]:ray_end[0],
                        ray_start[1]-0.5*CORE_WIDTH/proper_box_size:ray_start[1]+
                        0.5*CORE_WIDTH/proper_box_size,
                        ray_start[2]-0.5*CORE_WIDTH/proper_box_size:ray_start[2]+
                        0.5*CORE_WIDTH/proper_box_size]
    elif (ray_index == 1):
        all_data = ds.r[ray_start[0]-0.5*CORE_WIDTH/proper_box_size:ray_start[0]+
                        0.5*CORE_WIDTH/proper_box_size,
                        ray_start[1]:ray_end[1],
                        ray_start[2]-0.5*CORE_WIDTH/proper_box_size:ray_start[2]+
                        0.5*CORE_WIDTH/proper_box_size]
    elif (ray_index == 2):
        all_data = ds.r[ray_start[0]-0.5*CORE_WIDTH/proper_box_size:ray_start[0]+
                        0.5*CORE_WIDTH/proper_box_size,
                        ray_start[1]-0.5*CORE_WIDTH/proper_box_size:ray_start[1]+
                        0.5*CORE_WIDTH/proper_box_size,
                        ray_start[2]:ray_end[2]]
    else:
        print('Your ray is bogus, try again!')

    dens = np.log10(all_data['density'].ndarray_view())
    temp = np.log10(all_data['temperature'].ndarray_view())
    metallicity = all_data['metallicity'].ndarray_view()

    # creates the phase_label as a set of nonsense strings.
    phase_label = new_categorize_by_temp(temp)
    metal_label = new_categorize_by_metals(metallicity)

    df = pd.DataFrame({'x':all_data['x'].ndarray_view() * proper_box_size,
                       'y':all_data['y'].ndarray_view() * proper_box_size,
                       'z':all_data['z'].ndarray_view() * proper_box_size,
                       'metallicity':metallicity,
                       'vx':all_data["x-velocity"].in_units('km/s'),
                       'vy':all_data["y-velocity"].in_units('km/s'),
                       'vz':all_data["z-velocity"].in_units('km/s'),
                       'cell_mass':all_data['cell_mass'].in_units('Msun'),
                       'temp':temp, 'dens':dens, 'phase_label':phase_label,
                       'metal_label':metal_label})

    df.phase_label = df.phase_label.astype('category')
    df.metal_label = df.metal_label.astype('category')

    # this is awful, but we have to add categories that don't exist to use them later.
    existing_categories = df.phase_label.unique()
    for label in phase_labels:
        if (not (label in existing_categories)):
            df.phase_label = df.phase_label.cat.add_categories([label])

    existing_categories = df.metal_label.unique()
    for label in metal_labels:
        if (not (label in existing_categories)):
            df.metal_label = df.metal_label.cat.add_categories([label])

    return df
