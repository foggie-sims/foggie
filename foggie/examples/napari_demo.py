"""
Filename: napari_demo.py
Author: Cassi
Date created: 1-24-22

This file shows a simple example of how to use the visualization tool Napari with a FOGGIE halo snapshot.
"""

import numpy as np
import yt
import argparse
import scipy.ndimage as ndimage

# These imports are FOGGIE-specific files
from foggie.utils.consistency import *
from foggie.utils.get_refine_box import get_refine_box
from foggie.utils.get_halo_center import get_halo_center
from foggie.utils.get_proper_box_size import get_proper_box_size
from foggie.utils.get_run_loc_etc import get_run_loc_etc
from foggie.utils.yt_fields import *
from foggie.utils.foggie_load import *
from foggie.utils.analysis_utils import *

# Set up command line argument parsing
def parse_args():
    '''Parse command line arguments. Returns args object.'''

    parser = argparse.ArgumentParser(description='Calculates and saves to file a bunch of fluxes.')

    # Optional arguments:
    parser.add_argument('--halo', metavar='halo', type=str, action='store', \
                        help='Which halo? Default is 8508 (Tempest)')
    parser.set_defaults(halo='8508')

    parser.add_argument('--run', metavar='run', type=str, action='store', \
                        help='Which run? Default is nref11c_nref9f')
    parser.set_defaults(run='nref11c_nref9f')

    parser.add_argument('--output', metavar='output', type=str, action='store', \
                        help='Which output? Default is the z=0 snapshot of Tempest, DD2427')
    parser.set_defaults(output='DD2427')

    parser.add_argument('--system', metavar='system', type=str, action='store', \
                        help='Which system are you on? Default is cassiopeia')
    parser.set_defaults(system='cassiopeia')

    parser.add_argument('--pwd', dest='pwd', action='store_true',
                        help='Just use the working directory?, Default is no')
    parser.set_defaults(pwd=False)

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    # Read command line arguments for halo, run, snapshot, and system, then find appropriate
    # file structure for this system
    args = parse_args()
    foggie_dir, output_dir, run_dir, code_path, trackname, haloname, spectra_dir, infofile = get_run_loc_etc(args)
    halo_c_v_name = code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/halo_c_v'

    # Load the snapshot
    snap_name = foggie_dir + run_dir + args.output + '/' + args.output
    ds, refine_box = foggie_load(snap_name, trackname, do_filter_particles=True, halo_c_v_name=halo_c_v_name)

    # Define a 3D FRB to be used with napari. The first few lines here are just to define refine_res to
    # be the same as FOGGIE's level 9 forced refinement. You could just put in whatever number you want
    # for cell sizes for dx instead.
    pix_res = float(np.min(refine_box[('gas','dx')].in_units('kpc')))  # cell size at level 11
    lvl1_res = pix_res*2.**11.              # convert to cell size at level 1
    level = 9                               # specify the level of refinement to match
    dx = lvl1_res/(2.**level)               # convert cell size to specified level (could just put in a number here instead of 3 previous lines)
    box_size = 400                          # define how big you want your box in kpc
    refine_res = int(box_size/dx)           # calculate resolution of FRB based on desired box size and cell size
    # Now define the actual FRB based on box_size and refine_res
    box = ds.covering_grid(level=level, left_edge=ds.halo_center_kpc-ds.arr([box_size/2.,box_size/2.,box_size/2.],'kpc'), dims=[refine_res, refine_res, refine_res])

    # Grab whatever fields we want to visualize
    density = np.log10(box['density'].v)
    temperature = np.log10(box['temperature'].v)
    radial_velocity = box['radial_velocity_corrected'].in_units('km/s').v

    # This next block describes how to grab a disk region and then expand it to grab areas just outside the disk too
    # Define ISM regions
    # disk_mask is a boolean 3D array with 1's for the disk and 0's elsewhere
    disk_mask = (density > np.log10(cgm_density_max*0.1))
    # struct is needed just to identify how the later dilation is done geometrically (ie in 3D)
    struct = ndimage.generate_binary_structure(3,3)
    # disk_mask_expanded is a boolean 3D array with 1's for the disk AND the region surrounding it
    disk_mask_expanded = ndimage.binary_dilation(disk_mask, structure=struct, iterations=3)
    # Dilation expands the cells to include the region surrounding the disk, then closing fills any holes
    disk_mask_expanded = ndimage.binary_closing(disk_mask_expanded, structure=struct, iterations=3)
    # disk_edges is a boolean 3D array with 1's for ONLY pixels surrounding ISM regions -- nothing inside ISM regions
    disk_edges = disk_mask_expanded & ~disk_mask

    # Set the density and temperature of everything other than the disk edges to minimum value for the visualization
    density_edges = np.copy(density)
    density_edges[~disk_edges] = -32
    temperature_edges = np.copy(temperature)
    temperature_edges[~disk_edges] = 3

    # Define the FOGGIE-specific temperature colormap (napari requires you to use vispy to define the color map)
    from vispy.color import Colormap
    temp_cmap = Colormap(['salmon', "#984ea3", "#4daf4a", "#ffe34d", 'darkorange'])

    # Set up the viewer with the first field
    import napari
    viewer = napari.view_image(density, name='density', colormap='viridis', contrast_limits=[-30,-20])
    # Add additional fields with napari.add_image
    temperature_layer = viewer.add_image(temperature, name='temperature', colormap=temp_cmap, contrast_limits=[4,7])
    density_edge_layer = viewer.add_image(density_edges, name='density_disk_edges', colormap='viridis', contrast_limits=[-30,-20])
    temperature_edge_layer = viewer.add_image(temperature_edges, name='temperature_disk_edges', colormap=temp_cmap, contrast_limits=[4,7])

    # In the above, 'name' is the name that will show up on the layer in the viewer,
    # 'colormap' is either a string name of a standard colormap or a list of colors if you want to define your own,
    # and 'contrast_limits' give the bounds of the color map
    # Note that if you want a log scale, plot the log of the field -- there doesn't appear to be a
    # way to tell napari to do log scaling on the color

    # You can add a points field for particles like stars, but the positions must be passed
    # in terms of the image coordinates
    # Load young star positions and shift to halo-centric coordinates
    '''ys_pos = refine_box[('young_stars','particle_position')].in_units('kpc') - ds.halo_center_kpc
    # Convert halo-centric coordinates to image resolution coordinates
    ys_pos = ys_pos.v/dx
    ys_pos += refine_res/2.
    # Now add the points in a new layer
    viewer.add_points(ys_pos, size=1, face_color='white', edge_color='white', name='young stars')'''

    # Finally, open the viewer with the above-defined fields!
    napari.run()

    # If running in ipython or a Jupyter notebook, you can continue adding fields as new layers
    # after opening the viewer.
