
"""
This file contains useful helper functions for FOGGIE
use it as:
import foggie.utils as futils

JT 081318
"""

import yt
import pandas as pd
from consistency import *

CORE_WIDTH = 20.


def ds_to_df(ds, ray_start, ray_end):
    """
    this is a utility function that accepts a yt dataset and the start and end
    points of a ray and returns a pandas dataframe that is useful for shading
    and other analysis.
    """
    current_redshift = ds.get_parameter('CosmologyCurrentRedshift')
    proper_box_size = ds.get_parameter('CosmologyComovingBoxSize') \
        / ds.get_parameter('CosmologyHubbleConstantNow') * 1000.
    print("PROPER BOX SIZE : ", proper_box_size)
    all_data = ds.r[ray_start[0]:ray_end[0],
                    ray_start[1]-0.5*CORE_WIDTH/proper_box_size:ray_start[1]+
                    0.5*CORE_WIDTH/proper_box_size,
                    ray_start[2]-0.5*CORE_WIDTH/proper_box_size:ray_start[2]+
                    0.5*CORE_WIDTH/proper_box_size]

    dens = np.log10(all_data['density'].ndarray_view())
    temp = np.log10(all_data['temperature'].ndarray_view())
    metallicity = all_data['metallicity'].ndarray_view()

    phase_label = new_categorize_by_temp(temp)
    metal_label = categorize_by_metallicity(metallicity)

    df = pd.DataFrame({'x':all_data['x'].ndarray_view() * proper_box_size,
                       'y':all_data['y'].ndarray_view() * proper_box_size,
                       'z':all_data['z'].ndarray_view() * proper_box_size,
                       'vx':all_data["x-velocity"].in_units('km/s'),
                       'vy':all_data["y-velocity"].in_units('km/s'),
                       'vz':all_data["z-velocity"].in_units('km/s'),
                       'cell_mass':all_data['cell_mass'].in_units('Msun'),
                       'temp':temp, 'dens':dens, 'phase_label':phase_label,
                       'metal_label':metal_label})
    df.phase_label = df.phase_label.astype('category')
    df.metal_label = df.metal_label.astype('category')

    return df
