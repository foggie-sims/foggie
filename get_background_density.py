

import yt
import numpy as np
import os 
os.sys.path.insert(0, os.environ['FOGGIE_REPO'])
from get_proper_box_size import get_proper_box_size
from yt.units.yt_array import YTQuantity

def get_background_density(ds):

    ad = ds.all_data()
    proper_box_size = YTQuantity(get_proper_box_size(ds)/1000., 'Mpc')
    baryon_mass, particle_mass = ad.quantities.total_quantity(["cell_mass", "particle_mass"])
    BoxTotalMass = (baryon_mass + particle_mass).in_units('Msun')
    BoxDensity = BoxTotalMass / (proper_box_size)**3

    return BoxDensity
