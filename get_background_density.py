

import yt
import numpy as np

from get_proper_box_size import get_proper_box_size

def get_background_density(ds):

    ad = ds.all_data()
    proper_box_size = get_proper_box_size(ds)
    baryon_mass, particle_mass = ad.quantities.total_quantity(["cell_mass", "particle_mass"])
    BoxTotalMass = (baryon_mass + particle_mass).in_units('Msun')
    BoxDensity = BoxTotalMass / (proper_box_size)**3

    return BoxDensity
