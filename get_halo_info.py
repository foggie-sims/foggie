import yt
from astropy.table import Table
from astropy.io import ascii
from yt.analysis_modules.star_analysis.api import StarFormationRate

from utils.get_refine_box import get_refine_box
from get_halo_center import get_halo_center
import numpy as np
import glob
import os
from enzoGalaxyProps import find_rvirial

import seaborn as sns
density_color_map = sns.blend_palette(
    ("black", "#4575b4", "#4daf4a", "#ffe34d", "darkorange"), as_cmap=True)
density_proj_min = 5e-2  # msun / pc^2
density_proj_max = 1e4

def _stars(pfilter, data):
    return data[(pfilter.filtered_type, "particle_type")] == 2

#this gets dark matter particles in zoom region only
def _darkmatter(pfilter, data):
    return data[(pfilter.filtered_type, "particle_type")] == 4

yt.add_particle_filter("stars",function=_stars, filtered_type='all',requires=["particle_type"])
yt.add_particle_filter("darkmatter",function=_darkmatter, filtered_type='all',requires=["particle_type"])


def loop_over_halos():
    '''
    This will get the halo_info for ALL of the outputs in the pwd
    '''
    print('assuming trackname is halo_track in the pwd')
    trackname = 'halo_track'
    print("opening track: " + trackname)
    track = Table.read(trackname, format='ascii')
    track.sort('col1')

    t = Table(dtype=('f8','S6', 'f8', 'f8', 'f8',
                    'f8', 'f8', 'f8','f8', 'f8'),
            names=('redshift', 'name', 'xc', 'yc', 'zc',
                    'Rvir', 'Mvir', 'Mstar', 'Mism', 'SFR'))

    outs = glob.glob(os.path.join('.', '?D0???/?D0???'))
    for snap in outs:
        print(snap)
        ds = yt.load(snap)
        ds.add_particle_filter('stars')
        ds.add_particle_filter('darkmatter')
        row = get_halo_info(ds, track)
        t.add_row(row)

    ascii.write(t,'halo_info', format='fixed_width', overwrite=True)

def get_halo_info(ds, track):
    '''
    Given a dataset and the trackfile, finds the halo center, R200, M200, and galaxy masses and SFRs
    '''
    zsnap = ds.get_parameter('CosmologyCurrentRedshift')
    comoving_box_size = ds.get_parameter('CosmologyComovingBoxSize')
    print('Comoving Box Size:', comoving_box_size)
    refine_box, refine_box_center, refine_width = get_refine_box(ds, zsnap, track)

    search_radius = 10.
    this_search_radius = search_radius / (1+ds.get_parameter('CosmologyCurrentRedshift'))
    center, velocity = get_halo_center(ds, refine_box_center, radius=this_search_radius)

    ## halo information
    rvir = find_rvirial(refine_box, ds, center)
    # Mvir is mass within rvir
    vir_sphere = ds.sphere(center,rvir)
    vir_dm_mass = vir_sphere[('darkmatter', 'particle_mass')].in_units('Msun')
    Mvir = vir_dm_mass.sum()

    ## define where the central galaxy is
    fgal = 0.07 ## fraction of the virial radius we are saying is the galaxy radius; totes made up
    gal_sphere = ds.sphere(center, fgal*rvir)
    # Mstar is sum of stellar mass; Mism is sum of gas mass; SFR is sum of SFR
    gal_stars_mass = gal_sphere[('stars', 'particle_mass')].in_units('Msun')
    Mstar = gal_stars_mass.sum()

    gal_ism_mass = gal_sphere['cell_mass'].in_units('Msun')
    Mism = gal_ism_mass.sum()

    sfr = StarFormationRate(ds, data_source=gal_sphere)

    row = [zsnap, 'RD0020',
            center[0], center[1], center[2],
            rvir, Mvir, Mstar, Mism, sfr.Msol_yr[-1]]
    return row


if __name__ == "__main__":

    loop_over_halos()
