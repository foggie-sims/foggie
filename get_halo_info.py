import yt
from astropy.table import Table
from astropy.io import ascii
from yt.analysis_modules.star_analysis.api import StarFormationRate

from get_halo_center import get_halo_center
import numpy as np

from enzoGalaxyProps import find_rvirial

import seaborn as sns
density_color_map = sns.blend_palette(
    ("black", "#4575b4", "#4daf4a", "#ffe34d", "darkorange"), as_cmap=True)
density_proj_min = 5e-2  # msun / pc^2
density_proj_max = 1e4

yt.add_particle_filter("stars",function=_stars, filtered_type='all',requires=["particle_type"])
yt.add_particle_filter("darkmatter",function=_darkmatter, filtered_type='all',requires=["particle_type"])


def loop_over_halos():
    ### do this way at later times
    snaplist = np.arange(earlysnap,latesnap+1)
    print(snaplist)

    t = Table(
        names=('redshift', 'name', 'xc', 'yc', 'zc',
        'rvir', 'Mvir', 'Mstar', 'Mism', 'SFR'))

    center_guess = first_center
    search_radius = 10. ### COMOVING KPC

    for isnap in snaplist:
        if (isnap > 999): name = 'DD'+str(isnap)
        if (isnap <= 999): name = 'DD0'+str(isnap)
        if (isnap <= 99): name = 'DD00'+str(isnap)
        if (isnap <= 9): name = 'DD000'+str(isnap)
        print(name)
        ds = yt.load(name+'/'+name)
        row = get_halo_info(ds)

def get_halo_info(ds, track):

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
    gal_dm_mass = gal_sphere[('darkmatter', 'particle_mass')].in_units('Msun')
    gal_dm_total_mass = gal_dm_mass.sum().value[()]

    ## define where the central galaxy is
    fgal = 0.07 ## fraction of the virial radius we are saying is the galaxy radius; totes made up
    gal_sphere = ds.sphere(center, fgal*rvir)
    # Mstar is sum of stellar mass; Mism is sum of gas mass; SFR is sum of SFR
    gal_stars_mass = gal_sphere[('stars', 'particle_mass')].in_units('Msun')
    gal_stars_total_mass = gal_stars_mass.sum().value[()]

    gal_ism_mass = gal_sphere['cell_mass'].in_units('Msun')
    gal_ism_total_mass = gal_ism_mass.sum().value[()]

    sfr = StarFormationRate(ds, data_source=gal_sphere)


    row = [zsnap, name,
        center[0], center[1], center[2],
        rvir, Mvir, Mstar, Mism, sfr.Msol_yr])

    return row


if __name__ == "__main__":

    loop_over_halos(first_center, 699, 580, 0.002)
