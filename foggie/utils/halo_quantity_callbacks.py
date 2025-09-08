
import yt 
from foggie.utils.consistency import * 

def halo_average_temperature(halo):
    sphere = halo.data_object    # this sphere will have been made for us by the "sphere" callback

    # if the halo was too small, the sphere callback will return a None, then just return 0 (in correct units)
    if sphere is None:
        return halo.halo_catalog.data_ds.quan(0, "K")

    return sphere.quantities.weighted_average_quantity(
        ("gas", "temperature"), ("gas", "cell_mass"))

def halo_average_metallicity(halo):
    sphere = halo.data_object    # this sphere will have been made for us by the "sphere" callback

    if sphere is None:
        return halo.halo_catalog.data_ds.quan(0, "dimensionless")

    return sphere.quantities.weighted_average_quantity(
        ("gas", "metallicity"), ("gas", "cell_mass"))

def halo_max_metallicity(halo):
    sphere = halo.data_object    # this sphere will have been made for us by the "sphere" callback

    if sphere is None:
        return halo.halo_catalog.data_ds.quan(0, "dimensionless")

    zmax = sphere.quantities.extrema(("gas", "metallicity")) / 0.02 
    return zmax[1]

def halo_max_gas_density(halo): 
    sphere = halo.data_object    # this sphere will have been made for us by the "sphere" callback

    if sphere is None:
        return halo.halo_catalog.data_ds.quan(0, "dimensionless")

    dmax = sphere.quantities.extrema(("gas", "density")) 
    return dmax[1]

def halo_max_dm_density(halo): 
    sphere = halo.data_object    # this sphere will have been made for us by the "sphere" callback

    if sphere is None:
        return halo.halo_catalog.data_ds.quan(0, "dimensionless")

    dm_max = sphere.quantities.extrema(("enzo", "Dark_Matter_Density")) 
    return dm_max[1]

def halo_total_gas_mass(halo): 
    sphere = halo.data_object    # this sphere will have been made for us by the "sphere" callback

    if sphere is None:
        return halo.halo_catalog.data_ds.quan(0, "Msun")

    return sphere.quantities.total_quantity(("gas", "cell_mass"))

def halo_ism_gas_mass(halo): 
    sphere = halo.data_object    # this sphere will have been made for us by the "sphere" callback

    if sphere is None:
        return halo.halo_catalog.data_ds.quan(0, "Msun")

    sphere = sphere.cut_region([ism_field_filter]) #ism field filter is defined in consistency.py

    return sphere.quantities.total_quantity(("gas", "cell_mass"))


def halo_cgm_gas_mass(halo): 
    sphere = halo.data_object    # this sphere will have been made for us by the "sphere" callback

    if sphere is None:
        return halo.halo_catalog.data_ds.quan(0, "Msun")

    sphere = sphere.cut_region([cgm_field_filter]) #cgm field filter is defined in consistency.py

    return sphere.quantities.total_quantity(("gas", "cell_mass"))


def halo_total_star_mass(halo): 
    sphere = halo.data_object    # this sphere will have been made for us by the "sphere" callback

    if sphere is None:
        return halo.halo_catalog.data_ds.quan(0, "Msun")

    return sphere.quantities.total_quantity(('stars', 'particle_mass'))

def halo_total_metal_mass(halo): 
    sphere = halo.data_object    # this sphere will have been made for us by the "sphere" callback

    if sphere is None:
        return halo.halo_catalog.data_ds.quan(0, "Msun")
    
    return sphere.quantities.total_quantity(('gas', 'metal_mass'))

def halo_total_young_stars7_mass(halo): 
    sphere = halo.data_object    # this sphere will have been made for us by the "sphere" callback

    if sphere is None:
        return halo.halo_catalog.data_ds.quan(0, "Msun")

    return sphere.quantities.total_quantity(('young_stars', 'particle_mass'))

def halo_sfr7(halo): 
    sphere = halo.data_object    # this sphere will have been made for us by the "sphere" callback

    if sphere is None:
        return halo.halo_catalog.data_ds.quan(0, "Msun")

    return sphere.quantities.total_quantity(('young_stars', 'particle_mass')) / yt.YTArray(1e7, 'yr')  

def halo_total_young_stars8_mass(halo): 
    sphere = halo.data_object    # this sphere will have been made for us by the "sphere" callback

    if sphere is None:
        return halo.halo_catalog.data_ds.quan(0, "Msun")

    return sphere.quantities.total_quantity(('young_stars8', 'particle_mass'))

def halo_sfr8(halo): 
    sphere = halo.data_object    # this sphere will have been made for us by the "sphere" callback
    
    if sphere is None:
        return halo.halo_catalog.data_ds.quan(0, "Msun")

    return sphere.quantities.total_quantity(('young_stars8', 'particle_mass')) / yt.YTArray(1e8, 'yr')  

def halo_average_fH2(halo):
    sphere = halo.data_object    

    if sphere is None:
        return halo.halo_catalog.data_ds.quan(0, "Msun")

    fH2 = sphere.quantities.weighted_average_quantity(("gas", "H2_fraction"), ("gas", "cell_mass"))

    return fH2
