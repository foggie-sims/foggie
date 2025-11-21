"""
Callbacks for Calculating Halo Quantities in FOGGIE

This module provides callback functions for computing various physical 
quantities associated with halos in FOGGIE simulations. Each function takes a 
halo object (with an attached yt data object) and returns a derived quantity, 
such as mass, metallicity, temperature, or star formation rate. These callbacks 
are designed to be used with yt's halo catalog analysis framework.

Functions include:
    - halo_total_mass: Total mass (gas + stars + dark matter).
    - halo_average_temperature: Mass-weighted average gas temperature.
    - halo_average_metallicity: Mass-weighted average gas metallicity.
    - halo_max_metallicity: Maximum gas metallicity (normalized to solar).
    - halo_max_gas_density: Maximum gas density.
    - halo_max_dm_density: Maximum dark matter density.
    - halo_total_gas_mass: Total gas mass.
    - halo_ism_gas_mass: Total ISM gas mass (using ISM filter).
    - halo_cgm_gas_mass: Total CGM gas mass (using CGM filter).
    - halo_cool_cgm_gas_mass: Total cool CGM gas mass (T=1.5e4–1e5 K).
    - halo_warm_cgm_gas_mass: Total warm CGM gas mass (T=1e5–1e6 K).
    - halo_hot_cgm_gas_mass: Total hot CGM gas mass (T=1e6–1e8 K).
    - halo_total_star_mass: Total stellar mass.
    - halo_total_metal_mass: Total metal mass in gas.
    - halo_total_young_stars7_mass: Total mass in young stars (<10^7 yr).
    - halo_sfr7: Star formation rate from young stars (<10^7 yr).
    - halo_total_young_stars8_mass: Total mass in young stars (<10^8 yr).
    - halo_sfr8: Star formation rate from young stars (<10^8 yr).
    - halo_average_fH2: Mass-weighted average molecular hydrogen fraction.

If the halo's data object is None (e.g., for very small halos), each function 
returns a zero-valued yt quantity with appropriate units.

Dependencies:
    - yt
    - foggie.utils.consistency (for ISM/CGM filters)
"""

import yt 
from foggie.utils.consistency import * 
from foggie.utils import foggie_utils as futils
import astropy.units as u


def halo_total_mass(halo, correct=True): 
    if (correct): 
        sphere = halo.halo_catalog.data_ds.sphere(halo.data_object.center, radius=halo.quantities["corrected_rvir"])
    else:
        sphere = halo.data_object  # this sphere will have been made for us by the "sphere" callback

    halo.halo_catalog.data_ds.current_redshift

    total_gas_mass = sphere.quantities.total_quantity(("gas", "cell_mass"))
    total_particle_mass = sphere.quantities.total_quantity(("nbody", "particle_mass"))

    # NOTE: in Enzo, the dark matter particles are in the "nbody" family, not "enzo"
    # and, star particles are already included in "nbody" fields so we don't double count them
    return total_gas_mass + total_particle_mass

def halo_overdensity(halo, correct=True):
    if (correct): 
        sphere = halo.halo_catalog.data_ds.sphere(halo.data_object.center, radius=halo.quantities["corrected_rvir"])
    else:
        sphere = halo.data_object  # this sphere will have been made for us by the "sphere" callback

    cosmic_matter_density = futils.cosmic_matter_density(halo.halo_catalog.data_ds.current_redshift,
                                                  h=halo.halo_catalog.data_ds.hubble_constant,
                                                  omega_m=halo.halo_catalog.data_ds.omega_matter)
    
    total_halo_mass = halo_total_mass(halo, correct=correct).to('Msun')    
    halo_mean_density = (total_halo_mass / (4. / 3. * 3.141592653589793 * sphere.radius**3)).to('Msun/Mpc**3')

    return (halo_mean_density.value / cosmic_matter_density)

def halo_average_temperature(halo, correct=True):
    if (correct): 
        sphere = halo.halo_catalog.data_ds.sphere(halo.data_object.center, radius=halo.quantities["corrected_rvir"])
    else:
        sphere = halo.data_object  # this sphere will have been made for us by the "sphere" callback

    # if the halo was too small, the sphere callback will return a None, then just return 0 (in correct units)
    if sphere is None:
        return halo.halo_catalog.data_ds.quan(0, "K")

    return sphere.quantities.weighted_average_quantity(
        ("gas", "temperature"), ("gas", "cell_mass"))

def halo_average_metallicity(halo, correct=True):
    if (correct): 
        sphere = halo.halo_catalog.data_ds.sphere(halo.data_object.center, radius=halo.quantities["corrected_rvir"])
    else:
        sphere = halo.data_object    # this sphere will have been made for us by the "sphere" callback

    if sphere is None:
        return halo.halo_catalog.data_ds.quan(0, "dimensionless")

    return sphere.quantities.weighted_average_quantity(
        ("gas", "metallicity"), ("gas", "cell_mass"))

def halo_max_metallicity(halo, correct=True):
    if (correct): 
        sphere = halo.halo_catalog.data_ds.sphere(halo.data_object.center, radius=halo.quantities["corrected_rvir"])
    else:
        sphere = halo.data_object    # this sphere will have been made for us by the "sphere" callback

    if sphere is None:
        return halo.halo_catalog.data_ds.quan(0, "dimensionless")

    zmax = sphere.quantities.extrema(("gas", "metallicity")) / 0.02 
    return zmax[1]

def halo_max_gas_density(halo, correct=True): 
    if (correct): 
        sphere = halo.halo_catalog.data_ds.sphere(halo.data_object.center, radius=halo.quantities["corrected_rvir"])
    else:
        sphere = halo.data_object    # this sphere will have been made for us by the "sphere" callback

    if sphere is None:
        return halo.halo_catalog.data_ds.quan(0, "dimensionless")

    dmax = sphere.quantities.extrema(("gas", "density")) 
    return dmax[1]
    
def halo_max_dm_density(halo, correct=True): 
    if (correct): 
        sphere = halo.halo_catalog.data_ds.sphere(halo.data_object.center, radius=halo.quantities["corrected_rvir"])
    else:
        sphere = halo.data_object    # this sphere will have been made for us by the "sphere" callback

    if sphere is None:
        return halo.halo_catalog.data_ds.quan(0, "dimensionless")

    dm_max = sphere.quantities.extrema(("enzo", "Dark_Matter_Density")) 
    return dm_max[1]

def halo_total_gas_mass(halo, correct=True): 
    if (correct): 
        sphere = halo.halo_catalog.data_ds.sphere(halo.data_object.center, radius=halo.quantities["corrected_rvir"])
    else:
        sphere = halo.data_object  # this sphere will have been made for us by the "sphere" callback

    if sphere is None:
        return halo.halo_catalog.data_ds.quan(0, "Msun")

    return sphere.quantities.total_quantity(("gas", "cell_mass"))

def halo_ism_gas_mass(halo, correct=True): 
    if (correct): 
        sphere = halo.halo_catalog.data_ds.sphere(halo.data_object.center, radius=halo.quantities["corrected_rvir"])
    else:
        sphere = halo.data_object  # this sphere will have been made for us by the "sphere" callback

    if sphere is None:
        return halo.halo_catalog.data_ds.quan(0, "Msun")

    sphere = sphere.cut_region([ism_field_filter_z(halo.halo_catalog.data_ds.current_redshift)]) #ism field filter is defined in consistency.py

    return sphere.quantities.total_quantity(("gas", "cell_mass"))

def halo_cgm_gas_mass(halo, correct=True): 
    if (correct): 
        sphere = halo.halo_catalog.data_ds.sphere(halo.data_object.center, radius=halo.quantities["corrected_rvir"])
    else:
        sphere = halo.data_object  # this sphere will have been made for us by the "sphere" callback

    if sphere is None:
        return halo.halo_catalog.data_ds.quan(0, "Msun")

    cgm_cut_region = cgm_field_filter_z(halo.halo_catalog.data_ds.current_redshift, tmin=200., tmax=1e8)

    sphere = sphere.cut_region([cgm_cut_region]) #cgm field filter is defined in consistency.py

    return sphere.quantities.total_quantity(("gas", "cell_mass"))

def halo_cold_cgm_gas_mass(halo, correct=True): 
    if (correct): 
        sphere = halo.halo_catalog.data_ds.sphere(halo.data_object.center, radius=halo.quantities["corrected_rvir"])
    else:
        sphere = halo.data_object  # this sphere will have been made for us by the "sphere" callback

    if sphere is None:
        return halo.halo_catalog.data_ds.quan(0, "Msun")

    sphere = sphere.cut_region([cgm_field_filter_z(halo.halo_catalog.data_ds.current_redshift, tmin=100, tmax=1.5e4)]) #cgm field filter is defined in consistency.py
    return sphere.quantities.total_quantity(("gas", "cell_mass"))

def halo_cool_cgm_gas_mass(halo, correct=True): 
    if (correct): 
        sphere = halo.halo_catalog.data_ds.sphere(halo.data_object.center, radius=halo.quantities["corrected_rvir"])
    else:
        sphere = halo.data_object  # this sphere will have been made for us by the "sphere" callback

    if sphere is None:
        return halo.halo_catalog.data_ds.quan(0, "Msun")
	
    sphere = sphere.cut_region([cgm_field_filter_z(halo.halo_catalog.data_ds.current_redshift, tmin=1.5e4, tmax=1e5)]) #cgm field filter is defined in consistency.py
    return sphere.quantities.total_quantity(("gas", "cell_mass"))

def halo_warm_cgm_gas_mass(halo, correct=True): 
    if (correct): 
        sphere = halo.halo_catalog.data_ds.sphere(halo.data_object.center, radius=halo.quantities["corrected_rvir"])
    else:
        sphere = halo.data_object  # this sphere will have been made for us by the "sphere" callback

    if sphere is None:
        return halo.halo_catalog.data_ds.quan(0, "Msun")
    
    sphere = sphere.cut_region([cgm_field_filter_z(halo.halo_catalog.data_ds.current_redshift, tmin=1e5, tmax=1e6)]) #cgm field filter is defined in consistency.py
    return sphere.quantities.total_quantity(("gas", "cell_mass"))

def halo_hot_cgm_gas_mass(halo, correct=True): 
    if (correct): 
        sphere = halo.halo_catalog.data_ds.sphere(halo.data_object.center, radius=halo.quantities["corrected_rvir"])
    else:
        sphere = halo.data_object  # this sphere will have been made for us by the "sphere" callback

    if sphere is None:
        return halo.halo_catalog.data_ds.quan(0, "Msun")
	
    sphere = sphere.cut_region([cgm_field_filter_z(halo.halo_catalog.data_ds.current_redshift, tmin=1e6, tmax=1e8)]) #cgm field filter is defined in consistency.py

    return sphere.quantities.total_quantity(("gas", "cell_mass"))

def halo_total_star_mass(halo, correct=True): 
    if (correct): 
        sphere = halo.halo_catalog.data_ds.sphere(halo.data_object.center, radius=halo.quantities["corrected_rvir"])
    else:
        sphere = halo.data_object  # this sphere will have been made for us by the "sphere" callback

    if sphere is None:
        return halo.halo_catalog.data_ds.quan(0, "Msun")
    return sphere.quantities.total_quantity(('stars', 'particle_mass'))

def halo_total_metal_mass(halo, correct=True): 
    if (correct): 
        sphere = halo.halo_catalog.data_ds.sphere(halo.data_object.center, radius=halo.quantities["corrected_rvir"])
    else:
        sphere = halo.data_object  # this sphere will have been made for us by the "sphere" callback

    if sphere is None:
        return halo.halo_catalog.data_ds.quan(0, "Msun")
    
    return sphere.quantities.total_quantity(('gas', 'metal_mass'))

def halo_total_young_stars7_mass(halo, correct=True): 
    if (correct): 
        sphere = halo.halo_catalog.data_ds.sphere(halo.data_object.center, radius=halo.quantities["corrected_rvir"])
    else:
        sphere = halo.data_object  # this sphere will have been made for us by the "sphere" callback

    if sphere is None:
        return halo.halo_catalog.data_ds.quan(0, "Msun")

    return sphere.quantities.total_quantity(('young_stars', 'particle_mass'))

def halo_sfr7(halo, correct=True): 
    if (correct): 
        sphere = halo.halo_catalog.data_ds.sphere(halo.data_object.center, radius=halo.quantities["corrected_rvir"])
    else:
        sphere = halo.data_object  # this sphere will have been made for us by the "sphere" callback

    if sphere is None:
        return halo.halo_catalog.data_ds.quan(0, "Msun")

    return sphere.quantities.total_quantity(('young_stars', 'particle_mass')) / yt.YTArray(1e7, 'yr')  

def halo_total_young_stars8_mass(halo, correct=True): 
    if (correct): 
        sphere = halo.halo_catalog.data_ds.sphere(halo.data_object.center, radius=halo.quantities["corrected_rvir"])
    else:
        sphere = halo.data_object  # this sphere will have been made for us by the "sphere" callback

    if sphere is None:
        return halo.halo_catalog.data_ds.quan(0, "Msun")

    return sphere.quantities.total_quantity(('young_stars8', 'particle_mass'))

def halo_sfr8(halo, correct=True): 
    if (correct): 
        sphere = halo.halo_catalog.data_ds.sphere(halo.data_object.center, radius=halo.quantities["corrected_rvir"])
    else:
        sphere = halo.data_object    # this sphere will have been made for us by the "sphere" callback

    if sphere is None:
        return halo.halo_catalog.data_ds.quan(0, "Msun")

    return sphere.quantities.total_quantity(('young_stars8', 'particle_mass')) / yt.YTArray(1e8, 'yr')  

def halo_average_fH2(halo, correct=True):
    if (correct): 
        sphere = halo.halo_catalog.data_ds.sphere(halo.data_object.center, radius=halo.quantities["corrected_rvir"])
    else:
        sphere = halo.data_object    

    if sphere is None:
        return halo.halo_catalog.data_ds.quan(0, "Msun")

    fH2 = sphere.quantities.weighted_average_quantity(("gas", "H2_fraction"), ("gas", "cell_mass"))

    return fH2

def halo_actual_baryon_fraction(halo, correct=True):
    if (correct): 
        sphere = halo.halo_catalog.data_ds.sphere(halo.data_object.center, radius=halo.quantities["corrected_rvir"])
    else:
        sphere = halo.data_object  # this sphere will have been made for us by the "sphere" callback 

    if sphere is None:
        return halo.halo_catalog.data_ds.quan(0, "Msun")

    baryon_fraction = (halo_total_gas_mass(halo, correct=correct) + halo_total_star_mass(halo, correct=correct)) / halo_total_mass(halo, correct=correct)

    return baryon_fraction

def halo_corrected_rvir(halo):
    sphere = halo.data_object    
    
    cosmic_matter_density = futils.cosmic_matter_density(halo.halo_catalog.data_ds.current_redshift,
                                                  h=halo.halo_catalog.data_ds.hubble_constant,
                                                  omega_m=halo.halo_catalog.data_ds.omega_matter)

    delta_r = 0.05 * sphere.radius.in_units('kpc')
    total_halo_mass = sphere.quantities.total_quantity(("gas", "cell_mass")) + sphere.quantities.total_quantity(("nbody", "particle_mass"))   
    halo_mean_density = (total_halo_mass / (4. / 3. * 3.141592653589793 * sphere.radius**3)).to('Msun/Mpc**3')
    overdensity = (halo_mean_density.value / cosmic_matter_density)
    new_radius = sphere.radius.in_units('kpc')
    
    while (overdensity > 200.):
        new_radius = new_radius + halo.halo_catalog.data_ds.quan(delta_r, "kpc") 
        new_sphere = halo.halo_catalog.data_ds.sphere(sphere.center, radius = new_radius) 
        total_halo_mass = new_sphere.quantities.total_quantity(("gas", "cell_mass")) + new_sphere.quantities.total_quantity(("nbody", "particle_mass"))   
        halo_mean_density = (total_halo_mass / (4. / 3. * 3.141592653589793 * new_radius**3)).to('Msun/Mpc**3')
        overdensity = (halo_mean_density.value / cosmic_matter_density)
         
    return new_radius