import functools
import numpy as np
import optparse
from yt.units.unit_registry import UnitRegistry
from yt.units.yt_array import YTQuantity
from yt.utilities.physical_ratios import \
    rho_crit_g_cm3_h2

refine_by = 2
unit_registry = UnitRegistry()

def quan(v, u):
    return YTQuantity(v, u, registry=unit_registry)

def get_smallest_appropriate_unit(v, quantity='distance',
                                  return_quantity=False):
    """
    Returns the largest whole unit smaller than the YTQuantity passed to
    it as a string.

    The quantity keyword can be equal to `distance` or `time`.  In the
    case of distance, the units are: 'Mpc', 'kpc', 'pc', 'au', 'rsun',
    'km', etc.  For time, the units are: 'Myr', 'kyr', 'yr', 'day', 'hr',
    's', 'ms', etc.

    If return_quantity is set to True, it finds the largest YTQuantity
    object with a whole unit and a power of ten as the coefficient, and it
    returns this YTQuantity.
    """
    good_u = None
    if quantity == 'distance':
        unit_list =['Ppc', 'Tpc', 'Gpc', 'Mpc', 'kpc', 'pc', 'au', 'rsun',
                    'km', 'cm', 'um', 'nm', 'pm']
    elif quantity == 'time':
        unit_list =['Yyr', 'Zyr', 'Eyr', 'Pyr', 'Tyr', 'Gyr', 'Myr', 'kyr',
                    'yr', 'day', 'hr', 's', 'ms', 'us', 'ns', 'ps', 'fs']
    else:
        raise SyntaxError("Specified quantity must be equal to 'distance'"\
                          "or 'time'.")
    for unit in unit_list:
        uq = quan(1.0, unit)
        if uq <= v:
            good_u = unit
            break
    if good_u is None and quantity == 'distance': good_u = 'cm'
    if good_u is None and quantity == 'time': good_u = 's'
    if return_quantity:
        unit_index = unit_list.index(good_u)
        # This avoids indexing errors
        if unit_index == 0: return quan(1, unit_list[0])
        # Number of orders of magnitude between unit and next one up
        OOMs = np.ceil(np.log10(quan(1, unit_list[unit_index-1]) /
                                quan(1, unit_list[unit_index])))
        # Backwards order of coefficients (e.g. [100, 10, 1])
        coeffs = 10**np.arange(OOMs)[::-1]
        for j in coeffs:
            uq = quan(j, good_u)
            if uq <= v:
                return uq
    else:
        return good_u

def print_info(options):

    print ("Cosmology parameters:")
    print ("Omega_cdm: %.4f" % options.omega_cdm)
    print ("  Omega_b: %.4f" % options.omega_b)
    print ("        h: %.4f" % options.hubble_constant)
    print ("")
    print ("Simulation parameters:")
    print ("                    Box size: %.3e Mpc (%.3e Mpc/h)." % \
      (options.box_size.to("Mpc"), options.box_size.to("Mpc/h")))
    print ("   Number of particles/cells: %.2f^3." % options.root_grid_size)
    print ("               Particle mass: %.3e Msun (%.3e Msun/h)." % \
      (options.root_particle_mass.to("Msun"),
       options.root_particle_mass.to("Msun/h")))
    if options.nested_levels > 0:
        print ("               Nested levels: %d." % options.nested_levels)
        print ("        Nested particle mass: %.3e Msun (%.3e Msun/h)." % \
          (options.nested_particle_mass.to("Msun"),
           options.nested_particle_mass.to("Msun/h")))
    print ("                  AMR levels: %d." % options.amr_levels)
    print ("                Total levels: %d." % (options.nested_levels + options.amr_levels))
    dx = calculate_spatial_resolution(options)
    dx.convert_to_units(get_smallest_appropriate_unit(dx, quantity='distance'))
    print ("     Max. spatial resolution: %.3e %s (%.3e %s/h)." % \
      (dx, dx.units, dx.to("%s/h" % dx.units), dx.units))
    if options.amr_cell_density is not None:
        print ("Cell mass at %.3e g/cm^3: %.3e Msun." % (options.amr_cell_density,
                                                         calculate_amr_cell_mass(options)))
    
def rho_crit_cosmo(options):
    return quan(rho_crit_g_cm3_h2 *
                      options.hubble_constant**2, "g/cm**3")

def calculate_spatial_resolution(options):
    "Calculate the maximum spatial resolution."

    return options.box_size / options.root_grid_size / \
      refine_by**(options.nested_levels + options.amr_levels)
      
def calculate_box_size(options):
    "Calculate root grid size from box size and number of particles."

    omega = options.omega_cdm
    if options.dm_only:
        omega += options.omega_b

    options.box_size = options.root_grid_size * \
      (options.root_particle_mass /
       (omega * rho_crit_cosmo(options)))**(1./3.)
    return options.box_size

def calculate_root_grid_size(options):
    "Calculate root grid size from box size and number of particles."

    omega = options.omega_cdm
    if options.dm_only:
        omega += options.omega_b

    return options.box_size * \
      (options.root_particle_mass / (omega * rho_crit_cosmo(options)))**(-1./3.)

def calculate_particle_mass(options, refined=False):
    "Calculate particle mass from box size and number of particles."

    omega = options.omega_cdm
    if options.dm_only:
        omega += options.omega_b

    particle_mass = rho_crit_cosmo(options) * omega * \
      (options.box_size / options.root_grid_size)**3

    if refined:
        particle_mass /= refine_by**(3 * options.nested_levels)

    return particle_mass

def calculate_nested_levels(options):
    """
    Calculate number of nested levels from root grid size, number of particles, 
    and particle mass.
    """

    my_levels = np.ceil(np.log2(calculate_root_grid_size(options) / options.root_grid_size))
    if my_levels <= 0:
        raise RuntimeError("No nested levels are required to reach this particle mass.")
    return my_levels

def calculate_amr_cell_mass(options):
    """
    Calculate the mass contained within the highest resolution cell at a given density.
    """

    cell_size = calculate_spatial_resolution(options)
    return options.amr_cell_density * cell_size**3

if __name__ == "__main__":
    usage = "usage: %prog [options]"
    usage += "\n\nSupply options to get the following:"
    usage += "\n1. particle mass: box size, number of particles, [number of nested levels]."
    usage += "\n2. box size: number of particles, particle mass."
    usage += "\n3. number of particles: box size, particle mass."
    usage += "\n3. number of nested levels: box size, number of particles, particle mass."
    parser = optparse.OptionParser(usage=usage)
    parser.add_option("-L", "--box-size", dest="box_size",
                      metavar="float", type="str",
                      help="Box size in comoving Mpc.")
    parser.add_option("-N", "--root-grid", dest="root_grid_size",
                      metavar="float", type="float",
                      help="Number of particles/cells on root grid^(1/3).")
    parser.add_option("-m", "--particle-mass", dest="root_particle_mass",
                      metavar="float", type="str",
                      help="Dark matter particle mass in Msun.")
    parser.add_option("", "--nested", dest="nested_levels",
                      metavar="int", type="int", default=0,
                      help="Number of static, nested refinement levels.")
    parser.add_option("", "--amr", dest="amr_levels",
                      metavar="int", type="int", default=0,
                      help="Number of AMR levels.")
    parser.add_option("", "--dm-only", dest="dm_only",
                      action="store_true", default=False,
                      help="Set for dark matter only simulation.  In this case Omega_cdm is set to Omega_m.")
    parser.add_option("", "--amr-mass-at-density", dest="amr_cell_density",
                      metavar="float", type="float",
                      help="Calculate the mass within the highest resolution cell at the given density in g/cm^3.")
    
    cosmology_parameters = optparse.OptionGroup(parser, "Cosmology Parameters")
    cosmology_parameters.add_option("", "--omega-cdm", dest="omega_cdm",
                                    metavar="float", type="float",
                                    default=0.236,
                                    help="Cold dark matter contribution.")
    cosmology_parameters.add_option("", "--omega-b", dest="omega_b",
                                    metavar="float", type="float",
                                    default=0.046,
                                    help="Baryon contribution.")
    cosmology_parameters.add_option("", "--hubble", dest="hubble_constant",
                                    metavar="float", type="float",
                                    default=0.697,
                                    help="Hubble constant (h) in 100 km/s/Mpc.")
    parser.add_option_group(cosmology_parameters)

    options, args = parser.parse_args()

    required = (options.box_size is not None) + \
      (options.root_grid_size is not None) + \
      (options.root_particle_mass is not None)

    unit_registry.modify("h", options.hubble_constant)

    if options.box_size is not None:
        if "/h" in options.box_size:
            options.box_size = float(options.box_size[:-2]) / \
              options.hubble_constant
        else:
            options.box_size = float(options.box_size)
        options.box_size = quan(options.box_size, "Mpc")

    if options.root_particle_mass is not None:
        if "/h" in options.root_particle_mass:
            options.root_particle_mass = \
              float(options.root_particle_mass[:-2]) / \
              options.hubble_constant
        else:
            options.root_particle_mass = \
              float(options.root_particle_mass)
        options.root_particle_mass = \
          quan(options.root_particle_mass, "Msun")

    if required < 2:
        raise RuntimeError("Invalid options given, run with -h for more information.")

    if options.box_size is None:
        options.box_size = calculate_box_size(options)
    elif options.root_grid_size is None:
        options.root_grid_size = calculate_root_grid_size(options)
    elif options.root_particle_mass is None:
        options.root_particle_mass = calculate_particle_mass(options)
        if options.nested_levels > 0:
            options.nested_particle_mass = calculate_particle_mass(options, refined=True)
    elif options.nested_levels == 0:
        options.nested_levels = calculate_nested_levels(options)
        options.nested_particle_mass = calculate_particle_mass(options, refined=True)
    else:
        raise RuntimeError("Invalid options given, run with -h for more information.")
    
    print_info(options)
