
"""
This file contains useful helper functions for FOGGIE
use it as:

from foggie.utils import foggie_utils as futils
you can then use, e.g. futils.ds_to_df() functions, etc.

JT 081318

"""
import pandas as pd
import numpy as np
import glob, os
import astropy.units as u
import argparse
from foggie.utils.consistency import phase_color_labels, metal_labels, \
    categorize_by_temp, categorize_by_metals
from astropy.table import Table

CORE_WIDTH = 20.

def get_halo_track(track_file):
    """ takes in a track file name (absolute path required)
    and returns an astropy table with the track"""

    print("opening track: " + track_file)
    track = Table.read(track_file, format='ascii')
    track.sort('col1')

    return track


def get_list_of_spectra(halo, run, wildcard):
    """ This helper function obtains a list of FOGGIE spectra
	in 'pkl' files as usually stored in the collab
	Dropbox. You need to set your FOGGIE_COLLAB env variable
	for it to work properly.

	Accepts halo number (as string) and the run (e.g.
	nref11c_nref9f and returns a list of files."""

    filelist = []
    path = os.environ['FOGGIE_COLLAB'] + '/plots_halo_00'+halo+'/'+run+'/spectra/random/'
    filelist = glob.glob(os.path.join(path,  wildcard+'.pkl'))

    return filelist

def get_list_of_trident_rays(halo, run, wildcard):
    """ This helper function obtains a list of trident
    ray HDF5 files from the FOGGIE pipeline. These will
    be called something_something_tri.h5.
    You need to set your FOGGIE_COLLAB env variable
	for it to work properly.

	Accepts halo number (as string) and the run (e.g.
	nref11c_nref9f and returns a list of files."""

    filelist = []
    path = os.environ['FOGGIE_COLLAB'] + '/plots_halo_00'+halo+'/'+run+'/spectra/random/'
    filelist = glob.glob(os.path.join(path,  wildcard+'_tri.h5'))

    return filelist


def get_ray_axis(ray_start, ray_end):
    """ takes in ray and returns an integer, 0, 1, 2 for x, y, z, orients"""

    axes_labels = ['x', 'y', 'z']
    second_axes = {'x': 'y', 'y': 'z', 'z': 'x'}

    ray_length = ray_end-ray_start
    if (ray_length[0] > 0.):
        ray_index = 0
        first_axis = axes_labels[ray_index]
        second_axis = second_axes[first_axis]
        print('I think this is an x-axis!')
        return ray_index, first_axis, second_axis
    elif (ray_length[1] > 0.):
        ray_index = 1
        first_axis = axes_labels[ray_index]
        second_axis = second_axes[first_axis]
        print('I think this is a y-axis!')
        return ray_index, first_axis, second_axis
    elif (ray_length[2] > 0.):
        ray_index = 2
        first_axis = axes_labels[ray_index]
        second_axis = second_axes[first_axis]
        print('I think this is a z-axis!')
        return ray_index, first_axis, second_axis
    else:
        print('Your ray is bogus, try again!')
        return False


def parse_args():
    '''
    Parse command line arguments.  Returns args object.
    '''
    parser = argparse.ArgumentParser(
        description="extracts spectra from refined region")

    parser.add_argument('--run', metavar='run', type=str, action='store',
                        help='which run? default is nref9f')
    parser.set_defaults(run="nref9f")

    parser.add_argument('--output', metavar='output', type=str, action='store',
                        help='which output? default is RD0020')
    parser.set_defaults(output="RD0020")

    parser.add_argument('--system', metavar='system', type=str, action='store',
                        help='which system are you on? default is oak')
    parser.set_defaults(system="oak")

    parser.add_argument('--fitsfile', metavar='fitsfile', type=str,
                        action='store',
                        help='what fitsfile would you like to read in? \
                                this does not work yet')

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
        ds_loc = ds_base + "halo_008508/nref11n/natural/" + \
            args.output + "/" + args.output
        output_dir = output_path + "plots_halo_008508/nref11n/natural/spectra/"
        haloname = "halo008508_nref11n"
    elif args.run == "nref10f":
        ds_loc = ds_base + "halo_008508/nref11n/nref11n_nref10f_refine200kpc/" + \
            args.output + "/" + args.output
        output_dir = output_path + \
            "plots_halo_008508/nref11n/nref11n_nref10f_refine200kpc/spectra/"
        haloname = "halo008508_nref11n_nref10f"
    elif args.run == "nref9f":
        path_part = "halo_008508/nref11n/nref11n_"+args.run+"_refine200kpc/"
        ds_loc = ds_base + path_part + args.output + "/" + args.output
        output_dir = output_path + \
            "plots_halo_008508/nref11n/nref11n_nref9f_refine200kpc/spectra/"
        haloname = "halo008508_nref11n_nref9f"
    elif args.run == "nref11f":
        ds_loc = ds_base + "halo_008508/nref11n/nref11f_refine200kpc/" + \
            args.output + "/" + args.output
        output_dir = output_path + \
            "plots_halo_008508/nref11n/nref11f_refine200kpc/spectra/"
        haloname = "halo008508_nref11f"

    return ds_loc, output_path, output_dir, haloname




def filter_particles(data_source, filter_particle_types = ['stars', 'dm'], load_particles = False, load_particle_types = ['stars'],\
                        load_particle_fields = []):
    """
    filters dark matter and star particles. optionally, loads particle data.
    if load_particles = True:
            pre-load a given list of load_particle_fields for a given list of load_particle_types.

    Example:
        from foggie.utils.foggie_utils import load_particle_data
        # this will filter particles into "stars" and "dm", new derived fields available to refine_box
        load_particle_data(refine_box)

        # this will filter particles into "stars" and "dm", new derived fields available to refine_box
        # and pre-load particle_index and particle_mass for 'stars'.
        load_particle_data(refine_box, load_particles = True, load_particle_type = ['stars'],
                                                               load_particle_fields = ['particle_index', 'particle_mass'])

    """
    if type(load_particle_types) == str: load_particle_types = [load_particle_types]
    if type(filter_particle_types) == str: filter_particle_types = [filter_particle_types]

    for ptype in filter_particle_types:
        if (ptype, 'particle_index') not in data_source.ds.derived_field_list:
            print ('filtering %s particles...'%ptype)
            import yt
            from foggie.utils import yt_fields
            if ptype == 'stars': func = yt_fields._stars
            elif ptype == 'young_stars': func = yt_fields._young_stars
            elif ptype == 'young_stars3': func = yt_fields._young_stars3
            elif ptype == 'young_stars7': func = yt_fields._young_stars7
            elif ptype == 'young_stars8': func = yt_fields._young_stars8
            elif ptype == 'old_stars': func = yt_fields._old_stars
            elif ptype =='dm': func = yt_fields._dm
            else:
                print ('particle type %s not known'%ptype)
                return
            yt.add_particle_filter(ptype,function=func, filtered_type='all',requires=["particle_type"])
            data_source.ds.add_particle_filter(ptype)

    if load_particles:
        for ptype in load_particle_types:
            print ('loading %s particle data...'%ptype)
            if len(load_particle_fields) == 0: print ('\tno particle fields specified to load...')
            for field_name in load_particle_fields:
              print ('\t loading ("%s", "%s")'%(ptype, field_name))
              data_source[ptype, field_name]
    return







def ds_to_df(ds, ray_start, ray_end):
    """
    this is a utility function that accepts a yt dataset and the start and end
    points of a ray and returns a pandas dataframe that is useful for shading
    and other analysis.
    """
    proper_box_size = ds.get_parameter('CosmologyComovingBoxSize') \
        / ds.get_parameter('CosmologyHubbleConstantNow') * 1000.

    ray_index, first_axis, second_axis = get_ray_axis(
        ray_start, ray_end)
    if (ray_index == 0):
        all_data = ds.r[ray_start[0]:ray_end[0],
                        ray_start[1] - 0.5*CORE_WIDTH/proper_box_size:
                        ray_start[1] + 0.5*CORE_WIDTH/proper_box_size,
                        ray_start[2] - 0.5*CORE_WIDTH/proper_box_size:
                        ray_start[2] + 0.5*CORE_WIDTH/proper_box_size]
    elif (ray_index == 1):
        all_data = ds.r[ray_start[0] - 0.5*CORE_WIDTH/proper_box_size:
                        ray_start[0] + 0.5*CORE_WIDTH/proper_box_size,
                        ray_start[1]:ray_end[1],
                        ray_start[2] - 0.5*CORE_WIDTH/proper_box_size:
                        ray_start[2] + 0.5*CORE_WIDTH/proper_box_size]
    elif (ray_index == 2):
        all_data = ds.r[ray_start[0] - 0.5*CORE_WIDTH/proper_box_size:
                        ray_start[0] + 0.5*CORE_WIDTH/proper_box_size,
                        ray_start[1] - 0.5*CORE_WIDTH/proper_box_size:
                        ray_start[1] + 0.5*CORE_WIDTH/proper_box_size,
                        ray_start[2]:ray_end[2]]
    else:
        print('Your ray is bogus, try again!')

    dens = np.log10(all_data['density'].ndarray_view())
    temp = np.log10(all_data['temperature'].ndarray_view())
    metallicity = all_data['metallicity'].ndarray_view()

    # creates the phase_label as a set of nonsense strings.
    phase_label = categorize_by_temp(temp)
    metal_label = categorize_by_metals(metallicity)

    df = pd.DataFrame({'x': all_data['x'].ndarray_view() * proper_box_size,
                       'y': all_data['y'].ndarray_view() * proper_box_size,
                       'z': all_data['z'].ndarray_view() * proper_box_size,
                       'metallicity': metallicity,
                       'vx': all_data["x-velocity"].in_units('km/s'),
                       'vy': all_data["y-velocity"].in_units('km/s'),
                       'vz': all_data["z-velocity"].in_units('km/s'),
                       'cell_mass': all_data['cell_mass'].in_units('Msun'),
                       'temp': temp, 'dens': dens, 'phase_label': phase_label,
                       'metal_label': metal_label})

    df.phase_label = df.phase_label.astype('category')
    df.metal_label = df.metal_label.astype('category')

    # this is awful, but have to add categories that don't exist to use later.
    existing_categories = df.phase_label.unique()
    for label in phase_color_labels:
        if (not (label in existing_categories)):
            df.phase_label = df.phase_label.cat.add_categories([
                                                               label])

    existing_categories = df.metal_label.unique()
    for label in metal_labels:
        if (not (label in existing_categories)):
            df.metal_label = df.metal_label.cat.add_categories([
                                                               label])

    return df


def cosmic_matter_density(z, h=0.678, omega_m=0.315):

    """
    Calculate the cosmic matter density at a given redshift z in M_sun/Mpc^3.
    
    Parameters:
    z (float): Redshift
    h (float): Hubble constant H_0 in units of 100 km/s/Mpc (default: 0.678, Planck 2018)
    omega_m (float): Matter density parameter at z=0 (default: 0.315, Planck 2018)
    
    Returns:
    float: Matter density in M_sun/Mpc^3
    """
    # Constants
    G = (6.674e-11 * u.m**3 / u.kg / u.s**2) # Gravitational constant in m^3 kg^-1 s^-2
    M_sun = 1.989e33 * u.g # Solar mass in kg
    # Hubble constant in SI units (s^-1)
    H_0 = (h * 100. * u.km / u.s / u.Mpc).to('1/s') #<----correct units! 
    
    # Critical density at z=0 in kg/m^3
    rho_crit_0 = (3 * H_0**2 / (8 * 3.1415926535 * G)).to('Msun/Mpc**3') 
    
    # Matter density at z=0
    rho_m_0 = omega_m * rho_crit_0
    
    # Matter density at redshift z
    rho_m_z = rho_m_0 * (1 + z)**3
    
    return rho_m_z.value