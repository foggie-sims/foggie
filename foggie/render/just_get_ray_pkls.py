"""
creates "core sample" velocity plots - JT 070318
"""
import copy
import datashader as dshader
import datashader.transfer_functions as tf
from datashader import reductions
import numpy as np
from scipy.signal import argrelextrema
import pickle
import glob
import os
import argparse
import astropy.units as u
import trident
import yt
from astropy.io import fits

from foggie.utils.consistency import *
mpl.rcParams['font.family'] = 'stixgeneral'
#import foggie.utils.cmap_utils as cmaps
import foggie.clouds.cloud_utils as clouds
import foggie.utils.foggie_utils as futils
from foggie.utils.get_run_loc_etc import get_run_loc_etc


def parse_args():
    '''
    Parse command line arguments.  Returns args object.
    '''
    parser = argparse.ArgumentParser(description="makes a bunch of plots")

    ## optional arguments
    parser.add_argument('--halo', metavar='halo', type=str, action='store',
                        help='which halo? default is 8508 (Tempest)')
    parser.set_defaults(halo="8508")

    ## clobber?
    parser.add_argument('--clobber', dest='clobber', action='store_true')
    parser.add_argument('--no-clobber', dest='clobber', action='store_false', help="default is no clobber")
    parser.set_defaults(clobber=False)

    ## what are we plotting and where is it
    parser.add_argument('--pwd', dest='pwd', action='store_true',
                        help='just use the pwd?, default is no')
    parser.set_defaults(pwd=False)

    parser.add_argument('--run', metavar='run', type=str, action='store',
                        help='which run? default is natural')
    parser.set_defaults(run="natural")

    parser.add_argument('--output', metavar='output', type=str, action='store',
                        help='which output? default is RD0020')
    parser.set_defaults(output="RD0020")

    parser.add_argument('--system', metavar='system', type=str, action='store',
                        help='which system are you on? default is oak')
    parser.set_defaults(system="oak")

    args = parser.parse_args()
    return args

def show_velphase(ds, ray_df, ray_start, ray_end, hdulist, fileroot):
    """ this is the master control program for the 'velphase' plots
        from FOGGIE paper 1. """
    impact = hdulist[0].header['IMPACT']




    size_dict['ray_df'] = ray_df

    pickle.dump(size_dict, open('.' + fileroot+"sizes.pkl", "wb"))


def pickle_ray_file(ds, filename):
    """
    opens a fits file containing a FOGGIE spectrum and returns a dataframe
    with useful quantities along the ray
    """
    print("pickle_ray_file is opening: ", filename)
    hdulist = fits.open(filename)
    ray_start_str = hdulist[0].header['RAYSTART']
    ray_end_str = hdulist[0].header['RAYEND']
    ray_start = [float(ray_start_str.split(",")[0].strip('unitary')),
                 float(ray_start_str.split(",")[1].strip('unitary')),
                 float(ray_start_str.split(",")[2].strip('unitary'))]
    ray_end = [float(ray_end_str.split(",")[0].strip('unitary')),
               float(ray_end_str.split(",")[1].strip('unitary')),
               float(ray_end_str.split(",")[2].strip('unitary'))]
    rs, re = np.array(ray_start), np.array(ray_end)
    rs = ds.arr(rs, "code_length")
    re = ds.arr(re, "code_length")
    ray = ds.ray(rs, re)
    rs = rs.ndarray_view()
    re = re.ndarray_view()
    ray['x-velocity'] = ray['x-velocity'].convert_to_units('km/s')
    ray['y-velocity'] = ray['y-velocity'].convert_to_units('km/s')
    ray['z-velocity'] = ray['z-velocity'].convert_to_units('km/s')
    ray['dx'] = ray['dx'].convert_to_units('cm')


    ### this needs to be updated
    ray_field_list = ["x", "y", "z", "density", "temperature",
                               "metallicity", "pressure", "entropy",
                               "cooling_time", "thermal_energy", # ('gas',"gravitational_potential"),
                               "HI_Density", "cell_mass", "dx",
                               "x-velocity", "y-velocity", "z-velocity",
                               "C_p2_number_density", "C_p3_number_density",
                               "H_p0_number_density", "Mg_p1_number_density",
                               "O_p5_number_density", "Si_p2_number_density",
                               "Si_p1_number_density", "Si_p3_number_density",
                               "Ne_p7_number_density"]
    ray_df = ray.to_dataframe(ray_field_list)

    ray_index, first_axis, second_axis = futils.get_ray_axis(rs, re)

    ray_df = ray_df.sort_values(by=[first_axis])

    size_dict['ray_df'] = ray_df

    print('here is the thing')
    fileroot = filename.strip('los.fits.gz').replace('.','')
    pickle.dump(size_dict, open('.' + fileroot+"sizes.pkl", "wb"))

    return ray_df, rs, re, first_axis, hdulist


def loop_over_rays(ds, dataset_list):
    for filename in dataset_list:
        ray_df, rs, re, axis_to_use, hdulist = pickle_ray_file(ds, filename)


def drive_velphase(ds_name, wildcard):
    """
    for running as imported module.
    """
    ds = yt.load(ds_name)
    trident.add_ion_fields(ds, ions=['Si II', 'Si III', 'Si IV', 'C II',
                                     'C III', 'C IV', 'O VI', 'Mg II',
                                     'Ne VIII'])

    dataset_list = glob.glob(os.path.join(os.getcwd(), wildcard))
    loop_over_rays(ds, dataset_list)



if __name__ == "__main__":
    """
    for running at the command line.
    """

    args = parse_args()
    foggie_dir, output_dir, run_loc, trackname, haloname, spectra_dir, infoname = get_run_loc_etc(args)
    if args.pwd:
        run_dir = '.'
    else:
        run_dir = foggie_dir + run_loc
    ds_loc = run_dir + args.output + "/" + args.output


    dataset_list = glob.glob(os.path.join('.', '*squall*rd0022*los*fits.gz'))
    print('there are ', len(dataset_list), 'files')

    ds = yt.load(ds_loc)

    trident.add_ion_fields(ds, ions=linelist_all) # just add everything

    print('going to loop over rays now')
    loop_over_rays(ds, dataset_list)
    print('~~~ all done! ~~~')
