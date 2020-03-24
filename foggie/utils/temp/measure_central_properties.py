#!/u/rcsimons/miniconda3/bin/python3.7
import astropy
from astropy.io import fits
import numpy as np
from numpy import *
import math
from foggie.utils.get_run_loc_etc import get_run_loc_etc
from foggie.utils.foggie_load import *
import os, sys, argparse
from collections import OrderedDict
import yt
import matplotlib.pyplot as plt
import trident
import numpy


def parse_args():
    '''
    Parse command line arguments
    ''' 
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='''\
                                Generate the cameras to use in Sunrise and make projection plots
                                of the data for some of these cameras. Then export the data within
                                the fov to a FITS file in a format that Sunrise understands.
                                ''')

    parser.add_argument('-system', '--system', metavar='system', type=str, action='store', \
                        help='Which system are you on? Default is Jase')
    parser.set_defaults(system="pleiades_raymond")

    parser.add_argument('-output_dir', '--output_dir', metavar='output_dir', type=str, action='store', \
                        help='where is the output located')
    parser.set_defaults(output_dir="~/need_location")

    parser.add_argument('--use_halo_c_v', dest='use_halo_c_v', action='store_true',
                        help='just use the pwd?, default is no')
    parser.set_defaults(use_halo_c_v=False)

    parser.add_argument('--run', metavar='run', type=str, action='store',
                        help='which run? default is natural')
    parser.set_defaults(run="nref11c_nref9f")

    parser.add_argument('--halo', metavar='halo', type=str, action='store',
                        help='which halo? default is 8508 (Tempest)')
    parser.set_defaults(halo="8508")

    parser.add_argument('--pwd', dest='pwd', action='store_true',
                        help='just use the pwd?, default is no')
    parser.set_defaults(pwd=False)


    parser.add_argument('--output', metavar='output', type=str, action='store',
                        help='which output? default is RD0020')
    parser.set_defaults(output="RD0020")


    args = parser.parse_args()
    return args





def weighted_quantile(values, quantiles, sample_weight=None, values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of initial array
    :param old_style: if True, will correct output to be consistent with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = numpy.array(values)
    quantiles = numpy.array(quantiles)
    if sample_weight is None:
        sample_weight = numpy.ones(len(values))
    sample_weight = numpy.array(sample_weight)
    assert numpy.all(quantiles >= 0) and numpy.all(quantiles <= 1), 'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = numpy.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = numpy.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= numpy.sum(sample_weight)
    return numpy.interp(quantiles, weighted_quantiles, values)



def write_mass_fits(ds, args, r_arr = np.array([1, 10, 100])):
    species_dict = {'dark_matter'    : ("darkmatter", "particle_mass"),
                    'gas_tot'        : ("gas", "cell_mass"),
                    'gas_metals'     : ("gas", "metal_mass"),
                    'stars_mass'     : ("stars", "particle_mass"),                    
                    'stars_youngmass': ("youngstars", "particle_mass"),
                    'gas_H'      : ("gas", 'H_mass'),
                    'gas_H0'     : ("gas", 'H_p0_mass'),
                    'gas_H1'     : ("gas", 'H_p1_mass'),
                    'gas_CII'    : ("gas", 'C_p1_mass'),
                    'gas_CIII'   : ("gas", 'C_p2_mass'),
                    'gas_CIV'    : ("gas", 'C_p3_mass'),
                    'gas_OVI'    : ("gas", 'O_p5_mass'),
                    'gas_OVII'   : ("gas", 'O_p6_mass'),
                    'gas_MgII'   : ("gas", 'Mg_p1_mass'),
                    'gas_SII'    : ("gas", "Si_p1_mass"),
                    'gas_SIII'   : ("gas", "Si_p2_mass"),
                    'gas_SIV'    : ("gas", "Si_p3_mass"),
                    'gas_NeVIII' : ("gas", 'Ne_p7_mass')}
    # can do the same thing with species_dict.keys(), but it essentially randomizes the order
    species_keys = ['dark_matter',
                    'gas_tot',        
                    'gas_metals',     
                    'stars_mass',     
                    'stars_youngmass',
                    'gas_H',      
                    'gas_H0',     
                    'gas_H1',     
                    'gas_CII',    
                    'gas_CIII',   
                    'gas_CIV',    
                    'gas_OVI',    
                    'gas_OVII',   
                    'gas_MgII',   
                    'gas_SII',    
                    'gas_SIII',   
                    'gas_SIV',    
                    'gas_NeVIII']

    fits_name = args.output_dir + '/%s_%s_%s_mass.fits'%(args.run, args.halo, args.output)
    #if os.path.exists(fits_name): return
    if not os.path.isfile(fits_name):
        master_hdulist = []
        prihdr = fits.Header()
        prihdr['COMMENT'] = "Storing the mass profiles in this FITS file."
        prihdr['run'] = args.run
        prihdr['halo'] = args.halo
        prihdr['output'] = args.output

        prihdu = fits.PrimaryHDU(header=prihdr)    
        master_hdulist.append(prihdu)
        sat_hdus = []
        masses = {}
        for key in species_keys: masses[key] = []
        for rr, r in enumerate(r_arr):        
            print (rr, r)
            print ('Calculating mass inside %i kpc sphere'%r)
            gc_sphere =  ds.sphere(ds.halo_center_kpc, ds.arr(r,'kpc'))
            for key in species_keys: 
                print (key)
                masses[key].append(gc_sphere.quantities.total_quantity([species_dict[key]]).to('Msun'))
        cols = []
        cols.append(fits.Column(name = 'radius', array =  np.array(r_arr), format = 'D'))
        for key in species_keys: cols.append(fits.Column(name = key, array =  np.array(masses[key]), format = 'D'))
        cols = fits.ColDefs(cols)            
        master_hdulist.append(fits.BinTableHDU.from_columns(cols, name = hd_name))
        thdulist = fits.HDUList(master_hdulist)
        print ('\tSaving to ' + fits_name)

        thdulist.writeto(fits_name, overwrite = True)


def load_sim(args):
    foggie_dir, output_dir, run_loc, trackname, haloname, spectra_dir, infofile = get_run_loc_etc(args)
    track_dir =  trackname.split('halo_tracks')[0]   + 'halo_infos/00' + args.halo + '/' + args.run + '/'
    snap_name = foggie_dir + run_loc + args.output + '/' + args.output
    ds, refine_box, refine_box_center, refine_width = load(snap = snap_name, 
                                                           trackfile = trackname, 
                                                           use_halo_c_v=args.use_halo_c_v, 
                                                           halo_c_v_name=track_dir + 'halo_c_v')
    return ds


if __name__ == '__main__':

    args = parse_args()
    output_directory, prefix = check_paths(args)
    ds = load_sim(args)

    print ('adding trident fields...')
    trident.add_ion_fields(ds, ions=['O VI', 'O VII', 'Mg II', 'Si II', 'C II', 'C III', 'C IV',  'Si III', 'Si IV', 'Ne VIII'])
    
    write_mass_fits(ds, args)























