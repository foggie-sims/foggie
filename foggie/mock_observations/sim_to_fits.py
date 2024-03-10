#### routine to load foggie snapshots and store them in a fits cube
#### written by ramona
#### latest modification 2024-03-11

import numpy as np
import sys
import os
import matplotlib as mpl
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})
mpl.rcParams['font.family'] = 'serif'
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import AxesGrid
import matplotlib.image as mpimg
from yt.units import kpc
from astropy.table import Table
from astropy.io import fits
from astropy.utils.data import download_file
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.convolution import Gaussian2DKernel
from astropy.modeling.models import Lorentz1D
from astropy.convolution import convolve_fft
from astropy.constants import c as speedoflight
import pickle
from functools import partial
import datashader as dshader
import datashader.transfer_functions as tf
from datashader import reductions
from datashader.utils import export_image
import pandas as pd
import trident
from foggie.utils.get_halo_center import get_halo_center
from foggie.utils.get_refine_box import get_refine_box
from foggie.utils.get_proper_box_size import get_proper_box_size
from foggie.utils.consistency import *
from foggie.utils.enzoGalaxyProps import find_rvirial
import yt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator)
import csv
from collections import OrderedDict as odict
mpl.rcParams['axes.linewidth']=1
mpl.rcParams['axes.edgecolor']='k'
from astropy.io import fits
from foggie.utils.foggie_load import foggie_load
from astropy.cosmology import FlatLambdaCDM
from foggie.utils.get_run_loc_etc import get_run_loc_etc
import argparse



def parse_args():
    '''Parse command line arguments. Returns args object.
    NOTE: Need to move command-line argument parsing to separate file.'''

    parser = argparse.ArgumentParser()
    # Optional arguments:
    parser.add_argument('--halo', metavar='halo', type=str, action='store', \
                        help='Which halo? Default is 8508 (Tempest)')
    parser.set_defaults(halo='8508')

    parser.add_argument('--run', metavar='run', type=str, action='store', \
                        help='Which run? Default is nref11c_nref9f. Alternative: nref11n_nref10f')
    parser.set_defaults(run='nref11c_nref9f')

    parser.add_argument('--output', metavar='output', type=str, action='store', \
                        help='Which output? Default is RD0032')
    parser.set_defaults(output='RD0027')

    parser.add_argument('--system', metavar='system', type=str, action='store', \
                        help='Which system are you on? Default is ramona')
    parser.set_defaults(system='ramona')

    parser.add_argument('--pwd', dest='pwd', action='store_true', \
                        help='Just use the working directory? Default is no')
    parser.set_defaults(pwd=False)

    parser.add_argument('--make_fits', dest='make_fits', action='store_true',
                        help="Do you want to create a fits file? Default is True")
    parser.set_defaults(make_fits=True)

    parser.add_argument('--need_halo_center', dest='need_halo_center', action='store_true',
                        help="Do you need the halo center? Default is no.")
    parser.set_defaults(custom_endwl=False)

    args = parser.parse_args()
    return args

def make_fits(args):
    halo = args.halo
    sim = args.run
    snap = args.output
    foggie_dir, output_dir, run_loc, code_path, trackname, haloname, spectra_dir, infofile = get_run_loc_etc(args)
    output_dir = output_dir+"mockobservations/"
    if not (os.path.exists(output_dir)): os.system('mkdir -p ' + output_dir)
    print(halo)
    fn = foggie_dir+'halo_00'+halo+'/'+sim+'/'+snap+'/'+snap
    #track_name = "/Users/raugustin/foggie/foggie/halo_tracks/008508/nref11n_selfshield_15/halo_track_600kpc_nref8"
    track_name = trackname
    os.chdir(output_dir)
    fullds, region = foggie_load(fn,track_name)
    #if args.need_halo_center==True:
    #    fullds, region = foggie_load(fn,track_name)
    #else:
    #    fullds, region = foggie_load(fn,track_name,find_halo_center=False)

    zsnap = fullds.get_parameter('CosmologyCurrentRedshift')
    properwidth = fullds.refine_width # in kpc
    print(properwidth)


    dat = region['gas', 'density'] #or whatever you like
    pix_size = float(np.min(region['gas','dx'].in_units('kpc')))
    res=properwidth/pix_size 
    frbdat = fullds.covering_grid(level=9, left_edge=fullds.halo_center_kpc, dims=[res, res, res]) #modify to whatever
    datnp = np.array(frbdat['gas','density'])

    hdr = fits.Header()

    hdr['HALO'] = halo
    hdr['SIM_OUT'] = sim
    hdr['SNAPSHOT'] = snap
    hdr['REDSHIFT'] = zsnap
    hdr['NAXIS'] = 3
    hdr['NAXIS1'] = res
    hdr['NAXIS2'] = res
    hdr['NAXIS3'] = res
    hdr['CRPIX1'] = pix_size
    hdr['CRPIX2'] = pix_size
    hdr['CD1_1'] = pix_size
    hdr['CD1_2'] = 0.
    hdr['CD2_1'] = 0.
    hdr['CD2_2'] = pix_size
    hdr['CUNIT1'] = 'kpc'
    hdr['CUNIT2'] = 'kpc'
    hdr['CRVAL1'] = 0.
    hdr['CRVAL2'] = 0.
    hdr['CUNIT3'] = 'kpc'
    hdr['CD3_3'] = pix_size
    hdr['CRPIX3'] = pix_size
    hdr['CRVAL3'] = 0.
    primary_hdu = fits.PrimaryHDU(header = hdr)
    data_hdu = fits.ImageHDU(datnp)

    hdulist = fits.HDUList([primary_hdu, data_hdu])
    fitsfile = halo+'_'+sim+'_'+snap+'_'+'.fits'
    if (os.path.exists(output_dir+fitsfile)): os.system('mv ' + output_dir+fitsfile + ' '+ output_dir+'old_'+fitsfile)
    hdulist.writeto(fitsfile)




################################################################################################################################################################################################################################
################################################################################################################################################################################################################################
################################################################################################################################################################################################################################
################################################################################################################################################################################################################################
################################################################################################################################################################################################################################

args = parse_args()
make_fits(args)
print('you just created a new fits file \(^o^)/')
