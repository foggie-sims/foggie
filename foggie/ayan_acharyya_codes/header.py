#!/usr/bin/env python3

"""

    Title :      header
    Notes :      Header file for importing packages/modulesand parsing args required for working with FOGGIE code.
    Author :     Ayan Acharyya
    Started :    January 2021

"""

import numpy as np
import multiprocessing as mproc
import os, sys, argparse, re, subprocess, time, math

from matplotlib import pyplot as plt

from pathlib import Path
from importlib import reload

from scipy import optimize as op
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.interpolate import LinearNDInterpolator as LND
from scipy.special import erf

from astropy.io import ascii, fits
from astropy.table import Table

from operator import itemgetter

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)

import yt
from yt.units import *
import yt.visualization.eps_writer as eps

from foggie.utils.get_run_loc_etc import *
from foggie.utils.consistency import *
from foggie.utils.yt_fields import *
from foggie.utils.foggie_load import *

# -------------------------------------------------------------------------------------------
def myprint(text, args):
    '''
    Function to direct the print output to stdout or a file, depending upon user args
    '''
    if not text[-1] == '\n': text += '\n'
    if not args.silent:
        if args.print_to_file:
            ofile = open(args.printoutfile, 'a')
            ofile.write(text)
            ofile.close()
        else:
            print(text)

# -------------------------------------------------------------------------------------------
def get_cube_output_path(args, diag='D16', Om=0.5):
    '''
    Function to deduce which specific directory (in this jungle of folders) a given ifu datacube should be stored
    '''
    cube_output_path = args.output_dir + 'fits/' + args.output + '/diagnostic_' + diag + '/Om_' + str(Om) + '/inc_' + \
        str(args.inclination) + '/spectral_res_' + str(args.base_spec_res) + '/spatial_res_' + str(args.base_spatial_res)
    Path(cube_output_path).mkdir(parents=True, exist_ok=True) # creating the directory structure, if doesn't exist already

    return cube_output_path

# -------------------------------------------------------------------------------------------
def write_fits(filename, data, args, fill_val=np.nan, for_qfits=True):
    '''
    Function to write a datacube to a FITS file
    '''
    if for_qfits and np.shape(data)[0] == np.shape(data)[1]: data = data.swapaxes(0,2) # QFitsView requires (wave, pos, pos) arrangement rather than (pos, pos, wave)  arrangement

    hdu = fits.PrimaryHDU(np.ma.filled(data, fill_value=fill_val))
    hdulist = fits.HDUList([hdu])
    if filename[-5:] != '.fits':
        filename += '.fits'
    hdulist.writeto(filename, clobber=True)
    myprint('Written file ' + filename + '\n', args)

# ----------------------------------------------------------------------------------------------
def pull_halo_center(args):
    '''
    Function to pull halo center from halo catalogue, if exists, otherwise compute halo center
    Adapted from utils.foggie_load()
    '''

    foggie_dir, output_dir, run_loc, code_path, trackname, haloname, spectra_dir, infofile = get_run_loc_etc(args)
    args.output_dir = output_dir # so that output_dir is automatically propagated henceforth as args
    halos_df_name = code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/' + 'halo_c_v'

    if os.path.exists(halos_df_name):
        halos_df = pd.read_table(halos_df_name, sep='|')
        halos_df.columns = halos_df.columns.str.strip() # trimming column names of extra whitespace
        halos_df['name'] = halos_df['name'].str.strip() # trimming column 'name' of extra whitespace

        if halos_df['name'].str.contains(args.output).any():
            print("Pulling halo center from catalog file")
            halo_ind = halos_df.index[halos_df['name'] == args.output][0]
            args.halo_center = halos_df.loc[halo_ind, ['xc', 'yc', 'zc']].values # in kpc units
            args.halo_velocity = halos_df.loc[halo_ind, ['xv', 'yv', 'zv']].values # in km/s units
            calc_hc = False
        else:
            print('This snapshot is not in the halos_df file, calculating halo center...')
            calc_hc = True
    else:
        print("This halos_df file doesn't exist, calculating halo center...")
        calc_hc = True
    if calc_hc:
        ds, refine_box = load_sim(args, region='refine_box')
        args.halo_center = ds.halo_center_kpc
        args.halo_velocity = ds.halo_velocity_kms
    return args

# --------------------------------------------------------------------------------------------------------------
def parse_args(haloname, RDname):
    '''
    Function to parse keyword arguments
    '''

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='''identify satellites in FOGGIE simulations''')
    # ---- common args used widely over the full codebase ------------
    parser.add_argument('--system', metavar='system', type=str, action='store', help='Which system are you on? Default is Jase')
    parser.set_defaults(system='ayan_local')

    parser.add_argument('--do', metavar='do', type=str, action='store', help='Which particles do you want to plot? Default is gas')
    parser.set_defaults(do='gas')

    parser.add_argument('--run', metavar='run', type=str, action='store', help='which run? default is natural')
    parser.set_defaults(run="nref11c_nref9f")

    parser.add_argument('--halo', metavar='halo', type=str, action='store', help='which halo? default is 8508 (Tempest)')
    parser.set_defaults(halo=haloname)

    parser.add_argument('--proj', metavar='proj', type=str, action='store', help='Which projection do you want to plot? Default is x')
    parser.set_defaults(proj='x')

    parser.add_argument('--output', metavar='output', type=str, action='store', help='which output? default is RD0020')
    parser.set_defaults(output=RDname)

    parser.add_argument('--pwd', dest='pwd', action='store_true', help='Just use the current working directory?, default is no')
    parser.set_defaults(pwd=False)

    parser.add_argument('--do_all_sims', dest='do_all_sims', action='store_true', help='Run the code on all simulation snapshots available?, default is no')
    parser.set_defaults(do_all_sims=False)

    parser.add_argument('--silent', dest='silent', action='store_true', help='Suppress all print statements?, default is no')
    parser.set_defaults(silent=False)

    # ------- args added for filter_star_properties.py ------------------------------
    parser.add_argument('--plot_proj', dest='plot_proj', action='store_true', help='plot projection map? default is no')
    parser.set_defaults(plot_proj=False)

    parser.add_argument('--clobber', dest='clobber', action='store_true', help='overwrite existing outputs with same name?, default is no')
    parser.set_defaults(clobber=False)

    parser.add_argument('--automate', dest='automate', action='store_true', help='automatically execute the next script?, default is no')
    parser.set_defaults(automate=False)

    # ------- args added for compute_hii_radii.py ------------------------------
    parser.add_argument('--galrad', metavar='galrad', type=float, action='store', help='radius of the galaxy, in kpc, i.e. the radial extent to which computations will be done; default is 50')
    parser.set_defaults(galrad=20.)

    parser.add_argument('--galthick', metavar='galthick', type=float, action='store', help='thickness of stellar disk, in kpc; default is 0.4 kpc = 400 pc')
    parser.set_defaults(galthick=0.4)

    parser.add_argument('--mergeHII', metavar='mergeHII', type=float, action='store', help='separation btwn HII regions below which to merge them, in kpc; default is None i.e., do not merge')
    parser.set_defaults(mergeHII=None)

    # ------- args added for lookup_flux.py ------------------------------
    parser.add_argument('--diag_arr', metavar='diag_arr', type=str, action='store', help='list of metallicity diagnostics to use')
    parser.set_defaults(diag_arr='D16')

    parser.add_argument('--Om_arr', metavar='Om_arr', type=float, action='store', help='list of Omega values to use')
    parser.set_defaults(Om_arr=0.5)

    parser.add_argument('--nooutliers', dest='nooutliers', action='store_true', help='discard outlier HII regions (according to D16 diagnostic)?, default is no')
    parser.set_defaults(nooutliers=False)

    parser.add_argument('--xratio', metavar='xratio', type=str, action='store', help='ratio of lines to plot on X-axis; default is None')
    parser.set_defaults(xratio=None)

    parser.add_argument('--yratio', metavar='yratio', type=str, action='store', help='ratio of lines to plot on Y-axis; default is None')
    parser.set_defaults(yratio=None)

    parser.add_argument('--fontsize', metavar='fontsize', type=int, action='store', help='fontsize of plot labels, etc.; default is 15')
    parser.set_defaults(fontsize=15)

    parser.add_argument('--plot_metgrad', dest='plot_metgrad', action='store_true', help='make metallicity gradient plot?, default is no')
    parser.set_defaults(plot_metgrad=False)

    parser.add_argument('--plot_phase_space', dest='plot_phase_space', action='store_true', help='make P-r phase space plot?, default is no')
    parser.set_defaults(plot_phase_space=False)

    parser.add_argument('--plot_obsv_phase_space', dest='plot_obsv_phase_space', action='store_true', help='overlay observed P-r phase space on plot?, default is no')
    parser.set_defaults(plot_obsv_phase_space=False)

    parser.add_argument('--plot_fluxgrid', dest='plot_fluxgrid', action='store_true', help='make flux ratio grid plot?, default is no')
    parser.set_defaults(plot_fluxgrid=False)

    parser.add_argument('--annotate', dest='annotate', action='store_true', help='annotate grid plot?, default is no')
    parser.set_defaults(annotate=False)

    parser.add_argument('--pause', dest='pause', action='store_true', help='pause after annotating each grid?, default is no')
    parser.set_defaults(pause=False)

    parser.add_argument('--plot_Zin_Zout', dest='plot_Zin_Zout', action='store_true', help='make input vs output metallicity plot?, default is no')
    parser.set_defaults(plot_Zin_Zout=False)

    parser.add_argument('--saveplot', dest='saveplot', action='store_true', help='save the plot?, default is no')
    parser.set_defaults(saveplot=False)

    parser.add_argument('--keep', dest='keep', action='store_true', help='keep previously displayed plots on screen?, default is no')
    parser.set_defaults(keep=False)

    parser.add_argument('--use_RGI', dest='use_RGI', action='store_true', help='kuse RGI interpolation vs LND?, default is no')
    parser.set_defaults(use_RGI=False)

    # ------- args added for make_ideal_datacube.py ------------------------------
    parser.add_argument('--wave_start', metavar='wave_start', type=float, action='store', help='starting (bluest) wavelength for the ifu datacube, in A; default is 6400 A')
    parser.set_defaults(wave_start=6400.)

    parser.add_argument('--wave_end', metavar='wave_end', type=float, action='store', help='last (reddest) wavelength for the ifu datacube, in A; default is 6800 A')
    parser.set_defaults(wave_end=6800.)

    parser.add_argument('--vel_disp', metavar='vel_disp', type=float, action='store', help='intrinsic velocity dispersion for each emission line, in km/s; default is 15 km/s')
    parser.set_defaults(vel_disp=15.)

    parser.add_argument('--nbin_cont', metavar='nbin_cont', type=int, action='store', help='no. of spectral bins to bin the continuum (witout emission lines) in to; default is 1000')
    parser.set_defaults(nbin_cont=1000)

    parser.add_argument('--vel_highres_win', metavar='vel_highres_win', type=float, action='store', help='velocity window on either side of each emission line, in km/s, within which the continuum is resolved into finer (nbin_highres_cont) spectral elements; default is 500 km/s')
    parser.set_defaults(vel_highres_win=500.)

    parser.add_argument('--nbin_highres_cont', metavar='nbin_highres_cont', type=int, action='store', help='no. of additonal spectral bins to introduce around each emission line; default is 100')
    parser.set_defaults(nbin_highres_cont=100)

    parser.add_argument('--base_spec_res', metavar='base_spec_res', type=float, action='store', help='base spectral resolution, in km/s, i.e. to be employed while making the ideal datacube; default is 30 km/s')
    parser.set_defaults(base_spec_res=30.)

    parser.add_argument('--base_spatial_res', metavar='base_spatial_res', type=float, action='store', help='base spatial resolution, in kpc, i.e. to be employed while making the ideal datacube; default is 0.04 kpc = 40 pc')
    parser.set_defaults(base_spatial_res=0.04)

    parser.add_argument('--inclination', metavar='inclination', type=float, action='store', help='inclination angle to rotate the galaxy by, on YZ plane (keeping X fixed), in degrees; default is 0')
    parser.set_defaults(inclination=0.)

    parser.add_argument('--print_to_file', dest='print_to_file', action='store_true', help='Redirect all print statements to a file?, default is no')
    parser.set_defaults(print_to_file=False)

    parser.add_argument('--printoutfile', metavar='printoutfile', type=str, action='store', help='file to write all print statements to; default is ./logfile.out')
    parser.set_defaults(printoutfile='./logfile.out')

    parser.add_argument('--debug', dest='debug', action='store_true', help='run in debug mode (lots of print checks)?, default is no')
    parser.set_defaults(debug=False)

    # ------- wrap up and processing args ------------------------------
    args = parser.parse_args()

    args.diag_arr = [item for item in args.diag_arr.split(',')]
    args.Om_arr = [float(item) for item in str(args.Om_arr).split(',')]
    args.mergeHII_text = '_mergeHII=' + str(args.mergeHII) + 'kpc' if args.mergeHII is not None else '' # to be used as filename suffix to denote whether HII regions have been merged
    args.without_outlier = '_no_outlier' if args.nooutliers else '' # to be used as filename suffix to denote whether outlier HII regions (as per D16 density criteria) have been discarded

    args = pull_halo_center(args) # pull details about center of the snapshot
    return args

# ------------declaring overall paths (can be modified on a machine/user basis)-----------
HOME = os.getenv('HOME')
mappings_lab_dir = HOME + '/Mappings/lab/' # if you are producing the MAPPINGS grid,
                                          # this is where your MAPPINGS executable .map51 is installed,
                                          # otherwise, this is where your MAPPINGS grid and your emission line list is
mappings_input_dir = HOME + '/Mappings/HIIGrid306/Q/inputs/' # if you are producing the MAPPINGS grid,
                                                             # this is where your MAPPINGS input/ directory is
                                                             # otherwise, ignore this variable
sb99_dir = HOME + '/SB99-v8-02/output/' # this is where your Starburst99 model outputs reside
                                        # this path is used only when you are using compute_hiir_radii.py or lookup_flux.py
sb99_model = 'starburst11'  # for fixed stellar mass input spectra = 1e6 Msun, run up to 10 Myr
sb99_mass = 1e6 # Msun, mass of star cluster in given SB99 model

# ------------declaring list of ALL simulations-----------
#all_sims = [('8508', 'RD0042'), ('5036', 'RD0039'), ('5016', 'RD0042'), ('4123', 'RD0031'), ('2878', 'RD0020'), ('2392', 'RD0030')] # only the latest (lowest z) available snapshot for each halo

#all_sims = [('8508', 'RD0030'), ('5036', 'RD0030'), ('5016', 'RD0030'), ('4123', 'RD0030'), ('2392', 'RD0030')] # all same z (=0.7) snapshots for each halo

all_sims = [('8508', 'RD0042'), ('8508', 'RD0039'), ('8508', 'RD0031'), ('8508', 'RD0030'), \
            ('5036', 'RD0039'), ('5036', 'RD0031'), ('5036', 'RD0030'), ('5036', 'RD0020'), \
            ('5016', 'RD0042'), ('5016', 'RD0039'), ('5016', 'RD0031'), ('5016', 'RD0030'), ('5016', 'RD0020'), \
            ('4123', 'RD0031'), ('4123', 'RD0030'), \
            ('2878', 'RD0020'), ('2878', 'RD0018'), \
            ('2392', 'RD0030'), \
            ] # all snapshots in the HD
