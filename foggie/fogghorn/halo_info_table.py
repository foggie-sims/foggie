'''
Filename: halo_info_table.py
Author: Cassi
Created: 6-12-24
Last modified: 6-12-24 by Cassi
This file works with fogghorn_analysis.py to make a table of various halo info properties that can be used for plotting.
'''

from __future__ import print_function

import numpy as np
import argparse
import os
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial
import multiprocessing as multi
from pathlib import Path

from astropy.table import Table
from astropy.io import ascii

from astropy.cosmology import Planck15 as cosmo

import yt
from yt.units import *
from yt import YTArray

from foggie.utils.consistency import *
from foggie.utils.get_refine_box import get_refine_box
from foggie.utils.get_halo_center import get_halo_center
from foggie.utils.get_proper_box_size import get_proper_box_size
from foggie.utils.get_run_loc_etc import get_run_loc_etc
from foggie.utils.yt_fields import *
from foggie.utils.foggie_load import *
from foggie.utils.analysis_utils import *

def make_table():
    data_names = ['snapshot','time','redshift','halo_x','halo_y','halo_z','halo_vx','halo_vy','halo_vz','halo_mass','halo_radius','stellar_mass','SFR']
    data_types = ['S6','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8']
    data_units = ['none','Myr','none','kpc','kpc','kpc','km/s','km/s','km/s','Msun','kpc','Msun','Msun/yr']
    data = Table(names=data_names, dtype=data_types)
    for i in range(len(data.keys())):
        key = data.keys()[i]
        data[key].unit = data_units[i]
    return data

def get_halo_info(ds, args):
    '''Calculates basic information about the halo: snapshot name, time, redshift, halo x,y,z location, halo vx,vy,vz bulk velocity, virial mass, virial radius, stellar mass, star formation rate.
    NOTE: The virial mass and radius as currently written will only work for the central galaxies! Rockstar is not being run to find satellite halos.'''
    
    # Load the table if it exists, or make it if it does not exist
    if (os.path.exists(args.save_directory + '/halo_data.txt')):
        data = Table.read(args.save_directory + '/halo_data.txt', format='ascii.fixed_width')
    else:
        data = make_table()

    # Determine if this snapshot has already had its information calculated
    if (args.snap in data['snapshot']):
        if not args.silent: print('Halo info for snapshot ' + args.snap + ' already calculated.', )
        if args.clobber:
            if not args.silent: print(' But we will re-calculate it...')
            calc = True
        else:
            if not args.silent: print(' So we will skip it.')
            calc = False
    else: calc = True

    if (calc):
        if not args.silent: print('About to calculate halo info for ' + args.snap + '...')

        row = [args.snap, ds.current_time.in_units('Myr').v, ds.get_parameter('CosmologyCurrentRedshift'), \
               ds.halo_center_kpc[0], ds.halo_center_kpc[1], ds.halo_center_kpc[2], \
               ds.halo_velocity_kms[0], ds.halo_velocity_kms[1], ds.halo_velocity_kms[2]]
        
        sph = ds.sphere(center = ds.halo_center_kpc, radius = (400., 'kpc'))
        filter_particles(sph)
        prof_dm = yt.create_profile(sph, ('index', 'radius'), fields = [('deposit', 'dm_mass')], \
                                    n_bins = 500, weight_field = None, accumulation = True)
        prof_stars = yt.create_profile(sph, ('index', 'radius'), fields = [('deposit', 'stars_mass')], \
                                    n_bins = 500, weight_field = None, accumulation = True)
        prof_young_stars = yt.create_profile(sph, ('index', 'radius'), fields = [('deposit', 'young_stars_mass')], \
                                    n_bins = 500, weight_field = None, accumulation = True)
        prof_gas = yt.create_profile(sph, ('index', 'radius'), fields = [('gas', 'cell_mass')],\
                                    n_bins = 500, weight_field = None, accumulation = True)

        internal_density =  (prof_dm[('deposit', 'dm_mass')].to('g') + prof_stars[('deposit', 'stars_mass')].to('g') + \
                             prof_gas[('gas', 'cell_mass')].to('g'))/(4*np.pi*prof_dm.x.to('cm')**3./3.)

        rho_crit = cosmo.critical_density(ds.current_redshift)
        rvir = prof_dm.x[np.argmin(abs(internal_density.value - 200*rho_crit.value))]
        Mdm_rvir    = prof_dm[('deposit', 'dm_mass')][np.argmin(abs(internal_density.value - 200*rho_crit.value))]
        Mstars_rvir = prof_stars[('deposit', 'stars_mass')][np.argmin(abs(internal_density.value - 200*rho_crit.value))]
        Mgas_rvir   = prof_gas[('gas', 'cell_mass')][np.argmin(abs(internal_density.value - 200*rho_crit.value))]
        Mvir = Mdm_rvir + Mstars_rvir + Mgas_rvir
        Myoung_stars = prof_young_stars[('deposit','young_stars_mass')][np.where(prof_young_stars.x.to('kpc') >= 20.)[0][0]]
        SFR = Myoung_stars.to('Msun').v/1e7

        row.append(Mvir.to('Msun').v)
        row.append(rvir.to('kpc').v)
        row.append(Mstars_rvir.to('Msun').v)
        row.append(SFR)

        data.add_row(row)
        data.write(args.save_directory + '/halo_data.txt', format='ascii.fixed_width', append=True)