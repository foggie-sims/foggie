from __future__ import print_function

import numpy as np
from scipy import stats

import yt

import argparse
import os
import glob
import sys

try:
    import cPickle as pickle
except ImportError:
    import pickle

from astropy.table import Table

from consistency import *
from get_refine_box import get_refine_box
from get_halo_center import get_halo_center
from get_proper_box_size import get_proper_box_size

import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})

yt.enable_parallelism()


def parse_args():
    '''
    Parse command line arguments.  Returns args object.
    '''
    parser = argparse.ArgumentParser(description="makes a bunch of plots")

    ## optional arguments
    parser.add_argument('--halo', metavar='halo', type=str, action='store',
                        help='which halo? default is 8508 (Tempest)')
    parser.set_defaults(halo="8508")

    ## what are we plotting and where is it
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

#-----------------------------------------------------------------------------------------------------
####################    PARTICLES          ######################################
@yt.particle_filter(requires=["particle_type"], filtered_type='all')
def stars(pfilter, data):
    filter = data[(pfilter.filtered_type, "particle_type")] == 2
    return filter

## these are the must refine particles; no dm particle type 0's should be there!
@yt.particle_filter(requires=["particle_type"], filtered_type='all')
def dm(pfilter, data):
    filter = data[(pfilter.filtered_type, "particle_type")] == 4
    return filter

####################    FLUXES          ######################################
@derived_field(name="gas_density_in", units="Msun/kpc**3", force_override=True)
def _gas_density_in(field, data):
    gas_density_in = data['density']
    values_flags = data['radial_velocity'] > 0.  ## positive radial velocity = gas going out
    gas_density_in[values_flags] = 0
    return gas_density_in

@derived_field(name="gas_density_out", units="Msun/kpc**3", force_override=True)
def _gas_density_out(field, data):
    gas_density_out = data['density']
    values_flags = data['radial_velocity'] < 0.  ## positive radial velocity = gas going out
    gas_density_out[values_flags] = 0
    return gas_density_out

@derived_field(name="metal_density_in", units="Msun/kpc**3", force_override=True)
def _metal_density_in(field, data):
    metal_density_in = data['metal_density']
    values_flags = data['radial_velocity'] > 0.  ## positive radial velocity = metal going out
    metal_density_in[values_flags] = 0
    return metal_density_in

@derived_field(name="metal_density_out", units="Msun/kpc**3", force_override=True)
def _metal_density_out(field, data):
    metal_density_out = data['metal_density']
    values_flags = data['radial_velocity'] < 0.  ## positive radial velocity = metal going out
    metal_density_out[values_flags] = 0
    return metal_density_out

@derived_field(name="hot_gas_density", units="Msun/kpc**3", force_override=True)
def _hot_gas_density(field, data):
    gas_density = data['density']
    values_flags = (data['temperature'].in_units('K') < 1.e6)
    gas_density[values_flags] = 0
    return gas_density

@derived_field(name="hot_gas_density_in", units="Msun/kpc**3", force_override=True)
def _hot_gas_density_in(field, data):
    gas_density_in = data['density']
    values_flags = (data['radial_velocity'] > 0.) | (data['temperature'].in_units('K') < 1.e6)
    gas_density_in[values_flags] = 0
    return gas_density_in

@derived_field(name="hot_gas_density_out", units="Msun/kpc**3", force_override=True)
def _hot_gas_density_out(field, data):
    gas_density_out = data['density']
    values_flags = (data['radial_velocity'] < 0.) | (data['temperature'].in_units('K') < 1.e6)
    gas_density_out[values_flags] = 0
    return gas_density_out

@derived_field(name="warm_gas_density", units="Msun/kpc**3", force_override=True)
def _warm_gas_density(field, data):
    gas_density = data['density']
    values_flags =(data['temperature'].in_units('K') < 1.e5) | (data['temperature'].in_units('K') >= 1.e6)
    gas_density[values_flags] = 0
    return gas_density

@derived_field(name="warm_gas_density_in", units="Msun/kpc**3", force_override=True)
def _warm_gas_density_in(field, data):
    gas_density_in = data['density']
    values_flags = (data['radial_velocity'] > 0.) | (data['temperature'].in_units('K') < 1.e5) | (data['temperature'].in_units('K') >= 1.e6)
    gas_density_in[values_flags] = 0
    return gas_density_in

@derived_field(name="warm_gas_density_out", units="Msun/kpc**3", force_override=True)
def _warm_gas_density_out(field, data):
    gas_density_out = data['density']
    values_flags = (data['radial_velocity'] < 0.) | (data['temperature'].in_units('K') < 1.e5) | (data['temperature'].in_units('K') >= 1.e6)
    gas_density_out[values_flags] = 0
    return gas_density_out

@derived_field(name="cool_gas_density", units="Msun/kpc**3", force_override=True)
def _cool_gas_density(field, data):
    gas_density = data['density']
    values_flags = (data['temperature'].in_units('K') < 1.e4) | (data['temperature'].in_units('K') >= 1.e5)
    gas_density[values_flags] = 0
    return gas_density

@derived_field(name="cool_gas_density_in", units="Msun/kpc**3", force_override=True)
def _cool_gas_density_in(field, data):
    gas_density_in = data['density']
    values_flags = (data['radial_velocity'] > 0.) | (data['temperature'].in_units('K') < 1.e4) | (data['temperature'].in_units('K') >= 1.e5)
    gas_density_in[values_flags] = 0
    return gas_density_in

@derived_field(name="cool_gas_density_out", units="Msun/kpc**3", force_override=True)
def _cool_gas_density_out(field, data):
    gas_density_out = data['density']
    values_flags = (data['radial_velocity'] < 0.) | (data['temperature'].in_units('K') < 1.e4) | (data['temperature'].in_units('K') >= 1.e5)
    gas_density_out[values_flags] = 0
    return gas_density_out

@derived_field(name="cold_gas_density", units="Msun/kpc**3", force_override=True)
def _cold_gas_density(field, data):
    gas_density = data['density']
    values_flags = (data['temperature'].in_units('K') >= 1.e4)
    gas_density[values_flags] = 0
    return gas_density

@derived_field(name="cold_gas_density_in", units="Msun/kpc**3", force_override=True)
def _cold_gas_density_in(field, data):
    gas_density_in = data['density']
    values_flags = (data['radial_velocity'] > 0.) | (data['temperature'].in_units('K') >= 1.e4)
    gas_density_in[values_flags] = 0
    return gas_density_in

@derived_field(name="cold_gas_density_out", units="Msun/kpc**3", force_override=True)
def _cold_gas_density_out(field, data):
    gas_density_out = data['density']
    values_flags = (data['radial_velocity'] < 0.) | (data['temperature'].in_units('K') >= 1.e4)
    gas_density_out[values_flags] = 0
    return gas_density_out


#-----------------------------------------------------------------------------------------------------

def calc_ang_mom_and_fluxes(halo, foggie_dir, output_dir, run, **kwargs):
    outs = kwargs.get("outs", "all")
    trackname = kwargs.get("trackname", "halo_track")

    ### set up the table of all the stuff we want
    data = Table(names=('redshift', 'radius', 'nref_mode', \
                        'net_mass_flux', 'net_metal_flux', \
                        'mass_flux_in', 'mass_flux_out', \
                        'metal_flux_in', 'metal_flux_out', \
                        'net_cold_mass_flux', 'cold_mass_flux_in', 'cold_mass_flux_out', \
                        'net_cool_mass_flux', 'cool_mass_flux_in', 'cool_mass_flux_out', \
                        'net_warm_mass_flux', 'warm_mass_flux_in', 'warm_mass_flux_out', \
                        'net_hot_mass_flux', 'hot_mass_flux_in', 'hot_mass_flux_out', \
                        'annular_ang_mom_gas_x', 'annular_ang_mom_gas_y','annular_ang_mom_gas_z', \
                        'annular_spec_ang_mom_gas_x', 'annular_spec_ang_mom_gas_y','annular_spec_ang_mom_gas_z',\
                        'annular_ang_mom_dm_x', 'annular_ang_mom_dm_y','annular_ang_mom_dm_z', \
                        'annular_spec_ang_mom_dm_x', 'annular_spec_ang_mom_dm_y', 'annular_spec_ang_mom_dm_z', \
                        'outside_ang_mom_gas_x', 'outside_ang_mom_gas_y', 'outside_ang_mom_gas_z',  \
                        'outside_spec_ang_mom_gas_x', 'outside_spec_ang_mom_gas_y', 'outside_spec_ang_mom_gas_z', \
                        'outside_ang_mom_dm_x', 'outside_ang_mom_dm_y','outside_ang_mom_dm_z',\
                        'outside_spec_ang_mom_dm_x', 'outside_spec_ang_mom_dm_y', 'outside_spec_ang_mom_dm_z', \
                        'inside_ang_mom_stars_x', 'inside_ang_mom_stars_y', 'inside_ang_mom_stars_z', \
                        'inside_spec_ang_mom_stars_x', 'inside_spec_ang_mom_stars_y', 'inside_spec_ang_mom_stars_z'),
                  dtype=('f8', 'f8', 'i8',
                         'f8', 'f8', 'f8', 'f8', 'f8', 'f8',
                         'f8', 'f8', 'f8', 'f8', 'f8', 'f8',
                         'f8', 'f8', 'f8', 'f8', 'f8', 'f8',
                         'f8', 'f8', 'f8', 'f8', 'f8', 'f8',
                         'f8', 'f8', 'f8', 'f8', 'f8', 'f8',
                         'f8', 'f8', 'f8', 'f8', 'f8', 'f8',
                         'f8', 'f8', 'f8', 'f8', 'f8', 'f8',
                         'f8', 'f8', 'f8', 'f8', 'f8', 'f8'
                        ))


    print(foggie_dir)
    track_name = foggie_dir + 'halo_00' + str(halo) + '/' + run + '/' + trackname
    if args.system == "pleiades":
        track_name = foggie_dir + "halo_008508/nref11f_refine200kpc_z4to2/halo_track"

    print("opening track: " + track_name)
    track = Table.read(track_name, format='ascii')
    track.sort('col1')

    ## default is do allll the snaps in the directory
    ## want to add flag for if just one
    run_dir = foggie_dir + 'halo_00' + str(halo) + '/' + run
    if halo == "8508":
        prefix = output_dir + 'plots_halo_008508/' + run + '/'
    else:
        prefix = output_dir + 'other_halo_plots/' + str(halo) + '/' + run + '/'
    if not (os.path.exists(prefix)):
        os.system("mkdir " + prefix)

    if outs == "all":
        print("looking for outputs in ", run_dir)
        outs = glob.glob(os.path.join(run_dir, '?D????/?D????'))
    else:
        print("outs = ", outs)
        new_outs = [glob.glob(os.path.join(run_dir, snap)) for snap in outs]
        print("new_outs = ", new_outs)
        new_new_outs = [snap[0] for snap in new_outs]
        outs = new_new_outs

    for snap in outs:
        # load the snapshot
        print('opening snapshot '+ snap)
        ds = yt.load(snap)

        # add all the things
        ds.add_particle_filter('stars')
        ds.add_particle_filter('dm')
        ds.add_field(('gas_density_in'), function=_gas_density_in, units="Msun/kpc**3", force_override=True)
        ds.add_field(('gas_density_out'), function=_gas_density_out, units="Msun/kpc**3", force_override=True)
        ds.add_field(('metal_density_in'), function=_metal_density_in, units="Msun/kpc**3", force_override=True)
        ds.add_field(('metal_density_out'), function=_metal_density_out, units="Msun/kpc**3", force_override=True)
        ds.add_field(('hot_gas_density'), function=_hot_gas_density, units="Msun/kpc**3", force_override=True)
        ds.add_field(('hot_gas_density_in'), function=_hot_gas_density_in, units="Msun/kpc**3", force_override=True)
        ds.add_field(('hot_gas_density_out'), function=_hot_gas_density_out, units="Msun/kpc**3", force_override=True)
        ds.add_field(('warm_gas_density'), function=_warm_gas_density, units="Msun/kpc**3", force_override=True)
        ds.add_field(('warm_gas_density_in'), function=_warm_gas_density_in, units="Msun/kpc**3", force_override=True)
        ds.add_field(('warm_gas_density_out'), function=_warm_gas_density_out, units="Msun/kpc**3", force_override=True)
        ds.add_field(('cool_gas_density'), function=_cool_gas_density, units="Msun/kpc**3", force_override=True)
        ds.add_field(('cool_gas_density_in'), function=_cool_gas_density_in, units="Msun/kpc**3", force_override=True)
        ds.add_field(('cool_gas_density_out'), function=_cool_gas_density_out, units="Msun/kpc**3", force_override=True)
        ds.add_field(('cold_gas_density'), function=_cold_gas_density, units="Msun/kpc**3", force_override=True)
        ds.add_field(('cold_gas_density_in'), function=_cold_gas_density_in, units="Msun/kpc**3", force_override=True)
        ds.add_field(('cold_gas_density_out'), function=_cold_gas_density_out, units="Msun/kpc**3", force_override=True)


        # create all the regions
        zsnap = ds.get_parameter('CosmologyCurrentRedshift')
        proper_box_size = get_proper_box_size(ds)

        refine_box, refine_box_center, refine_width_code = get_refine_box(ds, zsnap, track)
        refine_width = refine_width_code * proper_box_size

        # center is trying to be the center of the halo
        halo_center, halo_velocity = get_halo_center(ds, refine_box_center)

        ### OK, now want to set up some spheres of some sizes and get the stuff
        radii = refine_width_code*0.5*np.arange(0.9, 0.1, -0.1)  # 0.5 because radius
        small_sphere = ds.sphere(halo_center, 0.05*refine_width_code) # R=10ckpc/h
        big_sphere = ds.sphere(halo_center, 0.45*refine_width_code)

        for radius in radii:
            this_sphere = ds.sphere(halo_center, radius)
            if radius != np.max(radii):
                surface = ds.surface(big_sphere, 'radius', (radius, 'code_length'))
                nref_mode = stats.mode(surface[('index', 'grid_level')])
                gas_flux = surface.calculate_flux("velocity_x", "velocity_y", "velocity_z", "density")
                metal_flux = surface.calculate_flux("velocity_x", "velocity_y", "velocity_z", "metal_density")
                ## also want to filter based on radial velocity to get fluxes in and mass flux out
                gas_flux_in = surface.calculate_flux("velocity_x", "velocity_y", "velocity_z", "gas_density_in")
                metal_flux_in = surface.calculate_flux("velocity_x", "velocity_y", "velocity_z", "metal_density_in")
                gas_flux_out = surface.calculate_flux("velocity_x", "velocity_y", "velocity_z", "gas_density_out")
                metal_flux_out = surface.calculate_flux("velocity_x", "velocity_y", "velocity_z", "metal_density_out")


                ## aaand want to filter based on temperature
                hot_gas_flux = surface.calculate_flux("velocity_x", "velocity_y", "velocity_z", "hot_gas_density")
                hot_gas_flux_in = surface.calculate_flux("velocity_x", "velocity_y", "velocity_z", "hot_gas_density_in")
                hot_gas_flux_out = surface.calculate_flux("velocity_x", "velocity_y", "velocity_z", "hot_gas_density_out")
                warm_gas_flux = surface.calculate_flux("velocity_x", "velocity_y", "velocity_z", "warm_gas_density")
                warm_gas_flux_in = surface.calculate_flux("velocity_x", "velocity_y", "velocity_z", "warm_gas_density_in")
                warm_gas_flux_out = surface.calculate_flux("velocity_x", "velocity_y", "velocity_z", "warm_gas_density_out")
                cool_gas_flux = surface.calculate_flux("velocity_x", "velocity_y", "velocity_z", "cool_gas_density")
                cool_gas_flux_in = surface.calculate_flux("velocity_x", "velocity_y", "velocity_z", "cool_gas_density_in")
                cool_gas_flux_out = surface.calculate_flux("velocity_x", "velocity_y", "velocity_z", "cool_gas_density_out")
                cold_gas_flux = surface.calculate_flux("velocity_x", "velocity_y", "velocity_z", "cold_gas_density")
                cold_gas_flux_in = surface.calculate_flux("velocity_x", "velocity_y", "velocity_z", "cold_gas_density_in")
                cold_gas_flux_out = surface.calculate_flux("velocity_x", "velocity_y", "velocity_z", "cold_gas_density_out")

                # annuli
                big_annulus = big_sphere - this_sphere
                # note that refine_fracs is in decreasing order!
                this_annulus = last_sphere - this_sphere
                inside_ang_mom_stars_x = this_sphere['stars', 'particle_angular_momentum_x'].sum()
                inside_ang_mom_stars_y = this_sphere['stars', 'particle_angular_momentum_y'].sum()
                inside_ang_mom_stars_z = this_sphere['stars', 'particle_angular_momentum_z'].sum()
                inside_spec_ang_mom_stars_x = this_sphere['stars', 'particle_specific_angular_momentum_x'].mean()
                inside_spec_ang_mom_stars_y = this_sphere['stars', 'particle_specific_angular_momentum_y'].mean()
                inside_spec_ang_mom_stars_z = this_sphere['stars', 'particle_specific_angular_momentum_z'].mean()

                ## ok want angular momenta
                annular_ang_mom_gas_x = this_annulus[('gas', 'angular_momentum_x')].sum()
                annular_ang_mom_gas_y = this_annulus[('gas', 'angular_momentum_y')].sum()
                annular_ang_mom_gas_z = this_annulus[('gas', 'angular_momentum_z')].sum()
                annular_spec_ang_mom_gas_x = this_annulus[('gas', 'specific_angular_momentum_x')].mean()
                annular_spec_ang_mom_gas_y = this_annulus[('gas', 'specific_angular_momentum_y')].mean()
                annular_spec_ang_mom_gas_z = this_annulus[('gas', 'specific_angular_momentum_z')].mean()

                annular_ang_mom_dm_x = this_annulus[('dm', 'particle_angular_momentum_x')].sum()
                annular_ang_mom_dm_y = this_annulus[('dm', 'particle_angular_momentum_y')].sum()
                annular_ang_mom_dm_z = this_annulus[('dm', 'particle_angular_momentum_z')].sum()
                annular_spec_ang_mom_dm_x = this_annulus[('dm', 'particle_specific_angular_momentum_x')].mean()
                annular_spec_ang_mom_dm_y = this_annulus[('dm', 'particle_specific_angular_momentum_y')].mean()
                annular_spec_ang_mom_dm_z = this_annulus[('dm', 'particle_specific_angular_momentum_z')].mean()

                outside_ang_mom_gas_x = big_annulus[('gas', 'angular_momentum_x')].sum()
                outside_ang_mom_gas_y = big_annulus[('gas', 'angular_momentum_y')].sum()
                outside_ang_mom_gas_z = big_annulus[('gas', 'angular_momentum_z')].sum()
                outside_spec_ang_mom_gas_x = big_annulus[('gas', 'specific_angular_momentum_x')].mean()
                outside_spec_ang_mom_gas_y = big_annulus[('gas', 'specific_angular_momentum_y')].mean()
                outside_spec_ang_mom_gas_z = big_annulus[('gas', 'specific_angular_momentum_z')].mean()

                outside_ang_mom_dm_x = big_annulus[('dm', 'particle_angular_momentum_x')].sum()
                outside_ang_mom_dm_y = big_annulus[('dm', 'particle_angular_momentum_y')].sum()
                outside_ang_mom_dm_z = big_annulus[('dm', 'particle_angular_momentum_z')].sum()
                outside_spec_ang_mom_dm_x = big_annulus[('dm', 'particle_specific_angular_momentum_x')].mean()
                outside_spec_ang_mom_dm_y = big_annulus[('dm', 'particle_specific_angular_momentum_y')].mean()
                outside_spec_ang_mom_dm_z = big_annulus[('dm', 'particle_specific_angular_momentum_z')].mean()

                # let's add everything to the giant table!
                data.add_row([zsnap, radius, int(nref_mode[0][0]), gas_flux, metal_flux, \
                                gas_flux_in, gas_flux_out, metal_flux_in, metal_flux_out, \
                                cold_gas_flux, cold_gas_flux_in, cold_gas_flux_out, \
                                cool_gas_flux, cool_gas_flux_in, cool_gas_flux_out, \
                                warm_gas_flux, warm_gas_flux_in, warm_gas_flux_out, \
                                hot_gas_flux, hot_gas_flux_in, hot_gas_flux_out,
                                annular_ang_mom_gas_x, annular_ang_mom_gas_y,annular_ang_mom_gas_z, \
                                annular_spec_ang_mom_gas_x, annular_spec_ang_mom_gas_y,annular_spec_ang_mom_gas_z,\
                                annular_ang_mom_dm_x, annular_ang_mom_dm_y,annular_ang_mom_dm_z, \
                                annular_spec_ang_mom_dm_x, annular_spec_ang_mom_dm_y, annular_spec_ang_mom_dm_z, \
                                outside_ang_mom_gas_x, outside_ang_mom_gas_y, outside_ang_mom_gas_z,  \
                                outside_spec_ang_mom_gas_x, outside_spec_ang_mom_gas_y, outside_spec_ang_mom_gas_z, \
                                outside_ang_mom_dm_x, outside_ang_mom_dm_y,outside_ang_mom_dm_z,\
                                outside_spec_ang_mom_dm_x, outside_spec_ang_mom_dm_y, outside_spec_ang_mom_dm_z, \
                                inside_ang_mom_stars_x, inside_ang_mom_stars_y, inside_ang_mom_stars_z, \
                                inside_spec_ang_mom_stars_x, inside_spec_ang_mom_stars_y, inside_spec_ang_mom_stars_z])

                # this apparently makes fluxes work in a loop?
                surface._vertices = None
            last_sphere = this_sphere

    # perhaps we should save the table?
    tablename = run + '_angular_momenta_and_fluxes.dat'
    ascii.write(data, tablename, format='fixed_width')

    return "whooooo angular momentum wheeeeeeee"


#-----------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    if args.system == "oak":
        foggie_dir = "/astro/simulations/FOGGIE/"
        output_path = "/Users/molly/Dropbox/foggie-collab/"
    elif args.system == "dhumuha" or args.system == "palmetto":
        foggie_dir = "/Users/molly/foggie/"
        output_path = "/Users/molly/Dropbox/foggie-collab/"
    elif args.system == "harddrive":
        foggie_dir = "/Volumes/foggie/"
        output_path = "/Users/molly/Dropbox/foggie-collab/"
    elif args.system == "nmearl":
        foggie_dir = "/Users/nearl/data/"
        output_path = "/Users/nearl/Desktop/"
    elif args.system == "pleiades":
        foggie_dir = "/nobackup/mpeeples/"
        output_path = "/nobackup/mpeeples/"

    if args.run == "natural":
        run_loc = "nref11n/natural/"
        trackname = "halo_track"
        haloname = "halo008508_nref11n"
    elif args.run == "nref10f":
        run_loc = "nref11n/nref11n_nref10f_refine200kpc/"
        trackname = "halo_008508/nref11n/nref11n_nref10f_refine200kpc/halo_track"
        haloname = "halo008508_nref11n_nref10f"
    elif args.run == "nref11n_selfshield":
        run_loc = "nref11n/nref11n_selfshield/"
        trackname = "halo_008508/nref11n/nref11n_selfshield/halo_track"
        haloname = "halo008508_nref11n_selfshield"
        if args.system == "pleiades":
            trackname = "halo_008508/nref11f_refine200kpc/halo_track"
            run_loc = "nref11n_selfshield/"
    elif args.run == "nref10n_nref8f_selfshield":
        run_loc = "nref10n/nref10n_nref8f_selfshield/"
        trackname = "halo_008508/nref10n/nref10n_nref8f_selfshield/halo_track"
        haloname = "halo008508_nref10n_nref8f_selfshield"
        if args.system == "pleiades":
            trackname = "halo_008508/nref10n_nref8f_selfshield/halo_track"
            run_loc = "nref10n_nref8f_selfshield/"
    elif args.run == "nref11f":
        run_loc = "nref11n/nref11f_refine200kpc/"
        trackname =  "halo_008508/nref11n/nref11f_refine200kpc/halo_track"
        haloname = "halo008508_nref11f"
        if args.system == "pleiades":
            trackname = "halo_008508/nref11f_refine200kpc/halo_track"
            run_loc = "nref11f_refine200kpc_z4to2/"

    print("---->>>>> for now I am assuming you are using the Tempest halo even if you passed in something different")

    if args.output == "all":
        message = calc_ang_mom_and_fluxes(args.halo, foggie_dir, output_path, run_loc, outs=args.output)
    else:
        message = calc_ang_mom_and_fluxes(args.halo, foggie_dir, output_path, run_loc, outs=[args.output + "/" + args.output])

    sys.exit(message)
