from __future__ import print_function

import numpy as np
from scipy import stats

import yt
from yt import derived_field

import argparse
import os
import glob
import sys

from astropy.table import Table
from astropy.io import ascii
import h5py.highlevel

from foggie.utils.consistency import *
from foggie.utils.get_refine_box import get_refine_box
from foggie.utils.get_halo_center import get_halo_center
from foggie.utils.get_proper_box_size import get_proper_box_size
from foggie.utils.get_run_loc_etc import get_run_loc_etc

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

####################    FLUX CALCULATION   ######################################
def calc_ang_mom_and_fluxes(halo, foggie_dir, run, **kwargs):
    outs = kwargs.get("outs", "all")
    trackname = kwargs.get("trackname", "halo_track")

    ### set up the table of all the stuff we want
    data = Table(names=('redshift', 'radius', 'nref_mode', \
                        'net_mass_flux', 'net_metal_flux', \
                        'mass_flux_in', 'mass_flux_out', \
                        'metal_flux_in', 'metal_flux_out', \
                        'net_kinetic_energy_flux','net_thermal_energy_flux','net_entropy_flux',\
                        'kinetic_energy_flux_in','kinetic_energy_flux_out',\
                        'thermal_energy_flux_in','thermal_energy_flux_out',\
                        'entropy_flux_in','entropy_flux_out',\
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
                  dtype=('f8', 'f8', 'i8','f8', 'f8', 'f8',
                         'f8', 'f8', 'f8','f8', 'f8', 'f8',
                         'f8', 'f8', 'f8', 'f8', 'f8', 'f8',
                         'f8', 'f8', 'f8', 'f8', 'f8', 'f8',
                         'f8', 'f8', 'f8', 'f8', 'f8', 'f8',
                         'f8', 'f8', 'f8', 'f8', 'f8', 'f8',
                         'f8', 'f8', 'f8', 'f8', 'f8', 'f8',
                         'f8', 'f8', 'f8', 'f8', 'f8', 'f8',
                         'f8', 'f8', 'f8', 'f8', 'f8', 'f8',
                         'f8', 'f8', 'f8', 'f8', 'f8', 'f8'
                        ))
    data2 = Table(names=('redshift', 'radius',
                        'net_mass_flux', 'net_metal_flux', \
                        'mass_flux_in', 'mass_flux_out', \
                        'metal_flux_in', 'metal_flux_out', \
                        'net_cold_mass_flux', 'cold_mass_flux_in', 'cold_mass_flux_out', \
                        'net_cool_mass_flux', 'cool_mass_flux_in', 'cool_mass_flux_out', \
                        'net_warm_mass_flux', 'warm_mass_flux_in', 'warm_mass_flux_out', \
                        'net_hot_mass_flux', 'hot_mass_flux_in', 'hot_mass_flux_out', \
                        'annular_ang_mom_gas_x', 'annular_ang_mom_gas_y','annular_ang_mom_gas_z', \
                        'annular_spec_ang_mom_gas_x', 'annular_spec_ang_mom_gas_y','annular_spec_ang_mom_gas_z',\
                        'outside_ang_mom_gas_x', 'outside_ang_mom_gas_y', 'outside_ang_mom_gas_z',  \
                        'outside_spec_ang_mom_gas_x', 'outside_spec_ang_mom_gas_y', 'outside_spec_ang_mom_gas_z'), \
                    dtype=('f8', 'f8', 'f8','f8', 'f8', 'f8',
                         'f8', 'f8', 'f8','f8', 'f8', 'f8',
                         'f8', 'f8', 'f8', 'f8', 'f8', 'f8',
                         'f8', 'f8', 'f8', 'f8', 'f8', 'f8',
                         'f8', 'f8', 'f8', 'f8', 'f8', 'f8',
                         'f8', 'f8'))

    print('foggie_dir = ',foggie_dir, 'run = ',run, 'trackname = ',trackname)
    track_name = foggie_dir + '/' + run + '/' + trackname
    if args.system == "pleiades":
        track_name = foggie_dir + "halo_008508/nref11n_nref10f_selfshield_z6/halo_track"
        output_dir = './'

    print("opening track: " + track_name)
    track = Table.read(track_name, format='ascii')
    track.sort('col1')

    ## default is do allll the snaps in the directory
    ## want to add flag for if just one
    run_dir = foggie_dir + run
    if args.system == "pleiades":
        track_name = foggie_dir + "halo_008508/nref11n_nref10f_selfshield_z6/halo_track"
        run_dir = foggie_dir  + run
        output_dir = './'
    print('run_dir = ', run_dir)

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

        # add the particle filters
        ds.add_particle_filter('stars')
        ds.add_particle_filter('dm')

        # create all the regions
        zsnap = ds.get_parameter('CosmologyCurrentRedshift')
        #proper_box_size = get_proper_box_size(ds)
        # another option than the function:
        proper_box_size = ds.quan(1.,'code_length').to('kpc')

        refine_box, refine_box_center, refine_width_code = get_refine_box(ds, zsnap, track)
        refine_width = refine_width_code * proper_box_size

        # center is trying to be the center of the halo
        halo_center, halo_velocity = get_halo_center(ds, refine_box_center)

        ### OK, now want to set up some spheres of some sizes and get the stuff
        radii = refine_width*0.5*np.arange(0.9, 0.1, -0.01)  # 0.5 because radius
        print('radii: ',radii)
        small_sphere = ds.sphere(halo_center, 0.05*refine_width_code) # R=10ckpc/h
        big_sphere = ds.sphere(halo_center, 0.45*refine_width_code)
        print('small sphere radius: ',small_sphere.radius.in_units('kpc'))

        # we want to subtract the bulk velocity from the radial velocities
        bulk_velocity = small_sphere.quantities["BulkVelocity"]()

        # find number of cells for the FRB
        # by default, it uses nref10 as the cell size for the frb
        # then create the 3D FRB for calculating the fluxes
        print('big sphere unique:', np.unique(big_sphere['dx'].in_units('kpc')))
        cell_size = np.unique(big_sphere['dx'].in_units('kpc'))[1] ###
        box_width = ds.quan(0.9*refine_width,'kpc')
        nbins = int(np.ceil(box_width/cell_size).value)
        print('there will be ',nbins**3,' bins')

        halo_center = ds.arr(halo_center,'code_length')
        xL,xR = halo_center[0]-box_width/2.,halo_center[0]+box_width/2.
        yL,yR = halo_center[1]-box_width/2.,halo_center[1]+box_width/2.
        zL,zR = halo_center[2]-box_width/2.,halo_center[2]+box_width/2.
        jnbins = complex(0,nbins)
        box = ds.r[xL:xR:jnbins,yL:yR:jnbins,zL:zR:jnbins]
        box.set_field_parameter("center",halo_center)
        box.set_field_parameter("bulk_velocity",bulk_velocity)

        print('setting up gas fields...')
        ### OK, now want to call the fields that we'll need for the fluxes
        ### otherwise, the code crashes when trying to select subsets of the data
        ## GAS FIELDS
        temperature = box['Temperature'].flatten()
        cell_mass = box['cell_mass'].to("Msun").flatten()
        metal_mass = box[('gas', 'metal_mass')].to("Msun").flatten()
        radius = box['radius'].to("kpc").flatten()
        radial_velocity = box['radial_velocity'].to('kpc/yr').flatten()
        cell_volume = box['cell_volume'].flatten()
        #grid_levels = box['index', 'grid_level']
        print('here ????')
        gas_ang_mom_x = box[('gas', 'angular_momentum_x')].flatten()
        gas_ang_mom_y = box[('gas', 'angular_momentum_y')].flatten()
        gas_ang_mom_z = box[('gas', 'angular_momentum_z')].flatten()
        print('i have gotten to here')
        gas_spec_ang_mom_x = box[('gas','specific_angular_momentum_x')].flatten()
        gas_spec_ang_mom_y = box[('gas','specific_angular_momentum_y')].flatten()
        gas_spec_ang_mom_z = box[('gas','specific_angular_momentum_z')].flatten()
        kinetic_energy = box['gas','kinetic_energy'].flatten()
        print('and now to here')
        kinetic_energy = (kinetic_energy*cell_volume/cell_mass).to('erg/g')
        thermal_energy = box['gas','thermal_energy'].flatten()
        entropy = box['entropy'].flatten()
        hden = box['H_nuclei_density'].flatten()

        print('setting up star fields...')
        ## STAR PARTICLE FIELDS
        star_ang_mom_x = big_sphere['stars', 'particle_angular_momentum_x'].flatten()
        star_ang_mom_y = big_sphere['stars', 'particle_angular_momentum_y'].flatten()
        star_ang_mom_z = big_sphere['stars', 'particle_angular_momentum_z'].flatten()
        star_spec_ang_mom_x = big_sphere['stars', 'particle_specific_angular_momentum_x'].flatten()
        star_spec_ang_mom_y = big_sphere['stars', 'particle_specific_angular_momentum_y'].flatten()
        star_spec_ang_mom_z = big_sphere['stars', 'particle_specific_angular_momentum_z'].flatten()
        star_distance = np.sqrt((big_sphere['stars','particle_position_x']-halo_center[0])**2. +
                        (big_sphere['stars','particle_position_y']-halo_center[1])**2. +
                        (big_sphere['stars','particle_position_z']-halo_center[2])**2.).to("kpc")


        print('setting up dark matter fields...')
        ## DM PARTICLE FIELDS
        dm_ang_mom_x = big_sphere['dm', 'particle_angular_momentum_x'].flatten()
        dm_ang_mom_y = big_sphere['dm', 'particle_angular_momentum_y'].flatten()
        dm_ang_mom_z = big_sphere['dm', 'particle_angular_momentum_z'].flatten()
        dm_spec_ang_mom_x = big_sphere['dm', 'particle_specific_angular_momentum_x'].flatten()
        dm_spec_ang_mom_y = big_sphere['dm', 'particle_specific_angular_momentum_y'].flatten()
        dm_spec_ang_mom_z = big_sphere['dm', 'particle_specific_angular_momentum_z'].flatten()
        dm_distance   = np.sqrt((big_sphere['dm','particle_position_x']-halo_center[0])**2. +
                        (big_sphere['dm','particle_position_y']-halo_center[1])**2. +
                        (big_sphere['dm','particle_position_z']-halo_center[2])**2.).to("kpc")


        for rad in radii:
            #this_sphere = ds.sphere(halo_center, rad)
            print('doing radius ', rad, '.......')
            if rad != np.max(radii):
                if rad == radii[-1]:
                    minrad,maxrad = ds.quan(1,'kpc'),rad
                else:
                    idI = np.where(radii == rad)[0]
                    maxrad,minrad = rad,radii[idI[0]+1]
                # some radius / geometry things
                dr = maxrad - minrad
                rad_here = (minrad+maxrad) / 2.

                # find the indices that I'm going to need
                idR = np.where((radius >= minrad) & (radius < maxrad))[0]
                idCd = np.where((radius >= minrad) & (radius < maxrad) & (temperature <= 1e4))[0]
                idCl = np.where((radius >= minrad) & (radius < maxrad) & (temperature >1e4) & (temperature <= 1e5))[0]
                idW =  np.where((radius >= minrad) & (radius < maxrad) & (temperature >1e5) & (temperature <= 1e6))[0]
                idH = np.where((radius >= minrad) & (radius < maxrad) & (temperature >= 1e6))
                idRdm = np.where((dm_distance >= minrad) & (dm_distance < maxrad))[0]
                big_annulusGAS = np.where(radius >= rad_here)[0]
                big_annulusDM  = np.where(dm_distance >= rad_here)[0]
                inside = np.where(star_distance < rad_here)[0]

                # most common refinement level
                #nref_mode = stats.mode(grid_levels[idR])
                print("min,max = ", minrad, maxrad)
                sp_out, sp_in = ds.sphere(halo_center, maxrad), ds.sphere(halo_center,minrad)
                shell = sp_out - sp_in
                print('nref mode = ', stats.mode(shell['index','grid_level']))
                nref_mode = stats.mode(shell['index','grid_level'])[0]
                # mass fluxes
                gas_flux = (np.sum(cell_mass[idR]*radial_velocity[idR])/dr).to("Msun/yr")
                metal_flux = (np.sum(metal_mass[idR]*radial_velocity[idR])/dr).to("Msun/yr")
                kinetic_energy_flux = (np.sum(kinetic_energy[idR]*radial_velocity[idR])/dr).to("erg/(g*yr)")
                thermal_energy_flux = (np.sum(thermal_energy[idR]*radial_velocity[idR])/dr).to("erg/(g*yr)")
                entropy_flux = (np.sum(entropy[idR]*radial_velocity[idR])/dr)

                ## also filter based off radial velocity
                idVin = np.where(radial_velocity[idR] <= 0. )[0]
                idVout = np.where(radial_velocity[idR] > 0.)[0]
                gas_flux_in = (np.sum(cell_mass[idR][idVin]*radial_velocity[idR][idVin])/dr).to("Msun/yr")
                gas_flux_out = (np.sum(cell_mass[idR][idVout]*radial_velocity[idR][idVout])/dr).to("Msun/yr")
                metal_flux_in = (np.sum(metal_mass[idR][idVin]*radial_velocity[idR][idVin])/dr).to("Msun/yr")
                metal_flux_out = (np.sum(metal_mass[idR][idVout]*radial_velocity[idR][idVout])/dr).to("Msun/yr")
                kinetic_energy_flux_in = (np.sum(kinetic_energy[idR][idVin]*radial_velocity[idR][idVin])/dr).to("erg/(g*yr)")
                kinetic_energy_flux_out = (np.sum(kinetic_energy[idR][idVout]*radial_velocity[idR][idVout])/dr).to("erg/(g*yr)")
                thermal_energy_flux_in = (np.sum(thermal_energy[idR][idVin]*radial_velocity[idR][idVin])/dr).to("erg/(g*yr)")
                thermal_energy_flux_out = (np.sum(thermal_energy[idR][idVout]*radial_velocity[idR][idVout])/dr).to("erg/(g*yr)")
                entropy_flux_in = (np.sum(entropy[idR][idVin]*radial_velocity[idR][idVin])/dr)
                entropy_flux_out = (np.sum(entropy[idR][idVout]*radial_velocity[idR][idVout])/dr)

                ## and filter on temperature! and velocity! woo!
                idVin = np.where(radial_velocity[idH] <= 0. )[0]
                idVout = np.where(radial_velocity[idH] > 0.)[0]
                hot_gas_flux = (np.sum(cell_mass[idH]*radial_velocity[idH])/dr).to("Msun/yr")
                hot_gas_flux_in  = (np.sum(cell_mass[idH][idVin]*radial_velocity[idH][idVin])/dr).to("Msun/yr")
                hot_gas_flux_out = (np.sum(cell_mass[idH][idVout]*radial_velocity[idH][idVout])/dr).to("Msun/yr")

                idVin = np.where(radial_velocity[idW] <= 0. )[0]
                idVout = np.where(radial_velocity[idW] > 0.)[0]
                warm_gas_flux = (np.sum(cell_mass[idW]*radial_velocity[idW])/dr).to("Msun/yr")
                warm_gas_flux_in  = (np.sum(cell_mass[idW][idVin]*radial_velocity[idW][idVin])/dr).to("Msun/yr")
                warm_gas_flux_out = (np.sum(cell_mass[idW][idVout]*radial_velocity[idW][idVout])/dr).to("Msun/yr")

                idVin = np.where(radial_velocity[idCl] <= 0. )[0]
                idVout = np.where(radial_velocity[idCl] > 0.)[0]
                cool_gas_flux = (np.sum(cell_mass[idCl]*radial_velocity[idCl])/dr).to("Msun/yr")
                cool_gas_flux_in  = (np.sum(cell_mass[idCl][idVin]*radial_velocity[idCl][idVin])/dr).to("Msun/yr")
                cool_gas_flux_out = (np.sum(cell_mass[idCl][idVout]*radial_velocity[idCl][idVout])/dr).to("Msun/yr")

                idVin = np.where(radial_velocity[idCd] <= 0. )[0]
                idVout = np.where(radial_velocity[idCd] > 0.)[0]
                cold_gas_flux = (np.sum(cell_mass[idCd]*radial_velocity[idCd])/dr).to("Msun/yr")
                cold_gas_flux_in  = (np.sum(cell_mass[idCd][idVin]*radial_velocity[idCd][idVin])/dr).to("Msun/yr")
                cold_gas_flux_out = (np.sum(cell_mass[idCd][idVout]*radial_velocity[idCd][idVout])/dr).to("Msun/yr")

                ## GAS angular momentum!
                annular_ang_mom_gas_x = np.sum(gas_ang_mom_x[idR])
                annular_ang_mom_gas_y = np.sum(gas_ang_mom_y[idR])
                annular_ang_mom_gas_z = np.sum(gas_ang_mom_z[idR])
                annular_spec_ang_mom_gas_x = np.mean(gas_spec_ang_mom_x[idR])
                annular_spec_ang_mom_gas_y = np.mean(gas_spec_ang_mom_y[idR])
                annular_spec_ang_mom_gas_z = np.mean(gas_spec_ang_mom_z[idR])

                outside_ang_mom_gas_x = np.sum(gas_ang_mom_x[big_annulusGAS])
                outside_ang_mom_gas_y = np.sum(gas_ang_mom_y[big_annulusGAS])
                outside_ang_mom_gas_z = np.sum(gas_ang_mom_z[big_annulusGAS])
                outside_spec_ang_mom_gas_x = np.mean(gas_spec_ang_mom_x[big_annulusGAS])
                outside_spec_ang_mom_gas_y = np.mean(gas_spec_ang_mom_y[big_annulusGAS])
                outside_spec_ang_mom_gas_z = np.mean(gas_spec_ang_mom_z[big_annulusGAS])

                ## PARTICLE angular momentum calculations
                inside_ang_mom_stars_x = np.sum(star_ang_mom_x[inside])
                inside_ang_mom_stars_y = np.sum(star_ang_mom_y[inside])
                inside_ang_mom_stars_z = np.sum(star_ang_mom_z[inside])
                inside_spec_ang_mom_stars_x = np.sum(star_spec_ang_mom_x[inside])
                inside_spec_ang_mom_stars_y = np.sum(star_spec_ang_mom_y[inside])
                inside_spec_ang_mom_stars_z = np.sum(star_spec_ang_mom_z[inside])

                annular_ang_mom_dm_x = np.sum(dm_ang_mom_x[idRdm])
                annular_ang_mom_dm_y = np.sum(dm_ang_mom_y[idRdm])
                annular_ang_mom_dm_z = np.sum(dm_ang_mom_z[idRdm])
                annular_spec_ang_mom_dm_x = np.mean(dm_spec_ang_mom_x[idRdm])
                annular_spec_ang_mom_dm_y = np.mean(dm_spec_ang_mom_y[idRdm])
                annular_spec_ang_mom_dm_z = np.mean(dm_spec_ang_mom_z[idRdm])

                outside_ang_mom_dm_x = np.sum(dm_ang_mom_x[big_annulusDM])
                outside_ang_mom_dm_y = np.sum(dm_ang_mom_y[big_annulusDM])
                outside_ang_mom_dm_z = np.sum(dm_ang_mom_z[big_annulusDM])
                outside_spec_ang_mom_dm_x = np.mean(dm_spec_ang_mom_x[big_annulusDM])
                outside_spec_ang_mom_dm_y = np.mean(dm_spec_ang_mom_y[big_annulusDM])
                outside_spec_ang_mom_dm_z = np.mean(dm_spec_ang_mom_z[big_annulusDM])

                data.add_row([zsnap, rad, nref_mode, gas_flux, metal_flux, \
                                gas_flux_in, gas_flux_out, metal_flux_in, metal_flux_out, \
                                kinetic_energy_flux, thermal_energy_flux, entropy_flux,\
                                kinetic_energy_flux_in,kinetic_energy_flux_out,\
                                thermal_energy_flux_in,thermal_energy_flux_out,\
                                entropy_flux_in,entropy_flux_out,\
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

                ####### REPEATING THE CALCUATION WITHOUT THE DENSEST ISM-LIKE GAS #########
                idR = np.where((radius >= minrad) & (radius < maxrad) & (hden < 0.1))[0]
                idCd = np.where((radius >= minrad) & (radius < maxrad) & (temperature <= 1e4) & (hden < 0.1))[0]
                idCl = np.where((radius >= minrad) & (radius < maxrad) & (temperature >1e4) & (temperature <= 1e5) & (hden < 0.1))[0]
                idW =  np.where((radius >= minrad) & (radius < maxrad) & (temperature >1e5) & (temperature <= 1e6) & (hden < 0.1))[0]
                idH = np.where((radius >= minrad) & (radius < maxrad) & (temperature >= 1e6) & (hden < 0.1))
                big_annulusGAS = np.where((radius >= rad_here) & (hden < 0.1))[0]

                # mass fluxes
                gas_flux = (np.sum(cell_mass[idR]*radial_velocity[idR])/dr).to("Msun/yr")
                metal_flux = (np.sum(metal_mass[idR]*radial_velocity[idR])/dr).to("Msun/yr")

                ## also filter based off radial velocity
                idVin = np.where(radial_velocity[idR] <= 0. )[0]
                idVout = np.where(radial_velocity[idR] > 0.)[0]
                gas_flux_in = (np.sum(cell_mass[idR][idVin]*radial_velocity[idR][idVin])/dr).to("Msun/yr")
                gas_flux_out = (np.sum(cell_mass[idR][idVout]*radial_velocity[idR][idVout])/dr).to("Msun/yr")
                metal_flux_in = (np.sum(metal_mass[idR][idVin]*radial_velocity[idR][idVin])/dr).to("Msun/yr")
                metal_flux_out = (np.sum(metal_mass[idR][idVout]*radial_velocity[idR][idVout])/dr).to("Msun/yr")

                ## and filter on temperature! and velocity! woo!
                idVin = np.where(radial_velocity[idH] <= 0. )[0]
                idVout = np.where(radial_velocity[idH] > 0.)[0]
                hot_gas_flux = (np.sum(cell_mass[idH]*radial_velocity[idH])/dr).to("Msun/yr")
                hot_gas_flux_in  = (np.sum(cell_mass[idH][idVin]*radial_velocity[idH][idVin])/dr).to("Msun/yr")
                hot_gas_flux_out = (np.sum(cell_mass[idH][idVout]*radial_velocity[idH][idVout])/dr).to("Msun/yr")

                idVin = np.where(radial_velocity[idW] <= 0. )[0]
                idVout = np.where(radial_velocity[idW] > 0.)[0]
                warm_gas_flux = (np.sum(cell_mass[idW]*radial_velocity[idW])/dr).to("Msun/yr")
                warm_gas_flux_in  = (np.sum(cell_mass[idW][idVin]*radial_velocity[idW][idVin])/dr).to("Msun/yr")
                warm_gas_flux_out = (np.sum(cell_mass[idW][idVout]*radial_velocity[idW][idVout])/dr).to("Msun/yr")

                idVin = np.where(radial_velocity[idCl] <= 0. )[0]
                idVout = np.where(radial_velocity[idCl] > 0.)[0]
                cool_gas_flux = (np.sum(cell_mass[idCl]*radial_velocity[idCl])/dr).to("Msun/yr")
                cool_gas_flux_in  = (np.sum(cell_mass[idCl][idVin]*radial_velocity[idCl][idVin])/dr).to("Msun/yr")
                cool_gas_flux_out = (np.sum(cell_mass[idCl][idVout]*radial_velocity[idCl][idVout])/dr).to("Msun/yr")

                idVin = np.where(radial_velocity[idCd] <= 0. )[0]
                idVout = np.where(radial_velocity[idCd] > 0.)[0]
                cold_gas_flux = (np.sum(cell_mass[idCd]*radial_velocity[idCd])/dr).to("Msun/yr")
                cold_gas_flux_in  = (np.sum(cell_mass[idCd][idVin]*radial_velocity[idCd][idVin])/dr).to("Msun/yr")
                cold_gas_flux_out = (np.sum(cell_mass[idCd][idVout]*radial_velocity[idCd][idVout])/dr).to("Msun/yr")

                ## GAS angular momentum!
                annular_ang_mom_gas_x = np.sum(gas_ang_mom_x[idR])
                annular_ang_mom_gas_y = np.sum(gas_ang_mom_y[idR])
                annular_ang_mom_gas_z = np.sum(gas_ang_mom_z[idR])
                annular_spec_ang_mom_gas_x = np.mean(gas_spec_ang_mom_x[idR])
                annular_spec_ang_mom_gas_y = np.mean(gas_spec_ang_mom_y[idR])
                annular_spec_ang_mom_gas_z = np.mean(gas_spec_ang_mom_z[idR])

                outside_ang_mom_gas_x = np.sum(gas_ang_mom_x[big_annulusGAS])
                outside_ang_mom_gas_y = np.sum(gas_ang_mom_y[big_annulusGAS])
                outside_ang_mom_gas_z = np.sum(gas_ang_mom_z[big_annulusGAS])
                outside_spec_ang_mom_gas_x = np.mean(gas_spec_ang_mom_x[big_annulusGAS])
                outside_spec_ang_mom_gas_y = np.mean(gas_spec_ang_mom_y[big_annulusGAS])
                outside_spec_ang_mom_gas_z = np.mean(gas_spec_ang_mom_z[big_annulusGAS])


                data2.add_row([zsnap, rad,gas_flux, metal_flux, \
                                gas_flux_in, gas_flux_out, metal_flux_in, metal_flux_out, \
                                cold_gas_flux, cold_gas_flux_in, cold_gas_flux_out, \
                                cool_gas_flux, cool_gas_flux_in, cool_gas_flux_out, \
                                warm_gas_flux, warm_gas_flux_in, warm_gas_flux_out, \
                                hot_gas_flux, hot_gas_flux_in, hot_gas_flux_out,
                                annular_ang_mom_gas_x, annular_ang_mom_gas_y,annular_ang_mom_gas_z, \
                                annular_spec_ang_mom_gas_x, annular_spec_ang_mom_gas_y,annular_spec_ang_mom_gas_z,\
                                outside_ang_mom_gas_x, outside_ang_mom_gas_y, outside_ang_mom_gas_z,  \
                                outside_spec_ang_mom_gas_x, outside_spec_ang_mom_gas_y, outside_spec_ang_mom_gas_z])



    data = set_table_units(data)
    data2 = set_table_units(data2)
    tablename = run_dir + '/' + args.run + '_' + args.output + '_angular_momenta_and_fluxes.hdf5'
    print('writing to ',tablename)
    data.write(tablename,path='all_data',serialize_meta=True,overwrite=True)
    data2.write(tablename,path='noISM_data',serialize_meta=True,overwrite=False,append=True)

    return "whooooo angular momentum wheeeeeeee"

def set_table_units(table):
    table_units = {'redshift':None,'radius':'kpc','nref_mode':None,'net_mass_flux':'Msun/yr', \
             'net_metal_flux':'Msun/yr', 'mass_flux_in'  :'Msun/yr','mass_flux_out':'Msun/yr', \
             'metal_flux_in' :'Msun/yr', 'metal_flux_out':'Msun/yr','net_entropy_flux':'cm**2*keV/yr',\
             'net_kinetic_energy_flux':'erg/g*yr','net_thermal_energy_flux':'erg/g*yr',\
             'kinetic_energy_flux_in':'erg/g*yr','kinetic_energy_flux_out':'erg/g*yr',\
             'thermal_energy_flux_in':'erg/g*yr','thermal_energy_flux_out':'erg/g*yr',\
             'entropy_flux_in':'cm**2*keV/yr','entropy_flux_out':'cm**2*keV/yr',\
             'net_cold_mass_flux':'Msun/yr', 'cold_mass_flux_in':'Msun/yr', 'cold_mass_flux_out':'Msun/yr', \
             'net_cool_mass_flux':'Msun/yr', 'cool_mass_flux_in':'Msun/yr', 'cool_mass_flux_out':'Msun/yr', \
             'net_warm_mass_flux':'Msun/yr', 'warm_mass_flux_in':'Msun/yr', 'warm_mass_flux_out':'Msun/yr', \
             'net_hot_mass_flux' :'Msun/yr', 'hot_mass_flux_in' :'Msun/yr', 'hot_mass_flux_out' :'Msun/yr', \
             'annular_ang_mom_gas_x'     :'cm**2*g/s', 'annular_ang_mom_gas_y'     :'cm**2*g/s','annular_ang_mom_gas_z'     :'cm**2*g/s', \
             'annular_spec_ang_mom_gas_x':'cm**2/s',   'annular_spec_ang_mom_gas_y':'cm**2/s',  'annular_spec_ang_mom_gas_z':'cm**2/s',\
             'annular_ang_mom_dm_x'      :'cm**2*g/s', 'annular_ang_mom_dm_y'      :'cm**2*g/s','annular_ang_mom_dm_z'      :'cm**2*g/s', \
             'annular_spec_ang_mom_dm_x' :'cm**2/s',   'annular_spec_ang_mom_dm_y' :'cm**2/s',  'annular_spec_ang_mom_dm_z' :'cm**2/s', \
             'outside_ang_mom_gas_x'     :'cm**2*g/s', 'outside_ang_mom_gas_y'     :'cm**2*g/s','outside_ang_mom_gas_z'     :'cm**2*g/s',  \
             'outside_spec_ang_mom_gas_x':'cm**2/s',   'outside_spec_ang_mom_gas_y':'cm**2/s',  'outside_spec_ang_mom_gas_z':'cm**2/s', \
             'outside_ang_mom_dm_x'      :'cm**2*g/s', 'outside_ang_mom_dm_y'      :'cm**2*g/s','outside_ang_mom_dm_z'      :'cm**2*g/s',\
             'outside_spec_ang_mom_dm_x' :'cm**2/s',   'outside_spec_ang_mom_dm_y' :'cm**2/s',  'outside_spec_ang_mom_dm_z' :'cm**2/s', \
             'inside_ang_mom_stars_x'    :'cm**2*g/s', 'inside_ang_mom_stars_y'    :'cm**2*g/s','inside_ang_mom_stars_z'    :'cm**2*g/s', \
             'inside_spec_ang_mom_stars_x':'cm**2/s',  'inside_spec_ang_mom_stars_y':'cm**2/s', 'inside_spec_ang_mom_stars_z':'cm**2/s'}
    for key in table.keys():
        table[key].unit = table_units[key]
    return table


#-----------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    foggie_dir, output_dir, run_loc, trackname, haloname, spectra_dir = get_run_loc_etc(args)

    print("---->>>>> for now I am assuming you are using the Tempest halo even if you passed in something different")

    message = calc_ang_mom_and_fluxes(args.halo, foggie_dir, run_loc, outs=[args.output + "/" + args.output])

    sys.exit(message)
