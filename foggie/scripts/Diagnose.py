#!/usr/bin/env python
# coding: utf-8

import yt, numpy as np, os, matplotlib.pyplot as plt 
from foggie.utils.consistency import density_color_map, metal_color_map, o6_color_map, o6_min, o6_max, mg2_color_map, mg2_min, mg2_max 
from foggie.utils.foggie_load import *
from yt_astro_analysis.halo_analysis import HaloCatalog, add_quantity
from foggie.utils.halo_quantity_callbacks import * 
import os, argparse
from astropy.table import Table 
import trident 

def diagnosis_plots(snapname, halo_id, track_type, width): 

    print('Halo ID = ', halo_id) 

    if ('standard' in track_type): 
        TRACKNAME = '/nobackupnfs1/jtumlins/foggie/foggie/halo_tracks/'+halo_id+'/nref11n_selfshield_15/halo_track_200kpc_nref9' 
    if ('numerical' in track_type): 
        TRACKNAME = '/nobackupnfs1/jtumlins/foggie/foggie/halo_tracks/'+halo_id+'/root_tracks/root_track_H2numerical_z2' 
    if ('mechfix' in track_type): 
        TRACKNAME = '/nobackupnfs1/jtumlins/foggie/foggie/halo_tracks/'+halo_id+'/root_tracks/root_track_H2mechfix_z4' 
    if ('mechanical' in track_type): 
        TRACKNAME = '/nobackupnfs1/jtumlins/foggie/foggie/halo_tracks/'+halo_id+'/root_tracks/root_track_H2mechanical_z4' 
    if ('ff' in track_type): 
        TRACKNAME ='/u/jtumlins/nobackup/foggie/foggie/halo_tracks/'+halo_id+'/root_tracks/root_track_H2mechanical_ff_z1' 
        

    print('We have selected track: ') 
    print(TRACKNAME) 


    root_filename = '/nobackupnfs1/jtumlins/foggie/foggie/halo_tracks/'+halo_id+'/root_tracks/halo_'+halo_id+'_root_index.txt' 
    ds, region = foggie_load(snapname, TRACKNAME, halo_c_v_name=root_filename) 
    ad = ds.all_data()

    trident.add_ion_fields(ds, ions=['H I','C II', 'C III', 'C IV', 'O I', 'O II', 'O III', 'O IV', 'O V', 'O VI', 'O VII', 'O VIII', 'Mg II'])


    halo_center_code = ds.halo_center_code 
    print('halo_center_code = ', halo_center_code) 
    region = ds.r[ (halo_center_code.value[0]-0.02):(halo_center_code.value[0]+0.02), 
                       (halo_center_code.value[1]-0.02):(halo_center_code.value[1]+0.02), 
                       (halo_center_code.value[2]-0.02):(halo_center_code.value[2]+0.02)]

    run_name = (os.getcwd()).split("/")[-1]

    for axis in ['x', 'y', 'z']: 
        p = yt.ProjectionPlot(ds, axis, 'density', center=halo_center_code, data_source=region, width=(width, 'kpc'), weight_field='density') 
        p.set_cmap('density', density_color_map)
        p.annotate_timestamp(redshift=True)
        p.set_zlim('density', 1e-31, 1e-21) 
        p.annotate_title(run_name + '  ' + snapname + ',   z = ' + str(ds.current_redshift) ) 
        p.save() 

        p = yt.ProjectionPlot(ds, axis, ('gas', 'vel_mag_corrected'), weight_field='density', data_source=region, center=halo_center_code, width=(width, 'kpc'))
        p.annotate_timestamp(redshift=True)
        p.set_zlim(('gas', 'vel_mag_corrected'), 0,500)
        p.save()

        p = yt.ProjectionPlot(ds, axis, ('gas', 'vel_mag_corrected'), method='max', data_source=region, center=halo_center_code, width=(width, 'kpc'))
        p.annotate_timestamp(redshift=True)
        p.set_zlim(('gas', 'vel_mag_corrected'), 0, 3000)
        p.save()

        p = yt.ProjectionPlot(ds, axis, 'metallicity', center=halo_center_code, data_source=region, width=(width, 'kpc'), weight_field='density') 
        p.set_cmap('metallicity',metal_color_map)
        p.set_zlim('metallicity', 1e-8, 0.01) 
        p.annotate_timestamp(redshift=True)
        p.annotate_title(run_name + '  ' + snapname + ',   z = ' + str(ds.current_redshift) ) 
        p.save() 

        p = yt.ProjectionPlot(ds, axis, 'temperature', center=halo_center_code, data_source=region, width=(width, 'kpc'), weight_field='density') 
        p.set_cmap('temperature',temperature_color_map)
        p.set_zlim('temperature', 100, 1e6) 
        p.annotate_timestamp(redshift=True)
        p.annotate_title(run_name + '  ' + snapname + ',   z = ' + str(ds.current_redshift) ) 
        p.save() 
    
        if (ds.parameters['MultiSpecies'] == 2):
            p = yt.ProjectionPlot(ds, axis, ('gas', 'H2_density'), center=halo_center_code, data_source=region, width=(width, 'kpc')) 
            p.annotate_timestamp(redshift=True)
            p.annotate_title(run_name + '  ' + snapname + ',   z = ' + str(ds.current_redshift) ) 
            p.save() 
    
        p = yt.ProjectionPlot(ds, axis, ('gas', 'H_p0_number_density'), center=halo_center_code, data_source=region, width=(width, 'kpc')) 
        p.annotate_timestamp(redshift=True)
        p.set_zlim(('gas', 'H_p0_number_density'), 1e14,1e21) 
        p.annotate_title(run_name + '  ' + snapname + ',   z = ' + str(ds.current_redshift) ) 
        p.save() 

        p = yt.ProjectionPlot(ds, axis, 'O_p5_number_density', center = halo_center_code, width=(width, 'kpc'))
        p.set_cmap('O_p5_number_density', o6_color_map)
        p.set_zlim('O_p5_number_density', o6_min, o6_max)
        p.save()

        p = yt.ProjectionPlot(ds, axis, 'Mg_p1_number_density', center = ds.halo_center_code, width=(width, 'kpc'))
        p.set_cmap('Mg_p1_number_density', mg2_color_map)
        p.set_zlim('Mg_p1_number_density', mg2_min, mg2_max)
        p.set_background_color('Mg_p1_number_density', '#0A0780') 
        p.save()


    metallicity = region[('gas', 'metallicity')]
    density = region[('gas', 'density')]
    Metal_Density = region['Metal_Density'].in_units('g/cm**3') 
    Total_Density = region['Density'].in_units('g/cm**3') 
    cell_mass = region['cell_volume'].in_units('pc**3') * region['density'].in_units('Msun/pc**3') 
    HI_density = region[('enzo', 'HI_Density')].in_units('g/cm**3')  
    if (ds.parameters['MultiSpecies'] == 2):
        H2_fraction = region[('gas', 'H2_fraction')] 
        H2_density = region[('gas', 'H2_density')] 
    number_density = region[('gas', 'number_density')] 
    temperature = region[('gas', 'temperature')] 
    cooling_time = region[('gas', 'cooling_time')].in_units('yr') 
    
    print(snapname) 
    prefix = 'DD'+snapname.split('DD')[1]
    print(prefix) 

    plt.figure() 
    plt.scatter(np.log10(density), np.log10(metallicity), s = 0.1 )
    plt.ylim(-4,1.5) 
    plt.xlim(-30,-20) 
    plt.xlabel('log Density') 
    plt.ylabel('log Metallicity') 
    plt.title(run_name + '  ' + snapname + ',   z = ' + str(ds.current_redshift) ) 
    plt.savefig(prefix+'_metallicity_density.png') 
    
    if (ds.parameters['MultiSpecies'] == 2):
        plt.figure() 
        plt.scatter(np.log10(cell_mass), np.log10(H2_fraction), s=0.1) 
        plt.xlabel('log Cell Mass') 
        plt.ylabel('f_H2') 
        plt.xlim(0, 9) 
        plt.ylim(-6, 0.2) 
        plt.title(run_name + '  ' + snapname + ',   z = ' + str(ds.current_redshift) ) 
        plt.savefig(prefix +     '_fH2_cellmass.png') 

    plt.figure() 
    plt.scatter(np.log10(number_density), np.log10(H2_fraction), s=0.1, label='all cells', color='blue') 
    plt.scatter(np.log10(number_density[metallicity > 1e-6]), np.log10(H2_fraction[metallicity > 1e-6]), s=0.1, label='cells with metals', color='green') 
    plt.xlabel('log Number Density') 
    plt.ylabel('log f_H2') 
    plt.xlim(-2, 13) 
    plt.ylim(-5, 0.5) 
    plt.title(run_name + '  ' + snapname + ',   z = ' + str(ds.current_redshift) ) 
    plt.legend() 
    plt.savefig(prefix +     '_fH2_number_density.png') 

    plt.figure() 
    plt.scatter(np.log10(number_density), np.log10(temperature), s=0.1, label='all cells', color='blue') 
    plt.scatter(np.log10(number_density[metallicity > 1e-6]), np.log10(temperature[metallicity > 1e-6]), s=0.1, label='cells with metals', color='green') 
    plt.xlabel('log Number Density') 
    plt.ylabel('log Temperarture') 
    plt.xlim(0, 5) 
    plt.ylim(1, 6) 
    plt.title(run_name + '  ' + snapname + ',   z = ' + str(ds.current_redshift) ) 
    plt.legend() 
    plt.savefig(prefix +     '_temp_number_density_metalcode.png') 

    plt.figure() 
    plt.scatter(np.log10(number_density), np.log10(temperature), s=0.1, label='all cells', color='blue') 
    plt.scatter(np.log10(number_density[H2_fraction > 0.01]), np.log10(temperature[H2_fraction > 0.01]), s=0.1, label='cells with fH2 > 0.01', color='green') 
    plt.xlabel('log Number Density') 
    plt.ylabel('log Temperarture') 
    plt.xlim(0, 5) 
    plt.ylim(1, 6) 
    plt.title(run_name + '  ' + snapname + ',   z = ' + str(ds.current_redshift) ) 
    plt.legend() 
    plt.savefig(prefix +     '_temp_number_density_H2code.png') 

    plt.figure() 
    plt.scatter(np.log10(number_density), np.log10(cooling_time), s=0.1, label='all cells', color='blue') 
    plt.scatter(np.log10(number_density[metallicity > 1e-6]), np.log10(cooling_time[metallicity > 1e-6]), s=0.1, label='cells with metals', color='green') 
    plt.xlabel('log Number Density') 
    plt.ylabel('log Cooling Time') 
    plt.xlim(0, 5) 
    plt.ylim(2, 8) 
    plt.title(run_name + '  ' + snapname + ',   z = ' + str(ds.current_redshift) ) 
    plt.legend() 
    plt.savefig(prefix +     '_tcool_number_density_metalcode.png') 

    plt.figure() 
    plt.scatter(np.log10(cell_mass), np.log10(HI_density), s=0.1) 
    plt.xlabel('log Cell Mass') 
    plt.ylabel('log HI density') 
    plt.xlim(0, 9) 
    plt.ylim(-30,-20) 
    plt.title(run_name + '  ' + snapname + ',   z = ' + str(ds.current_redshift) ) 
    plt.savefig(prefix +     '_HIdensity_cellmass.png') 
    
    plt.figure() 
    plt.scatter(np.log10(HI_density), np.log10(H2_density), s=0.1) 
    plt.xlabel('log HI density') 
    plt.ylabel('log H2 density') 
    plt.xlim(-30,-20) 
    plt.ylim(-30,-20) 
    plt.title(run_name + '  ' + snapname + ',   z = ' + str(ds.current_redshift) ) 
    plt.savefig(prefix +     '_H2density_HIdensity.png') 
    
    plt.figure() 
    plt.scatter(np.log10(cell_mass), np.log10(H2_density), s=0.1) 
    plt.xlabel('log Cell Mass') 
    plt.ylabel('log H2 density') 
    plt.xlim(0, 9) 
    plt.ylim(-30,-20) 
    plt.title(run_name + '  ' + snapname + ',   z = ' + str(ds.current_redshift) ) 
    plt.savefig(prefix +     '_H2density_cellmass.png') 

    plt.figure() 
    plt.scatter(np.log10(Total_Density), np.log10(Metal_Density), s=0.1) 
    plt.plot([-35,-15],[-35,-15]) 
    plt.xlabel('log Total_Density') 
    plt.ylabel('log Metal Density') 
    plt.xlim(-35,-15) 
    plt.ylim(-35,-15) 
    plt.title(run_name + '  ' + snapname + ',   z = ' + str(ds.current_redshift) ) 
    plt.savefig(prefix +     '_Metal_Density_Density.png') 

    
    star_particle_mass = region[('stars', 'particle_mass')].in_units('Msun') 
    cell_size_in_pc = ((region['cell_volume']).in_units('pc**3'))**(1./3.) 
    
    plt.figure() 
    plt.scatter(cell_size_in_pc, np.log10(metallicity)) 
    plt.xlim(0, 300) 
    plt.ylim(-4, 1.5) 
    plt.xlabel('Cell size [pc]') 
    plt.ylabel('log Metallicity') 
    plt.title(run_name + '  ' + snapname + ',   z = ' + str(ds.current_redshift) ) 
    plt.savefig(prefix +     '_metallicity_cell_size.png') 
    
    star_particle_mass = region[('stars', 'particle_mass')].in_units('Msun') 
    star_particle_time = region[('stars', 'creation_time')].in_units('yr')
    star_particle_z = region[('stars', 'metallicity_fraction')] 
    all_star_particle_mass = ad[('stars', 'particle_mass')].in_units('Msun')
    all_star_particle_time = ad[('stars', 'creation_time')].in_units('yr')
    all_star_particle_z = ad[('stars', 'metallicity_fraction')] 
    if (ds.parameters['StarParticleCreation'] == 1):  minimum_mass = ds.parameters['StarMakerMinimumMass'] 
    if (ds.parameters['StarParticleCreation'] == 2048):  minimum_mass = ds.parameters['H2StarMakerMinimumMass'] 
    
    plt.figure() 
    plt.scatter(all_star_particle_time, np.log10(all_star_particle_mass), s=0.1, color='orange') 
    plt.scatter(star_particle_time, np.log10(star_particle_mass), s=0.1, color='blue') 
    plt.plot([0, 3e9], [np.log10(minimum_mass), np.log10(minimum_mass)], linestyle='dashed') 
    plt.xlabel('Star Particle Creation Time') 
    plt.ylabel('Stellar Mass [Msun]') 
    plt.xlim(0, 3e9) 
    plt.ylim(-0.5, 7.5) 
    plt.title(run_name + '  ' + snapname + ',   z = ' + str(ds.current_redshift) ) 
    plt.savefig(prefix +     '_starmass_startime.png') 

    plt.figure() 
    plt.scatter(np.log10(all_star_particle_z), np.log10(all_star_particle_mass), s=0.1, color='orange') 
    plt.scatter(np.log10(star_particle_z), np.log10(star_particle_mass), s=0.1, color='blue') 
    plt.plot([-10.5, 1.5], [np.log10(minimum_mass), np.log10(minimum_mass)], linestyle='dashed') 
    plt.xlabel('log Stellar Metallicity') 
    plt.ylabel('Stellar Mass [Msun]') 
    plt.xlim(-10.5, 1.5) 
    plt.ylim(-0.5, 7.5) 
    plt.title(run_name + '  ' + snapname + ',   z = ' + str(ds.current_redshift) ) 
    plt.savefig(prefix +     '_starmass_starz.png') 

    plt.figure() 
    plt.scatter(all_star_particle_time, np.log10(all_star_particle_z), s=0.1, color='orange') 
    plt.scatter(star_particle_time, np.log10(star_particle_z), s=0.1, color='blue') 
    plt.xlabel('Star Particle Creation Time') 
    plt.ylabel('log Stellar Metallicity') 
    plt.xlim(0, 3e9) 
    plt.ylim(-10.5, 1.5) 
    plt.title(run_name + '  ' + snapname + ',   z = ' + str(ds.current_redshift) ) 
    plt.savefig(prefix +     '_starz_startime.png') 


parser = argparse.ArgumentParser()
parser.add_argument('--snap_number', type=int, required=True)
parser.add_argument('--halo', type=int, required=True)
parser.add_argument('--track', type=str, required=True)
parser.add_argument('--path', type=str, required=False) 
parser.add_argument('--width', type=int, required=False, default=200) 
args = parser.parse_args()
if (args.snap_number < 10000): snap_string='DD'+str(args.snap_number)
if (args.snap_number < 1000): snap_string='DD0'+str(args.snap_number)
if (args.snap_number < 100): snap_string='DD00'+str(args.snap_number)
if (args.snap_number < 10): snap_string='DD000'+str(args.snap_number)
print('Hello your snap_number is:', args.snap_number, snap_string)
print('       your track type is:', args.track) 
print('Your path is: ', args.path) 
print('Your width is: ', args.width) 

ds_name = args.path+'/'+snap_string+'/'+snap_string

halo_id = '00'+str(args.halo) 

diagnosis_plots(ds_name, halo_id, args.track, args.width) 


