'''
    Filename: diagnosis_plots.py
    Author: JT
    Created: 11-62-25
    Last modified: 10-126-25 by JT
    This file works with fogghorn_analysis.py to make useful diagnostic plots for FOGGIE simulations.
    If you add a new function to this script, then please also add the function name to the dictionary in fogghorn_analysis.py.

    NOTE: unlike the other FOGGHORN plotting scripts, this one generates multiple plots per function call, so the function
    here is called 'phase_shade' but it makes many different shade maps. This prevents the FOGGHORN script from checking for 
    existing output files and skipping them, so be careful when using this script to avoid overwriting existing plots.
    This is a temporary solution until we can refactor the shade map code to be more modular, but was the most expedient way to
    reuse the existing rendering code from foggie/scripts/Diagnosis.py
'''

from foggie.fogghorn.header import *
from foggie.fogghorn.util import *

def diagnosis_plots(ds, region, args, output_filename):
    """ not yet documented
    """

    run_name = (os.getcwd()).split("/")[-1]

    ad = ds.all_data()

    metallicity = region[('gas', 'metallicity')]
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
    
    prefix = ds.snapname

    plt.figure() 
    plt.scatter(np.log10(number_density), np.log10(H2_fraction), s=0.1, label='all cells', color='blue') 
    plt.scatter(np.log10(number_density[metallicity > 1e-6]), np.log10(H2_fraction[metallicity > 1e-6]), s=0.1, label='cells with metals', color='green') 
    plt.xlabel('log Number Density') 
    plt.ylabel('log f_H2') 
    plt.xlim(-2, 13) 
    plt.ylim(-5, 0.5) 
    plt.title(run_name + '  ' + ds.snapname + ',   z = ' + str(ds.current_redshift) ) 
    plt.legend() 
    plt.savefig('plots/'+prefix +     '_fH2_number_density_DIAGNOSIS.png') 

    plt.figure() 
    plt.scatter(np.log10(number_density), np.log10(temperature), s=0.1, label='all cells', color='blue') 
    plt.scatter(np.log10(number_density[metallicity > 1e-6]), np.log10(temperature[metallicity > 1e-6]), s=0.1, label='cells with metals', color='green') 
    plt.xlabel('log Number Density') 
    plt.ylabel('log Temperarture') 
    plt.xlim(0, 5) 
    plt.ylim(1, 6) 
    plt.title(run_name + '  ' + ds.snapname + ',   z = ' + str(ds.current_redshift) ) 
    plt.legend() 
    plt.savefig('plots/'+prefix +     '_temp_number_density_metalcode_DIAGNOSIS.png') 

    plt.figure() 
    plt.scatter(np.log10(number_density), np.log10(cooling_time), s=0.1, label='all cells', color='blue') 
    plt.scatter(np.log10(number_density[metallicity > 1e-6]), np.log10(cooling_time[metallicity > 1e-6]), s=0.1, label='cells with metals', color='green') 
    plt.xlabel('log Number Density') 
    plt.ylabel('log Cooling Time') 
    plt.xlim(0, 5) 
    plt.ylim(2, 8) 
    plt.title(run_name + '  ' + ds.snapname + ',   z = ' + str(ds.current_redshift) ) 
    plt.legend() 
    plt.savefig('plots/'+prefix +     '_tcool_number_density_metalcode_DIAGNOSIS.png') 
    
    plt.figure() 
    plt.scatter(np.log10(HI_density), np.log10(H2_density), s=0.1) 
    plt.xlabel('log HI density') 
    plt.ylabel('log H2 density') 
    plt.xlim(-30,-20) 
    plt.ylim(-30,-20) 
    plt.title(run_name + '  ' + ds.snapname + ',   z = ' + str(ds.current_redshift) ) 
    plt.savefig('plots/'+prefix +     '_H2density_HIdensity_DIAGNOSIS.png') 
    
    plt.figure() 
    plt.scatter(np.log10(cell_mass), np.log10(H2_density), s=0.1) 
    plt.xlabel('log Cell Mass') 
    plt.ylabel('log H2 density') 
    plt.xlim(0, 9) 
    plt.ylim(-30,-20) 
    plt.title(run_name + '  ' + ds.snapname + ',   z = ' + str(ds.current_redshift) ) 
    plt.savefig('plots/'+prefix +     '_H2density_cellmass_DIAGNOSIS.png') 

    plt.figure() 
    plt.scatter(np.log10(Total_Density), np.log10(Metal_Density), s=0.1) 
    plt.plot([-35,-15],[-35,-15]) 
    plt.xlabel('log Total_Density') 
    plt.ylabel('log Metal Density') 
    plt.xlim(-35,-15) 
    plt.ylim(-35,-15) 
    plt.title(run_name + '  ' + ds.snapname + ',   z = ' + str(ds.current_redshift) ) 
    plt.savefig('plots/'+prefix +     '_Metal_Density_Density_DIAGNOSIS.png') 
    
    star_particle_mass = region[('stars', 'particle_mass')].in_units('Msun') 
    star_particle_time = region[('stars', 'creation_time')].in_units('yr')
    star_particle_z = region[('stars', 'metallicity_fraction')] 
    all_star_particle_mass = ad[('stars', 'particle_mass')].in_units('Msun')
    all_star_particle_time = ad[('stars', 'creation_time')].in_units('yr')
    all_star_particle_z = ad[('stars', 'metallicity_fraction')] 
    if (ds.parameters['StarParticleCreation'] == 1):  minimum_mass = ds.parameters['StarMakerMinimumMass'] 
    if (ds.parameters['StarParticleCreation'] == 2048):  
        if ('H2StarMakerMinimumMass' in ds.parameters): minimum_mass = ds.parameters['H2StarMakerMinimumMass'] 
        else: minimum_mass = ds.parameters['StarMakerMinimumMass'] 
    
    plt.figure() 
    plt.scatter(all_star_particle_time, np.log10(all_star_particle_mass), s=0.1, color='orange') 
    plt.scatter(star_particle_time, np.log10(star_particle_mass), s=0.1, color='blue') 
    plt.plot([0, 3e9], [np.log10(minimum_mass), np.log10(minimum_mass)], linestyle='dashed') 
    plt.xlabel('Star Particle Creation Time') 
    plt.ylabel('Stellar Mass [Msun]') 
    plt.xlim(0, 13e9) 
    plt.ylim(-0.5, 10) 
    plt.title(run_name + '  ' + ds.snapname + ',   z = ' + str(ds.current_redshift) ) 
    plt.savefig('plots/'+prefix +     '_starmass_startime_DIAGNOSIS.png') 

    plt.figure() 
    plt.scatter(np.log10(all_star_particle_z), np.log10(all_star_particle_mass), s=0.1, color='orange') 
    plt.scatter(np.log10(star_particle_z), np.log10(star_particle_mass), s=0.1, color='blue') 
    plt.plot([-10.5, 1.5], [np.log10(minimum_mass), np.log10(minimum_mass)], linestyle='dashed') 
    plt.xlabel('log Stellar Metallicity') 
    plt.ylabel('Stellar Mass [Msun]') 
    plt.xlim(-10.5, 1.5) 
    plt.ylim(-0.5, 10) 
    plt.title(run_name + '  ' + ds.snapname + ',   z = ' + str(ds.current_redshift) ) 
    plt.savefig('plots/'+prefix +     '_starmass_starz_DIAGNOSIS.png') 

    plt.figure() 
    plt.scatter(all_star_particle_time, np.log10(all_star_particle_z), s=0.1, color='orange') 
    plt.scatter(star_particle_time, np.log10(star_particle_z), s=0.1, color='blue') 
    plt.xlabel('Star Particle Creation Time') 
    plt.ylabel('log Stellar Metallicity') 
    plt.xlim(0, 13e9) 
    plt.ylim(-10.5, 1.5) 
    plt.title(run_name + '  ' + ds.snapname + ',   z = ' + str(ds.current_redshift) ) 
    plt.savefig('plots/'+prefix +     '_starz_startime_DIAGNOSIS.png') 