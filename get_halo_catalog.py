
import yt 
from yt.analysis_modules.halo_analysis.api import HaloCatalog
from astropy.table import Table 
from astropy import units 
import numpy as np 
import matplotlib.pyplot as plt 

def get_halo_catalog(ds): 

    hc = HaloCatalog(data_ds=ds, finder_method='hop')
    hc.create() 

    t = Table()
    t['id'] = [1., 4]
    t['x'] = [1., 4]
    t['y'] = [1., 4]
    t['z'] = [1., 4]
    t['mass'] = [1., 4]
    t['rvir'] = [1., 4]
    
    for i in hc.catalog: 
        print i['particle_identifier'],i['particle_position_x'],i['particle_position_y'], \
              i['particle_position_z'],i['particle_mass'], i['virial_radius']
        t.add_row([i['particle_identifier'],\
                   i['particle_position_x']/units.kpc.to('cm'), 
                   i['particle_position_y']/units.kpc.to('cm'), 
                   i['particle_position_z']/units.kpc.to('cm'), 
                   i['particle_mass']/units.M_sun.to('g'), 
                   i['virial_radius']/units.kpc.to('cm')])
    
     
    return t[2:], hc 


def get_stellar_masses(ds, halo_table): 

    if ~('Mstar' in halo_table.keys()): halo_table['Mstar'] = halo_table['mass'] * 0.0 
 
    for halo0 in halo_table: 

        halo_center = np.array( [ halo0['x'], halo0['y'], halo0['z'] ] ) / 143886. 
        ad = ds.sphere(halo_center, radius=(halo0['rvir'],'kpc')) 

        x = ad['particle_position_x'].convert_to_units('kpc')
        y = ad['particle_position_y'].convert_to_units('kpc')
        z = ad['particle_position_z'].convert_to_units('kpc')
        ptype = ad['particle_type']
        mass = ad['particle_mass'].convert_to_units('Msun') 
        StarMass = np.sum(mass[ptype == 2])
        halo0['Mstar'] = StarMass 
        print halo0 

    ms = halo_table[[halo_table['Mstar'] > 1e3]]

    return ms 

def plot_baryon_ratios(ds, halo_table): 

    plt.plot(np.log10(halo_table['mass']), halo_table['Mstar'] / halo_table['mass'] / 0.161, 's')
    plt.xlim((9, 13)) 
    plt.ylim((0, 1.2)) 


