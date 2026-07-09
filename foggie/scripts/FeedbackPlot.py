#!/usr/bin/env python
# coding: utf-8

import yt, unyt, numpy as np, os, matplotlib.pyplot as plt, h5py
from datetime import datetime
from foggie.utils.foggie_load import *
from foggie.utils.consistency import *
from yt.funcs import mylog
import glob
from astropy.table import Table, vstack, Column

def catalogs_to_table_old(run, prefix='DD'):

    halo_tables = glob.glob(run+'/halo_catalogs/*/'+prefix+'????.0.fits')

    big_table = Table.read(halo_tables[0].split('.')[0]+'.0.fits')
    new_ds = yt.load(halo_tables[0].split('.')[0]+'.0.h5')
    big_table['z'] = new_ds['CosmologyCurrentRedshift']
    big_table = big_table[0]

    for tt in halo_tables[1:]:
        this_table = Table.read(tt.split('.')[0]+'.0.fits')
        this_ds = yt.load(tt.split('.')[0]+'.0.h5')
        this_table['z'] = this_ds['CosmologyCurrentRedshift']
        big_table = vstack([big_table, this_table[0]])

    big_table.sort('z', reverse=True)

    return big_table

def catalogs_to_table(run, prefix='DD'):

    halo_tables = glob.glob(run+'/halo_catalogs/*/'+prefix+'????.0.fits')

    tables = []
    for tt in halo_tables:
        this_table = Table.read(tt.split('.')[0]+'.0.fits')
        with h5py.File(tt.split('.')[0]+'.0.h5', 'r') as f:
            this_table['z'] = f.attrs['CosmologyCurrentRedshift']
        tables.append(this_table[0])

    big_table = vstack(tables)
    big_table.sort('z', reverse=True)

    return big_table

def baryons_vs_z(table, zrange=[3, 1.5], title='Baryon Budget', filename='a.png'):

    plt.figure() 
    plt.scatter([0], [0])
    plt.xlim(zrange[0], zrange[1])
    plt.ylim(0, 0.3)

    plt.xlabel('Redshift')
    plt.ylabel('Fraction of Halo Total Mass')

    total_mass = table['total_mass'].to('Msun') 

    #HOT
    baryon_sum = table['total_warm_cgm_gas_mass'].to('Msun') + table['total_hot_cgm_gas_mass'].to('Msun') \
        + table['total_cold_cgm_gas_mass'].to('Msun') + table['total_cool_cgm_gas_mass'].to('Msun') \
        + table['total_ism_gas_mass'].to('Msun') +  table['total_star_mass'].to('Msun')

    plt.plot(table['z'], baryon_sum / table['total_mass'].to('Msun'), color='#f2dc61', linewidth=2)
    plt.fill_between(table['z'], baryon_sum / total_mass, color='#f2dc61', label='Hot CGM')

    #WARM
    baryon_sum = table['total_warm_cgm_gas_mass'].to('Msun') \
        + table['total_cold_cgm_gas_mass'].to('Msun') + table['total_cool_cgm_gas_mass'].to('Msun') \
        + table['total_ism_gas_mass'].to('Msun') +  table['total_star_mass'].to('Msun')
    warm_cgm_fraction = table['total_warm_cgm_gas_mass'].to('Msun') / total_mass 

    plt.plot(table['z'], baryon_sum / table['total_mass'].to('Msun'), color='#659B4d')
    plt.fill_between(table['z'], baryon_sum / table['total_mass'].to('Msun'), color='#659B4d', label='Warm CGM')
    plt.text(table['z'][-1], baryon_sum[-1] / total_mass[-1], str(warm_cgm_fraction[-1]*100.)[0:3]+'%', color='#659B4d')

    #COOL
    baryon_sum = table['total_cool_cgm_gas_mass'].to('Msun') \
        + table['total_cold_cgm_gas_mass'].to('Msun') \
        + table['total_ism_gas_mass'].to('Msun') +  table['total_star_mass'].to('Msun')
    cool_cgm_fraction = table['total_cool_cgm_gas_mass'].to('Msun') / total_mass 

    plt.plot(table['z'], baryon_sum / table['total_mass'].to('Msun'), color='#6f427b')
    plt.fill_between(table['z'], baryon_sum / total_mass, color='#6f427b', label='Cool CGM')
    plt.text(table['z'][-1], baryon_sum[-1] / total_mass[-1], str(cool_cgm_fraction[-1]*100.)[0:3]+'%', color='#6f427b')


    #COLD
    baryon_sum = table['total_cold_cgm_gas_mass'].to('Msun') \
        + table['total_ism_gas_mass'].to('Msun') +  table['total_star_mass'].to('Msun')
    cold_cgm_fraction = table['total_cold_cgm_gas_mass'].to('Msun') / total_mass 

    plt.plot(table['z'], baryon_sum / table['total_mass'].to('Msun'), color='#C66D64')
    plt.fill_between(table['z'], baryon_sum / total_mass, color='#C66D64', label='Cold CGM')
    plt.text(table['z'][-1], baryon_sum[-1]/ total_mass[-1], str(cold_cgm_fraction[-1]*100.)[0:3]+'%', color='#C66D64')

    #ISM
    baryon_sum = table['total_ism_gas_mass'].to('Msun') +  table['total_star_mass'].to('Msun')
    ism_fraction = table['total_ism_gas_mass'].to('Msun') / total_mass 

    plt.plot(table['z'], baryon_sum / total_mass, color='#4a6091')
    plt.fill_between(table['z'], baryon_sum / total_mass, color='#4a6091', label='ISM')
    plt.text(table['z'][-1], baryon_sum[-1] / total_mass[-1], str(ism_fraction[-1]*100.)[0:3]+'%', color='#4a6091')

    #STARS
    baryon_sum = table['total_star_mass'].to('Msun')
    star_fraction = baryon_sum / total_mass 

    plt.plot(table['z'], baryon_sum / table['total_mass'].to('Msun'), color='#9e302c')
    plt.fill_between(table['z'], baryon_sum / total_mass, color='#9e302c', label='Stars')
    plt.text(table['z'][-1], baryon_sum[-1]/total_mass[-1], str(star_fraction[-1]*100.)[0:3]+'%', color='#9e302c')

    plt.plot([0, 7], [0.0461 / 0.285, 0.0461 / 0.285], linestyle='dashed', color='orange')
    plt.title(title)
    plt.legend(frameon=0, loc='upper right', ncols=3)
    plt.savefig(filename) 

    return 


pr63_table = catalogs_to_table('/u/jtumlins/nobackup/pr63/H2radtest')
print('Read the PR63 table') 
therm_table = catalogs_to_table('H2therm_ff')
print('Read the therm tables') 
mech_table = catalogs_to_table('H2mech_tab_cont_ff')
print('Read the mech_tab_cont table') 
rad_table = catalogs_to_table('H2radtest100')
print('Read the radtest100 table') 
default_table = catalogs_to_table('H2mech_tab_cont_cassi')
print('Read the default table') 
mom5_table = catalogs_to_table('H2mech_tab_cont_5mom_ff')
print('Read the Mom x5 table') 
mom_rad_table = catalogs_to_table('H2mech_tab_cont_mom3x_rad100_ff')
print('Read the Mom Rad table') 
radius3_table = catalogs_to_table('H2mech_tab_cont_radius3_ff')
print('Read the Feedback Radius = 3 table') 
rad3_rad100_table = catalogs_to_table('H2mech_tab_cont_radius3_rad100_ff')
print('Read the Feedback Radius = 3 Rad = 100 table') 

numerical_table = catalogs_to_table('H2numerical')
print('Read the Full Res Numerical Run') 

#Plot the >500 km/s outflow mass vs. SFR7 
plt.figure() 
plt.scatter(np.log10(therm_table['sfr7'].to('Msun/yr').value), np.log10(therm_table['outflow_mass_500'].to('Msun').value), s = 7, label='Thermal', color='blue')
plt.scatter(np.log10(mech_table['sfr7'].to('Msun/yr').value), np.log10(mech_table['outflow_mass_500'].to('Msun').value), s = 7, label='Mech', color='orange')
plt.scatter(np.log10(rad_table['sfr7'].to('Msun/yr').value), np.log10(rad_table['outflow_mass_500'].to('Msun').value), s = 7, label='Rad100', color='green')
plt.scatter(np.log10(mom5_table['sfr7'].to('Msun/yr').value), np.log10(mom5_table['outflow_mass_500'].to('Msun').value), s = 7, label='Mom x5', color='darkslateblue')
plt.scatter(np.log10(mom_rad_table['sfr7'].to('Msun/yr').value), np.log10(mom_rad_table['outflow_mass_500'].to('Msun').value), s = 7, label='Mom+Rad', color='pink')
plt.scatter(np.log10(radius3_table['sfr7'].to('Msun/yr').value), np.log10(radius3_table['outflow_mass_500'].to('Msun').value), s = 7, label='Radius = 3', color='red')
plt.scatter(np.log10(rad3_rad100_table['sfr7'].to('Msun/yr').value), np.log10(rad3_rad100_table['outflow_mass_500'].to('Msun').value), s = 7, label='Rad=3,Rad100', color='purple')

plt.legend()
plt.xlabel('log SFR7')
_ = plt.ylabel(' Outflow Mass (> 500 km/s)')
plt.xlim(0, 1.5)
plt.ylim(3, 9)
plt.title('Outflow Mass > 500 vs. SFR7') 
plt.savefig('outflow_mass_500_vs_sfr7.png') 
print('outflow_mass_500_vs_sfr7') 
plt.close() 

#Plot the SFR7 vs redshift 
plt.figure() 
plt.plot(therm_table['z'], therm_table['sfr7'].to('Msun/yr'), label='Thermal', color='blue', linestyle='solid')
plt.plot(mech_table['z'], mech_table['sfr7'].to('Msun/yr'), label='Mech', color='orange', linestyle='dashed')
plt.plot(rad_table['z'], rad_table['sfr7'].to('Msun/yr'), label='Rad100', color='green', linestyle='dashed')
plt.plot(default_table['z'], default_table['sfr7'].to('Msun/yr'), color='orange', linestyle='solid')
plt.plot(mom5_table['z'], mom5_table['sfr7'].to('Msun/yr'), label = 'Mom x5', color='darkslateblue', linestyle='solid')
plt.plot(mom_rad_table['z'], mom_rad_table['sfr7'].to('Msun/yr'), label = 'Mom+Rad', color='pink', linestyle='solid')
plt.plot(radius3_table['z'], radius3_table['sfr7'].to('Msun/yr'), label = 'Radius=3', color='red', linestyle='solid')
plt.plot(rad3_rad100_table['z'], rad3_rad100_table['sfr7'].to('Msun/yr'), label = 'Rad=3,Rad100', color='purple', linestyle='solid')

plt.xlim(3.2, 0.0)
plt.ylim(-0.02, 50)
_ = plt.xlabel('Redshift')
_ = plt.ylabel('SFR [Msun/yr]')
plt.title('SFR and Outflows vs. Feedback Scheme')
plt.legend()
plt.savefig('SFR7_vs_redshift.png') 
print('SFR7_vs_redshift') 
plt.close() 


#Plot the cumulative sum of > 500 km/s mass
plt.figure() 
plt.plot(therm_table['z'], therm_table['outflow_mass_500'].cumsum().to('Msun')/1e11, label='Thermal', color='blue', linestyle='dashed')
plt.plot(mech_table['z'], mech_table['outflow_mass_500'].cumsum().to('Msun')/1e11, label='Mech', color='orange', linestyle='dashed')
plt.plot(rad_table['z'], rad_table['outflow_mass_500'].cumsum().to('Msun')/1e11, label='Rad100', color='green', linestyle='dashed')
plt.plot(mom5_table['z'], mom5_table['outflow_mass_500'].cumsum().to('Msun')/1e11, label='Mom x5', color='darkslateblue', linestyle='dashed')
plt.plot(mom_rad_table['z'], mom_rad_table['outflow_mass_500'].cumsum().to('Msun')/1e11, label='Mom+Rad', color='pink', linestyle='dashed')
plt.plot(radius3_table['z'], radius3_table['outflow_mass_500'].cumsum().to('Msun')/1e11, label='Radius=3', color='red', linestyle='dashed')
plt.plot(rad3_rad100_table['z'], rad3_rad100_table['outflow_mass_500'].cumsum().to('Msun')/1e11, label='Rad3,Rad100', color='purple', linestyle='dashed')

plt.xlim(3.2, 0.0)
plt.ylim(-0.02, 1)
_ = plt.xlabel('Redshift')
_ = plt.ylabel('Cumulative Outflow Mass (>500) / 1e11')
_ = plt.title('Cumulative Outflow Mass vs Feedback Scheme')
plt.legend(loc='upper left')
plt.savefig('Outflow500_cumulative.png') 
print('SFR7_vs_redshift') 



#Plot the > 500 km/s mass vs z 
plt.figure() 
plt.plot(therm_table['z'], therm_table['outflow_mass_500'].to('Msun') / 1e8, label='Thermal', color='blue', linestyle='solid')
plt.plot(mech_table['z'], mech_table['outflow_mass_500'].to('Msun') / 1e8, label='Mech', color='orange', linestyle='dashed')
plt.plot(rad_table['z'], rad_table['outflow_mass_500'].to('Msun') / 1e8, label='Rad100', color='green', linestyle='dashed')
plt.plot(default_table['z'], default_table['outflow_mass_500'].to('Msun') / 1e8, color='orange', linestyle='dashed')
plt.plot(mom5_table['z'], mom5_table['outflow_mass_500'].to('Msun') / 1e8, label='Mom x5', color='darkslateblue', linewidth=3, linestyle='solid')
plt.plot(mom_rad_table['z'], mom_rad_table['outflow_mass_500'].to('Msun') / 1e8, label='Mom+Rad', color='pink', linewidth=3, linestyle='solid')
plt.plot(radius3_table['z'], radius3_table['outflow_mass_500'].to('Msun') / 1e8, label='Radius=3', color='red', linewidth=3, linestyle='solid')
plt.plot(rad3_rad100_table['z'], rad3_rad100_table['outflow_mass_500'].to('Msun') / 1e8, label='Rad=3, Rad100', color='purple', linewidth=3, linestyle='solid')

plt.xlim(3.2, 0.0)
plt.ylim(-0.02, 5)
_ = plt.xlabel('Redshift')
_ = plt.ylabel('Outflow Mass 500 / 1e8')
_ = plt.title('Outflows > 500 km/s vs. Feedback Scheme')
plt.legend()
plt.savefig('outflow500_vs_z.png') 
print('outflow500_vs_z.png') 
plt.close() 


plt.figure() 
plt.plot(therm_table['z'], therm_table['outflow_mass_300'].to('Msun') / 1e9, label='Thermal', color='blue', linestyle='dashed')
plt.plot(mech_table['z'], mech_table['outflow_mass_300'].to('Msun') / 1e9, label='Mech', color='orange', linestyle='dashed')
plt.plot(rad_table['z'], rad_table['outflow_mass_300'].to('Msun') / 1e9, label='Rad100', color='green', linestyle='dashed')
plt.plot(default_table['z'], default_table['outflow_mass_300'].to('Msun') / 1e9, color='orange', linestyle='dashed')
plt.plot(mom5_table['z'], mom5_table['outflow_mass_300'].to('Msun') / 1e9, label='Mom 5x', color='darkslateblue', linestyle='solid')
plt.plot(mom_rad_table['z'], mom_rad_table['outflow_mass_300'].to('Msun') / 1e9, label='Mom+Rad', color='pink', linestyle='solid')
plt.plot(radius3_table['z'], radius3_table['outflow_mass_300'].to('Msun') / 1e9, label='Radius=3', color='red', linestyle='solid')
plt.plot(rad3_rad100_table['z'], rad3_rad100_table['outflow_mass_300'].to('Msun') / 1e9, label='Rad=3,Rad100', color='purple', linestyle='solid')

plt.xlim(3.2, 0.0)
plt.ylim(-0.02, 5)
_ = plt.xlabel('Redshift')
_ = plt.ylabel('Outflow Mass (300) / 1e9')
_ = plt.title('Outflows vs. Feedback Scheme')
plt.legend()
plt.savefig('outflow_mass_300.png') 
print('outflow500_vs_z.png') 


plt.figure() 
plt.plot(therm_table['z'], therm_table['total_star_mass'].to('Msun')/1e10, label='Thermal', color='blue', linestyle='solid')
plt.plot(mech_table['z'], (mech_table['total_star_mass']).to('Msun')/1e10, label='Mech', color='orange', linestyle='solid')
plt.plot(rad_table['z'], (rad_table['total_star_mass']).to('Msun')/1e10, label='Rad100', color='green', linestyle='solid')
plt.plot(default_table['z'], (default_table['total_star_mass']).to('Msun')/1e10,  color='orange', linestyle='solid')
plt.plot(mom5_table['z'], (mom5_table['total_star_mass']).to('Msun')/1e10,  label='Mom x5', color='darkslateblue', linestyle='dashed')
plt.plot(mom_rad_table['z'], (mom_rad_table['total_star_mass']).to('Msun')/1e10, label='Mom+Rad',  color='pink', linestyle='dashed')
plt.plot(radius3_table['z'], (radius3_table['total_star_mass']).to('Msun')/1e10, label='Radius=3',  color='red', linestyle='dashed')
plt.plot(rad3_rad100_table['z'], (rad3_rad100_table['total_star_mass']).to('Msun')/1e10, label='Radius=3, Rad100',  color='purple', linestyle='dashed')

_ = plt.xlabel('Redshift')
_ = plt.ylabel('Mstar / 1e10')
_ = plt.legend(loc = 'upper left')
_ = plt.text(0.8, 0.6, 'Linear scale')
_ = plt.title('Stellar Mass vs. Feedback Scheme')
_ = plt.xlim(3.2, 0.0)
_ = plt.ylim(0,10)
plt.savefig('total_star_mass.png') 
print('total_star_mass.png') 
plt.close() 


plt.figure() 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
ax1.plot(therm_table['z'], therm_table['total_cgm_gas_mass'].to('Msun')/1e10, color='blue', label='Therm')
ax1.plot(mech_table['z'], mech_table['total_cgm_gas_mass'].to('Msun')/1e10, color='orange', label='Mech')
ax1.plot(mom5_table['z'], mom5_table['total_cgm_gas_mass'].to('Msun')/1e10, color='darkslateblue', label='Mom x5')
ax1.plot(mom_rad_table['z'], mom_rad_table['total_cgm_gas_mass'].to('Msun')/1e10, color='pink', label='Mom+Rad')
ax1.plot(radius3_table['z'], radius3_table['total_cgm_gas_mass'].to('Msun')/1e10, color='red', label='Radius=3')
ax1.plot(rad3_rad100_table['z'], rad3_rad100_table['total_cgm_gas_mass'].to('Msun')/1e10, color='purple', label='Rad=3,Rad100')
ax1.legend()

ax2.plot(therm_table['z'], therm_table['total_cgm_gas_mass_2rvir'].to('Msun')/1e10, color='blue', label='Therm')
ax2.plot(mech_table['z'], mech_table['total_cgm_gas_mass_2rvir'].to('Msun')/1e10, color='orange', label='Mech')
ax2.plot(mom5_table['z'], mom5_table['total_cgm_gas_mass_2rvir'].to('Msun')/1e10, color='darkslateblue', label='Mom x5')
ax2.plot(mom_rad_table['z'], mom_rad_table['total_cgm_gas_mass_2rvir'].to('Msun')/1e10, color='pink', label='Mom+Rad')
ax2.plot(radius3_table['z'], radius3_table['total_cgm_gas_mass_2rvir'].to('Msun')/1e10, color='red', label='Radius=3')
ax2.plot(rad3_rad100_table['z'], rad3_rad100_table['total_cgm_gas_mass_2rvir'].to('Msun')/1e10, color='purple', label='Rad=3,Rad100')
ax2.legend()

ax1.set_box_aspect(1)
ax2.set_box_aspect(1)
ax1.set_xlim(3.2, 0.0)
ax2.set_xlim(3.2, 0.0)
ax1.set_ylim(0, 4)
ax2.set_ylim(0, 4)

ax1.set_title('CGM Mass R < Rvir')
ax2.set_title('CGM Mass R < 2Rvir')

ax1.set_xlabel('Redshift')
ax2.set_xlabel('Redshift')
_ = ax1.set_ylabel('CGM Total Mass in Msun / 1e10')
plt.savefig('total_cgm_gas_mass.png') 
print('total_cgm_gas_mass.png') 
plt.close() 


plt.figure() 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
ax1.plot(therm_table['z'], therm_table['total_warm_cgm_gas_mass'].to('Msun')/1e10, color='blue', label='Therm')
ax1.plot(mech_table['z'], mech_table['total_warm_cgm_gas_mass'].to('Msun')/1e10, color='orange', label='Mech')
ax1.plot(mom5_table['z'], mom5_table['total_warm_cgm_gas_mass'].to('Msun')/1e10, color='darkslateblue', label='Mom 5x')
ax1.plot(mom_rad_table['z'], mom_rad_table['total_warm_cgm_gas_mass'].to('Msun')/1e10, color='pink', label='Mom+Rad')
ax1.plot(radius3_table['z'], radius3_table['total_warm_cgm_gas_mass'].to('Msun')/1e10, color='red', label='Radius=3')
ax1.plot(rad3_rad100_table['z'], rad3_rad100_table['total_warm_cgm_gas_mass'].to('Msun')/1e10, color='purple', label='Rad=3,Rad100')
ax1.legend()

ax2.plot(therm_table['z'], therm_table['total_warm_cgm_gas_mass_2rvir'].to('Msun')/1e10, color='blue', label='Therm')
ax2.plot(mech_table['z'], mech_table['total_warm_cgm_gas_mass_2rvir'].to('Msun')/1e10, color='orange', label='Mech')
ax2.plot(mom5_table['z'], mom5_table['total_warm_cgm_gas_mass_2rvir'].to('Msun')/1e10, color='darkslateblue', label='Mom 5x')
ax1.plot(mom_rad_table['z'], mom_rad_table['total_warm_cgm_gas_mass_2rvir'].to('Msun')/1e10, color='pink', label='Mom+Rad')
ax1.plot(radius3_table['z'], radius3_table['total_warm_cgm_gas_mass_2rvir'].to('Msun')/1e10, color='red', label='Radius=3')
ax2.plot(rad3_rad100_table['z'], rad3_rad100_table['total_warm_cgm_gas_mass_2rvir'].to('Msun')/1e10, color='purple', label='Radius=3,Rad100')
ax2.legend()

ax1.set_box_aspect(1)
ax2.set_box_aspect(1)
ax1.set_xlim(3.2, 0.0)
ax2.set_xlim(3.2, 0.0)
ax1.set_ylim(0, 1.2)
ax2.set_ylim(0, 1.2)

ax1.set_title('CGM Warm Mass R < Rvir')
ax2.set_title('CGM Warm Mass R < 2Rvir')

ax1.set_xlabel('Redshift')
ax2.set_xlabel('Redshift')
_ = ax1.set_ylabel('CGM Warm Mass in Msun / 1e10')
plt.savefig('total_warm_cgm_gas_mass.png') 
print('total_warm_cgm_gas_mass.png') 
plt.close() 


plt.figure() 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
ax1.plot(therm_table['z'], therm_table['total_hot_cgm_gas_mass'].to('Msun')/1e9, color='blue', label='Rvir Therm')
ax1.plot(mech_table['z'], mech_table['total_hot_cgm_gas_mass'].to('Msun')/1e9, color='orange', label='Rvir Mech')
ax1.plot(mom5_table['z'], mom5_table['total_hot_cgm_gas_mass'].to('Msun')/1e9, color='darkslateblue', label='Rvir Mom x5')
ax1.plot(mom_rad_table['z'], mom_rad_table['total_hot_cgm_gas_mass'].to('Msun')/1e9, color='pink', label='Rvir Mom+Rad')
ax1.plot(radius3_table['z'], radius3_table['total_hot_cgm_gas_mass'].to('Msun')/1e9, color='red', label='Radius=3')
ax1.plot(rad3_rad100_table['z'], rad3_rad100_table['total_hot_cgm_gas_mass'].to('Msun')/1e9, color='purple', label='Rad=3,Rad100')

ax2.plot(therm_table['z'], therm_table['total_hot_cgm_gas_mass_2rvir'].to('Msun')/1e9, color='blue', label='2Rvir Therm')
ax2.plot(mech_table['z'], mech_table['total_hot_cgm_gas_mass_2rvir'].to('Msun')/1e9, color='orange', label='2Rvir Mech')
ax2.plot(mom5_table['z'], mom5_table['total_hot_cgm_gas_mass_2rvir'].to('Msun')/1e9, color='darkslateblue', label='2Rvir Mom x5')
ax2.plot(mom_rad_table['z'], mom_rad_table['total_hot_cgm_gas_mass_2rvir'].to('Msun')/1e9, color='pink', label='2Rvir Mom+Rad')
ax2.plot(radius3_table['z'], radius3_table['total_hot_cgm_gas_mass_2rvir'].to('Msun')/1e9, color='red', label='Radius=3')
ax2.plot(rad3_rad100_table['z'], rad3_rad100_table['total_hot_cgm_gas_mass_2rvir'].to('Msun')/1e9, color='purple', label='Rad=3,Rad100')

for a in [ax1, ax2]:
    a.set_box_aspect(1)
    a.set_xlim(3.2, 0.4)
    a.set_ylim(0, 1.2)
    a.set_xlabel('Redshift')
    a.legend()

ax1.set_title('CGM Hot Mass R < Rvir')
ax2.set_title('CGM Hot Mass R < 2Rvir')
_ = ax1.set_ylabel('CGM Hot Mass in Msun / 1e9')
plt.savefig('total_hot_cgm_gas_mass.png') 
print('total_hot_cgm_gas_mass.png') 



plt.figure() 
plt.plot(therm_table['z'], therm_table['total_ism_gas_mass'].to('Msun')/1e10, label='Thermal', color='blue', linestyle='solid')
plt.plot(default_table['z'], default_table['total_ism_gas_mass'].to('Msun')/1e10,  color='orange', linestyle='solid')
plt.plot(mech_table['z'], mech_table['total_ism_gas_mass'].to('Msun')/1e10, label='Mech', color='orange', linestyle='solid')
plt.plot(rad_table['z'], rad_table['total_ism_gas_mass'].to('Msun')/1e10, label='Rad100', color='green', linestyle='solid')
plt.plot(mom5_table['z'], mom5_table['total_ism_gas_mass'].to('Msun')/1e10, label='Mom x5', color='darkslateblue', linestyle='solid')
plt.plot(mom_rad_table['z'], mom_rad_table['total_ism_gas_mass'].to('Msun')/1e10, label='Mom+Rad', color='pink', linestyle='solid')
plt.plot(radius3_table['z'], radius3_table['total_ism_gas_mass'].to('Msun')/1e10, label='Radius=3', color='red', linestyle='solid')
plt.plot(rad3_rad100_table['z'], rad3_rad100_table['total_ism_gas_mass'].to('Msun')/1e10, label='Rad=3,Rad100', color='purple', linestyle='solid')

_ = plt.xlabel('Redshift')
_ = plt.ylabel('M_ISM / 1e10')
plt.legend()
plt.title('ISM Mass vs. Feedback Scheme')
_ = plt.xlim(3.2, 0.0)
_ = plt.ylim(0, 4)
plt.savefig('total_ism_gas_mass.png') 
print('total_ism_gas_mass.png') 
plt.close() 



plt.figure() 
plt.plot(therm_table['z'], therm_table['total_ism_HI_mass'].to('Msun')/1e10,  color='blue', linestyle='dotted')
plt.plot(mech_table['z'], mech_table['total_ism_HI_mass'].to('Msun')/1e10, color='orange', linestyle='dotted')
plt.plot(mom5_table['z'], mom5_table['total_ism_HI_mass'].to('Msun')/1e10, color='darkslateblue', linestyle='dotted') 
plt.plot(rad_table['z'], rad_table['total_ism_HI_mass'].to('Msun')/1e10, color='green', linestyle='dotted')
plt.plot(mom_rad_table['z'], mom_rad_table['total_ism_HI_mass'].to('Msun')/1e10,  color='pink', linestyle='dotted')
plt.plot(radius3_table['z'], radius3_table['total_ism_HI_mass'].to('Msun')/1e10, color='red', linestyle='dotted')
plt.plot(rad3_rad100_table['z'], rad3_rad100_table['total_ism_HI_mass'].to('Msun')/1e10, color='purple', linestyle='dotted')
plt.plot(default_table['z'], default_table['total_ism_HI_mass'].to('Msun')/1e10, color='orange', linestyle='dotted')
plt.plot(pr63_table['z'], pr63_table['total_ism_HI_mass'].to('Msun')/1e10, color='cyan', linestyle='solid')

plt.plot(therm_table['z'], therm_table['total_ism_H2_mass'].to('Msun')/1e10, label='Thermal', color='blue', linestyle='dashed')
plt.plot(mech_table['z'], mech_table['total_ism_H2_mass'].to('Msun')/1e10, label='Mech', color='orange', linestyle='dashed')
plt.plot(mom5_table['z'], mom5_table['total_ism_H2_mass'].to('Msun')/1e10, label='Mom x5', color='darkslateblue', linestyle='dashed') 
plt.plot(rad_table['z'], rad_table['total_ism_H2_mass'].to('Msun')/1e10, label='Rad100', color='green', linestyle='dashed')
plt.plot(mom_rad_table['z'], mom_rad_table['total_ism_H2_mass'].to('Msun')/1e10, label='Mom+Rad', color='pink', linestyle='dashed')
plt.plot(radius3_table['z'], radius3_table['total_ism_H2_mass'].to('Msun')/1e10, label='Radius=3', color='red', linestyle='dashed')
plt.plot(rad3_rad100_table['z'], rad3_rad100_table['total_ism_H2_mass'].to('Msun')/1e10, label='Rad=3,Rad100', color='purple', linestyle='dashed')
plt.plot(default_table['z'], default_table['total_ism_H2_mass'].to('Msun')/1e10,  color='orange', linestyle='solid')
plt.plot(pr63_table['z'], pr63_table['total_ism_H2_mass'].to('Msun')/1e10, color='cyan', linestyle='solid')

_ = plt.xlabel('Redshift')
_ = plt.ylabel('M_HI or M_H2 / 1e10')
_ = plt.text(3.0, 2.25, 'dotted = HI')
_ = plt.text(3.0, 2.15, 'dashed = H2')
plt.legend(loc='upper left', ncols=3)
plt.title('ISM Mass vs. Feedback Scheme')
plt.xlim(3.2, 0.0)
plt.ylim(0, 3)
plt.savefig('total_ism_HI_H2_gas_mass.png') 
print('total_ism_HI_H2_gas_mass.png') 
plt.close() 


plt.figure() 
plt.plot(therm_table['z'], therm_table['total_ism_H2_mass'] / (therm_table['total_ism_HI_mass'] + therm_table['total_ism_H2_mass']) , label='Thermal', color='blue', linestyle='dotted')
plt.plot(mech_table['z'], mech_table['total_ism_H2_mass'] / (mech_table['total_ism_HI_mass'] + mech_table['total_ism_H2_mass']) , label='Mech', color='orange', linestyle='dotted')
plt.plot(rad_table['z'], rad_table['total_ism_H2_mass'] / (rad_table['total_ism_HI_mass'] + rad_table['total_ism_H2_mass']) , label='Rad100', color='green', linestyle='dotted')
plt.plot(mom5_table['z'], mom5_table['total_ism_H2_mass'] / (mom5_table['total_ism_HI_mass'] + mom5_table['total_ism_H2_mass']) , label='Mom x5', color='darkslateblue', linestyle='dotted')
plt.plot(mom_rad_table['z'], mom_rad_table['total_ism_H2_mass'] / (mom_rad_table['total_ism_HI_mass'] + mom_rad_table['total_ism_H2_mass']) , label='Mom+Rad', color='pink', linestyle='dashed')
plt.plot(radius3_table['z'], radius3_table['total_ism_H2_mass'] / (radius3_table['total_ism_HI_mass'] + radius3_table['total_ism_H2_mass']) , label='Radius=3', color='red', linestyle='dashed')
plt.plot(rad3_rad100_table['z'], rad3_rad100_table['total_ism_H2_mass'] / (rad3_rad100_table['total_ism_HI_mass'] + rad3_rad100_table['total_ism_H2_mass']) , label='Rad=3,Rad100', color='purple', linestyle='dashed')
plt.plot(pr63_table['z'], pr63_table['total_ism_H2_mass'] / (pr63_table['total_ism_HI_mass'] + pr63_table['total_ism_H2_mass']) , label='PR63 Rad', color='cyan', linestyle='solid')

_ = plt.xlabel('Redshift')
_ = plt.ylabel('Molecular Mass Fraction')
plt.legend()
plt.title('Molecular Mass Fraction vs. Feedback Scheme')
plt.xlim(3.2, 0.0)
plt.ylim(0, 1)
plt.savefig('molecular_fraction.png') 
print('molecular_fraction.png') 
plt.close() 

plt.figure() 
plt.plot(therm_table['z'], np.log10(therm_table['average_metallicity']), label='Thermal', color='blue', linestyle='solid')
plt.plot(mech_table['z'], np.log10(mech_table['average_metallicity']), label='Mech', color='orange', linestyle='solid')
plt.plot(rad_table['z'], np.log10(rad_table['average_metallicity']), label='Rad100', color='green', linestyle='solid')
plt.plot(default_table['z'], np.log10(default_table['average_metallicity']),  color='orange', linestyle='solid')
plt.plot(mom5_table['z'], np.log10(mom5_table['average_metallicity']), label='Mom x5',  color='darkslateblue', linestyle='solid')
plt.plot(mom_rad_table['z'], np.log10(mom_rad_table['average_metallicity']), label='Mom+Rad',  color='pink', linestyle='solid')
plt.plot(radius3_table['z'], np.log10(radius3_table['average_metallicity']), label='Radius=3',  color='red', linestyle='solid')
plt.plot(rad3_rad100_table['z'], np.log10(rad3_rad100_table['average_metallicity']), label='Rad=3,Rad100',  color='purple', linestyle='solid')


_ = plt.xlabel('Redshift')
_ = plt.ylabel('Average Metallicity')
plt.legend()
plt.title('ISM Metallicity vs. Feedback Scheme')
plt.xlim(3.2, 0.0)
plt.savefig('average_metallicity.png') 
print('average_metallicity.png') 
plt.close() 


plt.figure() 
plt.plot(np.log10(therm_table['total_star_mass'].to('Msun').value), np.log10(therm_table['sfr7'].to('Msun/yr').value), label='SFR7 Thermal', color='blue', linestyle='solid')
plt.plot(np.log10(mech_table['total_star_mass'].to('Msun').value), np.log10(mech_table['sfr7'].to('Msun/yr').value), label='SFR7 Mech', color='orange', linestyle='dotted')
plt.plot(np.log10(rad_table['total_star_mass'].to('Msun').value), np.log10(rad_table['sfr7'].to('Msun/yr').value), label='SFR7 Rad100', color='green', linestyle='dotted')
plt.plot(np.log10(default_table['total_star_mass'].to('Msun').value), np.log10(default_table['sfr7'].to('Msun/yr').value), color='orange', linestyle='dotted')
plt.plot(np.log10(mom5_table['total_star_mass'].to('Msun').value), np.log10(mom5_table['sfr7'].to('Msun/yr').value), color='darkslateblue', linestyle='dotted')
plt.plot(np.log10(mom_rad_table['total_star_mass'].to('Msun').value), np.log10(mom_rad_table['sfr7'].to('Msun/yr').value), color='pink', linestyle='dotted')
plt.plot(np.log10(radius3_table['total_star_mass'].to('Msun').value), np.log10(radius3_table['sfr7'].to('Msun/yr').value), color='red', linestyle='dotted')
plt.plot(np.log10(rad3_rad100_table['total_star_mass'].to('Msun').value), np.log10(rad3_rad100_table['sfr7'].to('Msun/yr').value), color='purple', linestyle='dotted')
plt.plot(np.log10(pr63_table['total_star_mass'].to('Msun').value), np.log10(pr63_table['sfr7'].to('Msun/yr').value), color='cyan', linestyle='solid')

plt.xlim(7.8, 11.9)
plt.ylim(-1.4, 3.4)
_ = plt.xlabel('log Mstar [Msun]')
_ = plt.ylabel('SFR [Msun/yr]')
plt.title('SFR and Outflows vs. Feedback Scheme')
plt.legend()
plt.savefig('SFR_vs_Mstar.png') 
print('SFR_vs_Mstar.png') 
plt.close() 


# NEW Tacconi plot
fig, ax = plt.subplots()
fig.patch.set_facecolor('none')
ax.set_facecolor('none')

ax.plot(np.log10(1.+therm_table['z']), np.log10(therm_table['total_ism_H2_mass'].to('Msun').value / therm_table['total_star_mass'].to('Msun').value), label=' Thermal', color='blue', linestyle='solid')
ax.plot(np.log10(1.+mech_table['z']), np.log10(mech_table['total_ism_H2_mass'].to('Msun').value / mech_table['total_star_mass'].to('Msun').value), label='Mech', color='orange', linestyle='dotted')
ax.plot(np.log10(1.+rad_table['z']), np.log10(rad_table['total_ism_H2_mass'].to('Msun').value / rad_table['total_star_mass'].to('Msun').value), label='Rad100', color='green', linestyle='dotted')
ax.plot(np.log10(1.+default_table['z']), np.log10(default_table['total_ism_H2_mass'].to('Msun').value / default_table['total_star_mass'].to('Msun').value), color='orange', linestyle='dotted')
ax.plot(np.log10(1.+mom5_table['z']), np.log10(mom5_table['total_ism_H2_mass'].to('Msun').value / mom5_table['total_star_mass'].to('Msun').value), label='Mom x5', color='darkslateblue', linestyle='solid')
ax.plot(np.log10(1.+mom_rad_table['z']), np.log10(mom_rad_table['total_ism_H2_mass'].to('Msun').value / mom_rad_table['total_star_mass'].to('Msun').value), label='Mom + Rad', color='pink', linestyle='solid')
ax.plot(np.log10(1.+radius3_table['z']), np.log10(radius3_table['total_ism_H2_mass'].to('Msun').value / radius3_table['total_star_mass'].to('Msun').value), label='Radius=3', color='red', linestyle='solid')
ax.plot(np.log10(1.+rad3_rad100_table['z']), np.log10(rad3_rad100_table['total_ism_H2_mass'].to('Msun').value / rad3_rad100_table['total_star_mass'].to('Msun').value), label='Rad=3,Rad100', color='purple', linestyle='solid')
ax.plot(np.log10(1.+pr63_table['z']), np.log10(pr63_table['total_ism_H2_mass'].to('Msun').value / pr63_table['total_star_mass'].to('Msun').value), label='PR63 Rad', color='cyan', linestyle='solid')

ax.set_xlim(-0.1, 0.9)
ax.set_ylim(-2, 1)
ax.set_xlabel('log z')
ax.set_ylabel('log [M_mol / Mstar]')
ax.set_title('Molecular Mass to Stellar Mass')
ax.legend(loc='lower right')

plt.savefig('Mmol_over_Mstar.png', dpi=200, bbox_inches='tight', transparent=True)
print('Mmol_over_Mstar.png')
plt.close()


plt.figure() 
plt.plot(therm_table['z'], therm_table['total_star_mass'] / therm_table['total_star_mass'], color='blue', label='Therm')
n = np.min([therm_table['z'].size, mech_table['z'].size])
plt.plot(mech_table['z'][0:n], mech_table['total_star_mass'][0:n] / therm_table['total_star_mass'][0:n], color='orange', label='Mech')
n = np.min([therm_table['z'].size, rad_table['z'].size])
plt.plot(rad_table['z'][0:n], rad_table['total_star_mass'][0:n] / therm_table['total_star_mass'][0:n], color='green', label='Rad100') 
n = np.min([therm_table['z'].size, mom5_table['z'].size])
plt.plot(mom5_table['z'][0:n], mom5_table['total_star_mass'][0:n] / therm_table['total_star_mass'][0:n], color='darkslateblue', label='Mom x5')
n = np.min([therm_table['z'].size, mom_rad_table['z'].size])
plt.plot(mom_rad_table['z'][0:n], mom_rad_table['total_star_mass'][0:n] / therm_table['total_star_mass'][0:n], color='pink', label='Mom+Rad')
n = np.min([therm_table['z'].size, radius3_table['z'].size])
plt.plot(radius3_table['z'][0:n], radius3_table['total_star_mass'][0:n] / therm_table['total_star_mass'][0:n], color='red', label='Mom5+Radius=3')
n = np.min([therm_table['z'].size, rad3_rad100_table['z'].size])
plt.plot(rad3_rad100_table['z'][0:n], rad3_rad100_table['total_star_mass'][0:n] / therm_table['total_star_mass'][0:n], color='purple', label='Radius=3,Rad100')
n = np.min([therm_table['z'].size, pr63_table['z'].size])
plt.plot(pr63_table['z'][0:n], pr63_table['total_star_mass'][0:n] / therm_table['total_star_mass'][0:n], color='cyan', label='Radius=3,Rad100')
print('pr63_table') 
print(pr63_table['z'][0:n]) 

print('therm_table') 
print(therm_table['z'][0:n]) 

plt.legend(loc='upper right', ncols=3) 
plt.xlim(3.2, 0.0)
plt.ylim(0.0, 1.2) 
plt.xlabel('Redshift') 
plt.ylabel('Stellar Mass Ratio to Thermal') 
plt.savefig('Mstar_ratio.png') 
print('Mstar_ratio.png') 
plt.close() 


baryons_vs_z(therm_table, zrange=[3, 0.0], title='Tempest Baryon Budget with Thermal Feedback', filename='baryon_budget_thermal.png')
baryons_vs_z(mech_table, zrange=[3, 0.0], title='Tempest Baryon Budget with Mechanical Feedback', filename='baryon_budget_mechanical.png')
baryons_vs_z(rad_table, zrange=[3, 0.0], title='Tempest Baryon Budget with 100x Rad & Thermal Feedback', filename='baryon_budget_rad100.png')
baryons_vs_z(mom5_table, zrange=[3, 0.0], title='Tempest Baryon Budget with Momemtum x5 Feedback', filename='baryon_budget_mom5x.png')
baryons_vs_z(mom_rad_table, zrange=[3, 0.0], title='Tempest Baryon Budget with Momemtum x3 + Rad100 Feedback', filename='baryon_budget_mom3x_rad.png')
baryons_vs_z(radius3_table, zrange=[3, 0.0], title='Tempest Baryon Budget with Momemtum x5 + Radius=3 ', filename='baryon_budget_mom5x_radius3.png')
baryons_vs_z(rad3_rad100_table, zrange=[3, 0.0], title='Tempest Baryon Budget with Mom x5,Radius=3,Rad100 ', filename='baryon_budget_mom5x_radius3_rad100.png')
baryons_vs_z(numerical_table, zrange=[3, 0.0], title='Tempest Baryon Budget with H2numerical nref11n ', filename='baryon_budget_H2numerical.png')
baryons_vs_z(pr63_table, zrange=[3, 0.0], title='Tempest Baryon Budget with PR63 Radiation', filename='baryon_budget_PR63.png')

