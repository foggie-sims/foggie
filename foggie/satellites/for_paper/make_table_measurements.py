import glob
from glob import glob
import yt
from astropy.io import ascii
from yt.units import kpc
import foggie
from foggie.utils.foggie_utils import filter_particles
import matplotlib.pyplot as plt
from astropy.cosmology import Planck15 as cosmo
import numpy as np
from numpy import *
import joblib
from joblib import Parallel, delayed



plt.close('all')
plt.ioff()

simtype = 'nref11c_nref9f'





def find_rvir(halo, DDname):
    print (halo)
    sim_dir = '/Users/rsimons/Desktop/foggie/sims'

    sat_cat = ascii.read('/Users/rsimons/Dropbox/foggie/catalogs/satellite_properties.cat')

    sat_cat_halo = sat_cat[sat_cat['halo'] == int(halo)]
    sat_center = sat_cat_halo[sat_cat_halo['id'] == '0']


    fl = glob(sim_dir + '/halo_00%s/%s/%s/%s'%(halo, simtype, DDname, DDname))[0]
    ds = yt.load(fl)

    ad = ds.all_data()
    center = yt.YTArray([sat_center['x'][0], sat_center['y'][0], sat_center['z'][0]], 'kpc')

    sp_find_rvir = ds.sphere(center = center, radius = 200*kpc)
    filter_particles(sp_find_rvir)
    prof = yt.create_profile(sp_find_rvir, ('dm', 'particle_radius'), fields = [('dm', 'particle_mass')], n_bins = 100, weight_field = None, accumulation = True)


    fig, ax = plt.subplots(1,1, figsize = (7,7))


    ax.plot(prof.x.to('kpc'), prof[('dm', 'particle_mass')].to('Msun'), 'k-')

    ax2 = ax.twinx()

    internal_density =  prof[('dm', 'particle_mass')].to('g')/(4*np.pi*prof.x.to('cm')**3./3.)
    ax2.plot(prof.x.to('kpc'),  internal_density.to('g * cm**-3'), 'b-')


    ax2.yaxis.label.set_color('blue')
    ax2.spines['right'].set_color('blue')
    ax2.tick_params(axis='y', colors='blue')
    rho_crit = cosmo.critical_density(ds.current_redshift)

    rvir = prof.x[argmin(abs(internal_density.value - 200*rho_crit.value))]
    Mvir = prof[('dm', 'particle_mass')][argmin(abs(internal_density.value - 200*rho_crit.value))]

    ax2.axvline(rvir.to('kpc'), color = 'darkblue')


    ax2.axhline(y = 200*rho_crit.value, color = 'darkblue', linestyle = '--')

    ax2.set_ylabel(r'$\rho_{DM}$ ($<$r) (g cm$^{-3}$)')
    ax2.set_yscale('log')

    ax2.annotate(r'R$_{200}$ = %.1f kpc'%(rvir.to('kpc')), (rvir.to('kpc').value * 1.05, 200*rho_crit.value * 20.), ha = 'left', va = 'bottom',\
                 color = 'darkblue', fontsize = 20 )
    ax2.annotate(r'M$_{200}$ = %.2f x $10^{12}$ M$_{\odot}$'%(Mvir.to('Msun')/1.e12), (rvir.to('kpc').value * 1.05, 200*rho_crit.value * 20.), ha = 'left', va = 'top', \
                color = 'black', fontsize = 20)



    ax2.annotate(r'200 x $\rho_{crit}$', (175, 200*rho_crit.value * 1.05), ha = 'right', va = 'bottom',\
                 color = 'darkblue', fontsize = 20 )
    ax.set_xlabel('distance from central (kpc)')
    ax.set_ylabel(r'M$_{dm}$ ($<$ r) (M$_{\odot}$)')

    fig.savefig('/Users/rsimons/Dropbox/foggie/figures/virial_radii/%s_%.2f_virial_radius.png'%(halo, ds.current_redshift), dpi = 300)


    '''
    sp2 = ds.sphere(center = center, radius = 20*kpc)
    sp_cold = sp2.cut_region(["(obj['temperature'] < {} )".format(1.5e4)])



    prof_all_gas = yt.create_profile(sp2, ['radius'], fields = [('gas', 'cell_mass')], n_bins = n_bins, weight_field = None, accumulation = True)
    print (halo, 'all_gas (<20 kpc) = ', '%.4f x 1.e10 Msun'%(prof_all_gas['gas', 'cell_mass'][-1].to('Msun')/(1.e10)))

    prof_cold = yt.create_profile(sp_cold, ['radius'], fields = [('gas', 'cell_mass')], n_bins = n_bins, weight_field = None, accumulation = True)
    print (halo,'cold_gas (<20 kpc) = ', '%.4f x 1.e10 Msun'%(prof_cold['gas', 'cell_mass'][-1].to('Msun')/(1.e10)))    
    '''











def find_masses(halo, DDname, rvir):
    print (halo)
    sim_dir = '/Users/rsimons/Desktop/foggie/sims'

    sat_cat = ascii.read('/Users/rsimons/Dropbox/foggie/catalogs/satellite_properties.cat')

    sat_cat_halo = sat_cat[sat_cat['halo'] == int(halo)]
    sat_center = sat_cat_halo[sat_cat_halo['id'] == '0']


    fl = glob(sim_dir + '/halo_00%s/%s/%s/%s'%(halo, simtype, DDname, DDname))[0]
    ds = yt.load(fl)

    ad = ds.all_data()
    center = yt.YTArray([sat_center['x'][0], sat_center['y'][0], sat_center['z'][0]], 'kpc')

    filter_particles(sp_find_rvir)


    sp_find_galmasses = ds.sphere(center = center, radius = 20*kpc)
    sp_find_galmasses = sp_find_galmasses.cut_region(["(obj['temperature'] < {} )".format(1.5e4)])

    sp_find_cgmmasses = ds.sphere(center = center, radius = rvir*kpc)
    sp_find_cgmmasses = sp_find_cgmmasses.cut_region(["(obj['temperature'] > {} )".format(1.5e4)])
    

    cold_gas_mass, star_mass = sp_find_galmasses.quantities.total_quantity(["cell_mass", ("stars", "particle_mass")])
    cgm_gas_mass = sp_find_cgmmasses.quantities.total_quantity(["cell_mass"])


    print (halo, '%.3f %.3f %.3f'%(cold_gas_mass.to('Msun'), cgm_gas_mass.to('Msun'), star_mass.to('Msun')))



inputs = [('2392', 'DD0581'),
          ('2878', 'DD0581'), 
          ('4123', 'DD0581'),
          ('5016', 'DD0581'), 
          ('5036', 'DD0581'),
          ('8508', 'DD0487')]


#Parallel(n_jobs = -1, backend = 'threading')(delayed(find_rvir) (halo = halo, DDname = DDname) for (halo, DDname) in inputs)



inputs_wrvi = [('2392', 'DD0581',71.7),
               ('2878', 'DD0581',88.4), 
               ('4123', 'DD0581',77.2),
               ('5016', 'DD0581',51.8), 
               ('5036', 'DD0581',68.5),
               ('8508', 'DD0487',48.5)]





def find_masses(halo, DDname, rvir):
    print (halo)
    sim_dir = '/Users/rsimons/Desktop/foggie/sims'

    sat_cat = ascii.read('/Users/rsimons/Dropbox/foggie/catalogs/satellite_properties.cat')

    sat_cat_halo = sat_cat[sat_cat['halo'] == int(halo)]
    sat_center = sat_cat_halo[sat_cat_halo['id'] == '0']


    fl = glob(sim_dir + '/halo_00%s/%s/%s/%s'%(halo, simtype, DDname, DDname))[0]
    ds = yt.load(fl)

    ad = ds.all_data()
    center = yt.YTArray([sat_center['x'][0], sat_center['y'][0], sat_center['z'][0]], 'kpc')



    sp_find_galmasses = ds.sphere(center = center, radius = 20*kpc)
    filter_particles(sp_find_galmasses)



    sp_find_galmasses_cold = sp_find_galmasses.cut_region(["(obj['temperature'] < {} )".format(1.5e4)])
    

    sp_find_cgmmasses = ds.sphere(center = center, radius = rvir*kpc)
    sp_find_cgmmasses = sp_find_cgmmasses - sp_find_galmasses_cold
    for sat_row in sat_cat_halo:
          if sat_row['id'] == '0':
              x_center = sat_row['x']
              y_center = sat_row['y']
              z_center = sat_row['z']
              continue

          x_sat = sat_row['x']
          y_sat = sat_row['y']
          z_sat = sat_row['z']
          
          dist = np.sqrt((x_sat - x_center)**2. + (y_sat - y_center)**2. + (z_sat - z_center)**2.)
          if dist > rvir: continue
          center_sat = yt.YTArray([x_sat, y_sat, z_sat], 'kpc')
          sp_sat =  ds.sphere(center = center_sat, radius = 4*kpc)
          sp_sat = sp_sat.cut_region(["(obj['temperature'] > {} )".format(1.5e4)])
          sp_find_cgmmasses = sp_find_cgmmasses - sp_sat




    cold_gas_mass = sp_find_galmasses_cold.quantities.total_quantity(["cell_mass"])
    star_mass     = sp_find_galmasses.quantities.total_quantity([("deposit", "stars_mass")])
    cgm_gas_mass  = sp_find_cgmmasses.quantities.total_quantity(["cell_mass"])





    print ('masses: ', halo, '%.2f %.2f %.2f'%(star_mass.to('Msun')/(1.e10), cold_gas_mass.to('Msun')/(1.e10), cgm_gas_mass.to('Msun')/(1.e10)))


Parallel(n_jobs = 1, backend = 'threading')(delayed(find_masses) (halo = halo, DDname = DDname, rvir = rvir) for (halo, DDname, rvir) in inputs_wrvi)























