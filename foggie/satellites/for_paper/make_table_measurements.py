#Make measurements for satellites paper

import glob
from glob import glob
import yt
from astropy.io import ascii
from yt.units import kpc
from foggie.utils.get_run_loc_etc import get_run_loc_etc
import foggie
from foggie.utils.foggie_utils import filter_particles
import matplotlib.pyplot as plt
from astropy.cosmology import Planck15 as cosmo
import numpy as np
from numpy import *
import joblib
from joblib import Parallel, delayed
import argparse



plt.close('all')
plt.ioff()

simtype = 'nref11c_nref9f'





def find_rvir(ds, halo_center = None, do_fig = False, figdir = '.'):

    from yt.units import kpc
    sp_find_rvir = ds.sphere(center = halo_center, radius = 200*kpc)
    filter_particles(sp_find_rvir)

    print ('measuring dm mass profile')
    prof_dm = yt.create_profile(sp_find_rvir, ('index', 'radius'), fields = [('deposit', 'dm_mass')],  n_bins = 200, weight_field = None, accumulation = True)
    print ('measuring stars mass profile')
    prof_stars = yt.create_profile(sp_find_rvir, ('index', 'radius'), fields = [('deposit', 'stars_mass')],  n_bins = 200, weight_field = None, accumulation = True)
    print ('measuring gas mass profile')
    prof_gas     = yt.create_profile(sp_find_rvir, ('index', 'radius'), fields = [('gas', 'cell_mass')],   n_bins = 200, weight_field = None, accumulation = True)


    internal_density =  (prof_dm[('deposit', 'dm_mass')].to('g') + prof_stars[('deposit', 'stars_mass')].to('g') + prof_gas[('gas', 'cell_mass')].to('g'))/(4*np.pi*prof_dm.x.to('cm')**3./3.)

    rho_crit = cosmo.critical_density(ds.current_redshift)
    rvir = prof_dm.x[argmin(abs(internal_density.value - 200*rho_crit.value))]
    Mdm_rvir    = prof_dm[('deposit', 'dm_mass')][argmin(abs(internal_density.value - 200*rho_crit.value))]
    Mstars_rvir = prof_stars[('deposit', 'stars_mass')][argmin(abs(internal_density.value - 200*rho_crit.value))]
    Mgas_rvir   = prof_gas[('gas', 'cell_mass')][argmin(abs(internal_density.value - 200*rho_crit.value))]
    Mvir = Mdm_rvir + Mstars_rvir + Mgas_rvir

    res = {}
    res['rvir'] = rvir.to('kpc')
    res['Mvir'] = Mvir.to('Msun')
    res['Mgas_rvir'] = Mgas_rvir.to('Msun')
    res['Mdm_rvir'] = Mdm_rvir.to('Msun')
    res['Mstars_rvir'] = Mstars_rvir.to('Msun')


    if do_fig:
      plot_colors = {}
      plot_colors['total']  = 'black'
      plot_colors['dark']   = 'grey'
      plot_colors['stars']  = 'goldenrod'
      plot_colors['gas']    = 'green'


      fig, ax = plt.subplots(1,1, figsize = (7,7))

      #plot total mass profile
      ax.plot(prof_dm.x.to('kpc'), prof_dm[('deposit', 'dm_mass')].to('Msun') + prof_stars[('deposit', 'stars_mass')].to('Msun') +  prof_gas[('gas', 'cell_mass')].to('Msun'), \
              color = plot_colors['total'], linewidth = 3.5, linestyle = '-')

      #plot component mass profiles
      ax.plot(prof_dm.x.to('kpc'), prof_dm[('deposit', 'dm_mass')].to('Msun'),\
              color = plot_colors['dark'], linewidth = 1.5,linestyle = '--')
      ax.plot(prof_stars.x.to('kpc'), prof_stars[('deposit', 'stars_mass')].to('Msun'),\
              color = plot_colors['stars'], linewidth = 1.5,linestyle = '--')
      ax.plot(prof_gas.x.to('kpc'), prof_gas[('gas', 'cell_mass')].to('Msun'),\
              color = plot_colors['gas'], linewidth = 1.5, linestyle = '--')

      ax2 = ax.twinx()

      ax2.plot(prof_dm.x.to('kpc'),  internal_density.to('g * cm**-3'), color = 'blue', linestyle = '-', linewidth = 3.5)


      ax2.yaxis.label.set_color('blue')
      ax2.spines['right'].set_color('blue')
      ax2.tick_params(axis='y', colors='blue')



      ax2.axvline(rvir.to('kpc'), color = 'darkblue', alpha = 0.4)
      ax2.axhline(y = 200*rho_crit.value, color = 'darkblue', alpha = 0.4)



      ax2.set_ylabel(r'$\rho$ ($<$r) (g cm$^{-3}$)')
      ax2.set_yscale('log')

      fs = 15
      ax2.annotate(r'R$_{200}$ = %.1f kpc'%(rvir.to('kpc')), (0.98, 0.6),  xycoords = 'axes fraction', ha = 'right', va = 'top',\
                   color = plot_colors['total'], fontsize = fs )
      ax2.annotate(r'M$_{200, tot}$ = %.2f x $10^{11}$ M$_{\odot}$'%(Mvir.to('Msun')/1.e11), (0.98, 0.56), xycoords = 'axes fraction', ha = 'right', va = 'top', \
                  color = plot_colors['total'], fontsize = fs)
      ax2.annotate(r'M$_{200, dark}$ = %.2f x $10^{11}$ M$_{\odot}$'%(Mdm_rvir.to('Msun')/1.e11), (0.98, 0.52), xycoords = 'axes fraction', ha = 'right', va = 'top', \
                  color = plot_colors['dark'], fontsize = fs)
      ax2.annotate(r'M$_{200, stars}$ = %.2f x $10^{11}$ M$_{\odot}$'%(Mstars_rvir.to('Msun')/1.e11),(0.98, 0.48), xycoords = 'axes fraction',  ha = 'right', va = 'top', \
                  color = plot_colors['stars'], fontsize = fs)
      ax2.annotate(r'M$_{200, gas}$ = %.2f x $10^{11}$ M$_{\odot}$'%(Mgas_rvir.to('Msun')/1.e11), (0.98, 0.44),  xycoords = 'axes fraction', ha = 'right', va = 'top', \
                  color = plot_colors['gas'], fontsize = fs)



      ax2.annotate(r'200 x $\rho_{crit}$', (198, 200*rho_crit.value * 1.04), ha = 'right', va = 'bottom',\
                   color = 'darkblue', fontsize = 20 )
      ax.set_xlabel('distance from central (kpc)')
      ax.set_ylabel(r'M ($<$ r) (M$_{\odot}$)')

      fig.tight_layout()
      halo =  ds.fullpath.split('/')[-3]
      run =  ds.fullpath.split('/')[-2]
      ddname =  ds.fullpath.split('/')[-1] 
      fig.savefig(figdir + '/%s_%s_%s_r200.png'%(halo, run, ddname), dpi = 300)

    return res

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



def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='''identify satellites in FOGGIE simulations''')
    parser.add_argument('-system', '--system', metavar='system', type=str, action='store', \
                        help='Which system are you on? Default is Jase')
    parser.set_defaults(system="jase")

    parser.add_argument('--run', metavar='run', type=str, action='store',
                        help='which run? default is natural')
    parser.set_defaults(run="nref11c_nref9f")

    parser.add_argument('--halo', metavar='halo', type=str, action='store',
                        help='which halo? default is 8508 (Tempest)')
    parser.set_defaults(halo="8508")

    parser.add_argument('--pwd', dest='pwd', action='store_true',
                        help='just use the pwd?, default is no')
    parser.set_defaults(pwd=False)

    parser.add_argument('--run_all', dest='run_all', action='store_true',
                        help='just use the pwd?, default is no')
    parser.set_defaults(pwd=False)

    parser.add_argument('--output', metavar='output', type=str, action='store',
                        help='which output? default is RD0020')
    parser.set_defaults(output="RD0020")


    args = parser.parse_args()
    return args



if __name__ == '__main__':


  args = parse_args()

  inputs = [('2392', 'DD0581', 'nref11c_nref9f'),
            ('2878', 'DD0581', 'nref11c_nref9f'), 
            ('4123', 'DD0581', 'nref11c_nref9f'),
            ('5016', 'DD0581', 'nref11c_nref9f'), 
            ('5036', 'DD0581', 'nref11c_nref9f'),
            ('8508', 'DD0487', 'nref11c_nref9f')]

  inputs = inputs[:1]

  '''
  for (args.halo, args.output, args.run) in inputs:
    foggie_dir, output_dir, run_loc, trackname, haloname, spectra_dir, infofile = get_run_loc_etc(args)

    run_dir = foggie_dir + run_loc
    run_dir = run_dir.replace('nref11n', 'natural')

    ds_loc = run_dir + args.output + "/" + args.output
    ds = yt.load(ds_loc)
    figdir = '/Users/rsimons/Dropbox/foggie/figures/virial_radii'


    sat_cat = ascii.read('/Users/rsimons/Dropbox/foggie/catalogs/satellite_properties.cat')
    sat_cat_halo = sat_cat[sat_cat['halo'] == int(args.halo)]
    sat_center = sat_cat_halo[sat_cat_halo['id'] == '0']
    cenx = sat_center['x'][0]
    ceny = sat_center['y'][0]
    cenz = sat_center['z'][0]
    halo_center = yt.YTArray([cenx, ceny, cenz], 'kpc')



    res = find_rvir(ds, halo_center = halo_center, do_fig = True, figdir = figdir)

    print ('\t %s %s %s'%(args.halo, args.run, args.output))
    print ('\t Rvir (kpc) = %.1f'%(res['rvir'].to('kpc')))
    print ('\t Mvir (10^11 Msun) = %.3f'%(res['Mvir'].to('Msun')/1.e11))
    print ('\t Msta (10^11 Msun) = %.3f'%(res['Mstars_rvir'].to('Msun')/1.e11))
    print ('\t Mgas (10^11 Msun) = %.3f'%(res['Mgas_rvir'].to('Msun')/1.e11))
    print ('\t Mdar (10^11 Msun) = %.3f'%(res['Mdm_rvir'].to('Msun')/1.e11))
    print ('\t Mbary/Mdark = %.3f'%((res['Mgas_rvir'] + res['Mstars_rvir'])/res['Mdm_rvir']))
    print ('\t Mbary/Mtot  = %.3f'%((res['Mgas_rvir'] + res['Mstars_rvir'])/res['Mvir']))

  #Parallel(n_jobs = -1, backend = 'threading')(delayed(find_rvir) (halo = halo, DDname = DDname) for (halo, DDname) in inputs)
  '''







  inputs_wrvi = [('2392', 'DD0581',82.2),
                 ('2878', 'DD0581',91.5), 
                 ('4123', 'DD0581',77.2),
                 ('5016', 'DD0581',55.3), 
                 ('5036', 'DD0581',69.9),
                 ('8508', 'DD0487',51.8)]





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























