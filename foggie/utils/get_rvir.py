'''
Filename: get_rvir.py
Author: Raymond
Created: 01-16-19
Last modified:  01-16-19

This file calculates the virial radius (R200) and the total/dark/gas/star mass inside Rvir.
'''
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
import argparse
from foggie.utils.foggie_load import *




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


    parser.add_argument('--use_halo_c_v', dest='use_halo_c_v', action='store_true',
                        help='just use the pwd?, default is no')
    parser.set_defaults(use_halo_c_v=False)


    parser.add_argument('--output', metavar='output', type=str, action='store',
                        help='which output? default is RD0020')
    parser.set_defaults(output="RD0020")

    parser.add_argument('--do_fig', dest='do_fig', action='store_true',
                        help='just use the pwd?, default is no')
    parser.set_defaults(do_fig=False)


    parser.add_argument('--figdir', metavar='figdir', type=str, action='store',
                        help='which output? default is RD0020')
    parser.set_defaults(figdir='.')



    args = parser.parse_args()
    return args




def find_rvir(ds, halo_center = None, do_fig = False, sphere_radius = 250*kpc, figdir = '.', n_bins = 500):
    """Calculate rvir and M(<rvir), return results in a dictionary
    """
    from yt.units import kpc
    sp_find_rvir = ds.sphere(center = halo_center, radius = sphere_radius)
    filter_particles(sp_find_rvir)

    print ('measuring dm mass profile')
    prof_dm = yt.create_profile(sp_find_rvir, ('index', 'radius'), fields = [('deposit', 'dm_mass')], \
                                n_bins = n_bins, weight_field = None, accumulation = True)
    print ('measuring stars mass profile')
    prof_stars = yt.create_profile(sp_find_rvir, ('index', 'radius'), fields = [('deposit', 'stars_mass')], \
                                  n_bins = n_bins, weight_field = None, accumulation = True)
    print ('measuring gas mass profile')
    prof_gas     = yt.create_profile(sp_find_rvir, ('index', 'radius'), fields = [('gas', 'cell_mass')],\
                                     n_bins = n_bins, weight_field = None, accumulation = True)


    internal_density =  (prof_dm[('deposit', 'dm_mass')].to('g') + prof_stars[('deposit', 'stars_mass')].to('g') + prof_gas[('gas', 'cell_mass')].to('g'))/(4*np.pi*prof_dm.x.to('cm')**3./3.)

    rho_crit = cosmo.critical_density(ds.current_redshift)
    rvir = prof_dm.x[argmin(abs(internal_density.value - 200*rho_crit.value))]
    Mdm_rvir    = prof_dm[('deposit', 'dm_mass')][argmin(abs(internal_density.value - 200*rho_crit.value))]
    Mstars_rvir = prof_stars[('deposit', 'stars_mass')][argmin(abs(internal_density.value - 200*rho_crit.value))]
    Mgas_rvir   = prof_gas[('gas', 'cell_mass')][argmin(abs(internal_density.value - 200*rho_crit.value))]
    Mvir = Mdm_rvir + Mstars_rvir + Mgas_rvir

    res = {}
    res['rvir']        = rvir.to('kpc')
    res['Mvir']        = Mvir.to('Msun')
    res['Mgas_rvir']   = Mgas_rvir.to('Msun')
    res['Mdm_rvir']    = Mdm_rvir.to('Msun')
    res['Mstars_rvir'] = Mstars_rvir.to('Msun')


    if do_fig:

      plt.close('all')
      plt.ioff()
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

      ax.set_xlim(0, float(sphere_radius.value))
      ax2 = ax.twinx()

      ax2.plot(prof_dm.x.to('kpc'),  internal_density.to('g * cm**-3'), color = 'blue', linestyle = '-', linewidth = 3.5)


      ax2.yaxis.label.set_color('blue')
      ax2.spines['right'].set_color('blue')
      ax2.tick_params(axis='y', colors='blue')



      ax2.axvline(rvir.to('kpc'), color = 'darkblue', alpha = 0.4)
      ax2.axhline(y = 200*rho_crit.value, xmin = 0.0, xmax = 0.77, color = 'darkblue', alpha = 0.4)



      ax2.set_ylabel(r'$\rho$ ($<$r) (g cm$^{-3}$)')
      ax2.set_yscale('log')

      fs = 15
      ax2.set_ylim(1.e-30, 1.e-18)
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

      ax.scatter(rvir.to('kpc'), Mvir.to('Msun'), c = plot_colors['total'], s = 20)
      ax.scatter(rvir.to('kpc'), Mdm_rvir.to('Msun'), c = plot_colors['dark'], s = 20)
      ax.scatter(rvir.to('kpc'), Mgas_rvir.to('Msun'), c = plot_colors['gas'], s = 20)
      ax.scatter(rvir.to('kpc'), Mstars_rvir.to('Msun'), c = plot_colors['stars'], s = 20)

      ax2.plot(rvir.to('kpc'),200*rho_crit.value,  'x', color = 'b', markersize = 20)



      ax2.annotate(r'200 x $\rho_{crit}$', (float(sphere_radius.value) * 0.98, 200*rho_crit.value), ha = 'right', va = 'center',\
                   color = 'darkblue', fontsize = 18 )
      ax.set_xlabel('distance from central (kpc)')
      ax.set_ylabel(r'M ($<$ r) (M$_{\odot}$)')

      fig.tight_layout()
      halo =  ds.fullpath.split('/')[-3]
      run =  ds.fullpath.split('/')[-2]
      ddname =  ds.fullpath.split('/')[-1] 
      fig.savefig(figdir + '/%s_%s_%s_r200.png'%(halo, run, ddname), dpi = 300)

    return res



if __name__ == '__main__':


  args = parse_args()

  ds, refine_box = sim_load(args)
  res = find_rvir(ds, halo_center = ds.halo_center_kpc, do_fig = args.do_fig, figdir = args.figdir)

  print ('\t %s %s %s'%(args.halo, args.run, args.output))
  print ('\t Rvir (kpc) = %.1f'%(res['rvir'].to('kpc')))
  print ('\t Mvir (10^11 Msun) = %.3f'%(res['Mvir'].to('Msun')/1.e11))
  print ('\t Msta (10^11 Msun) = %.3f'%(res['Mstars_rvir'].to('Msun')/1.e11))
  print ('\t Mgas (10^11 Msun) = %.3f'%(res['Mgas_rvir'].to('Msun')/1.e11))
  print ('\t Mdar (10^11 Msun) = %.3f'%(res['Mdm_rvir'].to('Msun')/1.e11))
  print ('\t Mbary/Mdark = %.3f'%((res['Mgas_rvir'] + res['Mstars_rvir'])/res['Mdm_rvir']))
  print ('\t Mbary/Mtot  = %.3f'%((res['Mgas_rvir'] + res['Mstars_rvir'])/res['Mvir']))

































