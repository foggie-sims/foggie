'''
Filename: get_rvir.py
Author: Raymond
Created: 01-16-19
Last modified:  04-23-19

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
from scipy.interpolate import interp1d 



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

    parser.add_argument('--use_catalog_profile', dest='use_catalog_profile', action='store_true',
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
    parser.set_defaults(figdir='/Users/rsimons/Dropbox/foggie/figures/rvir_check')



    args = parser.parse_args()
    return args


def make_fig(args, masses_trim, internal_density, rho_crit, data):
    plt.close('all')
    plt.ioff()
    fig, ax = plt.subplots(1,1, figsize = (7,7))
    if len(data) > 1:
      rvir = data['radius'][-1]
      Mvir = data['total_mass'][-1]
      Mdm_rvir = data['dm_mass'][-1]
      Mgas_rvir = data['gas_mass'][-1]
      Mstars_rvir = data['young_stars_mass'][-1]
    else:
      rvir = data['radius']
      Mvir = data['total_mass']
      Mdm_rvir = data['dm_mass']
      Mgas_rvir = data['gas_mass']
      Mstars_rvir = data['young_stars_mass']


    plot_colors = {}
    plot_colors['total']  = 'black'
    plot_colors['dark']   = 'grey'
    plot_colors['stars']  = 'goldenrod'
    plot_colors['gas']    = 'green'



    #plot total mass profile
    ax.plot(masses_trim['radius'], masses_trim['total_mass'], \
            color = plot_colors['total'], linewidth = 3.5, linestyle = '-')
    #plot component mass profiles
    ax.plot(masses_trim['radius'], masses_trim['dm_mass'],\
            color = plot_colors['dark'], linewidth = 1.5,linestyle = '--')
    ax.plot(masses_trim['radius'], masses_trim['stars_mass'],\
            color = plot_colors['stars'], linewidth = 1.5,linestyle = '--')
    ax.plot(masses_trim['radius'], masses_trim['gas_mass'],\
            color = plot_colors['gas'], linewidth = 1.5, linestyle = '--')

    ax.set_xlim(0, 200.)
    ax2 = ax.twinx()

    ax2.plot(masses_trim['radius'],  internal_density, color = 'blue', linestyle = '-', linewidth = 3.5)


    ax2.yaxis.label.set_color('blue')
    ax2.spines['right'].set_color('blue')
    ax2.tick_params(axis='y', colors='blue')



    ax2.axvline(rvir, color = 'darkblue', alpha = 0.4)
    ax2.axhline(y = 200*rho_crit.value, xmin = 0.0, xmax = 0.77, color = 'darkblue', alpha = 0.4)



    ax2.set_ylabel(r'$\rho$ ($<$r) (g cm$^{-3}$)')
    ax2.set_yscale('log')

    fs = 15
    ax2.set_ylim(1.e-30, 1.e-18)



    ax2.annotate(r'z = %.2f'%(data['redshift'][-1]), (0.98, 0.98),  xycoords = 'axes fraction', ha = 'right', va = 'top',\
                 color = plot_colors['total'], fontsize = fs )
    ax2.annotate(r'%s'%(data['snapshot'][-1]), (0.98, 0.94),  xycoords = 'axes fraction', ha = 'right', va = 'top',\
                 color = plot_colors['total'], fontsize = fs )

    ax2.annotate(r'R$_{200}$ = %.1f kpc'%(rvir), (0.98, 0.90),  xycoords = 'axes fraction', ha = 'right', va = 'top',\
                 color = plot_colors['total'], fontsize = fs )
    ax2.annotate(r'M$_{200, tot}$ = %.4f x $10^{11}$ M$_{\odot}$'%(Mvir/1.e11), (0.98, 0.86), xycoords = 'axes fraction', ha = 'right', va = 'top', \
                color = plot_colors['total'], fontsize = fs)
    ax2.annotate(r'M$_{200, dark}$ = %.4f x $10^{11}$ M$_{\odot}$'%(Mdm_rvir/1.e11), (0.98, 0.82), xycoords = 'axes fraction', ha = 'right', va = 'top', \
                color = plot_colors['dark'], fontsize = fs)
    ax2.annotate(r'M$_{200, stars}$ = %.4f x $10^{11}$ M$_{\odot}$'%(Mstars_rvir/1.e11),(0.98, 0.78), xycoords = 'axes fraction',  ha = 'right', va = 'top', \
                color = plot_colors['stars'], fontsize = fs)
    ax2.annotate(r'M$_{200, gas}$ = %.4f x $10^{11}$ M$_{\odot}$'%(Mgas_rvir/1.e11), (0.98, 0.74),  xycoords = 'axes fraction', ha = 'right', va = 'top', \
                color = plot_colors['gas'], fontsize = fs)






    ax.scatter(rvir, Mvir, c = plot_colors['total'], s = 20)
    ax.scatter(rvir, Mdm_rvir, c = plot_colors['dark'], s = 20)
    ax.scatter(rvir, Mgas_rvir, c = plot_colors['gas'], s = 20)
    ax.scatter(rvir, Mstars_rvir, c = plot_colors['stars'], s = 20)

    ax2.plot(rvir,200*rho_crit,  'x', color = 'b', markersize = 20)



    ax2.annotate(r'200 x $\rho_{crit}$', (148, 200*rho_crit.value), ha = 'right', va = 'center',\
                 color = 'darkblue', fontsize = 18 )

    ax.set_xlabel('distance from central (kpc)')
    ax.set_ylabel(r'M ($<$ r) (M$_{\odot}$)')

    fig.tight_layout()
    fig.savefig('%s/%s_%s_%s_r200.png'%(args.figdir, args.halo, args.run, args.output), dpi = 300)


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


    return res



def find_rvir_catalogs(args, data, halo_infos_dir, figdir = '.'):

    from astropy.table import Table
    masses = Table.read('%s/masses_z-gtr-2.hdf5'%halo_infos_dir)
    gd = where(masses['snapshot'] == args.output)[0]
    if len(gd) == 0:
      #snapshot less than 2
      masses = Table.read('%s/masses_z-less-2.hdf5'%halo_infos_dir)
      gd = where(masses['snapshot'] == args.output)[0]


    radius  = masses['radius'][gd]
    total_mass = masses['total_mass'][gd]
    redshift = masses['redshift'][gd][0]

    internal_density =  total_mass.to('g')/(4*np.pi*radius.to('cm')**3./3.)
    rho_crit = cosmo.critical_density(redshift)
    #argrvir = argmin(abs(internal_density - 200*rho_crit))
    interp_fnc = interp1d(internal_density, radius)

    rvir = interp_fnc(200*rho_crit)#radius[argrvir] * radius_unit
    res = []
    for key in masses.keys():
      if (key == 'redshift') | (key == 'snapshot'):
        res.append(masses[key][gd][0])
      else:
        interp_fnc = interp1d(radius, masses[key][gd])
        res.append(interp_fnc(rvir))

    data.add_row(res)

    if args.do_fig: make_fig(args, masses[gd], internal_density, rho_crit, data)


    return data

    



if __name__ == '__main__':


  args = parse_args()


  if args.use_catalog_profile:
    _, _, _, code_path, _, _, _, _ = get_run_loc_etc(args)
    halo_infos_dir = code_path + '/halo_infos/00%s/%s'%(args.halo, args.run)
    data = Table(names=('redshift', 'snapshot', 'radius', 'total_mass', 'dm_mass', \
                        'stars_mass', 'young_stars_mass', 'old_stars_mass', 'sfr', 'gas_mass', \
                        'gas_metal_mass', 'gas_H_mass', 'gas_HI_mass', 'gas_HII_mass', 'gas_CII_mass', \
                        'gas_CIII_mass', 'gas_CIV_mass', 'gas_OVI_mass', 'gas_OVII_mass', 'gas_MgII_mass', \
                        'gas_SiII_mass', 'gas_SiIII_mass', 'gas_SiIV_mass', 'gas_NeVIII_mass'), \
                 dtype=('f8', 'S6', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8'))
    from foggie.utils.get_mass_profile import set_table_units

    data = set_table_units(data)


  if args.use_catalog_profile:
    DDoutputs = ['DD%.4i'%i for i in np.arange(44, 2428,1)]
    masses1 = Table.read('%s/masses_z-gtr-2.hdf5'%halo_infos_dir)
    masses2 = Table.read('%s/masses_z-less-2.hdf5'%halo_infos_dir)
    list1 = list(masses1['snapshot'])
    list2 = list(masses2['snapshot'])
    full_list = np.array(list1 +   list2)
    
    outputs = unique(full_list)
    for args.output in outputs:
        if 'DD' in args.output:
          if float(args.output.strip('DD'))%200 == 0:  args.do_fig = True
          else: args.do_fig = False
        else: 
          args.do_fig = True


        data = find_rvir_catalogs(args, data, halo_infos_dir)
        print ('\t %s %s %s Rvir (kpc) = %.2f'%(args.halo, args.run, args.output, data['radius'][-1]))

    data.write(halo_infos_dir + '/rvir_masses.hdf5', path='all_data', serialize_meta=True, overwrite=True)
  



  else:
    ds, refine_box = sim_load(args)
    res = find_rvir(ds, halo_center = ds.halo_center_kpc, do_fig = args.do_fig, figdir = args.figdir)


  
    if False:
      print ('\t Rvir (kpc) = %.2f'%(res['rvir'].to('kpc').value))
      print ('\t Mvir (10^11 Msun) = %.3f'%(res['Mvir'].to('Msun').value/1.e11))
      print ('\t Msta (10^11 Msun) = %.3f'%(res['Mstars_rvir'].to('Msun').value/1.e11))
      print ('\t Mgas (10^11 Msun) = %.3f'%(res['Mgas_rvir'].to('Msun').value/1.e11))
      print ('\t Mdar (10^11 Msun) = %.3f'%(res['Mdm_rvir'].to('Msun').value/1.e11))
      print ('\t Mbary/Mdark = %.3f'%((res['Mgas_rvir'] + res['Mstars_rvir'])/res['Mdm_rvir']))
      print ('\t Mbary/Mtot  = %.3f'%((res['Mgas_rvir'] + res['Mstars_rvir'])/res['Mvir']))
     































