'''
Create sightlines through FOGGIE halo
author: Raymond C. Simons
'''





import yt
from get_halo_center import get_halo_center
import numpy as np
from astropy.cosmology import Planck13 as cosmo
import numpy as np
from numpy import *
from astropy import constants as c
import matplotlib
import matplotlib.pyplot as plt
import os, sys, argparse

plt.ioff()
plt.close('all')


def parse():
    '''
    Parse command line arguments
    ''' 
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='''\
                                Generate the cameras to use in Sunrise and make projection plots
                                of the data for some of these cameras. Then export the data within
                                the fov to a FITS file in a format that Sunrise understands.
                                ''')
    parser.add_argument('-simdir', '--simdir', default='/nobackupp2/mpeeples', help='simulation output directory')
    parser.add_argument('-simname', '--simname', default=None, help='Simulation to be analyzed.')
    parser.add_argument('-haloname', '--haloname', default='halo_008508', help='halo_name')
    parser.add_argument('-DD', '--DD', default=None, help='DD')



    args = vars(parser.parse_args())
    return args


def vmax_profile(ds, DDname, center, start_rad = 5, end_rad = 220., delta_rad = 5):

    rs = np.arange(start_rad, end_rad, delta_rad)
    r_arr = zeros(len(rs))
    m_arr = zeros(len(r_arr))

    for rr, r in enumerate(rs):
        print (rr, r)
        r0 = ds.arr(r, 'kpc')
        r_arr[rr] = r0
        critical_density = cosmo.critical_density(ds.current_redshift).value
        r0 =  ds.arr(delta_rad, 'kpc')
        v_sphere = ds.sphere(center, r0)
        cell_mass, particle_mass = v_sphere.quantities.total_quantity(["cell_mass", "particle_mass"])

        m_arr[rr] = cell_mass.in_units('Msun') + particle_mass.in_units('Msun')


    m_arr = yt.YTArray(m_arr, 'Msun')
    r_arr = yt.YTArray(r_arr, 'kpc')
    to_save = {}
    to_save['m'] = m_arr
    to_save['r'] = r_arr
    G = yt.YTArray([c.G.value], 'm**3/kg/s**2') 
    to_save['v'] = sqrt(2 * G * m_arr/r_arr).to('km/s')

    np.save('/nobackupp2/rcsimons/foggie_momentum/catalogs/vescape/%s_%s_vescape.npy'%(DDname, simname), to_save)



if __name__ == '__main__':
    args = parse()
    haloname = args['haloname']
    simname = args['simname']
    simdir = args['simdir']
    DD = args['DD']
    DDname = 'DD%s'%DD
    if simname == 'natural':      enzo_simname = 'natural'
    elif simname == 'natural_v2': enzo_simname = 'nref11n_v2_selfshield_z15'
    elif simname == 'natural_v3': enzo_simname = 'nref11n_v3_selfshield_z15'
    elif simname == 'natural_v4': enzo_simname = 'nref11n_v4_selfshield_z15'
    else: enzo_simname = simname

    if 'natural' in simname: interp_name = 'natural'
    else: interp_name = simname



    if True:
        ds = yt.load('%s/%s/%s/%s/%s'%(simdir, haloname, enzo_simname,  DDname, DDname))


        cen_fits = np.load('/nobackupp2/rcsimons/foggie_momentum/catalogs/sat_interpolations/%s_interpolations_DD0150_new.npy'%interp_name, allow_pickle = True)[()]

        central_x = cen_fits['CENTRAL']['fxe'](DD)
        central_y = cen_fits['CENTRAL']['fye'](DD)
        central_z = cen_fits['CENTRAL']['fze'](DD)


        cen_central = yt.YTArray([central_x, central_y, central_z], 'kpc')
        v_sphere = ds.sphere(cen_central, (100, 'kpc'))  
        cen_bulkv = v_sphere.quantities.bulk_velocity().to('km/s') 
    if True: vmax_profile(ds, DDname, cen_central)



    if True:
        ray_l = 400
        ray_w = 10

        for aa, axs in enumerate(['x', 'y', 'z']):
            for i in np.arange(2):
                if i == 0:
                    if axs == 'x':
                        box = ds.r[cen_central[0] - 0.5 * yt.YTArray(ray_l, 'kpc'):   cen_central[0], \
                                   cen_central[1] - 0.5 * yt.YTArray(ray_w,  'kpc'):  cen_central[1] + 0.5 * yt.YTArray(ray_l,  'kpc') , \
                                   cen_central[2] - 0.5 * yt.YTArray(ray_w,  'kpc'):  cen_central[2] + 0.5 * yt.YTArray(ray_w,  'kpc')]
                        ax_plot = 'y'

                    if axs == 'y':
                        box = ds.r[cen_central[0] - 0.5 * yt.YTArray(ray_w, 'kpc'):   cen_central[0] + 0.5 * yt.YTArray(ray_w, 'kpc'), \
                                   cen_central[1] - 0.5 * yt.YTArray(ray_l,  'kpc'):  cen_central[1], \
                                   cen_central[2] - 0.5 * yt.YTArray(ray_w,  'kpc'):  cen_central[2] + 0.5 * yt.YTArray(ray_w,  'kpc')]
                        ax_plot = 'z'

                    if axs == 'z':
                        box = ds.r[cen_central[0] - 0.5 * yt.YTArray(ray_w, 'kpc'):   cen_central[0] + 0.5 * yt.YTArray(ray_w, 'kpc'), \
                                   cen_central[1] - 0.5 * yt.YTArray(ray_w,  'kpc'):  cen_central[1] + 0.5 * yt.YTArray(ray_w,  'kpc'), \
                                   cen_central[2] - 0.5 * yt.YTArray(ray_l,  'kpc'):  cen_central[2]]
                        ax_plot = 'y'


                if i == 1:
                    if axs == 'x':
                        box = ds.r[cen_central[0]:   cen_central[0] + 0.5 * yt.YTArray(ray_l, 'kpc'), \
                                   cen_central[1] - 0.5 * yt.YTArray(ray_w,  'kpc'):  cen_central[1] + 0.5 * yt.YTArray(ray_w,  'kpc'), \
                                   cen_central[2] - 0.5 * yt.YTArray(ray_w,  'kpc'):  cen_central[2] + 0.5 * yt.YTArray(ray_w,  'kpc')]
                        ax_plot = 'y'

                    if axs == 'y':
                        box = ds.r[cen_central[0] - 0.5 * yt.YTArray(ray_w, 'kpc'):   cen_central[0] + 0.5 * yt.YTArray(ray_w, 'kpc'), \
                                   cen_central[1]:  cen_central[1] + 0.5 * yt.YTArray(ray_l,  'kpc'), \
                                   cen_central[2] - 0.5 * yt.YTArray(ray_w,  'kpc'):  cen_central[2] + 0.5 * yt.YTArray(ray_w,  'kpc')]
                        ax_plot = 'z'

                    if axs == 'z':
                        box = ds.r[cen_central[0] - 0.5 * yt.YTArray(ray_w, 'kpc'):   cen_central[0] + 0.5 * yt.YTArray(ray_w, 'kpc'), \
                                   cen_central[1] - 0.5 * yt.YTArray(ray_w,  'kpc'):  cen_central[1] + 0.5 * yt.YTArray(ray_w,  'kpc'), \
                                   cen_central[2]:  cen_central[2] + 0.5 * yt.YTArray(ray_l,  'kpc')]
                        ax_plot = 'y'

                p = yt.ProjectionPlot(ds, ax_plot, ("gas","density"), data_source = box, center = cen_central, width = (ray_l, 'kpc'))


                p.save('/nobackupp2/rcsimons/foggie_momentum/figures/plunges/%s_%s_%i_%s_tunnel.png'%(DDname,axs,i, simname))


                to_save = {}
                to_save['d'] = box['gas', axs].to('kpc') - cen_central[aa]
                to_save['dens'] = box['gas', 'density'].to('g/cm**3')
                if i == 0: sn = 1.
                if i == 1: sn = -1.
                to_save['vel'] = sn * (box['enzo', '%s-velocity'%axs].to('km/s') - cen_bulkv[aa]).to('km/s')

                np.save('/nobackupp2/rcsimons/foggie_momentum/catalogs/plunge/%s_%s_%i_%s.npy'%(DDname,axs,i, simname), to_save)


    if True:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        for aa, axs in enumerate(['x', 'y', 'z']):
            for i in np.arange(2):
                plunge = np.load('/nobackupp2/rcsimons/foggie_momentum/catalogs/plunge/%s_%s_%i_%s.npy'%(DDname, axs,i, simname), allow_pickle = True)[()]
                vmax = np.load('/nobackupp2/rcsimons/foggie_momentum/catalogs/vescape/%s_%s_vescape.npy'%(DDname, simname), allow_pickle = True)[()]
                dinner = yt.YTArray(200., 'kpc')
                dt = yt.YTArray(2.e7, 'yr')
                M = 0
                tot_Ms = []
                ts = []
                for t in arange(0, 1000):
                    douter = dinner
                    vmax_interp = yt.YTArray(np.interp(douter, vmax['r'], vmax['v']), 'km/s')
                    dinner = douter - (vmax_interp * dt.to('s')).to('kpc')
                    print (douter, dinner)
                    if dinner <0 : break
                    gd = where((plunge['d'] > dinner) & (plunge['d'] < douter))[0]

                    dvel = np.mean(plunge['vel'] + vmax_interp)
                    dens = np.mean(plunge['dens'])
                    P = dens * dvel**2.
                    M += P * dt
                    tot_Ms.append(M.value)
                    print (t)
                    ts.append((t * dt.to('s')).to('yr'))
                ax.plot(ts, tot_Ms ,'k-')    
        fig.savefig('/nobackupp2/rcsimons/foggie_momentum/figures/plunges/%s_%s_%i_%s.png'%(DDname, axs,i, simname), dpi = 300)






















