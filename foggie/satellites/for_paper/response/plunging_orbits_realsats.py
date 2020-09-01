import yt
from yt.units import kpc, Mpc
import joblib
from joblib import Parallel, delayed
import os
from astropy.io import ascii
import matplotlib.pyplot as plt
import numpy as np
from numpy import *
from astropy import constants as c
from astropy.cosmology import Planck13 as cosmo
import scipy
from scipy import stats
from scipy import interpolate
from scipy.interpolate import interp1d
import astropy.units as u
plt.ioff()
plt.close('all')
np.random.seed(1)


def run_plunge(ds, halo, orbits,halo_bulkv, center, ax_inversions):
    to_save = {}
    to_save[halo] = {}
    for sat in orbits[halo]:
        print (halo, sat)
        to_save[halo][sat] = {}
        for ii, ((i1, n1), (i2,n2), (i3,n3)) in enumerate(ax_inversions):
            to_save[halo][sat][ii] = {}
            to_save[halo][sat][ii]['dens'] = []
            to_save[halo][sat][ii]['vel'] = []
            to_save[halo][sat][ii]['RP'] = []
            to_save[halo][sat][ii]['t'] = []
            to_save[halo][sat][ii]['r'] = []
            N = len(orbits[halo][sat][i1])
            print (N)
            for i in np.arange(N):
                x = center[0].value + orbits[halo][sat][i1][i]
                y = center[1].value + orbits[halo][sat][i2][i]
                z = center[2].value + orbits[halo][sat][i3][i]
                t = orbits[halo][sat]['t'][i]
                vx = orbits[halo][sat]['v'+i1][i]
                vy = orbits[halo][sat]['v'+i2][i]
                vz = orbits[halo][sat]['v'+i3][i]
                r =  orbits[halo][sat]['r'][i]
                if r < 100:
                    center = yt.YTArray([x,y,z], 'kpc')
                    cen_bulkv = yt.YTArray([vx,vy,vz], 'km/s')
                    sphere = ds.sphere(center, (2., 'kpc'))
                    s_vx = sphere.quantities.weighted_average_quantity(('gas', 'velocity_x'), ('gas', 'cell_volume'))
                    s_vy = sphere.quantities.weighted_average_quantity(('gas', 'velocity_y'), ('gas', 'cell_volume'))
                    s_vz = sphere.quantities.weighted_average_quantity(('gas', 'velocity_z'), ('gas', 'cell_volume'))
                    dens = sphere.quantities.weighted_average_quantity(('gas', 'density'), ('gas', 'cell_volume'))

                    s_v =  yt.YTArray([s_vx,s_vy,s_vz], 'km/s')
                    sat_v = halo_bulkv + cen_bulkv
                    d_v = sat_v - s_v
                    norm_sat_v = sat_v/(np.sqrt(sum(sat_v**2.)))
                    face_on_v = sum(d_v * norm_sat_v)
                    RP = dens * max([0., face_on_v])**2.

                    to_save[halo][sat][ii]['dens'].append(dens)
                    to_save[halo][sat][ii]['vel'].append(face_on_v)
                    to_save[halo][sat][ii]['RP'].append(RP)
                    to_save[halo][sat][ii]['t'].append(t)
                    to_save[halo][sat][ii]['r'].append(r)
                else:
                    to_save[halo][sat][ii]['dens'].append(np.nan)
                    to_save[halo][sat][ii]['vel'].append(np.nan)
                    to_save[halo][sat][ii]['RP'].append(np.nan)
                    to_save[halo][sat][ii]['t'].append(np.nan)
                    to_save[halo][sat][ii]['r'].append(np.nan)


    np.save('/Users/rsimons/Dropbox/git/foggie/foggie/satellites/for_paper/response/%s_plunges.npy'%halo, to_save)

    return


if __name__ == '__main__':

    halonames = array([('8508', 'Tempest',  'DD0487')])
    for (haloname, name, DDname) in halonames:
        #create_plunging_tunnels(haloname, DDname)
        pass

    simname = 'nref11c_nref9f'
    center_dic =  np.load('/Users/rsimons/Dropbox/foggie/outputs/centers/halo_00%s_nref11c_nref9f_%s.npy'%(haloname,DDname), allow_pickle = True)[()]
    orbits = np.load('/Users/rsimons/Dropbox/git/foggie/foggie/satellites/for_paper/response/satellite_orbits.npy')[()]

    flname = '/Users/rsimons/Desktop/foggie/sims/%s/%s/%s/%s'%(haloname, simname, DDname, DDname)

    ds = yt.load(flname)
    center = ds.arr(center_dic, 'code_length').to('kpc')
    v_sphere = ds.sphere(center, (100, 'kpc'))  
    halo_bulkv = v_sphere.quantities.bulk_velocity().to('km/s') 

    xx = ('x', 0)
    yy = ('y', 1)
    zz = ('z', 2)

    ax_inversions = [(xx, yy, zz),\
                     (xx, zz, yy),\
                     (yy, xx, zz),\
                     (yy, zz, xx),\
                     (zz, xx, yy),\
                     (zz, yy, xx)]
    to_save_total = {}
    for h, halo in enumerate(orbits):
        print (halo)
        run_plunge(ds, halo, orbits, halo_bulkv, center, ax_inversions = ax_inversions) 
        x = np.load('/Users/rsimons/Dropbox/git/foggie/foggie/satellites/for_paper/response/%s_plunges.npy'%halo)[()]
        to_save_total[halo] = x[halo]
    np.save('/Users/rsimons/Dropbox/git/foggie/foggie/satellites/for_paper/response/plunges.npy', to_save_total)
    #Parallel(n_jobs = -1)(delayed(run_plunge)(ds, halo, orbits, halo_bulkv, center, ax_inversions = ax_inversions) for (halo) in orbits)
    #to_save = {}
    #for h, halo in enumerate(orbits):
    #    x = np.load('/Users/rsimons/Dropbox/git/foggie/foggie/satellites/for_paper/response/%s_plunges.npy'%halo)[()]
    #    to_save[halo] = x[halo]
    #np.save('/Users/rsimons/Dropbox/git/foggie/foggie/satellites/for_paper/response/plunges.npy', to_save)



    '''
    to_save = {}

    for h, halo in enumerate(orbits):
        #if h == 0.: continue
        to_save[halo] = {}
        for sat in orbits[halo]:
            print (halo, sat)
            to_save[halo][sat] = {}
            for ii, ((i1, n1), (i2,n2), (i3,n3)) in enumerate(ax_inversions):
                to_save[halo][sat][ii] = {}
                to_save[halo][sat][ii]['dens'] = []
                to_save[halo][sat][ii]['vel'] = []
                to_save[halo][sat][ii]['RP'] = []
                to_save[halo][sat][ii]['t'] = []
                to_save[halo][sat][ii]['r'] = []
                N = len(orbits[halo][sat][i1])
                print (N)
                for i in np.arange(N):
                    x = center[0].value + orbits[halo][sat][i1][i]
                    y = center[1].value + orbits[halo][sat][i2][i]
                    z = center[2].value + orbits[halo][sat][i3][i]
                    t = orbits[halo][sat]['t'][i]
                    vx = orbits[halo][sat]['v'+i1][i]
                    vy = orbits[halo][sat]['v'+i2][i]
                    vz = orbits[halo][sat]['v'+i3][i]
                    r =  orbits[halo][sat]['r'][i]
                    if r < 100:
                        center = yt.YTArray([x,y,z], 'kpc')
                        cen_bulkv = yt.YTArray([vx,vy,vz], 'km/s')
                        sphere = ds.sphere(center, (1., 'kpc'))
                        sphere_bulkv = sphere.quantities.bulk_velocity().to('km/s') 

                        s_vx = sphere.quantities.weighted_average_quantity(('gas', 'velocity_x'), ('gas', 'cell_mass'))
                        s_vy = sphere.quantities.weighted_average_quantity(('gas', 'velocity_y'), ('gas', 'cell_mass'))
                        s_vz = sphere.quantities.weighted_average_quantity(('gas', 'velocity_z'), ('gas', 'cell_mass'))
                        dens = sphere.quantities.weighted_average_quantity(('gas', 'density'), ('gas', 'cell_volume'))

                        s_v =  yt.YTArray([s_vx,s_vy,s_vz], 'km/s')
                        sat_v = halo_bulkv + cen_bulkv
                        d_v = sat_v - s_v
                        norm_sat_v = sat_v/(np.sqrt(sum(sat_v**2.)))
                        face_on_v = sum(d_v * norm_sat_v)
                        RP = dens * max([0., face_on_v])**2.

                        to_save[halo][sat][ii]['dens'].append(dens)
                        to_save[halo][sat][ii]['vel'].append(face_on_v)
                        to_save[halo][sat][ii]['RP'].append(RP)
                        to_save[halo][sat][ii]['t'].append(t)
                        to_save[halo][sat][ii]['r'].append(r)


    np.save('/Users/rsimons/Dropbox/git/foggie/foggie/satellites/for_paper/response/plunges.npy', to_save)
    '''
    #Parallel(n_jobs = 2)(delayed(create_plunging_tunnels)(haloname, DDname, nrays = nrays) for (haloname,  name,DDname) in halonames)
    #Parallel(n_jobs = -1)(delayed(simulate_plunging_orbits)(haloname,DDname,  ms, nrays = nrays) for (haloname, name, DDname) in halonames)


    #for (haloname, name, DDname) in halonames:
    #        simulate_plunging_orbits(haloname,DDname,  ms, nrays = 5)

    #Parallel(n_jobs = -1)(delayed(plot_plunging_orbits)(haloname, name, DDname,   nrays = nrays) for (haloname, name, DDname) in halonames)
    #for (haloname, name, DDname) in halonames:
    #    plot_plunging_orbits(haloname, name, DDname,  nrays = 5)

















