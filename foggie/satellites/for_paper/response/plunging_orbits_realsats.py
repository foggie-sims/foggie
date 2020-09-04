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


def run_plunge(ds, orig_halo, orbits,halo_bulkv, center, haloname, ax_inversions):
    to_save = {}
    to_save[orig_halo] = {}
    for sat in orbits[orig_halo]:
        to_save[orig_halo][sat] = {}
        for ii, ((i1, n1), (i2,n2), (i3,n3)) in enumerate(ax_inversions):
            print('%s-%s in %s: %i'%(orig_halo, sat, haloname, 100*ii/len(ax_inversions)) + r'% complete') 
            to_save[orig_halo][sat][ii] = {}
            to_save[orig_halo][sat][ii]['dens'] = []
            to_save[orig_halo][sat][ii]['vel'] = []
            to_save[orig_halo][sat][ii]['RP'] = []
            to_save[orig_halo][sat][ii]['t'] = []
            to_save[orig_halo][sat][ii]['r'] = []
            N = len(orbits[orig_halo][sat][i1])
            for i in np.arange(N):
                x = center[0].value + orbits[orig_halo][sat][i1][i]
                y = center[1].value + orbits[orig_halo][sat][i2][i]
                z = center[2].value + orbits[orig_halo][sat][i3][i]
                t = orbits[orig_halo][sat]['t'][i]
                vx = orbits[orig_halo][sat]['v'+i1][i]
                vy = orbits[orig_halo][sat]['v'+i2][i]
                vz = orbits[orig_halo][sat]['v'+i3][i]
                r =  orbits[orig_halo][sat]['r'][i]
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

                    to_save[orig_halo][sat][ii]['dens'].append(dens)
                    to_save[orig_halo][sat][ii]['vel'].append(face_on_v)
                    to_save[orig_halo][sat][ii]['RP'].append(RP)
                    to_save[orig_halo][sat][ii]['t'].append(t)
                    to_save[orig_halo][sat][ii]['r'].append(r)
                else:
                    to_save[orig_halo][sat][ii]['dens'].append(np.nan)
                    to_save[orig_halo][sat][ii]['vel'].append(np.nan)
                    to_save[orig_halo][sat][ii]['RP'].append(np.nan)
                    to_save[orig_halo][sat][ii]['t'].append(np.nan)
                    to_save[orig_halo][sat][ii]['r'].append(np.nan)

        print('%s-%s in %s: 100'%(orig_halo, sat, haloname) + r'% complete')


    np.save('/Users/rsimons/Dropbox/git/foggie/foggie/satellites/for_paper/response/%s_plunges_in_%s.npy'%(orig_halo, haloname), to_save)

    return


def run_all(haloname, DDname, simname = 'nref11c_nref9f', on_jase = True):    
    center_dic =  np.load('/Users/rsimons/Dropbox/foggie/outputs/centers/halo_00%s_nref11c_nref9f_%s.npy'%(haloname,DDname), allow_pickle = True)[()]
    orbits = np.load('/Users/rsimons/Dropbox/git/foggie/foggie/satellites/for_paper/response/satellite_orbits.npy', allow_pickle = True)[()]

    flname = '/Users/rsimons/Desktop/foggie/sims/halo_00%s/%s/%s/%s'%(haloname, simname, DDname, DDname)
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
    for h, orig_halo in enumerate(orbits):
        run_plunge(ds, orig_halo, orbits, halo_bulkv, center, haloname, ax_inversions = ax_inversions) 
        x = np.load('/Users/rsimons/Dropbox/git/foggie/foggie/satellites/for_paper/response/%s_plunges_in_%s.npy'%(orig_halo, haloname), allow_pickle = True)[()]
        to_save_total[orig_halo] = x[orig_halo]
        os.system('rm /Users/rsimons/Dropbox/git/foggie/foggie/satellites/for_paper/response/%s_plunges_in_%s.npy'%(orig_halo, haloname))
    np.save('/Users/rsimons/Dropbox/git/foggie/foggie/satellites/for_paper/response/plunges_in_%s.npy'%haloname, to_save_total)



if __name__ == '__main__':


    halonames = array([('8508', 'DD0487'),
                       ('2392', 'DD0581'), 
                       ('2878', 'DD0581'), 
                       ('4123', 'DD0581'), 
                       ('5016', 'DD0581'), 
                       ('5036', 'DD0581')])

    Parallel(n_jobs = -1)(delayed(run_all)(haloname, DDname) for (haloname, DDname) in halonames)
    '''
    for (haloname, DDname) in halonames[:1]:
        print ('simulating real satellite orbits in %s'%haloname)
        run_all(haloname, DDname)
    '''
