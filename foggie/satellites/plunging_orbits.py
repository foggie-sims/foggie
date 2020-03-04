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
from scipy.interpolate import interp1d
plt.ioff()
plt.close('all')
np.random.seed(1)



def create_plunging_tunnels(haloname, DDname, simnames = ['nref11c_nref9f'],  ray_l = 400, ray_s = 10, nrays = 100):

    center_dic =  np.load('/Users/rsimons/Dropbox/foggie/outputs/centers/%s_nref11c_nref9f_%s.npy'%(haloname,DDname), allow_pickle = True)[()]
    to_save = {}

    for simname in simnames:
        to_save_sim = {}
        print (haloname, simname)

        flname = '/Users/rsimons/Desktop/foggie/sims/%s/%s/%s/%s'%(haloname, simname, DDname, DDname)

        ds = yt.load(flname)
        center = ds.arr(center_dic, 'code_length').to('kpc')

        v_sphere = ds.sphere(center, (100, 'kpc'))  
        cen_bulkv = v_sphere.quantities.bulk_velocity().to('km/s') 


        ax_plots = ['y', 'z', 'y']

        np.random.seed(1)

        ts_random = np.random.random(nrays)*np.pi*2
        ps_random = array([np.math.acos(2*np.random.random()-1) for i in arange(nrays)])





        #save a ray that represents the mean density and velocity profiles, subtracting off known satellites
        to_save_sim_average = {}

        outer_sphere = ds.sphere(center, (ray_l, 'kpc'))  
        inner_sphere = ds.sphere(center, (ray_s, 'kpc'))  
        sphere = outer_sphere# - inner_sphere
        sat_cat = ascii.read('/Users/rsimons/Dropbox/foggie/catalogs/satellite_properties.cat')

        sat_cat_halo = sat_cat[sat_cat['halo'] == int(haloname.replace('halo_00', ''))]
        for sat in sat_cat_halo:
            if sat['id'] == '0': continue
            center = ds.arr([sat['x'], sat['y'], sat['z']], 'kpc')
            sat_sphere = ds.sphere(center, (5, 'kpc'))
            sphere = sphere - sat_sphere

        sphere.set_field_parameter('center', center)
        sphere.set_field_parameter('bulk_velocity', cen_bulkv)


        n_bins = int((ray_l))/1. #want a bin every 1 kpc
        prof_gas     = yt.create_profile(sphere, ('index', 'radius'), fields = [('gas', 'density'), ('gas', 'radial_velocity')],\
                                         n_bins = n_bins, weight_field = ('index', 'cell_volume'), accumulation = False)

        to_save_sim_average['density'] = prof_gas[('gas', 'density')].to('g/cm**3')
        to_save_sim_average['dist_r']  = prof_gas.x.to('kpc')
        to_save_sim_average['vel_r']   = prof_gas[('gas', 'radial_velocity')].to('km/s')

        to_save_sim['average'] = to_save_sim_average




        for i, (theta, phi) in enumerate(zip(ts_random, ps_random)):
            #randomly sample rays starting along a sphere of ray_l and ending at ray_s
            to_save_sim_ray = {}
            end = [(center[0].value + ray_l * np.cos(theta)*np.sin(phi), "kpc"),\
                   (center[1].value + ray_l * np.sin(theta)*np.sin(phi), "kpc"),\
                   (center[2].value + ray_l * np.cos(phi),               "kpc")]

            start = [(center[0].value + ray_s * np.cos(theta)*np.sin(phi), "kpc"),\
                     (center[1].value + ray_s * np.sin(theta)*np.sin(phi), "kpc"),\
                     (center[2].value + ray_s * np.cos(phi),               "kpc")]


            ray = ds.r[start:end]

            ray.set_field_parameter('center', center)
            ray.set_field_parameter('bulk_velocity', cen_bulkv)
            #save gas densities

            # distances in a coordinate system centered at central galaxy
            # velocities in the rest-frame of the central galaxy
            dist_r = ray['index', 'radius'].to('kpc')
            vel_r  = ray['gas', 'radial_velocity'].to('km/s')

            to_save_sim_ray['density'] = ray['gas', 'density'].to('g/cm**3')
            to_save_sim_ray['dist_r'] = ray['index', 'radius'].to('kpc')
            to_save_sim_ray['vel_r'] = ray['gas', 'radial_velocity'].to('km/s')

            to_save_sim['ray_%s'%i] = to_save_sim_ray

        to_save_sim['theta'] = ts_random
        to_save_sim['phi']   = ps_random

        to_save[simname] = to_save_sim
    np.save('/Users/rsimons/Dropbox/foggie/outputs/plunge_tunnels/%s_rays.npy'%(haloname), to_save)




def simulate_plunging_orbits(haloname, DDname, simnames = ['nref11c_nref9f'], dinner_start = 100., nrays = 5):


    rays = np.load('/Users/rsimons/Dropbox/foggie/outputs/plunge_tunnels/%s_rays.npy'%(haloname), allow_pickle = True)[()]
    vesc = np.load('/Users/rsimons/Dropbox/foggie/outputs/vesc_profile/%s_vmax.npy'%(haloname), allow_pickle = True)[()]
    to_save = {}



    for s, simname in enumerate(simnames):
        to_save_simname = {}
        for i in arange(nrays+1):

            if i <  nrays: 
                dinner_crit = 9
                ray_name = 'ray_%s'%i
            else: 
                dinner_crit = 0.2
                ray_name = 'average'

            to_save_rays = {}

            if i%25 == 0: print (haloname, simname, i, '/', nrays)
            dinner = yt.YTArray(dinner_start, 'kpc')
            dt = yt.YTArray(1.e6, 'yr')
            M, t = 0, 0
            tot_Ms, P_all, ts, dmids = [], [], [], []

            vel_r  = rays[simname][ray_name]['vel_r']
            dist_r = rays[simname][ray_name]['dist_r']


            while True:
                douter = dinner

                vmax_interp = yt.YTArray(np.interp(douter, vesc['r_%s'%simname], vesc['vesc_%s'%simname]), 'km/s')
                dinner = douter - (vmax_interp * dt.to('s')).to('kpc')
                if dinner < dinner_crit : break
                gd = argmin(abs(dist_r - (dinner + douter)/2.))

                dens = rays[simname][ray_name]['density'][gd]

                #dvel = vel_r[gd] + yt.YTArray(vmax_interp, 'km/s')
                dvel = yt.YTArray(min([(vel_r[gd] - yt.YTArray(vmax_interp, 'km/s')).value, 0]), 'km/s')

                P = (dens * dvel**2.).to('dyne * cm**-2')
                M += (P * dt).to('Msun * km/s * 1/kpc**2')
                tot_Ms.append(M.value)
                P_all.append(P.value)
                dmids.append((dinner + douter).value/2.)
                ts.append(t * dt.to('Myr').value)
                t+=1


            to_save_rays['time'] = array(ts)
            to_save_rays['tot_M'] = array(tot_Ms)
            to_save_rays['P'] = array(P_all)
            to_save_rays['dmids'] = array(dmids)

            to_save_simname[ray_name] = to_save_rays




        to_save[simname] = to_save_simname

    np.save('/Users/rsimons/Dropbox/foggie/outputs/plunge_tunnels/%s_simulated_plunge.npy'%(haloname), to_save)






def plot_plunging_orbits(haloname, name, DDname, simnames = ['nref11c_nref9f'], dinner_start = 100., nrays = 5):
    import matplotlib.pyplot as plt
    plt.rcParams['ytick.minor.size'] = 3.
    plt.rcParams['ytick.major.size'] = 5.
    plt.rcParams['xtick.minor.size'] = 3.
    plt.rcParams['xtick.major.size'] = 5.

    plt.rcParams['xtick.labelsize'] = 12.
    plt.rcParams['ytick.labelsize'] = 12.
    fig, axes = plt.subplots(2,1, figsize = (5.5,10))

    plt.ioff()
    plunge_sim = np.load('/Users/rsimons/Dropbox/foggie/outputs/plunge_tunnels/%s_simulated_plunge.npy'%(haloname), allow_pickle = True)[()]

    clrs = ['black', 'red', 'blue']
    to_save = {}
    for s, simname in enumerate(simnames):

        ts_all = array([])
        P_all = array([])
        m_all  = array([])
        dmids_all  = array([])


        for i in arange(nrays+1):
            if i <  nrays: 
                ray_name = 'ray_%s'%i
                lw = 0.1
                ls = '-'
                alp = 0.5
                clrs[s] = 'black'
            else: 
                ray_name = 'average'
                lw = 3.0
                ls = '--'
                alp = 1.0
                clrs[s] = 'red'

            ts      = plunge_sim[simname][ray_name]['time']
            tot_Ms  = plunge_sim[simname][ray_name]['tot_M']
            P     = plunge_sim[simname][ray_name]['P']
            dmids   = plunge_sim[simname][ray_name]['dmids']

            ts_all  = concatenate((ts_all, ts))
            P_all  = concatenate((P_all, P))
            m_all  = concatenate((m_all, tot_Ms))
            dmids_all = concatenate((dmids_all, dmids))

            
            axes[0].plot(ts, P, color = clrs[s], linestyle = ls, alpha = alp, linewidth = lw)    
            axes[1].plot(ts, tot_Ms, color = clrs[s], linestyle = ls, alpha = alp, linewidth = lw)    

            if ray_name == 'average': 
                print ()
                tot_ms_average = log10(tot_Ms[dmids > 10][-1])



        tmin = min(ts_all)
        tmax = max(ts_all)

        smoothed_P_5 = []
        smoothed_P_16 = []
        smoothed_P_50 = []
        smoothed_P_84 = []
        smoothed_P_95 = []



        smoothed_M_5 = []
        smoothed_M_16 = []
        smoothed_M_50 = []
        smoothed_M_84 = []
        smoothed_M_95 = []




        smoothed_t = []
        smoothed_t_all = arange(tmin, tmax, 5)


        for t in smoothed_t_all:
            gd = where((ts_all > t) & (ts_all < t+5) & (P_all > 0.)& (dmids_all > 10.))[0]
            gd2 = where((ts_all > t) & (ts_all < t+5) & (dmids_all > 10.))[0]

            if len(gd) == 0: 
                continue

            perc_P = np.percentile(log10(P_all[gd]), [5, 16, 50, 84, 95])
            perc_M = np.percentile(log10(m_all[gd2]), [5, 16, 50, 84, 95])


            smoothed_P_5.append( perc_P[0])
            smoothed_P_16.append( perc_P[1])
            smoothed_P_50.append(perc_P[2])
            smoothed_P_84.append( perc_P[3])
            smoothed_P_95.append( perc_P[4])



            smoothed_M_5.append(perc_M[0])
            smoothed_M_16.append(perc_M[1])
            smoothed_M_50.append(perc_M[2])
            smoothed_M_84.append(perc_M[3])
            smoothed_M_95.append(perc_M[4])

            smoothed_t.append(t)





        if True:
            smoothed_t      = np.array(smoothed_t)


            smoothed_P_5    = np.array(smoothed_P_5 )
            smoothed_P_16   = np.array(smoothed_P_16)
            smoothed_P_50   = np.array(smoothed_P_50)
            smoothed_P_84   = np.array(smoothed_P_84)
            smoothed_P_95   = np.array(smoothed_P_95)


            smoothed_M_5    = np.array(smoothed_M_5)
            smoothed_M_16   = np.array(smoothed_M_16)
            smoothed_M_50   = np.array(smoothed_M_50)
            smoothed_M_84   = np.array(smoothed_M_84)
            smoothed_M_95   = np.array(smoothed_M_95)



        clr = 'black'

        axes[0].plot(smoothed_t + 2.5, 10**(smoothed_P_16), color = clr, linewidth = 1.5)
        axes[0].plot(smoothed_t + 2.5, 10**(smoothed_P_50), color = clr, linewidth = 3)
        axes[0].plot(smoothed_t + 2.5, 10**(smoothed_P_84), color = clr, linewidth = 1.5)

        axes[1].plot(smoothed_t + 2.5, 10**(smoothed_M_16), color = clr,  linewidth = 1.5)
        axes[1].plot(smoothed_t + 2.5, 10**(smoothed_M_50), color = clr, linewidth = 3)
        axes[1].plot(smoothed_t + 2.5, 10**(smoothed_M_84), color = clr,  linewidth = 1.5)

    axes[0].set_ylabel(r'Surface Ram Pressure (dyne cm$^{-2}$)')
    axes[1].set_ylabel(r'Cumulative Momentum Imparted (M$_{\odot}$ km s$^{-1}$ kpc$^{-2}$)')

    for a, ax in enumerate(axes.ravel()):
        ax.set_yscale('log')
        ax2 = ax.twiny()

        ax2_ticks = arange(0, 105,  20)

        ax2_tickstr = array(['%i'%d for d in ax2_ticks])


        x = array(dmids)
        y = array(ts)


        interp = interp1d(x, y, fill_value = 'extrapolate')

        ax2_tick_locs = array([interp(d) for d in ax2_ticks])

        ax2.set_xticks(ax2_tick_locs)
        


        ax.set_xlim(ax2_tick_locs[-1], ax2_tick_locs[0])
        ax2.set_xlim(ax.get_xlim())

        ax.axvspan(xmin = max(smoothed_t+2.5), xmax = ax.get_xlim()[1], color = 'grey', alpha = 1.0)


        if a == 0:
            ax2.set_xticklabels(ax2_tickstr)
            ax.set_xticklabels([''])
            ax.set_xlabel('')
            #ax.set_ylim(5.e-17, 8.e-11)
            ax2.set_xlabel('Distance from Central Galaxy (kpc)')
            ax.set_yscale('symlog', linthreshy=1.e-16)
            #ax.set_ylim(-1.e-18, 5.e-12)
            #ax.set_yticks([0., 1.e-17, 1.e-16, 1.e-15, 1.e-14, 1.e-13, 1.e-12])
            ax.set_ylim(-1.e-17, 3.e-11)
            ax.set_yticks([0., 1.e-16, 1.e-15, 1.e-14, 1.e-13, 1.e-12, 1.e-11])


            minorticks_y = array([])
            minorticks_y = np.concatenate((minorticks_y, np.arange(0, 1.e-16, 0.1e-16)))

            for i in arange(-15, -10, 1):
                minorticks_y = np.concatenate((minorticks_y, np.arange(10.**(i-1), 10.**(i), 10.**(i - 1))))

            ax.set_yticks(minorticks_y, minor = True)


        else:
            ax2.set_xticklabels([''])
            ax2.set_xlabel('')
            ax.set_xlabel('Elapsed Time (Myr)')
            ax.set_ylim(1.e3, 1.e9)


    axes[0].annotate(name, (0.05, 0.88), ha = 'left', va = 'bottom', xycoords = 'axes fraction', fontsize = 30, color = 'black', fontweight = 'bold')
    axes[0].annotate('%i radial trajectories'%nrays, (0.05, 0.87), ha = 'left', va = 'top', xycoords = 'axes fraction', fontsize = 16, color = 'grey')


    fig.tight_layout()
    fig.subplots_adjust(hspace = 0.10)
    #print ('saving %s'%haloname)
    print ('%s %.2f  %.2f  %.2f  %.2f  %.2f  %.2f'%(haloname, smoothed_M_5[-1], smoothed_M_16[-1], smoothed_M_50[-1], smoothed_M_84[-1],smoothed_M_95[-1], tot_ms_average))
    if haloname == 'halo_008508':
        fig_dir = '/Users/rsimons/Dropbox/foggie/figures/for_paper'

        fig.savefig(fig_dir + '/%s_nref11c_nref9f.png'%haloname, dpi = 300)

    fig.savefig('/Users/rsimons/Dropbox/foggie/figures/simulated_plunges/%s_nref11c_nref9f.png'%haloname, dpi = 300)








if __name__ == '__main__':

    halonames = array([('halo_002392', 'Hurricane', 'DD0581'), 
                       ('halo_002878', 'Cyclone',  'DD0581'), 
                       ('halo_004123', 'Blizzard',  'DD0581'), 
                       ('halo_005016', 'Squall',  'DD0581'), 
                       ('halo_005036', 'Maelstrom',  'DD0581'), 
                       ('halo_008508', 'Tempest',  'DD0487')])




    #halonames = halonames[-1:]

    nrays = 100
    halonames = halonames
    #Parallel(n_jobs = 2)(delayed(create_plunging_tunnels)(haloname, DDname, nrays = nrays) for (haloname,  name,DDname) in halonames)
    #Parallel(n_jobs = -1)(delayed(simulate_plunging_orbits)(haloname,DDname,  nrays = nrays) for (haloname, name, DDname) in halonames)
    Parallel(n_jobs = -1)(delayed(plot_plunging_orbits)(haloname, name, DDname,   nrays = nrays) for (haloname, name, DDname) in halonames)
    #plot_plunging_orbits(halonames[0][0], halonames[0][1],halonames[0][2],   nrays = nrays)

















