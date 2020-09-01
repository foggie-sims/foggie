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
from astropy.convolution import convolve_fft, Gaussian1DKernel
plt.ioff()
plt.close('all')
np.random.seed(1)



def create_plunging_tunnels(haloname, DDname, simname = 'nref11c_nref9f',  ray_l = 110., ray_s = 10, nrays = 100, smooth = 1., center_dic = None):

    to_save = {}

    flname = '/Users/rsimons/Desktop/foggie/sims/%s/%s/%s/%s'%(haloname.strip('halo_00'), simname, DDname, DDname)

    ds = yt.load(flname)

    center_dic =  np.load('/Users/rsimons/Dropbox/foggie/outputs/centers/%s_nref11c_nref9f_%s.npy'%(haloname,DDname), allow_pickle = True)[()]

    to_save_sim = {}
    print (haloname, simname)

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


    dic = {}
    dic[('index', 'radius')] = False
    n_bins = int((ray_l))/1. #want a bin every "smooth" kpc
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

        xs = ray['index', 'x'].to('kpc')
        ys = ray['index', 'y'].to('kpc')
        zs = ray['index', 'z'].to('kpc')


        ray.set_field_parameter('center', center)
        ray.set_field_parameter('bulk_velocity', cen_bulkv)
        dist_r = ray['index', 'radius'].to('kpc')
        vel_r  = ray['gas', 'radial_velocity'].to('km/s')

        dens = []
        vel = []
        dist_r = []
        import time
        c = time.time()
        print (len(xs))
        for count, (xi, yi, zi) in enumerate(zip(xs, ys, zs)):
            if count%(int(smooth/0.5)) !=0: continue
            cen = ds.arr([xi,yi,zi], 'kpc')
            sphere = ds.sphere(cen, (smooth, 'kpc'))
            sphere.set_field_parameter('center', center)
            sphere.set_field_parameter('bulk_velocity', cen_bulkv)
            #save gas densities

            # distances in a coordinate system centered at central galaxy
            # velocities in the rest-frame of the central galaxy
            dens.append(sphere.quantities.weighted_average_quantity(('gas', 'density'), ('index', 'cell_volume')).to('g/cm**3'))
            vel.append(sphere.quantities.weighted_average_quantity(('gas', 'radial_velocity'), ('index', 'cell_volume')).to('km/s'))
            dist_r.append(sphere.quantities.weighted_average_quantity(('index', 'radius'), ('index', 'cell_volume')).to('kpc'))
        d = time.time()
        print (smooth, i, d-c)
        #to_save_sim_ray['density'] = ray['gas', 'density'].to('g/cm**3')
        #to_save_sim_ray['vel_r'] = ray['gas', 'radial_velocity'].to('km/s')
        #to_save_sim_ray['dist_r'] = ray['index', 'radius'].to('kpc')

        to_save_sim_ray['density'] = dens
        to_save_sim_ray['vel_r'] = vel
        to_save_sim_ray['dist_r'] = dist_r

        to_save_sim['ray_%s'%i] = to_save_sim_ray

    to_save_sim['theta'] = ts_random
    to_save_sim['phi']   = ps_random

    to_save[simname] = to_save_sim
    np.save('/Users/rsimons/Dropbox/foggie/outputs/plunge_tunnels/%s_rays_%.1f.npy'%(haloname, smooth), to_save)




def simulate_plunging_orbits(haloname, DDname, ms, simnames = ['nref11c_nref9f'], dinner_start = 100., nrays = 100, smooth = False):


    rays = np.load('/Users/rsimons/Dropbox/foggie/outputs/plunge_tunnels/%s_rays_%.1f.npy'%(haloname, smooth), allow_pickle = True)[()]
    vesc = np.load('/Users/rsimons/Dropbox/foggie/outputs/vesc_profile/%s_vmax.npy'%(haloname), allow_pickle = True)[()]
    to_save = {}


    for s, simname in enumerate(simnames):
        to_save_simname = {}

        for i in arange(nrays+1):
            if i <  nrays: 
                dinner_crit = 9 * u.kpc
                ray_name = 'ray_%s'%i
            else: 
                dinner_crit = 9* u.kpc
                ray_name = 'average'

            to_save_rays = {}

            if i%5 == 0: print (haloname, simname, i, '/', nrays)
            dinner = dinner_start*u.kpc
            dt = 1.e6 * u.yr
            M, t = 0, 0
            tot_Ms, P_all, ts, dmids = [], [], [], []

            vel_r  = np.array(rays[simname][ray_name]['vel_r'])*u.km/u.s
            dist_r = np.array(rays[simname][ray_name]['dist_r'])*u.kpc
            dens_r = np.array(rays[simname][ray_name]['density']) * u.g/u.cm**3.

            arg = argsort(dist_r)[::-1]
            vel_r  = vel_r[arg]
            dist_r = dist_r[arg]
            dens_r = dens_r[arg]


            '''
            if not smooth: pass
            else:
                n_smooth = smooth/(dist_r[-2] - dist_r[-1])
                #return
                kern = Gaussian1DKernel(n_smooth.value)
                vel_r = convolve_fft(vel_r, kern) * u.km/u.s
                dens_r = convolve_fft(dens_r, kern) * u.g/u.cm**3.
            '''

            while True:
                douter = dinner

                vmax_interp = np.interp(douter.value, vesc['r_%s'%simname].value, vesc['vesc_%s'%simname].value) * u.km/u.s
                dinner = douter - (vmax_interp * dt.to('s')).to('kpc')
                if dinner < dinner_crit : break
                gd = argmin(abs(dist_r.value - (dinner.value + douter.value)/2.))


                if ray_name == 'average':
                    #dens = rays[simname][ray_name]['density'][gd]
                    #dvel = vel_r[gd] + yt.YTArray(vmax_interp, 'km/s')
                    #dvel = yt.YTArray(min([(vel_r[gd] - yt.YTArray(vmax_interp, 'km/s')).value, 0]), 'km/s')
                    dens = 10**(ms[haloname][0]*log10(dist_r[gd].value) + ms[haloname][1]) * u.g/u.cm**3.
                    dvel = vmax_interp
                else:
                    dens = dens_r[gd]
                    #dvel = vel_r[gd] + yt.YTArray(vmax_interp, 'km/s')
                    dvel = min([(vel_r[gd].value -  vmax_interp.value), 0]) * u.km/u.s




                P = (dens * dvel**2.).to('dyne * cm**-2')
                M += (P * dt).to('Msun * km/s/kpc**2')
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
    if not smooth: flname = '%s_simulated_plunge.npy'%(haloname)
    else:    flname = '%s_simulated_plunge_%.1f.npy'%(haloname, smooth)
         
    np.save('/Users/rsimons/Dropbox/foggie/outputs/plunge_tunnels/%s'%(flname), to_save)






def plot_plunging_orbits(haloname, name, DDname, simnames = ['nref11c_nref9f'], dinner_start = 100., nrays = 100., smooth = False):
    if not smooth: flname = '%s_simulated_plunge.npy'%(haloname)
    else:    flname = '%s_simulated_plunge_%.1f.npy'%(haloname, smooth)
         
    plt.rcParams['ytick.minor.size'] = 3.
    plt.rcParams['ytick.major.size'] = 5.
    plt.rcParams['xtick.minor.size'] = 3.
    plt.rcParams['xtick.major.size'] = 5.
    fs = 12
    plt.rcParams['xtick.labelsize'] = 11
    plt.rcParams['ytick.labelsize'] = 11.
    fig, axes = plt.subplots(2,1, figsize = (5.5,10))

    plt.ioff()
    plunge_sim = np.load('/Users/rsimons/Dropbox/foggie/outputs/plunge_tunnels/%s'%flname, allow_pickle = True)[()]


    to_save = {}
    for s, simname in enumerate(simnames):

        ts_all = array([])
        P_all = array([])
        m_all  = array([])
        dmids_all  = array([])


        for i in arange(nrays+1):
            if i <  nrays: 
                ray_name = 'ray_%s'%int(i)
                lw = 0.1
                ls = '-'
                alp = 0.5
                clr = 'black'
                ts      = plunge_sim[simname][ray_name]['time']
                tot_Ms  = plunge_sim[simname][ray_name]['tot_M']
                P     = plunge_sim[simname][ray_name]['P']
                dmids   = plunge_sim[simname][ray_name]['dmids']
            else: 
                ray_name = 'average'
                lw = 3.0
                ls = '--'
                alp = 1.0
                clr = 'red'
                '''
                average = plunge_sim[simname]['average']                                                        
                average_dist_r = average['dmids']
                average_P = average['P']
                average_totM = plunge_sim[simname][ray_name]['tot_M']
                (m, b), v = np.polyfit(log10(average_dist_r), np.log10(average_P), deg = 1, cov = True)

                fit_x = log10(np.linspace(1, 100, 1000))
                fit_y = m*fit_x + b#*fit_x + bb
                interp_P_average = interpolate.interp1d(10**fit_x, fit_y, bounds_error = False, fill_value = fit_y[0])

                P       = 10**interp_P_average(average_dist_r)
                ts      = plunge_sim[simname][ray_name]['time']
                dt      = 1.e6
                tot_Ms  = cumsum((P * dt * 1.51112067e12))
                '''
            dmids   = plunge_sim[simname][ray_name]['dmids']
            ts      = plunge_sim[simname][ray_name]['time']
            tot_Ms  = plunge_sim[simname][ray_name]['tot_M']
            P     = plunge_sim[simname][ray_name]['P']
            dmids   = plunge_sim[simname][ray_name]['dmids']
            ts_all  = concatenate((ts_all, ts))
            P_all  = concatenate((P_all, P))
            m_all  = concatenate((m_all, tot_Ms))
            dmids_all = concatenate((dmids_all, dmids))

            
            axes[0].plot(ts, P, color = clr, linestyle = ls, alpha = alp, linewidth = lw, zorder = 2)    
            axes[1].plot(ts, tot_Ms, color = clr, linestyle = ls, alpha = alp, linewidth = lw, zorder = 3)    






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

    axes[0].set_ylabel(r'Surface Ram Pressure (dyne cm$^{-2}$)', fontsize = fs)
    axes[1].set_ylabel(r'Cumulative Momentum Imparted (M$_{\odot}$ km s$^{-1}$ kpc$^{-2}$)', fontsize = fs)

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

        ax.axvspan(xmin = max(smoothed_t+2.5), xmax = ax.get_xlim()[1], color = 'grey', alpha = 1.0, zorder = 10)


        if a == 0:
            ax2.set_xticklabels(ax2_tickstr)
            ax.set_xticklabels([''])
            ax.set_xlabel('')
            #ax.set_ylim(5.e-17, 8.e-11)
            ax2.set_xlabel('Distance from Central Galaxy (kpc)', fontsize = fs)
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
            ax.set_xlabel('Elapsed Time (Myr)', fontsize = fs)
            ax.set_ylim(1.e3, 1.e9)


    axes[0].annotate(name, (0.05, 0.88), ha = 'left', va = 'bottom', xycoords = 'axes fraction', fontsize = 25, color = 'black', fontweight = 'bold')
    axes[1].annotate('true CGM', (0.90, 0.12), ha = 'right', va = 'bottom', xycoords = 'axes fraction', fontsize = 18, color = 'grey')
    axes[1].annotate('spherically-averaged CGM', (0.90, 0.10), ha = 'right', va = 'top', xycoords = 'axes fraction', fontsize = 18, color = 'red')


    fig.tight_layout()
    fig.subplots_adjust(hspace = 0.10)
    #print ('saving %s'%haloname)
    if haloname == 'halo_004123': 
          hm =  0.48 
          sm =  5.27
    if haloname == 'halo_002878': 
          hm =  0.73 
          sm =  6.72
    if haloname == 'halo_002392': 
          hm =  0.52 
          sm =  2.83
    if haloname == 'halo_005036': 
          hm =  0.33 
          sm =  3.28
    if haloname == 'halo_005016': 
          hm =  0.16 
          sm =  1.07
    if haloname == 'halo_008508': 
          hm =  0.14 
          sm =  1.58
    print ('%s %.2f  %.2f  %.2f %.2f  %.2f  %.2f  %.2f  %.2f  %.2f'%(haloname, smooth, hm, sm, smoothed_M_5[-1], smoothed_M_16[-1], smoothed_M_50[-1], smoothed_M_84[-1],smoothed_M_95[-1], tot_ms_average))
    if haloname == 'halo_008508':
        fig_dir = '/Users/rsimons/Dropbox/foggie/figures/for_paper'

        fig.savefig(fig_dir + '/%s_nref11c_nref9f_%.1f.png'%(haloname, smooth), dpi = 300)
    print ('saving')
    fig.savefig('/Users/rsimons/Dropbox/foggie/figures/simulated_plunges/%s_nref11c_nref9f_%.1f.png'%(haloname, smooth), dpi = 300)








if __name__ == '__main__':

    halonames = array([('halo_002392', 'Hurricane', 'DD0581'), 
                       ('halo_002878', 'Cyclone',  'DD0581'), 
                       ('halo_004123', 'Blizzard',  'DD0581'), 
                       ('halo_005016', 'Squall',  'DD0581'), 
                       ('halo_005036', 'Maelstrom',  'DD0581'), 
                       ('halo_008508', 'Tempest',  'DD0487')])




    halonames = halonames[-1:]

    nrays = 100
    halonames = halonames
    ms = {}
    #average density profile
    #        fit_x = log10(np.linspace(10, 300, 1000))
    #        fit_y = m*fit_x + b#*fit_x + bb
    ms['halo_002392'] = [-1.588654056892834, -24.615833933058113]
    ms['halo_002878'] = [-0.9224813099551425, -25.899655180513736]
    ms['halo_004123'] = [-2.0563930627738847, -23.745455437461473]
    ms['halo_005016'] = [ -1.6095668505200567, -24.717412198736657]
    ms['halo_005036'] = [-1.5884352273933084, -24.64630120455833]
    ms['halo_008508'] = [ -1.5188196045400442, -24.967176740412583]



    '''
    for smooth in [5., 10., 15., 20., 30.][:1]:
        for (haloname, name, DDname) in halonames:
            ray = create_plunging_tunnels(haloname, DDname, nrays = nrays, smooth = smooth)
    '''

    smooths = [0.5, 1., 3., 5., 10., 15., 25.]
    smooths = [2., 7., 20.]
    smooths = [12., 30., 50.]
    #smooths = [1.0, 3.0, 5.0, 10.0]
    #for smooth in smooths:
    #smooth = 1.
    for (haloname, name, DDname) in halonames:
        Parallel(n_jobs = -1)(delayed(create_plunging_tunnels)(haloname = haloname, DDname = DDname,  nrays = nrays, smooth = smooth) for smooth in smooths)
        #for smooth in smooths:
        #    simulate_plunging_orbits(haloname,DDname,  ms, nrays = nrays, smooth = smooth)
        Parallel(n_jobs = -1)(delayed(simulate_plunging_orbits)(haloname,DDname,  ms, nrays = nrays, smooth = smooth) for smooth in smooths)
        Parallel(n_jobs = -1)(delayed(plot_plunging_orbits)(haloname, name, DDname, nrays = nrays, smooth = smooth) for smooth in smooths)

    #Parallel(n_jobs = 2)(delayed(create_plunging_tunnels)(haloname, DDname, nrays = nrays) for (haloname,  name,DDname) in halonames)
    #Parallel(n_jobs = -1)(delayed(simulate_plunging_orbits)(haloname,DDname,  ms, nrays = nrays, smooth = 1.) for (haloname, name, DDname) in halonames)
    #Parallel(n_jobs = -1)(delayed(simulate_plunging_orbits)(haloname,DDname,  ms, nrays = nrays, smooth = 10.) for (haloname, name, DDname) in halonames)
    '''
    for smooth in [0.1, 1., 5., 10., 15., 20., 30.]:
        for (haloname, name, DDname) in halonames:
            simulate_plunging_orbits(haloname,DDname,  ms, smooth = 30.)

        #Parallel(n_jobs = -1)(delayed(plot_plunging_orbits)(haloname, name, DDname,   nrays = nrays) for (haloname, name, DDname) in halonames)
        for (haloname, name, DDname) in halonames:
            plot_plunging_orbits(haloname, name, DDname,  smooth = 30.)

    '''















