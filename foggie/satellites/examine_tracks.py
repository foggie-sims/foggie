import numpy as np
import glob
from glob import glob
from astropy.io import ascii
from numpy import *
import yt
from yt.units import kpc
import matplotlib.pyplot as plt
plt.ioff()
plt.close('all')


halos = ['8508',
         '2878',
         '2392',
         '5016',
         '5036',
         '4123']

#halos = ['8508']


def running(DD, mass, half_window = 10):
    percs = []
    for d in DD:
        gd = where(abs(DD - d) < half_window)[0]

        percs.append(np.percentile(mass[gd], [16, 50, 84]))
    return np.array(percs)

sat_cat = ascii.read('/Users/rsimons/Desktop/foggie/catalogs/satellite_properties.cat')
for h, halo in enumerate(halos):
    sat_cat_halo = sat_cat[sat_cat['halo'] == int(halo)]
    fls = glob('/Users/rsimons/Desktop/foggie/catalogs/sat_track_locations/%s/%s_DD????.npy'%(halo, halo))
    fls = np.sort(fls)

    for sat in sat_cat_halo['id']:
        fig, axes = plt.subplots(1,1, figsize = (6,6))
        axes2 = axes.twinx()
        inner_all, outer_all, DD_all, ram_all = [], [], [], []

        for f, fl in enumerate(fls):
            track = np.load(fl, allow_pickle = True)[()]
            track_sat = track[sat]
            DD = int(fl.split('_')[-1].strip('.npy').strip('DD'))
            if f == 0: enter_DD = DD

            if np.isnan(track_sat['x']): continue
            if not track_sat['in_refine_box']: enter_DD = DD

            if (min(track_sat['cold_mass_dist']) > 0.25) | (np.isnan(min(track_sat['cold_mass_dist']))): continue
            DD_all.append(DD)
            for (r1, r2) in [(0, 0.3), (0.3, 0.5)]:
                gd = where((track_sat['cold_mass_dist'] > r1) & (track_sat['cold_mass_dist'] < r2))[0]
                cold_r1 =  track_sat['cold_gas_mass'][gd[0]]
                cold_r2 =  track_sat['cold_gas_mass'][gd[-1]]
                shell_volume = 4./3 * pi*(r2**3. - r1**3.) * kpc**3.
                cold_gas_density = (cold_r2 - cold_r1)/shell_volume

                if r1 == 0: inner_all.append(cold_gas_density)
                else: outer_all.append(cold_gas_density)
            gd = where(track_sat['ray_dist'] > 1)[0]  
            vel_ram = track_sat['ray_vel'][gd]
            den_ram = track_sat['ray_den'][gd]
            vel_ram[vel_ram > 0] = 0.
            ram_all.append(mean(vel_ram)**2. * mean(den_ram))




        DD_all = np.array(DD_all)
        inner_all = np.array(inner_all)
        outer_all = np.array(outer_all)
        ram_all = np.array(ram_all)


        axes.scatter(DD_all, inner_all, color = 'darkblue',  s = 4, alpha = 0.1)
        axes.scatter(DD_all, outer_all, color = 'darkred',   s = 4, alpha = 0.1)
        axes2.scatter(DD_all, ram_all, color = 'black',   s = 4, alpha = 0.1)




        inner_percs = running(DD_all, inner_all)
        outer_percs = running(DD_all, outer_all)
        ram_percs   = running(DD_all, ram_all)


        axes.plot(DD_all, inner_percs[:,1], color = 'darkblue', linestyle = '-')
        axes.plot(DD_all, outer_percs[:,1], color = 'darkred', linestyle = '-')
        axes2.plot(DD_all, ram_percs[:,1], color = 'black', linestyle = '-', zorder = 10)
        axes.fill_between(DD_all, y1 = inner_percs[:, 0], y2 = inner_percs[:, 2], color = 'darkblue', alpha = 0.6)
        axes.fill_between(DD_all, y1 = outer_percs[:, 0], y2 = outer_percs[:, 2], color = 'darkred', alpha =  0.6)
        axes2.fill_between(DD_all, y1 = ram_percs[:, 0], y2 = ram_percs[:, 2], color = 'black', alpha =  0.6, zorder = 10)




        #axes.axvline(x = enter_DD, color = 'grey', linestyle = 'dashed', zorder = 20)


        enter_DD = max([axes.get_xlim()[0], enter_DD])
        mid = (enter_DD - axes.get_xlim()[0])/(axes.get_xlim()[1] - axes.get_xlim()[0])


        '''
        axes.annotate("",
                    xy=(0, 0.01), xycoords='axes fraction',
                    xytext=(mid, 0.01), textcoords='axes fraction',
                    arrowprops=dict(edgecolor = "black", linewidth = 5, arrowstyle = "-",
                              connectionstyle="arc3, rad=0"),)
        '''
        axes.annotate("",
                    xy=(mid, 0.01), xycoords='axes fraction',
                    xytext=(1.0, 0.01), textcoords='axes fraction',
                    arrowprops=dict(edgecolor = "black", linewidth = 5, arrowstyle = "-"),)



        if mid + 0.02 < 0.75:
            axes.annotate("inside refine box", (min([mid + 0.02, 0.8]), 0.04), \
                         xycoords = 'axes fraction', ha = 'left', va = 'center', color = 'black')
        else:
            axes.annotate("inside\nrefine\nbox", (0.98, 0.07), \
                         xycoords = 'axes fraction', ha = 'right', va = 'center', color = 'black')


        axes2.set_ylim(1.e-28, 1.e-18)
        axes.annotate('%s-%s'%(halo, sat), (0.95, 0.95), xycoords = 'axes fraction', ha = 'right', va = 'top', fontsize = 20)
        axes.set_xlabel('DD')
        axes.set_ylabel(r'cold gas mass density (M$_{\odot}$ kpc$^{-3}$)')
        axes2.set_ylabel(r'ram pressure')
        axes.set_yscale('log')
        axes2.set_yscale('log')
        fig.tight_layout()
        fig.savefig('/Users/rsimons/Desktop/foggie/figures/cold_mass_evolution/%s_%s.png'%(halo, sat), dpi = 300)













