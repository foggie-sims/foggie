import numpy as np
import glob
from glob import glob
from astropy.io import ascii
from numpy import *
import yt
from yt.units import kpc
from scipy import interpolate



def running(DD, mass, half_window = 10):
    percs = []
    for d in DD:
        gd = where(abs(DD - d) < half_window)[0]

        percs.append(np.percentile(mass[gd], [16, 50, 84]))
    return np.array(percs)



halos = ['8508',
         '2878',
         '2392',
         '5016',
         '5036',
         '4123']


sat_cat = ascii.read('/Users/rsimons/Dropbox/foggie/catalogs/satellite_properties.cat')
combine_all = np.load('/Users/rsimons/Dropbox/foggie/catalogs/sat_track_locations/combine_all.npy', allow_pickle = True)[()]


rp_tracks = {}


for h, halo in enumerate(halos):
    rp_tracks[halo] = {}
    sat_cat_halo = sat_cat[sat_cat['halo'] == int(halo)]
    for sat in sat_cat_halo['id']:
        rp_tracks[halo][sat] = {}
        if sat == '0': continue
        tracks = combine_all[halo][sat]
        min_DD = combine_all[halo][sat]['min_DD']
        max_DD = combine_all[halo][sat]['max_DD']
        enter_DD = combine_all[halo][sat]['enter_DD']

        ram_all = []
        time_all = []

        for DD in np.arange(enter_DD, max_DD):
            try: track = tracks[DD]
            except: continue
            v = yt.YTArray(np.sqrt(track['vx']**2. + \
                                   track['vy']**2. + \
                                   track['vz']**2.), 'km/s')
            
            dx = v * yt.YTArray(5, 'Myr')
            start_ray = 2.
            gd = where((track['ray_dist'] > start_ray) & (track['ray_dist'] < start_ray + dx.to('kpc').value))[0]  

            vel_ram = track['ray_vel'][gd]
            den_ram = track['ray_den'][gd]
            vel_ram[vel_ram > 0] = 0.
            ram_all.append(mean(vel_ram)**2. * mean(den_ram))            
            time_all.append(track['time'].value - tracks['enter_time'].value)
        rp_tracks[halo][sat]['time'] = np.array(time_all)
        rp_tracks[halo][sat]['ram'] = np.array(yt.YTArray(ram_all).to('dyne*cm**-2'))

        mom_gained = []
        if not np.isnan(time_all[0]):
            ram_all = yt.YTArray(ram_all).to('dyne*cm**-2')
            #axes.ravel()[cnt].plot(time_all, ram_all, color = 'black', linewidth = 1)
            interp = interpolate.interp1d(time_all, ram_all)
            dt = 1.e-4
            time_interp = np.arange(min(time_all), max(time_all), dt)
            ram_interp = interp(time_interp)
            dmom = (yt.YTArray(ram_interp, 'dyne*cm**-2') * yt.YTArray(dt, 'Gyr')).to('Msun*km/s/kpc**2')
            tot_mom = sum(dmom)
            mom_gained.append(dmom)
            rp_tracks[halo][sat]['ram_interp'] = yt.YTArray(ram_interp, 'dyne*cm**-2') 
            rp_tracks[halo][sat]['mom_interp'] = mom_gained[0]
            rp_tracks[halo][sat]['time_interp'] = time_interp
        else:            
            rp_tracks[halo][sat]['mom_interp'] = [np.nan]
            rp_tracks[halo][sat]['time_interp'] = [np.nan]

    
        rp_tracks[halo][sat]['dt'] =  yt.YTArray(dt, 'Gyr')





np.save('/Users/rsimons/Dropbox/foggie/catalogs/sat_track_locations/rp_refinebox.npy', rp_tracks)



























