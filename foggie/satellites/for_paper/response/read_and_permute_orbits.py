import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from astropy.convolution import convolve_fft, Gaussian1DKernel

inputs = [('8508', 44, 493), 
          ('2878', 398, 583), 
          ('5036', 272, 586),
          ('4123', 355, 589)]
results = {}

for (halo, ddmin, ddmax) in inputs:
    results[halo] = {}

    cat_min  = np.load('./tars/%s/%s_DD%.4i.npy'%(halo, halo, ddmin), allow_pickle = True)[()]
    sats_use = []
    for sat in cat_min.keys():
        if sat == '0': continue
        sats_use.append(sat)
        '''
        try:
            if cat_min[sat]['in_refine_box']:
                sats_use.append(sat)
        except:
            pass
        '''
    print (halo, ddmin, ddmax)
    print (sats_use)

    names = ['t', 'dd', 'x', 'y', 'z', 'r', 'vx', 'vy', 'vz', 'peri_index']

    for sat in sats_use:
        results[halo][sat] = {}
        for name in names: 
            results[halo][sat][name] = []


    for dd in np.arange(ddmin, ddmax):
        cat  = np.load('./tars/%s/%s_DD%.4i.npy'%(halo, halo, dd), allow_pickle = True)[()]
        cat_0 = cat['0']
        for sat in sats_use:
            results[halo][sat]['dd'].append(dd)
            results[halo][sat]['t'].append((dd - ddmin) * (5.))
            x = cat[sat]['x'] - cat_0['x']
            y = cat[sat]['y'] - cat_0['y']
            z = cat[sat]['z'] - cat_0['z']
            r = np.sqrt(x**2. +  y**2. +  z**2.)
            results[halo][sat]['x'].append(x)
            results[halo][sat]['y'].append(y)
            results[halo][sat]['z'].append(z)
            results[halo][sat]['r'].append(r)           
            results[halo][sat]['vx'].append(cat[sat]['vx'] - cat_0['vx'])
            results[halo][sat]['vy'].append(cat[sat]['vy'] - cat_0['vy'])
            results[halo][sat]['vz'].append(cat[sat]['vz'] - cat_0['vz'])

    for sat in sats_use:
        x = np.array(results[halo][sat]['x'])
        y = np.array(results[halo][sat]['y'])
        z = np.array(results[halo][sat]['z'])
        r = np.sqrt(x**2. +  y**2. +  z**2.)
        dd = np.array(results[halo][sat]['dd'])
        kern = Gaussian1DKernel(3.)
        r_conv = convolve_fft(r, kern)
        r_conv[np.isnan(r)] = np.nan
        peaks, _ = find_peaks(1./r_conv, height=0.)



        for p in peaks: 
            if r[p] < 40.:
                results[halo][sat]['peri_index'].append(p)

        if True:
            #plot the orbits, and pericenters
            t = np.array(results[halo][sat]['t'])
            per_index = results[halo][sat]['peri_index']
            fig, ax = plt.subplots(1,1, figsize = (10,10))
            ax.plot(t, r, 'k-')
            ax.plot(t, r_conv, 'k--', alpha = 0.3)
            ax.set_ylim(0, 200.)
            ax.set_xlabel('time (Myr)')
            ax.set_ylabel('distance from center (kpc)')

            for p in per_index: ax.axvline(x = t[p], color = 'blue', linestyle = '-', lw = 3.)
            fig.savefig('/Users/rsimons/temp/{}_{}.png'.format(halo, sat))

np.save('satellite_orbits.npy', results)

'''
if True:
    c
    for halo in results.keys():
        for sat in results[halo].keys():
            x = np.array(results[halo][sat]['x'])
            y = np.array(results[halo][sat]['y'])
            z = np.array(results[halo][sat]['z'])
            r = np.sqrt(x**2. +  y**2. +  z**2.)
'''







