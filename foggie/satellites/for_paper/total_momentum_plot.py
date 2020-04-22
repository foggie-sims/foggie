import astropy
from astropy.io import ascii
import matplotlib.pyplot as plt
import yt
import numpy as np
import numpy as np
from astropy.modeling.models import Sersic2D
import matplotlib.pyplot as plt
from astropy.constants import G
import astropy.units as u
from foggie.utils.consistency import halo_dict
from numpy import pi
from numpy import *

np.random.seed(1)

plt.rcParams['text.usetex'] = True






def Re_lange(mass, a = 6.347, b = 0.327, sigma = 0.10):
  return 10**(np.log10(a * (mass.value * 1.e-10)**b) + np.random.normal(0, sigma))






 
plt.close('all')
plt.ioff()

cat = ascii.read('/Users/rsimons/Dropbox/foggie/from_desktop/outputs/plunge_tunnels/final_catalog.cat')

fig_dir = '/Users/rsimons/Dropbox/foggie/figures/for_paper'

fig, ax = plt.subplots(1,1, figsize = (8,5))

ax.errorbar(np.log10(cat['hmass'] * 1.e12), cat['p50'], yerr = [cat['p50'] - cat['p16'], cat['p84'] - cat['p50']],  fmt = 'o', color = 'black', lw = 2.5, markersize = 8)

ax.plot(np.log10(cat['hmass'] * 1.e12), cat['average'], marker = 's', color = 'red', markersize = 10, linestyle = 'None')


for c in cat:
    xcoord = np.log10(c['hmass'] * 1.e12) - 0.025

    ycoord = c['p16']
    print (xcoord, ycoord)
    haloname = halo_dict[c['name'].strip('halo_00')]
    #if haloname == 'Blizzard': xcoord+=0.035
    ax.annotate(haloname, (xcoord, ycoord), ha = 'left', va = 'bottom', color = 'black', rotation = 90, xycoords = 'data', fontweight = 'bold')




toy_models = [(1.e10 * u.Msun, "RP-stripped beyond R$_{e}$ in toy satellites of \n" + r"M$_*$/M$_{\odot}$ = 10$^{10}$"),\
              (1.e9 * u.Msun,   r"10$^{9}$"),\
              (1.e8 * u.Msun,   r"10$^{8}$"),\
              (1.e7 * u.Msun,    r"10$^{7}$")]



mbary_to_mtot = 1/2.
mstar_to_mgas = 1/1.


for m, (mstar, ann_str) in enumerate(toy_models):
    mgas = mstar/mstar_to_mgas
    Re_arr = np.array([Re_lange(mstar) for i in np.arange(1000)])*u.kpc

    alpha = Re_arr/1.67
    sigma_0 = mgas/pi/alpha**2./2.
    sigma_re = sigma_0 * np.exp(-(Re_arr/alpha))

    mtot = (mstar + mgas)/mbary_to_mtot/2.
    vesc = np.sqrt(2*G * mtot/(Re_arr))


    log_mom_n1 = log10((vesc * sigma_re).to('Msun * km * s**-1 * kpc**-2').value)


    sigma_re = mgas/7.2/np.pi/(Re_arr**2.)

    mtot = (mstar + mgas)/mbary_to_mtot
    vesc = np.sqrt(2*G * mtot/(Re_arr))

    log_mom_n4 = np.log10((vesc * sigma_re).to('Msun * km * s**-1 * kpc**-2').value)


    log_mom_tot = concatenate((log_mom_n1, log_mom_n4))
    mom_tot_perc = np.percentile(log_mom_tot, [16, 50, 84])

    ax.axhspan(ymin = mom_tot_perc[0], ymax = mom_tot_perc[-1], xmin = 0, xmax = 1, color = 'grey', zorder = 0., alpha = 0.3)
    #ax.axhline(y = log_mom, xmin = 0.15, xmax = 1.0, color = 'grey', linestyle = 'dashed')
    xann = 11.98
    yann = mom_tot_perc[0] * 0.998
    fs = 16
    if m == len(toy_models): ax.annotate(ann_str, (xann, yann), ha = 'right', va = 'bottom', fontsize = fs, xycoords = 'data', color = 'grey', fontweight = 'bold')
    else: ax.annotate(ann_str, (xann, yann), ha = 'right', va = 'bottom', fontsize = fs, xycoords = 'data', color = 'grey', fontweight = 'bold')








ax.set_ylabel(r'$\log$ Momentum Imparted (M$_{\odot}$ km s$^{-1}$ kpc$^{-2}$)', fontsize = 15)
ax.set_xlabel(r'$\log$ M$_{200}$/M$_{\odot}$ $({z=2})$', fontsize = 15)

#ax.set_xlim(0.3, 7)
ax.set_xlim(11.0, 12)
ax.set_xticks(np.arange(11, 12.2, 0.2))
ax.set_ylim(6.6, 9.8)

from mpl_toolkits.axes_grid.inset_locator import inset_axes


ax_inset = inset_axes(ax,width=2.0, height=0.8,
                    bbox_to_anchor=(0.00, 1.0),
                    bbox_transform=ax.transAxes, loc=2, borderpad=0)

ax_inset.errorbar([0.5], [0.5], yerr = [[0.15], [0.30]], color = 'black', marker = 'o')
ax_inset.plot([0.5], [0.65], color = 'red', marker = 's')
#ax_inset.annotate("16th", (0.55, 0.3), ha = 'left', va = 'center', color = 'black')
ax_inset.annotate("true CGM", (0.52, 0.5), ha = 'left', va = 'center', color = 'black')
ax_inset.annotate("spherically-averaged CGM", (0.52, 0.65), ha = 'left', va = 'center', color = 'red')
#ax_inset.annotate("94th", (0.55, 0.65), ha = 'left', va = 'center', color = 'black')

#ax_inset.axis('off')

ax_inset.set_ylim(0.25, 0.9)
ax_inset.set_xlim(0.45, 0.8)

ax_inset.set_xticks([])
ax_inset.set_yticks([])


fig.tight_layout()
fig.savefig(fig_dir + '/momentum_versus_mass_halonames_test3.png', dpi = 400)









