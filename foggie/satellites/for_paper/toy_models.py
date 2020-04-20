import astropy
from astropy.io import ascii
import matplotlib.pyplot as plt
import yt
import numpy as np
import numpy as np
from astropy.modeling.models import Sersic1D, Sersic2D
import matplotlib.pyplot as plt
from astropy.constants import G
import astropy.units as u
from foggie.utils.consistency import halo_dict
from numpy import pi
from numpy import *
from scipy.special import iv, kv
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

plt.close('all')

def Re_lange(mass, a = 6.347, b = 0.327, sigma = 0.16):
  return 10**(np.log10(a * (mass.value * 1.e-10)**b)) #+ np.random.normal(0, sigma)


def integrate_model(model):
    dx = 0.03
    res_for_int = [(x, x+dx) for x in np.arange(0, 30, dx)]

    masses_shell = []
    r_shell = []
    for (r0, r1) in res_for_int:
        r_shell.append(np.mean([r0, r1]))
        masses_shell.append((model(np.mean([r0, r1])) * pi * (r1**2. - r0**2.) * u.kpc**2.).value)
    r_shell = np.array(r_shell)
    masses_shell = np.array(masses_shell)
    masses_cumsum = np.cumsum(masses_shell)
    return r_shell,  masses_shell, masses_cumsum

def find_potential(model, r_shell, alpha):
    phi = []
    for r in r_shell:
        y = alpha * r/2.
        I0 = iv(0, y)
        I1 = iv(1, y)
        K0 = kv(0, y)
        K1 = kv(1, y)
        phi_r = -pi * G * model(0.) * r * u.kpc * (I0*K1 - I1 * K0)
        phi.append(phi_r.to('km**2./s**2.').value)

    return np.array(phi) * u.km**2./u.s**2.




toy_models = [(1.e10 * u.Msun, "RP-stripped beyond R$_{e}$ in toy satellites of \n" + r"M$_*$/M$_{\odot}$ = 10$^{10}$"),\
              (1.e9 * u.Msun,   r"10$^{9}$"),\
              (1.e8 * u.Msun,   r"10$^{8}$"),\
              (1.e7 * u.Msun,    r"10$^{7}$")]

toy_models = [1.e7 * u.Msun, 1.e8* u.Msun, 1.e9* u.Msun, 1.e10* u.Msun]


mbary_to_mtot = 1/2.
mstar_to_mgas = 1/1.

fig, ax = plt.subplots(1,1, figsize = (8,8))
clrs = ['black', 'darkblue', 'darkgreen', 'darkred']

for m, mstar in enumerate(toy_models):
    mgas = mstar/mstar_to_mgas
    mtot = (mstar + mgas)/mbary_to_mtot
    log_mom_test = np.linspace(7, 12, 100)
    nits = 3
    f_mass_left_both = np.zeros((nits, len(log_mom_test)))
    for it in arange(nits):    
        Re    = Re_lange(mstar)
        if it == 0: Re-=0.16
        if it == 1: Re = Re
        if it == 2: Re+=0.16

        alpha = Re/1.7

        n1_model_bland = Sersic1D(amplitude = 1. * u.Msun/(u.kpc)**2.,  r_eff = Re, n = 1.)
        r_shell,  masses_shell, masses_cumsum = integrate_model(n1_model_bland)

        n1_model_normalized = Sersic1D(amplitude = (mtot/(masses_cumsum[-1] * u.Msun)) * u.Msun/(u.kpc)**2.,  r_eff = Re, n = 1.)
        r_shell,  masses_shell, masses_cumsum = integrate_model(n1_model_normalized)
        phi = find_potential(n1_model_normalized, r_shell, alpha)

        log_mom_needed = log10((np.sqrt(-2*phi).to('km/s') * n1_model_normalized(r_shell)).to('Msun * km * s**-1 * kpc**-2').value)

        f_mass_left = []
        masses_cumsum_inside_Re = masses_cumsum[r_shell < Re]
        log_mom_needed_inside_Re        = log_mom_needed[r_shell < Re]
        for lmom in log_mom_test:
            gd = where(lmom > log_mom_needed_inside_Re)[0]
            if len(gd) > 0:
                m_total    = masses_cumsum_inside_Re[-1]
                m_left     = masses_cumsum_inside_Re[gd[0]]
                m_stripped = m_total - m_left
                f_mass_left.append(m_left/m_total)
            else:
                f_mass_left.append(1.)

        f_mass_left_both[it] = f_mass_left
    #if it == 0:
    #    lbl = r'$\log_{10}$ M$_{*}$/M$_{\odot}$' + '= %.1f'%(np.log10(mstar.value))
    #else:
    #    lbl = None
    lbl = r'$\log_{10}$ M$_{*}$/M$_{\odot}$' + '= %.1f'%(np.log10(mstar.value))
    ax.fill_between(log_mom_test, y1 = f_mass_left_both[0], y2 = f_mass_left_both[2], color = clrs[m], label = lbl, alpha = 0.7)
    ax.plot(log_mom_test, f_mass_left_both[0], color = clrs[m], label = None, alpha = 1.)
    ax.plot(log_mom_test, f_mass_left_both[2], color = clrs[m], label = None, alpha = 1.)


ax.tick_params(axis="x", labelsize=15)
ax.tick_params(axis="y", labelsize=15)

#ax.set_xlim(0, 2*Re)

ax.annotate('r $\,<$ effective radii', (0.95, 0.98), xycoords = 'axes fraction', ha = 'right', va = 'top', fontsize = 25)

ax.annotate('toy model parameters', (0.05, 0.78), xycoords = 'axes fraction', ha = 'left', va = 'bottom', fontsize = 25, fontweight = 'bold')
ax.annotate('exponential disk\nM$_{\mathrm{baryons}}$/M$_{\mathrm{total}}$ = 0.5\nM$_{\mathrm{star}}$/M$_{\mathrm{gas}}$ = 1.', (0.08, 0.76), xycoords = 'axes fraction', ha = 'left', va = 'top', fontsize = 15)
ax.set_xlabel(r'$\log$ Surface Momentum Imparted' + '\n' + r'(M$_{\odot}$ km s$^{-1}$ kpc$^{-2}$)', fontsize = 20)
ax.set_ylabel(r'M$_{\mathrm{gas, final}}$/M$_{\mathrm{gas, initial}}$', fontsize = 20)

ax.legend(loc = 2)
ax.set_ylim(-0.05, 1.8)
ax.set_yticks(np.arange(0., 2.0, 0.5))
ax.axhline(y = 0.0, color = 'grey', alpha = 0.3)
fig.tight_layout()
fig.savefig('/Users/rsimons/toy_model_RPS_inside_Re.png', dpi = 200)





















