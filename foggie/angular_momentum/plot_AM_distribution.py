import numpy as np
from numpy import *
import glob
from glob import glob
from joblib import Parallel, delayed
import os
import argparse




def get_lbl_name(name):
    if name == 'hot': lbl_name = 'hot ($>10^6$ K)'
    elif name == 'warmhot': lbl_name = 'warm-hot ($10^5-10^6$ K)'
    elif name == 'warm': lbl_name = r'warm ($1.5\times 10^4-10^5$ K)'
    elif name == 'cold': lbl_name = r'cold ($<1.5\times 10^4$ K)'
    elif name == 'stars': lbl_name = 'stars'
    elif name == 'dm': lbl_name = 'dark matter'
    elif name == 'all': lbl_name = 'all gas'
    return lbl_name


def make_adist_plot(DDname, ddtime, mass_types, figname, overwrite = False):
    print (DDname)
    fl = '/nobackupp2/rcsimons/foggie/angular_momentum/profiles/8508/Lprof_8508_%s.npy'%DDname

    figname = fl.replace('npy', 'png').replace('/Lprof_', '/figures/%s/%s_'%(figname, figname))
    if (not overwrite) & (os.path.exists(figname)): return

    L_all = np.load(fl, allow_pickle = True)[()]

    fig, axes = plt.subplots(2,3, figsize = (9, 6))
    z = float(ddtime[1,ddtime[0] == float(DDname.strip('DD'))])

    for nn, (low_temp, high_temp, name, clr) in enumerate(mass_types):
        ax = axes.ravel()[nn]
        nbins = 1000

        xvar, xmn, xmx, xlbl =           L_all[name]['adist']['thel'], -180, 180, r'$\theta_{\mathrm{L}}$ (deg.)'
        yvar, ymn, ymx, ylbl =           L_all[name]['adist']['phil'], 0, 180, r'$\phi_{\mathrm{L}}$ (deg.)'

        weights = L_all[name]['adist']['ltot']
        cmap = plt.cm.viridis
        binsx = np.linspace(xmn, xmx, nbins)
        binsy = np.linspace(ymn, ymx, nbins)

        hst = histogram2d(xvar, yvar, nbins, [[xmn, xmx], [ymn, ymx]], weights = weights)[0]
        vmin, vmax = np.percentile(hst.ravel(), [2, 98])
        ax.hist2d(xvar, yvar, bins = [binsx, binsy], \
                  norm = matplotlib.colors.Normalize(vmin, vmax),\
                  weights = weights, cmap = cmap)
        ax.annotate(name, (0.98, 0.05), xycoords = 'axes fraction', ha = 'right', va = 'bottom', \
                    color = 'white', fontweight = 'bold', fontsize = 22)
        if nn ==0: ax.annotate('z = %.2f \n%s'%(z, DDname), (0.05, 0.05), xycoords = 'axes fraction', ha = 'left', va = 'bottom', \
                    color = 'white', fontsize = 10)

    for ax in axes[:,0]:
        ax.set_ylabel(ylbl, fontsize = 15)
    for ax in axes[:,1:].ravel():
        ax.set_yticklabels([''])
        
    for ax in axes[1]:
        ax.set_xlabel(xlbl, fontsize = 15)
    for ax in axes[0].ravel():
        ax.set_xticklabels([''])
    ax.set_yticks([0, 45, 90, 135, 180])
        
    fig.subplots_adjust(hspace = 0.05, wspace = 0.05)
    fig.set_dpi(300)
    fig.tight_layout()
    fig.savefig(figname)
    plt.close(fig)

def make_accum_plot(DDname, ddtime, mass_types, figname, Lmax = -99, Mmax = -99, overwrite = False):
    print (DDname)
    fl = '/nobackupp2/rcsimons/foggie/angular_momentum/profiles/8508/Lprof_8508_%s.npy'%DDname

    figname = fl.replace('npy', 'png').replace('/Lprof_', '/figures/%s/%s_'%(figname, figname))
    if (not overwrite) & (os.path.exists(figname)): return

    fig, ax = plt.subplots(3,1, figsize = (6,9))
    L_all = np.load(fl, allow_pickle = True)[()]
    nbins = len(L_all['cold']['rprof']['r'])
    y1 = np.zeros(nbins)
    m1 = np.zeros(nbins)
    z = float(ddtime[1,ddtime[0] == float(DDname.strip('DD'))])

    for (low_temp, high_temp, name, clr) in mass_types:
        r = L_all[name]['rprof']['r']
        Lx = L_all[name]['rprof']['Lx']
        Ly = L_all[name]['rprof']['Ly']    
        Lz = L_all[name]['rprof']['Lz']
        m = L_all[name]['rprof']['mass']
            
        L_shell = np.stack(([Lx, Ly, Lz]), axis = 1)
        
        Lx_cs = cumsum(Lx)
        Ly_cs = cumsum(Ly)
        Lz_cs = cumsum(Lz)
        
        L_cs = np.sqrt(Lx_cs**2. + Ly_cs**2. + Lz_cs**2.)
        m_cs = cumsum(m)
        y2 = y1 + L_cs
        m2 = m1 + m_cs
        lbl_name = get_lbl_name(name)
        ax[0].fill_between(r, y1 = y1, y2 = y2, label = lbl_name, color = clr)
        ax[1].fill_between(r, y1 = m1, y2 = m2, label = lbl_name, color = clr)
        y1 = y2
        m1 = m2
        loc = argmin(abs(r.value-5.))
        L_disk = np.array([sum(L_all['cold']['rprof']['L%s'%s][:loc]) for s in ['x', 'y', 'z']])
        L_disk/=np.linalg.norm(L_disk)

        theta = np.arccos(np.array([np.dot(L_shl, L_disk)/np.linalg.norm(L_shl) for L_shl in L_shell]))* 180./pi
        ax[2].plot(r, theta, label = lbl_name, color = clr)

        
    fs = 12
    for a in ax:
        a.set_xlabel('radial distance (kpc)', fontsize = fs)

    ax[0].set_ylabel('L($<$r, g cm$^2$ s$^{-1}$)', fontsize = fs)
    ax[1].set_ylabel('M($<$r, M$_{\odot}$)', fontsize = fs)
    ax[2].set_ylabel(r'$\theta$ (deg.)', fontsize = fs)
    ax[2].set_ylim(0, 180)
    #ax[2].axhline(y = 90, color = 'black')
    for a in ax:
        handles, labels = a.get_legend_handles_labels()
        #a.legend(handles[::-1], labels[::-1],loc = 2)
        a.set_xlim(0,100)
    if Lmax < 0:
        ax[0].set_ylim(0, ax[0].get_ylim()[1])
    else:
        ax[0].set_ylim(0, Lmax)

    if Mmax < 0:
        ax[1].set_ylim(0, ax[1].get_ylim()[1])
    else:
        ax[1].set_ylim(0, Mmax)



    ax[0].annotate('z = %.2f \n%s'%(z, DDname), (0.05, 0.95), xycoords = 'axes fraction', ha = 'left', va = 'top', \
                   color = 'black', fontweight = 'bold', fontsize = 12)

    fig.set_dpi(300)
    fig.tight_layout()
    fig.savefig(figname)
    plt.close(fig)

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='''identify satellites in FOGGIE simulations''')
    parser.add_argument('-n1', '--n1', metavar='n1', type=int, action='store')
    parser.set_defaults(n1=0)
    parser.add_argument('-n2', '--n2', metavar='n2', type=int, action='store')
    parser.set_defaults(n2=0)
    parser.add_argument('-n3', '--n3', metavar='n3', type=int, action='store')
    parser.set_defaults(n3=1)
    parser.add_argument('-n_jobs', '--n_jobs', metavar='n_jobs', type=int, action='store')
    parser.set_defaults(n_jobs=-1)
    parser.add_argument('-overwrite', '--overwrite', dest='overwrite', action='store_true',
                        help='just use the pwd?, default is no')
    parser.set_defaults(overwrite=False)
    parser.add_argument('-run_series', '--run_series', dest='run_series', action='store_true',
                        help='just use the pwd?, default is no')
    parser.set_defaults(overwrite=False)

    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()
    import matplotlib
    import matplotlib.pyplot as plt

    plt.rcParams['text.usetex'] = True

    mass_types = [(-1., -1., 'dm', 'black'),
              (-1., -1., 'stars', 'goldenrod'),
              (0., 1.5e4, 'cold', 'darkblue'),
             (1.5e4, 1.e5, 'warm', 'blue'),
             (1.e5, 1.e6, 'warmhot', 'red'),
             (1.e6, 1.e10, 'hot', 'darkred'),              
             ]
    halo = '8508'
    ddtime = np.load('/nobackupp2/rcsimons/foggie_momentum/catalogs/DD_time.npy', allow_pickle = True)[()]

    ddnums = np.arange(args.n1, args.n2, args.n3)
    n_jobs = args.n_jobs
    overwrite = args.overwrite
    if args.run_series:
        for ddnum in ddnums:
            make_adist_plot('DD%.4i'%ddnum, ddtime, mass_types,     figname = 'thel-phil', overwrite = overwrite)

    else:
        Parallel(n_jobs = n_jobs)(delayed(make_adist_plot)('DD%.4i'%ddnum, ddtime, mass_types,     figname = 'thel-phil', overwrite = overwrite) for ddnum in ddnums)
        Parallel(n_jobs = n_jobs)(delayed(make_accum_plot)('DD%.4i'%ddnum, ddtime, mass_types,     figname = 'rprof_all',     Lmax = 1.e74, Mmax = 4.e11, overwrite = overwrite) for ddnum in ddnums)
        Parallel(n_jobs = n_jobs)(delayed(make_accum_plot)('DD%.4i'%ddnum, ddtime, mass_types[1:], figname = 'rprof_baryons', Lmax = 3.e73, Mmax = 7.e10, overwrite = overwrite) for ddnum in ddnums)
        Parallel(n_jobs = n_jobs)(delayed(make_accum_plot)('DD%.4i'%ddnum, ddtime, mass_types[2:], figname = 'rprof_gas',     Lmax = 3.e73, Mmax = 2.e10 , overwrite = overwrite) for ddnum in ddnums)















