import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from glob import glob
from astropy.table import Table
from joblib import Parallel, delayed
import argparse
from utils import *
import numpy as np
import copy
plt.ioff()

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-profdir', '--profdir', metavar='profdir', type=str, action='store',
                        default = "/Users/rsimons/Dropbox/foggie/angular_momentum/profiles")

    parser.add_argument('-halo', '--halo', metavar='halo', type=str, action='store',
                        default = "8508")

    parser.add_argument('-run_par', '--run_par', dest='run_par', action='store_true',
                        default=False)

    parser.add_argument('-outdir', '--outdir', metavar='outdir', type=str, action='store',
                        default="/Users/rsimons/Dropbox/foggie/angular_momentum/figures/thel_phil")

    parser.add_argument('-ddmin', '--ddmin', metavar='ddmin', type=int, action='store',
                        default=-99)

    parser.add_argument('-ddmax', '--ddmax', metavar='ddmax', type=int, action='store',
                        default=-99)

    parser.add_argument('-dduse', '--dduse', metavar='dduse', type=int, action='store',
                        default=-99)

    args = parser.parse_args()
    return args




def thel_phil(DD, args, mass_types):
        print ('creating thel_phil plots for %s DD%.4i..'%(args.halo, DD))
        prof_fl = '%s/%s/Lprof_%s_DD%.4i.npz'%(args.profdir, args.halo, args.halo, DD)
        Lprof    = np.load(prof_fl, allow_pickle = True)['a'][()]

        fig, axes = plt.subplots(1,3, figsize = (12, 4), facecolor = 'white')
        figname = '%s/%s/thel_phil_DD%.4i.png'%(args.outdir, args.halo, DD)
        cm = copy.copy(mpl.cm.get_cmap("viridis"))
        cm.set_bad('k')
        map_type = 'L'
        for mm, mtype in enumerate(mass_types[:3]):
            hst_full = Lprof[mtype]['r_dist']['%s_hst'%map_type][:10]
            dim_tuple = tuple(np.arange(hst_full.ndim-2))
            hst_center  = np.rot90(np.nansum(hst_full, axis = dim_tuple))            
            hst_rvl = hst_center.ravel()
            vmn, vmx = 0.0, np.nanpercentile(hst_rvl, [99.5])[0]
            axes[mm].imshow(hst_center, cmap = cm, vmin = vmn, vmax = vmx)
            axes[mm].annotate(mtype.replace('_', ' '), (0.98, 0.05), xycoords = 'axes fraction', ha = 'right', va = 'bottom', \
                        color = 'white', fontweight = 'bold', fontsize = 22)
            
            '''
            Lhst_full = Lprof['warm']['r_dist']['L_hst'][3:10,:2,:1,:]
            Mhst_full = Lprof['warm']['r_dist']['M_hst'][3:10,:2,:1,:]
            dim_tuple = tuple(np.arange(Lhst_full.ndim-2))
            Lhst  = np.rot90(np.nansum(Lhst_full, axis = dim_tuple))
            Mhst  = np.rot90(np.log10(np.nansum(Mhst_full, axis = dim_tuple)))
            
            Lhst_rvl = Lhst.ravel()
            Mhst_rvl = Mhst.ravel()
            if len(Mhst_rvl[isfinite(Mhst_rvl)]) < 3: continue
            vmn, vmx = np.nanpercentile(Mhst_rvl[isfinite(Mhst_rvl)], [5])[0], np.nanpercentile(Mhst_rvl[isfinite(Mhst_rvl)], [95])[0]

            axes[1].imshow(Mhst, cmap = cmp, vmin = vmn, vmax = vmx)
            axes[1].annotate('warm', (0.98, 0.05), xycoords = 'axes fraction', ha = 'right', va = 'bottom', \
                        color = 'white', fontweight = 'bold', fontsize = 22)
            '''
            
            
            
        for ax in axes:
            ax.set_xlim(0, 199)
            ax.set_ylim(199, 0)

            ax.set_xlabel(r'$\theta_{\mathrm{L}}$ (deg.)')
            ax.set_ylabel(r'$\phi_{\mathrm{L}}$ (deg.)')

            xtck_use  = np.arange(-180, 240, 60)
            xtcks_real = np.interp(xtck_use, [-180, 180],  [0, 199])
            ax.set_xticks(xtcks_real)
            ax.set_xticklabels(['%i'%tck for tck in xtck_use])

            ytck_use  = np.arange(0, 240, 60)
            ytcks_real = np.interp(ytck_use, [0, 180],  [0, 199])
            ax.set_yticks(ytcks_real)
            ax.set_yticklabels(['%i'%tck for tck in ytck_use])

            binsx = np.arange(199)
            binsy = np.arange(199)

            YY, XX = np.meshgrid(binsy, binsx)

        #axes[1].contour(YY, XX, Lhst_center/np.max(Lhst_center), 
        #           levels = [0.10], zorder = 10, colors = 'white', 
        #           alpha = 0.5, linewidth = 0.5)

        
        fig.subplots_adjust(hspace = 0.05, wspace = 0.05)
        fig.set_dpi(300)
        fig.tight_layout()
        fig.savefig(figname)
        plt.close(fig)
        


if __name__ == '__main__':
    args = parse_args()
    mass_types = ['cold', 'young_stars', 'stars', 'warm', 'warmhot', 'hot', 'dm']



    if args.dduse > 0: thel_phil(args.dduse, args, mass_types)

    if args.run_par:
        fls = np.sort(glob('%s/%s/Lprof*npz'%(args.profdir, args.halo)))
        DDs = [int(fl.split('_')[-1].rstrip('.npz').lstrip('DD')) for fl in fls]
        Parallel(n_jobs = 5)(delayed(thel_phil)(DD, args, mass_types) for DD in DDs)



























