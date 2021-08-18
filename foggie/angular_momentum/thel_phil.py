import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from glob import glob
from astropy.table import Table
from joblib import Parallel, delayed
import multiprocessing as multi
import argparse
from utils import *
import numpy as np
from numpy import *
from astropy.table import Table
mpl.use('Agg')
plt.rcParams['text.usetex'] = False
plt.rcParams['xtick.labelsize'] = 6
plt.rcParams['ytick.labelsize'] = 6


import copy
plt.ioff()

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-profdir', '--profdir', metavar='profdir', type=str, action='store',
                        default = "/Users/rsimons/Dropbox/foggie/angular_momentum/profiles")

    parser.add_argument('-halo', '--halo', metavar='halo', type=str, action='store',
                        default = "8508")

    parser.add_argument('-run_series', '--run_series', dest='run_series', action='store_true',
                        default=False)

    parser.add_argument('-protect', '--protect', dest='protect', action='store_true',
                        default=False)
    parser.add_argument('-outdir', '--outdir', metavar='outdir', type=str, action='store',
                        default="/Users/rsimons/Dropbox/foggie/angular_momentum/figures/thel_phil")

    parser.add_argument('-ddmin', '--ddmin', metavar='ddmin', type=int, action='store',
                        default=-99)

    parser.add_argument('-ddmax', '--ddmax', metavar='ddmax', type=int, action='store',
                        default=-99)

    parser.add_argument('-cores', '--cores', metavar='cores', type=int, action='store',
                        default=1)

    parser.add_argument('-situation', '--situation', metavar='situation', type=str, action='store',
                        default='inner')

    parser.add_argument('-halo_info_dir', '--halo_info_dir', metavar='halo_info_dir', type=str, action='store',
                        default='/Users/rsimons/Dropbox/git/foggie/foggie/halo_infos')

    parser.add_argument('-system', '--system', metavar='system', type=str, action='store',
                        default='pleiades_raymond')


    args = parser.parse_args()
    return args




def thel_phil(DD, args, mass_types, halo_c_v):
        print ('creating thel_phil plots for %s DD%.4i..'%(args.halo, DD))
        figname = '%s/%s/thel_phil_DD%.4i_%s.png'%(args.outdir, args.halo, DD, args.situation)
        if (args.protect) & (os.path.exists(figname)): return
        prof_fl = '%s/%s/Lprof_%s_DD%.4i.npz'%(args.profdir, args.halo, args.halo, DD)
        Lprof    = np.load(prof_fl, allow_pickle = True)['a'][()]

        fig, axes = plt.subplots(2,4, figsize = (8, 4), dpi = 350,facecolor = 'white')

        
        cm = copy.copy(mpl.cm.get_cmap("viridis"))
        cm.set_bad('k')
        map_type = 'L'
        for mm, mtype in enumerate(mass_types):
            ax  = axes.ravel()[mm]
            if args.situation == 'outflow':
                ###fast outflow, vr > 250 km/s
                hst_full = Lprof[mtype]['r_dist']['%s_hst'%map_type][:10,5:]
                if ('stars' in mtype) | ('dm' in mtype): continue
            if args.situation == 'inflow':
                ###metal-poor inflow, Z < 0.02 Zsun & vr < -100 km/s
                hst_full = Lprof[mtype]['r_dist']['%s_hst'%map_type][:30,:2,:1]
                if ('stars' in mtype) | ('dm' in mtype): continue
            if args.situation == 'inner':
                hst_full = Lprof[mtype]['r_dist']['%s_hst'%map_type][:10]
            if args.situation == 'full':
                hst_full = Lprof[mtype]['r_dist']['%s_hst'%map_type]
        
            dim_tuple = tuple(np.arange(hst_full.ndim-2))
            hst_center  = np.rot90(np.nansum(hst_full, axis = dim_tuple))            
            hst_rvl = hst_center.ravel()
            vmn, vmx = 0.0, np.nanpercentile(hst_rvl, [99.5])[0]
            ax.imshow(hst_center, cmap = cm, vmin = vmn, vmax = vmx)
            ax.imshow(np.log10(hst_center), cmap = cm, vmin = 65, vmax = 70)
            fs = 12
            ax.annotate(mtype.replace('_', ' '), (0.98, 0.05), xycoords = 'axes fraction', ha = 'right', va = 'bottom', \
                        color = 'white', fontweight = 'bold', fontsize = fs)
            z = float(halo_c_v['col2'][halo_c_v['col3'] == 'DD%.4i'%DD])
        axes.ravel()[-1].annotate('%s 11c9f\nDD%.4i\nz = %.2f'%(args.halo, DD,z), (0.80, 0.30), xycoords = 'axes fraction', ha = 'right', va = 'bottom', \
                                  color = 'black', fontweight = 'bold', fontsize = fs)

            
        for aa, ax in enumerate(axes.ravel()):
            ax.set_xlim(0, 199)
            ax.set_ylim(199, 0)


            xtck_use  = np.arange(-180, 240, 60)
            #xtck_use  = np.arange(-90, 180, 90)
            xtcks_real = np.interp(xtck_use, [-180, 180],  [0, 199])
            ax.set_xticks(xtcks_real)

            ytck_use  = np.arange(0, 240, 60)
            ytcks_real = np.interp(ytck_use, [0, 180],  [0, 199])
            ax.set_yticks(ytcks_real)

            binsx = np.arange(199)
            binsy = np.arange(199)

            YY, XX = np.meshgrid(binsy, binsx)


            if aa > 3: 
                ax.set_xlabel(r'$\theta_{\mathrm{L}}$ (deg.)')
                tcklbl = ['%i'%tck for tck in xtck_use]
                tcklbl[0] = ''
                tcklbl[2] = ''
                tcklbl[4] = ''
                tcklbl[6] = ''
                #tcklbl[0] = ''
                #tcklbl[-1] = ''
                #if aa > 4: tcklbl[0] = ''
                ax.set_xticklabels(tcklbl)
            else:
                ax.set_xticklabels([])

            if aa%4 == 0: 
                ax.set_ylabel(r'$\phi_{\mathrm{L}}$ (deg.)')
                ax.set_yticklabels(['%i'%tck for tck in ytck_use])
            else:
                ax.set_yticklabels([])

        #axes[1].contour(YY, XX, Lhst_center/np.max(Lhst_center), 
        #           levels = [0.10], zorder = 10, colors = 'white', 
        #           alpha = 0.5, linewidth = 0.5)
        axes[1,3].axis('off')
        
        fig.subplots_adjust(hspace = 0.08, wspace = 0.08, top = 0.95, right = 0.95)
        #fig.tight_layout()
        fig.savefig(figname)
        plt.close(fig)
        


if __name__ == '__main__':
    args = parse_args()
    mass_types = ['cold', 'warm', 'warmhot', 'hot', 'young_stars', 'stars', 'dm']

    if args.system == 'pleiades_raymond':
        args.halo_info_dir = '/nobackupp2/rcsimons/git/foggie/foggie/halo_infos'
        args.profdir       = '/nobackupp2/rcsimons/foggie/angular_momentum/profiles'
        args.outdir        = '/nobackupp2/rcsimons/foggie/angular_momentum/figures/thel_phil'
    halo_c_v = Table.read('%s/00%s/nref11c_nref9f/halo_c_v'%(args.halo_info_dir, args.halo), format = 'ascii')[1:]
    if (args.ddmax == -99) & (args.ddmin > 0): args.ddmax = args.ddmin  + 1


    if not args.run_series:
        fls = np.sort(glob('%s/%s/Lprof*npz'%(args.profdir, args.halo)))
        print ('number of files:', len(fls))
        DDs = [int(fl.split('_')[-1].rstrip('.npz').lstrip('DD')) for fl in fls]
        Parallel(n_jobs = args.cores, backend='multiprocessing')(delayed(thel_phil)(DD, args, mass_types, halo_c_v) for DD in DDs)

    else:
        for DD in np.arange(args.ddmin, args.ddmax, 1):
            thel_phil(DD, args, mass_types, halo_c_v)




