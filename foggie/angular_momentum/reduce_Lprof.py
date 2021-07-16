import numpy as np
from numpy import *
import glob
from glob import glob
from joblib import Parallel, delayed
import os
import argparse



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
    parser.add_argument('--halo', metavar='halo', type=str, action='store',
                        help='which halo? default is 8508 (Tempest)')
    parser.set_defaults(halo="8508")

    args = parser.parse_args()
    return args





if __name__ == '__main__':
    args = parse_args()

    nbins = 200
    for DD in np.arange(args.n1, args.n2, 1):
        L_all = np.load('/nobackupp2/rcsimons/foggie/angular_momentum/profiles/%s/Lprof_%s_DD%.4i.npy'%(args.halo,args.halo, DD), allow_pickle = True)[()]
        L_all_new = L_all.copy()
        nbins = 200
        for name in L_all_new.keys():
            xvar, xmn, xmx = L_all[name]['adist']['thel'], -180, 180
            yvar, ymn, ymx = L_all[name]['adist']['phil'],    0, 180
            rvar, rmn, rmx = L_all[name]['adist']['r'],       0, 100
            L_weights = L_all[name]['adist']['ltot']
            M_weights = L_all[name]['adist']['mass']
            L_hst = np.histogramdd((rvar, xvar, yvar), bins = (np.linspace(rmn, rmx, nbins), np.linspace(xmn, xmx, nbins), np.linspace(ymn, ymx, nbins)), weights = L_weights)
            M_hst = np.histogramdd((rvar, xvar, yvar), bins = (np.linspace(rmn, rmx, nbins), np.linspace(xmn, xmx, nbins), np.linspace(ymn, ymx, nbins)), weights = M_weights)

            L_all_new[name]['adist'] = {}
            L_all_new[name]['adist']['hst_bins'] = L_hst[1]
            L_all_new[name]['adist']['L_hst']    = L_hst[0]
            L_all_new[name]['adist']['M_hst']    = M_hst[0]


        np.save('/nobackupp2/rcsimons/foggie/angular_momentum/profiles/%s/Lprof_%s_DD%.4i_reduced.npy'%(args.halo,args.halo, DD), L_all_new)
















