from glob import glob
import matplotlib as mpl
import os
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
from numpy import *
import warnings
import matplotlib.colors as colors
import copy
import time
import astropy.units as u






def make_rprof_all(halo):
    indir  = '/Users/rsimons/Dropbox/foggie/angular_momentum/profiles/{}/rdist'.format(halo)
    outdir = '/Users/rsimons/Dropbox/foggie/angular_momentum/profiles/{}'.format(halo)

    all_rprof = {}
    fls = sort(glob(indir + '/Lprof_{}_DD????_rdist.npy'.format(halo)))
    DDs = np.array([fl.split('_')[-2].strip('DD') for fl in fls])
    mtypes = ['cold', 'warm', 'warmhot', 'hot', 'stars', 'young_stars', 'dm']
    for (DD, fl) in zip(DDs, fls):
        all_rprof[DD] = {}
        a = np.load(fl, allow_pickle = True)[()]
        for mtype in mtypes: 
            all_rprof[DD][mtype] = a[mtype]['rprof']
    np.save(outdir + '/all_rprof_{}.npy'.format(halo), all_rprof)
halos = ['5036', '5016', '8508', '2392', '4123', '2878']

for halo in halos:
    Parallel(n_jobs = 6)(delayed(make_rprof_all)(halo = halo) for halo in halos)

