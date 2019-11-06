### calculate the column denisty of a halo from inside view
# with random sightlines using ray from trident
# python column_density_inview_ray.py
#
# 10/31/2019, Yong Zheng, UCB.

import sys
import numpy as np
from foggie.mocky_way.core_funcs import calc_ray_ion_column_density
from foggie.mocky_way.core_funcs import calc_ray_end

ion = sys.argv[1]
ion_list = ['HI', 'SiII', 'SiIII', 'SiIV', 'CII', 'CIV', 'OVI', 'NV',
            'OVII', 'OVIII', 'NeVII', 'NeVIII']
if ion not in ion_list:
    print("Sorry, %s is not on my ion list, please add it first. "%(ion))
    sys.exit()

from foggie.mocky_way.core_funcs import prepdata
ds, ds_paras = prepdata('DD2175')

nlos = 3  # the total number of random sightlines
all_N = []
all_l = []
all_b = []
all_r = []

# now let's do a total number of nlos sightlines, and save it/overwrite it per
# step, so that data would get lost on pleiades
los_ray_start = ds_paras['offcenter_location']

los_l_rand = [30.3, 60, 350.]
los_b_rand = [-80.7, -20.2, 50.]
los_kpc_rand = [10., 34., 110.]

for i in range(nlos):
    #los_l_deg = np.random.uniform(low=0., high=360.)
    #los_b_deg = np.random.uniform(low=-90., high=90.)
    #los_length_kpc = np.random.uniform(low=3., high=160.)

    los_l_deg = los_l_rand[i]
    los_b_deg = los_b_rand[i]
    los_length_kpc = los_kpc_rand[i]

    los_ray_end, los_unit_vector = calc_ray_end(ds, ds_paras, los_l_deg, los_b_deg,
                                                los_ray_start, los_length_kpc)

    Nion, other_info = calc_ray_ion_column_density(ds, ion,
                                                   los_ray_start,
                                                   los_ray_end)

    ## let's save it step by step
    all_N.append(Nion)
    all_l.append(los_l_deg)
    all_b.append(los_b_deg)
    all_r.append(los_length_kpc)

    # save the data to fits file
    ##### now saving the data ####
    import astropy.io.fits as fits
    c1 = fits.Column(name='N', array=np.asarray(all_N), format='D')
    c2 = fits.Column(name='l', array=np.asarray(all_l), format='D')
    c3 = fits.Column(name='b', array=np.asarray(all_b), format='D')
    c4 = fits.Column(name='r', array=np.asarray(all_r), format='D')

    all_cols = [c1, c2, c3, c4]
    t = fits.BinTableHDU.from_columns(all_cols)
    fig_dir = 'figs/Nr_inview/fits'
    tb_name = 'nref11n_nref10f_DD2175_N%s_inview_local.fits'%(ion)

    save_to_file = '%s/%s'%(fig_dir, tb_name)
    print("%s: I am saving it to %s"%(i, save_to_file))
    t.writeto(save_to_file, overwrite=True)
