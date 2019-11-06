# calculate the radial density profile for a list of ions
# 10/25/2019, Yong Zheng, UCB.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'stixgeneral'
from foggie.mocky_way.core_funcs import calc_mean_median_3sig_2sig_1sig

run_data = False
plot_data = True
#ion_list = ['HI', 'SiII', 'SiIII', 'SiIV', 'NV', 'CII', 'CIV',
#            'OVI', 'OVII', 'OVIII', 'NeVII', 'NeVIII']

ion_list = ['SiIII', 'OVI']

if run_data == True:
    from foggie.mocky_way.core_funcs import prepdata
    ds, ds_paras = prepdata('DD2175')
    halo = ds.sphere(ds_paras['halo_center'], (120, 'kpc'))

    nion_list = []
    from foggie.utils import consistency
    for ion_tag in ion_list:
        field = consistency.species_dict[ion_tag]
        nion = halo[field]
        nion_list.append(nion)

    # position and position vector of each cell
    halo_center = ds_paras['halo_center']
    x = halo["gas", "x"].in_units("code_length")
    y = halo["gas", "y"].in_units("code_length")
    z = halo["gas", "z"].in_units("code_length")
    los_x = (x - halo_center[0]).in_units('kpc')
    los_y = (y - halo_center[1]).in_units('kpc')
    los_z = (z - halo_center[2]).in_units('kpc')
    los_r = np.sqrt(los_x**2 + los_y**2 + los_z**2)

    ### Now binned the data
    rion = los_r.flatten()
    dr = 2
    rbins = np.mgrid[0:122:dr]
    for ion_tag, nion in zip(ion_list, nion_list):
        print(ion_tag)
        n_mean = np.zeros(rbins.size-1)
        n_med = np.zeros(rbins.size-1)
        n_1sig_up = np.zeros(rbins.size-1)
        n_1sig_low = np.zeros(rbins.size-1)
        n_2sig_up = np.zeros(rbins.size-1)
        n_2sig_low = np.zeros(rbins.size-1)
        n_3sig_up = np.zeros(rbins.size-1)
        n_3sig_low = np.zeros(rbins.size-1)

        threesig = 0.9973
        twosig = 0.95
        onesig = 0.68

        for ir in range(rbins.size-1):

            rin = rbins[ir]
            rout = rbins[ir+1]
            ind = np.all([rion>=rin, rion<rout], axis=0)
            i_nion = nion[ind]
            i_nion = i_nion[i_nion > 0]

            data_stat = calc_mean_median_3sig_2sig_1sig(i_nion)
            n_mean[ir] = data_stat['mean']
            n_med[ir] = data_stat['median']
            n_3sig_up[ir] = data_stat['3sig_up']
            n_3sig_low[ir] = data_stat['3sig_low']
            n_2sig_up[ir] = data_stat['2sig_up']
            n_2sig_low[ir] = data_stat['2sig_low']
            n_1sig_up[ir] = data_stat['1sig_up']
            n_1sig_low[ir] = data_stat['1sig_low']

        ### save to fits file
        ##### now saving the data ####
        import astropy.io.fits as fits
        c1 = fits.Column(name='rbins (kpc)', array=rbins[:-1], format='D')
        c2 = fits.Column(name='n_mean', array=n_mean, format='D')
        c3 = fits.Column(name='n_median', array=n_med, format='D')
        c4 = fits.Column(name='n_1sig_up', array=n_1sig_up, format='D')
        c5 = fits.Column(name='n_1sig_low', array=n_1sig_low, format='D')
        c6 = fits.Column(name='n_2sig_up', array=n_2sig_up, format='D')
        c7 = fits.Column(name='n_2sig_low', array=n_2sig_low, format='D')
        c8 = fits.Column(name='n_3sig_up', array=n_3sig_up, format='D')
        c9 = fits.Column(name='n_3sig_low', array=n_3sig_low, format='D')

        all_cols = [c1, c2, c3, c4, c5, c6, c7, c8, c9]
        t = fits.BinTableHDU.from_columns(all_cols)
        fig_dir = 'figs/nr_Nr/fits/'
        tb_name = 'nref11n_nref10f_DD2175_all_nr_%s.fits'%(ion_tag)
        save_to_file = '%s/%s'%(fig_dir, tb_name)
        print("I am saving it to ", save_to_file)
        t.writeto(save_to_file, overwrite=True)

#######################
if plot_data == True:

    for ion_tag in ion_list:
        # ok, now let's plot things in a pretty way
        ion_file = 'figs/nr_Nr/fits/nref11n_nref10f_DD2175_all_nr_%s.fits'%(ion_tag)

        from astropy.table import Table
        ion_data = Table.read(ion_file, format='fits')
        rbins = ion_data['rbins (kpc)']
        n_mean = np.log10(ion_data['n_mean'])
        n_median = np.log10(ion_data['n_median'])
        n_1sig_up = np.log10(ion_data['n_1sig_up'])
        n_1sig_low = np.log10(ion_data['n_1sig_low'])
        n_2sig_up = np.log10(ion_data['n_2sig_up'])
        n_2sig_low = np.log10(ion_data['n_2sig_low'])
        n_3sig_up = np.log10(ion_data['n_3sig_up'])
        n_3sig_low = np.log10(ion_data['n_3sig_low'])

        ## plot ##
        ### plot the mean, std, and confidence levels ###
        fig = plt.figure(figsize=(5, 3.5))
        ax = fig.add_subplot(111)
        ax.fill_between(rbins, n_3sig_up, n_3sig_low, edgecolor=None,
                    facecolor='k', alpha=0.1, label=None)
        ax.fill_between(rbins, n_2sig_up, n_2sig_low, edgecolor=None,
                        facecolor='k', alpha=0.15, label=None)
        ax.fill_between(rbins, n_1sig_up, n_1sig_low, edgecolor=None,
                    facecolor='k', alpha=0.25, label=None)
        ax.plot(rbins, n_mean, color=plt.cm.Reds(0.6), lw=3, ls='--', label='Mean')
        ax.plot(rbins, n_median, color='k', lw=0.8, ls='-', label='Median')

        fontsize = 18
        # ax.set_xlim(0, 10) # in unit of kpc
        # ax.set_ylim(1e-10, 1e-4) # in unit of cm-3
        # ax.set_yscale('log')
        ax.legend(fontsize=fontsize-2, loc='upper right')
        ax.set_xlabel('r (kpc)', fontsize=fontsize)
        ax.set_ylabel(r'log [n(r) (cm$^{-3}$)]', fontsize=fontsize)
        ax.minorticks_on()

        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize-2)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize-2)
        ax.set_title(ion_tag, fontsize=fontsize)
        ax.set_xlim(0, 100)
        ax.set_ylim(-19, -5)
        ax.grid(linestyle='--', color=plt.cm.Greys(0.5), alpha=0.5)
        fig.tight_layout()
        figname = 'figs/nr_Nr/nref11n_nref10f_DD2175_all_nr_%s.pdf'%(ion_tag)
        fig.savefig(figname)
        print(figname)
        plt.close()
