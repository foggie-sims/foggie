from __future__ import print_function

import numpy as np
import MISTY
import logging
import sys
import os
import glob

import matplotlib as mpl
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})
mpl.rcParams['font.family'] = 'stixgeneral'
mpl.rcParams['font.size'] = 12.
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

from astropy.table import Table
from astropy.io import fits
from astropy.convolution import Gaussian1DKernel, convolve
import astropy.units as u
c = 299792.458 * u.Unit('km/s')

from spectacle.analysis import Resample
from spectacle.core.spectrum import Spectrum1DModel

# from consistency import *

def plot_misty_spectra(hdulist, **kwargs):
    outname = kwargs.get('outname', 'test.png')
    overplot = kwargs.get('overplot', False)
    offset = kwargs.get('offset', 0.0)
    box_redshift = kwargs.get('box_redshift', 2.0)

    ### these are the lines we want
    line_list = ['H I 1216', 'H I 919', \
             'Si II 1260', 'Si IV 1394', 'C IV 1548', 'O VI 1032', ]
    ## line_list = ['Si II 1260']
    # Nlines = np.int(hdulist[0].header['Nlines'])
    # print("there are ", Nlines, ' lines')
    # start by setting up plots
    fig = plt.figure(dpi=300)
    fig.set_figheight(12)
    fig.set_figwidth(6)
    gs = gridspec.GridSpec(6, 1)

    try:
        zsnap = np.median(hdulist[3].data['redshift_obs'])  ## hack
    except:
        zsnap = np.median(hdulist[3].data['redshift'])
    zmin, zmax = (zsnap-0.004), (zsnap+0.004)
    vmin, vmax = -700, 700
    print(zsnap)

    for line, line_name in enumerate(line_list):
        ax_spec = fig.add_subplot(gs[line, 0])
        try:
            # Construct spectacle spectrum
            spectrum = Spectrum1DModel(redshift=zsnap)
            ext = hdulist[line_name]
            name = ext.header['LINENAME']
            lambda_0 = ext.header['RESTWAVE'] * u.AA
            f_value = ext.header['F_VALUE']
            gamma = ext.header['GAMMA']

            for i in range(len([x for x in ext.header if 'FITLCEN' in x])):
                delta_v = ext.header['FITLCEN{}'.format(i)] * u.Unit('km/s')
                col_dens = ext.header['FITCOL{}'.format(i)]
                v_dop = ext.header['FITB{}'.format(i)]
                print(ext.header['FITLCEN{}'.format(i)], ext.header['FITCOL{}'.format(i)], ext.header['FITB{}'.format(i)])
                if col_dens > 100.:
                    col_dens = np.log10(col_dens)
                if v_dop < 1000.:
                    v_dop = v_dop*u.Unit('km/s')
                else:
                    v_dop = v_dop*u.Unit('cm/s')
                print('i:',i,delta_v, col_dens, v_dop)
                spectrum.add_line(name=name, lambda_0=lambda_0, f_value=f_value,
                                  gamma=gamma, column_density=col_dens, v_doppler=v_dop,
                                  delta_v=delta_v/(1+box_redshift))
                ax_spec.plot([delta_v.value+offset, delta_v.value+offset], [1.05, 0.95], color='k')

            ### _lsf.fits files have '_obs' while _los.fits files don't
            # ax_spec.step(redshift, flux, color='darkorange',lw=1)

            try:
                redshift = hdulist[line_name].data['redshift']
                flux = hdulist[line_name].data['flux']
                wavelength_obs = hdulist[line_name].data['disp'] * u.AA
            except:
                redshift = hdulist[line_name].data['redshift_obs']
                flux = hdulist[line_name].data['flux_obs']
                wavelength_obs = hdulist[line_name].data['disp_obs'] * u.AA
            with u.set_enabled_equivalencies(u.equivalencies.doppler_relativistic(lambda_0)):
                velocity = (wavelength_obs / (1 + zsnap)).to('km/s') * (1 + zsnap)
            ax_spec.plot(velocity + offset*u.Unit('km/s'), np.ones(len(velocity)),color='k',lw=1, ls=":")
            ax_spec.step(velocity + offset*u.Unit('km/s'), flux, color='#984ea3')
            ax_spec.step(velocity + offset*u.Unit('km/s'), spectrum.flux(velocity), color='darkorange', lw=1, ls="--", dashes=(5, 2))
            if overplot:
                ax_spec.step(hdulist[line_name].data['redshift_obs'], spectrum.flux(hdulist[line_name].data['disp_obs'] * u.AA), color='darkorange', lw=1)

            ax_spec.text(vmin + 50, 0, line_name, fontsize=12.)
        except Exception as e:
            logging.error("Plotting failed because: \n%s", e)
        ## eventually want to plot in velocity space and actually label the bottom axis but :shrug:
        if line < 5:
            ax_spec.xaxis.set_major_locator(ticker.NullLocator())
        plt.xlim(vmin, vmax)
        plt.ylim(-0.05, 1.05)
        plt.subplots_adjust(wspace=None, hspace=None)
    fig.tight_layout()
    plt.savefig(outname)
    plt.close(fig)


if __name__ == "__main__":

    # dataset_list = ['hlsp_misty_foggie_halo008508_nref11n_nref10f_rd0020_axx_i012.3-a3.40_v5_rsp.fits.gz']
    # dataset_list = ['hlsp_misty_foggie_halo008508_nref11n_nref10f_rd0018_axy_i018.2-a2.09_v5_rsp.fits.gz']
    dataset_list = ['hlsp_misty_foggie_halo008508_nref11n_nref10f_rd0018_axx_i010.4-a2.25_v5_rsp.fits.gz']
    output_dir = '/Users/molly/Dropbox/foggie-collab/papers/absorption_peeples/Figures/'

    for filename in dataset_list:
        plotname = output_dir + filename.strip('rsp.fits.gz') + 'rsp.png'
        print('plotting spectra in ', filename, ' and saving as ', plotname)
        hdulist = fits.open(filename)
        plot_misty_spectra(hdulist, overplot=False, outname=plotname, offset=-500.,box_redshift=2.5)
        hdulist.close()
