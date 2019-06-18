from __future__ import print_function

import numpy as np
import MISTY
import logging
import sys
import os
import glob
from scipy.signal import argrelextrema

import matplotlib as mpl
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 14.
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

from astropy.table import Table
from astropy.io import fits
import astropy.units as u

def plot_misty_spectra(hdulist, **kwargs):
    outname = kwargs.get('outname', 'test.png')
    overplot = kwargs.get('overplot', False)

    ## how many lines are there?
    Nlines = np.int(hdulist[0].header['Nlines'])
    Nlines = 6
    print("there are ", Nlines, ' lines')

    fig = plt.figure(dpi=300)
    fig.set_figheight(11)
    fig.set_figwidth(5)
    gs = gridspec.GridSpec(Nlines, 1)

    zsnap = hdulist[0].header['REDSHIFT']
    zmin, zmax = (zsnap-0.004), (zsnap+0.004)
    vmin, vmax = -300, 300

    for line in range(Nlines):
        
        height = 1./Nlines - 0.012
        ax_spec = fig.add_axes([0.15, 0.06 + (Nlines-line-1)*height, 0.82, height], xlim=(-300, 300), ylim=(-0.05,1.05))

        try:
 
            ext = hdulist[line+2]
            name = ext.header['LINENAME']
            lambda_0 = ext.header['RESTWAVE'] * u.AA
            f_value = ext.header['F_VALUE']
            gamma = ext.header['GAMMA']
            tot_column = ext.header['TOT_COLUMN']
            print(line, name)

            redshift = hdulist[name].data['redshift']
            flux = hdulist[name].data['flux']
            wavelength_obs = hdulist[name].data['wavelength'] * u.AA
            with u.set_enabled_equivalencies(u.equivalencies.doppler_relativistic(lambda_0*(1+zsnap))):
                velocity = (wavelength_obs).to('km/s')

            ax_spec.plot(velocity, np.ones(len(velocity)),color='k',lw=1, ls=":")
            ax_spec.step(velocity, flux, color='#984ea3')
            vi = argrelextrema(flux, np.less)[0]
            vmin = velocity[vi[flux[vi] < 0.95]]
            ax_spec.plot(vmin, np.array(vmin)*0.0 + 1, '|',color='black')
            if overplot:
                ax_spec.step(velocity, flux, color='darkorange', lw=1, ls="--", dashes=(5, 2))
            ax_spec.text(vmin + 25, 0.15, name, fontsize=16.)
            coldens = "N = " + "{:4.2f}".format(tot_column)
            ax_spec.text(vmin + 25, 0, coldens, fontsize=16.)

        except Exception as e:
            logging.error("Plotting failed because: \n%s", e)
        
        if line < (Nlines-1):
            ax_spec.xaxis.set_major_locator(ticker.NullLocator())
        else:
            plt.xlabel('velocity [km/s]',fontsize=24.)

    print("                   ")
    print("before fig commands")
    print("                   ")

    fig.text(0.02, 0.5, 'normalized flux', fontsize=24., va='center', rotation='vertical')
    plt.savefig(outname)
    plt.close(fig)




if __name__ == "__main__":

    long_dataset_list = glob.glob(os.path.join(".", 'hlsp*rd00*ax*v6*lsf.fits.gz'))
    long_dataset_list = glob.glob(os.path.join(".", 'hlsp*rd00*ax*.fits.gz'))

    dataset_list = long_dataset_list
    ### dataset_list = ['./hlsp_misty_foggie_halo008508_nref11n_nref10f_rd0018_axy_dx053.4_dz039.5_v6_lsf.fits.gz']

    for filename in dataset_list:
        plotname = '.' + filename.strip('los.fits.gz') + 'lsf.png'
        print('plotting spectra in ', filename, ' and saving as ', plotname)
        hdulist = fits.open(filename)
        plot_misty_spectra(hdulist, overplot=True, outname=plotname)
        hdulist.close()
