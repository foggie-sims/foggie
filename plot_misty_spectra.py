import numpy as np
import MISTY
import sys
import os
import glob

import matplotlib as mpl
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})
mpl.rcParams['font.family'] = 'stixgeneral'
mpl.rcParams['font.size'] = 6.
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

from astropy.table import Table
from astropy.io import fits
from astropy.convolution import Gaussian1DKernel, convolve
import astropy.units as u

from spectacle.modeling import Resample

# from consistency import *

def plot_misty_spectra(hdulist, **kwargs):
    outname = kwargs.get('outname', 'test.png')

    ## 0.0267 = (2 km/s)/ c) *4005 angstrom

    ## how many lines are there?
    Nlines = np.int(hdulist[0].header['Nlines'])
    print "there are ", Nlines, ' lines'
    # start by setting up plots
    fig = plt.figure(dpi=300)
    fig.set_figheight(11)
    fig.set_figwidth(6)
    # creates grid on which the figure will be plotted
    gs = gridspec.GridSpec(Nlines, 1)

    zsnap = np.median(hdulist[3].data['redshift'])  ## hack
    zmin, zmax = (zsnap-0.004), (zsnap+0.004)
    vmin, vmax = -1000, 1000

    for line in np.arange(Nlines):
        ax_spec = fig.add_subplot(gs[line, 0])
        try:
            redshift = (hdulist[line+2].data['wavelength'] / hdulist[line+2].header['restwave']) - 1
            print('redshift: ',np.min(redshift), np.max(redshift))
            velocity = (redshift - zsnap) * 299792.458
            flux = hdulist[line+2].data['flux']
            print('flux for line ',hdulist[line+2].header['LINENAME'],': ',np.min(flux),np.max(flux))
            # v_conv = convolve(velocity, g)
            # ax_spec1.step(v_conv, flux, color="#4575b4",lw=2)
            ax_spec.step(redshift, flux, color='darkorange',lw=1)
            ax_spec.text(zmin + 0.0001, 0, hdulist[line+2].header['LINENAME'], fontsize=10.)
        except:
            print('plotting faaaaailed :-()')
        # nstep = np.round((vmax - vmin)/2.)  ## 2 km/s
        # new_vel = np.linspace(np.min(velocity), np.max(velocity), nstep) * u.km / u.s
        # new_flux = Resample(velocity, new_vel)(flux)
        # ax_spec.step(new_vel, new_flux, color='purple',lw=1)
        ax_spec.xaxis.set_major_locator(ticker.NullLocator())
        plt.xlim(zmin, zmax)
        plt.ylim(-0.05, 1.05)
        plt.subplots_adjust(wspace=None, hspace=None)
    fig.tight_layout()
    plt.savefig(outname)
    plt.close(fig)


if __name__ == "__main__":

    long_dataset_list = glob.glob(os.path.join(".", 'hlsp*los.fits'))
    dataset_list = long_dataset_list

    for filename in dataset_list:
        plotname = '.' + filename.strip('los.fits') + 'los.png'
        print('plotting spectra in ', filename, ' and saving as ', plotname)
        hdulist = fits.open(filename)
        plot_misty_spectra(hdulist, outname=plotname)
        hdulist.close()
