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

from spectacle.analysis import Resample

# from consistency import *

def plot_misty_spectra(hdulist, **kwargs):
    outname = kwargs.get('outname', 'test.png')
    overplot = kwargs.get('overplot', False)

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
            ### _lsf.fits files have '_obs' while _los.fits files don't
            # ax_spec.step(redshift, flux, color='darkorange',lw=1)
            ax_spec.step(hdulist[line+2].data['redshift_obs'], hdulist[line+2].data['flux_obs'], color='purple', lw=1)
            if overplot:
                ax_spec.step(hdulist[line+2].data['redshift_obs'], hdulist[line+2].data['flux_obs'], color='darkorange', lw=1)
            ax_spec.text(zmin + 0.0001, 0, hdulist[line+2].header['LINENAME'], fontsize=10.)
        except:
            print('plotting faaaaailed :-()')
        ## eventually want to plot in velocity space and actually label the bottom axis but :shrug:
        ax_spec.xaxis.set_major_locator(ticker.NullLocator())
        plt.xlim(zmin, zmax)
        plt.ylim(-0.05, 1.05)
        plt.subplots_adjust(wspace=None, hspace=None)
    fig.tight_layout()
    plt.savefig(outname)
    plt.close(fig)


if __name__ == "__main__":

    long_dataset_list = glob.glob(os.path.join(".", 'hlsp*lsf.fits.gz'))
    dataset_list = long_dataset_list

    for filename in dataset_list:
        plotname = '.' + filename.strip('lsf.fits.gz') + 'lsf.png'
        print('plotting spectra in ', filename, ' and saving as ', plotname)
        hdulist = fits.open(filename)
        plot_misty_spectra(hdulist, overplot=False, outname=plotname)
        hdulist.close()
