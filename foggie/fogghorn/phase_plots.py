'''
    Filename: phase_plots.py
    Author: Cassi
    Created: 6-12-24
    Last modified: 6-12-24 by Cassi
    This file works with fogghorn_analysis.py to make a set of phase plots.
'''

from foggie.fogghorn.header import *

def den_temp_phase(ds, region, args):
    '''Makes a 2D histogram of density and temperature in the region of interest.'''

    output_filename = args.save_directory + '/' + args.snap + '_density_temperature_phase.png'

    density = np.log10(region['gas', 'density'].in_units('g/cm**3').v)
    temperature = np.log10(region['gas','temperature'].in_units('K').v)
    mass = np.log10(region['gas','cell_mass'].in_units('Msun').v)
    vol = np.log10(region['gas','cell_volume'].in_units('kpc**3').v)

    fig = plt.figure(figsize=(16,6), dpi=200)
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    ax1.hist2d(density, temperature, bins=(200,200), cmap='Blues', norm=matplotlib.colors.LogNorm(), weights=mass)
    ax2.hist2d(density, temperature, bins=(200,200), cmap='Blues', norm=matplotlib.colors.LogNorm(), weights=vol)
    ax1.set_xlabel('log Density [g/cm$^3$]', fontsize=16)
    ax1.set_ylabel('log Temperature [K]', fontsize=16)
    ax1.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14)
    ax1.set_title('Mass-weighted', fontsize=16)
    ax2.set_title('Volume-weighted', fontsize=16)
    ax2.set_xlabel('log Density [g/cm$^3$]', fontsize=16)
    ax2.set_ylabel('log Temperature [K]', fontsize=16)
    ax2.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()