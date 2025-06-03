'''
    Filename: time_evol_plots.py
    Author: Cassi
    Created: 6-12-24
    Last modified: 3-31-25 by Cassi
    This file works with fogghorn_analysis.py to make the set of plots for halo scaling relations.
    If you add a new function to this scripts, then please also add the function name to the appropriate list at the end of fogghorn/header.py

'''

from foggie.fogghorn.header import *
from foggie.fogghorn.util import *

# --------------------------------------------------------------------------------------------------------------------
def plot_SFMS(args, output_filename):
    '''
    Plots the star-forming main sequence of the simulated galaxy -- one point per output on the plot -- and compares
    to best fit curves from observations at different redshifts.
    '''

    # Read in the previously-saved halo information
    data = Table.read(args.save_directory + '/central_galaxy_info.txt', format='ascii.ecsv')

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(1,1,1)

    colormap = plt.cm.rainbow_r
    normalize = matplotlib.colors.Normalize(vmin=0., vmax=13.759)

    # Plot the observed best-fit values for a handful of redshifts
    Mstar_list = np.arange(8.,12.5,0.5)
    tlist = [13.759, 10.788, 8.657, 5.925, 3.330, 1.566]
    zlist = [0., 0.25, 0.5, 1., 2., 4.]
    SFR_list = []
    for i in range(len(tlist)):
        SFR_list.append([])
        for j in range(len(Mstar_list)):
            SFR_list[i].append((0.84-0.026*tlist[i])*Mstar_list[j] - (6.51-0.11*tlist[i]))  # This equation comes from Speagle et al. (2014), first row of Table 9
        ax.plot(Mstar_list, SFR_list[i], '-', lw=1, color=colormap(normalize(tlist[i])), label='z=%.2f' % zlist[i])
        ax.fill_between(Mstar_list, np.array(SFR_list[i])-0.2, y2=np.array(SFR_list[i])+0.2, color=colormap(normalize(tlist[i])), alpha=0.2)

    # Plot the simulated galaxy's location in stellar mass-SFR space as a scatterplot color-coded by redshift
    ax.scatter(np.log10(data['stellar_mass']), np.log10(data['SFR']), c=data['time']/1000., cmap=colormap, norm=normalize, s=20)

    ax.set_xlabel('$\log M_\star$ [$M_\odot$]', fontsize=16)
    ax.set_ylabel('$\log$ SFR [$M_\odot$/yr]', fontsize=16)
    ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, top=True, right=True)
    ax.legend(loc=2, frameon=False, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    print('Saved figure ' + output_filename)
    plt.close()

# --------------------------------------------------------------------------------------------------------------------
def plot_SMHM(args, output_filename):
    '''
    Plots the stellar mass-halo mass relation of the simulated galaxy -- one point per output on the plot -- and compares
    to best fit curves from observations at different redshifts.
    '''

    # Read in the previously-saved halo information
    data = Table.read(args.save_directory + '/central_galaxy_info.txt', format='ascii.ecsv')

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(1,1,1)

    colormap = plt.cm.rainbow_r
    normalize = matplotlib.colors.LogNorm(vmin=0.5, vmax=13.759)

    # Plot the observed best-fit values for a handful of redshifts
    tlist = [12.447, 5.925, 3.330, 2.182, 1.566, 1.193, 0.947]
    zlist = [0.1, 1., 2., 3., 4., 5., 6.]
    # These values come from digitizing Figure 7 of Behroozi et al. (2013)
    Mhalo_list = [[10.010, 10.379, 10.746, 11.131, 11.500, 11.874, 12.239, 12.627, 13.002, 13.380, 13.751, 14.122, 14.510, 14.876],
                  [10.876, 11.257, 11.675, 12.000, 12.374, 12.750, 13.129, 13.504, 13.875, 14.251, 14.633],
                  [11.266, 11.637, 11.998, 12.375, 12.762, 13.124, 13.505, 13.875],
                  [11.510, 11.875, 12.247, 12.624, 13.002, 13.386],
                  [11.376, 11.753, 12.119, 12.503, 12.875, 13.251],
                  [11.256, 11.634, 12.006, 12.374, 12.757],
                  [11.136, 11.504, 11.881, 12.251]]
    Mstar_list = [[7.246, 7.750, 8.291, 8.901, 9.697, 10.275, 10.633, 10.821, 10.937, 11.078, 11.152, 11.224, 11.290, 11.372],
                  [8.306, 8.918, 9.882, 10.420, 10.774, 10.955, 11.045, 11.114, 11.151, 11.205, 11.229],
                  [8.780, 9.569, 10.364, 10.789, 10.960, 11.055, 11.102, 11.136],
                  [9.349, 10.091, 10.626, 10.862, 10.970, 11.040],
                  [9.240, 9.981, 10.478, 10.704, 10.822, 10.886],
                  [9.213, 9.904, 10.323, 10.515, 10.644],
                  [9.190, 9.799, 10.161, 10.317]]
    for i in range(len(tlist)):
        ax.plot(Mhalo_list[i], Mstar_list[i], '-', lw=1, color=colormap(normalize(tlist[i])), label='z=%.2f' % zlist[i])
        ax.fill_between(Mhalo_list[i], np.array(Mstar_list[i])-0.15, y2=np.array(Mstar_list[i])+0.15, color=colormap(normalize(tlist[i])), alpha=0.2)

    # Plot the simulated galaxy's location in stellar mass-SFR space as a scatterplot color-coded by redshift
    ax.scatter(np.log10(data['halo_mass']), np.log10(data['stellar_mass']), c=data['time']/1000., cmap=colormap, norm=normalize, s=20)

    ax.set_xlabel('$\log M_\mathrm{halo}$ [$M_\odot$]', fontsize=16)
    ax.set_ylabel('$\log M_\star$ [$M_\odot$]', fontsize=16)
    ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, top=True, right=True)
    ax.legend(loc=2, frameon=False, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    print('Saved figure ' + output_filename)
    plt.close()

# --------------------------------------------------------------------------------------------------------------------
def plot_MZR(args, output_filename):
    '''
    Plots global gas metallicity vs gas mass relation.
    Returns nothing. Saves output as png file
    '''

    # --------- Setting up the figure ---------
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.subplots_adjust(hspace=0.05, wspace=0.05, right=0.95, top=0.95, bottom=0.1, left=0.15)

    # Ayan will add stuff here

    # ---------annotate and save the figure----------------------
    plt.text(0.97, 0.95, 'z = %.2F' % args.current_redshift, ha='right', transform=ax.transAxes, fontsize=args.fontsize)
    plt.savefig(output_filename)
    print('Saved figure ' + output_filename)
    plt.close()
