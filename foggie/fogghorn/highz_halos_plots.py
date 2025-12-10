'''
    Filename: highz_halos_plots.py
    Author: Cassi
    Created: 4-14-25
    Last modified: 4-14-25 by Cassi
    This file works with fogghorn_analysis.py to make the set of plots for various scaling relations for all halos in the high resolution region.
    If you add a new function to this script, then please also add the function name to the dictionary in fogghorn_analysis.py.

'''

from foggie.fogghorn.header import *
from foggie.fogghorn.util import *
from datetime import datetime
import seaborn as sns

def get_halo_catalog(ds, args, snap):
    '''This function either checks if the halo catalog already exists for this snapshot 'snap'
    and returns it if so, or runs the halo finder and both saves it to file and returns it.'''

    print(args.directory)
    halo_catalog_path = args.directory + '/halo_catalogs/' + snap + '/' + snap + '.0.h5'
    print('Checking for halo catalog at: ' + halo_catalog_path)
    if os.path.exists(args.directory + '/halo_catalogs/' + snap + '/' + snap + '.0.h5'):
        print('Halo catalog found for snapshot ' + snap)
        hc = yt.load(args.directory + '/halo_catalogs/' + snap + '/' + snap + '.0.h5')
        return hc
    else:
        print('No halo catalog found, creating halo catalog for snapshot ' + snap)

        return None
    
def halos_density_projection(ds, region, args, output_filename):
    '''Plots a density projection with all the halos in the halo catalog
    overplotted as circles.'''

    box_length = ds.quan(500., 'kpc')
    box = ds.r[ds.halo_center_kpc[0]-box_length:ds.halo_center_kpc[0]+box_length, 
            ds.halo_center_kpc[1]-box_length:ds.halo_center_kpc[1]+box_length, 
            ds.halo_center_kpc[2]-box_length:ds.halo_center_kpc[2]+box_length]
    hc = get_halo_catalog(ds, args, args.snap)
    p = yt.ProjectionPlot(ds, 'z', ('gas','density'), weight_field=('gas','density'), data_source=box, center=ds.halo_center_code, width=(500, 'kpc'))
    p.set_cmap('density', density_color_map)
    p.annotate_title(args.snap)
    p.annotate_timestamp(redshift=True)
    p.annotate_halos(hc) 
    p.save(output_filename)

def halos_SMHM(ds, region, args, output_filename):
    '''Plots the stellar-mass halo-mass relation for all halos in the halo catalog.'''

    hc = get_halo_catalog(ds, args, args.snap)
    all_data = hc.all_data()

    total_halo_mass = all_data[('halos','particle_mass')].in_units('Msun')
    total_star_mass = all_data[('halos','total_star_mass')].in_units('Msun')

    colormap = plt.cm.rainbow_r
    normalize = matplotlib.colors.LogNorm(vmin=0.9, vmax=5.)

    # Plot the halos in this snap
    plt.scatter(np.log10(total_halo_mass), np.log10(total_star_mass), color=colormap(normalize(float(ds.current_time.in_units('Gyr')))))

    # Plot the observed best-fit values for a handful of redshifts
    tlist = [3.330, 2.182, 1.566, 1.193, 0.947]
    zlist = [2., 3., 4., 5., 6.]
    # These values come from digitizing Figure 7 of Behroozi et al. (2013)
    Mhalo_list = [[11.266, 11.637, 11.998, 12.375, 12.762, 13.124, 13.505, 13.875],
                  [11.510, 11.875, 12.247, 12.624, 13.002, 13.386],
                  [11.376, 11.753, 12.119, 12.503, 12.875, 13.251],
                  [11.256, 11.634, 12.006, 12.374, 12.757],
                  [11.136, 11.504, 11.881, 12.251]]
    Mstar_list = [[8.780, 9.569, 10.364, 10.789, 10.960, 11.055, 11.102, 11.136],
                  [9.349, 10.091, 10.626, 10.862, 10.970, 11.040],
                  [9.240, 9.981, 10.478, 10.704, 10.822, 10.886],
                  [9.213, 9.904, 10.323, 10.515, 10.644],
                  [9.190, 9.799, 10.161, 10.317]]
    for i in range(len(tlist)):
        # Extrapolate the Behroozi relations linearly to lower Mhalo
        slope = (Mstar_list[i][1] - Mstar_list[i][0]) / (Mhalo_list[i][1] - Mhalo_list[i][0])
        intercept = Mstar_list[i][0] - slope * Mhalo_list[i][0]
        low_Mh = np.array([7.5, Mhalo_list[i][0]])
        low_Mstar = slope * low_Mh + intercept
        plt.plot(Mhalo_list[i], Mstar_list[i], '-', lw=1, color=colormap(normalize(tlist[i])))
        plt.plot(low_Mh, low_Mstar, '--', lw=1, color=colormap(normalize(tlist[i])), label='z=%.2f' % zlist[i])
        plt.fill_between(Mhalo_list[i], np.array(Mstar_list[i])-0.15, y2=np.array(Mstar_list[i])+0.15, color=colormap(normalize(tlist[i])), alpha=0.2)
    plt.xlabel(r'log Halo Mass [$M_\odot$]', fontsize=16)
    plt.ylabel(r'log Stellar Mass [$M_\odot$]', fontsize=16)
    plt.title('$z = %.2f$' % ds.get_parameter('CosmologyCurrentRedshift'))
    plt.axis([7.5,12.5,2,11.5])
    plt.legend(loc=2, frameon=False, fontsize=14)
    plt.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, top=True, right=True)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()

def halos_SFMS(ds, region, args, output_filename):
    '''Plots the star formation rate main sequence relation for all halos in the halo catalog.'''

    hc = get_halo_catalog(ds, args, args.snap)
    all_data = hc.all_data()

    total_star_mass = all_data[('halos','total_star_mass')].in_units('Msun')
    SFR = all_data[('halos','sfr7')].in_units('Msun/yr')

    colormap = plt.cm.rainbow_r
    normalize = matplotlib.colors.LogNorm(vmin=0.9, vmax=5.)

    # Plot the halos in this snap
    plt.scatter(np.log10(total_star_mass), np.log10(SFR), color=colormap(normalize(float(ds.current_time.in_units('Gyr')))))

    # Plot the observed best-fit values for a handful of redshifts
    Mstar_list = np.arange(8.,12.5,0.5)
    low_Mstar_list = np.arange(2.,8.5,0.5)
    tlist = [3.330, 1.566]
    zlist = [2., 4.]
    SFR_list = []
    low_SFR_list = []
    for i in range(len(tlist)):
        SFR_list.append([])
        for j in range(len(Mstar_list)):
            SFR_list[i].append((0.84-0.026*tlist[i])*Mstar_list[j] - (6.51-0.11*tlist[i]))  # This equation comes from Speagle et al. (2014), first row of Table 9
        plt.plot(Mstar_list, SFR_list[i], '-', lw=1, color=colormap(normalize(tlist[i])), label='z=%.2f' % zlist[i])
        plt.fill_between(Mstar_list, np.array(SFR_list[i])-0.2, y2=np.array(SFR_list[i])+0.2, color=colormap(normalize(tlist[i])), alpha=0.2)
        # Extrapolate to lower Mstar
        low_SFR_list.append([])
        for j in range(len(low_Mstar_list)):
            low_SFR_list[i].append((0.84-0.026*tlist[i])*low_Mstar_list[j] - (6.51-0.11*tlist[i]))
        plt.plot(low_Mstar_list, low_SFR_list[i], '--', lw=1, color=colormap(normalize(tlist[i])))
    plt.xlabel(r'log Stellar Mass [$M_\odot$]', fontsize=16)
    plt.ylabel(r'$\log$ SFR [$M_\odot$/yr]', fontsize=16)
    plt.title('$z = %.2f$' % ds.get_parameter('CosmologyCurrentRedshift'))
    plt.axis([2.,12.,-5,4])
    plt.legend(loc=2, frameon=False, fontsize=14)
    plt.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, top=True, right=True)
    plt.tight_layout()
    plt.legend(loc=2, frameon=False, fontsize=14)
    plt.savefig(output_filename, dpi=300)
    plt.close()

def halos_MZR(ds, region, args, output_filename):
    '''Plots the mass-metallicity relation for all halos in the halo catalog.'''

    hc = get_halo_catalog(ds, args, args.snap)
    all_data = hc.all_data()

    halo_metallicity = all_data[('halos','average_metallicity')].in_units('Zsun')
    total_star_mass = all_data[('halos','total_star_mass')].in_units('Msun')

    colormap = plt.cm.rainbow_r
    normalize = matplotlib.colors.LogNorm(vmin=0.9, vmax=5.)

    # Plot the halos in this snap
    plt.scatter(np.log10(total_star_mass), halo_metallicity, color=colormap(normalize(float(ds.current_time.in_units('Gyr')))))

    # Plot the observed best-fit values for a handful of redshifts
    # These data come from Nakajima et al. 2023, Fig. 10 digitized
    tlist = [2.896, 1.962, 1.566]
    zlist = [2.3, 3.3, 4.]
    Mstar_list = [[9.007, 9.186, 9.417, 9.615, 9.828, 10.048, 10.291, 10.449],
                  [9.156, 9.320, 9.488, 9.618, 9.754, 9.914, 10.062, 10.190, 10.325, 10.464],
                  [7.338, 7.675, 7.895, 8.133, 8.310, 9.031, 9.421, 9.567]]
    Z_list = [[0.3324, 0.3743, 0.4371, 0.5063, 0.5815, 0.6756, 0.7947, 0.8883],
              [0.2976, 0.3296, 0.3703, 0.4073, 0.4421, 0.4915, 0.5421, 0.5963, 0.6524, 0.7119],
              [0.0797, 0.0958, 0.1088, 0.1244, 0.1381, 0.2028, 0.2573, 0.2793]]
    for i in range(len(tlist)):
        # Extrapolate the relations linearly to lower Mstar
        slope = (np.log10(Z_list[i][1]) - np.log10(Z_list[i][0])) / (Mstar_list[i][1] - Mstar_list[i][0])
        intercept = np.log10(Z_list[i][0]) - slope * Mstar_list[i][0]
        low_Mstar = np.array([2., Mstar_list[i][0]])
        low_Z = 10**(slope * low_Mstar + intercept)
        plt.plot(Mstar_list[i], Z_list[i], '-', lw=1, color=colormap(normalize(tlist[i])))
        plt.plot(low_Mstar, low_Z, '--', lw=1, color=colormap(normalize(tlist[i])), label='z=%.2f' % zlist[i])
    plt.xlabel(r'log Stellar Mass [$M_\odot$]', fontsize=16)
    plt.ylabel(r'Average Gas Metallicity [$Z_\odot$]', fontsize=16)
    plt.title('$z = %.2f$' % ds.get_parameter('CosmologyCurrentRedshift'))
    plt.axis([2.,12.0,1e-8,1e1])
    plt.yscale('log')
    plt.legend(loc=4, frameon=False, fontsize=14)
    plt.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, top=True, right=True)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()

def halos_gasMHM(ds, region, args, output_filename):
    '''Plots total gas mass vs. halo mass for all halos in the halo catalog.'''

    hc = get_halo_catalog(ds, args, args.snap)
    all_data = hc.all_data()

    total_gas_mass = all_data[('halos','total_gas_mass')].in_units('Msun')
    total_halo_mass = all_data[('halos','particle_mass')].in_units('Msun')

    colormap = plt.cm.rainbow_r
    normalize = matplotlib.colors.LogNorm(vmin=0.9, vmax=5.)

    # Plot the halos in this snap
    plt.scatter(np.log10(total_halo_mass), np.log10(total_gas_mass), color=colormap(normalize(float(ds.current_time.in_units('Gyr')))))

    plt.xlabel(r'log Halo Mass [$M_\odot$]', fontsize=16)
    plt.ylabel(r'log Gas Mass [$M_\odot$]', fontsize=16)
    plt.title('$z = %.2f$' % ds.get_parameter('CosmologyCurrentRedshift'))
    plt.axis([7.5,12.5,5,11.5])
    plt.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, top=True, right=True)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()

def halos_ismMHM(ds, region, args, output_filename):
    '''Plots total gas mass vs. halo mass for all halos in the halo catalog.'''

    hc = get_halo_catalog(ds, args, args.snap)
    all_data = hc.all_data()

    total_ism_gas_mass = all_data[('halos','total_ism_gas_mass')].in_units('Msun')
    total_halo_mass = all_data[('halos','particle_mass')].in_units('Msun')

    colormap = plt.cm.rainbow_r
    normalize = matplotlib.colors.LogNorm(vmin=0.9, vmax=5.)

    # Plot the halos in this snap
    plt.scatter(np.log10(total_halo_mass), np.log10(total_ism_gas_mass), color=colormap(normalize(float(ds.current_time.in_units('Gyr')))))

    plt.xlabel(r'log Halo Mass [$M_\odot$]', fontsize=16)
    plt.ylabel(r'log ISM Gas Mass [$M_\odot$]', fontsize=16)
    plt.title('$z = %.2f$' % ds.get_parameter('CosmologyCurrentRedshift'))
    plt.axis([7.5,12.5,5,11.5])
    plt.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, top=True, right=True)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()

def halos_cgmMHM(ds, region, args, output_filename):
    '''Plots total gas mass vs. halo mass for all halos in the halo catalog.'''

    hc = get_halo_catalog(ds, args, args.snap)
    all_data = hc.all_data()

    total_cgm_gas_mass = all_data[('halos','total_cgm_gas_mass')].in_units('Msun')
    total_halo_mass = all_data[('halos','particle_mass')].in_units('Msun')

    colormap = plt.cm.rainbow_r
    normalize = matplotlib.colors.LogNorm(vmin=0.9, vmax=5.)

    # Plot the halos in this snap
    plt.scatter(np.log10(total_halo_mass), np.log10(total_cgm_gas_mass), color=colormap(normalize(float(ds.current_time.in_units('Gyr')))))

    plt.xlabel(r'log Halo Mass [$M_\odot$]', fontsize=16)
    plt.ylabel(r'log CGM Gas Mass [$M_\odot$]', fontsize=16)
    plt.title('$z = %.2f$' % ds.get_parameter('CosmologyCurrentRedshift'))
    plt.axis([7.5,12.5,5,11.5])
    plt.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, top=True, right=True)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()

def baryon_budget(ds, region, args, output_filename):

    current_datetime = datetime.now()

    hc = get_halo_catalog(ds, args, args.snap)
    all_data = hc.all_data()

    total_halo_mass = all_data["halos", "total_mass"].in_units('Msun')
    total_star_mass = all_data["halos", "total_star_mass"].in_units('Msun')
    total_ism_gas_mass = all_data["halos", "total_ism_gas_mass"].in_units('Msun')
    total_cold_cgm_gas_mass = all_data["halos", "total_cold_cgm_gas_mass"].in_units('Msun')
    total_cool_cgm_gas_mass = all_data["halos", "total_cool_cgm_gas_mass"].in_units('Msun')
    total_warm_cgm_gas_mass = all_data["halos", "total_warm_cgm_gas_mass"].in_units('Msun')
    total_hot_cgm_gas_mass = all_data["halos", "total_hot_cgm_gas_mass"].in_units('Msun')
    total_star_mass = all_data["halos", "total_star_mass"].in_units('Msun')

    print("Total Halo Masses: ", total_halo_mass)
    print("Total Star Masses: ", total_star_mass)


    ism_fraction = total_ism_gas_mass / total_halo_mass
    cold_cgm_fraction = total_cold_cgm_gas_mass / total_halo_mass
    cool_cgm_fraction = total_cool_cgm_gas_mass / total_halo_mass
    warm_cgm_fraction = total_warm_cgm_gas_mass / total_halo_mass
    hot_cgm_fraction = total_hot_cgm_gas_mass / total_halo_mass
    star_fraction = total_star_mass / total_halo_mass

    ism_fraction = total_ism_gas_mass / total_halo_mass

    total_baryon_fraction = (total_ism_gas_mass + total_cold_cgm_gas_mass + total_cool_cgm_gas_mass + total_warm_cgm_gas_mass + total_hot_cgm_gas_mass + total_star_mass) / total_halo_mass

    print("Total Baryon Fraction: ", total_baryon_fraction)

    sns.set_style("whitegrid")

    plt.scatter([0], [0]) 
    plt.xlim(9, 12.5) 
    plt.ylim(0, 0.3) 
    plt.xlabel('Total Halo Mass [Msun]') 
    plt.ylabel('Baryon Fraction') 
    plt.title('FOGGIE v2.0 Baryon Budgets   ' + str(current_datetime)[0:10]  )
    i = 0 # do it once to get the legends right 
    plt.bar(np.log10(total_halo_mass[i]), star_fraction[i], width=0.2, bottom = 0., color='#9e302c', label='Stars') 
    plt.bar(np.log10(total_halo_mass[i]), ism_fraction[i], width=0.2, bottom = star_fraction[i], color='#4a6091', label='ISM') 
    plt.bar(np.log10(total_halo_mass[i]), cold_cgm_fraction[i], width=0.2, bottom = star_fraction[i]+ism_fraction[i], color="#C66D64", label='Cold CGM')
    plt.bar(np.log10(total_halo_mass[i]), cool_cgm_fraction[i], width=0.2, bottom = star_fraction[i]+ism_fraction[i]+cold_cgm_fraction[i], color='#6f427b', label='Cool CGM')
    plt.bar(np.log10(total_halo_mass[i]), warm_cgm_fraction[i], width=0.2, bottom = star_fraction[i]+ism_fraction[i]+cold_cgm_fraction[i]+cool_cgm_fraction[i], color='#659B4d', label='Warm CGM')
    plt.bar(np.log10(total_halo_mass[i]), hot_cgm_fraction[i], width=0.2, bottom = star_fraction[i]+ism_fraction[i]+cold_cgm_fraction[i]+cool_cgm_fraction[i]+warm_cgm_fraction[i], color='#f2dc61', label='Hot CGM')

    for i in np.arange(7): 
        plt.bar(np.log10(total_halo_mass[i]), star_fraction[i], width=0.2, bottom = 0., color='#9e302c')  
        plt.bar(np.log10(total_halo_mass[i]), ism_fraction[i], width=0.2, bottom = star_fraction[i], color='#4a6091')  
        plt.bar(np.log10(total_halo_mass[i]), cold_cgm_fraction[i], width=0.2, bottom = star_fraction[i]+ism_fraction[i], color="#C66D64") 
        plt.bar(np.log10(total_halo_mass[i]), cool_cgm_fraction[i], width=0.2, bottom = star_fraction[i]+ism_fraction[i]+cold_cgm_fraction[i], color='#6f427b') 
        plt.bar(np.log10(total_halo_mass[i]), warm_cgm_fraction[i], width=0.2, bottom = star_fraction[i]+ism_fraction[i]+cold_cgm_fraction[i]+cool_cgm_fraction[i], color='#659B4d') 
        plt.bar(np.log10(total_halo_mass[i]), hot_cgm_fraction[i], width=0.2, bottom = star_fraction[i]+ism_fraction[i]+cold_cgm_fraction[i]+cool_cgm_fraction[i]+warm_cgm_fraction[i], color='#f2dc61') 

    plt.legend(frameon=0, loc='upper right', ncols=3)
    plt.plot([9, 12.5], [0.0461 / 0.285, 0.0461 / 0.285], linestyle='dashed', color='blue')
    
    plt.savefig(output_filename, dpi=300)
    plt.close()

def halos_h2_frac(ds, region, args, output_filename):
    '''Plots average H2 fraction vs. halo mass for all halos in the halo catalog.'''

    if (ds.parameters['MultiSpecies'] == 2): 

        hc = get_halo_catalog(ds, args, args.snap)
        all_data = hc.all_data()

        avg_h2_frac = all_data[('halos','average_fH2')]
        total_halo_mass = all_data[('halos','particle_mass')].in_units('Msun')

        colormap = plt.cm.rainbow_r
        normalize = matplotlib.colors.LogNorm(vmin=0.9, vmax=5.)

        # Plot the halos in this snap
        plt.scatter(np.log10(total_halo_mass), np.log10(avg_h2_frac), color=colormap(normalize(float(ds.current_time.in_units('Gyr')))))

        plt.xlabel(r'log Halo Mass [$M_\odot$]', fontsize=16)
        plt.ylabel(r'log $f_{\mathrm{H}_2}$', fontsize=16)
        plt.title('$z = %.2f$' % ds.get_parameter('CosmologyCurrentRedshift'))
        plt.axis([7.5,12.5,-10,0])
        plt.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, top=True, right=True)
        plt.tight_layout()
        plt.savefig(output_filename, dpi=300)
        plt.close()
    else: 
        print('Snapshot does not have MultiSpecies = 2 so f_H2 plot will not be made')

        
########################################################################
# The rest of this file is for if this file is run independently from fogghorn_analysis.py.
# If the halo catalogs have already been made, this file can be called to create the same plots,
# but combining multiple halos on each plot.
#
# Call this script like this:
#   > python highz_halos_plots.py --directory /Users/clochhaas/Documents/Research/FOGGIE/FOGGHORN/mech_and_h2 --halos ['halo_008508','halo_005016','halo_005036','halo_004123'] --output RD0014 --run tab_mech --all_plots
# where "directory" is the root directory where the halo catalogs are saved, in the structure [directory]/[halos]/[run]/plots/halo_catalogs/[output],
# "halos" is the list of halos to be put on each plot, "output" is the output to use (will be the same across all halos), "run" is the name of the run,
# and "all_plots" produces all the scaling relation plots.

def parse_halos_args():
    '''Parse command line arguments. Returns args object.'''

    parser = argparse.ArgumentParser(description='Plots the same snapshot of all halos on galaxy scaling relations. Halo catalogs must have already been made.')

    parser.add_argument('--directory', metavar='directory', type=str, action='store', default='', help='What is the top level directory where the halo catalogs are stored?')
    parser.add_argument('--save_directory', metavar='save_directory', type=str, action='store', default='.', help='Where do you want to store the plots? Default is to put them in whatever directory this code is run from.')
    parser.add_argument('--output', metavar='output', type=str, action='store', default=None, help='If you want to make the plots for specific output/s then specify those here separated by comma (e.g., DD0030,DD0040). Otherwise (default) it will make plots for ALL outputs in that directory')
    parser.add_argument('--nproc', metavar='nproc', type=int, action='store', default=1, help='How many processes do you want? Default is 1 (no parallelization), if multiple processors are specified, code will run one output per processor')
    parser.add_argument('--halos', metavar='halos', type=str, action='store', default=None, help='What halos do you want on the plots? Use a comma-separated list, no spaces.')
    parser.add_argument('--run', metavar='run', type=str, action='store', default='nref11c_nref9f', help='What run do you want to plot? All halos must have the same run name.')

    parser.add_argument('--all_plots', dest='all_plots', action='store_true', default=False, help='Make all the plots? Default is no.')
    parser.add_argument('--make_plots', metavar='make_plots', type=str, action='store', default='', help='Which plots to make? Comma-separated names of the plotting routines to call. Default is none.')

    args = parser.parse_args()
    return args

def all_halos_SMHM(snap, args):
    '''Plots the stellar-mass halo-mass relation for all halos in the halo catalog.'''

    colormap = plt.cm.rainbow_r
    normalize = matplotlib.colors.LogNorm(vmin=0.9, vmax=5.)

    halo_names = {'halo_008508':'Tempest', 'halo_005016':'Squall', 'halo_005036':'Maelstrom', 'halo_004123':'Blizzard', 'halo_002392':'Hurricane'}
    halo_colors = {'halo_008508':'#8FDC97','halo_005016':'#6A0136','halo_005036':'#188FA7','halo_004123':'#CC3F0C', 'halo_002392':'#D5A021'}

    for i in range(len(args.halo_list)):
        halo = args.halo_list[i]
        hc = yt.load(args.directory + '/' + halo + '/' + args.run + '/plots/halo_catalogs/' + snap + '/' + snap + '.0.h5')
        all_data = hc.all_data()

        total_halo_mass = all_data[('halos','particle_mass')].in_units('Msun')
        total_star_mass = all_data[('halos','total_star_mass')].in_units('Msun')

        # Plot the halos in this snap
        plt.scatter(np.log10(total_halo_mass), np.log10(total_star_mass), color=halo_colors[halo], label=halo_names[halo], s=10)

    # Plot the observed best-fit values for a handful of redshifts
    tlist = [3.330, 2.182, 1.566, 1.193, 0.947]
    zlist = [2., 3., 4., 5., 6.]
    # These values come from digitizing Figure 7 of Behroozi et al. (2013)
    Mhalo_list = [[11.266, 11.637, 11.998, 12.375, 12.762, 13.124, 13.505, 13.875],
                  [11.510, 11.875, 12.247, 12.624, 13.002, 13.386],
                  [11.376, 11.753, 12.119, 12.503, 12.875, 13.251],
                  [11.256, 11.634, 12.006, 12.374, 12.757],
                  [11.136, 11.504, 11.881, 12.251]]
    Mstar_list = [[8.780, 9.569, 10.364, 10.789, 10.960, 11.055, 11.102, 11.136],
                  [9.349, 10.091, 10.626, 10.862, 10.970, 11.040],
                  [9.240, 9.981, 10.478, 10.704, 10.822, 10.886],
                  [9.213, 9.904, 10.323, 10.515, 10.644],
                  [9.190, 9.799, 10.161, 10.317]]
    for i in range(len(tlist)):
        # Extrapolate the Behroozi relations linearly to lower Mhalo
        slope = (Mstar_list[i][1] - Mstar_list[i][0]) / (Mhalo_list[i][1] - Mhalo_list[i][0])
        intercept = Mstar_list[i][0] - slope * Mhalo_list[i][0]
        low_Mh = np.array([7.5, Mhalo_list[i][0]])
        low_Mstar = slope * low_Mh + intercept
        plt.plot(Mhalo_list[i], Mstar_list[i], '-', lw=1, color=colormap(normalize(tlist[i])))
        plt.plot(low_Mh, low_Mstar, '--', lw=1, color=colormap(normalize(tlist[i])), label='z=%.2f' % zlist[i])
        plt.fill_between(Mhalo_list[i], np.array(Mstar_list[i])-0.15, y2=np.array(Mstar_list[i])+0.15, color=colormap(normalize(tlist[i])), alpha=0.2)
    plt.xlabel(r'log Halo Mass [$M_\odot$]', fontsize=14)
    plt.ylabel(r'log Stellar Mass [$M_\odot$]', fontsize=14)
    plt.title('$z = %.2f$' % hc.current_redshift, fontsize=14)
    plt.axis([7.5,12.0,2,10])
    plt.legend(loc=2, ncols=2, frameon=False, fontsize=12)
    plt.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=12, top=True, right=True)
    plt.tight_layout()
    plt.savefig(args.save_directory + '/' + snap + '_SMHM_all-halos.png', dpi=300)
    plt.close()

def all_halos_SFMS(snap, args):
    '''Plots the star formation rate main sequence relation for all halos in the halo catalog.'''

    colormap = plt.cm.rainbow_r
    normalize = matplotlib.colors.LogNorm(vmin=0.9, vmax=5.)

    halo_names = {'halo_008508':'Tempest', 'halo_005016':'Squall', 'halo_005036':'Maelstrom', 'halo_004123':'Blizzard', 'halo_002392':'Hurricane'}
    halo_colors = {'halo_008508':'#8FDC97','halo_005016':'#6A0136','halo_005036':'#188FA7','halo_004123':'#CC3F0C', 'halo_002392':'#D5A021'}

    for i in range(len(args.halo_list)):
        halo = args.halo_list[i]
        hc = yt.load(args.directory + '/' + halo + '/' + args.run + '/plots/halo_catalogs/' + snap + '/' + snap + '.0.h5')
        all_data = hc.all_data()

        total_star_mass = all_data[('halos','total_star_mass')].in_units('Msun')
        SFR = all_data[('halos','sfr7')].in_units('Msun/yr')

        # Plot the halos in this snap
        plt.scatter(np.log10(total_star_mass), np.log10(SFR), color=halo_colors[halo], label=halo_names[halo], s=10)

    # Plot the observed best-fit values for a handful of redshifts
    Mstar_list = np.arange(8.,12.5,0.5)
    low_Mstar_list = np.arange(2.,8.5,0.5)
    tlist = [3.330, 1.566]
    zlist = [2., 4.]
    SFR_list = []
    low_SFR_list = []
    for i in range(len(tlist)):
        SFR_list.append([])
        for j in range(len(Mstar_list)):
            SFR_list[i].append((0.84-0.026*tlist[i])*Mstar_list[j] - (6.51-0.11*tlist[i]))  # This equation comes from Speagle et al. (2014), first row of Table 9
        plt.plot(Mstar_list, SFR_list[i], '-', lw=1, color=colormap(normalize(tlist[i])), label='z=%.2f' % zlist[i])
        plt.fill_between(Mstar_list, np.array(SFR_list[i])-0.2, y2=np.array(SFR_list[i])+0.2, color=colormap(normalize(tlist[i])), alpha=0.2)
        # Extrapolate to lower Mstar
        low_SFR_list.append([])
        for j in range(len(low_Mstar_list)):
            low_SFR_list[i].append((0.84-0.026*tlist[i])*low_Mstar_list[j] - (6.51-0.11*tlist[i]))
        plt.plot(low_Mstar_list, low_SFR_list[i], '--', lw=1, color=colormap(normalize(tlist[i])))
    plt.xlabel(r'log Stellar Mass [$M_\odot$]', fontsize=14)
    plt.ylabel(r'$\log$ SFR [$M_\odot$/yr]', fontsize=14)
    plt.title('$z = %.2f$' % hc.current_redshift, fontsize=14)
    plt.axis([2.,12.5,-5,4])
    plt.legend(loc=2, frameon=False, fontsize=12)
    plt.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, top=True, right=True)
    plt.tight_layout()
    plt.savefig(args.save_directory + '/' + snap + '_SFMS_all-halos.png', dpi=300)
    plt.close()

def all_halos_MZR(snap, args):
    '''Plots the mass-metallicity relation for all halos in the halo catalog.'''

    colormap = plt.cm.rainbow_r
    normalize = matplotlib.colors.LogNorm(vmin=0.9, vmax=5.)

    halo_names = {'halo_008508':'Tempest', 'halo_005016':'Squall', 'halo_005036':'Maelstrom', 'halo_004123':'Blizzard', 'halo_002392':'Hurricane'}
    halo_colors = {'halo_008508':'#8FDC97','halo_005016':'#6A0136','halo_005036':'#188FA7','halo_004123':'#CC3F0C', 'halo_002392':'#D5A021'}

    for i in range(len(args.halo_list)):
        halo = args.halo_list[i]
        hc = yt.load(args.directory + '/' + halo + '/' + args.run + '/plots/halo_catalogs/' + snap + '/' + snap + '.0.h5')
        all_data = hc.all_data()

        halo_metallicity = all_data[('halos','average_metallicity')].in_units('Zsun')
        total_star_mass = all_data[('halos','total_star_mass')].in_units('Msun')

        # Plot the halos in this snap
        plt.scatter(np.log10(total_star_mass), halo_metallicity, color=halo_colors[halo], label=halo_names[halo], s=10)

    # Plot the observed best-fit values for a handful of redshifts
    # These data come from Nakajima et al. 2023, Fig. 10 digitized
    tlist = [2.896, 1.962, 1.566]
    zlist = [2.3, 3.3, 4.]
    Mstar_list = [[9.007, 9.186, 9.417, 9.615, 9.828, 10.048, 10.291, 10.449],
                  [9.156, 9.320, 9.488, 9.618, 9.754, 9.914, 10.062, 10.190, 10.325, 10.464],
                  [7.338, 7.675, 7.895, 8.133, 8.310, 9.031, 9.421, 9.567]]
    Z_list = [[0.3324, 0.3743, 0.4371, 0.5063, 0.5815, 0.6756, 0.7947, 0.8883],
              [0.2976, 0.3296, 0.3703, 0.4073, 0.4421, 0.4915, 0.5421, 0.5963, 0.6524, 0.7119],
              [0.0797, 0.0958, 0.1088, 0.1244, 0.1381, 0.2028, 0.2573, 0.2793]]
    for i in range(len(tlist)):
        # Extrapolate the relations linearly to lower Mstar
        slope = (np.log10(Z_list[i][1]) - np.log10(Z_list[i][0])) / (Mstar_list[i][1] - Mstar_list[i][0])
        intercept = np.log10(Z_list[i][0]) - slope * Mstar_list[i][0]
        low_Mstar = np.array([2., Mstar_list[i][0]])
        low_Z = 10**(slope * low_Mstar + intercept)
        plt.plot(Mstar_list[i], Z_list[i], '-', lw=1, color=colormap(normalize(tlist[i])))
        plt.plot(low_Mstar, low_Z, '--', lw=1, color=colormap(normalize(tlist[i])), label='z=%.2f' % zlist[i])
    plt.xlabel(r'log Stellar Mass [$M_\odot$]', fontsize=14)
    plt.ylabel(r'Average Gas Metallicity [$Z_\odot$]', fontsize=14)
    plt.title('$z = %.2f$' % hc.current_redshift, fontsize=14)
    plt.axis([2.,12.0,1e-8,1e1])
    plt.yscale('log')
    plt.legend(loc=4, frameon=False, fontsize=12)
    plt.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=12, top=True, right=True)
    plt.tight_layout()
    plt.savefig(args.save_directory + '/' + snap + '_MZR_all-halos.png', dpi=300)
    plt.close()

def all_halos_gasMHM(snap, args):
    '''Plots total gas mass vs. halo mass for all halos in the halo catalog.'''

    halo_names = {'halo_008508':'Tempest', 'halo_005016':'Squall', 'halo_005036':'Maelstrom', 'halo_004123':'Blizzard', 'halo_002392':'Hurricane'}
    halo_colors = {'halo_008508':'#8FDC97','halo_005016':'#6A0136','halo_005036':'#188FA7','halo_004123':'#CC3F0C', 'halo_002392':'#D5A021'}

    for i in range(len(args.halo_list)):
        halo = args.halo_list[i]
        hc = yt.load(args.directory + '/' + halo + '/' + args.run + '/plots/halo_catalogs/' + snap + '/' + snap + '.0.h5')
        all_data = hc.all_data()

        total_gas_mass = all_data[('halos','total_gas_mass')].in_units('Msun')
        total_halo_mass = all_data[('halos','particle_mass')].in_units('Msun')

        # Plot the halos in this snap
        plt.scatter(np.log10(total_halo_mass), np.log10(total_gas_mass), color=halo_colors[halo], label=halo_names[halo], s=10)

    plt.xlabel(r'log Halo Mass [$M_\odot$]', fontsize=14)
    plt.ylabel(r'log Gas Mass [$M_\odot$]', fontsize=14)
    plt.title('$z = %.2f$' % hc.current_redshift, fontsize=14)
    plt.axis([7.5,12.0,5,11])
    plt.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=12, top=True, right=True)
    plt.legend(loc=4, frameon=False, fontsize=12)
    plt.tight_layout()
    plt.savefig(args.save_directory + '/' + snap + '_gasMHM_all-halos.png', dpi=300)
    plt.close()

def all_halos_h2_frac(snap, args):
    '''Plots average H2 fraction vs. halo mass for all halos in the halo catalog.'''

    halo_names = {'halo_008508':'Tempest', 'halo_005016':'Squall', 'halo_005036':'Maelstrom', 'halo_004123':'Blizzard', 'halo_002392':'Hurricane'}
    halo_colors = {'halo_008508':'#8FDC97','halo_005016':'#6A0136','halo_005036':'#188FA7','halo_004123':'#CC3F0C', 'halo_002392':'#D5A021'}

    for i in range(len(args.halo_list)):
        halo = args.halo_list[i]
        hc = yt.load(args.directory + '/' + halo + '/' + args.run + '/plots/halo_catalogs/' + snap + '/' + snap + '.0.h5')
        all_data = hc.all_data()

        avg_h2_frac = all_data[('halos','average_fH2')]
        total_halo_mass = all_data[('halos','particle_mass')].in_units('Msun')

        # Plot the halos in this snap
        plt.scatter(np.log10(total_halo_mass), np.log10(avg_h2_frac), color=halo_colors[halo], label=halo_names[halo], s=10)

    plt.xlabel(r'log Halo Mass [$M_\odot$]', fontsize=14)
    plt.ylabel(r'log $f_{\mathrm{H}_2}$', fontsize=14)
    plt.title('$z = %.2f$' % hc.current_redshift, fontsize=14)
    plt.axis([7.5,12.0,-16,0])
    plt.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=12, top=True, right=True)
    plt.legend(loc=4, frameon=False, fontsize=12)
    plt.tight_layout()
    plt.savefig(args.save_directory + '/' + snap + '_h2-frac_all-halos.png', dpi=300)
    plt.close()

def make_all_halos_plots(snap, args):
    if (args.all_plots) or ('SMHM' in args.make_plots):
        all_halos_SMHM(snap, args)
    if (args.all_plots) or ('SFMS' in args.make_plots):
        all_halos_SFMS(snap, args)
    if (args.all_plots) or ('MZR' in args.make_plots):
        all_halos_MZR(snap, args)
    if (args.all_plots) or ('gasMHM' in args.make_plots):
        all_halos_gasMHM(snap, args)
    if (args.all_plots) or ('h2_frac' in args.make_plots):
        all_halos_h2_frac(snap, args)

if __name__ == '__main__':
    args = parse_halos_args()

    outs = make_output_list(args.output)
    args.halo_list = args.halos.split(',')

    if (args.nproc==1):
        for i in range(len(outs)):
            snap = outs[i]
            make_all_halos_plots(snap, args)
    else:
        # Split into a number of groupings equal to the number of processors
        # and run one process per processor
        for i in range(len(outs)//args.nproc):
            threads = []
            for j in range(args.nproc):
                snap = outs[args.nproc*i+j]
                threads.append(multi.Process(target=make_all_halos_plots, args=[snap, args]))
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        # For any leftover snapshots, run one per processor
        threads = []
        for j in range(len(outs)%args.nproc):
            snap = outs[-(j+1)]
            threads.append(multi.Process(target=make_all_halos_plots, args=[snap, args]))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
