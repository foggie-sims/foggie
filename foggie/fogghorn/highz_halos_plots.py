'''
    Filename: highz_halos_plots.py
    Author: Cassi
    Created: 4-14-25
    Last modified: 4-14-25 by Cassi
    This file works with fogghorn_analysis.py to make the set of plots for various scaling relations for all halos in the high resolution region.
    If you add a new function to this script, then please also add the function name to the dictionary in fogghorn_analysis.py.

'''
['halos_density_projection','halos_SFMS','halos_SMHM','halos_MZR','halos_h2_frac','halos_gasMHM']
from foggie.fogghorn.header import *
from foggie.fogghorn.util import *

def get_halo_catalog(ds, args, snap):
    '''This function either checks if the halo catalog already exists for this snapshot 'snap'
    and returns it if so, or runs the halo finder and both saves it to file and returns it.'''

    if os.path.exists(args.save_directory + '/halo_catalogs/' + snap + '/' + snap + '.0.h5'):
        hc = yt.load(args.save_directory + '/halo_catalogs/' + snap + '/' + snap + '.0.h5')
        return hc

    else:
        box_length = ds.quan(500., 'kpc')
        box = ds.r[ds.halo_center_kpc[0]-box_length:ds.halo_center_kpc[0]+box_length, 
            ds.halo_center_kpc[1]-box_length:ds.halo_center_kpc[1]+box_length, 
            ds.halo_center_kpc[2]-box_length:ds.halo_center_kpc[2]+box_length]
        hc = HaloCatalog(data_ds=ds, finder_method='hop', finder_kwargs={"subvolume": box}, output_dir=args.save_directory + "/halo_catalogs")
        hc.create()

        hds = yt.load(args.save_directory + '/halo_catalogs/' + snap + '/' + snap + '.0.h5')
        hc = HaloCatalog(data_ds=ds, halos_ds=hds, output_dir=args.save_directory + "/halo_catalogs")
        hc.add_callback("sphere")

        add_quantity("average_temperature", halo_average_temperature)
        add_quantity("average_metallicity", halo_average_metallicity)
        add_quantity("max_metallicity", halo_max_metallicity)
        add_quantity("total_gas_mass", halo_total_gas_mass)
        add_quantity("total_star_mass", halo_total_star_mass)
        add_quantity("total_metal_mass", halo_total_metal_mass)
        add_quantity("total_young_stars7_mass", halo_total_young_stars7_mass)
        add_quantity("sfr7", halo_sfr7)
        add_quantity("total_young_stars8_mass", halo_total_young_stars8_mass)
        add_quantity("sfr8", halo_sfr8)
        add_quantity("average_fH2", halo_average_fH2)

        hc.add_quantity("average_temperature")
        hc.add_quantity("average_metallicity")
        hc.add_quantity("total_gas_mass")
        hc.add_quantity("total_star_mass")
        hc.add_quantity("total_metal_mass")
        hc.add_quantity("total_young_stars7_mass")
        hc.add_quantity("sfr7")
        hc.add_quantity("total_young_stars8_mass")
        hc.add_quantity("sfr8")
        hc.add_quantity("average_fH2")

        hc.create()

        hc = yt.load(args.save_directory + '/halo_catalogs/' + snap + '/' + snap + '.0.h5')
        return hc
    
def halos_density_projection(ds, region, args, output_filename):
    '''Plots a density projection with all the halos in the halo catalog
    overplotted as circles.'''

    box_length = ds.quan(500., 'kpc')
    box = ds.r[ds.halo_center_kpc[0]-box_length:ds.halo_center_kpc[0]+box_length, 
            ds.halo_center_kpc[1]-box_length:ds.halo_center_kpc[1]+box_length, 
            ds.halo_center_kpc[2]-box_length:ds.halo_center_kpc[2]+box_length]
    hc = get_halo_catalog(ds, args, args.snap)
    p = yt.ProjectionPlot(ds, 'z', ('gas','density'), weight_field=('gas','density'), data_source=box, center=ds.halo_center_code, width=(200, 'kpc'))
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
    plt.xlabel('log Halo Mass [$M_\odot$]', fontsize=16)
    plt.ylabel('log Stellar Mass [$M_\odot$]', fontsize=16)
    plt.title('$z = %.2f$' % ds.get_parameter('CosmologyCurrentRedshift'))
    plt.axis([7.5,10.5,5,10])
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
    low_Mstar_list = np.arange(5.,8.5,0.5)
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
    plt.xlabel('log Stellar Mass [$M_\odot$]', fontsize=16)
    plt.ylabel('$\log$ SFR [$M_\odot$/yr]', fontsize=16)
    plt.title('$z = %.2f$' % ds.get_parameter('CosmologyCurrentRedshift'))
    plt.axis([5.,12.5,-3,4])
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
    # These values come from digitizing Figure 7 of Behroozi et al. (2013)
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
        low_Mstar = np.array([5., Mstar_list[i][0]])
        low_Z = 10**(slope * low_Mstar + intercept)
        plt.plot(Mstar_list[i], Z_list[i], '-', lw=1, color=colormap(normalize(tlist[i])))
        plt.plot(low_Mstar, low_Z, '--', lw=1, color=colormap(normalize(tlist[i])), label='z=%.2f' % zlist[i])
    plt.xlabel('log Stellar Mass [$M_\odot$]', fontsize=16)
    plt.ylabel('Average Gas Metallicity [$Z_\odot$]', fontsize=16)
    plt.title('$z = %.2f$' % ds.get_parameter('CosmologyCurrentRedshift'))
    plt.axis([5.,10.5,1e-8,1e1])
    plt.yscale('log')
    plt.legend(loc=2, frameon=False, fontsize=14)
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

    plt.xlabel('log Halo Mass [$M_\odot$]', fontsize=16)
    plt.ylabel('log Gas Mass [$M_\odot$]', fontsize=16)
    plt.title('$z = %.2f$' % ds.get_parameter('CosmologyCurrentRedshift'))
    plt.axis([7.5,10.5,5,11])
    plt.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, top=True, right=True)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()

def halos_h2_frac(ds, region, args, output_filename):
    '''Plots average H2 fraction vs. halo mass for all halos in the halo catalog.'''

    hc = get_halo_catalog(ds, args, args.snap)
    all_data = hc.all_data()

    avg_h2_frac = all_data[('halos','average_fH2')]
    total_halo_mass = all_data[('halos','particle_mass')].in_units('Msun')

    colormap = plt.cm.rainbow_r
    normalize = matplotlib.colors.LogNorm(vmin=0.9, vmax=5.)

    # Plot the halos in this snap
    plt.scatter(np.log10(total_halo_mass), np.log10(avg_h2_frac), color=colormap(normalize(float(ds.current_time.in_units('Gyr')))))

    plt.xlabel('log Halo Mass [$M_\odot$]', fontsize=16)
    plt.ylabel('log $f_{\mathrm{H}_2}$', fontsize=16)
    plt.title('$z = %.2f$' % ds.get_parameter('CosmologyCurrentRedshift'))
    plt.axis([7.5,10.5,-16,0])
    plt.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, top=True, right=True)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()