'''
    Filename: star_formation_plots.py
    Author: Cassi
    Created: 6-12-24
    Last modified: 7-22-24 by Cassi
    This file works with fogghorn_analysis.py to make a set of plots for investigating star formation.
    If you add a new function to this scripts, then please also add the function name to the appropriate list at the end of fogghorn/header.py
'''

from foggie.fogghorn.header import *
from foggie.fogghorn.util import *

# --------------------------------------------------------------------------------------------------------------------
def young_stars_density_projection_x(ds, region, args, output_filename):
    young_stars_density_projection(ds, region, args, output_filename, 'x')

def young_stars_density_projection_y(ds, region, args, output_filename):
    young_stars_density_projection(ds, region, args, output_filename, 'y')

def young_stars_density_projection_z(ds, region, args, output_filename):
    young_stars_density_projection(ds, region, args, output_filename, 'z')

def young_stars_density_projection_x_disk(ds, region, args, output_filename):
    young_stars_density_projection(ds, region, args, output_filename, 'x-disk')

def young_stars_density_projection_y_disk(ds, region, args, output_filename):
    young_stars_density_projection(ds, region, args, output_filename, 'y-disk')

def young_stars_density_projection_z_disk(ds, region, args, output_filename):
    young_stars_density_projection(ds, region, args, output_filename, 'z-disk')

# --------------------------------------------------------------------------------------------------------------------
def young_stars_density_projection(ds, region, args, output_filename, projection):
    '''
    Plots a young stars density projection of the galaxy disk.
    '''

    if '-disk' in projection:
        if 'x' in projection:
            p_dir = ds.x_unit_disk
            north_vector = ds.z_unit_disk
        if 'y' in projection:
            p_dir = ds.y_unit_disk
            north_vector = ds.z_unit_disk
        if 'z' in projection:
            p_dir = ds.z_unit_disk
            north_vector = ds.x_unit_disk
        p = yt.ProjectionPlot(ds, p_dir, ('deposit', 'young_stars_cic'), width=(args.proj_width, 'kpc'), center=ds.halo_center_code, north_vector=north_vector)
    else: p = yt.ProjectionPlot(ds, projection, ('deposit', 'young_stars_cic'), width=(args.proj_width, 'kpc'), center=ds.halo_center_code)
    p.set_unit(('deposit','young_stars_cic'),'Msun/kpc**2')
    p.set_zlim(('deposit','young_stars_cic'),1000,1000000)
    p.set_cmap(('deposit','young_stars_cic'), density_color_map)
    p.save(output_filename)
    print('Saved figure ' + output_filename)

# --------------------------------------------------------------------------------------------------------------------
def KS_relation_x(ds, region, args, output_filename):
    KS_relation(ds, region, args, output_filename, 'x')

def KS_relation_y(ds, region, args, output_filename):
    KS_relation(ds, region, args, output_filename, 'y')

def KS_relation_z(ds, region, args, output_filename):
    KS_relation(ds, region, args, output_filename, 'z')

def KS_relation_x_disk(ds, region, args, output_filename):
    KS_relation(ds, region, args, output_filename, 'x-disk')

def KS_relation_y_disk(ds, region, args, output_filename):
    KS_relation(ds, region, args, output_filename, 'y-disk')

def KS_relation_z_disk(ds, region, args, output_filename):
    KS_relation(ds, region, args, output_filename, 'z-disk')

# --------------------------------------------------------------------------------------------------------------------
def KS_relation(ds, region, args, output_filename, projection):
    '''
    Plots the KS relation from the dataset as compared to a curve taken from Krumholz, McKee, & Tumlinson (2009), ApJ 699, 850.
    '''

    # Make a projection and convert to FRB
    if '-disk' in projection:
        if 'x' in projection:
            p_dir = ds.x_unit_disk
            north_vector = ds.z_unit_disk
        if 'y' in projection:
            p_dir = ds.y_unit_disk
            north_vector = ds.z_unit_disk
        if 'z' in projection:
            p_dir = ds.z_unit_disk
            north_vector = ds.x_unit_disk
        p = yt.ProjectionPlot(ds, p_dir, ('gas', 'density'), width=(args.proj_width, 'kpc'), center=ds.halo_center_code, north_vector=north_vector, buff_size=[500,500])
    else: p = yt.ProjectionPlot(ds, projection, ('gas', 'density'), width=(args.proj_width, 'kpc'), center=ds.halo_center_code, buff_size=[500,500])
    proj_frb = p.frb
    # Pull out the gas surface density and the star formation rate of the young stars
    projected_density = proj_frb['density'].in_units('Msun/pc**2')
    ks_nh1 = proj_frb['H_p0_number_density'].in_units('pc**-2') * yt.YTArray(1.67e-24/1.989e33, 'Msun')
    young_stars = proj_frb[('deposit', 'young_stars_cic')].in_units('Msun/kpc**2')
    ks_sfr = young_stars / yt.YTArray(3e6, 'yr') + yt.YTArray(1e-6, 'Msun/kpc**2/yr')

    # These values are pulled from KMT09 Figure 2, the log cZ' = 0.2 curve
    log_sigma_gas = [0.5278, 0.6571, 0.8165, 1.0151, 1.2034, 1.4506, 1.6286, 1.9399, 2.2663, 2.7905, 3.5817]
    log_sigma_sfr = [-5.1072, -4.4546, -3.5572, -2.7926, -2.3442, -2.0185, -1.8253, -1.5406, -1.0927, -0.3801, 0.6579]
    c = Polynomial.fit(log_sigma_gas, log_sigma_sfr, deg=5)

    # Make the plot
    plt.plot(np.log10(ks_nh1), np.log10(ks_sfr), '.')
    plt.plot(log_sigma_gas, log_sigma_sfr, marker='o', color='red')
    plt.xlabel('$\Sigma _{g} \,\, (M_{\odot} / pc^2)$', fontsize=16)
    plt.ylabel('$\dot{M} _{*} \,\, (M_{\odot} / yr / kpc^2)$', fontsize=16)
    plt.axis([-1,5,-6,3])
    plt.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, top=True, right=True)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    print('Saved figure ' + output_filename)
    plt.close()