'''
    Filename: feedback_plots.py
    Author: Cassi
    Created: 6-12-24
    Last modified: 7-22-24 by Cassi
    This file works with fogghorn_analysis.py to make a set of plots for investigating feddback.
    If you add a new function to this scripts, then please also add the function name to the appropriate list at the end of fogghorn/header.py
'''

from foggie.fogghorn.header import *
from foggie.fogghorn.util import *

# --------------------------------------------------------------------------------------------------------------------
def outflow_rates(ds, region, args, output_filename):
    '''
    Plots the mass and metals outflow rates, both as a function of radius centered on the galaxy
    and as a function of height through 20x20 kpc horizontal planes above and below the disk of young stars.
    Uses only gas with outflow velocities greater than 50 km/s.
    '''

    # Load needed fields into arrays
    radius = region['gas','radius_corrected'].in_units('kpc')
    x = region['gas', 'x_disk'].in_units('kpc').v
    y = region['gas', 'y_disk'].in_units('kpc').v
    z = region['gas', 'z_disk'].in_units('kpc').v
    vx = region['gas','vx_disk'].in_units('kpc/yr').v
    vy = region['gas','vy_disk'].in_units('kpc/yr').v
    vz = region['gas','vz_disk'].in_units('kpc/yr').v
    mass = region['gas', 'cell_mass'].in_units('Msun').v
    metals = region['gas','metal_mass'].in_units('Msun').v
    rv = region['gas','radial_velocity_corrected'].in_units('km/s').v
    hv = region['gas','vz_disk'].in_units('km/s').v

    # Define radius and height lists
    radii = np.linspace(0.5, 20., 40)
    heights = np.linspace(0.5, 20., 40)

    # Calculate new positions of gas cells 10 Myr later
    dt = 10.e6
    new_x = vx*dt + x
    new_y = vy*dt + y
    new_z = vz*dt + z
    new_radius = np.sqrt(new_x**2. + new_y**2. + new_z**2.)

    # Sum the mass and metals passing through the boundaries
    mass_sph = []
    metal_sph = []
    mass_horiz = []
    metal_horiz = []
    for i in range(len(radii)):
        r = radii[i]
        mass_sph.append(np.sum(mass[(radius < r) & (new_radius > r) & (rv > 50.)])/dt)
        metal_sph.append(np.sum(metals[(radius < r) & (new_radius > r) & (rv > 50.)])/dt)
    for i in range(len(heights)):
        h = heights[i]
        mass_horiz.append(np.sum(mass[(np.abs(z) < h) & (np.abs(new_z) > h) & (np.abs(hv) > 50.)])/dt)
        metal_horiz.append(np.sum(metals[(np.abs(z) < h) & (np.abs(new_z) > h) & (np.abs(hv) > 50.)])/dt)

    # Plot the outflow rates
    fig = plt.figure(1, figsize=(10,4))
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    ax1.plot(radii, mass_sph, 'k-', lw=2, label='Mass')
    ax1.plot(radii, metal_sph, 'k--', lw=2, label='Metals')
    ax1.set_ylabel(r'Mass outflow rate [$M_\odot$/yr]', fontsize=16)
    ax1.set_xlabel('Radius [kpc]', fontsize=16)
    ax1.set_yscale('log')
    ax1.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14)
    ax1.legend(loc=1, frameon=False, fontsize=16)
    ax2.plot(heights, mass_horiz, 'k-', lw=2, label='Mass')
    ax2.plot(heights, metal_horiz, 'k--', lw=2, label='Metals')
    ax2.set_ylabel(r'Mass outflow rate [$M_\odot$/yr]', fontsize=16)
    ax2.set_xlabel('Height from disk midplane [kpc]', fontsize=16)
    ax2.set_yscale('log')
    ax2.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    print('Saved figure ' + output_filename)
    plt.close()