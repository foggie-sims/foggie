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
    Plots the mass, metals, and specific thermal energy outflow rates, both as a function of radius centered on the galaxy
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
    kinetic_energy = region['gas','kinetic_energy_corrected'].in_units('erg').v
    thermal_energy = (region['gas','specific_thermal_energy']*region['gas','cell_mass']).in_units('erg').v

    # Define radius and height lists
    radii = np.linspace(0.5, 20., 40)
    heights = np.linspace(0.5, 20., 40)

    # Calculate new positions of gas cells 10 Myr later
    dt = 10.e6
    new_x = vx*dt + x
    new_y = vy*dt + y
    new_z = vz*dt + z
    new_radius = np.sqrt(new_x**2. + new_y**2. + new_z**2.)

    # Sum the mass, metals, and energy passing through the boundaries
    mass_sph = []
    metal_sph = []
    mass_horiz = []
    metal_horiz = []
    therm_sph = []
    therm_horiz = []
    kin_sph = []
    kin_horiz = []
    for i in range(len(radii)):
        r = radii[i]
        mass_sph.append(np.sum(mass[(radius < r) & (new_radius > r) & (rv > 50.)])/dt)
        metal_sph.append(np.sum(metals[(radius < r) & (new_radius > r) & (rv > 50.)])/dt)
        # Divide energies by mass *after* summing to get specific energy flow rates
        therm_sph.append(np.sum(thermal_energy[(radius < r) & (new_radius > r) & (rv > 50.)])/dt/mass_sph[-1])
        kin_sph.append(np.sum(kinetic_energy[(radius < r) & (new_radius > r) & (rv > 50.)])/dt/mass_sph[-1])
    for i in range(len(heights)):
        h = heights[i]
        mass_horiz.append(np.sum(mass[(np.abs(z) < h) & (np.abs(new_z) > h) & (np.abs(hv) > 50.)])/dt)
        metal_horiz.append(np.sum(metals[(np.abs(z) < h) & (np.abs(new_z) > h) & (np.abs(hv) > 50.)])/dt)
        # Divide energies by mass *after* summing to get specific energy flow rates
        therm_horiz.append(np.sum(thermal_energy[(np.abs(z) < h) & (np.abs(new_z) > h) & (np.abs(hv) > 50.)])/dt/mass_horiz[-1])
        kin_horiz.append(np.sum(kinetic_energy[(np.abs(z) < h) & (np.abs(new_z) > h) & (np.abs(hv) > 50.)])/dt/mass_horiz[-1])

    # Plot the outflow rates
    fig = plt.figure(1, figsize=(10,8))
    ax1 = fig.add_subplot(2,2,1)
    ax2 = fig.add_subplot(2,2,2)
    ax3 = fig.add_subplot(2,2,3)
    ax4 = fig.add_subplot(2,2,4)
    ax1.plot(radii, mass_sph, 'k-', lw=2, label='Mass')
    ax1.plot(radii, metal_sph, 'k--', lw=2, label='Metals')
    ax1.set_ylabel(r'Mass outflow rate [$M_\odot$/yr]', fontsize=14)
    ax1.set_xlabel('Radius [kpc]', fontsize=14)
    ax1.set_yscale('log')
    ax1.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=12)
    ax1.legend(loc=1, frameon=False, fontsize=14)
    ax2.plot(heights, mass_horiz, 'k-', lw=2, label='Mass')
    ax2.plot(heights, metal_horiz, 'k--', lw=2, label='Metals')
    ax2.set_ylabel(r'Mass outflow rate [$M_\odot$/yr]', fontsize=14)
    ax2.set_xlabel('Height from disk midplane [kpc]', fontsize=14)
    ax2.set_yscale('log')
    ax2.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=12)
    ax3.plot(radii, therm_sph, 'k-', lw=2, label='Thermal')
    ax3.plot(radii, kin_sph, 'k--', lw=2, label='Kinetic')
    ax3.set_ylabel(r'Specific energy outflow rate [erg/g/yr]', fontsize=14)
    ax3.set_xlabel('Radius [kpc]', fontsize=14)
    ax3.set_yscale('log')
    ax3.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=12)
    ax3.legend(loc=4, frameon=False, fontsize=14)
    ax4.plot(heights, therm_horiz, 'k-', lw=2, label='Thermal')
    ax4.plot(heights, kin_horiz, 'k--', lw=2, label='Kinetic')
    ax4.set_ylabel(r'Specific energy outflow rate [erg/g/yr]', fontsize=14)
    ax4.set_xlabel('Height from disk midplane [kpc]', fontsize=14)
    ax4.set_yscale('log')
    ax4.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=12)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    print('Saved figure ' + output_filename)
    plt.close()

# --------------------------------------------------------------------------------------------------------------------
def rad_vel_temp_colored(ds, region, args, output_filename):
    '''
    Makes a datashader plot of radial velocity radial profile color-coded by temperature.
    '''

    # Load needed fields into arrays
    radius = region['gas','radius_corrected'].in_units('kpc')
    rv = region['gas','radial_velocity_corrected'].in_units('km/s').v
    temperature = np.log10(region['gas','temperature'].in_units('K').v)

    color_func = categorize_by_temp
    color_key = new_phase_color_key
    cmin = temperature_min_datashader
    cmax = temperature_max_datashader
    color_ticks = [50,300,550]
    color_ticklabels = ['4','5','6']
    field_label = 'log T [K]'
    color_log = True
    x_range=[0,100]
    y_range=[-500,3000]

    data_frame = pd.DataFrame({})
    data_frame['radius'] = radius
    data_frame['rv'] = rv
    data_frame['temperature'] = temperature
    data_frame['color'] = color_func(data_frame['temperature'])
    data_frame.color = data_frame.color.astype('category')
    cvs = dshader.Canvas(plot_width=1000, plot_height=800, x_range=x_range, y_range=y_range)
    agg = cvs.points(data_frame, 'radius', 'rv', dshader.count_cat('color'))
    img = tf.spread(tf.shade(agg, color_key=color_key, how='eq_hist',min_alpha=40), shape='square', px=0)
    export_image(img, output_filename[:-4])
    fig = plt.figure(figsize=(10,9),dpi=300)
    ax = fig.add_subplot(1,1,1)
    image = plt.imread(output_filename)
    ax.imshow(image, extent=[x_range[0],x_range[1],y_range[0],y_range[1]])
    ax.set_aspect(8*abs(x_range[1]-x_range[0])/(10*abs(y_range[1]-y_range[0])))
    ax.set_xlabel('Radius [kpc]', fontsize=16)
    ax.set_ylabel('Radial velocity [km/s]', fontsize=16)
    ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, \
      top=True, right=True)
    ax2 = fig.add_axes([0.52, 0.9, 0.4, 0.1])
    cmap = create_foggie_cmap(cmin, cmax, color_func, color_key, color_log) # This function is defined in utils/analysis_utils.py and all it does is add the colorbar
    ax2.imshow(np.flip(cmap.to_pil(), 1))
    ax2.set_xticks(color_ticks)
    ax2.set_xticklabels(color_ticklabels, fontsize=14)
    ax2.text(400, 150, field_label, fontsize=16, ha='center', va='center')
    ax2.spines["top"].set_color('white')
    ax2.spines["bottom"].set_color('white')
    ax2.spines["left"].set_color('white')
    ax2.spines["right"].set_color('white')
    ax2.set_ylim(60, 180)
    ax2.set_xlim(-10, 750)
    ax2.set_yticklabels([])
    ax2.set_yticks([])
    plt.subplots_adjust(left=0.13, bottom=0.08,right=0.95)
    plt.savefig(output_filename, dpi=300)
    print('Saved figure ' + output_filename)

    plt.close()