import astropy
from astropy.io import ascii
import numpy as np
import matplotlib.pyplot as plt

fig_dir = '.'

cat = ascii.read('satellite_properties.cat')

halos = np.unique(cat['halo'])

proj_axes = ['x', 'y', 'z']

for halo in halos:
    cat_halo = cat[cat['halo'] == halo]
    center_index = np.where(cat_halo['id'] == '0')
    center_position = np.array([cat_halo[center_index]['x'], 
                                cat_halo[center_index]['y'], 
                                cat_halo[center_index]['z']])

    sats = np.unique(cat_halo['id'])
    fig, axes = plt.subplots(1, len(proj_axes), figsize = (5*len(proj_axes), 5))
    for p, proj_axis in enumerate(proj_axes):
        axes[p].annotate('{}-projection'.format(proj_axis), (0.98, 0.98), xycoords = 'axes fraction', ha = 'right', va = 'top', fontsize = 20)
        axes[p].annotate('{}'.format(halo), (0.02, 0.98), xycoords = 'axes fraction', ha = 'left', va = 'top', fontsize = 20)
        for sat in sats:
            if sat == '0': continue
            sat_index = np.where(cat_halo['id'] == sat)
            sat_position = np.array([cat_halo[sat_index]['x'], 
                                        cat_halo[sat_index]['y'], 
                                        cat_halo[sat_index]['z']])

            if proj_axis == 'x': 
                proj_i = sat_position[1] - center_position[1]
                proj_j = sat_position[2] - center_position[2]
                proj_i_lbl = 'y (kpc)'
                proj_j_lbl = 'z (kpc)'

            if proj_axis == 'y': 
                proj_i = sat_position[0] - center_position[0]
                proj_j = sat_position[2] - center_position[2]
                proj_i_lbl = 'x (kpc)'
                proj_j_lbl = 'z (kpc)'

            if proj_axis == 'z': 
                proj_i = sat_position[0] - center_position[0]
                proj_j = sat_position[1] - center_position[1]
                proj_i_lbl = 'x (kpc)'
                proj_j_lbl = 'y (kpc)'

            axes[p].plot(proj_i, proj_j, marker = 'o', color = 'black', markeredgecolor = 'darkblue', markersize = 12)
            axes[p].annotate(sat, (proj_i, proj_j), ha = 'center', va = 'center', color = 'white', zorder = 10)

        axes[p].plot(0, 0, 'kx', markersize = 20)
        axes[p].set_xlim(-100, 100)
        axes[p].set_ylim(-100, 100)
        axes[p].set_xlabel(proj_i_lbl)
        axes[p].set_ylabel(proj_j_lbl)
    fig.tight_layout()
    fig.savefig(fig_dir + '/{}_projections.png'.format(halo), dpi = 400)









