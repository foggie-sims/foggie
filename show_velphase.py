"""
creates "core sample" velocity plots
JT 070318
"""
import datashader as dshader
import datashader.transfer_functions as tf
from datashader import reductions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from consistency import phase_color_key, species_dict

CORE_WIDTH = 20.

def show_velphase(ds, ray_df, ray_start, ray_end, line_dict, fileroot):
    """ the docstring is missing, is it??? """

    ray_s = ray_start.ndarray_view()
    ray_e = ray_end.ndarray_view()

    # take out a "core sample" along the ray with a width given by core_width
    proper_box_size = ds.get_parameter('CosmologyComovingBoxSize') \
        / ds.get_parameter('CosmologyHubbleConstantNow') * 1000. # in kpc
    all_data = ds.r[ray_s[0]:ray_e[0],
                    ray_s[1]-0.5*CORE_WIDTH/proper_box_size:ray_s[1]+
                    0.5*CORE_WIDTH/proper_box_size,
                    ray_s[2]-0.5*CORE_WIDTH/proper_box_size:ray_s[2]+
                    0.5*CORE_WIDTH/proper_box_size]

    dens = np.log10(all_data['density'].ndarray_view())
    temp = np.log10(all_data['temperature'].ndarray_view())

    phase = np.chararray(np.size(temp), 4)
    phase[temp < 19.] = 'hot'
    phase[temp < 6.] = 'warm'
    phase[temp < 5.] = 'cool'
    phase[temp < 4.] = 'cold'

    df = pd.DataFrame({'x':all_data['x'].ndarray_view() * proper_box_size,
                       'y':all_data['y'].ndarray_view() * proper_box_size,
                       'z':all_data['z'].ndarray_view() * proper_box_size,
                       'vx':all_data["x-velocity"].in_units('km/s'),
                       'vy':all_data["y-velocity"].in_units('km/s'),
                       'vz':all_data["z-velocity"].in_units('km/s'),
                       'temp':temp, 'dens':dens, 'phase':phase})
    df.phase = df.phase.astype('category')

    #establish the grid of plots and obtain the axis objects
    plt.figure(figsize=(8,6))
    gs = GridSpec(2, 4, width_ratios=[1, 5, 5, 5],height_ratios=[4, 1])
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax1.set_title('HI Lya')
    ax2 = plt.subplot(gs[2])
    ax2.set_title('Si II 1260')
    ax3 = plt.subplot(gs[3])
    ax3.set_title('O VI 1032')
    ax4 = plt.subplot(gs[4])
    ax4.spines["top"].set_color('white')
    ax4.spines["bottom"].set_color('white')
    ax4.spines["left"].set_color('white')
    ax4.spines["right"].set_color('white')
    ax5 = plt.subplot(gs[5])
    ax5.set_ylabel('Flux')
    ax5.set_xlabel(' ')
    ax6 = plt.subplot(gs[6])
    ax6.set_xlabel('Velocity [km / s]')
    ax7 = plt.subplot(gs[7])
    ax7.set_xlabel(' ')


    # this one makes the datashaded "core sample" with phase coloring
    cvs = dshader.Canvas(plot_width=800, plot_height=200,
                         x_range=(np.min(df['x']), np.max(df['x'])),
                         y_range=(np.mean(df['y'])-CORE_WIDTH/0.695,
                                  np.mean(df['y'])+CORE_WIDTH/0.695))
    agg = cvs.points(df, 'x', 'y', dshader.count_cat('phase'))
    img = tf.shade(agg, color_key=phase_color_key)
    x_y = tf.spread(img, px=2, shape='square')

    ax0.imshow(np.rot90(x_y.to_pil()))

    ytext = ax0.set_ylabel('y [cKpc]', fontname='Arial', fontsize=10)
    ax0.set_yticks([0, 200, 400, 600, 800])
    ax0.set_yticklabels([ str(int(s)) for s in [0, 50, 100, 150, 200]],
                            fontname='Arial', fontsize=8)
    ax0.set_xticks([0, 100, 200])
    ax0.set_xticklabels([ str(int(s)) for s in [-50, 0, 50]],
                            fontname='Arial', fontsize=8)

    # render x vs. vx but don't show it yet.
    cvs = dshader.Canvas(plot_width=800, plot_height=300,
                         x_range=(np.min(df['x']), np.max(df['x'])),
                         y_range=(-300, 300)) # < ----- what units?
    agg = cvs.points(df, 'x', 'vx', dshader.count_cat('phase'))
    x_vx = tf.spread(tf.shade(agg, color_key=phase_color_key), shape='square')


    #now iterate over the species to get the ion fraction plots
    for species, ax, lax in zip(['HI', 'SiIII', 'OVI'], [ax1, ax2, ax3], [ax5, ax6, ax7]):

        print("Current species: ", species)
        cvs = dshader.Canvas(plot_width=800, plot_height=300,
                             x_range=(ray_s[0], ray_e[0]),
                             y_range=(-300,300))
        vx_render = tf.shade(cvs.points(ray_df, 'x', 'x-velocity',
                                        agg=reductions.mean(species_dict[species])),
                                        how='eq_hist')
        ray_vx = tf.spread(vx_render, px=3, shape='square')

        ax.imshow(np.rot90(x_vx.to_pil()))
        ax.imshow(np.rot90(ray_vx.to_pil()))
        ax.set_yticklabels([' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '])
        ax.set_xticks([0, 50, 100, 150, 200, 250, 300])
        ax.set_xticklabels(['-300', '-200', '-100', '0', '100', '200', '300'])


    vel = (line_dict['H I 1216'].lambda_field.ndarray_view()/3. - 1215.67) / 1215.67 * 3e5
    ax5.plot(vel, line_dict['H I 1216'].flux_field)
    ax5.set_xlim(-300,300)
    ax5.set_ylim(0,1)

    vel = (line_dict['Si II 1260'].lambda_field.ndarray_view()/3. - 1260.4221) / 1260.4221 * 3e5
    ax6.plot(vel, line_dict['Si II 1260'].flux_field)
    ax6.set_xlim(-300,300)
    ax6.set_ylim(0,1)
    ax6.set_yticklabels([' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '])

    vel = (line_dict['O VI 1032'].lambda_field.ndarray_view()/3. - 1031.9261) / 1031.9261 * 3e5
    ax7.plot(vel, line_dict['O VI 1032'].flux_field)
    ax7.set_xlim(-300,300)
    ax7.set_ylim(0,1)
    ax7.set_yticklabels([' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '])

    ax4.set_yticklabels([' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '])
    ax4.set_xticklabels([' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '])


    plt.savefig(fileroot+'_velphase.png', dpi=300)
