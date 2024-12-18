#!/usr/bin/env python3

"""

    Title :      datashader_mwe
    Notes :      Minimal working example of how to make a quick datashader plot with matplotlib interface
                 This is designed for debugging, and to be easily used by everyone within FOGGIE, i.e., without depending on all of Ayan Acharyya's dependencies (that is how this is different drom datashader_quickplot.py)
    Output :     datashader plot as png file
    Author :     Ayan Acharyya
    Started :    Aug 2023
    Examples :   run datashader_mwe.py --system ayan_local --halo 8508 --output RD0030 --xcol rad --ycol metal --colorcol vrad

"""
from matplotlib import pyplot as plt
import yt
from yt.units import *
import datashader as dsh
from datashader.mpl_ext import dsshow


from foggie.utils.consistency import *
from foggie.utils.foggie_load import *
import pandas as pd
import argparse, os, time, datetime
start_time = time.time()
HOME = os.getenv('HOME')

# -------------------------------------------------------------------------------
def get_df_from_ds(ds, args):
    '''
    Function to make a pandas dataframe from the yt dataset based on the given field list and color category
    The dataframe can be written to file for faster access in future
    This function is somewhat based on foggie.utils.prep_dataframe.prep_dataframe()
    :return: dataframe
    '''
    print('Creating datframe..')
    all_fields = [args.xcol, args.ycol, args.colorcol]
    df = pd.DataFrame()
    for index,field in enumerate(all_fields):
        print('Doing property: ' + field + ', which is ' + str(index + 1) + ' of the ' + str(len(all_fields)) + ' fields..')
        arr = ds[field_dict[field]].in_units(unit_dict[field]).ndarray_view()
        if field == 'metal': arr = np.log10(arr)
        df[field] = arr

    return df
# --------------------------------------------------------------------------------
def make_datashader_plot_mpl(df, args):
    '''
    Function to make data shader plot of y_field vs x_field, colored in bins of color_field
    This function is based on foggie.render.shade_maps.render_image()
    :return figure
    '''
    # --------to make the main datashader plot--------------------------
    fig, ax = plt.subplots()
    color_key = [to_hex(item) for item in args.colormap.colors]
    artist = dsshow(df, dsh.Point(args.xcol, args.ycol), dsh.mean(args.colorcol), norm='linear', cmap=color_key, x_range=(args.xmin, args.xmax), y_range=(args.ymin, args.ymax), vmin=args.cmin, vmax=args.cmax, aspect = 'auto', ax=ax)

    # ------to make the axes-------------
    ax.xaxis.set_label_text(label_dict[args.xcol])
    ax.yaxis.set_label_text(label_dict[args.ycol])

    # ------to make the colorbar axis-------------
    cax_xpos, cax_ypos, cax_width, cax_height = 0.7, 0.835, 0.25, 0.035
    cax = fig.add_axes([cax_xpos, cax_ypos, cax_width, cax_height])
    plt.colorbar(artist, cax=cax, orientation='horizontal')

    cax.set_xticklabels(['%.0F' % index for index in cax.get_xticks()])
    fig.text(cax_xpos + cax_width / 2, cax_ypos + cax_height + 0.005, label_dict[args.colorcol], ha='center', va='bottom')

    # ---------to annotate and save the figure----------------------
    plt.savefig(HOME + '/Downloads/test_' + args.xcol + '_vs_' + args.ycol + '_colorby_' + args.colorcol + '.png', transparent=False)
    plt.show(block=False)

    return fig

# ----------to initialise dicts-----------------------
field_dict = {'rad': ('gas', 'radius_corrected'), 'metal': ('gas', 'metallicity'), 'vrad': ('gas', 'radial_velocity_corrected')}
unit_dict = {'rad': 'kpc', 'metal': r'Zsun', 'vrad': 'km/s'}
label_dict = {'rad': 'Radius (kpc)', 'metal': 'Log Metallicity (Zsun)', 'vrad': 'Radial velocity (km/s)'}
bounds_dict = {'rad': (0, 20), 'metal': (-2, 1), 'vrad': (-400, 400)}
colormap_dict = {'vrad': outflow_inflow_discrete_cmap}

# -----main code-----------------
if __name__ == '__main__':

    # ------------- arg parse -------------------------------
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='''load FOGGIE simulations''')
    parser.add_argument('--system', metavar='system', type=str, action='store', default='ayan_local', help='Which system are you on? Default is ayan_pleiades')
    parser.add_argument('--halo', metavar='halo', type=str, action='store', default='8508', help='which halo?')
    parser.add_argument('--run', metavar='run', type=str, action='store', default='nref11c_nref9f', help='which run?')
    parser.add_argument('--pwd', dest='pwd', action='store_true', default=False, help='Just use the current working directory?, default is no')
    parser.add_argument('--output', metavar='output', type=str, action='store', default='RD0030', help='which output?')
    parser.add_argument('--xcol', metavar='xcol', type=str, action='store', default='rad', help='x axis quantity; default is rad')
    parser.add_argument('--ycol', metavar='ycol', type=str, action='store', default='metal', help='y axis quantity; default is metal')
    parser.add_argument('--colorcol', metavar='colorcol', type=str, action='store', default='vrad', help='color axis quantity; default is vrad')
    parser.add_argument('--galrad', metavar='galrad', type=float, action='store', default=20., help='the radial extent (in each spatial dimension) to which computations will be done, in kpc; default is 20')

    args = parser.parse_args()

    # ----------to load the simulation---------------
    halos_df_name = HOME + '/Work/astro/ayan_codes/foggie/foggie/halo_infos/00' + args.halo + '/' + args.run + '/halo_cen_smoothed'
    ds, refine_box = load_sim(args, region='refine_box', halo_c_v_name=halos_df_name)

    # ----------to choose a box ---------------
    box_center = ds.arr(ds.halo_center_kpc, kpc)
    box_width_kpc = ds.arr(2 * args.galrad, 'kpc')
    box = ds.r[box_center[0] - box_width_kpc / 2.: box_center[0] + box_width_kpc / 2., box_center[1] - box_width_kpc / 2.: box_center[1] + box_width_kpc / 2., box_center[2] - box_width_kpc / 2.: box_center[2] + box_width_kpc / 2., ]

    # ----------to initialise axes limits and colormaps-----------------------
    args.xmin, args.xmax = bounds_dict[args.xcol]
    args.ymin, args.ymax = bounds_dict[args.ycol]
    args.cmin, args.cmax = bounds_dict[args.colorcol]
    args.colormap = colormap_dict[args.colorcol]

    # ----------to make dataframe from dataset-----------------------
    df = get_df_from_ds(box, args)

    # ----------to make datashader plot-----------------------
    fig = make_datashader_plot_mpl(df, args)

    print('Completed in %s' % (datetime.timedelta(minutes=(time.time() - start_time) / 60)))
