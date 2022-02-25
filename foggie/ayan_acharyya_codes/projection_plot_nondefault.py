##!/usr/bin/env python3

"""

    Title :      projection_plot
    Notes :      make projection plots for new bigbox runs
    Output :     projection plots as png files
    Author :     Ayan Acharyya
    Started :    January 2021
    Example :    run /nobackup/aachary2/ayan_codes/foggie/foggie/ayan_acharyya_codes/projection_plot_nondefault.py --system ayan_pleiades --foggie_dir bigbox --run 25Mpc_DM_256-L1 --halo test --output RD0111 --width 1000 --do dm --proj x --width 500 --center 0.54212825,0.45856575,0.504577
                 run projection_plot_nondefault.py --halo test --width 1000 --run 25Mpc_DM_256-L1 --center 0.54212825,0.45856575,0.5045
                 run projection_plot_nondefault.py --halo test --width 1000 --run 25Mpc_DM_256-L2 --center 0.54212825,0.462472,0.50848325
                 run projection_plot_nondefault.py --halo test --width 1000 --run 25Mpc_DM_256-L3 --center 0.54212825,0.45856575,0.50848325
                 run projection_plot_nondefault.py --halo 8894 --annotate_grid --run 25Mpc_DM_256-L3 --center 0.5442269,0.45738622,0.50917259 --output RD0021
                 run projection_plot_nondefault.py --halo 5133 --width 1000 --run 25Mpc_DM_256-L3 --center_offset " 117,93,-73"

"""
from header import *
from util import *

# -----------------------------------------
def projection_plot(args):
    '''
    Function to generate a projection plot for simulations that are NOT from the original FOGGIE halos
    '''
    start_time = time.time()
    # ------------- paths, dict, etc. set up -------------------------------
    if args.system == 'ayan_hd' or args.system == 'ayan_local': args.root_dir = '/Users/acharyya/models/simulation_output/'
    elif args.system == 'ayan_pleiades': args.root_dir = '/nobackup/aachary2/'
    halo_name = 'halo_' + args.halo
    output_path = args.root_dir + args.foggie_dir + '/' + halo_name + '/'
    snap_name = args.root_dir + args.foggie_dir + '/' + halo_name + '/' + args.run + '/' + args.output + '/' + args.output

    field_dict = {'dm':('deposit', 'all_density'), 'gas': ('gas', 'density'), 'cellsize':'d' + args.projection}
    cmap_dict = {'gas': density_color_map, 'dm': plt.cm.gist_heat, 'cellsize': discrete_cmap}
    zlim_dict = {'gas': (1e-5, 5e-2), 'dm': (1e-4, 1e-1), 'cellsize': (1e-1, 1e1)}

    # ----------- read halo catalogue, to get center -------------------
    try:
        halos = Table.read('/nobackup/jtumlins/CGM_bigbox/25Mpc_256_shielded-L0/BigBox_z2_rockstar/out_0.list', format='ascii', header_start=0)
        index = [halos['ID'] == int(args.halo[:4])]
        thishalo = halos[index]
        center_L0 = np.array([thishalo['X'][0]/25., thishalo['Y'][0]/25., thishalo['Z'][0]/25.]) # divided by 25 to convert Mpc units to code units
        print('Center for L0_gas =', center_L0)
    except:
        pass

    if args.center is None: center = center_L0
    else: center = args.center
    center = center + np.array(args.center_offset) / 255.
    print('Offset =', args.center_offset, '\nCenter for current plot =', center)

    # ------------- main plotting -------------------------------
    ds = yt.load(snap_name)  # last output
    if args.width is None:
        box = ds.r[(center[0] - ds.arr(0.5 * 1, 'Mpc').in_units('code_length').value.tolist()): (center[0] + ds.arr(0.5 * 1, 'Mpc').in_units('code_length').value.tolist()), 0:1, 0:1] # slicing out 1 Mpc chunk along LoS
        if args.do == 'cellsize': p = yt.SlicePlot(ds, args.projection, field_dict[args.do], center=center, data_source=box)
        else: p = yt.ProjectionPlot(ds, args.projection, field_dict[args.do], center=center, data_source=box)
        width_text = ''
    else:
        box = ds.r[(center[0] - ds.arr(0.5 * args.width, 'kpc').in_units('code_length').value.tolist()): (center[0] + ds.arr(0.5 * args.width, 'kpc').in_units('code_length').value.tolist()), 0:1, 0:1]
        if args.do == 'cellsize': p = yt.SlicePlot(ds, args.projection, field_dict[args.do], center=center, width=(args.width, 'kpc'))#, data_source=box)
        else: p = yt.ProjectionPlot(ds, args.projection, field_dict[args.do], center=center, data_source=box, width=(args.width, 'kpc'))
        width_text = '_width' + str(args.width) + 'kpc'

    # -------------annotations and limits -------------------------------
    if args.annotate_grids: p.annotate_grids(min_level=args.min_level)
    p.annotate_text((0.06, 0.12), args.halo, coord_system="axis")
    if not args.do == 'dm': p.set_cmap(field_dict[args.do], cmap_dict[args.do])

    # if args.do == 'cellsize': p.plots[field_dict[args.do]].cb.set_label('cell size (kpc)')
    if args.do == 'cellsize': p.set_unit(field_dict[args.do], 'kpc')
    try: p.set_zlim(field_dict[args.do], zmin=zlim_dict[args.do][0], zmax=zlim_dict[args.do][1])
    except: pass

    # -------------optional annotations (if Rvir and M info exists) -------------------------------
    try:
        p.annotate_sphere(center, radius=(thishalo['Rvir'], "kpc"), circle_args={"color": "white"})
        p.annotate_text((0.06, 0.08), "M = "+"{0:.2e}".format(thishalo['Mvir'][0]), coord_system="axis")
    except:
        pass

    p.save(output_path + halo_name + '_' + args.run + '_' + args.output + '_' + args.projection + '_' + args.do + width_text + '.png', mpl_kwargs={'dpi': 500})
    print_master('Completed in %s minutes' % ((time.time() - start_time) / 60), args)

    return p

# -----main code-----------------
if __name__ == '__main__':
    # ------------- arg parse -------------------------------
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='''identify satellites in FOGGIE simulations''')

    parser.add_argument('--system', metavar='system', type=str, action='store', default='ayan_pleiades', help='Which system are you on? Default is ayan_pleiades')
    parser.add_argument('--foggie_dir', metavar='foggie_dir', type=str, action='store', default='bigbox', help='Specify which directory the dataset lies in')
    parser.add_argument('--halo', metavar='halo', type=str, action='store', default='test', help='which halo?')
    parser.add_argument('--run', metavar='run', type=str, action='store', default='25Mpc_DM_256-L1', help='which run?')
    parser.add_argument('--output', metavar='output', type=str, action='store', default='RD0111', help='which output?')
    parser.add_argument('--do', metavar='do', type=str, action='store', default='dm', help='Which particles do you want to plot? Default is gas')
    parser.add_argument('--projection', metavar='projection', type=str, action='store', default='x', help='Which projection do you want to plot, i.e., which axis is your line of sight? Default is x')
    parser.add_argument('--width', metavar='width', type=float, action='store', default=None, help='the extent to which plots will be rendered (each side of a square box), in kpc; default is 1000 kpc')
    parser.add_argument('--fullbox', dest='fullbox', action='store_true', default=False, help='Use full refine box, ignoring args.width?, default is no')
    parser.add_argument('--silent', dest='silent', action='store_true', default=False, help='Suppress all print statements?, default is no')
    parser.add_argument('--center', metavar='center', type=str, action='store', default=None, help='center of projection in code units')
    parser.add_argument('--center_offset', metavar='center_offset', type=str, action='store', default='0,0,0', help='offset from center in integer cell units')
    parser.add_argument('--print_to_file', dest='print_to_file', action='store_true', default=False, help='Redirect all print statements to a file?, default is no')
    parser.add_argument('--annotate_grids', dest='annotate_grids', action='store_true', default=False, help='annotate grids?, default is no')
    parser.add_argument('--min_level', dest='min_level', type=int, action='store', default=3, help='annotate grids min level, default is 3')

    args = parser.parse_args()
    if args.center is not None: args.center = [float(item) for item in args.center.split(',')]
    args.center_offset = [int(item) for item in args.center_offset.split(',')]

    p = projection_plot(args)