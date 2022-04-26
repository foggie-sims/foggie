##!/usr/bin/env python3

"""

    Title :      projection_plot
    Notes :      make projection plots for new bigbox runs
    Output :     projection plots as png files
    Author :     Ayan Acharyya
    Started :    January 2022
    Example :    run /nobackup/aachary2/ayan_codes/foggie/foggie/ayan_acharyya_codes/projection_plot_nondefault.py --system ayan_pleiades --foggie_dir bigbox --run 25Mpc_DM_256-L1 --halo test --output RD0111 --width 1000 --do dm --proj x --width 500 --center 0.54212825,0.45856575,0.504577
                 run projection_plot_nondefault.py --halo test --width 1000 --run 25Mpc_DM_256-L1 --center 0.54212825,0.45856575,0.5045
                 run projection_plot_nondefault.py --halo test --width 1000 --run 25Mpc_DM_256-L2 --center 0.54212825,0.462472,0.50848325
                 run projection_plot_nondefault.py --halo test --width 1000 --run 25Mpc_DM_256-L3 --center 0.54212825,0.45856575,0.50848325
                 run projection_plot_nondefault.py --halo 8894 --annotate_grid --run 25Mpc_DM_256-L3 --center 0.5442269,0.45738622,0.50917259 --output RD0021
                 run projection_plot_nondefault.py --halo 5133 --width 1000 --run 25Mpc_DM_256-L3 --center_offset " 117,93,-73"
                 run projection_plot_nondefault.py --halo 5205 --width 200 --run 25Mpc_DM_256-L3-gas --output RD0028 --get_center_track --do cellsize
                 run projection_plot_nondefault.py --halo 5205 --width 200 --run nref7c_nref7f --do_all_sims --use_onlyDD --nevery 5 --get_center_track --do gas
                 run projection_plot_nondefault.py --halo 5205 --width 2000 --run nref7c_nref7f --output RD0111 --get_center_track --do mrp --annotate_grids --annotate_box

"""
from header import *
from util import *
start_time = time.time()
sns.set_style('ticks') # instead of darkgrid, so that there are no grids overlaid on the projections

# -----------------------------------------------------------
def ptype4(pfilter, data):
    filter = data[(pfilter.filtered_type, "particle_type")] == 4
    return filter

# ---------------------------------------------------------
def annotate_box(p, width, ds, unit='kpc', projection='x', center=[0.5, 0.5, 0.5], linewidth=2, color='red'):
    '''
    Function to annotate a given yt plot with a box of a given size (width) centered on a given center
    '''
    width_code = ds.arr(width, unit).in_units('code_length').value.tolist()
    proj_dict = {'x': 1, 'y': 2, 'z': 0}

    for left_array, right_array in [[np.array([-1, -1, 0]), np.array([-1, +1, 0])], \
                                    [np.array([-1, +1, 0]), np.array([+1, +1, 0])], \
                                    [np.array([+1, +1, 0]), np.array([+1, -1, 0])], \
                                    [np.array([+1, -1, 0]), np.array([-1, -1, 0])]]:
        p.annotate_line(center + np.roll(left_array, proj_dict[projection]) * width_code/2, center + np.roll(right_array, proj_dict[projection]) * width_code/2, coord_system='data', plot_args={'color': color, 'linewidth': linewidth},)

    return p

# -----------------------------------------
def get_box(ds, projection, center, width):
    '''
    Function to slice out box of 'width' kpc along a given LoS 'projection', around 'center' from dataset 'ds'
    '''
    proj_dict = {'x': [1, 0, 0], 'y': [0, 1, 0], 'z': [0, 0, 1]}
    proj_array = np.array(proj_dict[projection])

    left_limit = center[np.where(proj_array == 1)[0][0]] - ds.arr(0.5 * width, 'kpc').in_units('code_length').value.tolist()
    right_limit = center[np.where(proj_array == 1)[0][0]] + ds.arr(0.5 * width, 'kpc').in_units('code_length').value.tolist()

    left_edge = np.multiply(np.tile(left_limit, 3), proj_array)
    right_edge = np.multiply(np.tile(right_limit, 3), proj_array) + np.array(proj_array ^ 1)
    print('Extracting box within left and right edges as', left_edge, right_edge)

    box = ds.r[left_edge[0]:right_edge[0], left_edge[1]:right_edge[1], left_edge[2]:right_edge[2]]
    return box

# -----------------------------------------
def projection_plot(args):
    '''
    Function to generate a projection plot for simulations that are NOT from the original FOGGIE halos
    '''
    start_time = time.time()

    field_dict = {'dm':('deposit', 'all_density'), 'gas': ('gas', 'density'), 'cellsize': ('gas', 'd' + args.projection), 'grid': ('index', 'grid_level'), 'mrp': ('deposit', 'ptype4_mass')}
    cmap_dict = {'gas': density_color_map, 'dm': plt.cm.gist_heat, 'cellsize': discrete_cmap, 'grid':'viridis', 'mrp':'viridis'}
    zlim_dict = {'gas': (1e-5, 5e-2), 'dm': (1e-4, 1e-1), 'cellsize': (1e-1, 1e1), 'grid': (1, 11), 'mrp': (1e57, 1e65)}

    # ----------- get halo center, either rough center at z=2 from halo catalogue or more accurate center from center track file-------------------
    ds = yt.load(args.snap_name)  # last output
    if 'pleiades' in args.system: halos_filename = '/nobackup/jtumlins/CGM_bigbox/25Mpc_256_shielded-L0/BigBox_z2_rockstar/out_0.list'
    elif args.system == 'ayan_local': halos_filename = '/Users/acharyya/Downloads/out_0.list'

    if 'mrp' in args.do:
        yt.add_particle_filter("ptype4", function=ptype4, requires=["particle_type"])
        ds.add_particle_filter('ptype4')

    halos = Table.read(halos_filename, format='ascii', header_start=0)
    thishalo = halos[halos['ID'] == int(args.halo[:4])]

    if args.center is None:
        if args.get_center_track:
            trackfile = args.root_dir + args.foggie_dir + '/' + args.halo_name + '/' + args.run + '/center_track_interp.dat'
            df_track_int = pd.read_table(trackfile, delim_whitespace=True)
            center = interp1d(df_track_int['redshift'], df_track_int[['center_x', 'center_y', 'center_z']], axis=0)(ds.current_redshift)
            print('Center from center track file =', center)
        else:
            center = np.array([thishalo['X'][0] / 25., thishalo['Y'][0] / 25., thishalo['Z'][0] / 25.])  # divided by 25 to convert Mpc units to code units
            print('Center for L0_gas =', center)
    else:
        center = args.center
    center = center + np.array(args.center_offset) / 255.
    print('Offset =', args.center_offset, '\nCenter for current plot =', center)

    # ------------- main plotting -------------------------------
    if args.width is None:
        box = get_box(ds, args.projection, center, 1000.) # slicing out 1 Mpc chunk along LoS anyway
        if args.do == 'cellsize': p = yt.SlicePlot(ds, args.projection, field_dict[args.do], center=center, data_source=box)
        else: p = yt.ProjectionPlot(ds, args.projection, field_dict[args.do], center=center, data_source=box)
        width_text = ''
    else:
        box = get_box(ds, args.projection, center, args.width)
        if args.do == 'cellsize': p = yt.SlicePlot(ds, args.projection, field_dict[args.do], center=center, width=(args.width, 'kpc'), data_source=box)
        else: p = yt.ProjectionPlot(ds, args.projection, field_dict[args.do], center=center, width=(args.width, 'kpc'), data_source=box)
        width_text = '_width' + str(args.width) + 'kpc'

    # -------------annotations and limits -------------------------------
    if args.annotate_grids:
        p.annotate_grids(min_level=args.min_level)

    if args.annotate_box:
        for thisbox in [200., 400.]: # comoving size at z=0 in kpc
            thisphys = thisbox / (1 + ds.current_redshift) / ds.hubble_constant # physical size at current redshift in kpc
            p = annotate_box(p, thisphys, ds, unit='kpc', projection=args.projection, center=center)

    p.annotate_text((0.06, 0.12), args.halo, coord_system='axis')
    p.annotate_timestamp(corner='lower_right', redshift=True, draw_inset_box=True)
    p.annotate_marker(center, coord_system='data')
    if not args.do == 'dm': p.set_cmap(field_dict[args.do], cmap_dict[args.do])

    # if args.do == 'cellsize': p.plots[field_dict[args.do]].cb.set_label('cell size (kpc)')
    if args.do == 'cellsize': p.set_unit(field_dict[args.do], 'kpc')
    try: p.set_zlim(field_dict[args.do], zmin=zlim_dict[args.do][0], zmax=zlim_dict[args.do][1])
    except: pass

    # -------------optional annotations (if Rvir and M info exists) -------------------------------
    if len(thishalo) > 0:
        p.annotate_sphere(center, radius=(thishalo['Rvir'], "kpc"), circle_args={"color": "white"})
        p.annotate_text((0.06, 0.08), "M = "+"{0:.2e}".format(thishalo['Mvir'][0]), coord_system="axis")

    target_dir = args.root_dir + args.foggie_dir + '/' + args.halo_name + '/figs/'
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    run = args.run.replace('/', '_')
    p.save(target_dir + args.halo_name + '_' + run + '_' + args.output + '_' + args.projection + '_' + args.do + width_text + '.png', mpl_kwargs={'dpi': 500})
    print('This snapshot completed in %s' % (datetime.timedelta(minutes=(time.time() - start_time) / 60)))

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
    parser.add_argument('--width', metavar='width', type=float, action='store', default=None, help='the extent to which plots will be rendered (each side of a square box), in kpc; default is None, which corresponds to the whole domain')
    parser.add_argument('--fullbox', dest='fullbox', action='store_true', default=False, help='Use full refine box, ignoring args.width?, default is no')
    parser.add_argument('--silent', dest='silent', action='store_true', default=False, help='Suppress all print statements?, default is no')
    parser.add_argument('--center', metavar='center', type=str, action='store', default=None, help='center of projection in code units')
    parser.add_argument('--center_offset', metavar='center_offset', type=str, action='store', default='0,0,0', help='offset from center in integer cell units')
    parser.add_argument('--print_to_file', dest='print_to_file', action='store_true', default=False, help='Redirect all print statements to a file?, default is no')
    parser.add_argument('--annotate_grids', dest='annotate_grids', action='store_true', default=False, help='annotate grids?, default is no')
    parser.add_argument('--annotate_box', dest='annotate_box', action='store_true', default=False, help='annotate box?, default is no')
    parser.add_argument('--min_level', dest='min_level', type=int, action='store', default=3, help='annotate grids min level, default is 3')
    parser.add_argument('--get_center_track', dest='get_center_track', action='store_true', default=False, help='get the halo cneter automatically from the center track file?, default is no')
    parser.add_argument('--do_all_sims', dest='do_all_sims', action='store_true', default=False, help='Run the code on all simulation snapshots available for a given halo?, default is no')
    parser.add_argument('--use_onlyRD', dest='use_onlyRD', action='store_true', default=False, help='Use only the RD snapshots available for a given halo?, default is no')
    parser.add_argument('--use_onlyDD', dest='use_onlyDD', action='store_true', default=False, help='Use only the DD snapshots available for a given halo?, default is no')
    parser.add_argument('--nevery', metavar='nevery', type=int, action='store', default=1, help='use every nth snapshot when do_all_sims is specified; default is 1 i.e., all snapshots will be used')

    args = parser.parse_args()
    if args.center is not None: args.center = [float(item) for item in args.center.split(',')]
    args.center_offset = [int(item) for item in args.center_offset.split(',')]
    args.output_arr = [item for item in args.output.split(',')]

    # ------------- paths, dict, etc. set up -------------------------------
    if args.system == 'ayan_hd' or args.system == 'ayan_local': args.root_dir = '/Users/acharyya/models/simulation_output/'
    elif args.system == 'ayan_pleiades': args.root_dir = '/nobackup/aachary2/'
    args.halo_name = 'halo_' + args.halo
    args.output_path = args.root_dir + args.foggie_dir + '/' + args.halo_name + '/' + args.run + '/'
    if args.do_all_sims: args.output_arr = np.array(get_all_sims_for_this_halo(args, given_path=args.output_path))[:, 1] # all snapshots of this particular halo

    for index, thisoutput in enumerate(args.output_arr):
        args.output = thisoutput
        print('Starting snapshot', args.output, 'i.e.,', index+1, 'out of', len(args.output_arr), 'snapshots..')
        args.snap_name = args.output_path + args.output + '/' + args.output
        p = projection_plot(args)

    print('All snapshots completed in %s' % (datetime.timedelta(minutes=(time.time() - start_time) / 60)))
