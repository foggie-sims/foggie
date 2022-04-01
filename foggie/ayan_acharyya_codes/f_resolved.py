##!/usr/bin/env python3

"""

    Title :      f_resolved
    Notes :      compute the fraction of cells, fraction of volume and fraction of mass resolved at different levels of refinement, based on Raymond's f_Resolved.py
    Output :     histograms, and STDOUT
    Author :     Ayan Acharyya
    Started :    Apr 2022
    Example :    run f_resolved.py --system ayan_local --halo 8508 --output RD0042 --foggie_dir foggie --fullbox
                 run f_resolved.py --system ayan_pleiades --halo 8508 --output RD0020 --fullbox
                 run f_resolved.py --system ayan_pleiades --foggie_dir bigbox --halo 5205 --run nref7c_nref7f --output RD0111 --width 200 --get_center_track

"""
from header import *
from util import *
start_time = time.time()

# ---------plotting the fractions-----------
def plothist(df, args):
    fig, ax = plt.subplots()
    bar_width = 0.25

    fracs_to_plot = [item for item in df.columns if 'f_' in item]
    for index, thisfrac in enumerate(fracs_to_plot): ax.bar(np.arange(len(df)) + index * bar_width, df[thisfrac], width=bar_width, label=thisfrac)

    plt.ylim(0, 1)
    plt.ylabel('Fraction', fontsize=args.fontsize)
    plt.xlabel('Refinement level', fontsize=args.fontsize)
    plt.xticks([r + bar_width for r in range(len(df))], df['level'])
    plt.legend(fontsize=args.fontsize)

    Path(output_path + 'figs/').mkdir(parents=True, exist_ok=True)
    figname = output_path + 'figs/halo_' + args.halo + '_output_' + args.output + '_width_%.2Fkpc_fresolved.png' % args.width
    fig.savefig(figname)
    plt.show(block=False)
    print('\nSaved figure at', figname)
    return fig

# -----main code-----------------
if __name__ == '__main__':
    # ------------- arg parse -------------------------------
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='''identify satellites in FOGGIE simulations''')

    parser.add_argument('--system', metavar='system', type=str, action='store', default='ayan_pleiades', help='Which system are you on? Default is ayan_pleiades')
    parser.add_argument('--foggie_dir', metavar='foggie_dir', type=str, action='store', default='bigbox', help='Specify which directory the dataset lies in')
    parser.add_argument('--halo', metavar='halo', type=str, action='store', default='test', help='which halo?')
    parser.add_argument('--run', metavar='run', type=str, action='store', default='nref11c_nref9f', help='which run?')
    parser.add_argument('--output', metavar='output', type=str, action='store', default='RD0111', help='which output?')
    parser.add_argument('--width', metavar='width', type=float, action='store', default=200, help='the extent to which plots will be rendered (each side of a square box), in kpc; default is 200 kpc')
    parser.add_argument('--fullbox', dest='fullbox', action='store_true', default=False, help='Use full refine box, ignoring args.width?, default is no')
    parser.add_argument('--silent', dest='silent', action='store_true', default=False, help='Suppress all print statements?, default is no')
    parser.add_argument('--pwd', dest='pwd', action='store_true', default=False, help='Just use the current working directory?, default is no')
    parser.add_argument('--center', metavar='center', type=str, action='store', default=None, help='center of projection in code units')
    parser.add_argument('--get_center_track', dest='get_center_track', action='store_true', default=False, help='get the halo cneter automatically from the center track file?, default is no')
    parser.add_argument('--fontsize', metavar='fontsize', type=int, action='store', default=15, help='fontsize of plot labels, etc.; default is 15')

    args = parser.parse_args()

    # -------read in correct halo and output at appropriate paths--------------
    original_halos = ['8508', '5036', '5016', '4123', '2878', '2392']
    if args.halo in original_halos:
        ds, refine_box = load_sim(args, region='refine_box', do_filter_particles=False)
        _, output_path, _, _, _, _, _, _ = get_run_loc_etc(args)
        center = refine_box.center
        if args.fullbox: args.width = ds.refine_width # in kpc
    else:
        if args.system == 'ayan_hd' or args.system == 'ayan_local': root_dir = '/Users/acharyya/models/simulation_output/'
        elif args.system == 'ayan_pleiades': root_dir = '/nobackup/aachary2/'
        output_path = root_dir + args.foggie_dir + '/halo_' + args.halo + '/' + args.run + '/'
        snap_name = output_path + args.output + '/' + args.output
        ds = yt.load(snap_name)

        if args.fullbox: args.width = 200 / (1 + ds.current_redshift) # converting 200 kpc comoving width to physical width at given z
        if args.center is not None:
            center = [float(item) for item in args.center.split(',')]
        else:
            df_track_int = pd.read_table(output_path + 'center_track_interp.dat', delim_whitespace=True)
            center = interp1d(df_track_int['redshift'], df_track_int[['center_x', 'center_y', 'center_z']], axis=0)(ds.current_redshift)

    # -------cut out a box around the center of width either given args.width or refine box width--------------
    halfbox = ds.arr(0.5 * args.width, 'kpc').in_units('code_length')
    left_edge = center - halfbox
    right_edge = center + halfbox
    box = ds.r[left_edge[0]:right_edge[0], left_edge[1]:right_edge[1], left_edge[2]:right_edge[2]]

    # ---------resolved fraction calculations-----------
    df = pd.DataFrame(columns=('level', 'f_cells', 'f_volume', 'f_mass'))
    grid_level = box['index', 'grid_level']
    cell_volume = box['index', 'cell_volume'].in_units('pc**3')
    cell_mass = box['gas', 'cell_mass'].in_units('Msun')

    for thislevel in np.unique(grid_level):
        atthislevel = np.where(grid_level == thislevel)
        f_cells = len(grid_level[atthislevel]) / len(grid_level)
        f_volume = np.sum(cell_volume[atthislevel]) / np.sum(cell_volume)
        f_mass = np.sum(cell_mass[atthislevel]) / np.sum(cell_mass)

        df.loc[len(df)] = [thislevel.value, f_cells, f_volume.value, f_mass.value]

    print('\nHalo:', args.halo, '; Output:', args.output, '; Redshift:', ds.current_redshift, '; Fractions:\n', df)

    fig = plothist(df, args)
    print('Completed in %s' % (datetime.timedelta(minutes=(time.time() - start_time) / 60)))
