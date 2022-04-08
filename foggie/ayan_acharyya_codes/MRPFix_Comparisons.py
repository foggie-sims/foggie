#!/usr/bin/env python
"""

    Title :      MRPFix_Comparisons
    Notes :      projection plots for comparing MRPFix region and Must Refine region, based on JT's MRPFix_Comparisons.ipynb
    Output :     saves yt plots
    Author :     Ayan Acharyya
    Started :    Apr 2022
    Example :    run MRPFix_Comparisons.py --system ayan_local --halo 8508 --output RD0042 --foggie_dir foggie --width 2000
                 run MRPFix_Comparisons.py --system ayan_pleiades --halo 8508 --output RD0020 --width 2000
                 run MRPFix_Comparisons.py --system ayan_pleiades --foggie_dir bigbox --halo 5205 --run nref7c_nref7f --output RD0111 --width 2000 --get_center_track

"""

from header import *
from util import *
from yt.funcs import mylog


# -----------------------------------------------------------
def ptype1(pfilter, data):
    filter = data[(pfilter.filtered_type, "particle_type")] == 1
    return filter

# -----------------------------------------------------------
def ptype2(pfilter, data):
    filter = data[(pfilter.filtered_type, "particle_type")] == 2
    return filter

# -----------------------------------------------------------
def ptype4(pfilter, data):
    filter = data[(pfilter.filtered_type, "particle_type")] == 4
    return filter

# -----main code-----------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='''identify satellites in FOGGIE simulations''')

    parser.add_argument('--system', metavar='system', type=str, action='store', default='ayan_pleiades', help='Which system are you on? Default is ayan_pleiades')
    parser.add_argument('--foggie_dir', metavar='foggie_dir', type=str, action='store', default=None, help='Specify which directory the dataset lies in')
    parser.add_argument('--halo', metavar='halo', type=str, action='store', default='test', help='which halo?')
    parser.add_argument('--run', metavar='run', type=str, action='store', default='nref11c_nref9f', help='which run?')
    parser.add_argument('--output', metavar='output', type=str, action='store', default='RD0111', help='which output?')
    parser.add_argument('--width', metavar='width', type=float, action='store', default=200, help='the extent to which plots will be rendered (each side of a square box), in kpc; default is 200 kpc')
    parser.add_argument('--fullbox', dest='fullbox', action='store_true', default=False, help='Use full refine box, ignoring args.width?, default is no')
    parser.add_argument('--silent', dest='silent', action='store_true', default=False, help='Suppress all print statements?, default is no')
    parser.add_argument('--pwd', dest='pwd', action='store_true', default=False, help='Just use the current working directory?, default is no')
    parser.add_argument('--center', metavar='center', type=str, action='store', default=None, help='center of projection in code units')
    parser.add_argument('--get_center_track', dest='get_center_track', action='store_true', default=False, help='get the halo cneter automatically from the center track file?, default is no')
    args = parse_args()

    args.original_halos = ['8508', '5036', '5016', '4123', '2878', '2392']
    if args.halo in args.original_halos:
        _, args.output_path, _, _, _, _, _, _ = get_run_loc_etc(args)
    else:
        if args.system == 'ayan_hd' or args.system == 'ayan_local': args.root_dir = '/Users/acharyya/models/simulation_output/'
        elif args.system == 'ayan_pleiades': args.root_dir = '/nobackup/aachary2/'
        args.output_path = args.root_dir + args.foggie_dir + '/halo_' + args.halo + '/' + args.run + '/'

    # -----------------------------------------------------------
    if args.halo in args.original_halos:
        ds, refine_box = load_sim(args, region='refine_box', do_filter_particles=False)
        center = refine_box.center
    else:
        snap_name = args.output_path + args.output + '/' + args.output
        ds = yt.load(snap_name)

        if args.center is not None:
            center = [float(item) for item in args.center.split(',')]
        else:
            df_track_int = pd.read_table(args.output_path + 'center_track_interp.dat', delim_whitespace=True)
            center = interp1d(df_track_int['redshift'], df_track_int[['center_x', 'center_y', 'center_z']], axis=0)(ds.current_redshift)
        center = ds.arr(center, 'code_length')

    # -----------------------------------------------------------
    yt.add_particle_filter("ptype1", function=ptype1, requires=["particle_type"])
    yt.add_particle_filter("ptype2", function=ptype2, requires=["particle_type"])
    yt.add_particle_filter("ptype4", function=ptype4, requires=["particle_type"])

    ds.add_particle_filter('ptype1')
    ds.add_particle_filter('ptype2')
    ds.add_particle_filter('ptype4')

    # -----------------------------------------------------------
    p = yt.ProjectionPlot(ds, 'x', 'density', method='mip', center=center, width=(args.width, 'kpc'))
    p.set_zlim('density', 1e-31, 1e-24)
    p.set_cmap('density', cmap=density_color_map)
    p.annotate_scale(coeff=150., unit='kpc', pos=(0.25, 0.05))
    p.annotate_timestamp(redshift=True)
    p.save(args.root_dir + args.foggie_dir + '/' + args.halo_name + '/' + args.halo_name + '_' + args.run.replace('/', '-') + '_' + args.output + '_density_width' + str(args.width) + 'kpc' + '.png', mpl_kwargs={'dpi': 500})

    # -----------------------------------------------------------
    p = yt.ProjectionPlot(ds, 'x', ('index', 'grid_level'), method='mip', center=center, width=(args.width, 'kpc'))
    p.set_zlim(('index', 'grid_level'), 1, 11)
    p.annotate_timestamp(redshift=True)
    p.annotate_scale(coeff=150., unit='kpc', pos=(0.25,0.05))
    p.save(args.root_dir + args.foggie_dir + '/' + args.halo_name + '/' + args.halo_name + '_' + args.run.replace('/', '-') + '_' + args.output + '_gridlevel_width' + str(args.width) + 'kpc' + '.png', mpl_kwargs={'dpi': 500})

    # -----------------------------------------------------------
    p = yt.ProjectionPlot(ds, 'x', ('deposit', 'ptype4_mass'), center=center, width=(args.width, 'kpc'))
    p.annotate_grids(min_level=4)
    p.annotate_timestamp(redshift=True, text_args={'color': 'black'})
    p.annotate_scale(coeff=40., unit='kpc', pos=(0.25, 0.05))
    p.save(args.root_dir + args.foggie_dir + '/' + args.halo_name + '/' + args.halo_name + '_' + args.run.replace('/', '-') + '_' + args.output + '_MRPmass_width' + str(args.width) + 'kpc' + '.png', mpl_kwargs={'dpi': 500})
