##!/usr/bin/env python3

"""

    Title :      projection_plot
    Notes :      make projection plots for new bigbox runs
    Output :     projection plots as png files
    Author :     Ayan Acharyya
    Started :    January 2021
    Example :    run /nobackup/aachary2/ayan_codes/foggie/foggie/ayan_acharyya_codes/projection_plot_nondefault.py --system ayan_pleiades --foggie_dir bigbox --run 25Mpc_DM_256-L1 --halo test --output RD0111 --do dm --proj x --galrad 500 --center 0.54212825,0.45856575,0.504577
                 run /nobackup/aachary2/ayan_codes/foggie/foggie/ayan_acharyya_codes/projection_plot_nondefault.py --halo test --run 25Mpc_DM_256-L1 --center 0.54212825,0.45856575,0.504577

"""
from header import *
from util import *
start_time = time.time()

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
    parser.add_argument('--galrad', metavar='galrad', type=float, action='store', default=500., help='the radial extent (in each spatial dimension) to which computations will be done, in kpc; default is 20')
    parser.add_argument('--fullbox', dest='fullbox', action='store_true', default=False, help='Use full refine box, ignoring args.galrad?, default is no')
    parser.add_argument('--silent', dest='silent', action='store_true', default=False, help='Suppress all print statements?, default is no')
    parser.add_argument('--center', metavar='center', type=str, action='store', default='-0.45787175,0.45856575,0.504577', help='center of projection in code units')
    parser.add_argument('--print_to_file', dest='print_to_file', action='store_true', default=False, help='Redirect all print statements to a file?, default is no')

    args = parser.parse_args()
    args.center = [float(item) for item in args.center.split(',')]

    # ------------- paths, dict, etc. set up -------------------------------
    if args.system == 'ayan_hd' or args.system == 'ayan_local': root_dir = '/Users/acharyya/Work/astro/'
    elif args.system == 'ayan_pleiades': root_dir = '/nobackup/aachary2/'
    halo_name = 'halo_' + args.halo
    output_path = root_dir + args.foggie_dir + '/' + halo_name + '/'
    snap_name = root_dir + args.foggie_dir + '/' + halo_name + '/' + args.run + '/' + args.output + '/' + args.output

    field_dict = {'dm':('deposit', 'all_density')}

    # ------------- main plotting -------------------------------
    ds = yt.load(snap_name)  # last output
    p = yt.ProjectionPlot(ds, args.projection, field_dict[args.do], center=args.center, width=(args.galrad, 'kpc'))
    p.save(output_path + args.run + '_' + args.output + '_' + args.projection + '_' + args.do + '_density.png', mpl_kwargs={'dpi': 500})

    print_master('Completed in %s minutes' % ((time.time() - start_time) / 60), args)
