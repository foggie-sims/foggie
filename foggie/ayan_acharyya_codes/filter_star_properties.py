##!/usr/bin/env python3

""""

    Title :      filter_star_properties
    Notes :      To extract physial properties of young (< 10Myr) stars e.g., position, velocity, mass etc. and output to an ASCII file
    Author:      Ayan Acharyya
    Started  :   January 2021
    Example :    run filter_star_properties.py --system ayan_local --halo 8508 --output RD0042

"""
from header import *
from projection_plot import *
start_time = time.time()

# ---------to parse keyword arguments----------
def parse_args(haloname, RDname):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='''identify satellites in FOGGIE simulations''')
    parser.add_argument('--system', metavar='system', type=str, action='store', help='Which system are you on? Default is Jase')
    parser.set_defaults(system='ayan_local')

    parser.add_argument('--do', metavar='do', type=str, action='store', help='Which particles do you want to plot? Default is gas')
    parser.set_defaults(do='gas')

    parser.add_argument('--run', metavar='run', type=str, action='store', help='which run? default is natural')
    parser.set_defaults(run="nref11c_nref9f")

    parser.add_argument('--halo', metavar='halo', type=str, action='store', help='which halo? default is 8508 (Tempest)')
    parser.set_defaults(halo=haloname)

    parser.add_argument('--proj', metavar='proj', type=str, action='store', help='Which projection do you want to plot? Default is x')
    parser.set_defaults(proj='x')

    parser.add_argument('--clobber', dest='clobber', action='store_true', help='overwrite existing outputs with same name?, default is no')
    parser.set_defaults(clobber=False)

    parser.add_argument('--output', metavar='output', type=str, action='store', help='which output? default is RD0020')
    parser.set_defaults(output=RDname)

    parser.add_argument('--plotmap', metavar='plotmap', action='store_true', help='plot projection map? default is no')
    parser.set_defaults(plotmap=False)

    args = parser.parse_args()
    return args

# -----------------------------------------------------------------------------------
if __name__ == '__main__':
    args = parse_args('8508', 'RD0042')

    foggie_dir, output_dir, run_loc, code_path, trackname, haloname, spectra_dir, infofile = get_run_loc_etc(args)
    outfilename = HOME + output_dir + 'txtfiles/' + args.output + '_young_star_properties.txt'

    if not os.path.exists(outfilename) or args.clobber:
        ds, refine_box = load_sim(args, region='refine_box')
        ad = ds.all_data()

        if args.plotmap:
            prj = make_projection_plots(ds=refine_box.ds, center=ds.refine_box_center, \
                                        refine_box=refine_box, x_width=ds.refine_width * kpc, \
                                        fig_dir=output_dir + 'figs/', haloname=args.output, name=halo_dict[args.halo], \
                                        fig_end='projection', do=[ar for ar in args.do.split(',')],
                                        axes=[ar for ar in args.proj.split(',')], is_central=True, add_arrow=False,
                                        add_velocity=False)

        print 'Extracting parameters for '+ str(len(ad['young_stars', 'particle_position_x'])) +' young stars...'
        xgrid = ad['young_stars', 'particle_position_x']
        zgrid = ad['young_stars', 'particle_position_z']
        ygrid = ad['young_stars', 'particle_position_y']

        px = xgrid.in_units('kpc')
        py = ygrid.in_units('kpc')
        pz = zgrid.in_units('kpc')

        vx = ad['young_stars', 'particle_velocity_x'].in_units('km/s')
        vy = ad['young_stars', 'particle_velocity_y'].in_units('km/s')
        vz = ad['young_stars', 'particle_velocity_z'].in_units('km/s')

        age = ad['young_stars', 'age'].in_units('Myr')
        mass = ad['young_stars', 'particle_mass'].in_units('Msun')

        coord = np.vstack([xgrid, ygrid, zgrid]).transpose()
        # ambient gas properties only at point where young stars are located:
        pres = ds.find_field_values_at_points([('gas', 'pressure')], coord)
        den = ds.find_field_values_at_points([('gas', 'density')], coord)
        temp = ds.find_field_values_at_points([('gas', 'temperature')], coord)
        Z = ds.find_field_values_at_points([('gas', 'metallicity')], coord)

        paramlist = pd.DataFrame({'pos_x':px, 'pos_y':py, 'pos_z':pz, 'vel_x':vx, 'vel_y':vy, 'vel_z':vz, 'age':age, 'mass':mass, 'gas_density':den, 'gas_pressure':pres, 'gas_temp':temp, 'gas_metal':Z})
        header = 'Units for the following columns: \n\
        pos_x, pos_y, pos_z: kpc \n\
        vel_x, vel_y, vel_z: km/s \n\
        age: Myr \n\
        mass: Msun \n\
        gas_density in a cell: simulation units \n\
        gas_pressure in a cell: simulation units \n\
        gas_temp in a cell: simulation units \n\
        gas_metal in a cell: simulation units'
        np.savetxt(outfilename, [], header=header, comments='#')
        paramlist.to_csv(outfilename, sep='\t', mode='a', index=None)
        print 'Saved file at', outfilename
    else:
        print 'Reading from existing file', outfilename
        paramlist = pd.read_table(outfilename, delim_whitespace=True, comment='#')

    print args.output + ' completed in %s minutes' % ((time.time() - start_time) / 60)