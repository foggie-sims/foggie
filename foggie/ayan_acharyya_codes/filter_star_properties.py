##!/usr/bin/env python3

"""

    Title :      filter_star_properties
    Notes :      To extract physial properties of young (< 10Myr) stars e.g., position, velocity, mass etc. and output to an ASCII file
    Output :     One pandas dataframe as a txt file
    Author :     Ayan Acharyya
    Started :    January 2021
    Example :    run filter_star_properties.py --system ayan_local --halo 8508 --output RD0042

"""
from header import *
from projection_plot import make_projection_plots
start_time = time.time()

# -----------------------------------------------------------------------------------
if __name__ == '__main__':
    args = parse_args('8508', 'RD0042')
    foggie_dir, output_dir, run_loc, code_path, trackname, haloname, spectra_dir, infofile = get_run_loc_etc(args)
    outfilename = output_dir + 'txtfiles/' + args.output + '_young_star_properties.txt'

    # ----------------------Reading in simulation data-------------------------------------------
    if not os.path.exists(outfilename) or args.clobber:
        if not os.path.exists(outfilename): print(outfilename + ' does not exist. Creating afresh..')
        elif args.clobber: print(outfilename + ' exists but over-writing..')

        ds, refine_box = load_sim(args, region='refine_box')
        ad = ds.all_data()

        if args.plotmap:
            prj = make_projection_plots(ds=refine_box.ds, center=ds.refine_box_center, \
                                        refine_box=refine_box, x_width=ds.refine_width * kpc, \
                                        fig_dir=output_dir + 'figs/', haloname=args.output, name=halo_dict[args.halo], \
                                        fig_end='projection', do=[ar for ar in args.do.split(',')],
                                        axes=[ar for ar in args.proj.split(',')], is_central=True, add_arrow=False,
                                        add_velocity=False)

        xgrid = ad['young_stars', 'particle_position_x']
        print('Extracting parameters for '+ str(len(xgrid)) +' young stars...')
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

        # saving the header (with units, etc.) first in a new txt file
        header = 'Units for the following columns: \n\
        pos_x, pos_y, pos_z: kpc \n\
        vel_x, vel_y, vel_z: km/s \n\
        age: Myr \n\
        mass: Msun \n\
        gas_density in a cell: ' + ds.field_info[('gas', 'density')].units + ' \n\
        gas_pressure in a cell: ' + ds.field_info[('gas', 'pressure')].units + ' \n\
        gas_temp in a cell: ' + ds.field_info[('gas', 'temperature')].units + ' \n\
        gas_metal in a cell: ' + ds.field_info[('gas', 'metallicity')].units
        np.savetxt(outfilename, [], header=header, comments='#')

        # creating and saving the dataframe itself to the file which already has the header
        paramlist = pd.DataFrame({'pos_x':px, 'pos_y':py, 'pos_z':pz, 'vel_x':vx, 'vel_y':vy, 'vel_z':vz, 'age':age, 'mass':mass, 'gas_density':den, 'gas_pressure':pres, 'gas_temp':temp, 'gas_metal':Z})
        paramlist.to_csv(outfilename, sep='\t', mode='a', index=None)
        print('Saved file at', outfilename)
    else:
        print('Reading from existing file', outfilename)
        paramlist = pd.read_table(outfilename, delim_whitespace=True, comment='#')

    print(args.output + ' completed in %s minutes' % ((time.time() - start_time) / 60))