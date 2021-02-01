
""" this is FOGGIE's main routine for generating a halo track from an
  initial guess and a set of snapshots - JT and MSP"""

import yt
from astropy.table import Table
from astropy.io import ascii

import argparse
import sys

from foggie.utils.consistency  import  *
from foggie.utils.get_halo_center import get_halo_center
import numpy as np


def parse_args():
    '''
    Parse command line arguments.  Returns args object.
    '''
    parser = argparse.ArgumentParser(description="finds the center track")

    ## optional arguments
    parser.add_argument('--halo', metavar='halo', type=str, action='store',
                        help='which halo? default is 5016 (Squall)')
    parser.set_defaults(halo="5016")

    args = parser.parse_args()
    return args


def get_center_track(first_center, latesnap, earlysnap, interval):

    ### do this way at high-redshift
    ### snaplist = np.flipud(np.arange(earlysnap,latesnap+1))

    ### do this way at later times
    snaplist = np.arange(earlysnap,latesnap+1)
    print(snaplist)

    t = Table([[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0], ['       ', '       ']],
        names=('redshift', 'x0', 'y0', 'z0', 'name'))

    center_guess = first_center
    search_radius = 10. ### COMOVING KPC

    for isnap in snaplist:
        if (isnap > 999): name = 'DD'+str(isnap)
        if (isnap <= 999): name = 'DD0'+str(isnap)
        if (isnap <= 99): name = 'DD00'+str(isnap)
        if (isnap <= 9): name = 'DD000'+str(isnap)

        print
        print
        print(name)
        ds = yt.load(name+'/'+name)
        comoving_box_size = ds.get_parameter('CosmologyComovingBoxSize')
        print('Comoving Box Size:', comoving_box_size)

        # decreased these from 500 to 100 because outputs so short spaced now
        this_search_radius = search_radius / (1+ds.get_parameter('CosmologyCurrentRedshift'))  ## search radius is in PHYSICAL kpc
        new_center, vel_center = get_halo_center(ds, center_guess, radius=this_search_radius, vel_radius=this_search_radius)
        print(new_center)

        t.add_row( [ds.get_parameter('CosmologyCurrentRedshift'),
            new_center[0], new_center[1],new_center[2], name])

        center_guess = new_center

        p = yt.ProjectionPlot(ds, 'x', 'density', center=new_center, width=(200., 'kpc'))
        p.set_unit(('gas','density'),'Msun/pc**2')
        p.set_zlim('density', density_proj_min, density_proj_max)
        p.set_cmap(field='density', cmap=density_color_map)
        p.annotate_timestamp(corner='upper_left', redshift=True, draw_inset_box=True)
        p.save()
        p = yt.ProjectionPlot(ds, 'y', 'density', center=new_center, width=(200., 'kpc'))
        p.set_unit(('gas','density'),'Msun/pc**2')
        p.set_zlim('density', density_proj_min, density_proj_max)
        p.set_cmap(field='density', cmap=density_color_map)
        p.annotate_timestamp(corner='upper_left', redshift=True, draw_inset_box=True)
        p.save()
        p = yt.ProjectionPlot(ds, 'z', 'density', center=new_center, width=(200., 'kpc'))
        p.set_unit(('gas','density'),'Msun/pc**2')
        p.set_zlim('density', density_proj_min, density_proj_max)
        p.set_cmap(field='density', cmap=density_color_map)
        p.annotate_timestamp(corner='upper_left', redshift=True, draw_inset_box=True)
        p.save()
        print(t)
        ascii.write(t, 'track_temp.dat', format='fixed_width_two_line', overwrite=True)
        print
        print

        ### just in case memory becomes a problem
        ds.index.clear_all_data()
        del ds.index.grid_dimensions
        del ds.index.grid_left_edge
        del ds.index.grid_right_edge
        del ds.index.grid_levels
        del ds.index.grid_particle_count
        del ds.index.grids


    t = t[2:]
    print(t)

    # now interpolate the track to the interval given as a parameter
    n_points = int((np.max(t['redshift']) - np.min(t['redshift']))  / interval)
    newredshift = np.min(t['redshift']) + np.arange(n_points+2) * interval
    newx = np.interp(newredshift, t['redshift'], t['x0'])
    newy = np.interp(newredshift, t['redshift'], t['y0'])
    newz = np.interp(newredshift, t['redshift'], t['z0'])

    tt = Table([ newredshift, newx, newy, newz], names=('redshift','x','y','z'))

    t.write('track.fits',overwrite=True)
    tt.write('track_interpolate.fits',overwrite=True)

    ascii.write(t, 'track.dat', format='fixed_width_two_line', overwrite=True)
    ascii.write(tt, 'track_interpolate.dat', format='fixed_width_two_line', overwrite=True)

    return t, tt


if __name__ == "__main__":
    ## first_center is the center at the last output; working backwards


    ### DD0493 for nref11n_selfshield_z15
    ### first_center = [0.49400806427001953, 0.48881053924560547, 0.50222492218017578]
    ### first_center = [ 0.49400806427, 0.488810539246,  0.50222492218 ]

    ### DD0752 for nref11n_selfshield_z15
    # first_center = [0.493138313293, 0.485363960266, 0.503630638123 ]

    ### DD0999 for nref11n_selfshield_z15
    #first_center = [0.492419242859, 0.482583045959, 0.504755973816]

    ### DD1088 for nref11n_selfshield_z15
    #first_center = [0.492186546326, 0.481648445129, 0.505097389221]

    ### DD1547 for nref11n_selfshield_z15
    # first_center = [0.49123287200927734, 0.4774351119995117, 0.5069074630737305 ]

    ## DD1795
    # first_center = [0.49077320098876953, 0.4754419326782226, 0.5077409744262695]

    ## DD2183
    #first_center = [0.4901914596557618,  0.4727659225463867, 0.5089044570922852]

    ### DD0580 for halo 5016
    #first_center = [0.515818595886, 0.475518226624, 0.497570991516]

    ### DD0984 for halo 5016
    #first_center = [ 0.5232095718383789, 0.46910381317138666,  0.5017900466918945 ]

    ### DD0580 for halo 5036
    #first_center = [0.485018730164, 0.502505302429, 0.503439903259]

    ### DD0580 for halo 2392
    #first_center = [0.497841835022, 0.497172355652, 0.479462623596]

    ### DD0580 for halo 2878
    # first_center = [0.508492469788, 0.509057044983, 0.489006996155]

    ### DD0580 for halo 4123
    # first_center = [0.485667228699, 0.478737831116, 0.485198020935]

    ### DD0689 for halo 5036
    # first_center = [0.4834909439086914, 0.5023832321166992, 0.5033254623413086 ]

    ### DD0770 for halo 5036
    # first_center = [ 0.4823884963989257, 0.5023069381713867,  0.503239631652832 ]

    ### DD0984 for halo 5016
    # first_center = [  0.5240983963012695,  0.4685277938842774, 0.5023050308227539 ]

    ### DD0797 for halo 5036
    #  first_center = [ 0.4820261001586914, 0.5022745132446289, 0.5032148361206055 ]

    ### DD0767 for halo  002392
    # first_center = [ 0.49776554107666016,  0.4940633773803711, 0.47627162933349615]

    ### DD0699 for halo 004123
    #first_center = [0.4843683242797852,  0.4765806198120117,  0.4836854934692382 ]

    ##  DD1175 for halo 005016
    #first_center = [0.5258378982543945 , 0.4669160842895508 ,  0.50400447845459 ]

    ##  DD1359 for halo 005016
    # first_center = [ 0.5284872055053711, 0.46451282501220703, 0.5057573318481445 ]

    ##  DD0850 for halo 005036
    # first_center = [ 0.4813394546508789, 0.5021867752075195, 0.5031805038452148]

    ## DD0779 for halo 4123
    # first_center = [0.48355197906494135,  0.4751977920532226,  0.4827165603637695]

    ## DD0902  for  5036
    # first_center =  [ 0.4807195663452148, 0.5021162033081055, 0.5032072067260742]

    ## DD0806 for  2392
    # first_center = [0.49770450592041016,  0.4933691024780274, 0.47570896148681646]

    ## DD0911 for 5036
    # first_center = [0.4806261062622071, 0.5021028518676758, 0.5032129287719728]

    ##  DD1589 for 5016
    #  first_center = [0.5314531326293944,  0.4612226486206054, 0.5076303482055663]

    ## DD0869 for 4123
    # first_center = [0.48263835906982416,  0.4736795425415039, 0.48167324066162115]

    ## DD0898 for 2392
    # first_center = [0.4975461959838867, 0.49187755584716797,  0.4743967056274414]

    ## DD1818 for 5016
    # first_center = [0.5341157913208008,  0.4584360122680664,  0.509373664855957]

    ## DD1020 for 5036
    # first_center = [0.47943973541259766, 0.5020551681518555, 0.5029783248901367]

    args = parse_args()
    if args.halo == "4123":
        first_center = [0.4731073379516602, 0.4573392868041993, 0.470881462097168 ]
        start_snap = 2200
        end_snap = 2400
    elif args.halo == "2392":
        first_center = [0.49654674530029297, 0.4821596145629883, 0.46587657928466797]
        start_snap = 1600
        end_snap = 1800
    elif args.halo == "2878":
        first_center = [0.5096139907836914, 0.5118303298950195, 0.48661327362060547]
        start_snap = 800
        end_snap = 900
    else:
        sys.exit("halo not found!")


    get_center_track(first_center, end_snap, start_snap, 0.002)
