import numpy as np
import yt
import sys
import os
import argparse

import pickle

from astropy.table import Table
from astropy.io import fits

from get_refine_box import get_refine_box
from get_proper_box_size import get_proper_box_size
from consistency import *


def parse_args():
    '''
    Parse command line arguments.  Returns args object.
    '''
    parser = argparse.ArgumentParser(description="makes a bunch of plots")

    parser.add_argument('--output', metavar='output', type=str, action='store',
                        help='which output? default is RD0020')
    parser.set_defaults(output="RD0020")

    args = parser.parse_args()
    return args


def calc_cddf(output):
    import trident
    # load the simulation
    outputds = output + '/' + output
    forced_ds = yt.load(outputds)

    track_name = "./halo_track"
    output_dir = "/nobackupp2/mpeeples/halo_008508/orig/nref11n_nref10f_orig/ovi_neviii/"
    ## output_dir = "/Users/molly/Dropbox/foggie-collab/plots/halo_008508/natural/nref11/spectra/"
    os.chdir(output_dir)
    track = Table.read(track_name, format='ascii')
    track.sort('col1')
    zsnap = forced_ds.get_parameter('CosmologyCurrentRedshift')

    proper_box_size = get_proper_box_size(forced_ds)
    forced_box, forced_c, width = get_refine_box(forced_ds, zsnap, track)
    width = width * proper_box_size
    print("width = ", width, "kpc")

    resolution = (1048,1048)
    radii = np.zeros(resolution)
    indices = np.indices(resolution)
    for x in range(0, 1048):
        for y in range(0, 1048):
            radii[x,y] = (width / 1048) * np.sqrt((524-x)**2 + (524-y)**2)
    big_radii = np.concatenate((radii, radii, radii), axis=None)
    pkl_name = 'radii_' + output + '_physicalkpc.pkl'
    print("saving to ", pkl_name)
    pickle.dump(big_radii, open( pkl_name, "wb" ) )

    trident.add_ion_fields(forced_ds, ions=['O VI', 'Ne VIII'])

    ions = ['H_p0_number_density', 'O_p5_number_density']
    fields = []
    for ion in ions:
        field = ('gas', ion)
        fields.append(field)

    dp_forced_x = yt.ProjectionPlot(forced_ds, 'x', fields, center=forced_c, \
                        width=(width,"kpc"), data_source=forced_box)
    dp_forced_y = yt.ProjectionPlot(forced_ds, 'y', fields, center=forced_c, \
                        width=(width,"kpc"), data_source=forced_box)
    dp_forced_z = yt.ProjectionPlot(forced_ds, 'z', fields, center=forced_c, \
                        width=(width,"kpc"), data_source=forced_box)


    for ion in ions:
        colr = np.array([])
        print("trying ",ion)

        frb = dp_forced_x.data_source.to_frb((width,'kpc'), resolution)
        forced = np.array(np.log10(frb[ion]))
        colr = np.append(colr, forced.ravel(), axis=None)

        frb = dp_forced_y.data_source.to_frb((width,'kpc'), resolution)
        forced = np.array(np.log10(frb[ion]))
        colr = np.append(colr, forced.ravel(), axis=None)

        frb = dp_forced_z.data_source.to_frb((width,'kpc'), resolution)
        forced = np.array(np.log10(frb[ion]))
        colr = np.append(colr, forced.ravel(), axis=None)

        colr[colr == -np.inf] = 1


        pkl_name = ion + '_nref10f_' + output + '_column_densities.pkl'
        print("saving to ", pkl_name)
        pickle.dump(colr, open( pkl_name, "wb" ) )

if __name__ == "__main__":
    args = parse_args()
    calc_cddf(args.output)
    sys.exit("~~~*~*~*~*~*~all done!!!! yay column densities!")
