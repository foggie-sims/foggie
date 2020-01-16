import numpy as np
import matplotlib.pyplot as plt
import argparse
from astropy.table import Table
from foggie.utils.foggie_utils import filter_particles
from foggie.utils.get_run_loc_etc import get_run_loc_etc
from foggie.utils.get_refine_box import get_refine_box
from foggie.utils.get_halo_center import get_halo_center
from foggie.utils.get_proper_box_size import get_proper_box_size
from foggie.utils.consistency import *
from foggie.utils import yt_fields
import yt
plt.ioff()
plt.close('all')

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='''identify satellites in FOGGIE simulations''')
    parser.add_argument('-system', '--system', metavar='system', type=str, action='store', \
                        help='Which system are you on? Default is Jase')
    parser.set_defaults(system="jase")

    parser.add_argument('--run', metavar='run', type=str, action='store',
                        help='which run? default is natural')
    parser.set_defaults(run="nref11c_nref9f")

    parser.add_argument('--halo', metavar='halo', type=str, action='store',
                        help='which halo? default is 8508 (Tempest)')
    parser.set_defaults(halo="8508")

    parser.add_argument('--pwd', dest='pwd', action='store_true',
                        help='just use the pwd?, default is no')
    parser.set_defaults(pwd=False)

    parser.add_argument('--output', metavar='output', type=str, action='store',
                        help='which output? default is RD0020')
    parser.set_defaults(output="RD0020")


    args = parser.parse_args()
    return args




if __name__ == '__main__':

    args = parse_args()
    
    args.halo = '8508'
    args.output = 'DD0487'






    foggie_dir, output_dir, run_loc, trackname, haloname, spectra_dir, blek = get_run_loc_etc(args)
    run_dir = foggie_dir + run_loc

    ds_loc = run_dir + args.output + "/" + args.output
    ds = yt.load(ds_loc)
    track = Table.read(trackname, format='ascii')
    track.sort('col1')
    zsnap = ds.get_parameter('CosmologyCurrentRedshift')

    refine_box, refine_box_center, x_width = get_refine_box(ds, zsnap, track)
    filter_particles(refine_box)




    fig, axes = plt.subplots(1,2,figsize = (10,5))



    fig.savefig('/Users/rsimons/Dropbox/foggie/figures/for_paper/tempest_2Ddistributions.png', dpi = 300)















