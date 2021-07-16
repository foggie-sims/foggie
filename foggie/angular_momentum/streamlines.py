import yt
import numpy as np
from numpy import *
from astropy.table import Table, Column
from foggie.utils.foggie_utils import filter_particles
from foggie.utils.get_run_loc_etc import get_run_loc_etc
from foggie.utils.get_refine_box import get_refine_box
from foggie.utils.get_halo_center import get_halo_center
from foggie.utils.get_proper_box_size import get_proper_box_size
from foggie.utils import yt_fields
import os, sys, argparse
from foggie.utils.consistency import *
import matplotlib.pyplot as plt
from foggie.utils.foggie_load import *
from yt.units import *
from yt.visualization.api import Streamlines
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def parse_args(haloname, DDname):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='''identify satellites in FOGGIE simulations''')
    parser.add_argument('-system', '--system', metavar='system', type=str, action='store', \
                        help='Which system are you on? Default is Jase')
    parser.set_defaults(system="jase")

    parser.add_argument('-do', '--do', metavar='do', type=str, action='store', \
                        help='Which system are you on? Default is Jase')
    parser.set_defaults(system="none")

    parser.add_argument('--run', metavar='run', type=str, action='store',
                        help='which run? default is natural')
    parser.set_defaults(run="nref11c_nref9f")

    parser.add_argument('--halo', metavar='halo', type=str, action='store',
                        help='which halo? default is 8508 (Tempest)')
    parser.set_defaults(halo=haloname.strip('halo_00'))

    parser.add_argument('--pwd', dest='pwd', action='store_true',
                        help='just use the pwd?, default is no')
    parser.set_defaults(pwd=False)

    parser.add_argument('--run_all', dest='run_all', action='store_true',
                        help='just use the pwd?, default is no')
    parser.set_defaults(pwd=False)


    parser.add_argument('--output', metavar='output', type=str, action='store',
                        help='which output? default is RD0020')
    parser.set_defaults(output=DDname)


    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args('', '')
    ds, refine_box = load_sim(args)
    fig_dir = '/Users/rsimons/Desktop'
    

    c   = ds.halo_center_kpc
    pos = c + ds.arr([15, 15, 15], 'kpc')
    xfield = ('gas', 'vx_corrected')
    yfield = ('gas', 'vy_corrected')
    zfield = ('gas', 'vz_corrected')
    print ('hi')
    streamlines = Streamlines(ds, pos, xfield, yfield, zfield, length=20*kpc, dx = 1*kpc, get_magnitude=True)
    print ('hi2')
    streamlines.integrate_through_volume()

    # Create a 3D plot, trace the streamlines through the 3D volume of the plot
    fig=plt.figure()
    ax = Axes3D(fig)
    for stream in streamlines.streamlines:
        stream = stream[np.all(stream != 0.0, axis=1)]
        ax.plot3D(stream[:,0], stream[:,1], stream[:,2], alpha=0.1)

    # Save the plot to disk.
    pl.savefig(fig_dir + '/streamlines.png')





