import trident
import numpy as np
import yt
import MISTY
import sys
import os

import matplotlib as mpl
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})
mpl.rcParams['font.family'] = 'stixgeneral'
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import AxesGrid

from astropy.table import Table
from astropy.io import fits

from get_refine_box import get_refine_box
from get_proper_box_size import get_proper_box_size
from consistency import *

def calc_cddf(**kwargs):
    ion = kwargs.get("ion","H I 1216")
    # load the simulation
    forced_ds = yt.load("/Users/molly/foggie/halo_008508/nref11n/nref11n_nref10f_refine200kpc_z4to2/RD0020/RD0020")
    track_name = "/Users/molly/foggie/halo_008508/nref11n/nref11n_nref10f_refine200kpc_z4to2/halo_track"
    output_dir = "/Users/molly/Dropbox/foggie-collab/plots_halo_008508/nref11n/comparisons/"
    natural_ds = yt.load("/Users/molly/foggie/halo_008508/nref11n/natural/RD0020/RD0020")
    ## output_dir = "/Users/molly/Dropbox/foggie-collab/plots/halo_008508/natural/nref11/spectra/"
    os.chdir(output_dir)
    track = Table.read(track_name, format='ascii')
    track.sort('col1')
    zsnap = forced_ds.get_parameter('CosmologyCurrentRedshift')

    proper_box_size = get_proper_box_size(forced_ds)
    forced_box, forced_c, width = get_refine_box(forced_ds, zsnap, track)
    natural_box, natural_c, width = get_refine_box(natural_ds, zsnap, track)
    width = width * proper_box_size

    # forced_box = forced_ds.box([xmin, ymin, zmin], [xmax, ymax, zmax])
    # forced_c = forced_ds.arr(halo_center,'code_length')
    # natural_box = natural_ds.box([xmin, ymin, zmin], [xmax, ymax, zmax])
    # natural_c = natural_ds.arr(halo_center,'code_length')
    # width = (197./forced_ds.hubble_constant)/(1+forced_ds.current_redshift)
    print "width = ", width, "kpc"

    axis = 'z'
    res = [1000,1000]

    trident.add_ion_fields(forced_ds, ions=['O VI'])
    trident.add_ion_fields(natural_ds, ions=['O VI'])
    # trident.add_ion_fields(forced_ds, ions=['C II'])
    # trident.add_ion_fields(natural_ds, ions=['C II'])

    ## start with HI

    ion = 'O_p5_number_density'

    dph_forced = yt.ProjectionPlot(forced_ds,axis,('gas',ion), center=forced_c, \
                            width=(width,"kpc"), data_source=forced_box)
    frb = dph_forced.frb['gas',ion]
    hi_forced = np.array(np.log10(frb))

    dph_natural = yt.ProjectionPlot(natural_ds,axis,('gas',ion), center=natural_c, \
                            width=(width,"kpc"), data_source=natural_box)
    frb = dph_natural.frb['gas',ion]
    hi_natural = np.array(np.log10(frb))

    # plot
    fig, ax = plt.subplots(figsize=(6,6))
    ax.hist(hi_forced.ravel(), cumulative=True, normed=True, bins=500, \
                    histtype='step', range=[12,19], lw=2, label="forced")
    ax.hist(hi_natural.ravel(), cumulative=True, normed=True, bins=500, \
                    histtype='step', range=[12,19],lw=2, label="natural")
    plt.legend()
    plt.xlabel('log OVI')
    plt.ylabel('cumulative fraction')
    plt.savefig('ovi_cddf_compare.png')



if __name__ == "__main__":
    calc_cddf()
    sys.exit("~~~*~*~*~*~*~all done!!!! yay column densities!")
