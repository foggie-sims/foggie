import yt
from yt.analysis_modules.halo_analysis.api import HaloCatalog
import matplotlib.pyplot as plt
import numpy as np

from get_proper_box_size import get_proper_box_size

def plot_cell_vs_radius(ds, r, cell_size):

    proper_box_size = get_proper_box_size(ds)
    temp = np.array(ds["temperature"])
    cell_vol = ds["cell_volume"]
    cell_size = np.array(cell_vol)**(1./3.)*proper_box_size

    # plot the cell-size vs. radius, broken down by phase
    plt.semilogy(r[temp > 1e6], 1.2*cell_size[temp > 1e6], '.', color='yellow')
    plt.semilogy(r[(temp > 1e5) & (temp < 1e6)], 1.1*cell_size[(temp > 1e5) & (temp < 1e6)], '.', color='#4daf4a')
    plt.semilogy(r[(temp > 1e4) & (temp < 1e5)], 1.0*cell_size[(temp > 1e4) & (temp < 1e5)], '.', color='#984ea3')
    plt.semilogy(r[temp < 1e4], 0.9*cell_size[temp < 1e4], '.', color='salmon')

    plt.xlim((0, 400))
    plt.ylim((0.1, 50))
    plt.ylabel('Cell Size [kpc]')
    plt.xlabel('Radius [kpc]')
    plt.title('Maximum Refinement = '+str(ds.get_parameter('MaximumRefinementLevel')))
    plt.savefig('cell_size_by_phase_nref'+str(maxrefine)+'.png')
    plt.close()
