

import yt
from yt.analysis_modules.halo_analysis.api import HaloCatalog
import matplotlib.pyplot as plt
import numpy as np

from get_proper_box_size import get_proper_box_size

# open the enzo output
ds = yt.load('DD0127/DD0127')
ad = ds.sphere([0.48984, 0.47133, 0.50956], (1000, 'kpc'))

# extract cell-by-cell physical info
proper_box_size = get_proper_box_size(ds)
cell_vol = ad["cell_volume"]
cell_size = np.array(cell_vol)**(1./3.)*proper_box_size
x = np.array(ad["x"])
y = np.array(ad["y"])
z = np.array(ad["z"])
temp = np.array(ad["temperature"])
r = ((x-0.48984)**2 + (y-0.47133)**2 + (z-0.50956)**2)**0.5 * proper_box_size



plt.semilogy(r[temp > 1e6], 1.2*cell_size[temp > 1e6], '.', color='yellow')
plt.semilogy(r[(temp > 1e5) & (temp < 1e6)], 1.1*cell_size[(temp > 1e5) & (temp < 1e6)], '.', color='#4daf4a')
plt.semilogy(r[(temp > 1e4) & (temp < 1e5)], 1.0*cell_size[(temp > 1e4) & (temp < 1e5)], '.', color='#984ea3')
plt.semilogy(r[temp < 1e4], 0.9*cell_size[temp < 1e4], '.', color='salmon')


plt.xlim((0, 400))
plt.ylim((0.1, 50))
plt.ylabel('Cell Size [kpc]')
plt.xlabel('Radius [kpc]')
plt.title('Maximum Refinement = '+str(ds.get_parameter('MaximumRefinementLevel')))
plt.savefig('cell_size_by_phase.png')
plt.close()







#hc = HaloCatalog(data_ds=ds, finder_method='hop')
#hc.create()
prj = yt.ProjectionPlot(ds, 'x', 'density', center=[0.48984, 0.47133, 0.50956], width=(80,'kpc'))
#prj.annotate_halos(hc)
prj.annotate_grids()
prj.save()

prj = yt.ProjectionPlot(ds, 'y', 'density', center=[0.48984, 0.47133, 0.50956], width=(80,'kpc'))
#prj.annotate_halos(hc)
prj.annotate_grids()
prj.save()

prj = yt.ProjectionPlot(ds, 'z', 'density', center=[0.48984, 0.47133, 0.50956], width=(80,'kpc'))
#prj.annotate_halos(hc)
prj.annotate_grids()
prj.save()



hot_ad = ad.cut_region(["(obj['temperature'] > 1e6)"])
warm_ad =ad.cut_region(["(obj['temperature'] < 1e6) & (obj['temperature'] > 1e5)"])
cool_ad =ad.cut_region(["(obj['temperature'] < 1e5) & (obj['temperature'] > 1e4)"])
cold_ad =ad.cut_region(["(obj['temperature'] < 1e4) & (obj['temperature'] > 1e2)"])


# density profiles
hot_plot = yt.ProfilePlot(hot_ad, "radius", "density", weight_field="cell_mass")
profile = hot_plot.profiles[0]
hot_dens = profile['density']
warm_plot = yt.ProfilePlot(warm_ad, "radius", "density", weight_field="cell_mass")
profile = hot_plot.profiles[0]
warm_dens = profile['density']
cool_plot = yt.ProfilePlot(cool_ad, "radius", "density", weight_field="cell_mass")
profile = hot_plot.profiles[0]
cool_dens = profile['density']
cold_plot = yt.ProfilePlot(cold_ad, "radius", "density", weight_field="cell_mass")
profile = hot_plot.profiles[0]
cold_dens = profile['density']




hot_plot.set_unit('radius', 'kpc').save('dens_hot.png')
warm_plot.set_unit('radius', 'kpc').save('dens_warm.png')
cool_plot.set_unit('radius', 'kpc').save('dens_cool.png')
cold_plot.set_unit('radius', 'kpc').save('dens_cold.png')


# pressure profiles
hot_plot = yt.ProfilePlot(hot_ad, "radius", "pressure", weight_field="cell_mass")
warm_plot = yt.ProfilePlot(warm_ad, "radius", "pressure", weight_field="cell_mass")
cool_plot = yt.ProfilePlot(cool_ad, "radius", "pressure", weight_field="cell_mass")
cold_plot = yt.ProfilePlot(cold_ad, "radius", "pressure", weight_field="cell_mass")

hot_plot.set_unit('radius', 'kpc').save('pres_hot.png')
warm_plot.set_unit('radius', 'kpc').save('pres_warm.png')
cool_plot.set_unit('radius', 'kpc').save('pres_cool.png')
cold_plot.set_unit('radius', 'kpc').save('pres_cold.png')


# entropy profiles
hot_plot = yt.ProfilePlot(hot_ad, "radius", "entropy", weight_field="cell_mass")
warm_plot = yt.ProfilePlot(warm_ad, "radius", "entropy", weight_field="cell_mass")
cool_plot = yt.ProfilePlot(cool_ad, "radius", "entropy", weight_field="cell_mass")
cold_plot = yt.ProfilePlot(cold_ad, "radius", "entropy", weight_field="cell_mass")

hot_plot.set_unit('radius', 'kpc').save('ent_hot.png')
warm_plot.set_unit('radius', 'kpc').save('ent_warm.png')
cool_plot.set_unit('radius', 'kpc').save('ent_cool.png')
cold_plot.set_unit('radius', 'kpc').save('ent_cold.png')
