import yt
import numpy as np, os
from foggie.utils.foggie_load import *
import matplotlib.pyplot as plt
from yt.funcs import mylog
from numpy.polynomial import Polynomial
mylog.setLevel(40)
import unyt

# ##### First we open a FOGGIE dataset for Tempest
dataset_name = 'run3/RD0042/RD0042'
trackname = os.getenv('DROPBOX_DIR') + '/FOGGIE/foggie/foggie/halo_tracks/008508/nref11n_selfshield_15/halo_track_200kpc_nref10'
ds, region = foggie_load(dataset_name, trackname) 
ad = ds.all_data() 

# ###### Now let's do a projection plot of the density for reference.
p = yt.ProjectionPlot(ds, 'z', 'density', data_source=region, width=(20, 'kpc'), center=ds.halo_center_code)
p.set_unit('density','Msun/pc**2')
p.set_cmap('density', density_color_map)
p.set_zlim('density',0.01,300)
p.save() 

# ###### Now we can do a projection of the "young stars" (with creation time in the last 3 Myr) for comparison. 
p = yt.ProjectionPlot(ds, 'z', ('deposit', 'young_stars_cic'), width=(20, 'kpc'), data_source=region, center=ds.halo_center_code)
p.set_unit(('deposit','young_stars_cic'),'Msun/kpc**2')
p.set_zlim(('deposit','young_stars_cic'),1000,1000000)
p.save() 

# #### Now, project the gas density to a surface density for comparison to the SFR. This is the x-axis of the KS plot. 
proj_frb = p.data_source.to_frb((100., "kpc"), 500)
projected_density = proj_frb['density'].in_units('Msun/pc**2')
ks_nh1 = proj_frb['H_p0_number_density'].in_units('pc**-2') * yt.YTArray(1.67e-24/1.989e33, 'Msun') 


# #### Now, project the young stars to a surface density for comparison to the gas. This is the y-axis of the KS plot. 
young_stars = proj_frb[('deposit', 'young_stars3_cic')].in_units('Msun/kpc**2')
ks_sfr = young_stars / yt.YTArray(3e6, 'yr') + yt.YTArray(1e-6, 'Msun/kpc**2/yr')

# #### Next, define the KS relation in two vectors that can be interpolated. These are values data-thiefed from the KMT09 plot. 
log_sigma_gas = [0.5278, 0.6571, 0.8165, 1.0151, 1.2034, 1.4506, 1.6286, 1.9399, 2.2663, 2.7905, 3.5817]
log_sigma_sfr = [-5.1072, -4.4546, -3.5572, -2.7926, -2.3442, -2.0185, -1.8253, -1.5406, -1.0927, -0.3801, 0.6579]
c = Polynomial.fit(log_sigma_gas, log_sigma_sfr, deg=5)

plt.plot(log_sigma_gas, log_sigma_sfr, marker='o')
plt.xlabel('log $\Sigma _{g} \,\, (M_{\odot} / pc^2)$')
plt.ylabel('log $\dot{M} _{*} \,\, (M_{\odot} / yr / kpc^2)$')

plt.scatter(0.5, c(0.5), color='red')
plt.scatter(0.8, c(0.8), color='red')
plt.scatter(1, c(1), color='red')
plt.scatter(1.5, c(1.5), color='red')
plt.scatter(2, c(2), color='red')
plt.scatter(3, c(3), color='red')
plt.scatter(3.5, c(3.5), color='red')
plt.xlim(-1, 5)
plt.ylim(-6,3)

# #### We now have all the elements of the KS relation, so make the plot. 
plt.plot(np.log10(ks_nh1), np.log10(ks_sfr), '.', markersize=0.5)
plt.plot(log_sigma_gas, log_sigma_sfr, marker='o', color='red')
plt.xlabel('$\Sigma _{g} \,\, (M_{\odot} / pc^2)$')
plt.ylabel('$\dot{M} _{*} \,\, (M_{\odot} / yr / kpc^2)$')
plt.xlim(-1,5)
plt.title('FOGGIE KS relation - '+dataset_name)
plt.ylim(-6,3)
plt.savefig(dataset_name[-6:]+'_KS_relation')


