import yt
from yt.analysis_modules.star_analysis.api import StarFormationRate

import numpy as np

from astropy.table import Table
import astropy.units as u

from consistency import *
from get_halo_center import get_halo_center
from utils.get_proper_box_size import get_proper_box_size
from modular_plots import get_refine_box

import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})
import matplotlib as mpl
mpl.rcParams['font.family'] = 'stixgeneral'
mpl.rcParams['font.size'] = 18.
import matplotlib.pyplot as plt

ref_color = 'darkorange' ###  '#4575b4' # purple
nat_color = '#4daf4a' # green
cool_color = '#4575b4' # blue

dsn = yt.load('/astro/simulations/FOGGIE/halo_008508/nref11n_selfshield_z15/natural/RD0018/RD0018')
#dsv2 = yt.load('/astro/simulations/FOGGIE/halo_008508/nref11n_selfshield_z15/natural/RD0028/RD0028')
dsv3 = yt.load('/astro/simulations/FOGGIE/halo_008508/nref11n_selfshield_z15/natural/RD0018/RD0018')
dsv4 = yt.load('/astro/simulations/FOGGIE/halo_008508/nref11n_selfshield_z15/natural/RD0018/RD0018')
dsr = yt.load('/astro/simulations/FOGGIE/halo_008508/nref11n_selfshield_z15/nref11n_nref10f_selfshield_z6/RD0018/RD0018')
dsc = yt.load('/astro/simulations/FOGGIE/halo_008508/nref11n_selfshield_z15/nref11c_nref9f_selfshield_z6/RD0018/RD0018')


track_name = "/astro/simulations/FOGGIE/halo_008508/nref11n_selfshield_z15/nref11n_nref10f_selfshield_z6/halo_track"
#track_name = "/astro/simulations/FOGGIE/halo_008508/nref11n/nref11n_nref10f_refine200kpc/halo_track"
track = Table.read(track_name, format='ascii')
track.sort('col1')
width = 15. #kpc

proper_box_size = get_proper_box_size(dsr)
refine_box, refine_box_center, refine_width = get_refine_box(dsr, dsr.current_redshift, track)
centerr, velocity = get_halo_center(dsr, refine_box_center)
refine_width = refine_width * proper_box_size
width_code = width / proper_box_size ## needs to be in code units
boxr = dsr.r[centerr[0] - 0.5*width_code : centerr[0] + 0.5*width_code, \
           centerr[1] - 0.5*width_code : centerr[1] + 0.5*width_code, \
           centerr[2] - 0.5*width_code : centerr[2] + 0.5*width_code]

proper_box_size = get_proper_box_size(dsn)
refine_box_natural, refine_box_center, refine_width = get_refine_box(dsn, dsn.current_redshift, track)
centern, velocity = get_halo_center(dsn, refine_box_center)
width_code = width / proper_box_size ## needs to be in code units
boxn = dsn.r[centern[0] - 0.5*width_code : centern[0] + 0.5*width_code, \
           centern[1] - 0.5*width_code : centern[1] + 0.5*width_code, \
           centern[2] - 0.5*width_code : centern[2] + 0.5*width_code]

# proper_box_size = get_proper_box_size(dsc)
# refine_box_cooling, refine_box_center, refine_width = get_refine_box(dsc, dsc.current_redshift, track)
# centerc, velocity = get_halo_center(dsc, refine_box_center)
# width_code = width / proper_box_size ## needs to be in code units
# boxc = dsc.r[centerc[0] - 0.5*width_code : centerc[0] + 0.5*width_code, \
#            centerc[1] - 0.5*width_code : centerc[1] + 0.5*width_code, \
#            centerc[2] - 0.5*width_code : centerc[2] + 0.5*width_code]


proper_box_size = get_proper_box_size(dsv3)
refine_box_v3, refine_box_center, refine_width = get_refine_box(dsn, dsv3.current_redshift, track)
centerv3, velocity = get_halo_center(dsn, refine_box_center)
width_code = width / proper_box_size ## needs to be in code units
boxv3 = dsn.r[centerv3[0] - 0.5*width_code : centerv3[0] + 0.5*width_code, \
           centerv3[1] - 0.5*width_code : centerv3[1] + 0.5*width_code, \
           centerv3[2] - 0.5*width_code : centerv3[2] + 0.5*width_code]

proper_box_size = get_proper_box_size(dsv4)
refine_box_v4, refine_box_center, refine_width = get_refine_box(dsn, dsv4.current_redshift, track)
centerv4, velocity = get_halo_center(dsn, refine_box_center)
width_code = width / proper_box_size ## needs to be in code units
boxn = dsn.r[centerv4[0] - 0.5*width_code : centerv4[0] + 0.5*width_code, \
           centerv4[1] - 0.5*width_code : centerv4[1] + 0.5*width_code, \
           centerv4[2] - 0.5*width_code : centerv4[2] + 0.5*width_code]


## star formation
spr = dsr.sphere(centerr,(50.,'kpc'))
spn = dsn.sphere(centern,(50.,'kpc'))
#spc = dsr.sphere(centerc,(50.,'kpc'))
spv3 = dsn.sphere(centerv3,(50.,'kpc'))
spv4 = dsn.sphere(centerv4,(50.,'kpc'))

sfrr = StarFormationRate(dsr, data_source=spr)
sfrn = StarFormationRate(dsn, data_source=spn)
#sfrc = StarFormationRate(dsc, data_source=spc)
sfrv3 = StarFormationRate(dsv3, data_source=spv3)
sfrv4 = StarFormationRate(dsv4, data_source=spv4)

fig = plt.figure(figsize=(6,6))
plt.plot(sfrr.time.to('Gyr'),sfrr.Msol_yr, lw=2, color=ref_color, label="nref10f")
#plt.plot(sfrc.time.to('Gyr'),sfrc.Msol_yr, lw=2, color=cool_color, label="nref11c")
plt.plot(sfrn.time.to('Gyr'),sfrn.Msol_yr, lw=2, color=nat_color, label="natural")
plt.plot(sfrv3.time.to('Gyr'),sfrv3.Msol_yr, lw=2, color='black', ls=':',label="v3")
plt.plot(sfrv4.time.to('Gyr'),sfrv4.Msol_yr, lw=2, color='#565656', ls='--',label="v4")
plt.legend(loc='upper left')
plt.xlabel('time [Gyr]')
plt.ylabel('star formation rate')
#plt.xlim(1,13.7)
plt.tight_layout()
plt.savefig('/Users/molly/Dropbox/foggie-collab/plots_halo_008508/nref11n_selfshield_z15/comparisons/sfh_t.png')


fig = plt.figure(figsize=(6,6))
plt.plot(sfrr.redshift,sfrr.Msol_cumulative, lw=2, color=ref_color, label="nref10f")
#plt.plot(sfrc.redshift,sfrc.Msol_cumulative, lw=2, color=cool_color, label="nref11c")
plt.plot(sfrn.redshift,sfrn.Msol_cumulative, lw=2, color=nat_color, label="natural")
plt.plot(sfrv3.redshift,sfrv3.Msol_cumulative, lw=2, color='black', ls=':',label="v3")
plt.plot(sfrv4.redshift,sfrv4.Msol_cumulative, lw=2, color='#565656', ls='--',label="v4")
plt.legend(loc='upper right')
plt.xlabel('redshift')
plt.ylabel('total stellar mass')
plt.xlim(2.45,5)
plt.tight_layout()
plt.savefig('/Users/molly/Dropbox/foggie-collab/plots_halo_008508/nref11n_selfshield_z15/comparisons/sfh_cumul_z.png')
