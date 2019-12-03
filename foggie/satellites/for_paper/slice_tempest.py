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
import PIL

plt.rcParams['ytick.minor.size'] = 3.
plt.rcParams['ytick.major.size'] = 5.
plt.rcParams['xtick.minor.size'] = 3.
plt.rcParams['xtick.major.size'] = 5.

plt.rcParams['xtick.labelsize'] = 12.
plt.rcParams['ytick.labelsize'] = 12.

plt.ioff()
plt.close('all')


fig_dir = '/Users/rsimons/Dropbox/foggie/figures/for_paper'

def parse_args(haloname, DDname):
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


haloname = 'halo_008508'
DDname = 'DD0487'
name = 'Tempest'
simname = 'nref11c_nref9f'


center_dic =  np.load('/Users/rsimons/Desktop/foggie/outputs/centers/%s_nref11c_nref9f_%s.npy'%(haloname,DDname), allow_pickle = True)[()]
flname = '/Users/rsimons/Desktop/foggie/sims/%s/%s/%s/%s'%(haloname, simname, DDname, DDname)
ds = yt.load(flname)
center = ds.arr(center_dic, 'code_length').to('kpc')

box = ds.r[center[0] - ds.arr(100, 'kpc'): center[0] + ds.arr(100, 'kpc'),
           center[1] - ds.arr(100, 'kpc'): center[1] + ds.arr(100, 'kpc'),
           center[2] - ds.arr(100, 'kpc'): center[2] + ds.arr(100, 'kpc')]

cen_bulkv = box.quantities.bulk_velocity().to('km/s') 
box.set_field_parameter('bulk_velocity', cen_bulkv)


do_slice = True
axs = 'x'

if do_slice:
    field = ('gas', 'density')
    cmap = density_color_map
    cmap.set_bad('k')


    slc = yt.SlicePlot(ds, axs, field, center = center, data_source = box, width = (200, 'kpc'))
    unit = 'Msun/pc**3'
    zmin = density_slc_min * 1.e-1
    zmax = density_slc_max
    slc.set_unit(field, unit)
    slc.set_zlim(field, zmin = zmin, zmax =  zmax)
    slc.set_cmap(field, cmap)


    slc.annotate_scale(size_bar_args={'color':'white'})
    slc.set_colorbar_label(field = field, label = r'gas density (M$_{\odot}$ pc$^{-3}$)') 
    slc.hide_axes()
    slc.annotate_velocity(factor=20)


ray_l = 100.

ray_s = 10.


thetas = [90 * pi/180., 90 * pi/180., 90 * pi/180., 90 * pi/180.]
phis   = [60. * pi/180., 240 * pi/180., 160. * pi/180., 340 * pi/180.]


print ('adding arrows')





vesc = np.load('/Users/rsimons/Desktop/foggie/outputs/vesc_profile/%s_vmax.npy'%(haloname), allow_pickle = True)[()]
dinner_start = ray_l
dinner = yt.YTArray(dinner_start, 'kpc')
dt = yt.YTArray(1.e6, 'yr')


fig, ax = plt.subplots(1,1, figsize = (6.5,5.5))
clrs = ['blue', 'darkblue', 'lightblue']
clrs = [ 'firebrick', 'red', 'darkblue', 'blue']

for i, (theta, phi) in enumerate(zip(thetas, phis)):

    end = ds.arr([center[0].value + ray_l * np.cos(theta)*np.sin(phi), \
                  center[1].value + ray_l * np.sin(theta)*np.sin(phi), \
                  center[2].value + ray_l * np.cos(phi)               ], 'kpc')

    start = ds.arr([center[0].value + ray_s * np.cos(theta)*np.sin(phi), \
                    center[1].value + ray_s * np.sin(theta)*np.sin(phi), \
                    center[2].value + ray_s * np.cos(phi)               ], 'kpc')

    ray = ds.r[start:end]

    ray.set_field_parameter('center', center)
    ray.set_field_parameter('bulk_velocity', cen_bulkv)
    dist_r = ray['index', 'radius'].to('kpc')
    vel_r  = ray['gas', 'radial_velocity'].to('km/s')

    dinner = yt.YTArray(dinner_start, 'kpc')
    dt = yt.YTArray(1.e6, 'yr')

    M, t = 0, 0
    tot_Ms, P_all, ts, dmids = [], [], [], []

    vel_r = ray['gas', 'radial_velocity'].to('km/s')
    dist_r = ray['index', 'radius'].to('kpc')

    dens_r = ray['gas', 'density'].to('g/cm**3')



    while True:
        douter = dinner

        vmax_interp = yt.YTArray(np.interp(douter, vesc['r_%s'%simname], vesc['vesc_%s'%simname]), 'km/s')
        dinner = douter - (vmax_interp * dt.to('s')).to('kpc')
        if dinner < 9 : break
        gd = argmin(abs(dist_r - (dinner + douter)/2.))

        dens = dens_r[gd]
        dvel = yt.YTArray(min([(vel_r[gd] - yt.YTArray(vmax_interp, 'km/s')).value, 0]), 'km/s')
        P = (dens * dvel**2.).to('dyne * cm**-2')
        M += (P * dt).to('Msun * km/s * 1/kpc**2')
        tot_Ms.append(M.value)
        P_all.append(float(P.value))
        dmids.append((dinner + douter).value/2.)
        ts.append(t * dt.to('Myr').value)
        t+=1


    ax.plot(dmids, P_all, 'k', linestyle = '-', color = clrs[i])
    ax.set_xlim(100, 0)
    
    ax.axvspan(xmin = 0 , xmax = 10, color = 'grey', alpha = 1.0, zorder = 10)


    ax.set_yscale('symlog', linthreshy=1.e-16)
    #ax.set_ylim(-1.e-18, 5.e-12)
    #ax.set_yticks([0., 1.e-17, 1.e-16, 1.e-15, 1.e-14, 1.e-13, 1.e-12])
    ax.set_ylim(-1.e-17, 3.e-11)
    ax.set_yticks([0., 1.e-16, 1.e-15, 1.e-14, 1.e-13, 1.e-12, 1.e-11])


    minorticks_y = array([])
    minorticks_y = np.concatenate((minorticks_y, np.arange(0, 1.e-16, 0.1e-16)))

    for j in arange(-15, -10, 1):
        minorticks_y = np.concatenate((minorticks_y, np.arange(10.**(j-1), 10.**(j), 10.**(j - 1))))

    ax.set_yticks(minorticks_y, minor = True)


    ax.set_xlabel(r'Distance from Central Galaxy (kpc)')
    ax.set_ylabel(r'Surface Ram Pressure (dyne cm$^{-2}$)')

    if do_slice:
        slc.annotate_arrow(pos = start, starting_pos = end, coord_system = 'data', plot_args = {'color': clrs[i], 'linewidth': 5})




extra = 0
for i in arange(4):
    if i == 2: extra = 0.35
    ax.plot([95, 85], [10**(-11. - 0.15 * i - extra), 10**(-11. - 0.15 * i - extra)], '-', linewidth = 4, color = clrs[i])

ax.annotate("against outflow", (82, 10**-11.075), ha = 'left', va = 'center', color = 'black', fontweight = 'bold', fontsize = 24)
ax.annotate("along inflow", (82, 10**-11.725), ha = 'left', va = 'center', color = 'black', fontweight = 'bold', fontsize = 24)



fig.tight_layout()
fig.subplots_adjust(top = 0.962)
fig.savefig(fig_dir + '/example_trajectories.png', dpi = 400)












print ('done')





if do_slice:
    slc.annotate_sphere(center, radius = (100, 'kpc'), coord_system='data', circle_args={'color':'white'})                     
    slc.annotate_sphere(center, radius = (10, 'kpc'), coord_system='data', circle_args={'color':'white'})                        
    slc.save(fig_dir + '/%s_%s_Tempest_ramslice.png'%(haloname, axs), mpl_kwargs = {'dpi': 500})







fls = [fig_dir + '/example_trajectories.png',\
       fig_dir + '/%s_%s_Tempest_ramslice.png'%(haloname, axs),]

imgs = [PIL.Image.open(fl) for fl in fls]



min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
imgs_comb = np.hstack( (np.asarray( i.resize(min_shape) ) for i in imgs) )

imgs_comb = PIL.Image.fromarray( imgs_comb)
imgs_comb.save(fig_dir + '/example_ram_4trajectories.png')    


























