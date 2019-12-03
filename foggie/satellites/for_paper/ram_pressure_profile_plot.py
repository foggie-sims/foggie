import matplotlib.pyplot as plt
import numpy as np
from numpy import *
from scipy import interpolate
from astropy.io import ascii
plt.ioff()
plt.close('all')

rp_tracks = np.load('/Users/rsimons/Dropbox/foggie/catalogs/sat_track_locations/rp_refinebox.npy', allow_pickle = True)[()]
sat_cat = ascii.read('/Users/rsimons/Dropbox/foggie/catalogs/satellite_properties.cat')



fig = plt.figure(figsize = (12,6))
ax1 =  plt.subplot2grid((3,4), (0,0), colspan = 2)
ax2 =  plt.subplot2grid((3,4), (1,0), colspan = 2)
ax3 =  plt.subplot2grid((3,4), (2,0), colspan = 2)
axes = [ax1, ax2, ax3]
axes2 = plt.subplot2grid((3,4), (0,2), colspan = 4, rowspan = 3)





def add_subplot_axes(ax,rect,axisbg='w'):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)    
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    subax = fig.add_axes([x,y,width,height])
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax


halos = ['8508',
         '2878',
         '2392',
         '5016',
         '5036',
         '4123']

cnt = 0
t_90 = []
examples = [('8508', 'b'), ('5036', 'b'), ('4123', 'd')]
examples = []
#examples = [('5036', 'b')]
#examples = [('8508', 'b'), ('5016', 'e'), ('4123', 'b')]


for h, halo in enumerate(halos):
    sat_cat_halo = sat_cat[sat_cat['halo'] == int(halo)]
    for sat in sat_cat_halo['id']:
        if sat == '0': continue
        rp_track = rp_tracks[halo][sat]
        if np.isnan(rp_track['time_interp'][0]): continue

        time_interp  = rp_track['time_interp'] * 1000.
        rp_interp = rp_track['ram_interp']
        dt        = rp_track['dt'].to('Myr')


        if max(time_interp) < 100.: continue
        rp_interp[np.isnan(rp_interp)] = 0.
        sort_arg = argsort(rp_interp)[::-1]


        sort_rp    = rp_interp[sort_arg]
        sort_time  = time_interp[sort_arg]

        sort_mom = cumsum(sort_rp * dt)
        tot_mom = sort_mom[-1]


        if (halo, sat) in examples:

            ###Example#####
            axes[cnt].plot(time_interp, rp_interp, color = 'black')


            perc_90_arg = argmin(abs(sort_mom - tot_mom * 0.90))

            print (min(time_interp), max(time_interp))

            for i in arange(0, perc_90_arg):

                axes[cnt].fill_between(x = time_interp, \
                                     y1 = rp_interp, 
                                     where = abs(time_interp - sort_time[i]) < 5*dt,
                                     color = 'blue', alpha = 0.1)
            cnt+=1

        t_plot = np.linspace(0, 1, len(sort_mom)) 
        arg_90 = argmin(abs(sort_mom/tot_mom - 0.9))
        t_90.append(t_plot[arg_90])



        axes2.plot(np.linspace(0, 1, len(sort_mom))[:arg_90], sort_mom[:arg_90]/tot_mom, 'b-')
        axes2.plot(np.linspace(0, 1, len(sort_mom))[arg_90:], sort_mom[arg_90:]/tot_mom, '-', color = 'grey')



axes[-2].set_ylabel('ram pressure (dyne cm$^{-2}$)', labelpad = 15)

axes[-1].set_xlabel('time (Myr)')



rect = [0.5,0.18,0.65,0.45]
axes22 = add_subplot_axes(axes2, rect)


axes22.hist(t_90, color = 'blue', bins = np.linspace(0, 1, 15), zorder = 5)
axes22.set_xlabel(r"t$_{90}$/t$_{total}$")
axes22.set_ylabel("")
axes22.set_yticks([])
axes22.set_xlim(0,1)

axes22.axvline(x = 0.9, linestyle = '--', color = 'black')
axes22.axvline(x = median(t_90), linestyle = '-', color = 'grey', zorder =10)
axes22.annotate('constant\nram pressure', (0.78, 0.6), xycoords = 'axes fraction',  rotation = 90, color = 'black')
axes22.annotate('median', (0.18, 0.65), xycoords = 'axes fraction',  rotation = 90, color = 'grey')


axes2.annotate('constant ram pressure', (0.58, 0.58), \
                 ha = 'left', va = 'bottom', rotation = 44, \
                 xycoords = 'axes fraction', fontsize = 20, color = 'black')


axes2.plot([0,0.58], [0.0,0.58], '--', color = 'black')
axes2.plot([0.95, 1.0], [0.95, 1.0], '--', color = 'black')
#axes2.plot([0.4,1.0], [0.4,1.0], '--', color = 'black')
axes2.set_xlim(-0.0,1)
axes2.set_ylim(-0.0,1)
axes2.set_xlabel('cumulative time\n(normalized; sorted by periods of highest to lowest ram pressure)')
axes2.set_ylabel('cumulative surface momentum imparted (normalized)')



#for ax in axes:
#    ax.set_xlim(0, 800)
ax1.set_xticklabels([])
ax2.set_xticklabels([])

#ax2.set_ylim(0, 0.7e-9)
#ax3.set_ylim(0, 0.7e-12)

for e, (halo, sat)in enumerate(examples):
    axes[e].annotate('%s-%s'%(halo, sat), (0.05, 0.9), ha = 'left', va = 'top', xycoords = 'axes fraction', color = 'black', fontsize = 14)

#ax3.annotate('90 percent of total\nmomentum imparted', \
#            (0.5, 0.9), ha = 'left', va = 'top', \
#            xycoords = 'axes fraction', color = 'blue', fontweight = 'bold', fontsize = 20)
fig.tight_layout()
fig.subplots_adjust(hspace = 0.12)
fig.savefig('/Users/rsimons/Dropbox/foggie/figures/for_paper/ram_pressure_cumulative.png', dpi = 350)



