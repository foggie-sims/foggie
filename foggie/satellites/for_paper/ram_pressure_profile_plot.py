import matplotlib.pyplot as plt
import numpy as np
from numpy import *
from scipy import interpolate
from astropy.io import ascii
import yt
import astropy.units as u
plt.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
plt.ioff()
plt.close('all')

rp_tracks = np.load('/Users/rsimons/Dropbox/foggie/catalogs/sat_track_locations/rp_refinebox_new.npy', allow_pickle = True)[()]
sat_cat = ascii.read('/Users/rsimons/Dropbox/foggie/catalogs/satellite_properties.cat')



fig = plt.figure(figsize = (10,5.2))
ax1 =  plt.subplot2grid((3,4), (0,0), colspan = 2)
ax2 =  plt.subplot2grid((3,4), (1,0), colspan = 2)
ax3 =  plt.subplot2grid((3,4), (2,0), colspan = 2)
axes = [ax1, ax2, ax3]
axes22 = plt.subplot2grid((3,4), (0,2), colspan = 4, rowspan = 3)

fig2, axes2 = plt.subplots(1,1, figsize = (8, 8))




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
    height *= rect[3]
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
t_90_ifaverage = []
examples = [('8508', 'b'), ('5036', 'b'), ('4123', 'd')]
#examples = []
flag = False
#halos = ['8508', '5036', '4123']

for h, halo in enumerate(halos):
    fig3, ax_check = plt.subplots(1,1, figsize = (8, 8))    
    sat_cat_halo = sat_cat[sat_cat['halo'] == int(halo)]

    average = np.load('/Users/rsimons/Dropbox/foggie/outputs/plunge_tunnels/halo_00%s_simulated_plunge.npy'%halo, allow_pickle =True)[()]['nref11c_nref9f']['average']                                                        
    average_dist_r = average['dmids']
    average_P = average['P']

    (m, b), v = np.polyfit(log10(average_dist_r), np.log10(average_P), deg = 1, cov = True)

    fit_x = log10(np.linspace(1, 100, 1000))
    fit_y = m*fit_x + b#*fit_x + bb
    ax_check.plot(average_dist_r, average_P, '--', linewidth = 2.0, color = 'blue', label = 'actual')
    ax_check.plot(10**fit_x, 10**fit_y, '--', linewidth = 2.0, color = 'darkblue', label = 'fit')

    ax_check.set_yscale('log')

    ax_check.legend()

    if flag:  interp_average = interpolate.interp1d(average_dist_r, average_P, bounds_error = False, fill_value = average_P[-1])
    else:     interp_average = interpolate.interp1d(10**fit_x, fit_y, bounds_error = False, fill_value = fit_y[0])

    for sat in sat_cat_halo['id']:
        #if (halo, sat) not in examples: continue

        

        if sat == '0': continue
        rp_track = rp_tracks[halo][sat]
        if np.isnan(rp_track['time_interp'][0]): continue

        time_interp  = rp_track['time_interp'] * 1000.
        rp_interp = rp_track['ram_interp'].value# * u.dyne/(u.cm**2.) 
        dt        = rp_track['dt'].to('Myr').value#*u.Myr

        dist_interp = rp_track['dist_interp']


        if flag: rp_interp_ifaverage = np.array([float(interp_average(d)) for d in dist_interp])
        else:    rp_interp_ifaverage = np.array([10**float(interp_average(d)) for d in dist_interp])


        if max(time_interp) < 100.: continue
        rp_interp[np.isnan(rp_interp)] = 0.
        sort_arg = argsort(rp_interp)[::-1]
        sort_rp    = rp_interp[sort_arg]
        sort_time  = time_interp[sort_arg]
        sort_mom = cumsum(sort_rp* dt)
        tot_mom = sort_mom[-1]



        rp_interp_ifaverage[np.isnan(rp_interp_ifaverage)] = 0.
        sort_arg_ifaverage = argsort(rp_interp_ifaverage)[::-1]
        sort_rp_ifaverage    = rp_interp_ifaverage[sort_arg_ifaverage]
        sort_time_ifaverage  = time_interp[sort_arg_ifaverage]
        sort_mom_ifaverage = cumsum(sort_rp_ifaverage * dt)
        tot_mom_ifaverage = sort_mom_ifaverage[-1]


        if (halo, sat) in examples:
            ###Example#####
            axes[cnt].plot(time_interp, rp_interp/(1.e-11), color = 'black', zorder = 10)
            axes[cnt].plot(time_interp, rp_interp_ifaverage/(1.e-11), color = 'red', zorder = 10)
            print (rp_interp_ifaverage[argmin(dist_interp)]/(1.e-11))

            perc_90_arg = argmin(abs(sort_mom - tot_mom * 0.90))

            print (min(time_interp), max(time_interp))

            if halo == '8508':
                gd = argmin(dist_interp)
                alp = 1.0
                zo = 20
                axes[cnt].axvline(x = time_interp[gd], color = 'grey', alpha = alp, zorder = zo)
                axes[cnt].annotate('periapsis', (time_interp[gd]*1.02, 0.65), rotation = -90, ha = 'left', va = 'top',  color = 'grey', fontsize = 10)

            if halo == '5036':
                gd = argmin(dist_interp)
                axes[cnt].axvline(x = time_interp[gd], color = 'grey', alpha = alp, zorder = zo)
            if halo == '4123':
                gd1 = argmin(rp_track['dist_interp'][rp_track['time_interp'] < 0.3])
                gd2 = argmin(rp_track['dist_interp'][(rp_track['time_interp'] < 0.6) & (rp_track['time_interp'] >0.4)])

                axes[cnt].axvline(x = time_interp[rp_track['time_interp'] < 0.3][gd1], color = 'grey', alpha = alp, zorder = zo)
                axes[cnt].axvline(x = time_interp[(rp_track['time_interp'] < 0.6) & (rp_track['time_interp'] >0.4)][gd2], color = 'grey', alpha = alp, zorder = zo)


            for i in arange(0, perc_90_arg):

                axes[cnt].fill_between(x = time_interp, \
                                       y1 = rp_interp/(1.e-11), 
                                      where = abs(time_interp - sort_time[i]) < 5*dt,
                                       color = 'blue', alpha = 0.1)

            cnt+=1  



        t_plot = np.linspace(0, 1, len(sort_mom)) 
        arg_90 = argmin(abs(sort_mom/tot_mom - 0.9))

        t_90.append(t_plot[arg_90])

        axes2.plot(np.linspace(0, 1, len(sort_mom))[:arg_90], sort_mom[:arg_90]/tot_mom, 'b-')
        axes2.plot(np.linspace(0, 1, len(sort_mom))[arg_90:], sort_mom[arg_90:]/tot_mom, '-', color = 'grey')
        arg_90_ifaverage = argmin(abs(sort_mom_ifaverage/tot_mom_ifaverage - 0.9))
        axes2.plot(np.linspace(0, 1, len(sort_mom_ifaverage))[:arg_90_ifaverage], sort_mom_ifaverage[:arg_90_ifaverage]/tot_mom_ifaverage, 'r-', alpha = 0.4)
        axes2.plot(np.linspace(0, 1, len(sort_mom_ifaverage))[arg_90_ifaverage:], sort_mom_ifaverage[arg_90_ifaverage:]/tot_mom_ifaverage, '-', color = 'grey')

        t_90_ifaverage.append(t_plot[arg_90_ifaverage])


    fig3.savefig('/Users/rsimons/Dropbox/foggie/figures/average_P/%s_averageP.png'%halo)



axes[-2].set_ylabel('ram pressure (10$^{-11}$ dyne cm$^{-2}$)', labelpad = 20)

axes[-1].set_xlabel('time (Myr)', fontsize = 18)



#rect = [0.46,0.18,0.65,0.45]
#axes22 = add_subplot_axes(axes2, rect)


bns = np.linspace(0, 1., 16)
counts1, edges1, bars1 = axes22.hist(t_90, color = 'blue', bins = bns, zorder = 5,  linewidth = 3)
counts2, edges2, bars2 = axes22.hist(t_90_ifaverage, color = 'red', bins = bns, zorder = 5, alpha = 1.0, linewidth = 3)



#axes22.hist(t_90, color = 'blue', bins = bns, zorder = 5,  ec = 'k', histtype = 'step',ls = 'solid', linewidth = 1)
#axes22.hist(t_90_ifaverage, color = 'red', bins = bns, zorder = 5,ec = 'k',histtype = 'step', ls = 'solid', alpha = 1.0, linewidth = 1)


'''
# set the z-order of each bar according to its relative height
x2_bigger = counts2 > counts1
for b1, b2, oo in zip(bars1, bars2, x2_bigger):
    if oo:
        # if bar 2 is taller than bar 1, place it behind bar 1
        b2.set_zorder(b1.get_zorder() - 1)
    else:
        # otherwise place it in front
        b2.set_zorder(b1.get_zorder() + 1)
'''





axes22.set_xlabel(r"$\Delta$t$_{90}$/t$_{total}$", fontsize = 18)
axes22.set_ylabel("number of satellites", fontsize = 18)
#axes22.set_yticks([])
axes22.set_xlim(0,1)

axes22.axvline(x = 0.9, ymin = 0.0, ymax = 1.0, linestyle = '--', zorder = 10, color = 'black')
#axes22.axvline(x = median(t_90), linestyle = '-', color = 'grey', zorder =10)
axes22.annotate('constant\nram pressure', (0.88, 0.96), ha = 'right', va = 'top', xycoords = 'axes fraction', color = 'black', fontsize = 18)
#axes22.annotate('median', (0.18, 0.65), xycoords = 'axes fraction',  rotation = 90, color = 'grey')


axes22.annotate('true\nsimulated\nCGM', (0.18, 0.90), ha = 'left', va = 'top', xycoords = 'axes fraction', color = 'blue', fontsize = 18)
axes22.annotate('spherically-\naveraged\nsimulated\nCGM', (0.63, 0.6), ha = 'right', va = 'top', xycoords = 'axes fraction', color = 'red', fontsize = 18)




axes2.annotate('constant ram pressure', (0.60, 0.58), \
                 ha = 'left', va = 'bottom', rotation = 44, \
                 xycoords = 'axes fraction', fontsize = 15, color = 'black')


axes2.axhspan(ymin = 0.9, ymax = 1.0, xmin = 0, xmax = 1.0, color = 'white')
#axes2.plot([0,0.58], [0.0,0.58], '--', color = 'black')
#axes2.plot([0.95, 1.0], [0.95, 1.0], '--', color = 'black')

axes2.plot([0,0.9], [0.0,0.9], '--', color = 'black')

#axes2.plot([0.4,1.0], [0.4,1.0], '--', color = 'black')
axes2.set_xlim(-0.0,1)
axes2.set_ylim(-0.0,1)
axes2.set_xlabel('cumulative time\n(normalized; sorted by periods of highest to lowest ram pressure)')
axes2.set_ylabel('cumulative surface momentum imparted (normalized)')

axes22.set_ylim(0, 14)

#for ax in axes:
#    ax.set_xlim(0, 800)
ax1.set_xticklabels([])
ax2.set_xticklabels([])

#ax2.set_ylim(0, 0.7e-9)
#ax3.set_ylim(0, 0.7e-12)

for e, (halo, sat)in enumerate(examples):
    if halo == '8508': haloname = 'Tempest'
    elif halo == '5036': haloname = 'Maelstrom'
    elif halo == '4123': haloname = 'Blizzard'




    axes[e].annotate('%s-%s'%(haloname, sat), (0.05, 0.9), ha = 'left', va = 'top', xycoords = 'axes fraction', color = 'black', fontsize = 14)

#ax3.annotate('90 percent of total\nmomentum imparted', \
#            (0.5, 0.9), ha = 'left', va = 'top', \
#            xycoords = 'axes fraction', color = 'blue', fontweight = 'bold', fontsize = 20)
fig.tight_layout()
fig.subplots_adjust(hspace = 0.12, right = 0.98)

axes2.axhspan(ymin = 0.9, ymax = 1.0, color = 'white', zorder = 3)
axes2.axhline(y = 0.9, color = 'black', zorder = 10)
axes2.annotate('accumulated 90\% of total momentum', (0.1, 0.92), zorder = 5, xycoords = 'data', ha = 'left', va = 'bottom', color = 'black')


x = np.linspace(550, 650, 1000)
from astropy.convolution import convolve_fft, Gaussian1DKernel
y = zeros(len(x))
y[300] = 1.
y[600] = 0.2
y[800] = 0.1



kern = Gaussian1DKernel(45)
y = convolve_fft(y, kern)

from mpl_toolkits.axes_grid.inset_locator import inset_axes






axes0_0 = inset_axes(axes[0],width=1.3, height=0.8,
                    bbox_to_anchor=(0.65, 0.35),
                    bbox_transform=axes[0].transAxes, loc=3, borderpad=0)
axes0_0.plot(x, y,'k-')

axes0_0.fill_between(x = x, \
                     y1 = y, 
                     where = y > 5.e-3,
                     color = 'blue', alpha = 1.0)


xlm = axes[0].get_xlim()

axes[0].axvspan(xmin = 800, xmax = xlm[1], ymin = 0.35, ymax = 1.0, alpha = 1.0, color = 'whitesmoke')

axes[0].set_xlim(xlm)

axes0_0.set_ylim(-0.008, 0.018)
axes0_0.set_xlim(500, 650)
axes0_0.axis('off')

axes0_0.annotate(r'$\Delta$t$_{90}$', (0.30, 0.02), color = 'black', ha = 'right', xycoords = 'axes fraction', va = 'bottom')
#axes0_0.annotate(r't$_{\textrm{total}}$', (0.30, 0.85),color = 'black', ha = 'right',  xycoords = 'axes fraction', va = 'top')
axes0_0.annotate(r't$_{total}$', (0.30, 0.85),color = 'black', ha = 'right',  xycoords = 'axes fraction', va = 'top')

axes0_0.annotate('90$\%$\nof total', (0.62, 0.72),color = 'blue', ha = 'left',  xycoords = 'axes fraction', va = 'top')

axes0_0.errorbar([600], [0.013], yerr = [0], xerr = [50], fmt = '', capsize=5, elinewidth=2, color = 'black')


xmn = min(x[y>5.e-3])
xmx = max(x[y>5.e-3])
xmd = (xmn+xmx)/2.
xer = abs(xmd - xmn)
axes0_0.errorbar([xmd], [-0.005], yerr = [0], xerr = xer,  fmt = '', capsize=5, elinewidth=2, color = 'black')




#axes0_0.plot([576.7, 550], [0, -0.01], '-', linewidth = 1, color = 'black', alpha = 0.3)
#axes0_0.plot([583.4, 550 + (583.4 - 576.7)], [0, -0.01], '-', linewidth = 1, color = 'black', alpha = 0.3)



#axes0_0.plot([597.2, 550 + (583.4 - 576.7)], [0, -0.01], '-', linewidth = 1, color = 'black', alpha = 0.3)
#axes0_0.plot([602.9, 550 + (583.4 - 576.7) +  (602.9 - 597.2)], [0, -0.01], '-', linewidth = 1, color = 'black', alpha = 0.3)

#axes0_0.axis('on')
axes22.annotate('more bursty\nram pressure', xy = (0.05,1.065), xytext = (0.15, 1.065), fontsize = 15,\
                 xycoords = 'axes fraction',textcoords='axes fraction', horizontalalignment = 'left', verticalalignment = 'bottom', ha = 'left', va = 'center', arrowprops=dict(arrowstyle="simple", color = 'black'), color = 'black')

axes22.annotate('less bursty\nram pressure', xy = (0.85, 1.065), xytext = (0.75, 1.065), fontsize = 15,\
                 xycoords = 'axes fraction',textcoords='axes fraction', horizontalalignment = 'left', verticalalignment = 'bottom', ha = 'right', va = 'center', arrowprops=dict(arrowstyle="simple", color = 'black'), color = 'black')


fig.subplots_adjust(top = 0.85)
fig.savefig('/Users/rsimons/Dropbox/foggie/figures/for_paper/ram_pressure_cumulative_test.png', dpi = 350)
#fig2.savefig('/Users/rsimons/Dropbox/foggie/figures/for_paper/cumulative_momentum_plot.png', dpi = 350)



















