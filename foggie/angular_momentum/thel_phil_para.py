from glob import glob
import matplotlib as mpl
import os
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
from numpy import *
import warnings
import matplotlib.colors as colors
import copy
import time
import astropy.units as u
warnings.filterwarnings("ignore")
plt.ioff()

def add_ax_lbls(ax):
    ax.set_xlim(0, 199)
    ax.set_ylim(199, 0)


    xtck_use  = np.arange(-180, 240, 60)
    xtcks_real = np.interp(xtck_use, [-180, 180],  [0, 199])
    ax.set_xticks(xtcks_real)

    ytck_use  = np.arange(0, 240, 60)
    ytcks_real = np.interp(ytck_use, [0, 180],  [0, 199])
    ax.set_yticks(ytcks_real)

    binsx = np.arange(199)
    binsy = np.arange(199)

    YY, XX = np.meshgrid(binsy, binsx)
    
    ax.set_xlabel(r'$\theta_{\mathrm{L}}$ (deg.)')
    tcklbl = ['%i'%tck for tck in xtck_use]
    tcklbl[0] = ''
    tcklbl[2] = ''
    tcklbl[4] = ''
    tcklbl[6] = ''
    ax.set_xticklabels(tcklbl)

    ax.set_ylabel(r'$\phi_{\mathrm{L}}$ (deg.)')
    ax.set_yticklabels(['%i'%tck for tck in ytck_use])
    return ax




def create_fig(fl, fig_dir, situation, DD_z_halo, mtype, cmap, map_type = 'L'):
    a = np.load(fl, allow_pickle = True)[()]

    aL = a[mtype][situation][map_type]
    if situation == 'inner':
        if map_type == 'L':        
            vmn = 68
            vmx = 71
        #if map_type == 'M':        
        #    vmn = 3
        #    vmx = 7
    if situation == 'outflow_inner':
        if map_type == 'L':        
            vmn = 66
            vmx = 69
 
    fig, ax = plt.subplots(1,1, facecolor = 'white', figsize = (4,4))
    aL[aL == 0.] = np.nan
    la = np.log10(aL)
    alpha = np.ones(shape(la))
    alpha[la < vmn] = 1 - (vmn - la[la < vmn])/3.
    alpha[alpha == 0] = 0.2
    
    ax.imshow(np.zeros(shape(la)) * np.nan, vmin = vmn, vmax = vmx, cmap = cmap)
    ax.imshow(la, vmin = vmn, vmax = vmx, cmap = cmap, alpha = alpha)



    ax = add_ax_lbls(ax)
    
    
    DDname = fl.split('/')[-1].split('_')[-2]
    redshift = float(DD_z_halo[DDname])
    ax.annotate('%s\n%s\nz = %.2f'%(halo, DDname, redshift), (0.05, 0.05), xycoords = 'axes fraction', color = 'white', 
                fontsize = 15, ha = 'left', va = 'bottom')
    ax.annotate('%s'%(mtype.replace('_', ' ')), (0.95, 0.05), xycoords = 'axes fraction', color = 'white', 
                fontsize = 20, ha = 'right', va = 'bottom', fontweight = 'bold')
    
    figname = fig_dir + '/' + fl.split('/')[-1].replace('_rdist.npy', '_%s_%s_%s.png'%(situation, mtype, map_type)).replace('Lprof_', '')
    fig.tight_layout()
    fig.savefig(figname, dpi = 500)


    plt.close(fig)
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def flatten_thelphil(Lhst):
    t = np.linspace(-180, 180, 199)
    p = np.linspace(0, 180, 199)
    
    

    TT, PP = np.meshgrid(t, p)

    
    TTrad = TT*pi/180
    PPrad = PP*pi/180
    x = Lhst * sin(PPrad) * cos(TTrad)
    y = Lhst * sin(PPrad) * sin(TTrad)
    z = Lhst * cos(PPrad)

    x_sum = np.nansum(x)*u.g*u.cm**2./u.s
    y_sum = np.nansum(y)*u.g*u.cm**2./u.s
    z_sum = np.nansum(z)*u.g*u.cm**2./u.s

    L_sum = np.sqrt(x_sum**2 + y_sum**2. + z_sum**2)
    thel_mean = np.arctan2(y_sum,x_sum).value*180./pi
    phil_mean = np.arccos(z_sum/L_sum).value*180./pi
    return L_sum, thel_mean, phil_mean

def create_fig_several(fl, fig_dir, situation, DD_z_halo, mtype, pt_means, cmap, map_type = 'L', make_fig = True):
    figname = fig_dir + '/' + fl.split('/')[-1].replace('_rdist.npy', '_%s_%s_%s.png'%(situation, mtype, map_type)).replace('Lprof_', '')
    DDname = fl.split('/')[-1].split('_')[-2]
    redshift = float(DD_z_halo[DDname])
    #if os.path.exists(figname): return
    a = np.load(fl, allow_pickle = True)[()]
    #fig, ax = plt.subplots(1,1, facecolor = 'white', figsize = (4,4))
    #ax1 = fig3.add_gridspec(3, 3)
    if make_fig:
        fig = plt.figure(facecolor = 'white', figsize = (6.,4))
        gs = fig.add_gridspec(nrows = 2, ncols = 4)
        ax = fig.add_subplot(gs[:,:3])
        ax2 = fig.add_subplot(gs[0,3])
        ax3 = fig.add_subplot(gs[1,3])
        ax_ind = [ax2, ax3]

    mtype_1 = mtype.split('_and_')[0]
    mtype_2 = mtype.split('_and_')[1]

    vmns = []
    vmxs = []

    for mm, mtype_i in enumerate([mtype_1, mtype_2]):

        if (mtype_i == 'cgm-inflow'):
            aL_i = a['warm']['cgm_inflow'][map_type] + a['cold']['cgm_inflow'][map_type]
            vmns.append(67)
            vmxs.append(71)

        if (mtype_i == 'galaxy-inflow'):
            aL_i = a['warm']['galaxy_inflow'][map_type] + a['cold']['galaxy_inflow'][map_type]
            vmns.append(65)
            vmxs.append(69)

        if mtype_i == 'cosmoflow':
            aL_i = a['cold']['galaxy_zpoor_inflow'][map_type] + a['cold']['cgm_zpoor_inflow'][map_type] +\
                   a['warm']['galaxy_zpoor_inflow'][map_type] + a['warm']['cgm_zpoor_inflow'][map_type]
            vmns.append(65)
            vmxs.append(70)
        if mtype_i == 'cgm-cosmoflow':
            aL_i = a['cold']['cgm_zpoor_inflow'][map_type] + a['warm']['cgm_zpoor_inflow'][map_type] + a['warm']['cgm_zpoor_inflow'][map_type]

            vmns.append(65)
            vmxs.append(70)
        if mtype_i == 'galaxy-cosmoflow':
            aL_i = a['cold']['galaxy_zpoor_inflow'][map_type] + a['warm']['galaxy_zpoor_inflow'][map_type]
            vmns.append(65)
            vmxs.append(70)

        if mtype_i == 'halo':
            aL_i = np.zeros((199, 199))
            for mtype_m in ['cold', 'warm', 'warmhot', 'hot', 'stars', 'dm']:
                for loc_type in ['galaxy', 'cgm']:
                    for flow_type in ['inflow', 'outflow']:
                        if (mtype_m == 'stars') | (mtype_m == 'dm'):
                                aL_i+=a[mtype_m]['{}_{}'.format(loc_type, flow_type)][map_type]
                        else:
                            for speed_type in ['s', 'f']:
                                aL_i+=a[mtype_m]['{}_{}{}'.format(loc_type, speed_type, flow_type)][map_type]

            vmns.append(69)
            vmxs.append(73)

        if mtype_i == 'halo-dm':
            aL_i = np.zeros((199, 199))
            for mtype_m in ['dm']:
                for loc_type in ['galaxy', 'cgm']:
                    for flow_type in ['inflow', 'outflow']:
                        if (mtype_m == 'stars') | (mtype_m == 'dm'):
                                aL_i+=a[mtype_m]['{}_{}'.format(loc_type, flow_type)][map_type]
                        else:
                            for speed_type in ['s', 'f']:
                                aL_i+=a[mtype_m]['{}_{}{}'.format(loc_type, speed_type, flow_type)][map_type]

            vmns.append(69)
            vmxs.append(73)


        if mtype_i == 'cgm-baryons':
            loc_type = 'cgm'
            aL_i = np.zeros((199, 199))
            for mtype_m in ['cold', 'warm', 'warmhot', 'hot', 'stars']:
                for flow_type in ['inflow', 'outflow']:
                    if (mtype_m == 'stars') | (mtype_m == 'dm'):
                            aL_i+=a[mtype_m]['{}_{}'.format(loc_type, flow_type)][map_type]
                    else:
                        for speed_type in ['s', 'f']:
                            aL_i+=a[mtype_m]['{}_{}{}'.format(loc_type, speed_type, flow_type)][map_type]

            vmns.append(67)
            vmxs.append(72)

        if mtype_i == 'galaxy-baryons':
            loc_type = 'galaxy'
            aL_i = np.zeros((199, 199))
            for mtype_m in ['cold', 'warm', 'warmhot', 'hot', 'stars']:
                for flow_type in ['inflow', 'outflow']:
                    if (mtype_m == 'stars') | (mtype_m == 'dm'):
                            aL_i+=a[mtype_m]['{}_{}'.format(loc_type, flow_type)][map_type]
                    else:
                        for speed_type in ['s', 'f']:
                            aL_i+=a[mtype_m]['{}_{}{}'.format(loc_type, speed_type, flow_type)][map_type]

            vmns.append(67)
            vmxs.append(72)


        if mtype_i == 'cgm-gas':
            loc_type = 'cgm'
            aL_i = np.zeros((199, 199))
            for mtype_m in ['cold', 'warm', 'warmhot', 'hot']:
                for flow_type in ['inflow', 'outflow']:
                    if (mtype_m == 'stars') | (mtype_m == 'dm'):
                            aL_i+=a[mtype_m]['{}_{}'.format(loc_type, flow_type)][map_type]
                    else:
                        for speed_type in ['s', 'f']:
                            aL_i+=a[mtype_m]['{}_{}{}'.format(loc_type, speed_type, flow_type)][map_type]

            vmns.append(65)
            vmxs.append(71)

        if mtype_i == 'galaxy-gas':
            loc_type = 'galaxy'
            aL_i = np.zeros((199, 199))
            for mtype_m in ['cold', 'warm', 'warmhot', 'hot']:
                for flow_type in ['inflow', 'outflow']:
                    if (mtype_m == 'stars') | (mtype_m == 'dm'):
                            aL_i+=a[mtype_m]['{}_{}'.format(loc_type, flow_type)][map_type]
                    else:
                        for speed_type in ['s', 'f']:
                            aL_i+=a[mtype_m]['{}_{}{}'.format(loc_type, speed_type, flow_type)][map_type]

            vmns.append(65)
            vmxs.append(71)


        if mtype_i == 'galaxy-gas-finflow':
            loc_type = 'galaxy'
            aL_i = np.zeros((199, 199))
            for mtype_m in ['cold', 'warm', 'warmhot', 'hot']:
                for flow_type in ['inflow']:
                    if (mtype_m == 'stars') | (mtype_m == 'dm'):
                            aL_i+=a[mtype_m]['{}_{}'.format(loc_type, flow_type)][map_type]
                    else:
                        for speed_type in ['f']:
                            aL_i+=a[mtype_m]['{}_{}{}'.format(loc_type, speed_type, flow_type)][map_type]

            vmns.append(65)
            vmxs.append(71)

        if mtype_i == 'galaxy-gas-sinflow':
            loc_type = 'galaxy'
            aL_i = np.zeros((199, 199))
            for mtype_m in ['cold', 'warm', 'warmhot', 'hot']:
                for flow_type in ['inflow']:
                    if (mtype_m == 'stars') | (mtype_m == 'dm'):
                            aL_i+=a[mtype_m]['{}_{}'.format(loc_type, flow_type)][map_type]
                    else:
                        for speed_type in ['s']:
                            aL_i+=a[mtype_m]['{}_{}{}'.format(loc_type, speed_type, flow_type)][map_type]

            vmns.append(65)
            vmxs.append(71)






        if mtype_i == 'cgm-coolgas':
            loc_type = 'cgm'
            aL_i = np.zeros((199, 199))
            for mtype_m in ['cold', 'warm']:
                for flow_type in ['inflow', 'outflow']:
                    if (mtype_m == 'stars') | (mtype_m == 'dm'):
                            aL_i+=a[mtype_m]['{}_{}'.format(loc_type, flow_type)][map_type]
                    else:
                        for speed_type in ['s', 'f']:
                            aL_i+=a[mtype_m]['{}_{}{}'.format(loc_type, speed_type, flow_type)][map_type]

            vmns.append(65)
            vmxs.append(71)


        if mtype_i == 'cgm-cold':
            loc_type = 'cgm'
            aL_i = np.zeros((199, 199))
            for mtype_m in ['cold']:
                for flow_type in ['inflow', 'outflow']:
                    if (mtype_m == 'stars') | (mtype_m == 'dm'):
                            aL_i+=a[mtype_m]['{}_{}'.format(loc_type, flow_type)][map_type]
                    else:
                        for speed_type in ['s', 'f']:
                            aL_i+=a[mtype_m]['{}_{}{}'.format(loc_type, speed_type, flow_type)][map_type]

            vmns.append(65)
            vmxs.append(71)

        if mtype_i == 'cgm-warmhot':
            loc_type = 'cgm'
            aL_i = np.zeros((199, 199))
            for mtype_m in ['warmhot', 'hot']:
                for flow_type in ['inflow', 'outflow']:
                    if (mtype_m == 'stars') | (mtype_m == 'dm'):
                            aL_i+=a[mtype_m]['{}_{}'.format(loc_type, flow_type)][map_type]
                    else:
                        for speed_type in ['s', 'f']:
                            aL_i+=a[mtype_m]['{}_{}{}'.format(loc_type, speed_type, flow_type)][map_type]

            vmns.append(65)
            vmxs.append(71)

        if mtype_i == 'galaxy-coolgas':
            loc_type = 'galaxy'
            aL_i = np.zeros((199, 199))
            for mtype_m in ['cold', 'warm']:
                for flow_type in ['inflow', 'outflow']:
                    if (mtype_m == 'stars') | (mtype_m == 'dm'):
                            aL_i+=a[mtype_m]['{}_{}'.format(loc_type, flow_type)][map_type]
                    else:
                        for speed_type in ['s', 'f']:
                            aL_i+=a[mtype_m]['{}_{}{}'.format(loc_type, speed_type, flow_type)][map_type]

            vmns.append(65)
            vmxs.append(71)


        if mtype_i == 'finflow-cgm':
            aL_i = a['cold']['cgm_finflow'][map_type] + a['warm']['cgm_finflow'][map_type]
            vmns.append(65)
            vmxs.append(71)

        if mtype_i == 'finflow-galaxy':
            aL_i = a['cold']['galaxy_finflow'][map_type] + a['warm']['galaxy_finflow'][map_type]
            vmns.append(65)
            vmxs.append(71)

        if mtype_i == 'outflow': 
            aL_i = a['hot']['galaxy_foutflow'][map_type] + a['warmhot']['galaxy_foutflow'][map_type]
            vmns.append(65)
            vmxs.append(68)

        if mtype_i == 'disk':    
            aL_i = a['young_stars']['galaxy_inflow'][map_type] + a['young_stars']['galaxy_outflow'][map_type]
            vmns.append(65)
            vmxs.append(69)


        if mtype_i == 'stars-galaxy':   
            aL_i = a['stars']['galaxy_inflow'][map_type] + a['stars']['galaxy_outflow'][map_type]
            vmns.append(67)
            vmxs.append(70)

        if mtype_i == 'stars-cgm':   
            aL_i = a['stars']['cgm_inflow'][map_type] + a['stars']['cgm_outflow'][map_type]
            vmns.append(67)
            vmxs.append(70)


        if mtype_i == 'stars-inflow':   
            aL_i = a['stars']['galaxy_inflow'][map_type] + a['stars']['cgm_inflow'][map_type]
            vmns.append(67)
            vmxs.append(70)

        if mtype_i == 'stars-inflow-galaxy':   
            aL_i = a['stars']['galaxy_inflow'][map_type]
            vmns.append(67)
            vmxs.append(70)

        if mtype_i == 'stars-inflow-cgm':   
            aL_i = a['stars']['galaxy_inflow'][map_type]
            vmns.append(67)
            vmxs.append(70)

        if mtype_i == 'cold':    
            aL_i = a['cold']['inner'][map_type]
            vmns.append(67)
            vmxs.append(70.5)

        if mtype_i == 'colddisk':    
            aL_i = a['cold']['galaxy_sinflow'][map_type] + a['cold']['galaxy_soutflow'][map_type]
            vmns.append(66)
            vmxs.append(70.5)

        if mm == 0: aL_s = aL_i
        if mm == 1: aL_ys = aL_i

    if make_fig:
        for ax_temp in [ax, ax2, ax3]:
            ax_temp.imshow(np.zeros(shape(aL_s)) * np.nan, cmap = cmap)
    

        cmap_parent1 = copy.copy(mpl.cm.get_cmap("hot"))
        cmap_parent1.set_bad('k')

        cmap_parent2 = copy.copy(mpl.cm.get_cmap("Blues_r"))
        cmap_parent2.set_bad('k')

        cmaps = [truncate_colormap(cmap_parent1, 0.2, 0.5), truncate_colormap(cmap_parent2, 0.2, 0.5)]
    
    for aa, aL in enumerate([aL_s, aL_ys]):
        to_sv = {}

        L_sum, thel_mean, phil_mean = flatten_thelphil(aL)

        to_sv['thel_mean'] = thel_mean
        to_sv['phil_mean'] =  phil_mean
        to_sv['Lsum'] = L_sum

        if make_fig:
            aL[aL == 0.] = np.nan
            vmn = vmns[aa]
            vmx = vmxs[aa]
            la = np.log10(aL)
            #if aa > 0:
            alp_low = 0.2
            alp_hig = 0.9
            #else:
            #    alp_low = 0.0
            #alpha = (np.ones(shape(la)) - alp_low)#/(vmn - vmx)

            alpha = (la - vmn)/(vmx - vmn)

            alpha[alpha > alp_hig] = alp_hig
            alpha[alpha < alp_low] = alp_low

            if len(pt_means) > 0:
                if len(where(pt_means[aa]['DD'] == DDname)[0]) > 0:
                    gd = pt_means[aa]['DD'] == DDname
                    thel_mean = pt_means[aa]['thel_smooth'][gd]*180./pi - 180
                    phil_mean = pt_means[aa]['phil_smooth'][gd]*180./pi
            ax.plot((thel_mean + 180)*(199/360.), phil_mean*(199/180.), marker = 'o', markersize = 10,
                color = 'white', markeredgecolor = cmaps[aa](1.0), linewidth = 3, alpha = 0.6)
            ax.imshow(la, vmin = vmn, vmax = vmx, cmap = cmaps[aa], alpha = alpha, zorder = aa + 1)

            #ax_ind[aa].plot((thel_mean + 180)*(199/360.), phil_mean*(199/180.), marker = 'o', markersize = 10,
            #    color = 'white', markeredgecolor = cmaps[aa](1.0), linewidth = 3, alpha = 0.6)
            ax_ind[aa].imshow(la, vmin = vmn, vmax = vmx, cmap = cmaps[aa], alpha = alpha, zorder = aa + 1)
            #else:
            #    ax.contourf(la, 2, colors='red', alpha = alpha)

            #    #ax.clab(la, levels = np.arange(vmn, vmx, 10), colors = 'r', alpha = 1.0, zorder = aa + 1) 

        mtype_i = [mtype_1, mtype_2][aa]
        sv_fle = '/Users/rsimons/Dropbox/foggie/angular_momentum/profiles/temp_save/' + fl.split('/')[-1].replace('_rdist.npy', '_%s.npy'%mtype_i)
        np.save(sv_fle, to_sv)

    if make_fig:
        ax = add_ax_lbls(ax)
        for ax_temp in ax_ind:
            add_ax_lbls(ax_temp)
            ax_temp.axis('off')
            ax_temp.annotate('z = %.2f'%(redshift), (0.05, 0.05), xycoords = 'axes fraction', color = 'white', 
            fontsize = 15, ha = 'left', va = 'bottom')
        
        ax.annotate('%s\n%s\nz = %.2f'%(halo, DDname, redshift), (0.05, 0.05), xycoords = 'axes fraction', color = 'white', 
                    fontsize = 15, ha = 'left', va = 'bottom')
        ax.annotate('%s'%(mtype.replace('_and_', '\n + \n')), (0.95, 0.05), xycoords = 'axes fraction', color = 'white', 
                    fontsize = 20, ha = 'right', va = 'bottom', fontweight = 'bold')
        #ax_ind[0].annotate('%s'%(mtype.split('_')[0]), (0.95, 0.05), xycoords = 'axes fraction', color = 'white', 
        #            fontsize = 20, ha = 'right', va = 'bottom', fontweight = 'bold')
        #ax_ind[1].annotate('%s'%(mtype.split('_')[-1]), (0.95, 0.05), xycoords = 'axes fraction', color = 'white', 
        #            fontsize = 20, ha = 'right', va = 'bottom', fontweight = 'bold')
        
        #fig.tight_layout()
        fig.subplots_adjust(left = 0.02, wspace = -0.2, hspace = 0.00, top = 0.98, right = 0.95, bottom = 0.15)
        fig.savefig(figname, dpi = 500)


        plt.close(fig)



cmap = copy.copy(mpl.cm.get_cmap("viridis"))
cmap.set_bad('k')
DD_z = np.load('/Users/rsimons/Dropbox/foggie/catalogs/DD_redshift.npy', allow_pickle = True)[()]
mov_dir = '/Users/rsimons/Dropbox/foggie/angular_momentum/figures/thel_phil/mp4s'

#situation, mtype = 'outflow_inner', 'hot'
#situation, mtype = 'mix', 'disk_and_inflow'
#situation, mtype = 'mix', 'stars_and_cold'
#situation, mtype = 'mix', 'stars_and_inflow'
#situation, mtype = 'mix', 'cold_and_inflow'
situation = 'mix'

mtypes = ['stars-cgm_and_stars-galaxy', 'cosmoflow_and_disk', 'disk_and_stars-galaxy', \
          'finflow-cgm_and_finflow-galaxy', 'outflow_and_disk', 'halo_and_disk', \
          'stars-galaxy_and_finflow-galaxy', 'stars-inflow-galaxy_and_finflow-galaxy',\
          'stars-galaxy_and_colddisk', 'cgm-baryons_and_galaxy-baryons',\
          'cgm-gas_and_galaxy-gas', 'cgm-coolgas_and_galaxy-coolgas', 'halo_and_disk',\
          'halo-dm_and_disk', 'cgm-gas_and_cosmoflow', 'stars-galaxy_and_stars-cgm',\
          'cgm-warmhot_and_cgm-cold', 'cgm-cosmoflow_and_galaxy-cosmoflow',\
          'galaxy-gas-sinflow_and_galaxy-gas-finflow', 'stars-galaxy_and_colddisk'][-1:]
#['stars_and_cold', 'stars_and_disk', 'stars_and_inflow', \
#              'cold_and_inflow', 'disk_and_inflow', 'outflow_and_disk',\
#              'cgm-inflow_and_galaxy-inflow'][-1:]
make_fig = True

for mtype in mtypes:
    for halo in ['5036', '5016', '8508', '2392', '4123', '2878'][-2:-1]:
        if make_fig: os.system('rm /Users/rsimons/.matplotlib/tex.cache/*lock*')
        print (halo, mtype)
        plt.close('all')
        pt_fl1 = '/Users/rsimons/Dropbox/foggie/angular_momentum/profiles/temp_save/smooth_pt_{}_{}.npy'.format(halo, mtype.split('_and_')[0])
        pt_fl2 = '/Users/rsimons/Dropbox/foggie/angular_momentum/profiles/temp_save/smooth_pt_{}_{}.npy'.format(halo, mtype.split('_and_')[1])
        if (os.path.exists(pt_fl1)) & (os.path.exists(pt_fl2)):
            pt_mean1 = np.load(pt_fl1, allow_pickle = True)[()]
            pt_mean2 = np.load(pt_fl2, allow_pickle = True)[()]
            pt_means = [pt_mean1, pt_mean2]
        else:
            pt_means = []
        map_type = 'L'
        rdist_dir = '/Users/rsimons/Dropbox/foggie/angular_momentum/profiles/%s/rdist'%halo
        fig_dir = '/Users/rsimons/Dropbox/foggie/angular_momentum/figures/thel_phil/%s/%s/%s'%(halo, situation, mtype)
        if not os.path.isdir(fig_dir): os.system('mkdir %s'%fig_dir)
        #fls = sort(glob(rdist_dir + '/*DD0500*npy'))
        fls = sort(glob(rdist_dir + '/*DD????*npy'))[::-1]
        #Parallel(n_jobs = -1)(delayed(create_fig)(fl, fig_dir, situation, DD_z[halo], mtype, cmap, map_type) for fl in fls)
        Parallel(n_jobs = -1)(delayed(create_fig_several)(fl, fig_dir, situation, DD_z[halo], mtype,pt_means, cmap, map_type, make_fig = make_fig) for fl in fls)
        #for fl in fls: create_fig_several(fl, fig_dir, situation, DD_z[halo], mtype, pt_means, cmap, map_type)
        if make_fig:
            mov_file = mov_dir + '/%s_%s_%s_%s.mp4'%(halo, situation, mtype, map_type)
            os.system("ffmpeg -y -f image2 -r 24 -pattern_type glob -i '%s/*%s_%s_%s.png' "%(fig_dir, situation, mtype, map_type) +
                      "-vcodec libx264 -crf 25  -pix_fmt yuv420p %s"%mov_file)

            plt.close('all')
            os.system('rm /Users/rsimons/.matplotlib/tex.cache/*lock*')






