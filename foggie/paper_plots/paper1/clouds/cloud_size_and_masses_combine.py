import glob
import os
import pickle
import numpy as np

#import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
#sns.set_style("whitegrid", {'axes.grid' : False})
mpl.rcParams['font.family'] = 'stixgeneral'
mpl.rcParams['font.size'] = 26.

h=0.695  # damn you little h!!

def plot_cloud_size_and_masses():
    h1_limit = 1.e13
    si2_limit = 1.e11
    c4_limit = 1.e12
    si4_limit = 1.e13
    o6_limit = 1.e13

    hi_color = 'salmon' ## '#984ea3' # purple
    ovi_color = '#4daf4a'  # green
    si2_color = '#984ea3' # 'darkorange'
    c4_color = "#4575b4" # blue 'darkorange'
    si4_color = "#4575b4" #'darkorange'

    nref10_cell = 1000. * 100 / (256 * np.power(2,10)) ## ckpc/h

    output_dir = '/Users/molly/Dropbox/foggie-collab/plots_halo_008508/nref11n/comparisons/clouds/'

    ### this will only work in python 3 !!
    filelist = glob.glob(os.path.join('/Users/molly/Dropbox/foggie-collab/plots_halo_008508/nref11n/natural/spectra/lls', '*.pkl'))
    # filelist = filelist[0:100]
    print('there are ',np.size(filelist),'files')
    natural_size_dict = pickle.load( open( filelist[0], "rb" ) )
    print(natural_size_dict.keys())

    natural_h1sizes = []
    natural_h1sizes_phys = []
    natural_h1_n_cells = []
    natural_h1masses = []
    natural_h1columns = []
    natural_h1cloudcolumns = []

    natural_o6sizes = []
    natural_o6sizes_phys = []
    natural_o6_n_cells = []
    natural_o6masses = []
    natural_o6columns = []
    natural_o6cloudcolumns = []

    natural_si2sizes = []
    natural_si2sizes_phys = []
    natural_si2_n_cells = []
    natural_si2masses = []
    natural_si2columns = []
    natural_si2cloudcolumns = []

    natural_c4sizes = []
    natural_c4sizes_phys = []
    natural_c4_n_cells = []
    natural_c4masses = []
    natural_c4columns = []
    natural_c4cloudcolumns = []

    natural_si4sizes = []
    natural_si4sizes_phys = []
    natural_si4_n_cells = []
    natural_si4masses = []
    natural_si4columns = []
    natural_si4cloudcolumns = []

    for file in filelist:
        natural_size_dict = pickle.load( open( file, "rb" ) )
        if "rd0020" in file:
            z = 2.0
        if "rd0018" in file:
            z = 2.5
        ### make this limit different for each ion
        if (natural_size_dict['nh1'] > h1_limit):
            if np.max(natural_size_dict['h1_kpcsizes']) < 200:
                for item in natural_size_dict['h1_kpcsizes']: natural_h1sizes.append(item)
                for item in natural_size_dict['h1_kpcsizes']: natural_h1sizes_phys.append(h*item/(1+z))
                for item in natural_size_dict['h1_cell_masses']: natural_h1masses.append(item)
                for item in natural_size_dict['h1_coldens']: natural_h1cloudcolumns.append(item)
                natural_h1_n_cells.append(natural_size_dict['h1_n_cells'])
                natural_h1columns.append(natural_size_dict['nh1'])

        if (natural_size_dict['nsi2'] > si2_limit):
            for item in natural_size_dict['si2_kpcsizes']: natural_si2sizes.append(item)
            for item in natural_size_dict['si2_kpcsizes']: natural_si2sizes_phys.append(h*item/(1+z))
            for item in natural_size_dict['si2_cell_masses']: natural_si2masses.append(item)
            for item in natural_size_dict['si2_coldens']: natural_si2cloudcolumns.append(item)
            natural_si2_n_cells.append(natural_size_dict['si2_n_cells'])
            natural_si2columns.append(natural_size_dict['nsi2'])

        if (natural_size_dict['no6'] > o6_limit):
            if np.max(natural_size_dict['o6_kpcsizes']) < 200:
                for item in natural_size_dict['o6_kpcsizes']: natural_o6sizes.append(item)
                for item in natural_size_dict['o6_kpcsizes']: natural_o6sizes_phys.append(h*item/(1+z))
                for item in natural_size_dict['o6_cell_masses']: natural_o6masses.append(item)
                for item in natural_size_dict['o6_coldens']: natural_o6cloudcolumns.append(item)
                natural_o6_n_cells.append(natural_size_dict['o6_n_cells'])
                natural_o6columns.append(natural_size_dict['no6'])

        if (natural_size_dict['nc4'] > c4_limit):
            if np.max(natural_size_dict['c4_kpcsizes']) < 200:
                for item in natural_size_dict['c4_kpcsizes']: natural_c4sizes.append(item)
                for item in natural_size_dict['c4_kpcsizes']: natural_c4sizes_phys.append(h*item/(1+z))
                for item in natural_size_dict['c4_cell_masses']: natural_c4masses.append(item)
                for item in natural_size_dict['c4_coldens']: natural_c4cloudcolumns.append(item)
                natural_c4_n_cells.append(natural_size_dict['c4_n_cells'])
                natural_c4columns.append(natural_size_dict['nc4'])

        # if (natural_size_dict['nsi4'] > si4_limit):
        #     if np.max(natural_size_dict['si4_kpcsizes']) < 200:
        #         for item in natural_size_dict['si4_kpcsizes']: natural_si4sizes.append(item)
        #         for item in natural_size_dict['si4_cell_masses']: natural_si4masses.append(item)
        #         for item in natural_size_dict['si4_coldens']: natural_si4cloudcolumns.append(item)
        #         natural_si4_n_cells.append(natural_size_dict['si4_n_cells'])
        #         natural_si4columns.append(natural_size_dict['nsi4'])


    filelist = glob.glob(os.path.join('/Users/molly/Dropbox/foggie-collab/plots_halo_008508/nref11n/nref11n_nref10f_refine200kpc/spectra/lls', '*.pkl'))
    # filelist = filelist[0:100]
    print('there are ',np.size(filelist),'files')
    nref10f_size_dict = pickle.load( open( filelist[0], "rb" ) )
    print(nref10f_size_dict.keys())

    nref10f_h1sizes = []
    nref10f_h1sizes_phys = []
    nref10f_h1_n_cells = []
    nref10f_h1masses = []
    nref10f_h1columns = []
    nref10f_h1cloudcolumns = []

    nref10f_o6sizes = []
    nref10f_o6sizes_phys = []
    nref10f_o6_n_cells = []
    nref10f_o6masses = []
    nref10f_o6columns = []
    nref10f_o6cloudcolumns = []

    nref10f_si2sizes = []
    nref10f_si2sizes_phys = []
    nref10f_si2_n_cells = []
    nref10f_si2masses = []
    nref10f_si2columns = []
    nref10f_si2cloudcolumns = []

    nref10f_c4sizes = []
    nref10f_c4sizes_phys = []
    nref10f_c4_n_cells = []
    nref10f_c4masses = []
    nref10f_c4columns = []
    nref10f_c4cloudcolumns = []

    nref10f_si4sizes = []
    nref10f_si4sizes_phys = []
    nref10f_si4_n_cells = []
    nref10f_si4masses = []
    nref10f_si4columns = []
    nref10f_si4cloudcolumns = []

    for file in filelist:
        nref10f_size_dict = pickle.load( open( file, "rb" ) )
        ### make this limit different for each ion
        if (nref10f_size_dict['nh1'] > h1_limit):
            if np.max(nref10f_size_dict['h1_kpcsizes']) < 200:
                for item in nref10f_size_dict['h1_kpcsizes']: nref10f_h1sizes.append(item)
                for item in nref10f_size_dict['h1_kpcsizes']: nref10f_h1sizes_phys.append(h*item/(1+z))
                for item in nref10f_size_dict['h1_cell_masses']: nref10f_h1masses.append(item)
                for item in nref10f_size_dict['h1_coldens']: nref10f_h1cloudcolumns.append(item)
                nref10f_h1_n_cells.append(nref10f_size_dict['h1_n_cells'])
                nref10f_h1columns.append(nref10f_size_dict['nh1'])

        if (nref10f_size_dict['nsi2'] > si2_limit):
            for item in nref10f_size_dict['si2_kpcsizes']: nref10f_si2sizes.append(item)
            for item in nref10f_size_dict['si2_kpcsizes']: nref10f_si2sizes_phys.append(h*item/(1+z))
            for item in nref10f_size_dict['si2_cell_masses']: nref10f_si2masses.append(item)
            for item in nref10f_size_dict['si2_coldens']: nref10f_si2cloudcolumns.append(item)
            nref10f_si2_n_cells.append(nref10f_size_dict['si2_n_cells'])
            nref10f_si2columns.append(nref10f_size_dict['nsi2'])

        if (nref10f_size_dict['no6'] > o6_limit):
            if np.max(nref10f_size_dict['o6_kpcsizes']) < 200:
                for item in nref10f_size_dict['o6_kpcsizes']: nref10f_o6sizes.append(item)
                for item in nref10f_size_dict['o6_kpcsizes']: nref10f_o6sizes_phys.append(h*item/(1+z))
                for item in nref10f_size_dict['o6_cell_masses']: nref10f_o6masses.append(item)
                for item in nref10f_size_dict['o6_coldens']: nref10f_o6cloudcolumns.append(item)
                nref10f_o6_n_cells.append(nref10f_size_dict['o6_n_cells'])
                nref10f_o6columns.append(nref10f_size_dict['no6'])

        if (nref10f_size_dict['nc4'] > c4_limit):
            if np.max(nref10f_size_dict['c4_kpcsizes']) < 200:
                for item in nref10f_size_dict['c4_kpcsizes']: nref10f_c4sizes.append(item)
                for item in nref10f_size_dict['c4_kpcsizes']: nref10f_c4sizes_phys.append(h*item/(1+z))
                for item in nref10f_size_dict['c4_cell_masses']: nref10f_c4masses.append(item)
                for item in nref10f_size_dict['c4_coldens']: nref10f_c4cloudcolumns.append(item)
                nref10f_c4_n_cells.append(nref10f_size_dict['c4_n_cells'])
                nref10f_c4columns.append(nref10f_size_dict['nc4'])

        # if (nref10f_size_dict['nsi4'] > si4_limit):
        #     if np.max(nref10f_size_dict['si4_kpcsizes']) < 200:
        #         for item in nref10f_size_dict['si4_kpcsizes']: nref10f_si4sizes.append(item)
        #         for item in nref10f_size_dict['si4_cell_masses']: nref10f_si4masses.append(item)
        #         for item in nref10f_size_dict['si4_coldens']: nref10f_si4cloudcolumns.append(item)
        #         nref10f_si4_n_cells.append(nref10f_size_dict['si4_n_cells'])
        #         nref10f_si4columns.append(nref10f_size_dict['nsi4'])


    ####################################################
    ####### cumulative histogram of sizes ##############
    ####################################################
    fig = plt.figure(figsize=(14,11))
    axtop = fig.add_axes([0.1, 0.52, 0.88, 0.42],
                   ylim=(0, 1.05), xlim=(0.1, 60))
    axbot = fig.add_axes([0.1, 0.1, 0.88, 0.42],
                   ylim=(0, 1.05), xlim=(0.1, 60))
    hist_bins = 0.5 * nref10_cell * (np.arange(2000)+1.)
    for i in np.arange(7)+1.:
        axtop.plot([0.5 * nref10_cell*2.**i, nref10_cell*2.**i],[0,2000],'--', lw=5, color='grey')
        axbot.plot([0.5 * nref10_cell*2.**i, nref10_cell*2.**i],[0,2000],'--', lw=5, color='grey')
    for i in np.arange(2)+2.:
        axtop.plot([0.5 * nref10_cell*2.**i, nref10_cell*2.**i],[0,2000],'--', lw=5, color='grey')
        axbot.plot([0.5 * nref10_cell*2.**i, nref10_cell*2.**i],[0,2000],'--', lw=5, color='grey')
    axtop.hist(natural_h1sizes, hist_bins, range=(0, 500), cumulative=-1,normed=True,color=hi_color, histtype='step',lw=3, label='H I')
    axtop.hist(natural_si2sizes, hist_bins, range=(0, 500), cumulative=-1,normed=True,color=si2_color, histtype='step',lw=3, label='Si II')
    axtop.hist(natural_c4sizes, hist_bins, range=(0, 500), cumulative=-1,normed=True,color=c4_color, histtype='step',lw=3, label='C IV')
    axtop.hist(natural_o6sizes, hist_bins, range=(0, 500), cumulative=-1,normed=True,color=ovi_color, histtype='step',lw=3, label='O VI')
    axtop.set_xscale('log')
    #axtop.set_xticks(())
    axtop.set_xticklabels(())
    axtop.set_yticks((0.25, 0.5, 0.75, 1.0))
    axtop.set_yticklabels(('0.25','0.5','0.75','1.00'))
    axtop.grid(True)
    axtop.legend(loc='upper right')
    fig.text(0.66, 0.91, 'standard resolution', fontsize=32, ha='right')


    axbot.hist(nref10f_h1sizes, hist_bins, range=(0, 500), cumulative=-1,normed=True,color=hi_color, histtype='step',lw=3, label='H I')
    axbot.hist(nref10f_si2sizes, hist_bins, range=(0, 500), cumulative=-1,normed=True,color=si2_color, histtype='step',lw=3, label='Si II')
    axbot.hist(nref10f_c4sizes, hist_bins, range=(0, 500), cumulative=-1,normed=True,color=c4_color, histtype='step',lw=3, label='C IV')
    axbot.hist(nref10f_o6sizes, hist_bins, range=(0, 500), cumulative=-1,normed=True,color=ovi_color, histtype='step',lw=3, label='O VI')
    axbot.set_xscale('log')
    axbot.set_xticks((0.1, 1, 10))
    axbot.set_xticklabels(('0.1','1','10'))
    axbot.grid(True)
    fig.text(0.66, 0.48, 'high resolution', fontsize=32, ha='right')

    fig.text(0.5, 0.02, 'cloud size [comoving kpc]', fontsize=34, ha='center')
    fig.text(0.02, 0.5, 'fraction of clouds with larger size', fontsize=34, va='center', rotation='vertical')

    # plt.tight_layout()
    plt.savefig(output_dir + 'cloud_size_cumulative_histogram.png')
    plt.savefig(output_dir + 'cloud_size_cumulative_histogram.pdf')




    ####################################################
    #######  histogram of sizes ########################
    ####################################################
    fig = plt.figure(figsize=(14,11), dpi=300)
    axtop = fig.add_axes([0.1, 0.52, 0.88, 0.42],
                   ylim=(0, 0.85), xlim=(0.1, 60))
    axbot = fig.add_axes([0.1, 0.1, 0.88, 0.42],
                   ylim=(0, 0.85), xlim=(0.1, 60))

    hist_bins = nref10_cell * (np.arange(2000)+1.) - 0.5*nref10_cell
    #hist_bins = np.concatenate((0.5*nref10_cell, hist_bins), axis=None)
    for i in np.arange(7)+1.:
        axtop.plot([0.5 * nref10_cell*2.**i, nref10_cell*2.**i],[0,2000],':',lw=4, color='grey',zorder=0)
        axbot.plot([0.5 * nref10_cell*2.**i, nref10_cell*2.**i],[0,2000],':',lw=4, color='grey',zorder=0)
    for i in np.arange(2)+2.:
        axtop.plot([0.5 * nref10_cell*2.**i, nref10_cell*2.**i],[0,2000],':', lw=4, color='grey',zorder=0)
        axbot.plot([0.5 * nref10_cell*2.**i, nref10_cell*2.**i],[0,2000],':', lw=4, color='grey',zorder=0)
    axtop.hist(natural_h1sizes, hist_bins, range=(0, 500), normed=True,edgecolor=hi_color,hatch='XX', lw=3,histtype='step',align='mid', label='H I')
    axtop.hist(natural_si2sizes, hist_bins, range=(0, 500), normed=True, edgecolor=si2_color, hatch='////', lw=3, histtype='step', align='mid',label='Si II')
    axtop.hist(natural_c4sizes, hist_bins, range=(0, 500), normed=True, edgecolor=c4_color, hatch='--', lw=3,histtype='step',align='mid',label='C IV')
    axtop.hist(natural_o6sizes, hist_bins, range=(0, 500), normed=True,edgecolor=ovi_color, hatch='\\\\', lw=3,histtype='step', align='mid',label='O VI')
    axtop.set_xscale('log')
    #axtop.set_xticks(())
    axtop.set_xticklabels(())
    axtop.set_yticks((0.2, 0.4, 0.6,0.8))
    axtop.set_yticklabels(('0.2','0.4','0.6','0.8'))
    axtop.grid(True)
    axtop.legend(loc='upper right')
    fig.text(0.11, 0.87, 'standard resolution', fontsize=32, ha='left', backgroundcolor='white')


    axbot.hist(nref10f_h1sizes, hist_bins, range=(0, 500), normed=True,edgecolor=hi_color,hatch='XX', lw=3,histtype='step',align='mid', label='H I',zorder=8)
    axbot.hist(nref10f_si2sizes, hist_bins, range=(0, 500), normed=True, edgecolor=si2_color, hatch='////', lw=3, histtype='step', align='mid',label='Si II',zorder=4)
    axbot.hist(nref10f_c4sizes, hist_bins, range=(0, 500), normed=True, edgecolor=c4_color, hatch='--', lw=3,histtype='step',align='mid',label='C IV',zorder=6)
    axbot.hist(nref10f_o6sizes, hist_bins, range=(0, 500), normed=True,edgecolor=ovi_color, hatch='\\\\', lw=3,histtype='step', align='mid',label='O VI',zorder=10)
    axbot.set_xscale('log')
    axbot.set_xticks((0.1, 1, 10))
    axbot.set_xticklabels(('0.1','1','10'))
    axbot.grid(True)
    fig.text(0.11, 0.46, 'high resolution', fontsize=32, ha='left', backgroundcolor='white')

    fig.text(0.5, 0.02, r'cloud size [$h^{-1}$ comoving kpc]', fontsize=34, ha='center')
    fig.text(0.02, 0.5, 'fraction of clouds', fontsize=34, va='center', rotation='vertical')

    # plt.tight_layout()
    plt.savefig(output_dir + 'cloud_size_histogram.png')
    plt.savefig(output_dir + 'cloud_size_histogram.pdf')



    ########################################################################
    #######  cumulative histogram of physical sizes ########################
    ########################################################################
    fig = plt.figure(figsize=(14,11))
    axtop = fig.add_axes([0.1, 0.52, 0.88, 0.42],
                   ylim=(0, 1.05), xlim=(0.1, 60))
    axbot = fig.add_axes([0.1, 0.1, 0.88, 0.42],
                   ylim=(0, 1.05), xlim=(0.1, 60))
    hist_bins = 0.5 * nref10_cell * (np.arange(2000)+1.)
    axtop.hist(natural_h1sizes_phys, hist_bins, range=(0, 500), cumulative=-1,normed=True,color=hi_color, histtype='step',lw=3, label='H I')
    axtop.hist(natural_si2sizes_phys, hist_bins, range=(0, 500), cumulative=-1,normed=True,color=si2_color, histtype='step',lw=3, label='Si II')
    axtop.hist(natural_c4sizes_phys, hist_bins, range=(0, 500), cumulative=-1,normed=True,color=c4_color, histtype='step',lw=3, label='C IV')
    axtop.hist(natural_o6sizes_phys, hist_bins, range=(0, 500), cumulative=-1,normed=True,color=ovi_color, histtype='step',lw=3, label='O VI')
    axtop.set_xscale('log')
    #axtop.set_xticks(())
    axtop.set_xticklabels(())
    axtop.set_yticks((0.2, 0.4, 0.6, 0.8, 1.0))
    axtop.set_yticklabels(('0.2','0.4','0.6','0.8','1.00'))
    axtop.grid(True)
    axtop.legend(loc='upper right')
    fig.text(0.73, 0.88, 'standard resolution', fontsize=32, ha='right')


    axbot.hist(nref10f_h1sizes_phys, hist_bins, range=(0, 500), cumulative=-1,normed=True,color=hi_color, histtype='step',lw=3, label='H I')
    axbot.hist(nref10f_si2sizes_phys, hist_bins, range=(0, 500), cumulative=-1,normed=True,color=si2_color, histtype='step',lw=3, label='Si II')
    axbot.hist(nref10f_c4sizes_phys, hist_bins, range=(0, 500), cumulative=-1,normed=True,color=c4_color, histtype='step',lw=3, label='C IV')
    axbot.hist(nref10f_o6sizes_phys, hist_bins, range=(0, 500), cumulative=-1,normed=True,color=ovi_color, histtype='step',lw=3, label='O VI')
    axbot.set_xscale('log')
    axbot.set_xticks((0.1, 1, 10))
    axbot.set_xticklabels(('0.1','1','10'))
    axbot.grid(True)
    fig.text(0.73, 0.45, 'high resolution', fontsize=32, ha='right')

    fig.text(0.5, 0.02, 'cloud size [physical kpc]', fontsize=34, ha='center')
    fig.text(0.02, 0.5, 'fraction of clouds with larger size', fontsize=34, va='center', rotation='vertical')

    # plt.tight_layout()
    plt.savefig(output_dir + 'cloud_size_physical_histogram.png')
    plt.savefig(output_dir + 'cloud_size_physical_histogram.pdf')




    ####################################################
    ####### histogram of numbers of cells ##############
    ####################################################
    fig = plt.figure(figsize=(14,11))
    axtop = fig.add_axes([0.1, 0.52, 0.88, 0.42],
                   ylim=(0, 1.05), xlim=(0, 200))
    axbot = fig.add_axes([0.1, 0.1, 0.88, 0.42],
                   ylim=(0, 1.05), xlim=(0, 200))
    axtop.hist(natural_h1_n_cells, 200, range=(0, 200), cumulative=-1,normed=True,edgecolor=hi_color,lw=3,histtype='step', label='H I')
    axtop.hist(natural_si2_n_cells, 200, range=(0, 200), cumulative=-1,normed=True, edgecolor=si2_color, lw=3, histtype='step', label='Si II')
    axtop.hist(natural_c4_n_cells, 200, range=(0, 200), cumulative=-1,normed=True, edgecolor=c4_color, lw=3,histtype='step',label='C IV')
    axtop.hist(natural_o6_n_cells, 200, range=(0, 200),cumulative=-1,normed=True, edgecolor=ovi_color, lw=3,histtype='step', label='O VI')
    #axtop.grid()
    #axtop.set_xticks(())
    axtop.set_xticklabels(())
    axtop.set_yticks((0.2, 0.4, 0.6, 0.8, 1.0))
    axtop.set_yticklabels(('0.2','0.4','0.6','0.8','1.0'))
    axtop.grid(True)
    axtop.legend(loc='upper right')
    fig.text(0.42, 0.88, 'standard resolution', fontsize=32, ha='center')


    axbot.hist(nref10f_h1_n_cells, 250, range=(0, 250), cumulative=-1,normed=True,edgecolor=hi_color,lw=3,histtype='step', label='H I')
    axbot.hist(nref10f_si2_n_cells, 250, range=(0, 250), cumulative=-1,normed=True, edgecolor=si2_color, lw=3, histtype='step', label='Si II')
    axbot.hist(nref10f_c4_n_cells, 250, range=(0, 250), cumulative=-1,normed=True, edgecolor=c4_color, lw=3,histtype='step',label='C IV')
    axbot.hist(nref10f_o6_n_cells, 250, range=(0, 250),cumulative=-1,normed=True, edgecolor=ovi_color, lw=3,histtype='step', label='O VI')
    #axbot.set_xscale('log')
    #axbot.set_yscale('log')
    #axbot.set_xticks((0.1, 1, 10))
    #axbot.set_xticklabels(('0.1','1','10'))
    axbot.grid(True)
    fig.text(0.42, 0.45, 'high resolution', fontsize=32, ha='center')

    fig.text(0.5, 0.02, 'number of cells giving 80 percent of column', fontsize=34, ha='center')
    fig.text(0.02, 0.5, 'fraction of sightlines with more cells', fontsize=34, va='center', rotation='vertical')

    # plt.tight_layout()
    plt.savefig(output_dir + 'cloud_cells_histogram.png')
    plt.savefig(output_dir + 'cloud_cells_histogram.pdf')

    ## calculate the fractions:
    h1f = np.array(nref10f_h1_n_cells)
    h1n = np.array(natural_h1_n_cells)
    print("natural HI cells fraction <= 5: ", len(h1n[h1n <= 5]) / len(h1n),"== 1: ",len(h1n[h1n == 1]) / len(h1n))
    print("refined HI cells fraction <= 5: ", len(h1f[h1f <= 5]) / len(h1f),"== 1: ",len(h1f[h1f == 1]) / len(h1f))
    si2f = np.array(nref10f_si2_n_cells)
    si2n = np.array(natural_si2_n_cells)
    print("natural SiII cells fraction <= 5: ", len(si2n[si2n <= 5]) / len(si2n),"== 1: ",len(si2n[si2n == 1]) / len(si2n))
    print("refined SiII cells fraction <= 5: ", len(si2f[si2f <= 5]) / len(si2f),"== 1: ",len(si2f[si2f == 1]) / len(si2f))
    c4f = np.array(nref10f_c4_n_cells)
    c4n = np.array(natural_c4_n_cells)
    print("natural CIV cells fraction <= 5: ", len(c4n[c4n <= 5]) / len(c4n),"== 1: ",len(c4n[c4n == 1]) / len(c4n))
    print("refined CIV cells fraction <= 5: ", len(c4f[c4f <= 5]) / len(c4f),"== 1: ",len(c4f[c4f == 1]) / len(c4f))
    o6f = np.array(nref10f_o6_n_cells)
    o6n = np.array(natural_o6_n_cells)
    print("natural OVI cells fraction <= 5: ", len(o6n[o6n <= 5]) / len(o6n),"== 1: ",len(o6n[o6n == 1]) / len(o6n))
    print("refined OVI cells fraction <= 5: ", len(o6f[o6f <= 5]) / len(o6f),"== 1: ",len(o6f[o6f == 1]) / len(o6f))



    #########################################
    ####### histogram of masses #############
    #########################################
    fig = plt.figure(figsize=(14,11))
    axtop = fig.add_axes([0.1, 0.52, 0.88, 0.42],
                 ylim=(0, 200), xlim=(0, 7.3))
    axbot = fig.add_axes([0.1, 0.1, 0.88, 0.42],
                   ylim=(0, 200), xlim=(0, 7.3))
    nbins = 100
    axtop.hist(np.log10(np.array(natural_o6masses)/1.989e33), nbins, range=(0,8), histtype='step', lw=3, edgecolor=ovi_color, hatch='\\\\', label='O VI',zorder=3)
    axtop.hist(np.log10(np.array(natural_c4masses)/1.989e33), nbins, range=(0,8), histtype='step',lw=3, edgecolor=c4_color, hatch='--', label='C IV',zorder=4)
    axtop.hist(np.log10(np.array(natural_si2masses)/1.989e33), nbins, range=(0,8), histtype='step',lw=3, edgecolor=si2_color, hatch='////', label='Si II',zorder=7)
    axtop.hist(np.log10(np.array(natural_h1masses)/1.989e33), nbins, range=(0,8), histtype='step', lw=3, edgecolor=hi_color,hatch='XX', label='H I',zorder=5)
    #n, bins, patches = plt.hist(np.array(si2masses)/1.989e33, 500, range=(0,10000), color=si2_color, histtype='step',lw=3, label=None)
    axtop.hist(np.log10(np.array(natural_o6masses)/1.989e33), nbins, range=(0,8), color=ovi_color, histtype='step',lw=3, label=None, zorder=9)
    #axtop.set_xticks(())
    axtop.set_yticks((50, 100, 150, 200))
    axtop.set_yticklabels(('50','100','150','200'))
    axtop.set_xticklabels(())
    axtop.grid(True)
    axtop.legend(loc='upper left')

    axbot.hist(np.log10(np.array(nref10f_o6masses)/1.989e33), nbins, range=(0,8), histtype='step', lw=3, edgecolor=ovi_color, hatch='\\\\', label='O VI',zorder=3)
    axbot.hist(np.log10(np.array(nref10f_c4masses)/1.989e33), nbins, range=(0,8), histtype='step',lw=3, edgecolor=c4_color, hatch='--', label='C IV',zorder=4)
    axbot.hist(np.log10(np.array(nref10f_si2masses)/1.989e33), nbins, range=(0,8), histtype='step',lw=3, edgecolor=si2_color, hatch='////', label='Si II',zorder=7)
    axbot.hist(np.log10(np.array(nref10f_h1masses)/1.989e33), nbins, range=(0,8), histtype='step', lw=3, edgecolor=hi_color,hatch='XX', label='H I',zorder=5)
    #n, bins, patches = plt.hist(np.array(si2masses)/1.989e33, 500, range=(0,10000), color=si2_color, histtype='step',lw=3, label=None)
    axbot.hist(np.log10(np.array(nref10f_o6masses)/1.989e33), nbins, range=(0,8), color=ovi_color, histtype='step',lw=3, label=None, zorder=9)
    axbot.grid(True)
    axbot.set_xticks((1,2,3,4,5,6,7))
    axbot.set_xticklabels(('10','100','1000',r'10$^4$',r'10$^5$',r'10$^6$',r'10$^7$'))

    fig.text(0.95, 0.88, 'standard resolution', fontsize=32, ha='right')
    fig.text(0.95, 0.45, 'high resolution', fontsize=32, ha='right')

    fig.text(0.5, 0.02, r'summed mass of cells along individual clouds [M$_{\odot}$]', fontsize=34, ha='center')
    fig.text(0.02, 0.5, 'number of clouds', fontsize=34, va='center', rotation='vertical')

    plt.savefig(output_dir + 'cloud_masses_histogram.png')
    plt.savefig(output_dir + 'cloud_masses_histogram.pdf')

    ## calculate the fractions:
    o6f = np.array(nref10f_o6masses)/1.989e33
    print("OVI fraction < 100: ", len(o6f[o6f < 100]) / len(o6f),"< 1000: ",len(o6f[o6f < 1000]) / len(o6f))


if __name__ == "__main__":
    plot_cloud_size_and_masses()
