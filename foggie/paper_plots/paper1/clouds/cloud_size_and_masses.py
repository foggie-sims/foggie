import glob
import os
import pickle
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
sns.set_style("whitegrid", {'axes.grid' : False})
mpl.rcParams['font.family'] = 'stixgeneral'
mpl.rcParams['font.size'] = 26.

def plot_cloud_size_and_masses():
    h1_limit = 1.e13
    si2_limit = 1.e11
    c4_limit = 1.e12
    o6_limit = 1.e13

    hi_color = 'salmon' ## '#984ea3' # purple
    ovi_color = '#4daf4a'  # green
    si2_color = '#984ea3' # 'darkorange'
    c4_color = "#4575b4" # blue 'darkorange'

    ### this will only work in python 3 !!
    filelist = glob.glob(os.path.join('.', '*vjt*.pkl'))
    # filelist = filelist[0:100]
    print('there are ',np.size(filelist),'files')
    size_dict = pickle.load( open( filelist[0], "rb" ) )
    print(size_dict.keys())

    h1sizes = []
    h1_n_cells = []
    h1masses = []
    h1columns = []
    h1cloudcolumns = []

    o6sizes = []
    o6_n_cells = []
    o6masses = []
    o6columns = []
    o6cloudcolumns = []

    si2sizes = []
    si2_n_cells = []
    si2masses = []
    si2columns = []
    si2cloudcolumns = []

    c4sizes = []
    c4_n_cells = []
    c4masses = []
    c4columns = []
    c4cloudcolumns = []

    nref10_cell = 1000. * 100 / (256 * np.power(2,10)) ## ckpc/h
    hist_bins = 0.5 * nref10_cell * (np.arange(2000)+1.)

    for file in filelist:

        size_dict = pickle.load( open( file, "rb" ) )

        ### make this limit different for each ion
        if (size_dict['nh1'] > h1_limit):
            if np.max(size_dict['h1_kpcsizes']) < 200:
                for item in size_dict['h1_kpcsizes']: h1sizes.append(item)
                for item in size_dict['h1_cell_masses']: h1masses.append(item)
                for item in size_dict['h1_coldens']: h1cloudcolumns.append(item)
                h1_n_cells.append(size_dict['h1_n_cells'])
                h1columns.append(size_dict['nh1'])

        if (size_dict['nsi2'] > si2_limit):
            for item in size_dict['si2_kpcsizes']: si2sizes.append(item)
            for item in size_dict['si2_cell_masses']: si2masses.append(item)
            for item in size_dict['si2_coldens']: si2cloudcolumns.append(item)
            si2_n_cells.append(size_dict['si2_n_cells'])
            si2columns.append(size_dict['nsi2'])

        if (size_dict['no6'] > o6_limit):
            if np.max(size_dict['o6_kpcsizes']) < 200:
                for item in size_dict['o6_kpcsizes']: o6sizes.append(item)
                for item in size_dict['o6_cell_masses']: o6masses.append(item)
                for item in size_dict['o6_coldens']: o6cloudcolumns.append(item)
                o6_n_cells.append(size_dict['o6_n_cells'])
                o6columns.append(size_dict['no6'])

        if (size_dict['nc4'] > c4_limit):
            if np.max(size_dict['c4_kpcsizes']) < 200:
                for item in size_dict['c4_kpcsizes']: c4sizes.append(item)
                for item in size_dict['c4_cell_masses']: c4masses.append(item)
                for item in size_dict['c4_coldens']: c4cloudcolumns.append(item)
                c4_n_cells.append(size_dict['c4_n_cells'])
                c4columns.append(size_dict['nc4'])

    ####################################################
    ####### cumulative histogram of sizes ##############
    ####################################################
    # fig = plt.figure(figsize=(14,7))
    fig = plt.figure(figsize=(12,14))
    ax = fig.add_subplot(111)
    hist_bins = 0.5 * nref10_cell * (np.arange(2000)+1.)
    for i in np.arange(5)+1.:
        plt.plot([nref10_cell*2.**i, nref10_cell*2.**i],[0,2000],'--', color='grey')
    for i in np.arange(2)+2.:
        plt.plot([nref10_cell*2.**i, nref10_cell*2.**i],[0,2000],'--', color='grey')
    plt.plot([nref10_cell, nref10_cell],[0,2000],'--', color='grey')
    #n, bins, patches = plt.hist(o6sizes, hist_bins, range=(0, 100), cumulative=-1, facecolor=ovi_color, alpha=0.75, label='O VI')
    #n, bins, patches = plt.hist(si2sizes, hist_bins, range=(0, 100), cumulative=-1, facecolor=si2_color, alpha=0.75, label='Si II')
    #n, bins, patches = plt.hist(h1sizes, hist_bins, range=(0, 100), cumulative=-1, facecolor=hi_color, alpha=0.75, label='H I')
    ax.hist(h1sizes, hist_bins, range=(0, 500), cumulative=-1,normed=True,color=hi_color, histtype='step',lw=3, label='H I')
    ax.hist(si2sizes, hist_bins, range=(0, 500), cumulative=-1,normed=True,color=si2_color, histtype='step',lw=3, label='Si II')
    ax.hist(c4sizes, hist_bins, range=(0, 500), cumulative=-1,normed=True,color=c4_color, histtype='step',lw=3, label='C IV')
    ax.hist(o6sizes, hist_bins, range=(0, 500), cumulative=-1,normed=True,color=ovi_color, histtype='step',lw=3, label='O VI')

    plt.xlabel('Cloud size [kpc]', fontsize=22)
    plt.ylabel('Fraction of clouds with larger size', fontsize=22)
    plt.axis([0.1, 60, 0, 1.05])
    #plt.text(13, 53, 'N$_{los}$ = '+str(np.size(o6columns)), fontsize='x-large')
    #plt.text(13, 46, 'N$_{clouds}(H I)$ = '+str(np.size(h1sizes)), fontsize='large')
    #plt.text(13, 42, 'N$_{clouds}(O VI)$ = '+str(np.size(o6sizes)), fontsize='large')
    plt.xscale('log')
    ax.set_xticks((0.1, 1, 10))
    ax.set_xticklabels(('0.1','1','10'))
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('cloud_size_cumulative_histogram.png')
    plt.savefig('cloud_size_cumulative_histogram.pdf')

    #########################################
    ####### histogram of sizes ##############
    #########################################
    # fig = plt.figure(figsize=(14,7))
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)
    hist_bins = nref10_cell * (np.arange(2000)+1.)
    hist_bins = np.concatenate((0.5*nref10_cell, hist_bins), axis=None)
    for i in np.arange(7)+1.:
        plt.plot([0.5*nref10_cell*2.**i, nref10_cell*2.**i],[0,2000],'--', color='grey')
    for i in np.arange(2)+2.:
        plt.plot([nref10_cell*2.**i, nref10_cell*2.**i],[0,2000],'--', color='grey')
    plt.plot([nref10_cell, nref10_cell],[0,2000],'--', color='grey')
    #n, bins, patches = plt.hist(o6sizes, hist_bins, range=(0, 100), cumulative=-1, facecolor=ovi_color, alpha=0.75, label='O VI')
    #n, bins, patches = plt.hist(si2sizes, hist_bins, range=(0, 100), cumulative=-1, facecolor=si2_color, alpha=0.75, label='Si II')
    #n, bins, patches = plt.hist(h1sizes, hist_bins, range=(0, 100), cumulative=-1, facecolor=hi_color, alpha=0.75, label='H I')
    ax.hist(h1sizes, hist_bins, range=(0, 500), normed=True,edgecolor=hi_color,hatch='XX', lw=2,histtype='step',align='left', label='H I')
    ax.hist(si2sizes, hist_bins, range=(0, 500), normed=True, edgecolor=si2_color, hatch='////', lw=2, histtype='step', align='left',label='Si II')
    ax.hist(c4sizes, hist_bins, range=(0, 500), normed=True, edgecolor=c4_color, hatch='--', histtype='step',align='left',label='C IV')
    ax.hist(o6sizes, hist_bins, range=(0, 500), normed=True,edgecolor=ovi_color, hatch='\\\\', histtype='step', align='left',label='O VI')

    plt.xlabel('Cloud size [kpc]', fontsize=26)
    plt.ylabel('Fraction of clouds', fontsize=26)
    plt.axis([0.1, 60, 0, 1.05])
    #plt.text(13, 53, 'N$_{los}$ = '+str(np.size(o6columns)), fontsize='x-large')
    #plt.text(13, 46, 'N$_{clouds}(H I)$ = '+str(np.size(h1sizes)), fontsize='large')
    #plt.text(13, 42, 'N$_{clouds}(O VI)$ = '+str(np.size(o6sizes)), fontsize='large')
    plt.xscale('log')
    ax.set_xticks((0.1, 1, 10))
    ax.set_xticklabels(('0.1','1','10'),fontsize=24)
    plt.grid(True)
    plt.legend(loc='upper right',fontsize=26)
    plt.tight_layout()
    plt.savefig('cloud_size_histogram.png')
    plt.savefig('cloud_size_histogram.pdf')

    ####################################################
    ####### histogram of numbers of cells ##############
    ####################################################
    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111)
    ax.hist(h1_n_cells, 100, range=(0,200), histtype='step', edgecolor=hi_color,hatch='XX', lw=2,label='H I',zorder=5)
    ax.hist(si2_n_cells, 100, range=(0,200), histtype='step', edgecolor=si2_color, hatch='////', lw=2, label='Si II',zorder=7)
    ax.hist(c4_n_cells, 100, range=(0,200), histtype='step',lw=2, edgecolor=c4_color, hatch='--', label='C IV',zorder=4)
    ax.hist(o6_n_cells, 100, range=(0,200), histtype='step', edgecolor=ovi_color, hatch='\\\\',lw=2, label='O VI',zorder=3)
    #n, bins, patches = plt.hist(o6_n_cells, 100, range=(0,200), facecolor=ovi_color, alpha=0.75, label='O VI')
    #n, bins, patches = plt.hist(si2_n_cells, 100, range=(0,200), facecolor=si2_color, alpha=0.75, label='Si II')
    #n, bins, patches = plt.hist(h1_n_cells, 100, range=(0,200), facecolor=hi_color, alpha=0.75, label='H I')
    #n, bins, patches = plt.hist(si2_n_cells, 100, range=(0,200), color=si2_color, histtype='step',lw=2, label=None)
    #n, bins, patches = plt.hist(h1_n_cells, 100, range=(0,200), color=hi_color, histtype='step',lw=2, label=None)
    #n, bins, patches = plt.hist(o6_n_cells, 100, range=(0,200), color=ovi_color, histtype='step',lw=2, label=None)
    plt.xlabel('Number of cells giving 80 percent of column',fontsize=22)
    plt.ylabel('Number of sightlines',fontsize=22)
    #plt.title(r'Cell distribution [natural]')
    plt.axis([0, 200, 0, 200])
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('cloud_cells_histogram.png')
    plt.savefig('cloud_cells_histogram.pdf')


    #########################################
    ####### histogram of masses #############
    #########################################
    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111)
    nbins = 100
    ax.hist(np.log10(np.array(o6masses)/1.989e33), nbins, range=(0,7), histtype='step', lw=2, edgecolor=ovi_color, hatch='\\\\', label='O VI',zorder=3)
    ax.hist(np.log10(np.array(c4masses)/1.989e33), nbins, range=(0,7), histtype='step',lw=2, edgecolor=c4_color, hatch='--', label='C IV',zorder=4)
    ax.hist(np.log10(np.array(si2masses)/1.989e33), nbins, range=(0,7), histtype='step',lw=2, edgecolor=si2_color, hatch='////', label='Si II',zorder=7)
    ax.hist(np.log10(np.array(h1masses)/1.989e33), nbins, range=(0,7), histtype='step', lw=2, edgecolor=hi_color,hatch='XX', label='H I',zorder=5)
    #n, bins, patches = plt.hist(np.array(si2masses)/1.989e33, 500, range=(0,10000), color=si2_color, histtype='step',lw=2, label=None)
    ax.hist(np.log10(np.array(o6masses)/1.989e33), nbins, range=(0,7), color=ovi_color, histtype='step',lw=2, label=None, zorder=9)
    #n, bins, patches = plt.hist(np.array(h1masses)/1.989e33, 500, range=(0,10000), color=hi_color, histtype='step',lw=2, label=None)
    plt.xlabel(r'Summed mass of cells along individual clouds [M$_{\odot}$]',fontsize=22)
    plt.ylabel('Number of clouds',fontsize=22)
    # plt.title('Mass of cells in each identified "cloud" [normal]')
    plt.axis([1,5.3, 0, 200])
    #plt.xlim(10,1e5)
    ##plt.xscale('log')
    ax.set_xticks((1,2,3,4,5,6,7))
    ax.set_xticklabels(('10','100','1000',r'10$^4$',r'10$^5$',r'10$^6$',r'10$^7$'))
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('cloud_masses_histogram.png')
    plt.savefig('cloud_masses_histogram.pdf')

    #########################################
    ####### 3d masses and sizes #############
    #########################################
    fig = plt.figure(figsize=(14,11))
    ax = fig.add_subplot(111)

    h1_cloud_ncells = np.array(h1sizes) / np.array(nref10_cell)
    si2_cloud_ncells = np.array(si2sizes) / np.array(nref10_cell)
    c4_cloud_ncells = np.array(c4sizes) / np.array(nref10_cell)
    o6_cloud_ncells = np.array(o6sizes) / np.array(nref10_cell)
    h1_implied_spherical_masses = (np.pi/6) * (h1_cloud_ncells**2) * np.array(h1masses)/1.989e33 / 2.
    si2_implied_spherical_masses = (np.pi/6) * (si2_cloud_ncells**2) * np.array(si2masses)/1.989e33 / 2.
    c4_implied_spherical_masses = (np.pi/6) * (c4_cloud_ncells**2) * np.array(c4masses)/1.989e33 / 2.
    o6_implied_spherical_masses = (np.pi/6) * (o6_cloud_ncells**2) * np.array(o6masses)/1.989e33 / 2.
    h1_implied_ncells = (np.pi/6) * ((np.random.normal(0,0.25,size=len(h1_cloud_ncells)) + h1_cloud_ncells)**2)
    si2_implied_ncells = (np.pi/6) * ((np.random.normal(0,0.25,size=len(si2_cloud_ncells)) + si2_cloud_ncells)**2)
    c4_implied_ncells = (np.pi/6) * ((np.random.normal(0,0.25,size=len(c4_cloud_ncells)) + c4_cloud_ncells)**2)
    o6_implied_ncells = (np.pi/6) * ((np.random.normal(0,0.25,size=len(o6_cloud_ncells)) + o6_cloud_ncells)**2)

    ax.scatter(np.log10(h1_implied_ncells), np.log10(h1_implied_spherical_masses), marker='*', alpha=0.6, color=hi_color, label='H I',zorder=10)
    ax.scatter(np.log10(si2_implied_ncells), np.log10(si2_implied_spherical_masses), marker='s', alpha=0.4, color=si2_color, label='Si II',zorder=10)
    ax.scatter(np.log10(c4_implied_ncells), np.log10(c4_implied_spherical_masses), marker='D', alpha=0.4, color=c4_color, label='C IV',zorder=5)
    ax.scatter(np.log10(o6_implied_ncells), np.log10(o6_implied_spherical_masses), marker='o', alpha=0.5, color=ovi_color, label='O VI',zorder=3)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.axis([0.1,4,0.5,8.2])
    ax.set_xticks((0,1,2,3,4))
    ax.set_xticklabels((r'1',r'10',r'100',r'1000',r'10$^4$'))
    ax.set_yticks((1,2,3,4,5,6,7,8))
    ax.set_yticklabels((r'10',r'100',r'1000',r'10$^4$',r'10$^5$',r'10$^6$',r'10$^7$',r'10$^8$'))
    plt.xlabel(r'number of cells in implied 3D cloud', fontsize=34)
    plt.ylabel(r'total mass of implied 3D cloud', fontsize=34)
    plt.legend(loc='lower right', handletextpad=0.1, markerscale=2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('cloud_3d.png')
    plt.savefig('cloud_3d.pdf')




if __name__ == "__main__":
    plot_cloud_size_and_masses()
