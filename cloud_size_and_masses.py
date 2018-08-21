import glob
import os
import pickle
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
sns.set_style("whitegrid", {'axes.grid' : False})
mpl.rcParams['font.family'] = 'stixgeneral'
mpl.rcParams['font.size'] = 16.

def plot_cloud_size_and_masses():
    h1_limit = 1.e13
    si2_limit = 1.e12
    c4_limit = 1.e12
    o6_limit = 1.e13

    hi_color = '#984ea3' # purple
    ovi_color = '#4daf4a'  # green
    si2_color = 'darkorange'



    ### this will only work in python 3 !!
    filelist = glob.glob(os.path.join('.', '*.pkl'))
    # filelist = filelist[0:100]
    print('there are ',np.size(filelist),'files')
    size_dict = pickle.load( open( filelist[0], "rb" ) )
    print(size_dict.keys())

    h1sizes = []
    h1_n_cells = []
    h1masses = []
    h1columns = []

    o6sizes = []
    o6_n_cells = []
    o6masses = []
    o6columns = []

    si2sizes = []
    si2_n_cells = []
    si2masses = []
    si2columns = []

    nref10_cell = 1000 * 100 / (256 * np.power(2,10)) ## ckpc/h
    hist_bins = nref10_cell * (np.arange(1000)+1.)

    for file in filelist:

        size_dict = pickle.load( open( file, "rb" ) )
        ### make this limit different for each ion
        if (size_dict['nh1'] > h1_limit):
for file in filelist:

    size_dict = pickle.load( open( file, "rb" ) )

    ### make this limit different for each ion
    if (size_dict['nh1'] > h1_limit):
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
        for item in size_dict['o6_kpcsizes']: o6sizes.append(item)
        for item in size_dict['o6_cell_masses']: o6masses.append(item)
        for item in size_dict['o6_coldens']: o6cloudcolumns.append(item)
        o6_n_cells.append(size_dict['o6_n_cells'])
        o6columns.append(size_dict['no6'])

    for item in size_dict['c4_coldens']: c4cloudcolumns.append(item)

    #########################################
    ####### histogram of sizes ##############
    #########################################
    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111)


    for i in np.arange(5)+1.:
        plt.plot([nref10_cell*2.**i, nref10_cell*2.**i],[0,2000],'--', color='grey')
    for i in np.arange(2)+2.:
        plt.plot([nref10_cell*2.**i, nref10_cell*2.**i],[0,2000],'--', color='grey')
    plt.plot([nref10_cell, nref10_cell],[0,2000],'--', color='grey')


    #n, bins, patches = plt.hist(o6sizes, hist_bins, range=(0, 100), cumulative=-1, facecolor=ovi_color, alpha=0.75, label='O VI')
    #n, bins, patches = plt.hist(si2sizes, hist_bins, range=(0, 100), cumulative=-1, facecolor=si2_color, alpha=0.75, label='Si II')
    #n, bins, patches = plt.hist(h1sizes, hist_bins, range=(0, 100), cumulative=-1, facecolor=hi_color, alpha=0.75, label='H I')
    n, bins, patches = plt.hist(h1sizes, hist_bins, range=(0, 500), cumulative=-1,normed=True,color=hi_color, histtype='step',lw=3, label='H I')
    n, bins, patches = plt.hist(si2sizes, hist_bins, range=(0, 500), cumulative=-1,normed=True,color=si2_color, histtype='step',lw=3, label='Si II')
    n, bins, patches = plt.hist(o6sizes, hist_bins, range=(0, 500), cumulative=-1,normed=True,color=ovi_color, histtype='step',lw=3, label='O VI')

    plt.xlabel('Sizes [kpc]')
    plt.ylabel('Fraction of clouds with larger size')
    plt.axis([0.1, 100, 0, 1.05])
    #plt.text(13, 53, 'N$_{los}$ = '+str(np.size(o6columns)), fontsize='x-large')
    #plt.text(13, 46, 'N$_{clouds}(H I)$ = '+str(np.size(h1sizes)), fontsize='large')
    #plt.text(13, 42, 'N$_{clouds}(O VI)$ = '+str(np.size(o6sizes)), fontsize='large')
    plt.xscale('log')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.savefig('cloud_size_histogram.png')
    plt.savefig('cloud_size_histogram.pdf')

    ####################################################
    ####### histogram of numbers of cells ##############
    ####################################################
    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111)

    n, bins, patches = plt.hist(o6_n_cells, 100, range=(0,200), facecolor=ovi_color, alpha=0.75, label='O VI')
    n, bins, patches = plt.hist(si2_n_cells, 100, range=(0,200), facecolor=si2_color, alpha=0.75, label='Si II')
    n, bins, patches = plt.hist(h1_n_cells, 100, range=(0,200), facecolor=hi_color, alpha=0.75, label='H I')
    n, bins, patches = plt.hist(si2_n_cells, 100, range=(0,200), color=si2_color, histtype='step',lw=2, label=None)
    n, bins, patches = plt.hist(h1_n_cells, 100, range=(0,200), color=hi_color, histtype='step',lw=2, label=None)
    n, bins, patches = plt.hist(o6_n_cells, 100, range=(0,200), color=ovi_color, histtype='step',lw=2, label=None)
    plt.xlabel('Number of cells giving 80 percent of column')
    plt.ylabel('N')
    #plt.title(r'Cell distribution [natural]')
    plt.axis([0, 200, 0, 35])
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig('cloud_cells_histogram.png')
    plt.savefig('cloud_cells_histogram.pdf')


    #########################################
    ####### histogram of masses #############
    #########################################
    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111)
    n, bins, patches = plt.hist(np.array(o6masses)/1.989e33, 500, range=(0,10000), facecolor=ovi_color, alpha=0.75, label='O VI')
    n, bins, patches = plt.hist(np.array(si2masses)/1.989e33, 500, range=(0,10000), facecolor=si2_color, alpha=0.75, label='Si II')
    n, bins, patches = plt.hist(np.array(h1masses)/1.989e33, 500, range=(0,10000), alpha=0.75, facecolor=hi_color, label='H I')
    n, bins, patches = plt.hist(np.array(si2masses)/1.989e33, 500, range=(0,10000), color=si2_color, histtype='step',lw=2, label=None)
    n, bins, patches = plt.hist(np.array(o6masses)/1.989e33, 500, range=(0,10000), color=ovi_color, histtype='step',lw=2, label=None)
    n, bins, patches = plt.hist(np.array(h1masses)/1.989e33, 500, range=(0,10000), color=hi_color, histtype='step',lw=2, label=None)
    plt.xlabel(r'Mass of cells along clouds [M$_{\odot}$]')
    plt.ylabel('Number of clouds')
    # plt.title('Mass of cells in each identified "cloud" [normal]')
    plt.axis([10, 1e5, 0, 40])
    #plt.xlim(10,1e5)
    plt.xscale('log')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig('cloud_masses_histogram.png')
    plt.savefig('cloud_masses_histogram.pdf')

if __name__ == "__main__":
    plot_cloud_size_and_masses()
