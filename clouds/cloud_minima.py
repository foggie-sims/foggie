import glob
import os
import pickle
import numpy as np
from astropy.io import fits

#import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
#sns.set_style("whitegrid", {'axes.grid' : False})
mpl.rcParams['font.family'] = 'stixgeneral'
mpl.rcParams['font.size'] = 20.

def plot_cloud_minima():
    h1_limit = 1.e13
    si2_limit = 1.e12
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
    natural_h1_nmin = []

    natural_o6sizes = []
    natural_o6sizes_phys = []
    natural_o6_n_cells = []
    natural_o6masses = []
    natural_o6columns = []
    natural_o6cloudcolumns = []
    natural_o6_nmin = []

    natural_si2sizes = []
    natural_si2sizes_phys = []
    natural_si2_n_cells = []
    natural_si2masses = []
    natural_si2columns = []
    natural_si2cloudcolumns = []
    natural_si2_nmin = []

    natural_c4sizes = []
    natural_c4sizes_phys = []
    natural_c4_n_cells = []
    natural_c4masses = []
    natural_c4columns = []
    natural_c4cloudcolumns = []
    natural_c4_nmin = []

    natural_si4sizes = []
    natural_si4sizes_phys = []
    natural_si4_n_cells = []
    natural_si4masses = []
    natural_si4columns = []
    natural_si4cloudcolumns = []
    natural_si4_nmin = []

    for file in filelist:
        natural_size_dict = pickle.load( open( file, "rb" ) )
        if "rd0020" in file:
            z = 2.0
        if "rd0018" in file:
            z = 2.5
        fileroot = file.strip('_sizes.pkl')
        lsffile = fileroot + '_lsf.fits.gz'
        ## print('opening ',lsffile)
        try:
            hdulist = fits.open(lsffile)
        except:
            print('opening ',lsffile, ' soooo did not work')
            continue
        ### make this limit different for each ion
        if (natural_size_dict['nh1'] > h1_limit):
            if np.max(natural_size_dict['h1_kpcsizes']) < 200:
                key = 'H I 1216'
                for item in natural_size_dict['h1_kpcsizes']: natural_h1sizes.append(item)
                for item in natural_size_dict['h1_kpcsizes']: natural_h1sizes_phys.append(item/(1+z))
                for item in natural_size_dict['h1_cell_masses']: natural_h1masses.append(item)
                for item in natural_size_dict['h1_coldens']: natural_h1cloudcolumns.append(item)
                natural_h1_n_cells.append(natural_size_dict['h1_n_cells'])
                natural_h1columns.append(natural_size_dict['nh1'])
                if key in hdulist:
                    natural_h1_nmin.append(hdulist[key].header['Nmin'])
                else:
                    natural_h1_nmin.append(0)

        if (natural_size_dict['nsi2'] > si2_limit):
            key = 'Si II 1260'
            for item in natural_size_dict['si2_kpcsizes']: natural_si2sizes.append(item)
            for item in natural_size_dict['si2_kpcsizes']: natural_si2sizes_phys.append(item/(1+z))
            for item in natural_size_dict['si2_cell_masses']: natural_si2masses.append(item)
            for item in natural_size_dict['si2_coldens']: natural_si2cloudcolumns.append(item)
            natural_si2_n_cells.append(natural_size_dict['si2_n_cells'])
            natural_si2columns.append(natural_size_dict['nsi2'])
            if key in hdulist:
                natural_si2_nmin.append(hdulist[key].header['Nmin'])
            else:
                natural_si2_nmin.append(0)

        if (natural_size_dict['no6'] > o6_limit):
            if np.max(natural_size_dict['o6_kpcsizes']) < 200:
                for item in natural_size_dict['o6_kpcsizes']: natural_o6sizes.append(item)
                for item in natural_size_dict['o6_kpcsizes']: natural_o6sizes_phys.append(item/(1+z))
                for item in natural_size_dict['o6_cell_masses']: natural_o6masses.append(item)
                for item in natural_size_dict['o6_coldens']: natural_o6cloudcolumns.append(item)
                natural_o6_n_cells.append(natural_size_dict['o6_n_cells'])
                natural_o6columns.append(natural_size_dict['no6'])

        if (natural_size_dict['nc4'] > c4_limit):
            if np.max(natural_size_dict['c4_kpcsizes']) < 200:
                for item in natural_size_dict['c4_kpcsizes']: natural_c4sizes.append(item)
                for item in natural_size_dict['c4_kpcsizes']: natural_c4sizes_phys.append(item/(1+z))
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
                for item in nref10f_size_dict['h1_kpcsizes']: nref10f_h1sizes_phys.append(item/(1+z))
                for item in nref10f_size_dict['h1_cell_masses']: nref10f_h1masses.append(item)
                for item in nref10f_size_dict['h1_coldens']: nref10f_h1cloudcolumns.append(item)
                nref10f_h1_n_cells.append(nref10f_size_dict['h1_n_cells'])
                nref10f_h1columns.append(nref10f_size_dict['nh1'])

        if (nref10f_size_dict['nsi2'] > si2_limit):
            for item in nref10f_size_dict['si2_kpcsizes']: nref10f_si2sizes.append(item)
            for item in nref10f_size_dict['si2_kpcsizes']: nref10f_si2sizes_phys.append(item/(1+z))
            for item in nref10f_size_dict['si2_cell_masses']: nref10f_si2masses.append(item)
            for item in nref10f_size_dict['si2_coldens']: nref10f_si2cloudcolumns.append(item)
            nref10f_si2_n_cells.append(nref10f_size_dict['si2_n_cells'])
            nref10f_si2columns.append(nref10f_size_dict['nsi2'])

        if (nref10f_size_dict['no6'] > o6_limit):
            if np.max(nref10f_size_dict['o6_kpcsizes']) < 200:
                for item in nref10f_size_dict['o6_kpcsizes']: nref10f_o6sizes.append(item)
                for item in nref10f_size_dict['o6_kpcsizes']: nref10f_o6sizes_phys.append(item/(1+z))
                for item in nref10f_size_dict['o6_cell_masses']: nref10f_o6masses.append(item)
                for item in nref10f_size_dict['o6_coldens']: nref10f_o6cloudcolumns.append(item)
                nref10f_o6_n_cells.append(nref10f_size_dict['o6_n_cells'])
                nref10f_o6columns.append(nref10f_size_dict['no6'])

        if (nref10f_size_dict['nc4'] > c4_limit):
            if np.max(nref10f_size_dict['c4_kpcsizes']) < 200:
                for item in nref10f_size_dict['c4_kpcsizes']: nref10f_c4sizes.append(item)
                for item in nref10f_size_dict['c4_kpcsizes']: nref10f_c4sizes_phys.append(item/(1+z))
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





    ##############################################################
    ####### number of cells versus number of minima ##############
    ##############################################################
    fig = plt.figure(figsize=(14,9))
    axtop = fig.add_axes([0.1, 0.52, 0.88, 0.42])
    axbot = fig.add_axes([0.1, 0.1, 0.88, 0.42])
    axbot.scatter(natural_si2_n_cells, natural_si2_nmin, color=si2_color,  label='Si II')
    axbot.plot([0,20],[0,20])
    #print(np.array(natural_si2_n_cells)/np.array/(natural_si2_nmin))
    #axbot.hist((np.array(natural_si2_n_cells)/np.array/(natural_si2_nmin)), 20,color=si2_color,  label='Si II')
    # plt.tight_layout()
    plt.savefig(output_dir + 'clouds_cells_nmin.png')

    ## calculate the fractions:
    si2n = np.array(natural_si2_n_cells)
    si2f = np.array(nref10f_si2_n_cells)
    print("Natural SiII fraction = 1: ", len(si2n[si2n < 2]) / len(si2n),"<= 5: ",len(si2n[si2n < 6]) / len(si2n))
    print("Forced SiII fraction = 1: ", len(si2f[si2f == 1]) / len(si2f),"<= 5: ",len(si2f[si2f <= 5]) / len(si2f))


if __name__ == "__main__":
    plot_cloud_minima()
