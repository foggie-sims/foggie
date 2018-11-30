import glob
import os
import pickle
import numpy as np

import astropy.units as u
from astropy.constants import k_B, m_p
from astropy.table import Table
from astropy.io import ascii
import pandas

#import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
#sns.set_style("whitegrid", {'axes.grid' : False})
mpl.rcParams['font.family'] = 'stixgeneral'
mpl.rcParams['font.size'] = 26.

h=0.695  # damn you little h!!
sigma_ratio = 2.0

def plot_cloud_size_and_masses():
    h1_limit = 1.e13
    si2_limit = 1.e11
    c4_limit = 1.e12
    si4_limit = 1.e13
    o6_limit = 1.e13

    simass = 28.0855
    cmass = 12.0107
    omass = 15.999

    hi_color = 'salmon' ## '#984ea3' # purple
    ovi_color = '#4daf4a'  # green
    si2_color = '#984ea3' # 'darkorange'
    c4_color = "#4575b4" # blue 'darkorange'
    si4_color = "#4575b4" #'darkorange'

    nref10_cell = 1000. * 100 / (256 * np.power(2,10)) ## ckpc/h

    output_dir = '/Users/molly/Dropbox/foggie-collab/papers/absorption_peeples/Figures/appendix/'

    ### this will only work in python 3 !!
    filelist = glob.glob(os.path.join('/Users/molly/Dropbox/foggie-collab/plots_halo_008508/nref11n/natural/spectra/lls', '*rd0018*.pkl'))
    # filelist = filelist[0:100]
    print('there are ',np.size(filelist),'files')
    natural_size_dict = pickle.load( open( filelist[0], "rb" ) )
    print(natural_size_dict.keys())

    natural_h1temps = []
    natural_h1vc = []
    natural_h1dv = []
    natural_h1cloudcolumns = []
    natural_h1_cloud_cells = []

    natural_o6temps = []
    natural_o6vc = []
    natural_o6dv = []
    natural_o6cloudcolumns = []
    natural_o6_cloud_cells = []

    natural_si2temps = []
    natural_si2vc = []
    natural_si2dv = []
    natural_si2cloudcolumns = []
    natural_si2_cloud_cells = []

    natural_c4temps = []
    natural_c4vc = []
    natural_c4dv = []
    natural_c4cloudcolumns = []
    natural_c4_cloud_cells = []

    natural_si4temps = []
    natural_si4vc = []
    natural_si4dv = []
    natural_si4cloudcolumns = []
    natural_si4_cloud_cells = []

    for file in filelist:
        natural_size_dict = pickle.load( open( file, "rb" ) )
        raydf = natural_size_dict['ray_df']
        if "rd0020" in file:
            z = 2.0
        if "rd0018" in file:
            z = 2.5
        if "axx" in file:
            this_velocity = 'x-velocity'
        elif "axy" in file:
            this_velocity = 'y-velocity'
        elif "axz" in file:
            this_velocity = 'z-velocity'
        ### make this limit different for each ion
        if (natural_size_dict['nh1'] > h1_limit):
            if np.max(natural_size_dict['h1_kpcsizes']) < 200:
                ## loop through the clouds
                for i, item in enumerate(natural_size_dict['h1_coldens']):
                    this_cloud = raydf[raydf['h1_cloud_flag'] == (i+1)]
                    m = m_p
                    dv = (np.max(this_cloud[this_velocity]) -
                          np.min(this_cloud[this_velocity])) * u.km / u.s
                    T = sum(this_cloud['temperature'] * this_cloud['H_p0_number_density']*this_cloud['dx'] / sum(this_cloud['H_p0_number_density']*this_cloud['dx'])) * u.K
                    vc = np.sqrt(k_B*T / m).to('km/s')
                    natural_h1temps.append(T.value)
                    natural_h1dv.append(dv.value)
                    natural_h1vc.append(vc.value)
                    natural_h1cloudcolumns.append(item)
                    natural_h1_cloud_cells.append(len(this_cloud))

        if (natural_size_dict['nsi2'] > si2_limit):
            if np.max(natural_size_dict['si2_kpcsizes']) < 200:
                ## loop through the clouds
                for i, item in enumerate(natural_size_dict['si2_coldens']):
                    this_cloud = raydf[raydf['si2_cloud_flag'] == (i+1)]
                    m = simass * m_p
                    dv = (np.max(this_cloud[this_velocity]) -
                          np.min(this_cloud[this_velocity])) * u.km / u.s
                    T = sum(this_cloud['temperature'] * this_cloud['Si_p1_number_density']*this_cloud['dx'] / sum(this_cloud['Si_p1_number_density']*this_cloud['dx'])) * u.K
                    vc = np.sqrt(k_B*T / m).to('km/s')
                    natural_si2temps.append(T.value)
                    natural_si2dv.append(dv.value)
                    natural_si2vc.append(vc.value)
                    natural_si2cloudcolumns.append(item)
                    natural_si2_cloud_cells.append(len(this_cloud))

        if (natural_size_dict['no6'] > o6_limit):
            if np.max(natural_size_dict['o6_kpcsizes']) < 200:
                ## loop through the clouds
                for i, item in enumerate(natural_size_dict['o6_coldens']):
                    this_cloud = raydf[raydf['o6_cloud_flag'] == (i+1)]
                    m = omass * m_p
                    dv = (np.max(this_cloud[this_velocity]) -
                          np.min(this_cloud[this_velocity])) * u.km / u.s
                    T = sum(this_cloud['temperature'] * this_cloud['O_p5_number_density']*this_cloud['dx'] / sum(this_cloud['O_p5_number_density']*this_cloud['dx'])) * u.K
                    vc = np.sqrt(k_B*T / m).to('km/s')
                    natural_o6temps.append(T.value)
                    natural_o6dv.append(dv.value)
                    natural_o6vc.append(vc.value)
                    natural_o6cloudcolumns.append(item)
                    natural_o6_cloud_cells.append(len(this_cloud))

        if (natural_size_dict['nc4'] > c4_limit):
            if np.max(natural_size_dict['c4_kpcsizes']) < 200:
                ## loop through the clouds
                for i, item in enumerate(natural_size_dict['c4_coldens']):
                    this_cloud = raydf[raydf['c4_cloud_flag'] == (i+1)]
                    m = cmass * m_p
                    dv = (np.max(this_cloud[this_velocity]) -
                          np.min(this_cloud[this_velocity])) * u.km / u.s
                    T = sum(this_cloud['temperature'] * this_cloud['C_p3_number_density']*this_cloud['dx'] / sum(this_cloud['C_p3_number_density']*this_cloud['dx'])) * u.K
                    vc = np.sqrt(k_B*T / m).to('km/s')
                    natural_c4temps.append(T.value)
                    natural_c4dv.append(dv.value)
                    natural_c4vc.append(vc.value)
                    natural_c4cloudcolumns.append(item)
                    natural_c4_cloud_cells.append(len(this_cloud))

    natural_h1_sampling = np.array(natural_h1dv) / (np.array(natural_h1vc))
    natural_si2_sampling = np.array(natural_si2dv) / (np.array(natural_si2vc))
    natural_c4_sampling =  np.array(natural_c4dv) / (np.array(natural_c4vc))
    natural_o6_sampling = np.array(natural_o6dv) / (np.array(natural_o6vc))
    natural_h1_ratio = np.array(natural_h1_sampling)  / ((np.array(natural_h1_cloud_cells) - 1)*sigma_ratio)
    natural_si2_ratio = np.array(natural_si2_sampling)  / ((np.array(natural_si2_cloud_cells) - 1)*sigma_ratio)
    natural_c4_ratio = np.array(natural_c4_sampling)  / ((np.array(natural_c4_cloud_cells) - 1)*sigma_ratio)
    natural_o6_ratio = np.array(natural_o6_sampling)  / ((np.array(natural_o6_cloud_cells) - 1)*sigma_ratio)
    natural_h1_ratio[natural_h1_ratio == np.nan] = 1e6
    natural_si2_ratio[natural_si2_ratio == np.nan] = 1e6
    natural_c4_ratio[natural_c4_ratio == np.nan] = 1e6
    natural_o6_ratio[natural_o6_ratio == np.nan] = 1e6
    natural_h1_ratio[natural_h1_ratio == np.inf] = 1e6
    natural_si2_ratio[natural_si2_ratio == np.inf] = 1e6
    natural_c4_ratio[natural_c4_ratio == np.inf] = 1e6
    natural_o6_ratio[natural_o6_ratio == np.inf] = 1e6


    filelist = glob.glob(os.path.join('/Users/molly/Dropbox/foggie-collab/plots_halo_008508/nref11n/nref11n_nref10f_refine200kpc/spectra/lls', '*.pkl'))
    # filelist = filelist[0:100]
    print('there are ',np.size(filelist),'files')
    nref10f_size_dict = pickle.load( open( filelist[0], "rb" ) )
    print(nref10f_size_dict.keys())


    nref10f_h1temps = []
    nref10f_h1vc = []
    nref10f_h1dv = []
    nref10f_h1cloudcolumns = []
    nref10f_h1_cloud_cells = []

    nref10f_o6temps = []
    nref10f_o6vc = []
    nref10f_o6dv = []
    nref10f_o6cloudcolumns = []
    nref10f_o6_cloud_cells = []

    nref10f_si2temps = []
    nref10f_si2vc = []
    nref10f_si2dv = []
    nref10f_si2cloudcolumns = []
    nref10f_si2_cloud_cells = []

    nref10f_c4temps = []
    nref10f_c4vc = []
    nref10f_c4dv = []
    nref10f_c4cloudcolumns = []
    nref10f_c4_cloud_cells = []

    nref10f_si4temps = []
    nref10f_si4vc = []
    nref10f_si4dv = []
    nref10f_si4cloudcolumns = []
    nref10f_si4_cloud_cells = []

    for file in filelist:
        nref10f_size_dict = pickle.load( open( file, "rb" ) )
        raydf = nref10f_size_dict['ray_df']
        if "rd0020" in file:
            z = 2.0
        if "rd0018" in file:
            z = 2.5
        if "axx" in file:
            this_velocity = 'x-velocity'
        elif "axy" in file:
            this_velocity = 'y-velocity'
        elif "axz" in file:
            this_velocity = 'z-velocity'
        ### make this limit different for each ion
        if (nref10f_size_dict['nh1'] > h1_limit):
            if np.max(nref10f_size_dict['h1_kpcsizes']) < 200:
                ## loop through the clouds
                for i, item in enumerate(nref10f_size_dict['h1_coldens']):
                    this_cloud = raydf[raydf['h1_cloud_flag'] == (i+1)]
                    m = m_p
                    dv = (np.max(this_cloud[this_velocity]) -
                          np.min(this_cloud[this_velocity])) * u.km / u.s
                    T = sum(this_cloud['temperature'] * this_cloud['H_p0_number_density']*this_cloud['dx'] / sum(this_cloud['H_p0_number_density']*this_cloud['dx'])) * u.K
                    vc = np.sqrt(k_B*T / m).to('km/s')
                    nref10f_h1temps.append(T.value)
                    nref10f_h1dv.append(dv.value)
                    nref10f_h1vc.append(vc.value)
                    nref10f_h1cloudcolumns.append(item)
                    nref10f_h1_cloud_cells.append(len(this_cloud))

        if (nref10f_size_dict['nsi2'] > si2_limit):
            if np.max(nref10f_size_dict['si2_kpcsizes']) < 200:
                ## loop through the clouds
                for i, item in enumerate(nref10f_size_dict['si2_coldens']):
                    this_cloud = raydf[raydf['si2_cloud_flag'] == (i+1)]
                    m = simass * m_p
                    dv = (np.max(this_cloud[this_velocity]) -
                          np.min(this_cloud[this_velocity])) * u.km / u.s
                    T = sum(this_cloud['temperature'] * this_cloud['Si_p1_number_density']*this_cloud['dx'] / sum(this_cloud['Si_p1_number_density']*this_cloud['dx'])) * u.K
                    vc = np.sqrt(k_B*T / m).to('km/s')
                    nref10f_si2temps.append(T.value)
                    nref10f_si2dv.append(dv.value)
                    nref10f_si2vc.append(vc.value)
                    nref10f_si2cloudcolumns.append(item)
                    nref10f_si2_cloud_cells.append(len(this_cloud))

        if (nref10f_size_dict['no6'] > o6_limit):
            if np.max(nref10f_size_dict['o6_kpcsizes']) < 200:
                ## loop through the clouds
                for i, item in enumerate(nref10f_size_dict['o6_coldens']):
                    this_cloud = raydf[raydf['o6_cloud_flag'] == (i+1)]
                    m = omass * m_p
                    dv = (np.max(this_cloud[this_velocity]) -
                          np.min(this_cloud[this_velocity])) * u.km / u.s
                    T = sum(this_cloud['temperature'] * this_cloud['O_p5_number_density']*this_cloud['dx'] / sum(this_cloud['O_p5_number_density']*this_cloud['dx'])) * u.K
                    vc = np.sqrt(k_B*T / m).to('km/s')
                    nref10f_o6temps.append(T.value)
                    nref10f_o6dv.append(dv.value)
                    nref10f_o6vc.append(vc.value)
                    nref10f_o6cloudcolumns.append(item)
                    nref10f_o6_cloud_cells.append(len(this_cloud))

        if (nref10f_size_dict['nc4'] > c4_limit):
            if np.max(nref10f_size_dict['c4_kpcsizes']) < 200:
                ## loop through the clouds
                for i, item in enumerate(nref10f_size_dict['c4_coldens']):
                    this_cloud = raydf[raydf['c4_cloud_flag'] == (i+1)]
                    m = cmass * m_p
                    dv = (np.max(this_cloud[this_velocity]) -
                          np.min(this_cloud[this_velocity])) * u.km / u.s
                    T = sum(this_cloud['temperature'] * this_cloud['C_p3_number_density']*this_cloud['dx'] / sum(this_cloud['C_p3_number_density']*this_cloud['dx'])) * u.K
                    vc = np.sqrt(k_B*T / m).to('km/s')
                    nref10f_c4temps.append(T.value)
                    nref10f_c4dv.append(dv.value)
                    nref10f_c4vc.append(vc.value)
                    nref10f_c4cloudcolumns.append(item)
                    nref10f_c4_cloud_cells.append(len(this_cloud))

    nref10f_h1_sampling = np.array(nref10f_h1dv) / (np.array(nref10f_h1vc))
    nref10f_si2_sampling = np.array(nref10f_si2dv) / (np.array(nref10f_si2vc))
    nref10f_c4_sampling =  np.array(nref10f_c4dv) / (np.array(nref10f_c4vc))
    nref10f_o6_sampling = np.array(nref10f_o6dv) / (np.array(nref10f_o6vc))
    nref10f_h1_ratio = np.array(nref10f_h1_sampling)  / ((np.array(nref10f_h1_cloud_cells) - 1)*sigma_ratio)
    nref10f_si2_ratio = np.array(nref10f_si2_sampling)  / ((np.array(nref10f_si2_cloud_cells) - 1)*sigma_ratio)
    nref10f_c4_ratio = np.array(nref10f_c4_sampling)  / ((np.array(nref10f_c4_cloud_cells) - 1)*sigma_ratio)
    nref10f_o6_ratio = np.array(nref10f_o6_sampling)  / ((np.array(nref10f_o6_cloud_cells) - 1)*sigma_ratio)
    nref10f_h1_ratio[nref10f_h1_ratio == np.nan] = 1e6
    nref10f_si2_ratio[nref10f_si2_ratio == np.nan] = 1e6
    nref10f_c4_ratio[nref10f_c4_ratio == np.nan] = 1e6
    nref10f_o6_ratio[nref10f_o6_ratio == np.nan] = 1e6
    nref10f_h1_ratio[nref10f_h1_ratio == np.inf] = 1e6
    nref10f_si2_ratio[nref10f_si2_ratio == np.inf] = 1e6
    nref10f_c4_ratio[nref10f_c4_ratio == np.inf] = 1e6
    nref10f_o6_ratio[nref10f_o6_ratio == np.inf] = 1e6


    ################################################################
    #######  histogram of velocity ratio ###########################
    ################################################################
    fig = plt.figure(figsize=(14,11))
    axtop = fig.add_axes([0.1, 0.52, 0.88, 0.42],
                        xlim=(0,4),ylim=(0,1.05))
    axbot = fig.add_axes([0.1, 0.1, 0.88, 0.42],
                        xlim=(0,4),ylim=(0,1.05))
    hist_bins = 5000
    axtop.hist(natural_h1_ratio, hist_bins, range=(0, 20), normed=True, cumulative=-1,edgecolor=hi_color, lw=3,histtype='step',align='mid', label='H I')
    axtop.hist(natural_si2_ratio, hist_bins, range=(0, 20), normed=True, cumulative=-1, edgecolor=si2_color,  lw=3, histtype='step', align='mid',label='Si II')
    axtop.hist(natural_c4_ratio, hist_bins, range=(0, 20), normed=True, cumulative=-1, edgecolor=c4_color,  lw=3,histtype='step',align='mid',label='C IV')
    axtop.hist(natural_o6_ratio, hist_bins, range=(0, 20), normed=True, cumulative=-1,edgecolor=ovi_color,  lw=3,histtype='step', align='mid',label='O VI')
    # axtop.hist(((np.array(natural_h1_cloud_cells) - 1)*sigma_ratio) / np.array(natural_h1_sampling) , hist_bins, range=(0, 20), normed=True, cumulative=-1,edgecolor=hi_color, lw=3,histtype='step',align='mid', label='H I')
    # axtop.hist(((np.array(natural_si2_cloud_cells) - 1)*sigma_ratio) / np.array(natural_si2_sampling) , hist_bins, range=(0, 20), normed=True, cumulative=-1, edgecolor=si2_color,  lw=3, histtype='step', align='mid',label='Si II')
    # axtop.hist(((np.array(natural_c4_cloud_cells) - 1)*sigma_ratio) / np.array(natural_c4_sampling) , hist_bins, range=(0, 20), normed=True, cumulative=-1, edgecolor=c4_color,  lw=3,histtype='step',align='mid',label='C IV')
    # axtop.hist(((np.array(natural_o6_cloud_cells) - 1)*sigma_ratio) / np.array(natural_o6_sampling) , hist_bins, range=(0, 20), normed=True, cumulative=-1,edgecolor=ovi_color,  lw=3,histtype='step', align='mid',label='O VI')
    axtop.set_xticklabels(())
    axtop.set_yticks((0.2, 0.4, 0.6,0.8,1))
    axtop.set_yticklabels(('0.2','0.4','0.6','0.8','1'))
    axtop.grid(True)
    fig.text(0.5, 0.87, 'standard resolution', fontsize=32, ha='left')


    axbot.hist(nref10f_h1_ratio, hist_bins, range=(0, 20), normed=True, cumulative=-1,edgecolor=hi_color, lw=3,histtype='step',align='mid', label='H I')
    axbot.hist(nref10f_si2_ratio, hist_bins, range=(0, 20), normed=True, cumulative=-1, edgecolor=si2_color,  lw=3, histtype='step', align='mid',label='Si II')
    axbot.hist(nref10f_c4_ratio, hist_bins, range=(0, 20), normed=True, cumulative=-1, edgecolor=c4_color, lw=3,histtype='step',align='mid',label='C IV')
    axbot.hist(nref10f_o6_ratio, hist_bins, range=(0, 20), normed=True, cumulative=-1,edgecolor=ovi_color, lw=3,histtype='step', align='mid',label='O VI')
    # axbot.hist(((np.array(nref10f_h1_cloud_cells) - 1)*sigma_ratio) / np.array(nref10f_h1_sampling) , hist_bins, range=(0, 20), normed=True, cumulative=-1,edgecolor=hi_color, lw=3,histtype='step',align='mid', label='H I')
    # axbot.hist(((np.array(nref10f_si2_cloud_cells) - 1)*sigma_ratio) / np.array(nref10f_si2_sampling) , hist_bins, range=(0, 20), normed=True, cumulative=-1, edgecolor=si2_color,  lw=3, histtype='step', align='mid',label='Si II')
    # axbot.hist(((np.array(nref10f_c4_cloud_cells) - 1)*sigma_ratio) / np.array(nref10f_c4_sampling) , hist_bins, range=(0, 20), normed=True, cumulative=-1, edgecolor=c4_color,  lw=3,histtype='step',align='mid',label='C IV')
    # axbot.hist(((np.array(nref10f_o6_cloud_cells) - 1)*sigma_ratio) / np.array(nref10f_o6_sampling) , hist_bins, range=(0, 20), normed=True, cumulative=-1,edgecolor=ovi_color,  lw=3,histtype='step', align='mid',label='O VI')
    axbot.set_yticks((0,0.2, 0.4, 0.6,0.8,1))
    axbot.set_yticklabels(('0','0.2','0.4','0.6','0.8','1'))
    axbot.legend(loc='upper right')
    #axbot.set_xscale('log')
    #axbot.set_xticks((0.1, 1, 10))
    #axbot.set_xticklabels(('0.1','1','10'))
    axbot.grid(True)
    fig.text(0.5, 0.46, 'high resolution', fontsize=32, ha='left')

    fig.text(0.5, 0.02, r'resolved velocity sampling ratio', fontsize=34, ha='center')
    fig.text(0.02, 0.5, 'fraction of clouds with larger ratio', fontsize=34, va='center', rotation='vertical')

    # plt.tight_layout()
    plt.savefig(output_dir + 'cloud_velocities.png')
    plt.savefig(output_dir + 'cloud_velocities.pdf')


    print("natural HI < 1: ", len(natural_h1_ratio[natural_h1_ratio < 1]) / len(natural_h1_ratio),"< 0.5: ",len(natural_h1_ratio[natural_h1_ratio < 0.5]) / len(natural_h1_ratio))
    print("natural SiII < 1: ", len(natural_si2_ratio[natural_si2_ratio < 1]) / len(natural_si2_ratio),"< 0.5: ",len(natural_si2_ratio[natural_si2_ratio < 0.5]) / len(natural_si2_ratio))
    print("natural CIV < 1: ", len(natural_c4_ratio[natural_c4_ratio < 1]) / len(natural_c4_ratio),"< 0.5: ",len(natural_c4_ratio[natural_c4_ratio < 0.5]) / len(natural_c4_ratio))
    print("natural OVI < 1: ", len(natural_o6_ratio[natural_o6_ratio < 1]) / len(natural_o6_ratio),"< 0.5: ",len(natural_o6_ratio[natural_o6_ratio < 0.5]) / len(natural_o6_ratio))
    print("ref HI < 1: ", len(nref10f_h1_ratio[nref10f_h1_ratio < 1]) / len(nref10f_h1_ratio),"< 0.5: ",len(nref10f_h1_ratio[nref10f_h1_ratio < 0.5]) / len(nref10f_h1_ratio))
    print("ref SiII < 1: ", len(nref10f_si2_ratio[nref10f_si2_ratio < 1]) / len(nref10f_si2_ratio),"< 0.5: ",len(nref10f_si2_ratio[nref10f_si2_ratio < 0.5]) / len(nref10f_si2_ratio))
    print("ref CIV < 1: ", len(nref10f_c4_ratio[nref10f_c4_ratio < 1]) / len(nref10f_c4_ratio),"< 0.5: ",len(nref10f_c4_ratio[nref10f_c4_ratio < 0.5]) / len(nref10f_c4_ratio))
    print("ref OVI < 1: ", len(nref10f_o6_ratio[nref10f_o6_ratio < 1]) / len(nref10f_o6_ratio),"< 0.5: ",len(nref10f_o6_ratio[nref10f_o6_ratio < 0.5]) / len(nref10f_o6_ratio))



if __name__ == "__main__":
    plot_cloud_size_and_masses()
