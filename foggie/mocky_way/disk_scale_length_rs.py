import yt
import numpy as np

def disk_gas_profile_n_r(ds, halo_center, normal_vector,
                         disk_r_kpc=50, disk_z_kpc=2,
                         field='H_nuclei_density'):
    """
    Calculate the hydrogen number density profile for a disk of size (r, +/-z).
    Here I'm only approprimately the radius with r, but to do it more carefully,
    I really do calculate the value of r projected onto the disk plane.

    input:
    normal_vector: also used as L_vec, the unit angular momentum vector of the
         galaxy
    disk_r_kpc: the radius of the disk cylinder
    disk_z_kpc: the extension of the disk +/-z, the thickness is 2z

    History:
    - 04/20/2019, UCB, Yong Zheng.
    - 08/19/2019, add the field parameter, because in the real MW case,
      it is the HI instead of the total H that is used to calculate the radial
      profile (see Kalberla+2008), Yong Zheng, UCB.
     - 10/07/2019, merging into foggie.mocky_way funcs, and add __name__ part. Yong Zheng.
    """

    # first, cut out a disk
    disk = ds.disk(halo_center,
                   normal_vector,
                   (disk_r_kpc, 'kpc'),
                   (disk_z_kpc, 'kpc'))
    disk.set_field_parameter('center', halo_center)

    # calculate the distance of particles to the center
    gas_x = (disk['gas', 'x'] - halo_center[0]).to('kpc')
    gas_y = (disk['gas', 'y'] - halo_center[1]).to('kpc')
    gas_z = (disk['gas', 'z'] - halo_center[2]).to('kpc')
    gas_r = np.sqrt(gas_x**2 + gas_y**2 + gas_z**2) # kpc
    gas_n = disk[('gas', field)] # total H number density

    return gas_r, gas_n

def profile_mean_med_std_3sig(gas_r, gas_n, minr=0, maxr=50, dr=0.5):
    """
    Calculate the mean, std, median, and 3sig range of gas density profile
    in either z or r direction.

    History:
    08/16/2019, Yong Zheng. UCB
    10/07/2019, merging into foggie.mocky_way, now nr_3sig/2sig/1sig have shapes
                of (2, gas_r.size), which = (upper lims, lower lims)
    """

    import numpy as np

    # now let's calculate the profile from this sphere
    threesig = 0.9973
    twosig = 0.95
    onesig = 0.68

    rbins = np.mgrid[minr:maxr+dr: dr]  # kpc
    nr_mean = np.zeros(rbins.size)
    nr_std = np.zeros(rbins.size)
    nr_med = np.zeros(rbins.size)
    nr_3sig = np.zeros((2, rbins.size))
    nr_2sig = np.zeros((2, rbins.size))
    nr_1sig = np.zeros((2, rbins.size))
    for ir in range(rbins.size):
        indr = np.all([gas_r>=rbins[ir], gas_r<=rbins[ir]+dr], axis=0)
        gas_n_shell = gas_n[indr]
        nr_mean[ir] = np.nanmean(gas_n_shell)
        nr_std[ir] = np.nanstd(gas_n_shell)

        # get the range of the gas_r that includes 99.73% (3sigma) of the points
        gas_n_shell = gas_n_shell[np.argsort(gas_n_shell)]
        all_index = np.arange(gas_n_shell.size)+1
        cum_frac = all_index/gas_n_shell.size

        indmed = np.argmin(np.abs(cum_frac-0.5))
        nr_med[ir] = gas_n_shell[indmed]

        indup = np.argmin(np.abs(cum_frac-(0.5+threesig/2.)))
        indlow = np.argmin(np.abs(cum_frac-(0.5-threesig/2.)))
        nr_3sig[0, ir] = gas_n_shell[indup]  # upper 3 sigma limit
        nr_3sig[1, ir] = gas_n_shell[indlow] # lower 3 sigmma limit

        indup = np.argmin(np.abs(cum_frac-(0.5+twosig/2.)))
        indlow = np.argmin(np.abs(cum_frac-(0.5-twosig/2.)))
        nr_2sig[0, ir] = gas_n_shell[indup]  # upper 2 sigma limit
        nr_2sig[1, ir] = gas_n_shell[indlow] # lower 2 sigmma limit

        indup = np.argmin(np.abs(cum_frac-(0.5+onesig/2.)))
        indlow = np.argmin(np.abs(cum_frac-(0.5-onesig/2.)))
        nr_1sig[0, ir] = gas_n_shell[indup]  # upper 3 sigma limit
        nr_1sig[1, ir] = gas_n_shell[indlow] # lower 3 sigmma limit

    stat_profiles = {'rbins': rbins,
                     'nr_mean': nr_mean,
                     'nr_std': nr_std,
                     'nr_med': nr_med,
                     'nr_3sig': nr_3sig,
                     'nr_2sig': nr_2sig,
                     'nr_1sig': nr_1sig
                    }
    return stat_profiles

def fit_nr_exp_profile(gas_r, gas_n, fit_minr=0.2, fit_maxr=10):
    """
    Fit some chunck of the pre-calculated gas_n/gas_r profiles with an
    exponential function. r could be z as well.

    fit_minr and fit_maxr are used to set the range of gas for fitting.

    History:
    08/15/2019, Yong Zheng, UCB.
    10/07/2019, merging into foggie.mocky_way, Yong Zheng.
    """

    ### only fit a small chunck of the whole profile
    import numpy as np
    indr = np.all([gas_r>=fit_minr, gas_r<=fit_maxr], axis=0)
    r_to_fit = gas_r[indr]
    n_to_fit = gas_n[indr]

    from scipy.optimize import curve_fit
    popt, pcov = curve_fit(ln_exp_profile, r_to_fit,
                           np.log(n_to_fit))
    psig = np.sqrt(np.diag(pcov))
    print('Fitted Scale rs: %.2f +/-%.2f kpc'%(popt[1], psig[1]))
    print('Fitted Density ln(n0): %.2f +/-%.2f cm-3'%(popt[0], psig[0]))

    return popt, psig

def ln_exp_profile(x, ln_n0, x0):
    """
    An exponential profile, used to fit the disk gas density profile,
    could be radial or vertical

    History:
    Created on 08/15/2019, Yong Zheng, UCB.
    10/07/2019, merging into foggie.mocky_way
    """
    y = ln_n0-x/x0
    return y

def plot_gas_profiles(stat_profiles, fit_popt, figname):
    """plotting stuff, always changing, not uniformly adjusted. Yong Zheng. """

    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rcParams['font.family'] = 'stixgeneral'

    rbins = stat_profiles['rbins']
    nr_mean = stat_profiles['nr_mean']
    nr_med = stat_profiles['nr_med']
    nr_3sig = stat_profiles['nr_3sig']
    nr_2sig = stat_profiles['nr_2sig']
    nr_1sig = stat_profiles['nr_1sig']

    ### plot the mean, std, and confidence levels ###
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
    ax.fill_between(rbins, nr_3sig[0, :], nr_3sig[1, :], edgecolor=None,
                    facecolor='k', alpha=0.1, label=None)
    ax.fill_between(rbins, nr_2sig[0, :], nr_2sig[1, :], edgecolor=None,
                    facecolor='k', alpha=0.15, label=None)
    ax.fill_between(rbins, nr_1sig[0, :], nr_1sig[1, :], edgecolor=None,
                    facecolor='k', alpha=0.25, label=None)
    ax.plot(rbins, nr_mean, color=plt.cm.Reds(0.6), lw=3, ls='--', label='Mean')
    ax.plot(rbins, nr_med, color='k', lw=0.8, ls='-', label='Median')

    ### plot the exponential fit ###
    xbins = np.mgrid[0:40:0.1]
    ln_n0 = fit_popt[0]
    r0 = fit_popt[1]
    ybins = np.exp(ln_exp_profile(xbins, ln_n0, r0))
    label = r'Exp fit, r$_{\rm s}$=%.1f kpc'%(r0)
    ax.plot(xbins, ybins, color='k', lw=2, label=label, linestyle='-')

    fontsize = 16
    # ax.set_xlim(0, 10) # in unit of kpc
    # ax.set_ylim(1e-10, 1e-4) # in unit of cm-3
    ax.set_yscale('log')
    ax.legend(fontsize=fontsize-2, loc='upper right')
    ax.set_xlabel('r (kpc)', fontsize=fontsize)
    ax.set_ylabel(r'n$_{\rm H}$(r) (cm$^{-3}$)', fontsize=fontsize)
    ax.minorticks_on()
    ax.set_title(r"Disk scale length $r_{\rm s}$=%.1f kpc"%(r0))

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize-2)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize-2)

    fig.savefig(figname)
    print("Saving the figure to ", figname)

############################################################################
if __name__ == "__main__":
    ### Read in the simulation data and find halo center  ###

    import sys
    import os 
    sim_name = sys.argv[1] # 'nref11n_nref10f'
    dd_name = sys.argv[2]  # 'DD2175'

    from core_funcs import data_dir_sys_dir
    data_dir, sys_dir = data_dir_sys_dir()
    ds_file = '%s/%s/%s/%s'%(data_dir, sim_name, dd_name, dd_name)
    if os.path.isfile(ds_file) == False:
        drive_dir = '/Volumes/Yong4TB/foggie/halo_008508'
        ds_file = '%s/%s/%s/%s'%(drive_dir, sim_name, dd_name, dd_name)

    ds = yt.load(ds_file)
    zsnap = ds.get_parameter('CosmologyCurrentRedshift')

    from core_funcs import find_halo_center_yz
    halo_center = find_halo_center_yz(ds, zsnap, sim_name, data_dir)

    ### find the disk normal vector ###
    from core_funcs import dict_sphere_for_gal_ang_mom
    from core_funcs import get_sphere_ang_mom_vecs
    r_for_L = dict_sphere_for_gal_ang_mom(dd_name, sim_name=sim_name)
    from core_funcs import default_random_seed
    random_seed = default_random_seed()
    dict_vecs = get_sphere_ang_mom_vecs(ds, halo_center, r_for_L,
                                        random_seed=random_seed)
    normal_vector = dict_vecs['L_vec']

    ### get the gas density profile ###
    gas_r, gas_n  = disk_gas_profile_n_r(ds, halo_center, normal_vector,
                                         disk_r_kpc=100,
                                         disk_z_kpc=0.5,
                                         field='H_nuclei_density')
    ### get some statistics distribution of it
    stat_profiles = profile_mean_med_std_3sig(gas_r, gas_n,
                                              minr=0, maxr=100, dr=1)
    ### an exponential fit of the profile
    fit_r = stat_profiles['rbins']
    fit_n = stat_profiles['nr_mean']
    # For DD2175
    popt, psig = fit_nr_exp_profile(fit_r, fit_n, fit_minr=5, fit_maxr=30)
    # For RD0039
    # popt, psig = fit_nr_exp_profile(fit_r, fit_n, fit_minr=10, fit_maxr=40)

    ### plotting stuff ####
    figname = 'figs/disk_rs_zs/%s_%s_disk_rs.pdf'%(sim_name, dd_name)
    plot_gas_profiles(stat_profiles, popt, figname)
