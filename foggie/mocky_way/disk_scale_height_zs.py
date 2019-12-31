import yt
import numpy as np

def disk_gas_profile_n_z(ds, halo_center, normal_vector,
                         disk_r_kpc=40, disk_z_kpc=50,
                         field='H_nuclei_density'):
    """
    calculate the density profile of disk along the z direction (perpendicular
    to the disk plane).

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
    gas_xyz = np.array([gas_x, gas_y, gas_z]) # shape of (3, N)
    gas_r = np.sqrt(gas_x**2 + gas_y**2 + gas_z**2) # kpc
    gas_n = disk[('gas', field)] # total H number density

    # then project this distance to the veritcal direction
    cos_theta = np.zeros(gas_r.size)
    for i in np.arange(gas_r.size):
        cos_theta[i] = np.dot(gas_xyz[:, i], normal_vector)/gas_r[i]
    gas_proj_z = gas_r*cos_theta # note that gas_proj_z is different from gas_z
                                 # because gas_z is the code unit

    return gas_proj_z, gas_n

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
    xbins = np.mgrid[0:6:0.1]
    ln_n0 = fit_popt[0]
    r0 = fit_popt[1]
    from disk_scale_length_rs import ln_exp_profile
    ybins = np.exp(ln_exp_profile(xbins, ln_n0, r0))
    label = r'Exp fit, z$_{\rm s}$=%.1f kpc'%(r0)
    ax.plot(xbins, ybins, color='k', lw=2, label=label, linestyle='-')

    fontsize = 16
    # ax.set_xlim(0, 10) # in unit of kpc
    # ax.set_ylim(1e-10, 1e-4) # in unit of cm-3
    ax.set_yscale('log')
    ax.legend(fontsize=fontsize-2, loc='upper right')
    ax.set_xlabel('height z (kpc)', fontsize=fontsize)
    ax.set_ylabel(r'n$_{\rm H}$(z) (cm$^{-3}$)', fontsize=fontsize)
    ax.minorticks_on()
    ax.set_title(r"Disk scale height $z_{\rm s}$=%.1f kpc"%(r0))

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize-2)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize-2)

    fig.savefig(figname)
    print("Saving the figure to ", figname)

############################################################################
if __name__ == "__main__":
    ### Read in the simulation data and find halo center  ###
    import sys, os
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
    gas_proj_z, gas_n  = disk_gas_profile_n_z(ds, halo_center, normal_vector,
                                              disk_r_kpc=20,
                                              disk_z_kpc=50,
                                              field='H_nuclei_density')

    from disk_scale_length_rs import profile_mean_med_std_3sig
    ### get some statistics distribution of it
    stat_profiles = profile_mean_med_std_3sig(gas_proj_z, gas_n,
                                              minr=0, maxr=49, dr=0.5)
    ### an exponential fit of the profile
    from disk_scale_length_rs import fit_nr_exp_profile
    fit_r = stat_profiles['rbins']
    fit_n = stat_profiles['nr_mean']
    popt, psig = fit_nr_exp_profile(fit_r, fit_n, fit_minr=0.01, fit_maxr=3)

    ### plotting stuff ####
    figname = 'figs/disk_rs_zs/%s_%s_disk_zs.pdf'%(sim_name, dd_name)
    plot_gas_profiles(stat_profiles, popt, figname)
