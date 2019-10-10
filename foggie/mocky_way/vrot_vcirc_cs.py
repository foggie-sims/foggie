def calc_vrot_slit(ds, dd_name, sim_name, halo_center, normal_vector,
                   los_vector, disk_bulkvel, maxr=100, dr=0.5):
    """
    Calculate rotation velocity of gas, following description in Sec 3.1 of
    El-Baldry+2018. This is to calculate the circular velocity in an observational
    way. We want to compare it with the result from calc_vrot_phi, which focuses
    on an approach that's theoretical.

    Inputs:
    normal_vector: the angular momentum vector, L_vec
    los_vector: line of sight vector we are gonna project the vrot onto
    disk_bulkvel: the bulk velocity of the galaxy,
                  of which the normal vector is calculated.
    maxr: maximum radius within which vcirc would be calculated
    dr: the radius interval

    Return:
    We save the data to a fits file for future uses. fitsfiles can be read with
    astropy.table func, and keywords are r, v_rotation, v_dispersion.

    History:
    03/27/2019, Yong Zheng, UCB, first write up the function to calculate roation/dispersion
    08/20/2019, Yong Zheng, UCB, move this func to mocky_way_modules
    10/08/2019, Yong Zheng, UCb, merging into foggie.mocky_way

    """

    from core_funcs import data_dir_sys_dir
    from tqdm import tqdm
    import numpy as np

    data_dir, sys_dir = data_dir_sys_dir()
    bulk_velocity = disk_bulkvel

    #### calculate the rotational velocity as if we are doing real observation
    # put a slit along the edgeon direction. Assuming a 2 kpc wide slit.
    disk = ds.disk(halo_center, normal_vector, (maxr, 'kpc'), (2, 'kpc'))
    disk.set_field_parameter('center', halo_center)
    disk.set_field_parameter('bulk_velocity', bulk_velocity)

    gas_vx = (disk[('gas', 'velocity_x')] - bulk_velocity[0]).to('km/s')
    gas_vy = (disk[('gas', 'velocity_y')] - bulk_velocity[1]).to('km/s')
    gas_vz = (disk[('gas', 'velocity_z')] - bulk_velocity[2]).to('km/s')
    gas_vec_velr = np.array([gas_vx, gas_vy, gas_vz]).T

    #### dot product to get line of sight velocity
    los_vel = np.dot(gas_vec_velr, los_vector) # in unit of km/s

    gas_x = (disk['gas', 'x'] - halo_center[0]).to('kpc')
    gas_y = (disk['gas', 'y'] - halo_center[1]).to('kpc')
    gas_z = (disk['gas', 'z'] - halo_center[2]).to('kpc')

    gas_vec_r = np.array([gas_x, gas_y, gas_z]).T
    gas_r = np.sqrt(gas_x**2 + gas_y**2 + gas_z**2) # kpc
    cos_theta = np.dot(gas_vec_r, los_vector)/gas_r.value
    sin_theta = np.sqrt(1-cos_theta**2)
    imp_para = gas_r*sin_theta  # kpc

    radii_slit = np.mgrid[0:maxr+dr:dr]  # kpc
    v_rot_slit = np.zeros(radii_slit.size) # rotation velocity along the list
    v_sig_slit = np.zeros(radii_slit.size) # dispersion
    for ir in tqdm(range(radii_slit.size)[1:]):
        rr = radii_slit[ir]
        ind_rr = np.all([imp_para>=rr, imp_para<=rr+dr], axis=0)
        v_rot_slit[ir] = np.nanmean(np.abs(los_vel[ind_rr]))
        v_sig_slit[ir] = np.nanstd(np.abs(los_vel[ind_rr]))

    # save the result to a table for future comparison
    from astropy.table import Table
    master = Table([radii_slit, v_rot_slit, v_sig_slit],
                    names=('r', 'v_rotation', 'v_dispersion'),
                    meta={'name': 'FOGGIE/%s/%s'%(sim_name, dd_name)})
    master['r'].format = '8.2f'
    master['r'].unit = 'kpc'
    master['v_rotation'].format = '8.2f'
    master['v_rotation'].unit = 'km/s'
    master['v_dispersion'].format = '8.2f'
    master['v_dispersion'].unit = 'km/s'

    save_dir = sys_dir+'/foggie/mocky_way/figs/vrot_vcirc_cs'
    filename = '%s_%s_vrotslit'%(sim_name, dd_name)
    fitsfile = '%s/%s.fits'%(save_dir, filename)
    master.write(fitsfile, overwrite=True)
    print("Saving to ", fitsfile)

    return fitsfile

def calc_vrot_phi(ds, dd_name, sim_name, halo_center, L_vec, sun_vec, phi_vec,
                  rvir, disk_bulkvel, maxr=100, dr=0.5):

    """
    Calculate rotation velocity of gas, following description in Sec 3.1 of
    El-Baldry+2018. This is to calculate the circular velocity in a theoretical
    way. We want to compare it with the result from calc_vrot_slit, which focuses
    on an approach that's more observational.

    Inputs:
    L_vec, sun_vec, phi_vec: the angular momentum vector, and the other two are
              on the galaxy plane
    rvir: virial radius of the galaxy, e.g., rvir=ds.quan(150, 'kpc')
    disk_bulkvel: the bulk velocity of the galaxy,
                  of which the normal vector is calculated.
    maxr: maximum radius within which vcirc would be calculated
    dr: the radius interval

    Output:
    We save the data to a fits file for future uses. fitsfiles can be read with
    astropy.table func, and keywords are r, v_rotation, v_dispersion.


    History:
    03/27/2019, Yong Zheng, UCB, first write up the function to calculate roation/dispersion
    08/20/2019, Yong Zheng, UCB, move this func to mocky_way_modules
    10/08/2019, Yong Zheng, UCB, merging into foggie.mocky_way

    """

    from core_funcs import data_dir_sys_dir
    from tqdm import tqdm
    import numpy as np
    import yt

    data_dir, sys_dir = data_dir_sys_dir()
    vecZ = L_vec
    vecY = sun_vec
    vecX = phi_vec

    big_sphere = ds.sphere(halo_center, rvir)
    ### Now, let's calculate the rotational velocity in a cylindrical coordinate
    gas_x = big_sphere['gas', 'x'] - halo_center[0]
    gas_y = big_sphere['gas', 'y'] - halo_center[1]
    gas_z = big_sphere['gas', 'z'] - halo_center[2]

    gas_vx = big_sphere['gas', 'velocity_x'] - disk_bulkvel[0]
    gas_vy = big_sphere['gas', 'velocity_y'] - disk_bulkvel[1]
    gas_vz = big_sphere['gas', 'velocity_z'] - disk_bulkvel[2]

    gas_pos_vector = yt.YTArray([gas_x, gas_y, gas_z]).T
    gas_vel_vector = yt.YTArray([gas_vx, gas_vy, gas_vz]).T

    # say for a vector r, its angle theta with the new axes are
    gas_newx = np.dot(gas_pos_vector, vecX) # this is only true when newX is a unit vector
    gas_newy = np.dot(gas_pos_vector, vecY)
    gas_newz = np.dot(gas_pos_vector, vecZ)

    gas_newvx = np.dot(gas_vel_vector, vecX)
    gas_newvy = np.dot(gas_vel_vector, vecY)
    gas_newvz = np.dot(gas_vel_vector, vecZ)

    # according to El-Badry+2018, section 3.1
    v_phi = (gas_newx*gas_newvy - gas_newy*gas_newvx)/np.sqrt(gas_newx**2 + gas_newy**2)
    v_phi = v_phi.in_units('km/s')
    new_r = np.sqrt(gas_newx**2 + gas_newy**2).in_units('kpc')
    gas_mass = big_sphere['gas', 'cell_mass']

    calc_radii = yt.YTArray(np.mgrid[0:maxr+dr:dr], 'kpc')
    v_rot_phi = yt.YTArray(np.zeros(calc_radii.size), 'km/s') # rotation velocity
    v_sig_phi = yt.YTArray(np.zeros(calc_radii.size), 'km/s') # velocity dispersion
    for ir in tqdm(range(calc_radii.size)[1:]):
        ind = np.all([new_r >= calc_radii[ir-1], new_r < calc_radii[ir]], axis=0)
        # gas mass weighted mean of the phi
        v_rot_phi[ir] = np.mean(np.abs(v_phi[ind]))
        v_sig_phi[ir] = np.sqrt(np.nanmean(v_phi[ind]**2) - v_rot_phi[ir]**2)

    # save the result to a table for future comparison
    from astropy.table import Table
    master = Table([calc_radii, v_rot_phi, v_sig_phi],
                    names=('r', 'v_rotation', 'v_dispersion'),
                    meta={'name': 'FOGGIE/%s/%s'%(sim_name, dd_name)})
    master['r'].format = '8.2f'
    master['r'].unit = 'kpc'
    master['v_rotation'].format = '8.2f'
    master['v_rotation'].unit = 'km/s'
    master['v_dispersion'].format = '8.2f'
    master['v_dispersion'].unit = 'km/s'

    save_dir = sys_dir+'/foggie/mocky_way/figs/vrot_vcirc_cs'
    filename = '%s_%s_vrotphi'%(sim_name, dd_name)
    fitsfile = '%s/%s.fits'%(save_dir, filename)

    print("Saving to ", fitsfile)
    master.write(fitsfile, overwrite=True)

    return fitsfile

def calc_vcirc(ds, dd_name, sim_name, halo_center, maxr=150, dr=0.5):
    """
    Calculate circular velocity of gas: vcirc = sqrt[G M(R<r)/R], save
    the data to a fits file for future uses. fitsfiles can be read with
    astropy.table func, and keywords are r, v_circ

    ds, ds_paras are from data source and related information from prepdata
    maxr is the maximum radius within within vcirc would be calculated
    dr is the radius interval

    History:
    03/27/2019, Yong Zheng, UCB, first write up the function to calculate circular velocity
    08/20/2019, Yong Zheng, UCB, move this func to mocky_way_modules
    10/08/2019, Yong Zheng, UCB, merging into foggie.mocky_way

    """
    import numpy as np
    from tqdm import tqdm
    import yt
    from yt.units import gravitational_constant_cgs as grav_const

    calc_radii = yt.YTArray(np.mgrid[0:maxr+dr:dr], 'kpc')

    # Part I: calculate the circular velocity of gas in this galaxy
    big_sphere = ds.sphere(halo_center, (maxr, 'kpc'))

    print("Gettting particle mass and positions")
    par_x = (big_sphere['particle_position_x'] - halo_center[0]).to('kpc')
    par_y = (big_sphere['particle_position_y'] - halo_center[1]).to('kpc')
    par_z = (big_sphere['particle_position_z'] - halo_center[2]).to('kpc')
    par_r = np.sqrt(par_x**2 + par_y**2 + par_z**2)
    par_mass = big_sphere['particle_mass'].to('Msun')

    print("Getting gas mass and positions")
    gas_x = (big_sphere[('gas', 'x')] - halo_center[0]).to('kpc')
    gas_y = (big_sphere[('gas', 'y')] - halo_center[1]).to('kpc')
    gas_z = (big_sphere[('gas', 'z')] - halo_center[2]).to('kpc')
    gas_r = np.sqrt(gas_x**2 + gas_y**2 + gas_z**2)
    gas_mass = big_sphere[('gas', 'cell_mass')].to('Msun')

    ### Now, let's calculate the rotational velocity
    print("Calculating circular velocity...")
    v_circ_cms = yt.YTArray(np.zeros(calc_radii.size), 'cm/s')
    for ir in tqdm(range(calc_radii.size)[1:]):
        rr = calc_radii[ir]
        ind_gas_rr = np.where(gas_r <= rr)[0]
        gas_mass_rr = np.sum(gas_mass[ind_gas_rr])
        ind_par_rr = np.where(par_r <= rr)[0]
        par_mass_rr = np.sum(par_mass[ind_par_rr])
        mass_rr = par_mass_rr + gas_mass_rr
        v_circ_cms[ir] = np.sqrt((grav_const * mass_rr.to('g') / rr.to('cm')))
        # print(v_circ_cms[ir])

    v_circ = v_circ_cms.to('km/s')

    # save the result to a table for future comparison
    from astropy.table import Table
    master = Table([calc_radii, v_circ],
                    names=('r', 'v_circ'),
                    meta={'name': 'FOGGIE/%s/%s'%(sim_name, dd_name)})
    master['r'].format = '8.2f'
    master['v_circ'].format = '8.2f'
    master['r'].unit = 'kpc'
    master['v_circ'].unit = 'km/s'

    save_dir = sys_dir+'/foggie/mocky_way/figs/vrot_vcirc_cs'
    filename = '%s_%s_vcirc'%(sim_name, dd_name)
    fitsfile = '%s/%s.fits'%(save_dir, filename)

    master.write(fitsfile, overwrite=True)
    print("Saving to ", fitsfile)

    return fitsfile

def calc_sound_speed(ds, dd_name, sim_name, halo_center, maxr=150, dr=0.5):
    """
    Calculate the sound speed of the gas, cs = sqrt[kT(r, r+dr)/m(r, r+dr)],
    where T(r, r+dr) is the mean temperature of gas from r to r+dr
    and m(r, r+dr) is the mean mass of gas from r to r+dr

    save to fits file, with keyword of r, and cs

    Note that in El-Baldry+2018, cs is calculated using HI instead of all gas

    History:
    08/20/2019, Yong Zheng, UCB, first write up of the sound speed function based on
          work by El-Baldry+2018.
    """
    import numpy as np
    from tqdm import tqdm
    import yt

    big_sphere = ds.sphere(halo_center, (maxr, 'kpc'))
    calc_radii = yt.YTArray(np.mgrid[0:maxr+dr:dr], 'kpc')

    print("Getting gas temperature and positions")
    gas_x = (big_sphere[('gas', 'x')] - halo_center[0]).to('kpc')
    gas_y = (big_sphere[('gas', 'y')] - halo_center[1]).to('kpc')
    gas_z = (big_sphere[('gas', 'z')] - halo_center[2]).to('kpc')
    gas_r = np.sqrt(gas_x**2 + gas_y**2 + gas_z**2) # in unit of  kpc
    gas_T = big_sphere[('gas', 'temperature')].to('K')

    calc_radii = yt.YTArray(np.mgrid[0:maxr+dr:dr], 'kpc')
    cs = yt.YTArray(np.zeros(calc_radii.size), 'cm/s') # rotation velocity

    from yt.units import kboltz as kb_erg_per_K
    from yt.units import mass_hydrogen as m_H_g
    mu_H = 1.3
    for ir in tqdm(range(calc_radii.size)[1:]):
        ind = np.all([gas_r>=calc_radii[ir-1], gas_r<calc_radii[ir]], axis=0)
        mean_T = np.mean(gas_T[ind])
        cs[ir] = np.sqrt(kb_erg_per_K * mean_T / mu_H / m_H_g)

    cs = cs.to('km/s')

    # save the result to a table for future comparison
    from astropy.table import Table
    master = Table([calc_radii, cs],
                    names=('r', 'cs'),
                    meta={'name': 'FOGGIE/%s/%s'%(sim_name, dd_name)})
    master['r'].format = '8.2f'
    master['cs'].format = '8.2f'
    master['r'].unit = 'kpc'
    master['cs'].unit = 'km/s'

    save_dir = sys_dir+'/foggie/mocky_way/figs/vrot_vcirc_cs'
    filename = '%s_%s_sound_speed'%(sim_name, dd_name)
    fitsfile = '%s/%s.fits'%(save_dir, filename)

    master.write(fitsfile, overwrite=True)
    print("Saving to ", fitsfile)

    return fitsfile

def plt_vrot(sim_name, dd_name, vrot_slit_fits, vrot_phi_fits, vcirc_fits, cs_fits):
    """Plot the rotation curves, will be always changing. No need to be fancy. Yong Zheng"""

    import numpy as np
    from astropy.table import Table
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rcParams['font.family'] = 'stixgeneral'

    vslit = Table.read(vrot_slit_fits, format='fits')
    vphi = Table.read(vrot_phi_fits, format='fits')
    vcirc = Table.read(vcirc_fits, format='fits')
    cs = Table.read(cs_fits, format='fits')

    grey = plt.cm.Greys(0.7)
    red = plt.cm.Reds(0.7)
    blue = plt.cm.Blues(0.6)
    lw = 1.5
    fs = 18

    # plot the result. Rotation
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(vcirc['r'], vcirc['v_circ'], label=r'$v_{\rm c}=\sqrt{GM(r<R)/R}$',
            color='k', lw=lw*3, linestyle=':')
    ax.plot(vphi['r'], vphi['v_rotation'], label=r'$v_{\rm rot, cylindrical}$',
            color=blue, lw=lw*3, linestyle='-')
    ax.plot(vphi['r'], vphi['v_dispersion'], label=r'$\sigma_{\rm cylindrical}$',
            color=blue, lw=lw, linestyle='-')
    ax.plot(vslit['r'], vslit['v_rotation'], label=r'$v_{\rm rot, slit}$',
            color=red, lw=lw*3, linestyle='--')
    ax.plot(vslit['r'], vslit['v_dispersion'], label=r'$\sigma_{\rm slit}$',
            color=red, lw=lw, linestyle='--')

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fs-2)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fs-2)

    ax.set_xlim(0, 30)
    ax.set_ylim(0, 400)
    ax.set_xlabel('R (kpc)', fontsize=fs)
    ax.set_ylabel(r'Velocity (km s$^{-1}$)', fontsize=fs)
    # ax.set_title(dd_name, fontsize=fs)
    ax.legend(fontsize=fs-4)
    ax.minorticks_on()
    fig.tight_layout()
    fig.savefig('figs/vrot_vcirc_cs/%s_%s_vcirc_vrot_vsigma.pdf'%(sim_name, dd_name))
    plt.close()

    # plot the result. Rotation
    linea = '-'
    lineb = ':'
    linec = '-.'
    lined = '--'
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(vcirc['r'], vcirc['v_circ'], label=r'$v_{\rm c}=\sqrt{GM(r<R)/R}$',
            color='k', lw=lw, linestyle=linea)
    ax.plot(cs['r'], cs['cs'], label=r'$c_{\rm s}\equiv\sqrt{kT/\mu m_{\rm p}}$',
            color=grey, lw=lw, linestyle=lined)
    ax.plot(vphi['r'], vphi['v_rotation'], label=r'$v_{\rm rot, cylindrical}$',
            color=blue, lw=lw, linestyle=lineb)
    ax.plot(vslit['r'], vslit['v_rotation'], label=r'$v_{\rm rot, slit}$',
            color=red, lw=lw, linestyle=linec)

    fs = 14
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 400)
    ax.set_xlabel('R (kpc)', fontsize=fs)
    ax.set_ylabel(r'Velocity (km s$^{-1}$)', fontsize=fs)
    ax.set_title(dd_name, fontsize=fs)
    ax.legend(fontsize=fs)
    fig.savefig('figs/vrot_vcirc_cs/%s_%s_vcirc_vrot_cs.pdf'%(sim_name, dd_name))
    plt.close()

    # plot the result, dispersion
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(vcirc['r'], vcirc['v_circ'], label=r'$v_{\rm c}=\sqrt{GM(r<R)/R}$',
            color='k', lw=lw, linestyle=linea)
    ax.plot(cs['r'], cs['cs'], label=r'$c_{\rm s}\equiv\sqrt{kT/\mu m_{\rm p}}$',
            color=grey, lw=lw, linestyle=lined)
    ax.plot(vphi['r'], vphi['v_dispersion'], label=r'$\sigma_{\rm cylindrical}$',
            color=blue, lw=lw, linestyle=lineb)
    ax.plot(vslit['r'], vslit['v_dispersion'], label=r'$\sigma_{\rm slit}$',
            color=red, lw=lw, linestyle=linec)

    fs = 14
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 400)
    ax.set_xlabel('R (kpc)', fontsize=fs)
    ax.set_ylabel(r'Velocity (km s$^{-1}$)', fontsize=fs)
    ax.set_title(dd_name, fontsize=fs)
    ax.legend(fontsize=fs)
    fig.savefig('figs/vrot_vcirc_cs/%s_%s_vcirc_vsigma_cs.pdf'%(sim_name, dd_name))
    plt.close()


if __name__ == "__main__":
    ### Read in the simulation data and find halo center  ###

    dd_name = 'RD0039'
    sim_name = 'nref11n_nref10f'
    run_funcs = False

    if run_funcs == True:
        from core_funcs import data_dir_sys_dir
        data_dir, sys_dir = data_dir_sys_dir()

        from core_funcs import prepdata
        ds, ds_paras = prepdata(dd_name, sim_name)

        maxr = 100
        dr = 0.5
        ### calculate rotation curves
        print("Calculating vrot through slit...")
        vrot_slit_fits = calc_vrot_slit(ds, dd_name, sim_name,
                                        ds_paras['halo_center'],
                                        ds_paras['L_vec'],
                                        ds_paras['sun_vec'],
                                        ds_paras['disk_bulkvel'],
                                        maxr=maxr, dr=dr)

        print("Calculating vrot in cylindrical way... ")
        vrot_phi_fits = calc_vrot_phi(ds, dd_name, sim_name,
                                      ds_paras['halo_center'],
                                      ds_paras['L_vec'],
                                      ds_paras['sun_vec'],
                                      ds_paras['phi_vec'],
                                      ds_paras['rvir'],
                                      ds_paras['disk_bulkvel'],
                                      maxr=maxr, dr=dr)

        print("Calculating circular velocity...")
        vcirc_fits = calc_vcirc(ds, dd_name, sim_name,
                                ds_paras['halo_center'],
                                maxr=maxr, dr=dr)

        print("Calculating the sound speed...")
        cs_fits = calc_sound_speed(ds, dd_name, sim_name,
                                   ds_paras['halo_center'],
                                   maxr=maxr, dr=dr)

    else:
        vrot_slit_fits = 'figs/vrot_vcirc_cs/nref11n_nref10f_RD0039_vrotslit.fits'
        vrot_phi_fits = 'figs/vrot_vcirc_cs/nref11n_nref10f_RD0039_vrotphi.fits'
        vcirc_fits = 'figs/vrot_vcirc_cs/nref11n_nref10f_RD0039_vcirc.fits'
        cs_fits = 'figs/vrot_vcirc_cs/nref11n_nref10f_RD0039_sound_speed.fits'

    print("Phew, finally plotting everything...")
    plt_vrot(dd_name, sim_name, vrot_slit_fits,
             vrot_phi_fits, vcirc_fits, cs_fits)
