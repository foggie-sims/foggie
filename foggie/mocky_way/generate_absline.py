def generate_absline(ds, ds_paras, los_rs, los_re,
                     lab_lambda=1393.7550,
                     lines=['Si IV 1394'],
                     line_snr=20, fontsize=16,
                     save_file='./spec.pdf'):
    """
    Generate absorption line at wavelength of lab_lambda

    Input:
    ds: data file from yt.load()
    ds_paras: a library object from prepdata.py, with info for the halo.
              The useful one from it is ds_paras["observer_bulkvel"],
              could make you onwself.
    los_rs: ray start point, in code_length unit
    los_re: ray end point, in code_length unit
    lab_lambda: rest wavelength of the line
    lines: the lines of interest, list object
    line_snr: the signal to noise ratio for the line in COS, Gaussian Noise

    Output:
    raw_wave: wavelength from trident
    raw_vel_obs: line-of-sight velocity (=vlsr), with gas bulk motion
    cos_flux: normalized flux assuming COS LSF and some SNR

    History:
    Early 2019, Yong wrote it, in mocky_way repo
    Feb 7, 2020, YZ merged to foggie.mocky_way
    """

    import trident
    import numpy as np
    from core_funcs import data_dir_sys_dir
    data_dir, sys_dir = data_dir_sys_dir()

    # lab_lambda = 1031.9261 # 1037.6167
    # add_lines = ['O VI 1032']

    #lab_lambda = 1548.2049  # 1550.7785
    #add_lines = ['C IV 1548']

    los_r = np.linalg.norm((los_rs-los_re).in_units("kpc"))
    save_ray = sys_dir+"/foggie/mocky_way/data/trident/ray.h5"
    tri_ray = trident.make_simple_ray(ds, start_position=los_rs.copy(),
                                      end_position=los_re.copy(),
                                      data_filename=save_ray,
                                      lines=lines)

    # the wavelength
    line_dir = '/Users/Yong/ForkedRepo/trident/trident/data/line_lists'
    ldb = trident.LineDatabase(line_dir+'/lines.txt')
    zsnap = ds.get_parameter("CosmologyCurrentRedshift")
    lambda_min = (1+zsnap)*(lab_lambda-6) # thisline * (1 + min(tri_ray.r['redshift_eff']).value) - 5
    lambda_max = (1+zsnap)*(lab_lambda+6) # thisline * (1 + max(tri_ray.r['redshift_eff']).value) + 5
    print('z=%.3f, (%.1f, %.1f) A'%(zsnap, lambda_min, lambda_max))

    # generate a spectrum
    sg = trident.SpectrumGenerator(lambda_min=lambda_min,
                                   lambda_max=lambda_max,
                                   dlambda=0.01,
                                   #line_database='atom_wave_gamma_f.dat'
                                   line_database=line_dir+'/lines.txt')
    sg.make_spectrum(tri_ray, lines=lines, min_tau=1.e-5, store_observables=True)

    # spectrum flux and wavelength within lsf and noise
    from astropy.constants import c as light_speed
    raw_wave = (sg.lambda_field/(1+zsnap)).copy()
    raw_flux = (sg.flux_field).copy()
    raw_vel = (raw_wave.value-lab_lambda)/lab_lambda*light_speed.to("km/s").value

    # correct tri-ray los_velocity to yt los velocity
    obs_bv = ds_paras["offcenter_bulkvel"].in_units("km/s")
    los_unit_vec = (los_re - los_rs)/np.linalg.norm(los_re - los_rs)
    cos_theta = np.dot(los_unit_vec, (obs_bv/np.linalg.norm(obs_bv)).value)
    obs_bv_rproj = ds.quan(np.linalg.norm(obs_bv)*cos_theta, 'km/s')
    raw_vel_obs = -raw_vel - obs_bv_rproj.value

    # add cos lsf first
    if lab_lambda <1450 and lab_lambda>1000:
        cos_lsf = sys_dir+'/foggie/mocky_way/cos_lsf/avg_COS_G130M_yz.txt'
    elif lab_lambda >=1450 and lab_lambda <1750:
        cos_lsf = sys_dir+'/foggie/mocky_way/cos_lsf/avg_COS_G160M_yz.txt'
    else:
        print('This line is not in G130M or G160M lsf range. Please check.')
        sys.exit(1)
    print('Appling COS LSF: ', cos_lsf)
    sg.apply_lsf(filename=cos_lsf)

    # then add Gaussian noise
    sg.add_gaussian_noise(line_snr)
    cos_flux = sg.flux_field

    # plot
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rcParams['font.family'] = 'stixgeneral'

    fig = plt.figure(figsize=(5, 5))
    ax1 = fig.add_subplot(211)
    xmin, xmax= -500, 500
    ymin, ymax = 0, 1.5

    ax1.step(raw_vel_obs, raw_flux, color=plt.cm.Greys(0.7))
    ax1.set_ylabel('Norm. Flux')
    ax1.set_title(r'Synthetic %s w Obj at %.1f kpc'%(lines[0], los_r),
                  fontsize=fontsize)

    ax2 = fig.add_subplot(212, sharex=ax1)
    ax2.step(raw_vel_obs, cos_flux, lw=0.7, color=plt.cm.Greys(0.7))
    ax2.text(xmin+0.03*(xmax-xmin), ymax-0.12*(ymax-ymin),
             'COS G130M LSF', fontsize=fontsize-5)
    ax2.text(xmin+0.03*(xmax-xmin), ymax-0.2*(ymax-ymin),
             'Gaussian Noise S/N=%d'%(line_snr), fontsize=fontsize-5)
    ax2.set_xlabel('LOS Velocity (km/s)', fontsize=fontsize)

    for ax in [ax1, ax2]:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.hlines(1.0, xmin, xmax, linestyle=':')
        ax.minorticks_on()
        ax.set_ylabel('Norm. Flux', fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
    plt.tight_layout()
    fig.savefig(save_file)

    return raw_wave, raw_vel_obs, cos_flux
