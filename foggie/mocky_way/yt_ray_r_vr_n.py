
def yt_ray_info(ds, ds_paras, los_rs, los_re, ion_list):
    # yt_ray only has t, xyz, xyz-vel, temperature, metallicity, H_number_density
    yt_ray = ds.ray(los_rs, los_re)
    yt_ray.set_field_parameter("observer_location", ds_paras["offcenter_location"])
    yt_ray.set_field_parameter("observer_bulkvel", ds_paras["offcenter_bulkvel"])

    # the distance between each cell and the observer
    # rule, for x/y/z, vx, vy, vz along the ray, they are all wrt the observer's frame.
    yt_sort = np.argsort(yt_ray["t"]) # from small r to large r
    yt_ray_x = yt_ray["gas", "x"].in_units("code_length") - los_rs[0]
    yt_ray_y = yt_ray["gas", "y"].in_units("code_length") - los_rs[1]
    yt_ray_z = yt_ray["gas", "z"].in_units("code_length") - los_rs[2]
    yt_ray_r = np.sqrt(yt_ray_x**2 + yt_ray_y**2 + yt_ray_z**2).in_units("kpc")
    yt_ray_r = yt_ray_r[yt_sort]
    yt_ray_dr = yt_ray_r[1:] - yt_ray_r[:-1]

    # quantities from derived fields
    ray_info = {}
    ray_info['los_dr'] = yt_ray_dr
    ray_info['los_r'] = yt_ray_r[:-1]
    ray_info['los_vr'] = yt_ray["gas", "los_velocity_mw"][yt_sort][:-1]
    ray_info['T'] = yt_ray['temperature'][yt_sort][:-1]
    ray_info['Z'] = yt_ray['metallicity'][yt_sort][:-1]


    from foggie.utils import consistency
    for ion in ion_list:
        ion_field = consistency.species_dict[ion]
        ray_info['n_%s'%(ion)] = yt_ray[ion_field][yt_sort][:-1]

    return ray_info

def logN_low_high_v(ion, ray_info, vmin=-100, vmax=100):
    """
    ray_info is a dictionary that includes T, los_vr, los_r, los_dr, Z, and
    ion column density. ion could be, e.g., SiIV
    """
    vr = ray_info['los_vr'].in_units('km/s').value
    dr = ray_info['los_dr'].in_units('cm').value
    n = ray_info['n_%s'%(ion)].in_units('1/cm**3').value

    ind_lowv = np.all([vr>=vmin, vr<=vmax], axis=0)
    ind_highv = np.logical_not(ind_lowv)

    N_lowv = (n[ind_lowv]*dr[ind_lowv]).sum()
    N_highv = (n[ind_highv]*dr[ind_highv]).sum()
    f_low_to_high = N_lowv / N_highv

    return f_low_to_high

if __name__ == '__name__':
    # generate random sightlines and calculate the low and high velocity logN ratio
    import sys
    import numpy as np
    from foggie.mocky_way.core_funcs import calc_ray_end

    nlos = np.float(sys.argv[1])

    from foggie.mocky_way.core_funcs import prepdata
    sim_name = 'nref11n_nref10f'
    dd_name = 'DD2175'
    ds, ds_paras = prepdata(dd_name)

    los_r_kpc = 160 # rvir
    los_rs = ds_paras['offcenter_location'].copy()

    ion_list = ['HI', 'SiIV']
    for i in range(nlos):
        los_l_deg = np.random.uniform(low=0., high=360.)
        los_b_deg = np.random.uniform(low=-90., high=90.)
        los_re, unit_vec = calc_ray_end(ds, ds_paras, los_l_deg, los_b_deg,
                                        los_rs, los_r_kpc)

        ray_info = yt_ray_info(ds, ds_paras, los_rs, los_re, ion_list)
        for ion in ion_list:
            f_low_to_high = logN_low_high_v(ion, ray_info, vmin=-100, vmax=100)
