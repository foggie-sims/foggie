def find_r200(dd_name, sim_name):
    """
    Find rvir (proper) for a given halo.
    Example1:
    > Command line: python find_r200.py nref11n_nref10f RD0039
    Example2:
    > In python
    > from foggie.mocky_way.find_r200 import find_r200
    > rvir = find_r200(dd_name, sim_name)

    History:
    10/04/2019, Yong Zheng, UCB
    10/06/2019, Yong Zheng added __name__ part so it can be used both as a module and from
                command line.
    """

    import yt
    from foggie.mocky_way.core_funcs import data_dir_sys_dir
    from foggie.mocky_way.core_funcs import find_halo_center_yz
    from foggie.mocky_way.core_funcs import calc_r200_proper
    data_dir, sys_dir = data_dir_sys_dir()

    ds_file = '%s/%s/%s/%s'%(data_dir, sim_name, dd_name, dd_name)
    if os.path.isfile(ds_file) == False:
        drive_dir = '/Volumes/Yong4TB/foggie/halo_008508'
        ds_file = '%s/%s/%s/%s'%(drive_dir, sim_name, dd_name, dd_name)
        
    ds = yt.load(ds_file)
    zsnap = ds.get_parameter('CosmologyCurrentRedshift')

    ## find halo center
    halo_center = find_halo_center_yz(ds, zsnap, sim_name, data_dir)

    ## now find r200
    print('\n'*3)
    print('OK, let us find the rvir for %s/%s!!!!'%(sim_name, dd_name))
    rvir_proper = calc_r200_proper(ds, halo_center,
                                   start_rad=120,  # in unit of 50 kpc
                                   delta_rad_coarse=20,
                                   delta_rad_fine=5,
                                   delta_rad_tiny=0.5)

    print("*** OK, now you have rvir for %s/%s"%(sim_name, dd_name))
    print("    go ahead add this rvir to core_funcs.dict_rvir_proper")
    return rvir_proper

if __name__ == "__main__":
    import sys
    sim_name = sys.argv[1]
    dd_name = sys.argv[2]
    rvir_proper = find_r200(dd_name, sim_name)
