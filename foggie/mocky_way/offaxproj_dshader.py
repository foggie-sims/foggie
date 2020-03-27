# code adopted from foggie/shader_map.py
import matplotlib as mpl
mpl.rcParams['font.family'] = 'stixgeneral'
import pandas
import numpy as np
from foggie.utils import consistency
import yt

def gas_imgx_imgy(all_data, ds_paras, obs_point='halo_center'):
    """
    Given observing point (obs_point) and line of sight vector (usually sun_vec),
    calculate the projected x and y on the image plane (phi_vec, L_vec),
    and also the 3D radius from the observing point

    Input:
    all_data: could be a sphere, disk, or whatever
    ds_paras: a pre-defined dictionary, most important items in it are the
              three vectors: L_vec is the angular momentum, sun_vec and phi_vec
              are two orthogonal vectors on the galactic plane. sun_vec, phi_vec,
              and L_vec follows right-hand rule.

    Return:
    image_x: project x coord on the phi vec
    image_y: project y coord on the L vec
    rr_kpc: 3D distance of cells to observing location

    History:
    10/01/2019, Yong Zheng, UCB
    10/10/2019, was originally calc_image_xy in mocky_way. rename. Yong Zheng.
    10/14/2019, add a random funcs to shake up the distribution just a bit to
                get rid of the grid influence. Yong. UCB.
    """
    x = all_data['x'].in_units('code_length').flatten()
    y = all_data['y'].in_units('code_length').flatten()
    z = all_data['z'].in_units('code_length').flatten()

    observer_location = ds_paras[obs_point]
    rx_vec = x-observer_location[0]
    ry_vec = y-observer_location[1]
    rz_vec = z-observer_location[2]
    r_vec = yt.YTArray([rx_vec, ry_vec, rz_vec])

    #rr_kpc = np.sqrt((rx_vec.in_units('kpc'))**2 + \
    #                 (ry_vec.in_units('kpc'))**2 +\
    #                 (rz_vec.in_units('kpc'))**2) # kpc
    rr_code = np.sqrt(rx_vec**2 + ry_vec**2 + rz_vec**2) # code_length
    rr_kpc = rr_code.in_units('kpc')

    # we want the projection toward sun_vec (what we use in Figure 1)
    # so we are projecting the r vector on to (phi, L) plane
    # image_x is along phi direction
    phi_vec = ds_paras['phi_vec'] # unit vector
    L_vec = ds_paras['L_vec']     # unit vector

    # image_x is along phi direction
    cos_theta_r_phi = np.dot(phi_vec, r_vec)/rr_code.value
    rphi_proj_kpc = rr_kpc * cos_theta_r_phi  # need to figure out the sign
    image_x = rphi_proj_kpc

    # image_y is along L direction
    cos_theta_r_L = np.dot(L_vec, r_vec)/rr_code.value
    rL_proj_kpc = rr_kpc * cos_theta_r_L  # need to figure out the sign
    image_y = rL_proj_kpc

    ### because the simulation grids are somewhat align with x, y, z,
    # let's mix them a bit. choose 0.5 kpc because that's about the common
    # resolution cell size. (some are smaller) -- 10/14/2019, Yong, UCB.
    rand_y = image_y.value+np.random.random(image_y.size)*0.5
    rand_x = image_x.value # +np.random.random(image_x.size)*0.2

    # return image_x, image_y, rr_kpc
    return rand_x, rand_y, rr_kpc

def prep_dataframe_velocity(all_data, ds_paras, obs_point='halo_center',
                            vel_tag='outflow'):
    """
    Add fields to the dataset, create dataframe for rendering
    And select data only with negative velocity for off axis projection.

    all_data: could be all the data, or a sphere, or a box or something

    History:
    Yong Zheng created sometime in the summer of 2019
    09/23/2019, Yong Zheng change prep_dataframe to adjust to the
                postive velocity requirement.
    10/01/2019, Yong Zheng deleted the refine_box and halo_vcenter lines, @UCB.
                also added obs_point for off_center observers
    10/10/2019, Yong Zheng simplified the module, and merge into foggie.mocky_way.
    10/15/2019, merging outflow and inflow. Yong. UCB.
    11/14/2019, add outflow_inflow as another vel_tag to plot outflow and inflow
                together. Yong. UCB.
    """

    # from offaxproj_dshader import gas_imgx_imgy
    gas_imgx, gas_imgy, gas_imgr = gas_imgx_imgy(all_data, ds_paras,
                                                 obs_point=obs_point)

    # put some into categories
    los_vel = all_data['los_velocity_mw']
    if vel_tag == 'outflow':
        cat_vel = consistency.categorize_by_outflow(los_vel)
    elif vel_tag == 'inflow':
        cat_vel = consistency.categorize_by_inflow(los_vel)
    elif vel_tag == 'outflow_inflow':
        cat_vel = consistency.categorize_by_outflow_inflow(los_vel)
    else:
        print("I do not know this vel_tag, please check.")
        sys.exit()

    # build dataframe with mandatory fields
    dataframe = pandas.DataFrame({'gas_imgx': gas_imgx,
                                  'gas_imgy': gas_imgy,
                                  # category entries
                                  'cat_vel': cat_vel})
    dataframe.cat_vel = dataframe.cat_vel.astype('category')
    return dataframe

def prep_dataframe_logT(all_data, ds_paras, obs_point='halo_center'):
    """
    Add fields to the dataset, create dataframe for rendering for temperature

    all_data: could be all the data, or a sphere, or a box or something

    History:
    Yong Zheng created sometime in the summer of 2019
    09/23/2019, Yong Zheng change prep_dataframe to adjust to the postive velocity requirement.
    10/01/2019, Yong Zheng deleted the refine_box and halo_vcenter lines, @UCB.
                also added obs_point for off_center observers
    10/10/2019, Yong Zheng simplified the module, and merge into foggie.mocky_way.
    """

    # from offaxproj_dshader import gas_imgx_imgy
    gas_imgx, gas_imgy, gas_imgr = gas_imgx_imgy(all_data, ds_paras,
                                                 obs_point=obs_point)
    logT = np.log10(all_data['temperature'])
    cat_logT = consistency.categorize_by_logT_mw(logT)

    # build dataframe with mandatory fields
    dataframe = pandas.DataFrame({'gas_imgx': gas_imgx,
                                  'gas_imgy': gas_imgy,
                                  # category entries
                                  'cat_logT': cat_logT})
    dataframe.cat_logT = dataframe.cat_logT.astype('category')
    return dataframe

def prep_dataframe_metallicity(all_data, ds_paras, obs_point='halo_center'):
    """
    Add fields to the dataset, create dataframe for rendering for temperature

    all_data: could be all the data, or a sphere, or a box or something

    History:
    Yong Zheng created sometime in the summer of 2019
    09/23/2019, Yong Zheng change prep_dataframe to adjust to the postive velocity requirement.
    10/01/2019, Yong Zheng deleted the refine_box and halo_vcenter lines, @UCB.
                also added obs_point for off_center observers
    10/10/2019, Yong Zheng simplified the module, and merge into foggie.mocky_way.
    """

    # from offaxproj_dshader import gas_imgx_imgy
    gas_imgx, gas_imgy, gas_imgr = gas_imgx_imgy(all_data, ds_paras,
                                                 obs_point=obs_point)
    metal_Zsun = all_data['metallicity']
    cat_metallicity = consistency.categorize_by_metallicity_mw(metal_Zsun)

    # build dataframe with mandatory fields
    dataframe = pandas.DataFrame({'gas_imgx': gas_imgx,
                                  'gas_imgy': gas_imgy,
                                  # category entries
                                  'cat_metallicity': cat_metallicity})
    dataframe.cat_metallicity = dataframe.cat_metallicity.astype('category')
    return dataframe


#########################################
if __name__ == '__main__':

    sim_name = 'nref11n_nref10f' # 'nref11c_nref9f_selfshield_z6'
    dd_name = 'DD2175'           # 'RD0039', 'RD0037'
    obj_tag = 'all-refined' # all-refined, disk, cgm-refined
    obs_point = 'halo_center' # halo_center, or offcenter_location
    vel_tag = 'outflow_inflow' # outflow or inflow, to decide which part to plot,
                        # outflow_inflow
    dshader_tag = 'velocity' # velocity, logT, metallicity
    test = False

    from core_funcs import prepdata
    ds, ds_paras = prepdata(dd_name, sim_name=sim_name)
    observer_location = ds_paras[obs_point]

    ### deciding whether to do cgm, disk, or both
    print("Taking %s out of the simulation..."%(obj_tag))
    from core_funcs import obj_source_all_disk_cgm
    sp = obj_source_all_disk_cgm(ds, ds_paras, obj_tag, test=test)
    if obs_point == 'halo_center':
        bv = ds_paras['disk_bulkvel']
    else:
        bv = ds_paras['observer_bulkvel']
    sp.set_field_parameter("observer_bulkvel", bv)
    sp.set_field_parameter("observer_location", observer_location)

    if vel_tag == 'outflow':
        obj_source = sp.cut_region(["obj['los_velocity_mw'] >= 0"])
    elif vel_tag == 'inflow':
        obj_source = sp.cut_region(["obj['los_velocity_mw'] < 0"])
    elif vel_tag == 'outflow_inflow':
        obj_source = sp.cut_region(["(obj['los_velocity_mw'] >-400) & (obj['los_velocity_mw'] < 400)"])
    else:
        print("I have no idea what you want to proj with vel_tag, please check.")
        sys.exit()

    ### Set up the phase diagram parameter
    dict_basic_args = {'x_field': 'gas_imgx',
                       'y_field': 'gas_imgy',
                       'x_range': [-120, 120],
                       'y_range': [-120, 120],
                       #'x_range': [-20, 20], # testing
                       #'y_range': [-20, 20], # testing
                       'x_label': r'x (kpc)',
                       'y_label': r'y (kpc)',
                       'image_x_width': 1000, # in pixels I think?
                       'image_y_width': 1000,
                       'export_path': 'figs/offaxproj_dshader',
                       # edgeon2 is to be consistent with offaxproj_ytfunc
                       'figname': '%s_%s_%s_%s_%s_edgeon2.pdf'%(sim_name,
                                                                dd_name,
                                                                obs_point,
                                                                vel_tag,
                                                                dshader_tag)}

    if dshader_tag == 'logT':
        print("Making data shader frame... color-coded in logT ...")
        df = prep_dataframe_logT(obj_source, ds_paras, obs_point=obs_point)
        categories = consistency.logT_color_labels_mw
        cticklabels = [s.decode('UTF-8').upper() for s in categories]
        dict_T_args = {'c_field': 'cat_logT',
                       'cmap': consistency.logT_discrete_cmap_mw,
                       'ckey': consistency.logT_color_key_mw,
                       'clabel': r'log [T (K)]',
                       'cticklabels': cticklabels}
        dict_extra_args = dict_T_args

    elif dshader_tag == 'metallicity':
        print("Making data shader frame... color-coded in metallicity ...")
        df = prep_dataframe_metallicity(obj_source, ds_paras, obs_point=obs_point)
        categories = consistency.metal_color_labels_mw
        cticklabels = [s.decode('UTF-8').upper() for s in categories]
        dict_Z_args = {'c_field': 'cat_metallicity',
                       'cmap': consistency.metal_discrete_cmap_mw,
                       'ckey': consistency.metal_color_key_mw,
                       'clabel': r'Z (Zsun)',
                       'cticklabels': cticklabels}
        dict_extra_args = dict_Z_args

    elif dshader_tag == 'velocity':
        print("Making data shader frame... color-coded in velocity ...")
        df = prep_dataframe_velocity(obj_source, ds_paras,
                                     obs_point=obs_point,
                                     vel_tag=vel_tag)
        if vel_tag == 'outflow':
            categories = consistency.outflow_color_labels
            cmap = consistency.outflow_discrete_cmap
            ckey = consistency.outflow_color_key
        elif vel_tag == 'inflow':
            categories = consistency.inflow_color_labels
            cmap = consistency.inflow_discrete_cmap
            ckey = consistency.inflow_color_key
        elif vel_tag == 'outflow_inflow':
            categories = consistency.outflow_inflow_color_labels
            cmap = consistency.outflow_inflow_discrete_cmap
            ckey = consistency.outflow_inflow_color_key
        else:
            print("I have no idea what you want with vel_tag")
            sys.exit()

        cticklabels = [s.decode('UTF-8').upper() for s in categories]
        dict_v_args = {'c_field': 'cat_vel', 'cmap': cmap, 'ckey': ckey,
                       'clabel': r'Velo w.r.t Obs (km/s)',
                       'cticklabels': cticklabels}
        print(len(cticklabels), len(dict_v_args['ckey']))
        dict_extra_args = dict_v_args

    else:
        print("Do not recognize this dshader_tag, please check...")
        import sys
        sys.exit()

    ######
    print("Phew, finally making offax projection plots with dshader...")
    from phase_diagram import dshader_noaxes
    dshader_noaxes(df, dict_basic_args, dict_extra_args)

    print("Putting axes on the data shader plot...")
    from phase_diagram import wrap_axes
    wrap_axes(dict_basic_args, dict_extra_args)
