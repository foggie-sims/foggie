# code adopted from foggie/shader_map.py
import matplotlib as mpl
mpl.rcParams['font.family'] = 'stixgeneral'
import pandas
import numpy as np
from foggie.utils import consistency
import yt

def gas_imgx_imgy(all_data, ds_paras, obs_point='halo_center'):
    """
    Given observing point and line of sight vector (L_vec),
    calculate the projected x and y on the image plane (phi_vec, L_vec),
    and also the 3D radius from the observing point

    Return:
    image_x: project x coord on the phi vec
    image_y: project y coord on the L vec
    rr_kpc: 3D distance of cells to observing location

    History:
    10/01/2019, Yong Zheng, UCB
    10/10/2019, was originally calc_image_xy in mocky_way. rename. Yong Zheng.
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
    cos_theta_r_phi = np.dot(phi_vec, r_vec)/rr_code
    rphi_proj_kpc = rr_kpc * cos_theta_r_phi  # need to figure out the sign
    image_x = rphi_proj_kpc

    # image_y is along L direction
    cos_theta_r_L = np.dot(L_vec, r_vec)/rr_code
    rL_proj_kpc = rr_kpc * cos_theta_r_L  # need to figure out the sign
    image_y = rL_proj_kpc

    return image_x, image_y, rr_kpc

def prep_dataframe_vel_pos(all_data, ds_paras, obs_point='halo_center'):
    """
    Add fields to the dataset, create dataframe for rendering
    And select data only with positive velocity for off axis projection.

    all_data: could be all the data, or a sphere, or a box or something

    History:
    Yong Zheng created sometime in the summer of 2019
    09/23/2019, Yong Zheng change prep_dataframe to adjust to the postive velocity requirement.
    10/01/2019, Yong Zheng deleted the refine_box and halo_vcenter lines, @UCB.
                also added obs_point for off_center observers
    10/10/2019, Yong Zheng simplified the module, and merge into foggie.mocky_way.
    """

    from offaxproj_dshader import gas_imgx_imgy
    gas_imgx, gas_imgy, gas_imgr = gas_imgx_imgy(all_data, ds_paras,
                                                 obs_point=obs_point)

    # put some into categories
    los_vel = all_data['line_of_sight_velocity']
    cat_vel_pos = consistency.categorize_by_vel_pos(los_vel)
    vel_filter = los_vel >= 0

    # build dataframe with mandatory fields
    dataframe = pandas.DataFrame({'gas_imgx': gas_imgx[vel_filter],
                                  'gas_imgy': gas_imgy[vel_filter],
                                  # category entries
                                  'cat_vel_pos': cat_vel_pos[vel_filter]})
    dataframe.cat_vel_pos = dataframe.cat_vel_pos.astype('category')
    return dataframe

def prep_dataframe_vel_neg(all_data, ds_paras, obs_point='halo_center'):
    """
    Add fields to the dataset, create dataframe for rendering
    And select data only with negative velocity for off axis projection.

    all_data: could be all the data, or a sphere, or a box or something

    History:
    Yong Zheng created sometime in the summer of 2019
    09/23/2019, Yong Zheng change prep_dataframe to adjust to the postive velocity requirement.
    10/01/2019, Yong Zheng deleted the refine_box and halo_vcenter lines, @UCB.
                also added obs_point for off_center observers
    10/10/2019, Yong Zheng simplified the module, and merge into foggie.mocky_way.
    """

    from offaxproj_dshader import gas_imgx_imgy
    gas_imgx, gas_imgy, gas_imgr = gas_imgx_imgy(all_data, ds_paras,
                                                 obs_point=obs_point)

    # put some into categories
    los_vel = all_data['line_of_sight_velocity']
    cat_vel_neg = consistency.categorize_by_vel_pos(los_vel)
    vel_filter = los_vel < 0

    # build dataframe with mandatory fields
    dataframe = pandas.DataFrame({'gas_imgx': gas_imgx[vel_filter],
                                  'gas_imgy': gas_imgy[vel_filter],
                                  # category entries
                                  'cat_vel_neg': cat_vel_neg[vel_filter]})
    dataframe.cat_vel_neg = dataframe.cat_vel_neg.astype('category')
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

    from offaxproj_dshader import gas_imgx_imgy
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

    from offaxproj_dshader import gas_imgx_imgy
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

    import sys
    sim_name = sys.argv[1] #'nref11n_nref10f', 'nref11c_nref9f_selfshield_z6'
    dd_name = sys.argv[2]  # 'RD0039', 'RD0037'
    obj_tag = 'all' # all, disk, cgm
    obs_point = 'halo_center' # halo_center, or offcenter_location
    dshader_tag = 'vel_pos' # vel_pos, vel_neg, logT, metallicity

    from core_funcs import prepdata
    ds, ds_paras = prepdata(dd_name, sim_name=sim_name)
    observer_location = ds_paras[obs_point]

    ### deciding whether to do cgm, disk, or both
    print("Taking %s out of the simulation..."%(obj_tag))
    from core_funcs import obj_source_all_disk_cgm
    obj_source = obj_source_all_disk_cgm(ds, ds_paras, obj_tag)
    obj_source.set_field_parameter("observer_location", observer_location)
    if obs_point == 'halo_center':
        bv = ds_paras['disk_bulkvel']
    else:
        bv = ds_paras['observer_bulkvel']
    obj_source.set_field_parameter("observer_bulkvel", bv)

    ### Set up the phase diagram parameter
    dict_basic_args = {'x_field': 'gas_imgx',
                       'y_field': 'gas_imgy',
                       'x_range': [-20, 20],
                       'y_range': [-20, 20],
                       'x_label': r'x (kpc)',
                       'y_label': r'y (kpc)',
                       'image_x_width': 1000, # in pixels I think?
                       'image_y_width': 1000}

    if dshader_tag == 'logT':
        print("Making data shader frame... color-coded in logT ...")
        df = prep_dataframe_logT(obj_source, ds_paras, obs_point=obs_point)
        categories = consistency.logT_color_labels_mw
        dict_T_args = {'c_field': 'cat_logT',
                       'cmap': consistency.logT_discrete_cmap_mw,
                       'ckey': consistency.logT_color_key_mw,
                       'clabel': r'log [T (K)]',
                       'cticklabels': [s.decode('UTF-8').upper() for s in categories],
                       'export_path': 'figs/offaxproj_dshader',
                       'figname': '%s_%s_%s_offaxproj_logT'%(sim_name, dd_name,
                                                             obj_tag)}
        dict_extra_args = dict_T_args

    elif dshader_tag == 'metallicity':
        print("Making data shader frame... color-coded in metallicity ...")
        df = prep_dataframe_metallicity(obj_source, ds_paras, obs_point=obs_point)
        categories = consistency.metal_color_labels_mw
        dict_Z_args = {'c_field': 'cat_metallicity',
                       'cmap': consistency.metal_discrete_cmap_mw,
                       'ckey': consistency.metal_color_key_mw,
                       'clabel': r'Z (Zsun)',
                       'cticklabels': [s.decode('UTF-8').upper() for s in categories],
                       'export_path': 'figs/offaxproj_dshader',
                       'figname': '%s_%s_%s_offaxproj_metallicity'%(sim_name, dd_name,
                                                                    obj_tag)}
        dict_extra_args = dict_Z_args

    elif dshader_tag == 'vel_pos':
        print("Making data shader frame... color-coded in vel_pos ...")
        df = prep_dataframe_vel_pos(obj_source, ds_paras, obs_point=obs_point)
        categories = consistency.vel_pos_color_labels
        dict_v_args = {'c_field': 'cat_vel_pos',
                       'cmap': consistency.vel_pos_discrete_cmap,
                       'ckey': consistency.vel_pos_color_key,
                       'clabel': r'V$_{los}$ (km/s)',
                       'cticklabels': [s.decode('UTF-8').upper() for s in categories],
                       'export_path': 'figs/offaxproj_dshader',
                       'figname': '%s_%s_%s_%s_offaxproj_vel_pos'%(sim_name, dd_name,
                                                                   obj_tag, obs_point)}
        dict_extra_args = dict_v_args

    elif dshader_tag == 'vel_neg':
        print("Making data shader frame... color-coded in vel_pos ...")
        df = prep_dataframe_vel_neg(obj_source, ds_paras, obs_point=obs_point)
        categories = consistency.vel_neg_color_labels
        dict_v_args = {'c_field': 'cat_vel_neg',
                       'cmap': consistency.vel_neg_discrete_cmap,
                       'ckey': consistency.vel_neg_color_key,
                       'clabel': r'V$_{los}$ (km/s)',
                       'cticklabels': [s.decode('UTF-8').upper() for s in categories],
                       'export_path': 'figs/offaxproj_dshader',
                       'figname': '%s_%s_%s_%s_offaxproj_vel_neg'%(sim_name, dd_name,
                                                                   obj_tag, obs_point)}
        dict_extra_args = dict_v_args

    else:
        print("Do not recognize this dshader_tag, please check...")
        import sys
        sys.exit(0)

    ######
    print("Phew, finally making offax projection plots with dshader...")
    from phase_diagram import dshader_noaxes
    dshader_noaxes(df, dict_basic_args, dict_extra_args)

    print("Putting axes on the data shader plot...")
    from phase_diagram import wrap_axes
    wrap_axes(dict_basic_args, dict_extra_args)
