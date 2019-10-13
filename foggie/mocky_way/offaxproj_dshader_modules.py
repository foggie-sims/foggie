#los_velocity_mwimport os
#from mocky_way_modules import data_dir_sys_dir
#data_dir, sys_dir = data_dir_sys_dir()
#os.sys.path.insert(0, sys_dir)
#########################################

# code adopted from foggie/shader_map.py
import matplotlib as mpl
mpl.rcParams['font.family'] = 'stixgeneral'
import pandas
import yt
#import trident
import numpy as np
from astropy.table import Table
import warnings

from foggie.utils import consistency
# import foggie.cmap_utils as cmaps

def calc_image_xy(all_data, ds_paras, obs_point='halo_center'):
    """
    Given observing point and line of sight vector (L_vec),
    calculate the projected x and y on the image plane (phi_vec, L_vec),
    and also the 3D radius from the observing point

    Return:
    image_x: project x coord on the phi vec
    image_y: project y coord on the L vec
    rr_kpc: 3D distance of cells to observing location

    History:
    10/01/2019, YZ, UCB
    """
    x = all_data['x'].in_units('code_length').flatten()
    y = all_data['y'].in_units('code_length').flatten()
    z = all_data['z'].in_units('code_length').flatten()

    wrt_center = ds_paras[obs_point]
    rx_vec = x-wrt_center[0]
    ry_vec = y-wrt_center[1]
    rz_vec = z-wrt_center[2]
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
    cos_theta_r_phi = np.dot(phi_vec, r_vec)/rr_code
    rphi_proj_kpc = rr_kpc * cos_theta_r_phi  # need to figure out the sign
    image_x = rphi_proj_kpc

    # image_y is along L direction
    L_vec = ds_paras['L_vec']     # unit vector
    cos_theta_r_L = np.dot(L_vec, r_vec)/rr_code
    rL_proj_kpc = rr_kpc * cos_theta_r_L  # need to figure out the sign
    image_y = rL_proj_kpc

    return image_x, image_y, rr_kpc

def prep_dataframe(all_data, ds_paras, obs_point='halo_center', fields=[]):
    """
    Add fields to the dataset, create dataframe for rendering
    code adopted from foggie/shader_map.py

    all_data: could be all the data, or a sphere, or a box or something
    ds_paras: info about this halo, need halo_center in code_lengh unit,
              and observer_location
    fields: The enzo fields x, y, z, temperature, density, cell_vol, cell_mass,
    and metallicity will always be included, others will be included
    if they are requested as fields.

    History:
    YZ created sometime in the summer of 2019
    10/01/2019, YZ deleted the refine_box and halo_vcenter lines, @UCB.
                also added obs_point for off_center observers
    """

    from offaxproj_dshader_modules import calc_image_xy
    image_x, image_y, rr_kpc = calc_image_xy(all_data, ds_paras, obs_point=obs_point)

    density = np.log10(all_data['density'])
    temperature = np.log10(all_data['temperature'])
    mass = np.log10(all_data['cell_mass'])
    metallicity_Zsun = all_data['gas', 'metallicity']

    # put some into categories
    cat_radius = consistency.categorize_by_radius(rr_kpc)
    los_velocity_mw = all_data['los_velocity_mw']
    cat_velocity = consistency.categorize_by_velocity(los_velocity_mw)
    # want to put metallicity into a category as well

    # build dataframe with mandatory fields
    dataframe = pandas.DataFrame({'image_x':image_x,
                                  'image_y':image_y,
                                  'r':rr_kpc,\
                                  'temperature':temperature, \
                                  'density':density, \
                                  'cell_mass': mass, \
                                  'los_velocity_mw': los_velocity_mw, \
                                  # category entries
                                  'cat_radius': cat_radius, \
                                  'cat_velocity': cat_velocity})
    dataframe.cat_radius = dataframe.cat_radius.astype('category')
    dataframe.cat_velocity = dataframe.cat_velocity.astype('category')
    print('You have put these columns in dataframe: ')
    print([ss for ss in dataframe.columns])

    #  add requested fields
    if len(fields) > 0:
        print("Now adding requested fields: ", fields)
        for ifield in fields:
            if ifield not in dataframe.columns:
                print("No field=%s in dataframe, adding it."%(ifield))
                if ifield in consistency.logfields:
                    print("%s is a log field."%(ifield))
                    dataframe[ifield] = np.log10(all_data[ifield])
                elif 'density' in ifield:
                    print("%s is a density field, take log."%(ifield))
                    dataframe[ifield] = np.log10(all_data[ifield])
                else:
                    if ('vel' in ifield):
                        print("%s is a velocity field, add km/s unit."%(ifield))
                        dataframe[ifield] = all_data[ifield].in_units('km/s')
                    else:
                        dataframe[ifield] = all_data[ifield]

    return dataframe

def prep_dataframe_vel_pos(all_data, ds_paras, obs_point='halo_center', fields=[]):
    """
    Add fields to the dataset, create dataframe for rendering
    code adopted from foggie/shader_map.py.
    And select data only with positive velocity

    all_data: could be all the data, or a sphere, or a box or something
    ds_paras: info about this halo, need halo_center in code_lengh unit,
              and observer_location
    fields: The enzo fields x, y, z, temperature, density, cell_vol, cell_mass,
    and metallicity will always be included, others will be included
    if they are requested as fields.

    History:
    YZ created sometime in the summer of 2019
    09/23/2019, YZ change prep_dataframe to adjust to the postive velocity requirement.
    10/01/2019, YZ deleted the refine_box and halo_vcenter lines, @UCB.
                also added obs_point for off_center observers
    """

    print("Preparing dataframe wrt %s" %(obs_point))
    from offaxproj_dshader_modules import calc_image_xy
    image_x, image_y, rr_kpc = calc_image_xy(all_data, ds_paras, obs_point=obs_point)

    density = np.log10(all_data['density'])
    temperature = np.log10(all_data['temperature'])
    mass = np.log10(all_data['cell_mass'])
    metallicity_Zsun = all_data['gas', 'metallicity']

    # put some into categories
    cat_radius = consistency.categorize_by_radius(rr_kpc)
    los_velocity_mw = all_data['los_velocity_mw']
    keep_ind_vel_pos = los_velocity_mw >= 0
    cat_vel_pos = consistency.categorize_by_vel_pos(los_velocity_mw)

    # build dataframe with mandatory fields
    dataframe = pandas.DataFrame({'image_x':image_x[keep_ind_vel_pos],
                                  'image_y':image_y[keep_ind_vel_pos],
                                  'r':rr_kpc[keep_ind_vel_pos],\
                                  'temperature':temperature[keep_ind_vel_pos],\
                                  'density':density[keep_ind_vel_pos], \
                                  'cell_mass': mass[keep_ind_vel_pos], \
                                  # category entries
                                  'cat_radius': cat_radius[keep_ind_vel_pos],\
                                  'cat_vel_pos': cat_vel_pos[keep_ind_vel_pos]})
    dataframe.cat_radius = dataframe.cat_radius.astype('category')
    dataframe.cat_vel_pos = dataframe.cat_vel_pos.astype('category')
    print('You have put these columns in dataframe: ')
    print([ss for ss in dataframe.columns])

    # now add additionally requested fields
    if len(fields) > 0:
        print("Now adding requested fields: ", fields)
        for ifield in fields:
            if ifield not in dataframe.columns:
                print("No field=%s in dataframe, adding it."%(ifield))
                if ifield in consistency.logfields:
                    print("%s is a log field."%(ifield))
                    dataframe[ifield] = np.log10(all_data[ifield][keep_ind_vel_pos])
                elif 'density' in ifield:
                    print("%s is a density field, take log."%(ifield))
                    dataframe[ifield] = np.log10(all_data[ifield][keep_ind_vel_pos])
                else:
                    if ('vel' in ifield):
                        print("%s is a velocity field, add km/s unit."%(ifield))
                        vel_field = all_data[ifield].in_units('km/s')
                        dataframe[ifield] = vel_field[keep_ind_vel_pos]
                    else:
                        dataframe[ifield] = all_data[ifield][keep_ind_vel_pos]

    return dataframe

def prep_dataframe_vel_neg(all_data, ds_paras, obs_point='halo_center', fields=[]):
    """
    Add fields to the dataset, create dataframe for rendering
    code adopted from foggie/shader_map.py.
    And select data only with negative velocity

    all_data: could be all the data, or a sphere, or a box or something
    ds_paras: info about this halo, need halo_center in code_lengh unit,
              and observer_location
    fields: The enzo fields x, y, z, temperature, density, cell_vol, cell_mass,
    and metallicity will always be included, others will be included
    if they are requested as fields.

    History:
    YZ created sometime in the summer of 2019
    09/23/2019, YZ change prep_dataframe to adjust to the negative velocity requirement.
    10/01/2019, YZ deleted the refine_box and halo_vcenter lines, @UCB.
                also added obs_point for off_center observers
    """

    print("Preparing dataframe wrt %s" %(obs_point))
    from offaxproj_dshader_modules import calc_image_xy
    image_x, image_y, rr_kpc = calc_image_xy(all_data, ds_paras, obs_point=obs_point)

    density = np.log10(all_data['density'])
    temperature = np.log10(all_data['temperature'])
    mass = np.log10(all_data['cell_mass'])
    metallicity_Zsun = all_data['gas', 'metallicity']

    # put some into categories
    cat_radius = consistency.categorize_by_radius(rr_kpc)
    los_velocity_mw = all_data['los_velocity_mw']
    keep_ind_vel_neg = los_velocity_mw < 0
    cat_vel_neg = consistency.categorize_by_vel_neg(los_velocity_bk)

    # build dataframe with mandatory fields
    dataframe = pandas.DataFrame({'image_x':image_x[keep_ind_vel_neg], \
                                  'image_y':image_y[keep_ind_vel_neg], \
                                  'r':rr_kpc[keep_ind_vel_neg],\
                                  'temperature':temperature[keep_ind_vel_neg],\
                                  'density':density[keep_ind_vel_neg], \
                                  'cell_mass': mass[keep_ind_vel_neg], \
                                  # category entries
                                  'cat_radius': cat_radius[keep_ind_vel_neg],\
                                  'cat_vel_neg': cat_vel_neg[keep_ind_vel_neg]})
    dataframe.cat_radius = dataframe.cat_radius.astype('category')
    dataframe.cat_vel_neg = dataframe.cat_vel_neg.astype('category')
    print('You have put these columns in dataframe: ')
    print([ss for ss in dataframe.columns])

    # now add additionally requested fields
    if len(fields) > 0:
        print("Now adding requested fields: ", fields)
        for ifield in fields:
            if ifield not in dataframe.columns:
                print("No field=%s in dataframe, adding it."%(ifield))
                if ifield in consistency.logfields:
                    print("%s is a log field."%(ifield))
                    dataframe[ifield] = np.log10(all_data[ifield][keep_ind_vel_neg])
                elif 'density' in ifield:
                    print("%s is a density field, take log."%(ifield))
                    dataframe[ifield] = np.log10(all_data[ifield][keep_ind_vel_neg])
                else:
                    if ('vel' in ifield):
                        print("%s is a velocity field, add km/s unit."%(ifield))
                        vel_field = all_data[ifield].in_units('km/s')
                        dataframe[ifield] = vel_field[keep_ind_vel_neg]
                    else:
                        dataframe[ifield] = all_data[ifield][keep_ind_vel_neg]

    return dataframe

def get_df_dicts(dshader_field, dd_name, location_tag):
    dict_cat_field = {'velocity': consistency.velocity_df_colname,
                  'vel_pos': consistency.vel_pos_df_colname,
                  'vel_neg': consistency.vel_neg_df_colname,
                  'radius': consistency.radius_df_colname}

    dict_c_label = {'velocity': r'V$_{los}$ (km/s)',
                'vel_pos': r'V$_{los}$ (km/s)',
                'vel_neg': r'V$_{los}$ (km/s)',
                'radius': 'R (kpc)'}

    dict_discrete_cmap = {'velocity': consistency.velocity_discrete_cmap,
                      'vel_pos': consistency.vel_pos_discrete_cmap,
                      'vel_neg': consistency.vel_neg_discrete_cmap,
                      'radius': consistency.radius_discrete_cmap}

    dict_categories = {'velocity': consistency.velocity_color_labels,
                   'vel_pos': consistency.vel_pos_color_labels,
                   'vel_neg': consistency.vel_neg_color_labels,
                   'radius': consistency.radius_color_labels}

    cat_field = dict_cat_field[dshader_field]
    c_label = dict_c_label[dshader_field]
    discrete_cmap = dict_discrete_cmap[dshader_field]
    categories = dict_categories[dshader_field]

    return cat_field, c_label, discrete_cmap, categories

def locate_xyticks(ax_range, ax_step=1, img_ax_width=1000):
    """
    Func used by wrap_axes to setup x/y axes tickes and ticklabels,
    by converting pixels to real units

    ax_range: [min, max] of x or y axes quantity, usually in log
    ax_step: how far you want the ticks to be, usually in log
    img_ax_width: inherited originally from phase_diagram_noaxes, to scale
            img pixel size to real units.
    """
    ax_min, ax_max = ax_range[0], ax_range[1]
    ax_step = 10 # for the offaxis projection plots, in unit  of kpc
    # if (ax_max > 10.): ax_step = 10
    # if (ax_max > 100.): ax_step = 100
    ax_frac = np.arange((ax_max-ax_min) + 1., step=ax_step)/(ax_max-ax_min)
    ticks = ax_frac*img_ax_width
    ticklabels = [str(int(ii)) for ii in ax_frac*(ax_max-ax_min)+ax_min]
    if len(ticklabels)>10: # if too crowded
        for j in range(len(ticklabels)):
            if j % 5 == 0:
                ticklabels[j] = ''
    return ticks, ticklabels


def offaxproj_noaxes(dataframe, x_field="density", x_range=[-15, 2],
                         y_field="temperature", y_range=[2, 8],
                         img_x_width=1000, img_y_width=1000,
                         cat_field="metallcity",
                         export_path='./', save_to_file='test.pdf'):
    """
    Renders density and temperature phase diagram with linear aggregation

    x_field: the x axis of the diagram, usually density or I would use
             H_number_density, in log
    y_field: usually temperature, in log
    cat_field: the field that will be aggregated over for each pair of
                  (x_field, y_field) point in ds canvas.
    img_x_width: the size of image, in pixels maybe, Same as img_y_width, will
                 be used in plot_width parameter of datashader.Canvas

    adopted from foggie/shader_maps.py/render_image func
    """
    import sys
    from functools import partial
    from datashader.utils import export_image
    import datashader as dshader
    import datashader.transfer_functions as tranfunc

    # check if field xyz exist in the data frame:
    for ifield in [x_field, y_field, cat_field]:
        if ifield not in dataframe.columns:
            print('No %s in dataframe, please check.'%ifield)
            sys.exit(0)

    # now make a canvas
    export = partial(export_image, background='white', export_path=export_path)
    cvs = dshader.Canvas(plot_width=img_x_width, plot_height=img_y_width,
                         x_range=x_range, y_range=y_range)
    agg = cvs.points(dataframe, x_field, y_field, dshader.count_cat(cat_field))

    all_color_keys = {'cat_frac': consistency.ion_frac_color_key, \
                      'cat_phase': consistency.new_phase_color_key, \
                      'cat_metal': consistency.new_metals_color_key, \
                      'cat_radius': consistency.radius_color_key, \
                      'cat_velocity': consistency.velocity_color_key, \
                      'cat_vel_pos': consistency.vel_pos_color_key, \
                      'cat_vel_neg': consistency.vel_neg_color_key}
    try:
        color_key = all_color_keys[cat_field]
    except KeyError:
        print("No color key for field=%s, please check."%(cat_field))
        sys.exit(0)

    img = tranfunc.shade(agg, color_key=color_key, how='log', min_alpha=230)
    export(img, save_to_file)
    return img

def wrap_axes(filename, discrete_cmap, x_range=[-15, 2], y_range=[2, 8],
              x_label=r'log n$_H$ (cm$^{-3}$)', y_label='log Temperature [K]',
              img_x_width=1000, img_y_width=1000, c_label='', c_ticklabels=[],
              draw_pressure=False,
              p_args={'T1': 1e6, 'T2': 10**7.5, 'fontsize': 14, 'rotation': -60}):
    """
    Intended to be run after phase_diagram_noaxes(i.e., render_image),
    take the image and wraps it in axes using matplotlib.

    filename: .png image file generated with phase_diagram_noaxes
    discrete_cmap: from sns color palette
    x_range: [xmin, xmax] of x axis in the phase diagram, usually for lognH or
              log Density
    y_range: [ymin, ymax], usually log Temperature
    x_label: name of x axis
    y_label: name of y axis
    img_x_width: size of image in pixels, from phase_diagram_noaxes, to scale
            image from pixel to real unit. Same for img_y_width.
    c_label: the name of the colorbar quantity
    c_ticklables: the names from the category from dataframe
    p_args: if draw_pressure: draw a constant pressure line on the plot
            with ploting arguments defined in p_args.
    """

    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    img = mpimg.imread(filename)
    # print('IMG', np.shape(img[:,:,0:3]))
    fig = plt.figure(figsize=(8,8), dpi=300)

    # main ax for image
    ax = fig.add_axes([0.1, 0.13, 0.7, 0.7])
    ax.imshow(np.flip(img[:,:,0:3],0), alpha=1., origin='lower')

    # x axis
    x_step = 1
    xticks, xticklabels = locate_xyticks(x_range, img_ax_width=img_x_width, ax_step=x_step)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=20)
    ax.set_xlabel(x_label, fontsize=22)

    # y axis
    y_step = 1
    yticks, yticklabels = locate_xyticks(y_range, img_ax_width=img_y_width, ax_step=y_step)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=20)
    ax.set_ylabel(y_label, fontsize=22)

    # ax for colorbar
    # first make a ghost image hidden behind actual color bar, this is an ugly trick..
    fxax = fig.add_axes([0.82, 0.14, 0.01, 0.01])
    n_colors = discrete_cmap.N
    fx1, fx2 = np.meshgrid(np.arange(n_colors+1), np.arange(n_colors+1))
    fximg = fxax.imshow(fx1, cmap=discrete_cmap)
    fxax.set_axis_off()

    # real color bar
    cax = fig.add_axes([0.82, 0.13, 0.03, 0.7])
    cb = fig.colorbar(fximg, cax=cax, orientation='vertical',
                      format='%d', ticks=np.mgrid[0:n_colors+1:1]+0.5)
    clbs = cb.ax.set_yticklabels(c_ticklabels, fontsize=15)  # vertically oriented colorbar
    cax.text(0, n_colors*1.01, c_label, fontsize=15)

    # draw a constant pressure line and arrow
    if draw_pressure == True:
        # find two pairs of points with the same pressure
        import astropy.units as u
        import astropy.constants as const

        T1 = p_args['T1']*u.K
        nH1 = 1e1/u.cm**3
        p = nH1*const.k_B*T1

        T2 = p_args['T2']*u.K
        nH2 = p/T2/const.k_B

        # convert the two points from phy units to pixel units
        x1 = x_step/(x_range[1]-x_range[0])*(np.log10(nH1.value)-x_range[0])*img_x_width
        y1 = y_step/(y_range[1]-y_range[0])*(np.log10(T1.value)-y_range[0])*img_y_width
        x2 = x_step/(x_range[1]-x_range[0])*(np.log10(nH2.value)-x_range[0])*img_x_width
        y2 = y_step/(y_range[1]-y_range[0])*(np.log10(T2.value)-y_range[0])*img_y_width
        ax.text(x2, y2, r'${\rm P = n_H k_B T = const.}$',
                rotation=p_args['rotation'], fontsize=p_args['fontsize'])
        ax.plot([x1, x2], [y1, y2], lw=2, color='k')
    plt.savefig('%s.pdf'%(filename[:-4]))

    import warnings
    warnings.warn('You should always check if colors and c_labels are consistent.')
