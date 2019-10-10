# HistoryL
# adapted from phase diagram codes from mocky_way, now merging into
# foggie.mocky_way
# 10/09/2019, Yong Zheng, UCB.

import numpy as np
import pandas
from foggie.utils import consistency # for plotting

def prep_dataframe_radius(all_data, observer_location):
    """
    Add fields to the dataset, create dataframe for rendering
    code adopted from foggie/shader_map.py

    all_data: could be all the data, or a sphere, or a box or something
    observer_location: coordinates of where the observer is in code_length,
          ususally ds_paras['halo_center'] or ds_paras['offcenter_location']

    History:
    Yong Zheng created sometime in the summer of 2019
    10/01/2019, Yong Zheng deleted the refine_box and halo_vcenter lines, @UCB.
                also added obs_point for off_center observers
    10/09/2019, Yong Zheng merging all vel, pos vel, and neg vel into the same func.

    """

    ## calculate the distance between cells and observer
    x_from_obs = all_data['x'].in_units('code_length') - observer_location[0]
    y_from_obs = all_data['y'].in_units('code_length') - observer_location[1]
    z_from_obs = all_data['z'].in_units('code_length') - observer_location[2]
    r_from_obs = np.sqrt((x_from_obs.in_units('kpc'))**2 + \
                         (y_from_obs.in_units('kpc'))**2 + \
                         (z_from_obs.in_units('kpc'))**2) # kpc
    log_nH = np.log10(all_data['H_nuclei_density'])
    log_T = np.log10(all_data['temperature'])

    # make radius categories
    cat_radius = consistency.categorize_by_radius(r_from_obs)

    # make velocity categories, build dataframe with all velocity
    df = pandas.DataFrame({'radius': r_from_obs, \
                           'log_T': log_T, \
                           'log_nH': log_nH, \
                           # category entries
                           'cat_radius': cat_radius})
    df.cat_radius = df.cat_radius.astype('category')

    return df

def prep_dataframe_vel_all(all_data, observer_location):
    """
    Add fields to the dataset, create dataframe for rendering
    code adopted from foggie/shader_map.py

    all_data: could be all the data, or a sphere, or a box or something
    observer_location: coordinates of where the observer is in code_length,
          ususally ds_paras['halo_center'] or ds_paras['offcenter_location']

    History:
    Yong Zheng created sometime in the summer of 2019
    10/01/2019, Yong Zheng deleted the refine_box and halo_vcenter lines, @UCB.
                also added obs_point for off_center observers
    10/09/2019, Yong Zheng merging all vel, pos vel, and neg vel into the same func.

    """

    log_nH = np.log10(all_data['H_nuclei_density'])
    log_T = np.log10(all_data['temperature'])
    los_vel = all_data['line_of_sight_velocity']

    # make radius categories
    cat_vel = consistency.categorize_by_velocity(los_vel)

    # make velocity categories, build dataframe with all velocity
    df = pandas.DataFrame({'log_T': log_T, \
                           'log_nH': log_nH, \
                           'los_vel': los_vel, \
                           # category entries
                           'cat_vel_all': cat_vel})
    df.cat_vel_all = df.cat_vel_all.astype('category')

    return df

def prep_dataframe_vel_pos(all_data, observer_location):
    """
    Add fields to the dataset, create dataframe for rendering
    code adopted from foggie/shader_map.py

    all_data: could be all the data, or a sphere, or a box or something
    observer_location: coordinates of where the observer is in code_length,
          ususally ds_paras['halo_center'] or ds_paras['offcenter_location']

    History:
    Yong Zheng created sometime in the summer of 2019
    10/01/2019, Yong Zheng deleted the refine_box and halo_vcenter lines, @UCB.
                also added obs_point for off_center observers
    10/09/2019, Yong Zheng merging all vel, pos vel, and neg vel into the same func.

    """

    log_nH = np.log10(all_data['H_nuclei_density'])
    log_T = np.log10(all_data['temperature'])
    los_vel = all_data['line_of_sight_velocity']

    # make radius categories
    cat_vel = consistency.categorize_by_vel_pos(los_vel)

    # make velocity categories, build dataframe with only positive velocity gases
    filter = los_vel>=0
    df = pandas.DataFrame({'log_T': log_T[filter], \
                           'log_nH': log_nH[filter], \
                           'los_vel': los_vel[filter], \
                           # category entries
                           'cat_vel_pos': cat_vel[filter]})
    df.cat_vel_pos = df.cat_vel_pos.astype('category')

    return df

def prep_dataframe_vel_neg(all_data, observer_location):
    """
    Add fields to the dataset, create dataframe for rendering
    code adopted from foggie/shader_map.py

    all_data: could be all the data, or a sphere, or a box or something
    observer_location: coordinates of where the observer is in code_length,
          ususally ds_paras['halo_center'] or ds_paras['offcenter_location']

    History:
    Yong Zheng created sometime in the summer of 2019
    10/01/2019, Yong Zheng deleted the refine_box and halo_vcenter lines, @UCB.
                also added obs_point for off_center observers
    10/09/2019, Yong Zheng merging all vel, pos vel, and neg vel into the same func.

    """

    log_nH = np.log10(all_data['H_nuclei_density'])
    log_T = np.log10(all_data['temperature'])
    los_vel = all_data['line_of_sight_velocity']

    # make radius categories
    cat_vel = consistency.categorize_by_vel_neg(los_vel)

    # make velocity categories, build dataframe with only positive velocity gases
    filter = los_vel<0
    df = pandas.DataFrame({'log_T': log_T[filter], \
                           'log_nH': log_nH[filter], \
                           'los_vel': los_vel[filter], \
                           # category entries
                           'cat_vel_neg': cat_vel[filter]})
    df.cat_vel_neg = df.cat_vel_neg.astype('category')

    return df

def dshader_noaxes(dataframe, dict_basic_args, dict_extra_args):
    """
    Renders density and temperature phase diagram with linear aggregation

    x_field: the x axis of the diagram, usually density or I would use
             H_number_density, in log
    y_field: usually temperature, in log
    cat_field: the field that will be aggregated over for each pair of
                  (x_field, y_field) point in ds canvas.
    img_x_width: the size of image, in pixels maybe, Same as img_y_width, will
                 be used in plot_width parameter of datashader.Canvas

    History:
    sometime Yong Zheng adopted from foggie/shader_maps.py/render_image func
    10/09/2019, Yong Zheng merging this into foggie.mocky_way
    """
    import sys
    from functools import partial
    from datashader.utils import export_image
    import datashader as dshader
    import datashader.transfer_functions as tranfunc

    # now make a canvas
    cvs = dshader.Canvas(plot_width=dict_basic_args['image_x_width'],
                         plot_height=dict_basic_args['image_y_width'],
                         x_range=dict_basic_args['x_range'],
                         y_range=dict_basic_args['y_range'])
    agg = cvs.points(dataframe,
                     dict_basic_args['x_field'],
                     dict_basic_args['y_field'],
                     dshader.count_cat(dict_extra_args['c_field']))
    color_key = dict_extra_args['ckey']
    img = tranfunc.shade(agg, color_key=color_key, how='log', min_alpha=230)

    export = partial(export_image,
                     background='white',
                     export_path=dict_extra_args['export_path'])
    export(img, dict_extra_args['figname'])
    print("Saving to %s/%s.png (on axes)"%(dict_extra_args['export_path'],
                                           dict_extra_args['figname']))

def wrap_axes(dict_basic_args, dict_extra_args, draw_pressure_line=False):
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
    import matplotlib as mpl
    mpl.rcParams['font.family'] = 'stixgeneral'

    figname = '%s/%s.png'%(dict_extra_args['export_path'],
                           dict_extra_args['figname'])
    img = mpimg.imread(figname)
    # print('IMG', np.shape(img[:,:,0:3]))
    fig = plt.figure(figsize=(8,8), dpi=300)

    # main ax for image
    ax = fig.add_axes([0.1, 0.13, 0.7, 0.7])
    ax.imshow(np.flip(img[:,:,0:3],0), alpha=1., origin='lower')

    # x axis
    x_step = 1
    xticks, xticklabels = locate_xyticks(dict_basic_args['x_range'],
                                         img_ax_width=dict_basic_args['image_x_width'],
                                         ax_step=x_step)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=20)
    ax.set_xlabel(dict_basic_args['x_label'], fontsize=22)

    # y axis
    y_step = 1
    yticks, yticklabels = locate_xyticks(dict_basic_args['y_range'],
                                         img_ax_width=dict_basic_args['image_y_width'],
                                         ax_step=y_step)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=20)
    ax.set_ylabel(dict_basic_args['y_label'], fontsize=22)

    # ax for colorbar
    # first make a ghost image hidden behind actual color bar, this is an ugly trick..
    fxax = fig.add_axes([0.82, 0.14, 0.01, 0.01])
    n_colors = dict_extra_args['cmap'].N
    fx1, fx2 = np.meshgrid(np.arange(n_colors+1), np.arange(n_colors+1))
    fximg = fxax.imshow(fx1, cmap=dict_extra_args['cmap'])
    fxax.set_axis_off()

    # real color bar
    cax = fig.add_axes([0.82, 0.13, 0.03, 0.7])
    cb = fig.colorbar(fximg, cax=cax, orientation='vertical',
                      format='%d', ticks=np.mgrid[0:n_colors+1:1]+0.5)
    clbs = cb.ax.set_yticklabels(dict_extra_args['cticklabels'], fontsize=15)  # vertically oriented colorbar
    cax.text(0, n_colors*1.01, dict_extra_args['clabel'], fontsize=15)


    # draw a constant pressure line and arrow
    if draw_pressure_line == True:
        p_args={'T1': 1e6, 'T2': 10**7.5, 'fontsize': 14, 'rotation': -59}
        x_range = dict_basic_args['x_range']
        y_range = dict_basic_args['y_range']
        img_x_width = dict_basic_args['image_x_width']
        img_y_width = dict_basic_args['image_y_width']

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

    figname = '%s/%s.pdf'%(dict_extra_args['export_path'],
                           dict_extra_args['figname'])
    plt.savefig(figname)

    import warnings
    warnings.warn('You should always check if colors and c_labels are consistent.')

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
    if (ax_max > 10.): ax_step = 10
    if (ax_max > 100.): ax_step = 100
    ax_frac = np.arange((ax_max-ax_min) + 1., step=ax_step)/(ax_max-ax_min)
    ticks = ax_frac*img_ax_width
    ticklabels = [str(int(ii)) for ii in ax_frac*(ax_max-ax_min)+ax_min]
    if len(ticklabels)>10: # if too crowded
        for j in range(len(ticklabels)):
            if j % 2 == 0:
                ticklabels[j] = ''
    return ticks, ticklabels

def yt_phase_diagram_sanity_check(ds, ds_paras):
    """
    To double check if dshader and the axes ticks are right.

    History:
    10/10/2019, Yong Zheng. UCB.
    """
    my_sphere = ds.sphere(ds_paras["halo_center"], ds_paras['rvir'])
    plot = yt.PhasePlot(my_sphere, "H_nuclei_density", "temperature", ["cell_mass"],
                        weight_field=None)
    plot.save('figs/phase_diagram/%s_%s_yt_phase_diagram.png'%(ds_paras['sim_name'],
                                                               ds_paras['dd_name']))

#################################################################
if __name__=='__main__':
    dd_name = 'RD0039'
    sim_name = 'nref11n_nref10f'
    obj_tag = 'cgm' # all, disk, cgm
    obs_point = 'halo_center' # halo_center, or offcenter_location
    dshader_tag = 'vel_neg' # radius, vel_pos, vel_neg

    print("hey!")
    from core_funcs import prepdata
    ds, ds_paras = prepdata(dd_name, sim_name=sim_name)
    observer_location = ds_paras[obs_point]

    # yt_phase_diagram_sanity_check(ds, ds_paras)

    ### deciding whether to do cgm, disk, or both
    print("Taking %s out of the simulatioin..."%(obj_tag))
    from core_funcs import obj_source_all_disk_cgm
    obj_source = obj_source_all_disk_cgm(ds, ds_paras, obj_tag)
    obj_source.set_field_parameter("observer_location", observer_location)
    if obs_point == 'halo_center':
        bv = ds_paras['disk_bulkvel']
    else:
        bv = ds_paras['observer_bulkvel']
    obj_source.set_field_parameter("observer_bulkvel", bv)

    ### Set up the phase diagram parameter
    dict_basic_args = {'x_field': 'log_nH',
                       'y_field': 'log_T',
                       'x_range': [-8, 2],
                       'y_range': [1, 8],
                       'x_label': r'log n$_H$ (cm$^{-3}$)',
                       'y_label': 'log Temperature [K]',
                       'image_x_width': 1000, # in pixels I think?
                       'image_y_width': 1000}

    ### ok, all prep work done, now actually making phase diagram, with radius
    if dshader_tag == 'radius':
        print("Making data shader frame... color-coded in radius...")
        df_radius = prep_dataframe_radius(obj_source, observer_location)
        categories = consistency.radius_color_labels
        dict_r_args = {'c_field': 'cat_radius',
                       'cmap': consistency.radius_discrete_cmap,
                       'ckey': consistency.radius_color_key,
                       'clabel': 'R (kpc)',
                       'cticklabels': [s.decode('UTF-8').upper() for s in categories],
                       'export_path': 'figs/phase_diagram',
                       'figname': '%s_%s_%s_%s_lognH_logT_r'%(sim_name, dd_name, obj_tag, obs_point)}
        df = df_radius
        dict_extra_args = dict_r_args

    elif dshader_tag == 'vel_pos':
        print("Making data shader frame... color-coded in vel_pos ...")
        df_vel_pos = prep_dataframe_vel_pos(obj_source, observer_location)
        categories = consistency.vel_pos_color_labels
        dict_v_args = {'c_field': 'cat_vel_pos',
                       'cmap': consistency.vel_pos_discrete_cmap,
                       'ckey': consistency.vel_pos_color_key,
                       'clabel': r'V$_{los}$ (km/s)',
                       'cticklabels': [s.decode('UTF-8').upper() for s in categories],
                       'export_path': 'figs/phase_diagram',
                       'figname': '%s_%s_%s_%s_lognH_logT_vpos'%(sim_name, dd_name, obj_tag, obs_point)}

        df = df_vel_pos
        dict_extra_args = dict_v_args

    elif dshader_tag == 'vel_neg':
        print("Making data shader frame... color-coded in vel_neg ...")
        df_vel_neg = prep_dataframe_vel_neg(obj_source, observer_location)
        categories = consistency.vel_neg_color_labels
        dict_v_args = {'c_field': 'cat_vel_neg',
                       'cmap': consistency.vel_neg_discrete_cmap,
                       'ckey': consistency.vel_neg_color_key,
                       'clabel': r'V$_{los}$ (km/s)',
                       'cticklabels': [s.decode('UTF-8').upper() for s in categories],
                       'export_path': 'figs/phase_diagram',
                       'figname': '%s_%s_%s_%s_lognH_logT_vneg'%(sim_name, dd_name, obj_tag, obs_point)}

        df = df_vel_neg
        dict_extra_args = dict_v_args
    else:
        print("Do not know what you need... check first please...")
        import sys
        sys.exit(0)

    print("Phew, finally making phase diagrames...")
    dshader_noaxes(df, dict_basic_args, dict_extra_args)

    print("Putting axes on the data shader plot...")
    wrap_axes(dict_basic_args, dict_extra_args, draw_pressure_line=True)
