#!/usr/bin/env python3

"""

    Title :      datashader_quickplot
    Notes :      To make a quick datashader plot of 3 given quantities from FOGGIE outputs
    Output :     datashader plot as png file
    Author :     Ayan Acharyya
    Started :    Oct 2021
    Examples :   run datashader_quickplot.py --system ayan_local --halo 8508 --galrad 20 --xcol rad --ycol metal --colorcol vrad
                 run datashader_quickplot.py --system ayan_local --halo 8508 --fullbox --xcol vrad --xmin -320 --xmax 500 --ycol metal --ymin -4 --ymax 0.5 --colorcol phi_disk --cmap viridis --ncolbins 7

"""
from header import *
from util import *
yt_ver = yt.__version__
start_time = time.time()

# -------------------------------------------------------------------------------
def get_df_from_ds(ds, args):
    '''
    Function to make a pandas dataframe from the yt dataset based on the given field list and color category,
    then writes dataframe to file for faster access in future
    This function is somewhat based on foggie.utils.prep_dataframe.prep_dataframe()
    :return: dataframe
    '''
    outfilename = args.output_dir + 'txtfiles/' + args.output + '_df_boxrad_%.2Fkpc_%s_vs_%s_colby_%s%s.txt' % (args.galrad, args.ycolname, args.xcolname, args.colorcolname, inflow_outflow_text)
    Path(args.output_dir + 'txtfiles/').mkdir(parents=True, exist_ok=True)  # creating the directory structure, if doesn't exist already

    if not os.path.exists(outfilename) or args.clobber:
        myprint('Creating file ' + outfilename + '..', args)
        df = pd.DataFrame()
        all_fields = [args.xcol, args.ycol, args.colorcol]
        for index,field in enumerate(all_fields):
            myprint('Doing property: ' + field + ', which is ' + str(index + 1) + ' of the ' + str(len(all_fields)) + ' fields..', args)
            if 'phi' in field and 'disk' in field: arr = np.abs(np.degrees(ds[field_dict[field]].v) - 90) # to convert from radian to degrees; and then restrict phi from 0 to 90
            elif 'theta' in field and 'disk' in field: arr = np.degrees(ds[field_dict[field]].v) # to convert from radian to degrees
            else: arr = ds[field_dict[field]].in_units(unit_dict[field]).ndarray_view()
            column_name = field

            if isfield_weighted_dict[field] and args.weight:
                weights = ds[field_dict[args.weight]].in_units(unit_dict[args.weight]).ndarray_view()
                arr = arr * weights * len(weights) / np.sum(weights)
                df[args.weight] = weights
                column_name = column_name + '_wtby_' + args.weight
            if islog_dict[field]:
                arr = np.log10(arr)
                column_name = 'log_' + column_name

            df[column_name] = arr

        df.to_csv(outfilename, sep='\t', index=None)
    else:
        myprint('Reading from existing file ' + outfilename, args)
        df = pd.read_table(outfilename, delim_whitespace=True, comment='#')

    return df

# ---------------------------------------------------------------------------------
def plot_1D_histogram(data, data_min, data_max, ax, vertical=False):
    '''
    Function to plot marginalised histograms using seaborn
    '''
    sns.kdeplot(data, ax=ax, legend=False, lw=1, vertical=vertical, color='k')
    ax.tick_params(axis='x', which='both', top=False)
    if vertical: ax.set_ylim(data_min, data_max)
    else: ax.set_xlim(data_min, data_max)

    return ax

# ---------------------------------------------------------------------------------
def make_coordinate_axis(colname, data_min, data_max, ax, fontsize):
    '''
    Function to make the coordinate axis
    Uses globally defined islog_dict and unit_dict
    '''
    log_text = 'Log ' if islog_dict[colname]else ''
    ax.set_label_text(log_text + labels_dict[colname] + ' (' + unit_dict[colname] + ')', fontsize=fontsize)

    nticks = 5
    ticks = np.linspace(data_min, data_max, nticks)
    ax.set_ticks(ticks)

    nticklabels = 5
    ticklabels = np.array(['     '] * nticks)
    ticklabel_every = int(nticks/nticklabels)
    ticklabels[:: ticklabel_every] = ['%.1F' % item for item in ticks[:: ticklabel_every]]

    ax.set_ticklabels(ticklabels, fontsize=fontsize)

    return ax

# ---------------------------------------------------------------------------------
def make_colorbar_axis_mpl(colname, artist, fig, fontsize):
    '''
    Function to make the coordinate axis
    Uses globally defined islog_dict and unit_dict
    Different from make_colorbar_axis() because this function uses native matplotlib support of datashader
    '''
    cax_xpos, cax_ypos, cax_width, cax_height = 0.7, 0.835, 0.25, 0.035
    cax = fig.add_axes([cax_xpos, cax_ypos, cax_width, cax_height])
    plt.colorbar(artist, cax=cax, orientation='horizontal')

    cax.set_xticklabels(['%.0F' % index for index in cax.get_xticks()], fontsize=fontsize / 1.5)  # , weight='bold')

    log_text = 'Log ' if islog_dict[colname] else ''
    fig.text(cax_xpos + cax_width / 2, cax_ypos + cax_height + 0.005, log_text + labels_dict[colname] + ' (' + unit_dict[colname] + ')', fontsize=fontsize/1.5, ha='center', va='bottom')

    return fig, cax

# --------------------------------------------------------------------------------
def make_datashader_plot_mpl(df, outfilename, args):
    '''
    Function to make data shader plot of y_field vs x_field, colored in bins of color_field
    This function is based on foggie.render.shade_maps.render_image()
    This is different from make_datashader_plot() in that this function uses the newest version of datashader which has matplotlib support
    So this function essentially combines make_datashader_plot() and wrap_axes(), because the latter is not needed anymore
    :return dataframe, figure
    '''
    # ----------to determine colorbar parameters--------------
    color_list = plt.get_cmap(args.cmap) if args.cmap is not None else colormap_dict[args.colorcol].colors
    ncol_bins = args.ncolbins if args.ncolbins is not None else len(color_list)
    color_list = color_list[::int(len(color_list)/ncol_bins)] # truncating color_list in to a length of rougly ncol_bins
    ncol_bins = len(color_list)

    # ----------to filter and categorize the dataframe--------------
    df = df[(df[args.xcolname].between(args.xmin, args.xmax)) & (df[args.ycolname].between(args.ymin, args.ymax)) & (df[args.colorcolname].between(args.cmin, args.cmax))]
    df[args.colorcol_cat] = categorize_by_quant(df[args.colorcolname], args.cmin, args.cmax, ncol_bins)
    df[args.colorcol_cat] = df[args.colorcol_cat].astype('category')

    # -----------------to initialise figure---------------------
    shift_right = 'phi' in args.ycol or 'theta' in args.ycol
    axes = sns.JointGrid(args.xcolname, args.ycolname, df, height=8)
    extra_space = 0.03 if shift_right else 0
    plt.subplots_adjust(hspace=0.05, wspace=0.05, right=0.95 + extra_space, top=0.95, bottom=0.1, left=0.1 + extra_space)
    fig, ax1 = plt.gcf(), axes.ax_joint

    # --------to make the main datashader plot--------------------------
    color_key = get_color_keys(args.cmin, args.cmax, color_list)
    #artist = dsshow(df, dsh.Point(args.xcolname, args.ycolname), dsh.count_cat(args.colorcol_cat), norm='eq_hist', color_key=color_key, x_range=(args.xmin, args.xmax), y_range=(args.ymin, args.ymax), vmin=args.cmin, vmax=args.cmax, aspect = 'auto', ax=ax1, alpha_range=(40, 255), shade_hook=partial(dstf.spread, shape='square')) # the 40 in alpha_range and `square` in shade_hook are to reproduce original-looking plots as if made with make_datashader_plot()
    artist = dsshow(df, dsh.Point(args.xcolname, args.ycolname), dsh.mean(args.colorcolname), norm='linear', cmap=list(color_key.values()), x_range=(args.xmin, args.xmax), y_range=(args.ymin, args.ymax), vmin=args.cmin, vmax=args.cmax, aspect = 'auto', ax=ax1) #, shade_hook=partial(dstf.spread, px=1, shape='square')) # the 40 in alpha_range and `square` in shade_hook are to reproduce original-looking plots as if made with make_datashader_plot()

    # ----------to plot 1D histogram on the top and right axes--------------
    axes.ax_marg_x = plot_1D_histogram(df[args.xcolname], args.xmin, args.xmax, axes.ax_marg_x, vertical=False)
    axes.ax_marg_y = plot_1D_histogram(df[args.ycolname], args.ymin, args.ymax, axes.ax_marg_y, vertical=True)

    # ------to make the axes-------------
    ax1.xaxis = make_coordinate_axis(args.xcol, args.xmin, args.xmax, ax1.xaxis, args.fontsize)
    ax1.yaxis = make_coordinate_axis(args.ycol, args.ymin, args.ymax, ax1.yaxis, args.fontsize)

    # ------to make the colorbar axis-------------
    fig, ax2 = make_colorbar_axis_mpl(args.colorcol, artist, fig, args.fontsize)

    # ---------to annotate and save the figure----------------------
    if args.current_redshift is not None: plt.text(0.033, 0.05, 'z = %.4F' % args.current_redshift, transform=ax1.transAxes, fontsize=args.fontsize)
    if args.current_time is not None: plt.text(0.033, 0.1, 't = %.3F Gyr' % args.current_time, transform=ax1.transAxes, fontsize=args.fontsize)
    filename = os.path.splitext(outfilename)[0] + '.png'
    plt.savefig(filename, transparent=False)
    myprint('Saved figure ' + filename, args)
    plt.show(block=False)

    return df, fig

# -------------------------------------------------------------------------------------------------------------
def get_color_labels(data_min, data_max, nbins):
    '''
    Function to determine the bin labels of every color category, to make the datashader plot
    '''
    edges = np.linspace(data_min, data_max, nbins + 1)
    color_labels = np.chararray(nbins, 13)
    for i in range(nbins):
        color_labels[i] = b'[%.2F,%.2F)' % (edges[i], edges[i + 1])

    return edges, color_labels

# -------------------------------------------------------------------------------------------------------------
def get_color_keys(data_min, data_max, color_list):
    '''
    Function to determine the actual color for every color category, based on a given discrete colormap
    In this script, this function is used such that the discrete colormap is the corresponding discrete colormap in consistency.py
    '''
    nbins = len(color_list)
    _, color_labels = get_color_labels(data_min, data_max, nbins)
    color_key = collections.OrderedDict()
    for i in range(nbins):
        color_key[color_labels[i]] = to_hex(color_list[i])

    return color_key

# -------------------------------------------------------------------------------------------------------------
def categorize_by_quant(data, data_min, data_max, nbins):
    '''
    Function to assign categories to the given data, to make the datashader plot color coding
    '''
    edges, color_labels = get_color_labels(data_min, data_max, nbins)
    category = np.chararray(np.size(data), 13)
    for i in range(nbins):
        category[(data >= edges[i]) & (data <= edges[i + 1])] = color_labels[i]

    return category

# -------------------------------------------------------------------------------------------------------------
def update_dicts(param, ds):
    '''
    Function to update the dicts with default values for a new parameter
    '''
    print(param + ' does not exist in dicts, therefore, updating dicts..')
    field_dict.update({param: ('gas', param)})
    unit_dict.update({param: ds.field_info[('gas', param)].units})
    labels_dict.update({param: param})

#-------- set variables and dictionaries such that they are available to other scripts importing this script-----------
field_dict = {'rad':('gas', 'radius_corrected'), 'density':('gas', 'density'), 'mass':('gas', 'mass'), \
              'metal':('gas', 'metallicity'), 'temp':('gas', 'temperature'), 'vrad':('gas', 'radial_velocity_corrected'), \
              'phi_L':('gas', 'angular_momentum_phi'), 'theta_L':('gas', 'angular_momentum_theta'), 'volume':('gas', 'volume'), \
              'phi_disk': ('gas', 'phi_pos_disk'), 'theta_disk': ('gas', 'theta_pos_disk')}
if yt_ver[0]=='3':
    field_dict['mass'] = ('gas','cell_mass')
    field_dict['volume'] = ('gas', 'cell_volume')
unit_dict = {'rad':'kpc', 'density':'g/cm**3', 'metal':r'Zsun', 'temp':'K', 'vrad':'km/s', 'phi_L':'deg', 'theta_L':'deg', 'PDF':'', 'mass':'Msun', 'volume':'pc**3', 'phi_disk':'deg', 'theta_disk':'deg'}
labels_dict = {'rad':'Radius', 'density':'Density', 'metal':'Metallicity', 'temp':'Temperature', 'vrad':'Radial velocity', 'phi_L':r'$\phi_L$', 'theta_L':r'$\theta_L$', 'PDF':'PDF', 'phi_disk':'Azimuthal Angle', 'theta_disk':r'$\theta_{\mathrm{diskrel}}$'}
islog_dict = defaultdict(lambda: False, metal=True, density=True, temp=True)
bin_size_dict = defaultdict(lambda: 1.0, metal=0.1, density=2, temp=1, rad=0.1, vrad=50)
colormap_dict = {'temp':temperature_discrete_cmap, 'metal':metal_discrete_cmap, 'density': density_discrete_cmap, 'vrad': outflow_inflow_discrete_cmap, 'rad': radius_discrete_cmap, 'phi_L': angle_discrete_cmap_pi, 'theta_L': angle_discrete_cmap_2pi, 'phi_disk':'viridis', 'theta_disk':angle_discrete_cmap_2pi}
isfield_weighted_dict = defaultdict(lambda: False, metal=True, temp=True, vrad=True, phi_L=True, theta_L=True, phi_disk=True, theta_disk=True)
bounds_dict = defaultdict(lambda: None, density=(1e-31, 1e-21), temp=(1e1, 1e8), metal=(1e-3, 1e1), vrad=(-400, 400), phi_L=(0, 180), theta_L=(-180, 180), phi_disk=(0, 90), theta_disk=(-180, 180))  # in g/cc, range within box; hard-coded for Blizzard RD0038; but should be broadly applicable to other snaps too

# -----main code-----------------
if __name__ == '__main__':
    args_tuple = parse_args('8508', 'RD0042')  # default simulation to work upon when comand line args not provided
    # ------------------------------------------------------------------------------
    if type(args_tuple) is tuple:
        args, ds, refine_box = args_tuple  # if the sim has already been loaded in, in order to compute the box center (via utils.pull_halo_center()), then no need to do it again
        print_mpi('ds ' + str(ds) + ' for halo ' + str(args.output) + ' was already loaded at some point by utils; using that loaded ds henceforth', args)
    else:
        args = args_tuple
        ds, refine_box = load_sim(args, region='refine_box', do_filter_particles=True, disk_relative=True)

    # -------create new fields for angular momentum vectors-----------
    ds.add_field(('gas', 'angular_momentum_phi'), function=phi_angular_momentum, sampling_type='cell', units='degree')
    ds.add_field(('gas', 'angular_momentum_theta'), function=theta_angular_momentum, sampling_type='cell', units='degree')
    
    # ----------to determine axes labels--------------
    if args.xcol == 'radius': args.xcol == 'rad'
    args.colorcol = args.colorcol[0]
    if args.xcol not in field_dict: update_dicts(args.xcol, ds)
    if args.ycol not in field_dict: update_dicts(args.ycol, ds)
    if args.colorcol not in field_dict: update_dicts(args.colorcol, ds)


    args.xcolname = 'log_' + args.xcol if islog_dict[args.xcol] else args.xcol
    args.ycolname = 'log_' + args.ycol if islog_dict[args.ycol] else args.ycol
    args.colorcolname = 'log_' + args.colorcol if islog_dict[args.colorcol] else args.colorcol
    if isfield_weighted_dict[args.xcol] and args.weight: args.xcolname += '_wtby_' + args.weight
    if isfield_weighted_dict[args.ycol] and args.weight: args.ycolname += '_wtby_' + args.weight
    if isfield_weighted_dict[args.colorcol] and args.weight: args.colorcolname += '_wtby_' + args.weight
    args.colorcol_cat = 'cat_' + args.colorcolname

    # ----------to determine box size--------------
    if args.fullbox:
        box_width = ds.refine_width  # kpc
        args.galrad = box_width / 2
        box = refine_box
    else:
        box_center = ds.arr(args.halo_center, kpc)
        box_width = args.galrad * 2  # in kpc
        box_width_kpc = ds.arr(box_width, 'kpc')
        box = ds.r[box_center[0] - box_width_kpc / 2.: box_center[0] + box_width_kpc / 2., box_center[1] - box_width_kpc / 2.: box_center[1] + box_width_kpc / 2., box_center[2] - box_width_kpc / 2.: box_center[2] + box_width_kpc / 2., ]

    # ----------to determine axes limits--------------
    bounds_dict.update(rad=(0, args.galrad))
    if args.xmin is None:
        args.xmin = np.log10(bounds_dict[args.xcol][0]) if islog_dict[args.xcol] else bounds_dict[args.xcol][0]
    if args.xmax is None:
        args.xmax = np.log10(bounds_dict[args.xcol][1]) if islog_dict[args.xcol] else bounds_dict[args.xcol][1]
    if args.ymin is None:
        args.ymin = np.log10(bounds_dict[args.ycol][0]) if islog_dict[args.ycol] else bounds_dict[args.ycol][0]
    if args.ymax is None:
        args.ymax = np.log10(bounds_dict[args.ycol][1]) if islog_dict[args.ycol] else bounds_dict[args.ycol][1]
    if args.cmin is None:
        args.cmin = np.log10(bounds_dict[args.colorcol][0]) if islog_dict[args.colorcol] else bounds_dict[args.colorcol][0]
    if args.cmax is None:
        args.cmax = np.log10(bounds_dict[args.colorcol][1]) if islog_dict[args.colorcol] else bounds_dict[args.colorcol][1]

    if args.inflow_only:
        box = box.cut_region(["obj[('gas', 'radial_velocity_corrected')] < 0"])
        inflow_outflow_text = '_inflow_only'
    elif args.outflow_only:
        box = box.cut_region(["obj[('gas', 'radial_velocity_corrected')] > 0"])
        inflow_outflow_text = '_outflow_only'
    else:
        inflow_outflow_text = ''

    # ----------to labels and paths--------------
    args.current_redshift = ds.current_redshift
    args.current_time = ds.current_time.in_units('Gyr')
    outfile_rootname = 'datashader_boxrad_%.2Fkpc_%s_vs_%s_colby_%s%s.png' % (args.galrad, args.ycolname, args.xcolname, args.colorcolname, inflow_outflow_text)

    fig_dir = args.output_dir + 'figs/' + args.output + '/'
    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    thisfilename = fig_dir + outfile_rootname

    # ----------to actually make the plot--------------
    if not args.keep: plt.close('all')

    df = get_df_from_ds(box, args)
    df, fig = make_datashader_plot_mpl(df, thisfilename, args)

    print_master('Completed in %s mins' % ((time.time() - start_time) / 60), args)

