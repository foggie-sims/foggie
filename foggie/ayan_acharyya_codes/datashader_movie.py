#!/usr/bin/env python3

"""

    Title :      datashader_movie
    Notes :      Attempt to make datashader plots of 3 given quantities from FOGGIE outputs and make movies
    Output :     datashader plots as png files (which can be later converted to a movie via animate_png.py)
    Author :     Ayan Acharyya
    Started :    July 2021
    Examples :   run datashader_movie.py --system ayan_hd --halo 8508 --galrad 20 --xcol rad --ycol metal --colorcol vrad --weight density --overplot_stars --do_all_sims --makemovie --delay 0.2
                 run datashader_movie.py --system ayan_hd --halo 8508 --galrad 20 --xcol rad --ycol temp --colorcol density,metal,vrad,phi_L,theta_L --output RD0042 --clobber_plot --overplot_stars --keep
                 run datashader_movie.py --system ayan_hd --halo 8508 --galrad 20 --xcol rad --ycol metal --colorcol vrad,density,temp,phi_L,theta_L --output RD0042 --clobber_plot --overplot_stars --keep
                 run datashader_movie.py --system ayan_hd --halo 8508 --do gas --galrad 20 --xcol rad --ycol temp --colorcol vrad --output RD0030 --clobber_plot --overplot_stars --interactive --combine
"""
from header import *
from util import *
from projection_plot import do_plot
from make_ideal_datacube import shift_ref_frame
from filter_star_properties import get_star_properties
start_time = time.time()

# ------------------------------------------------------------------------------
def weight_by(data, weights):
    '''
    Function to compute weighted array
    '''
    weighted_data = data * weights * len(weights) / np.sum(weights)
    return weighted_data

# -------------------------------------------------------------------------------
def get_df_from_ds(ds, args):
    '''
    Function to make a pandas dataframe from the yt dataset based on the given field list and color category,
    then writes dataframe to file for faster access in future
    This function is based upon foggie.utils.prep_dataframe.prep_dataframe()
    :return: dataframe
    '''
    outfilename = args.output_dir + 'txtfiles/' + args.output + '_df_boxrad_%.2Fkpc_%s_vs_%s_colcat_%s.txt' % (args.galrad, ycol, xcol, colorcol)
    if not os.path.exists(outfilename) or args.clobber:
        if not os.path.exists(outfilename):
            myprint(outfilename + ' does not exist. Creating afresh..', args)
        elif args.clobber:
            myprint(outfilename + ' exists but over-writing..', args)

        df = pd.DataFrame()
        for field in [args.xcol, args.ycol, args.colorcol]:
            column_name = field
            arr = ds[field_dict[field]].in_units(unit_dict[field]).ndarray_view()
            if weight_field_dict[field] and args.weight:
                weights = ds[field_dict[weight_field_dict[field]]].ndarray_view()
                arr = weight_by(arr, weights)
                column_name = column_name + '_wtby_' + weight_field_dict[field]
            if islog_dict[field]:
                arr = np.log10(arr)
                column_name = 'log_' + column_name
            df[column_name] = arr

        df.to_csv(outfilename, sep='\t', index=None)
    else:
        myprint('Reading from existing file ' + outfilename, args)
        df = pd.read_table(outfilename, delim_whitespace=True, comment='#')

    return df

# ----------------------------------------------------------------------------------
def get_radial_velocity(paramlist):
    '''
    Function to corrects the radial velocity for the halo center
    Takes in velocities that are already corrected for the bulk motion of the halo
    This function is based on Cassi's foggie.utils.yt_fields.radial_velocity_corrected()
    '''
    paramlist['rad'] = np.sqrt(paramlist['pos_x_cen'] ** 2 + paramlist['pos_y_cen'] ** 2 + paramlist['pos_z_cen'] ** 2)
    x_hat = paramlist['pos_x_cen'] / paramlist['rad']
    y_hat = paramlist['pos_y_cen'] / paramlist['rad']
    z_hat = paramlist['pos_z_cen'] / paramlist['rad']
    paramlist['vrad'] = paramlist['vel_x_cen'] * x_hat + paramlist['vel_y_cen'] * y_hat + paramlist['vel_z_cen'] * z_hat
    return paramlist

# ---------------------------------------------------------------------------------
def convert_to_datashader_frame(data, data_min, data_max, npix_datashader):
    '''
    Function to convert physical quantities to corresponding pixel values in the datashader image frame
    '''
    return (data - data_min) * npix_datashader / (data_max - data_min)

# ---------------------------------------------------------------------------------
def convert_from_datashader_frame(data, data_min, data_max, npix_datashader):
    '''
    Function to convert pixel values on a datashader image to corresponding physical quantities
    '''
    return data * (data_max - data_min) / npix_datashader + data_min

# ---------------------------------------------------------------------------------
def overplot_stars(x_min, x_max, y_min, y_max, c_min, c_max, npix_datashader, ax, args):
    '''
    Function to overplot young stars on existing datashader plot
    Uses globally defined colormap_dict
    '''
    starlistfile = args.output_dir + 'txtfiles/' + args.output + '_young_star_properties.txt'

    # -------------to read in simulation data------------
    if not os.path.exists(starlistfile):
        print_mpi(starlistfile + 'does not exist. Calling get_star_properties() first..', args)
        dummy = get_star_properties(args)  # this creates the infilename
    paramlist = pd.read_table(starlistfile, delim_whitespace=True, comment='#')

    # -------------to prep the simulation data------------
    paramlist = shift_ref_frame(paramlist, args)
    paramlist = paramlist.rename(columns={'gas_metal': 'metal', 'gas_density': 'density', 'gas_pressure': 'pressure', 'gas_temp': 'temp'})
    paramlist = get_radial_velocity(paramlist)
    for field in [args.xcol, args.ycol, args.colorcol]:
        if field not in paramlist.columns:
            print_mpi(field + ' does not exist in young star list, therefore cannot overplot stars..', args)
            return ax
        column_name = field
        if weight_field_dict[field] and args.weight:
            weightcol = weight_field_dict[field]
            if weightcol in paramlist.columns: weights = paramlist[weightcol]
            else: weights = np.ones(len(paramlist))
            paramlist[column_name + '_wtby_' + weightcol] = weight_by(paramlist[column_name], weights)
            column_name = column_name + '_wtby_' + weightcol
        if islog_dict[field]: paramlist['log_' + column_name] = np.log10(paramlist[column_name])
    paramlist = paramlist[(paramlist[xcol].between(x_min, x_max)) & (paramlist[ycol].between(y_min, y_max)) & (paramlist[colorcol].between(c_min, c_max))]

    # -------------to actually plot the simulation data------------
    x_on_plot = convert_to_datashader_frame(paramlist[xcol], x_min, x_max, npix_datashader) # because we need to stretch the binned x and y into npix_datashader dimensions determined by the datashader plot
    y_on_plot = convert_to_datashader_frame(paramlist[ycol], y_min, y_max, npix_datashader)
    ax.scatter(x_on_plot, y_on_plot, c=paramlist[colorcol], edgecolors='black', lw=0.2, s=15, cmap=colormap_dict[args.colorcol])
    print_mpi('Overplotted ' + str(len(paramlist)) + ' young stars..', args)

    return ax

# ---------------------------------------------------------------------------------
def overplot_binned(df, x_min, x_max, y_min, y_max, npix_datashader, ax, args):
    '''
    Function to overplot binned data on existing datashader plot
    Uses globally defined islog_dict
    '''
    x_bin_size = bin_size_dict[args.xcol]
    x_bins = np.arange(x_min, x_max + x_bin_size, x_bin_size)
    df['binned_cat'] = pd.cut(df[xcol], x_bins)
    if islog_dict[args.ycol]: df[args.ycol] = 10 ** df[ycol] # because otherwise args.ycol is not going to be present in df
    y_binned = df.groupby('binned_cat', as_index=False).agg(np.mean)[args.ycol] # so that the averaging is done in linear space (as opposed to log space, which would be incorrect)
    if islog_dict[args.ycol]: y_binned = np.log10(y_binned)

    # ----------to plot mean binned y vs x profile--------------
    x_bin_centers = x_bins[:-1] + x_bin_size / 2
    x_on_plot = convert_to_datashader_frame(x_bin_centers, x_min, x_max, npix_datashader) # because we need to stretch the binned x and y into npix_datashader dimensions determined by the datashader plot
    y_on_plot = convert_to_datashader_frame(y_binned, y_min, y_max, npix_datashader)
    ax.plot(x_on_plot, y_on_plot, color='black', lw=1)

    return ax

# ---------------------------------------------------------------------------------
def plot_1D_histogram(data, data_min, data_max, ax, vertical=False):
    '''
    Function to plot marginalised histograms using seaborn
    '''
    sns.kdeplot(data, ax=ax, legend=False, color='black', lw=1, vertical=vertical)
    ax.tick_params(axis='x', which='both', top=False)
    if vertical: ax.set_ylim(data_min, data_max)
    else: ax.set_xlim(data_min, data_max)

    return ax

# ---------------------------------------------------------------------------------
def make_coordinate_axis(colname, data_min, data_max, ax, npix_datashader, fontsize):
    '''
    Function to make the coordinate axis
    Uses globally defined islog_dict and unit_dict
    '''
    log_text = 'Log ' if islog_dict[colname] else ''
    nticks = 5
    #delta = 100 if data_max - data_min > 100 else 30 if data_max - data_min > 30 else 5 if data_max - data_min > 5 else 1
    ax.set_label_text(log_text + labels_dict[colname] + ' (' + unit_dict[colname] + ')', fontsize=fontsize)
    ticks = np.linspace(data_min, data_max, nticks)
    ax.set_ticks(convert_to_datashader_frame(ticks, data_min, data_max, npix_datashader))
    ax.set_ticklabels(['%.1F' % index for index in ticks], fontsize=fontsize)

    return ax

# -----------------------------------------------------------------------------------
def create_foggie_cmap(colname, c_min, c_max):
    '''
    Function to create the colorbar for the tiny colorbar axis on top right
    Uses globally defined colorkey_dict, categorize_by_dict
    This function is based on Cassi's foggie.pressure_support.create_foggie_cmap(), which is in turn based on Jason's foggie.render.cmap_utils.create_foggie_cmap()
    '''
    n = 100000
    color_key = colorkey_dict[colname]

    df = pd.DataFrame({'x':np.random.rand(n), 'y': np.random.rand(n)})
    df[colname] = np.array([random.uniform(c_min, c_max) for i in range(n)])
    if islog_dict[colname]: df[colname] = 10. ** df[colname] # skipping taking log of metallicity because it is categorised in linear space
    df['cat'] = categorize_by_dict[colname](df[colname])

    n_labels = np.size(list(color_key))
    sightline_length = np.max(df['x']) - np.min(df['x'])
    value = np.max(df['x'])

    for index in np.flip(np.arange(n_labels), 0):
        df['cat'][df['x'] > value - sightline_length * (1.*index+1)/n_labels] = list(color_key)[index]
    df.cat = df.cat.astype('category') ##

    cvs = dsh.Canvas(plot_width=750, plot_height=100, x_range=(np.min(df['x']), np.max(df['x'])), y_range=(np.min(df['y']), np.max(df['y'])))
    agg = cvs.points(df, 'x', 'y', dsh.count_cat('cat'))
    cmap = dstf.spread(dstf.shade(agg, color_key=color_key), px=2, shape='square')

    return cmap

# ---------------------------------------------------------------------------------
def make_colorbar_axis(colname, data_min, data_max, fig, fontsize):
    '''
    Function to make the coordinate axis
    Uses globally defined islog_dict and unit_dict
    '''
    color_field_cmap = create_foggie_cmap(colname, data_min, data_max)
    ax_xpos, ax_ypos, ax_width, ax_height = 0.7, 0.82, 0.25, 0.06
    ax = fig.add_axes([ax_xpos, ax_ypos, ax_width, ax_height])
    cbar_im = np.flip(color_field_cmap.to_pil(), 1)
    ax.imshow(cbar_im)
    log_text = 'Log ' if islog_dict[colname] else ''
    delta_c = 200 if data_max - data_min > 200 else 60 if data_max - data_min > 60 else 10 if data_max - data_min > 10 else 2
    
    ax.set_xticks(np.arange((data_max - data_min) + 1., step=delta_c) * np.shape(cbar_im)[1] / (data_max - data_min))
    ax.set_xticklabels(['%.0F' % index for index in np.arange(data_min, data_max + 1, delta_c)], fontsize=fontsize/1.5)#, weight='bold')
    
    fig.text(ax_xpos + ax_width / 2, ax_ypos + ax_height, log_text + labels_dict[args.colorcol] + ' (' + unit_dict[args.colorcol] + ')', fontsize=fontsize/1.5, ha='center', va='bottom')

    for item in ['top', 'bottom', 'left', 'right']: ax.spines[item].set_color('white')
    ax.set_yticklabels([])
    ax.set_yticks([])

    return fig, ax

# -----------------------------------------------------------------------------------
def wrap_axes(df, filename, npix_datashader, args):
    '''
    Function to read in raw datashader plot and wrap it in axes using matplotlib AND added x- and y- marginalised histograms using seaborn
    This function is partly based on foggie.render.shade_maps.wrap_axes()
    :return: fig
    '''
    # ----------to determine axes limits--------------
    x_min, x_max = bounds_dict[args.xcol]
    if islog_dict[args.xcol]: x_min, x_max = np.log10(x_min), np.log10(x_max)
    y_min, y_max = bounds_dict[args.ycol]
    if islog_dict[args.ycol]: y_min, y_max = np.log10(y_min), np.log10(y_max)
    c_min, c_max = bounds_dict[args.colorcol]
    if islog_dict[args.colorcol]: c_min, c_max = np.log10(c_min), np.log10(c_max)
    shift_right = 'phi' in args.ycol or 'theta' in args.ycol

    # ----------to filter and bin the dataframe--------------
    df = df[(df[xcol].between(x_min, x_max)) & (df[ycol].between(y_min, y_max))]
    df[xcol + '_in_pix'] = convert_to_datashader_frame(df[xcol], x_min, x_max, npix_datashader)
    df[ycol + '_in_pix'] = convert_to_datashader_frame(df[ycol], y_min, y_max, npix_datashader)

    # -----------------to initialise figure---------------------
    axes = sns.JointGrid(xcol + '_in_pix', ycol + '_in_pix', df, height=8)
    extra_space = 0.03 if shift_right else 0
    plt.subplots_adjust(hspace=0.05, wspace=0.05, right=0.95 + extra_space, top=0.95, bottom=0.1, left=0.1 + extra_space)
    fig, ax1 = plt.gcf(), axes.ax_joint

    # ------to plot datashader image from file-------------
    img = mpimg.imread(filename)
    ax1.imshow(np.flip(img, 0))

    # ----------to overplot young stars----------------
    if args.overplot_stars: ax1 = overplot_stars(x_min, x_max, y_min, y_max, c_min, c_max, npix_datashader, ax1, args)

    # ----------to overplot binned profile----------------
    ax1 = overplot_binned(df, x_min, x_max, y_min, y_max, npix_datashader, ax1, args)

    # ----------to plot 1D histogram on the top and right axes--------------
    axes.ax_marg_x = plot_1D_histogram(df[xcol + '_in_pix'], 0, npix_datashader, axes.ax_marg_x, vertical=False)
    axes.ax_marg_y = plot_1D_histogram(df[ycol + '_in_pix'], 0, npix_datashader, axes.ax_marg_y, vertical=True)

    # ------to make the axes-------------
    ax1.xaxis = make_coordinate_axis(args.xcol, x_min, x_max, ax1.xaxis, npix_datashader, args.fontsize)
    ax1.yaxis = make_coordinate_axis(args.ycol, y_min, y_max, ax1.yaxis, npix_datashader, args.fontsize)
    fig, ax2 = make_colorbar_axis(args.colorcol, c_min, c_max, fig, args.fontsize)

    # ---------to annotate and save the figure----------------------
    plt.text(0.033, 0.05, 'z = %.4F' % args.current_redshift , transform=ax1.transAxes, fontsize=args.fontsize)
    plt.text(0.033, 0.1, 't = %.3F Gyr' % args.current_time , transform=ax1.transAxes, fontsize=args.fontsize)
    plt.savefig(filename, transparent=False)
    myprint('Saved figure ' + filename, args)
    if not args.makemovie: plt.show(block=False)

    return fig

# --------------------------------------------------------------------------------
def make_datashader_plot(ds, outfilename, args, npix_datashader=1000):
    '''
    Function to make data shader plot of y_field vs x_field, colored in bins of color_field
    This function is based on foggie.render.shade_maps.render_image()
    :return dataframe, figure
    '''
    df = get_df_from_ds(ds, args)
    df[colorcol_cat] = categorize_by_dict[args.colorcol](df[colorcol])
    df[colorcol_cat] = df[colorcol_cat].astype('category')

    x_range = np.log10(bounds_dict[args.xcol]) if islog_dict[args.xcol] else bounds_dict[args.xcol]
    y_range = np.log10(bounds_dict[args.ycol]) if islog_dict[args.ycol] else bounds_dict[args.ycol]

    cvs = dsh.Canvas(plot_width=npix_datashader, plot_height=npix_datashader, x_range=x_range, y_range=y_range)
    agg = cvs.points(df, xcol, ycol, dsh.count_cat(colorcol_cat))
    img = dstf.spread(dstf.shade(agg, color_key=colorkey_dict[args.colorcol], how='eq_hist', min_alpha=40), shape='square')
    export_image(img, os.path.splitext(outfilename)[0])

    fig = wrap_axes(df, os.path.splitext(outfilename)[0] + '.png', npix_datashader, args)

    return df, fig

# ------------------------------------------------------------------
class SelectFromCollection:
    '''
    Object to interactively choose coordinate points in a given plot using the LassoSelector
    This is copied from Raymond's foggie.angular_momentum.lasso_data_selection.ipynb

    Select indices from a matplotlib collection using `LassoSelector`.

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
    Axes to interact with.
    collection : `matplotlib.collections.Collection` subclass
    Collection you want to select from.
    alpha_other : 0 <= float <= 1
    To highlight a selection, this tool sets all selected points to an
    alpha value of 1 and non-selected points to *alpha_other*.
    '''

    def __init__(self, ax, collection, npix_datashader, nbins, alpha_other=0.5):
        self.ax = ax
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        self.lasso = LassoSelector(ax, lambda verts: self.onselect(verts, npix_datashader, nbins))
        self.ind = []
        self.count = 0
        self.should_remove_highlight = False

        mask = np.zeros((nbins, nbins))
        extent = 0, npix_datashader, 0, npix_datashader
        self.mask_plot = self.ax.imshow(mask, cmap=plt.cm.Greys_r, alpha=0, interpolation='bilinear', extent=extent, zorder=10, aspect='auto', vmin=0, vmax=1)  # , origin='lower')
        self.helpful_text = self.ax.text(0.5 * npix_datashader, 0.95 * npix_datashader, '', ha='center', fontsize=15)
        self.blitman = BlitManager(self.ax.figure.canvas, [self.mask_plot, self.helpful_text])

    def onselect(self, verts, npix_datashader, nbins):
        path = mpl_Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.count += 1

        # -------to highlight the selected region-------------------
        mask = np.zeros((nbins, nbins))
        mask.ravel()[self.ind] = 1.
        self.mask_plot.set_data(mask)
        self.mask_plot.set_alpha(self.alpha_other)
        self.helpful_text.set_text('Press ENTER to proceed with this selection or select again')
        self.blitman.update()

    def disconnect(self):
        self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.ax.figure.canvas.draw()

# -------------------------------------------------------------------------------
def combine_lasso_outputs(args, selection_figname, projection_figname):
    '''
    Function to combine the figures produced by lasso selection and display in one frame and write that frame to file
    '''
    myprint('Preparing to combine projection plots based on highlighted selection..', args)
    nrow, ncol = 2, 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(8, 8))
    fig.subplots_adjust(hspace=0, wspace=0, right=1, top=1, bottom=0, left=0)

    for index, thisproj in enumerate(['dsh', 'x', 'y', 'z']):
        if index: this_figname = projection_figname.replace('proj_' + args.projection, 'proj_' + thisproj)
        else: this_figname = selection_figname
        image = mpimg.imread(this_figname)
        #image = imageio.imread(this_figname)
        # how to crop extra margin off image?

        rowid, colid = int(index / ncol), index % ncol
        # ---the only harccoded bits, for cropping---
        if index: # for yt projection plots generated by this script datashader_movie.py
            crop_x1, crop_x2 = 150, 3750
            crop_y1, crop_y2 = 50, 3000
        else: # for datashader plots generated by this script datashader_movie.py
            crop_x1, crop_x2 = 10, 1550
            crop_y1, crop_y2 = 80, 1550
        # -------------------------------------
        axes[rowid][colid].axis('off')
        axes[rowid][colid].imshow(image[crop_y1 : crop_y2, crop_x1 : crop_x2], origin='upper')
        fig.text(0.15, 0.15, 'LoS = ' + thisproj if index else 'Selection' , color='white', transform=axes[rowid][colid].transAxes, fontsize=args.fontsize, ha='left', va='bottom')

    outfile_rootname = os.path.split(projection_figname)[1].replace('proj_' + args.projection, 'proj_xyz')
    outfile_rootname = os.path.splitext(outfile_rootname)[0]
    outfile_rootname = 'dsh_' + outfile_rootname.replace('projection', 'combined')
    saveplot(fig, args, outfile_rootname, outputdir=os.path.split(selection_figname)[0] + '/')
    plt.show(block=False)

    return fig

# -------------------------------------------------------------------------------
def projplot_selection(box, args, npix_datashader, nbins, selector, xgrid, ygrid, outfilename, field_to_plot='density'):
    '''
    Function to accept a key press event and plot selected region on yt ProjectionPlot
    This is based on Raymond's foggie.angular_momentum.lasso_data_selection.ipynb
    '''
    # ----------to determine axes limits--------------
    x_min, x_max = bounds_dict[args.xcol]
    if islog_dict[args.xcol]: x_min, x_max = np.log10(x_min), np.log10(x_max)
    y_min, y_max = bounds_dict[args.ycol]
    if islog_dict[args.ycol]: y_min, y_max = np.log10(y_min), np.log10(y_max)

    # ------to cut out the selected region----------
    if args.debug: myprint('Preparing to cut regions based on highlighted selection..', args)
    cut_crit = ""
    for index, item in enumerate(selector.ind):
        x_lower_bound = convert_from_datashader_frame(xgrid.ravel()[item] - 0.5 * npix_datashader / nbins, x_min, x_max, npix_datashader)
        x_upper_bound = convert_from_datashader_frame(xgrid.ravel()[item] + 0.5 * npix_datashader / nbins, x_min, x_max, npix_datashader)
        y_lower_bound = convert_from_datashader_frame(ygrid.ravel()[item] - 0.5 * npix_datashader / nbins, y_min, y_max, npix_datashader)
        y_upper_bound = convert_from_datashader_frame(ygrid.ravel()[item] + 0.5 * npix_datashader / nbins, y_min, y_max, npix_datashader)

        if islog_dict[args.xcol]: x_lower_bound, x_upper_bound = 10 ** x_lower_bound, 10 ** x_upper_bound
        if islog_dict[args.ycol]: y_lower_bound, y_upper_bound = 10 ** y_lower_bound, 10 ** y_upper_bound

        if index: cut_crit += '|'
        cut_crit += "((obj[{}] > {}) & (obj[{}] < {}) & (obj[{}] > {}) & (obj[{}] < {}))".format(field_dict[args.xcol], x_lower_bound, field_dict[args.xcol], x_upper_bound, field_dict[args.ycol], y_lower_bound, field_dict[args.ycol], y_upper_bound)

    box_cut = box.cut_region([cut_crit])

    # ------to make new plots corresponding to the selected region----------
    if args.debug: myprint('Preparing to projection plot based on above cut regions: \n' + cut_crit, args)
    proj_arr = ['x', 'y', 'z'] if args.combine else [args.projection]

    for thisproj in proj_arr:
        prj = do_plot(box_cut.ds, field_dict[field_to_plot], thisproj, [], box_cut, box.center.in_units(kpc), 2 * args.galrad * kpc, \
                      cmap_dict[field_to_plot], halo_dict[args.halo], unit=projected_unit_dict[field_to_plot], zmin=projected_bounds_dict[field_to_plot][0], \
                      zmax=projected_bounds_dict[field_to_plot][1], weight_field=weight_field_dict[field_to_plot], iscolorlog=islog_dict[field_to_plot], \
                      noweight=args.weight is None)

        this_figname = outfilename.replace('proj_' + args.projection, 'proj_' + thisproj)
        prj.save(this_figname, mpl_kwargs={'dpi': 500})

    return prj

# ------------------------------------------------------------------
def on_key_press(event, box, ax, args, npix_datashader, nbins, selector, xgrid, ygrid):
    '''
    Function to accept a key press event and display the selected pixels and modify the axis title
    This is based on Raymond's foggie.angular_momentum.lasso_data_selection.ipynb
    '''
    if event.key == 'enter':
        start_time_on_press = time.time()
        # -----save the selection-----------
        output_selfig = fig_dir + 'datashader_boxrad_%.2Fkpc_%s_vs_%s_colby_%s_lasso%d.png' % (args.galrad, ycol, xcol, colorcol, selector.count)
        ax.figure.savefig(output_selfig) # to save the highlighted image
        nselection = len(selector.ind)
        myprint('Selected ' + str(nselection) + ' binned pixels:', args)
        if args.debug: print(selector.xys[selector.ind])

        # -----refresh the image for next selection, if any-----------
        selector.helpful_text.set_text('Select points by dragging mouse in a loop')
        selector.blitman.update()

        # -----make projection plots with the selection-----------
        if nselection > 0:
            field_to_plot = 'density'
            output_projfig = fig_dir + '%s' % (field_to_plot) + '_boxrad_%.2Fkpc' % (args.galrad) + '_proj_' + args.projection + '_lasso' + str(selector.count) + '_by_' + ycol + '_vs_' + xcol + '_projection.png'
            prj = projplot_selection(box, args, npix_datashader, nbins, selector, xgrid, ygrid, output_projfig, field_to_plot=field_to_plot)
            if args.combine: fig = combine_lasso_outputs(args, output_selfig, output_projfig)
        else:
            myprint('No data point selected; therefore, not proceeding any further.', args)

        myprint('Key press events completed in %s' % (datetime.timedelta(seconds = time.time() - start_time_on_press)), args)

    elif event.key == 'escape':
        myprint('Exiting interactive mode..', args)
        selector.helpful_text.set_text('')
        selector.blitman.update()
        selector.disconnect()

    else:
        myprint('You pressed ' + event.key + '; but you need to press ENTER to see your selection or ESC to exit interactive mode', args)

# -------------------------------------------------------------------------------------------------------------
def setup_interactive(box, ax, args, npix_datashader, nbins):
    '''
    Function to interactively select points from a live plot using LassoSelector
    This is based on Raymond's foggie.angular_momentum.lasso_data_selection.ipynb
    '''
    x = np.linspace(0, npix_datashader, nbins)
    y = np.linspace(0, npix_datashader, nbins)
    xgrid, ygrid = np.meshgrid(x, y)
    pts = ax.scatter(xgrid, ygrid, s=10, alpha=0.0)
    selector = SelectFromCollection(ax, pts, npix_datashader, nbins)

    fig = plt.gcf()
    fig.canvas.mpl_connect("key_press_event", lambda event: on_key_press(event, box, ax, args, npix_datashader, nbins, selector, xgrid, ygrid))

    selector.helpful_text.set_text('Select points by dragging mouse in a loop')
    selector.blitman.update()

    return fig

# -----main code-----------------
if __name__ == '__main__':
    npix_datashader, nselection_bins = 1000, 20
    # set variables and dictionaries
    field_dict = {'rad':('gas', 'radius_corrected'), 'density':('gas', 'density'), 'stars':('deposit', 'stars_density'), 'mass':('gas', 'mass'), \
                  'metal':('gas', 'metallicity'), 'temp':('gas', 'temperature'), 'vrad':('gas', 'radial_velocity_corrected'), \
                  'phi_L':('gas', 'angular_momentum_phi'), 'theta_L':('gas', 'angular_momentum_theta')}
    unit_dict = {'rad':'kpc', 'density':'g/cm**3', 'metal':r'Zsun', 'temp':'K', 'vrad':'km/s', 'phi_L':'deg', 'theta_L':'deg'}
    labels_dict = {'rad':'Radius', 'density':'Density', 'metal':'Metallicity', 'temp':'Temperature', 'vrad':'Radial velocity', 'phi_L':r'$\phi_L$', 'theta_L':r'$\theta_L$'}
    islog_dict = defaultdict(lambda: False, metal=True, density=True, temp=True)
    bin_size_dict = defaultdict(lambda: 1.0, metal=0.1, density=2, temp=1, rad=0.1, vrad=50)
    categorize_by_dict = {'temp':categorize_by_temp, 'metal':categorize_by_log_metals, 'density':categorize_by_den, 'vrad':categorize_by_outflow_inflow, 'rad':categorize_by_radius, 'phi_L':categorize_by_angle_pi, 'theta_L':categorize_by_angle_2pi}
    colorkey_dict = {'temp':new_phase_color_key, 'metal':new_metals_color_key, 'density': density_color_key, 'vrad': outflow_inflow_color_key, 'rad': radius_color_key, 'phi_L': angle_color_key_pi, 'theta_L': angle_color_key_2pi}
    colormap_dict = {'temp':temperature_discrete_cmap, 'metal':metal_discrete_cmap, 'density': density_discrete_cmap, 'vrad': outflow_inflow_discrete_cmap, 'rad': radius_discrete_cmap, 'phi_L': angle_discrete_cmap_pi, 'theta_L': angle_discrete_cmap_2pi}

    projected_unit_dict = defaultdict(lambda x: unit_dict[x], density='Msun/pc**2')
    cmap_dict = {'density':density_color_map, 'metal':metal_color_map, 'temp':temperature_color_map, 'vrad':velocity_discrete_cmap} # for projection plots, if any

    dummy_args_tuple = parse_args('8508', 'RD0042')  # default simulation to work upon when comand line args not provided
    if type(dummy_args_tuple) is tuple: dummy_args = dummy_args_tuple[0] # if the sim has already been loaded in, in order to compute the box center (via utils.pull_halo_center()), then no need to do it again
    else: dummy_args = dummy_args_tuple
    if dummy_args.xcol == 'radius': dummy_args.xcol == 'rad'
    if not dummy_args.keep: plt.close('all')

    if dummy_args.do_all_sims: list_of_sims = get_all_sims_for_this_halo(dummy_args) # all snapshots of this particular halo
    else: list_of_sims = [(dummy_args.halo, dummy_args.output)]
    total_snaps = len(list_of_sims)

    weight_field_dict = defaultdict(lambda: None, metal=dummy_args.weight, temp=dummy_args.weight, vrad=dummy_args.weight, phi_L=dummy_args.weight, theta_L=dummy_args.weight)

    # parse column names, in case log
    xcol = 'log_' + dummy_args.xcol if islog_dict[dummy_args.xcol] else dummy_args.xcol
    ycol = 'log_' + dummy_args.ycol if islog_dict[dummy_args.ycol] else dummy_args.ycol
    if weight_field_dict[dummy_args.xcol] and not dummy_args.noweight: xcol += '_wtby_' + weight_field_dict[dummy_args.xcol]
    if weight_field_dict[dummy_args.ycol] and not dummy_args.noweight: ycol += '_wtby_' + weight_field_dict[dummy_args.ycol]

    # parse paths and filenames
    fig_dir = dummy_args.output_dir + 'figs/' if dummy_args.do_all_sims else dummy_args.output_dir + 'figs/' + dummy_args.output + '/'
    Path(fig_dir).mkdir(parents=True, exist_ok=True)

    # --------domain decomposition; for mpi parallelisation-------------
    comm = MPI.COMM_WORLD
    ncores = comm.size
    rank = comm.rank
    print_master('Total number of MPI ranks = ' + str(ncores) + '. Starting at: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()), dummy_args)
    comm.Barrier() # wait till all cores reached here and then resume

    split_at_cpu = total_snaps - ncores * int(total_snaps/ncores)
    nper_cpu1 = int(total_snaps / ncores)
    nper_cpu2 = nper_cpu1 + 1
    if rank < split_at_cpu:
        core_start = rank * nper_cpu2
        core_end = (rank+1) * nper_cpu2 - 1
    else:
        core_start = split_at_cpu * nper_cpu2 + (rank - split_at_cpu) * nper_cpu1
        core_end = split_at_cpu * nper_cpu2 + (rank - split_at_cpu + 1) * nper_cpu1 - 1

    # --------------------------------------------------------------
    print_mpi('Operating on snapshots ' + str(core_start + 1) + ' to ' + str(core_end + 1) + ', i.e., ' + str(core_end - core_start + 1) + ' out of ' + str(total_snaps) + ' snapshots', dummy_args)

    for index in range(core_start + dummy_args.start_index, core_end + 1):
        start_time_this_snapshot = time.time()
        this_sim = list_of_sims[index]
        print_mpi('Doing snapshot ' + this_sim[1] + ' of halo ' + this_sim[0] + ' which is ' + str(index + 1 - core_start) + ' out of the total ' + str(core_end - core_start + 1) + ' snapshots...', dummy_args)
        try:
            if dummy_args.do_all_sims: args = parse_args(this_sim[0], this_sim[1])
            else: args = dummy_args_tuple # since parse_args() has already been called and evaluated once, no need to repeat it

            if type(args) is tuple:
                args, ds, refine_box = args  # if the sim has already been loaded in, in order to compute the box center (via utils.pull_halo_center()), then no need to do it again
                print_mpi('ds ' + str(ds) + ' for halo ' + str(this_sim[0]) + ' was already loaded at some point by utils; using that loaded ds henceforth', args)
            else:
                ds, refine_box = load_sim(args, region='refine_box', do_filter_particles=False)

            # -------create new fields for angular momentum vectors-----------
            ds.add_field(('gas', 'angular_momentum_phi'), function=phi_angular_momentum, sampling_type='cell', units='degree')
            ds.add_field(('gas', 'angular_momentum_theta'), function=theta_angular_momentum, sampling_type='cell', units='degree')

        except (FileNotFoundError, PermissionError) as e:
            print_mpi('Skipping ' + this_sim[1] + ' because ' + str(e), dummy_args)
            continue

        if args.fullbox:
            box_width = ds.refine_width  # kpc
            args.galrad = box.width / 2
            box = refine_box
        else:
            box_center = ds.arr(args.halo_center, kpc)
            box_width = args.galrad * 2  # in kpc
            box_width_kpc = ds.arr(box_width, 'kpc')
            box = ds.r[box_center[0] - box_width_kpc / 2.: box_center[0] + box_width_kpc / 2., box_center[1] - box_width_kpc / 2.: box_center[1] + box_width_kpc / 2., box_center[2] - box_width_kpc / 2.: box_center[2] + box_width_kpc / 2., ]

        bounds_dict = defaultdict(lambda: None, rad=(0, args.galrad), density=(1e-31, 1e-21), temp=(1e1, 1e8), metal=(1e-2, 1e1), vrad=(-400, 400), phi_L=(0, 180), theta_L=(-180, 180))  # in g/cc, range within box; hard-coded for Blizzard RD0038; but should be broadly applicable to other snaps too
        projected_bounds_dict = defaultdict(lambda x: bounds_dict[x], density=(density_proj_min, density_proj_max))

        args.current_redshift = ds.current_redshift
        args.current_time = ds.current_time.in_units('Gyr')
        colorcol_arr = args.colorcol

        for index, thiscolorcol in enumerate(colorcol_arr):
            args.colorcol = thiscolorcol
            colorcol = 'log_' + args.colorcol if islog_dict[args.colorcol] else args.colorcol
            if weight_field_dict[args.colorcol] and args.weight: colorcol += '_wtby_' + weight_field_dict[args.colorcol]
            colorcol_cat = 'cat_' + colorcol
            print_mpi('Plotting ' + xcol + ' vs ' + ycol + ', color coded by ' + colorcol + ' i.e., plot ' + str(index + 1) + ' of ' + str(len(colorcol_arr)) + '..', args)

            outfile_rootname = 'datashader_boxrad_%.2Fkpc_%s_vs_%s_colby_%s.png' % (args.galrad, ycol, xcol, colorcol)
            if args.do_all_sims: outfile_rootname = 'z=*_' + outfile_rootname

            thisfilename = fig_dir + outfile_rootname.replace('*', '%.5F' % (args.current_redshift))

            if not os.path.exists(thisfilename) or args.clobber_plot:
                if not os.path.exists(thisfilename):
                    print_mpi(thisfilename + ' plot does not exist. Creating afresh..', args)
                elif args.clobber_plot:
                    print_mpi(thisfilename + ' plot exists but over-writing..', args)

                df, fig = make_datashader_plot(box, thisfilename, args, npix_datashader=npix_datashader)
                if args.interactive:
                    myprint('This plot is now in interactive mode..', args)
                    fig = setup_interactive(box, fig.axes[0], args, npix_datashader=npix_datashader, nbins=nselection_bins)
            else:
                print_mpi('Skipping colorcol ' + thiscolorcol + ' because plot already exists (use --clobber_plot to over-write) at ' + thisfilename, args)

        print_mpi('This snapshot ' + this_sim[1] + ' completed in %s minutes' % ((time.time() - start_time_this_snapshot) / 60), args)

    comm.Barrier() # wait till all cores reached here and then resume

    if args.makemovie and args.do_all_sims:
        print_master('Finished creating snapshots, calling animate_png.py to create movie..', args)
        for thiscolorcol in colorcol_arr:
            args.colorcol = thiscolorcol
            colorcol = 'log_' + args.colorcol if islog_dict[args.colorcol] else args.colorcol
            outfile_rootname = 'z=*_datashader_boxrad_%.2Fkpc_%s_vs_%s_colby_%s.png' % (args.galrad, ycol, xcol, colorcol)
            subprocess.call(['python ' + HOME + '/Work/astro/ayan_codes/animate_png.py --inpath ' + fig_dir + ' --rootname ' + outfile_rootname + ' --delay ' + str(args.delay_frame) + ' --reverse'], shell=True)

    if ncores > 1: print_master('Parallely: time taken for datashading ' + str(total_snaps) + ' snapshots with ' + str(ncores) + ' cores was %s mins' % ((time.time() - start_time) / 60), dummy_args)
    else: print_master('Serially: time taken for datashading ' + str(total_snaps) + ' snapshots with ' + str(ncores) + ' core was %s mins' % ((time.time() - start_time) / 60), dummy_args)
