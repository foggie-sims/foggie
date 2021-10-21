#!/usr/bin/env python3

"""

    Title :      datashader_movie
    Notes :      Attempt to make datashader plots of 3 given quantities from FOGGIE outputs and make movies
    Output :     datashader plots as png files (which can be later converted to a movie via animate_png.py)
    Author :     Ayan Acharyya
    Started :    July 2021
    Examples :   run datashader_movie.py --system ayan_hd --halo 8508 --galrad 20 --xcol rad --ycol metal --colorcol vrad --weight density --overplot_stars --do_all_sims --makemovie --delay 0.2
                 run datashader_movie.py --system ayan_hd --halo 8508,5036 --fullbox --xcol rad --ycol metal --colorcol vrad --output RD0020,RD0030 --clobber_plot
                 run datashader_movie.py --system ayan_hd --halo 8508 --galrad 20 --xcol rad --ycol temp --colorcol density,metal,vrad,phi_L,theta_L --output RD0042 --clobber_plot --overplot_stars --keep
                 run datashader_movie.py --system ayan_hd --halo 8508 --galrad 20 --xcol rad --ycol metal --colorcol vrad,density,temp,phi_L,theta_L --output RD0042 --clobber_plot --overplot_stars --keep
                 run datashader_movie.py --system ayan_local --halo 8508 --do gas --galrad 20 --xcol rad --ycol metal --colorcol temp --output RD0030 --clobber_plot --overplot_stars --interactive --selcol --combine
                 run datashader_movie.py --system ayan_pleiades --halo 8508 --run nref10n.3000 --galrad 200 --xcol rad --ycol metal --colorcol temp,vrad,density --do_all_sims --clobber_plot
                 run datashader_movie.py --system ayan_pleiades --halo 8508 --run full_run_with_mrp_fix.64 --galrad 200 --xcol rad --ycol metal --colorcol temp,vrad,density --do_all_sims --clobber_plot

"""
from header import *
from util import *
from projection_plot import do_plot
from make_ideal_datacube import shift_ref_frame
from filter_star_properties import get_star_properties

from matplotlib.colors import to_hex
from matplotlib.widgets import LassoSelector
from matplotlib.widgets import SpanSelector
import matplotlib as mpl

from functools import partial

yt_ver = yt.__version__
start_time = time.time()

# ------------------------------------------------------------------------------
def weight_by(data, weights):
    '''
    Function to compute weighted array
    '''
    weighted_data = data * weights * len(weights) / np.sum(weights)
    return weighted_data

# ---------------------------------------------
def get_text_between_strings(full_string, left_string, right_string):
    '''
    Function to extract the string between two strings in a given parent string
    '''
    return full_string[full_string.find(left_string) + len(left_string):full_string.find(right_string)]

# ---------------------------------------------
def get_correct_tablename(args):
    '''
    Function to determine the correct tablename for a given set of args
    '''
    if args.quick: outfileroot = args.output_dir + 'txtfiles/' + args.output + '_df_boxrad_*kpc_%s_vs_%s_colby_%s%s.txt' % (args.ycolname, args.xcolname, args.colorcolname, inflow_outflow_text)
    else: outfileroot = args.output_dir + 'txtfiles/' + args.output + '_df_boxrad_*kpc.txt'

    outfile_list = glob.glob(outfileroot)
    if len(outfile_list) == 0:
        correct_rad_to_grab = args.galrad
    else:
        available_rads = np.sort([float(get_text_between_strings(item, 'boxrad_', 'kpc')) for item in outfile_list])
        try:
            index = np.where(available_rads >= args.galrad)[0][0]
            correct_rad_to_grab = available_rads[index]
        except IndexError:
            correct_rad_to_grab = args.galrad
            pass

    outfilename = outfileroot.replace('*', '%.2F' % correct_rad_to_grab)
    return outfilename

# -------------------------------------------------------------------------------
def extract_columns_from_df(df_allprop, args):
    '''
    Function to extract only the relevant fields/columns for this analysis
    :return: dataframe
    '''
    if args.inflow_only: df_allprop = df_allprop[df_allprop['vrad'] < 0.]
    elif args.outflow_only: df_allprop = df_allprop[df_allprop['vrad'] > 0.]

    df = pd.DataFrame()
    all_fields = [args.xcol, args.ycol, args.colorcol]
    if 'rad' not in all_fields: all_fields = ['rad'] + all_fields

    for field in all_fields:
        arr = df_allprop[field]
        column_name = field
        df[column_name] = arr
        if isfield_weighted_dict[field] and args.weight:
            df[args.weight] = df_allprop[args.weight]
            arr = weight_by(arr, df[args.weight])
            column_name = column_name + '_wtby_' + args.weight
        if islog_dict[field] and (field == args.colorcol or not args.use_cvs_log):
            arr = np.log10(arr)
            column_name = 'log_' + column_name
        if column_name not in df: df[column_name] = arr

    return df

# -------------------------------------------------------------------------------
def get_df_from_ds(ds, args):
    '''
    Function to make a pandas dataframe from the yt dataset based on the given field list and color category,
    then writes dataframe to file for faster access in future
    This function is somewhat based on foggie.utils.prep_dataframe.prep_dataframe()
    :return: dataframe
    '''
    # -------------read/write pandas df file with ALL fields-------------------
    Path(args.output_dir + 'txtfiles/').mkdir(parents=True, exist_ok=True)  # creating the directory structure, if doesn't exist already
    outfilename = get_correct_tablename(args)

    if not os.path.exists(outfilename) or args.clobber:
        if not os.path.exists(outfilename):
            myprint(outfilename + ' does not exist. Creating afresh..', args)
        elif args.clobber:
            myprint(outfilename + ' exists but over-writing..', args)

        if not args.quick: myprint('Extracting all gas, once and for all, and putting them into dataframe, so that this step is not required for subsequently plotting other parameters and therefore subsequent plotting is faster; this may take a while..', args)
        df_allprop = pd.DataFrame()

        all_fields = [args.xcol, args.ycol, args.colorcol] if args.quick else field_dict.keys() # only the relevant properties if in a hurry
        if 'rad' not in all_fields: all_fields = ['rad'] + all_fields

        for index,field in enumerate(all_fields):
            myprint('Doing property: ' + field + ', which is ' + str(index + 1) + ' of the ' + str(len(all_fields)) + ' fields..', args)
            if 'phi' in field and 'disk' in field: df_allprop[field] = np.abs(np.degrees(ds[field_dict[field]].v) - 90) # to convert from radian to degrees; and then restrict phi from 0 to 90
            elif 'theta' in field and 'disk' in field: df_allprop[field] = np.degrees(ds[field_dict[field]].v) # to convert from radian to degrees
            else: df_allprop[field] = ds[field_dict[field]].in_units(unit_dict[field]).ndarray_view()

        df_allprop.to_csv(outfilename, sep='\t', index=None)
    else:
        myprint('Reading from existing file ' + outfilename, args)
        df_allprop = pd.read_table(outfilename, delim_whitespace=True, comment='#')
        try:
            df_allprop = df_allprop[df_allprop['rad'].between(0, args.galrad)] # curtailing in radius space, in case this dataframe has been read in from a file corresponding to a larger chunk of the box
        except KeyError: # for files produced previously and therefore may not have a 'rad' column
            rad_picked_up = float(get_text_between_strings(outfilename, 'boxrad_', 'kpc'))
            if rad_picked_up == args.galrad: pass # if this file actually corresponds to the correct radius, then you're fine (even if the file itself doesn't have radius column)
            else: sys.exit('Please regenerate ' + outfilename + ', using the --clobber option') # otherwise throw error

    df = extract_columns_from_df(df_allprop, args)
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
def convert_to_datashader_frame(data, data_min, data_max, npix_datashader, log_scale=False):
    '''
    Function to convert physical quantities to corresponding pixel values in the datashader image frame
    '''
    if log_scale:
        data, data_min, data_max = np.log10(data), np.log10(data_min), np.log10(data_max)
    data_in_pix = (data - data_min) * npix_datashader / (data_max - data_min)
    return data_in_pix

# ---------------------------------------------------------------------------------
def convert_from_datashader_frame(data, data_min, data_max, npix_datashader):
    '''
    Function to convert pixel values on a datashader image to corresponding physical quantities
    '''
    return data * (data_max - data_min) / npix_datashader + data_min

# ---------------------------------------------------------------------------------
def load_stars_file(args):
    '''
    Function to load the young star parameters file
    '''
    starlistfile = args.output_dir + 'txtfiles/' + args.output + '_young_star_properties.txt'
    Path(args.output_dir + 'txtfiles/').mkdir(parents=True, exist_ok=True)  # creating the directory structure, if doesn't exist already

    # -------------to read in simulation data------------
    if not os.path.exists(starlistfile):
        print_mpi(starlistfile + 'does not exist. Calling get_star_properties() first..', args)
        dummy = get_star_properties(args)  # this creates the infilename
    else:
        print_mpi('Reading young star properties from ' + starlistfile + '..', args)
    starlist = pd.read_table(starlistfile, delim_whitespace=True, comment='#')

    # -------------to prep the simulation data------------
    starlist = shift_ref_frame(starlist, args)
    starlist = starlist.rename(columns={'gas_metal': 'metal', 'gas_density': 'density', 'gas_pressure': 'pressure', 'gas_temp': 'temp'})
    starlist = get_radial_velocity(starlist)
    starlist = starlist[starlist['rad'].between(0, args.galrad)] # to overplot only those young stars that are within the desired radius ('rad' is in kpc)

    if args.inflow_only: starlist = starlist[starlist['vrad'] < 0.]
    elif args.outflow_only: starlist = starlist[starlist['vrad'] > 0.]

    for field in [args.xcol, args.ycol, args.colorcol]:
        if field not in starlist.columns:
            print_mpi(field + ' does not exist in young star list, therefore cannot overplot stars..', args)
            return pd.DataFrame() # return empty dataframe
        column_name = field
        if isfield_weighted_dict[field] and args.weight:
            weightcol = args.weight
            if weightcol in starlist.columns: weights = starlist[weightcol]
            else: weights = np.ones(len(starlist))
            starlist[column_name + '_wtby_' + weightcol] = weight_by(starlist[column_name], weights)
            column_name = column_name + '_wtby_' + weightcol
        if islog_dict[field] and (field == args.colorcol or not args.use_cvs_log): starlist['log_' + column_name] = np.log10(starlist[column_name])

    return starlist

# ---------------------------------------------------------------------------------
def load_absorbers_file(args):
    '''
    Function to load the absorbers file made by Claire
    '''
    abslistfile = '/'.join(args.output_dir.split('/')[:-3]) + '/foggie_absorbers_200kpc_flow.csv' # the absorber file is in the directory output_path as defined in get_run_loc_etc.py for a given args.system

    # -------------to read in simulation data------------
    if not os.path.exists(abslistfile):
        print_mpi(abslistfile + 'does not exist. Cannot overplot absorbers.', args)
        return pd.DataFrame() # return empty dataframe
    else:
        print_mpi('Reading absorber properties from ' + abslistfile + '..', args)
    abslist = pd.read_csv(abslistfile, comment='#')

    # -------------to prep the simulation data------------
    abslist = abslist[abslist['name'] == 'H I'] # using HI absorbers only, because file itself has CIV and other absorbers too
    if args.current_redshift is not None:
        abslist = abslist[abslist['redshift'].between(args.current_redshift * 0.999, args.current_redshift * 1.001)]
        if len(abslist) == 0: myprint('Warning: no absorbers remaining to plot after the redshift cut', args)

    abslist = abslist.rename(columns={'metallicity': 'metal', 'col_dens': 'col_density', 'radius': 'rad', 'temperature': 'temp'})
    abslist['vrad'] = abslist['velocity_magnitude'] * abslist['radial_alignment'] / np.abs(abslist['radial_alignment'])

    abslist = abslist[['rad', 'vrad', 'metal', 'col_density', 'temp']]
    abslist['col_density'] = 10 ** abslist['col_density'] # because column density was reported as log in Claire's file
    abslist = abslist[abslist['rad'].between(0, args.galrad)] # to overplot only those absorbers that are within the desired radius ('rad' is in kpc)

    if args.inflow_only: abslist = abslist[abslist['vrad'] < 0.]
    elif args.outflow_only: abslist = abslist[abslist['vrad'] > 0.]

    for field in [args.xcol, args.ycol, args.colorcol]:
        if field not in abslist.columns:
            print_mpi(field + ' does not exist in absorbers list, therefore cannot overplot absorbers..', args)
            return pd.DataFrame() # return empty dataframe
        column_name = field
        if isfield_weighted_dict[field] and args.weight:
            weightcol = args.weight
            if weightcol in abslist.columns: weights = abslist[weightcol]
            else: weights = np.ones(len(abslist))
            abslist[column_name + '_wtby_' + weightcol] = weight_by(abslist[column_name], weights)
            column_name = column_name + '_wtby_' + weightcol
        if islog_dict[field] and (field == args.colorcol or not args.use_cvs_log): abslist['log_' + column_name] = np.log10(abslist[column_name])

    return abslist

# ---------------------------------------------------------------------------------
def overplot_stars(paramlist, axes, args, type='stars', npix_datashader=1000):
    '''
    Function to overplot young stars on existing datashader plot.
    Needs a pandas dataframe (paramlist) as an input; paramlist should have at least three columns
    = args.xcolname, args.ycolname and args.colorcolname. See the keys of the global dictionaries
    "field_dict" and "unit_dict" (below) for what should the column names exactly be and what units
    they need to be in, e.g., for a radial metallicity profile color-coded with temperature, paramlist
    should (at least) have the columns "rad", "metal" and "temp". However, if you wish the metallicity
    and temperature to be in log scale, it should already be in log-scale in the paramlist being input here
    AND the column names in the dataframe in that case need to be "rad", "log_metal" and "log_temperature", etc.
    It does not matter if paramlist has other extra columns. Also see the "bounds_dict" global dictionary
    for limits on each parameter. Values outside these ranges will not be plotted, so you may want to change that.
    '''
    marker_dict = {'stars':'o', 'absorbers':'s'}
    ax = axes.ax_joint
    init_len = len(paramlist)
    if init_len > 0:
        paramlist = paramlist[(paramlist[args.xcolname].between(args.xmin, args.xmax)) & (paramlist[args.ycolname].between(args.ymin, args.ymax)) & (paramlist[args.colorcolname].between(args.cmin, args.cmax))]

        # -------------to actually plot the simulation data------------
        if datashader_ver <= 11 or args.use_old_dsh:
            x_on_plot = convert_to_datashader_frame(paramlist[args.xcolname], args.xmin, args.xmax, npix_datashader, log_scale=islog_dict[args.xcol] and args.use_cvs_log) # because we need to stretch the binned x and y into npix_datashader dimensions determined by the datashader plot
            y_on_plot = convert_to_datashader_frame(paramlist[args.ycolname], args.ymin, args.ymax, npix_datashader, log_scale=islog_dict[args.ycol] and args.use_cvs_log)
            x_min_on_plot, x_max_on_plot = 0, npix_datashader
            y_min_on_plot, y_max_on_plot = 0, npix_datashader
        else:
            x_on_plot, y_on_plot = paramlist[args.xcolname], paramlist[args.ycolname]
            x_min_on_plot, x_max_on_plot = args.xmin, args.xmax
            y_min_on_plot, y_max_on_plot = args.ymin, args.ymax

        overplotted = ax.scatter(x_on_plot, y_on_plot, c=paramlist[args.colorcolname], vmin=args.cmin, vmax=args.cmax, edgecolors='black', lw=0 if type == 'absorbers' and len(paramlist) > 700 else 0.2, s=5 if type == 'absorbers' and len(paramlist) > 700 else 15, marker=marker_dict[type], cmap=args.cmap)

        # ----------to plot 1D histogram on the top and right axes--------------
        if type == 'absorbers':
            axes.ax_marg_x = plot_1D_histogram(x_on_plot, x_min_on_plot, x_max_on_plot, axes.ax_marg_x, vertical=False, type=type)
            axes.ax_marg_y = plot_1D_histogram(y_on_plot, y_min_on_plot, y_max_on_plot, axes.ax_marg_y, vertical=True, type=type)

        print_mpi('Overplotted ' + str(len(paramlist)) + ' of ' + str(init_len) + ', i.e., ' + '%.2F' % (len(paramlist) * 100 / init_len) + '% of ' + type + ' inside this box..', args)

    return overplotted, axes

# ---------------------------------------------------------------------------------
def overplot_binned(df, ax, args, npix_datashader=1000):
    '''
    Function to overplot binned data on existing datashader plot
    Uses globally defined islog_dict
    '''
    if args.xcol == 'rad':
        x_bins = np.logspace(np.log10(args.xmin + 0.001), np.log10(args.xmax), 200) # the 0.001 term is to avoid taking log of 0, in case args.xmin = 0
    else:
        x_bin_size = bin_size_dict[args.xcol]
        x_bins = np.arange(args.xmin, args.xmax + x_bin_size, x_bin_size)
    df['binned_cat'] = pd.cut(df[args.xcolname], x_bins)

    if isfield_weighted_dict[args.ycol] and args.weight: agg_func = lambda x: np.mean(weight_by(x, df.loc[x.index, args.weight])) # function to get weighted mean
    else: agg_func = np.mean
    y_binned = df.groupby('binned_cat', as_index=False).agg([(args.ycol, agg_func)])[args.ycol]
    if islog_dict[args.ycol] and not args.use_cvs_log: y_binned = np.log10(y_binned)

    # ----------to plot mean binned y vs x profile--------------
    x_bin_centers = x_bins[:-1] + np.diff(x_bins) / 2
    if datashader_ver <= 11 or args.use_old_dsh:
        x_on_plot = convert_to_datashader_frame(x_bin_centers, args.xmin, args.xmax, npix_datashader, log_scale=islog_dict[args.xcol] and args.use_cvs_log) # because we need to stretch the binned x and y into npix_datashader dimensions determined by the datashader plot
        y_on_plot = convert_to_datashader_frame(y_binned, args.ymin, args.ymax, npix_datashader, log_scale=islog_dict[args.ycol] and args.use_cvs_log)
    else:
        x_on_plot, y_on_plot = x_bin_centers, y_binned

    ax.plot(x_on_plot, y_on_plot, color='black', lw=1)

    return ax

# ---------------------------------------------------------------------------------
def plot_1D_histogram(data, data_min, data_max, ax, vertical=False, type='gas'):
    '''
    Function to plot marginalised histograms using seaborn
    '''
    linestyle_dict = {'gas':('solid', 'black'), 'stars':('dotted', 'brown'), 'absorbers':('dashed', 'darkgreen')}
    sns.kdeplot(data, ax=ax, legend=False, lw=1, vertical=vertical, linestyle=linestyle_dict[type][0], color=linestyle_dict[type][1])
    ax.tick_params(axis='x', which='both', top=False)
    if vertical: ax.set_ylim(data_min, data_max)
    else: ax.set_xlim(data_min, data_max)

    return ax

# ---------------------------------------------------------------------------------
def make_coordinate_axis(colname, data_min, data_max, ax, fontsize, npix_datashader=1000, dsh=True, log_scale=False):
    '''
    Function to make the coordinate axis
    Uses globally defined islog_dict and unit_dict
    '''
    log_text = 'Log ' if islog_dict[colname] and not log_scale else ''
    ax.set_label_text(log_text + labels_dict[colname] + ' (' + unit_dict[colname] + ')', fontsize=fontsize)

    nticks = 50 if log_scale else 5
    ticks = np.linspace(data_min, data_max, nticks)
    if dsh: ax.set_ticks(convert_to_datashader_frame(ticks, data_min, data_max, npix_datashader, log_scale=log_scale))
    else: ax.set_ticks(ticks)

    nticklabels = 6 if log_scale else 5
    ticklabels = np.array(['     '] * nticks)
    if log_scale:
        label_arr = np.linspace(np.log10(data_min), np.log10(data_max), nticklabels + 1)
        dig = np.digitize(np.log10(ticks), label_arr)
        bins = [np.where(dig == i)[0][0] for i in np.unique(dig)]
        for bin in bins: ticklabels[bin] = '%.1F' % ticks[bin]
    else:
        ticklabel_every = int(nticks/nticklabels)
        ticklabels[:: ticklabel_every] = ['%.1F' % item for item in ticks[:: ticklabel_every]]

    ax.set_ticklabels(ticklabels, fontsize=fontsize)

    return ax

# -----------------------------------------------------------------------------------
def create_foggie_cmap_old(c_min, c_max, color_list):
    '''
    Function to create the colorbar for the tiny colorbar axis on top right
    This function is based on Cassi's foggie.pressure_support.create_foggie_cmap(), which is in turn based on Jason's foggie.render.cmap_utils.create_foggie_cmap()
    '''
    n = 100000
    color_key = get_color_keys(c_min, c_max, color_list)
    n_labels = len(color_key)

    df = pd.DataFrame({'x':np.random.rand(n), 'y': np.random.rand(n)})
    edges = np.linspace(np.min(df['x']), np.max(df['x']), n_labels + 1)

    df['cat'] = ''
    for index in range(n_labels):
        df['cat'][(df['x'] >= edges[index]) & (df['x'] <= edges[index + 1])] = list(color_key)[index]
    df.cat = df.cat.astype('category')

    cvs = dsh.Canvas(plot_width=750, plot_height=100, x_range=(np.min(df['x']), np.max(df['x'])), y_range=(np.min(df['y']), np.max(df['y'])))
    agg = cvs.points(df, 'x', 'y', dsh.count_cat('cat'))
    cmap = dstf.spread(dstf.shade(agg, color_key=color_key, how='eq_hist', min_alpha=40), px=2, shape='square')

    return cmap

# ---------------------------------------------------------------------------------
def make_colorbar_axis_old(colname, data_min, data_max, fig, fontsize, color_list):
    '''
    Function to make the colorbar axis
    Uses globally defined islog_dict and unit_dict
    '''
    color_field_cmap = create_foggie_cmap_old(data_min, data_max, color_list)
    ax_xpos, ax_ypos, ax_width, ax_height = 0.7, 0.82, 0.25, 0.06
    ax = fig.add_axes([ax_xpos, ax_ypos, ax_width, ax_height])
    cbar_im = color_field_cmap.to_pil()
    ax.imshow(cbar_im)
    log_text = 'Log ' if islog_dict[colname] else ''
    delta_c = 200 if data_max - data_min > 200 else 60 if data_max - data_min > 60 else 10 if data_max - data_min > 10 else 2
    
    ax.set_xticks(np.arange((data_max - data_min) + 1., step=delta_c) * np.shape(cbar_im)[1] / (data_max - data_min))
    ax.set_xticklabels(['%.0F' % index for index in np.arange(data_min, data_max + 1, delta_c)], fontsize=fontsize/1.5)#, weight='bold')
    
    fig.text(ax_xpos + ax_width / 2, ax_ypos + ax_height, log_text + labels_dict[colname] + ' (' + unit_dict[colname] + ')', fontsize=fontsize/1.5, ha='center', va='bottom')

    for item in ['top', 'bottom', 'left', 'right']: ax.spines[item].set_color('white')
    ax.set_yticklabels([])
    ax.set_yticks([])

    return fig, ax


# ---------------------------------------------------------------------------------
def make_colorbar_axis(colname, data_min, data_max, fig, fontsize):
    '''
    Function to make the colorbar axis with discrete color values, but without using JT's create_foggie_cmap()
    Uses globally defined islog_dict and unit_dict
    '''
    discrete_cmap = colormap_dict[colname]
    normalised_cmap = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=data_min, vmax=data_max), cmap=discrete_cmap)
    cax_xpos, cax_ypos, cax_width, cax_height = 0.7, 0.82, 0.25, 0.06
    cax = fig.add_axes([cax_xpos, cax_ypos, cax_width, cax_height])
    plt.colorbar(normalised_cmap, cax=cax, orientation='horizontal')

    cax.set_xticklabels(['%.0F' % index for index in cax.get_xticks()], fontsize=fontsize / 1.5)

    log_text = 'Log ' if islog_dict[colname] else ''
    fig.text(cax_xpos + cax_width / 2, cax_ypos + cax_height + 0.005, log_text + labels_dict[colname] + ' (' + unit_dict[colname] + ')', fontsize=fontsize/1.5, ha='center', va='bottom')

    return fig, cax

# ---------------------------------------------------------------------------------
def make_colorbar_axis_mpl(colname, artist, fig, fontsize):
    '''
    Function to make the colorbar axis
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

# -----------------------------------------------------------------------------------
def wrap_axes(df, filename, npix_datashader, args, paramlist=None, abslist=None):
    '''
    Function to read in raw datashader plot and wrap it in axes using matplotlib AND added x- and y- marginalised histograms using seaborn
    This function is partly based on foggie.render.shade_maps.wrap_axes()
    :return: fig
    '''
    # ----------to get quantities in datashader pixel units--------------
    df[args.xcolname + '_in_pix'] = convert_to_datashader_frame(df[args.xcolname], args.xmin, args.xmax, npix_datashader, log_scale=islog_dict[args.xcol] and args.use_cvs_log)
    df[args.ycolname + '_in_pix'] = convert_to_datashader_frame(df[args.ycolname], args.ymin, args.ymax, npix_datashader, log_scale=islog_dict[args.ycol] and args.use_cvs_log)

    # -----------------to initialise figure---------------------
    shift_right = 'phi' in args.ycol or 'theta' in args.ycol
    axes = sns.JointGrid(args.xcolname + '_in_pix', args.ycolname + '_in_pix', df, height=8)
    extra_space = 0.03 if shift_right else 0
    plt.subplots_adjust(hspace=0.05, wspace=0.05, right=0.95 + extra_space, top=0.95, bottom=0.1, left=0.1 + extra_space)
    fig, ax1 = plt.gcf(), axes.ax_joint

    # ------to plot datashader image from file-------------
    img = mpimg.imread(filename)
    ax1.imshow(np.flip(img, 0))

    # ----------to overplot young stars----------------
    if args.overplot_stars: axes = overplot_stars(paramlist, axes, args, type='stars', npix_datashader=npix_datashader)

    # ----------to overplot young stars----------------
    if args.overplot_absorbers: axes = overplot_stars(abslist, axes, args, type='absorbers', npix_datashader=npix_datashader)

    # ----------to overplot binned profile----------------
    ax1 = overplot_binned(df, ax1, args, npix_datashader=npix_datashader)

    # ----------to plot 1D histogram on the top and right axes--------------
    axes.ax_marg_x = plot_1D_histogram(df[args.xcolname + '_in_pix'], 0, npix_datashader, axes.ax_marg_x, vertical=False)
    axes.ax_marg_y = plot_1D_histogram(df[args.ycolname + '_in_pix'], 0, npix_datashader, axes.ax_marg_y, vertical=True)

    # ------to make the axes-------------
    ax1.xaxis = make_coordinate_axis(args.xcol, args.xmin, args.xmax, ax1.xaxis, args.fontsize, npix_datashader=npix_datashader, dsh=True, log_scale=islog_dict[args.xcol] and args.use_cvs_log)
    ax1.yaxis = make_coordinate_axis(args.ycol, args.ymin, args.ymax, ax1.yaxis, args.fontsize, npix_datashader=npix_datashader, dsh=True, log_scale=islog_dict[args.ycol] and args.use_cvs_log)
    fig, ax2 = make_colorbar_axis(args.colorcol, args.cmin, args.cmax, fig, args.fontsize)

    # ---------to annotate and save the figure----------------------
    if args.current_redshift is not None: plt.text(0.033, 0.05, 'z = %.4F' % args.current_redshift, transform=ax1.transAxes, fontsize=args.fontsize)
    if args.current_time is not None: plt.text(0.033, 0.1, 't = %.3F Gyr' % args.current_time, transform=ax1.transAxes, fontsize=args.fontsize)
    plt.savefig(filename, transparent=False)
    myprint('Saved figure ' + filename, args)
    if not args.makemovie: plt.show(block=False)

    return fig

# --------------------------------------------------------------------------------
def make_datashader_plot(df, outfilename, args, npix_datashader=1000, paramlist=None, abslist=None):
    '''
    Function to make data shader plot of y_field vs x_field, colored in bins of color_field
    This function is based on foggie.render.shade_maps.render_image()
    :return dataframe, figure
    '''
    # ----------to filter and categorize the dataframe--------------
    df = df[(df[args.xcolname].between(args.xmin, args.xmax)) & (df[args.ycolname].between(args.ymin, args.ymax)) & (df[args.colorcolname].between(args.cmin, args.cmax))]
    df[args.colorcol_cat] = categorize_by_quant(df[args.colorcolname], args.cmin, args.cmax, args.ncolbins)
    df[args.colorcol_cat] = df[args.colorcol_cat].astype('category')

    cvs = dsh.Canvas(plot_width=npix_datashader, plot_height=npix_datashader, x_range=(args.xmin, args.xmax), y_range=(args.ymin, args.ymax), x_axis_type='log' if islog_dict[args.xcol] and args.use_cvs_log else 'linear', y_axis_type='log' if islog_dict[args.ycol] and args.use_cvs_log  else 'linear')
    agg = cvs.points(df, args.xcolname, args.ycolname, dsh.count_cat(args.colorcol_cat))
    color_key = get_color_keys(args.cmin, args.cmax, args.color_list)
    img = dstf.spread(dstf.shade(agg, color_key=color_key, how='eq_hist', min_alpha=40), shape='square')
    export_image(img, os.path.splitext(outfilename)[0])

    fig = wrap_axes(df, os.path.splitext(outfilename)[0] + '.png', npix_datashader, args, paramlist=paramlist, abslist=abslist)
    #fig = img # to bypass wrap_axes, comment out previous line and uncomment this line

    return df, fig

# --------------------------------------------------------------------------------
def make_datashader_plot_mpl(df, outfilename, args, paramlist=None, abslist=None):
    '''
    Function to make data shader plot of y_field vs x_field, colored in bins of color_field
    This function is based on foggie.render.shade_maps.render_image()
    This is different from make_datashader_plot() in that this function uses the newest version of datashader which has matplotlib support
    So this function essentially combines make_datashader_plot() and wrap_axes(), because the latter is not needed anymore
    :return dataframe, figure
    '''
    # ----------to filter and categorize the dataframe--------------
    df = df[(df[args.xcolname].between(args.xmin, args.xmax)) & (df[args.ycolname].between(args.ymin, args.ymax)) & (df[args.colorcolname].between(args.cmin, args.cmax))]
    df[args.colorcol_cat] = categorize_by_quant(df[args.colorcolname], args.cmin, args.cmax, args.ncolbins)
    df[args.colorcol_cat] = df[args.colorcol_cat].astype('category')

    # -----------------to initialise figure---------------------
    shift_right = 'phi' in args.ycol or 'theta' in args.ycol
    axes = sns.JointGrid(args.xcolname, args.ycolname, df, height=8)
    extra_space = 0.03 if shift_right else 0
    plt.subplots_adjust(hspace=0.05, wspace=0.05, right=0.95 + extra_space, top=0.95, bottom=0.1, left=0.1 + extra_space)
    fig, ax1 = plt.gcf(), axes.ax_joint

    # --------to make the main datashader plot--------------------------
    color_key = get_color_keys(args.cmin, args.cmax, args.color_list)
    if not args.nofoggie:
        #artist = dsshow(df, dsh.Point(args.xcolname, args.ycolname), dsh.count_cat(args.colorcol_cat), norm='eq_hist', color_key=color_key, x_range=(args.xmin, args.xmax), y_range=(args.ymin, args.ymax), vmin=args.cmin, vmax=args.cmax, aspect = 'auto', ax=ax1, alpha_range=(40, 255), shade_hook=partial(dstf.spread, shape='square')) # the 40 in alpha_range and `square` in shade_hook are to reproduce original-looking plots as if made with make_datashader_plot()
        artist = dsshow(df, dsh.Point(args.xcolname, args.ycolname), dsh.mean(args.colorcolname), norm='linear', cmap=list(color_key.values()), x_range=(args.xmin, args.xmax), y_range=(args.ymin, args.ymax), vmin=args.cmin, vmax=args.cmax, aspect = 'auto', ax=ax1) #, shade_hook=partial(dstf.spread, px=1, shape='square')) # the 40 in alpha_range and `square` in shade_hook are to reproduce original-looking plots as if made with make_datashader_plot()

    # ----------to overplot young stars----------------
    if args.overplot_stars: overplotted, axes = overplot_stars(paramlist, axes, args, type='stars')

    # ----------to overplot young stars----------------
    if args.overplot_absorbers: overplotted, axes = overplot_stars(abslist, axes, args, type='absorbers')

    if len(df) > 0:
        # ----------to overplot binned profile----------------
        ax1 = overplot_binned(df, ax1, args)

        # ----------to plot 1D histogram on the top and right axes--------------
        axes.ax_marg_x = plot_1D_histogram(df[args.xcolname], args.xmin, args.xmax, axes.ax_marg_x, vertical=False)
        axes.ax_marg_y = plot_1D_histogram(df[args.ycolname], args.ymin, args.ymax, axes.ax_marg_y, vertical=True)

    # ------to make the axes-------------
    #ax1.set_xlim(20, 0) #
    ax1.xaxis = make_coordinate_axis(args.xcol, args.xmin, args.xmax, ax1.xaxis, args.fontsize, dsh=False, log_scale=islog_dict[args.xcol] and args.use_cvs_log)
    ax1.yaxis = make_coordinate_axis(args.ycol, args.ymin, args.ymax, ax1.yaxis, args.fontsize, dsh=False, log_scale=islog_dict[args.ycol] and args.use_cvs_log)

    # ------to make the colorbar axis-------------
    #fig, ax2 = make_colorbar_axis(args.colorcol, args.cmin, args.cmax, fig, args.fontsize)
    fig, ax2 = make_colorbar_axis_mpl(args.colorcol, artist if not args.nofoggie else overplotted, fig, args.fontsize)

    # ---------to annotate and save the figure----------------------
    if args.current_redshift is not None: plt.text(0.033, 0.05, 'z = %.4F' % args.current_redshift, transform=ax1.transAxes, fontsize=args.fontsize)
    if args.current_time is not None: plt.text(0.033, 0.1, 't = %.3F Gyr' % args.current_time, transform=ax1.transAxes, fontsize=args.fontsize)
    filename = os.path.splitext(outfilename)[0] + '.png'
    plt.savefig(filename, transparent=False)
    myprint('Saved figure ' + filename, args)
    if not args.makemovie: plt.show(block=False)

    return df, fig

# ------------------------------------------------------------------
class SelectFromCollection:
    '''
    Object to interactively choose coordinate points in a given plot using the LassoSelector
    This is based on Raymond's foggie.angular_momentum.lasso_data_selection.ipynb

    Select indices from a matplotlib collection using `LassoSelector`.

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values).

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
    Axes to interact with.
    collection : `matplotlib.collections.Collection` subclass
    Collection you want to select from.
    alpha_other : 0 <= float <= 1
    '''
    def __init__(self, ax, xys, nbins,  x_range, y_range, alpha_other=0.3):
        self.ax = ax
        self.alpha_other = alpha_other
        self.xys = xys

        self.lasso = LassoSelector(ax, lambda verts: self.onselect(verts, nbins), useblit=True)
        self.ind = []

        mask = np.zeros((nbins, nbins))

        extent = x_range[0], x_range[1], y_range[0], y_range[1]
        text_xpos, text_ypos = x_range[0] + 0.5 * (x_range[1] - x_range[0]), y_range[0] + 0.95 * (y_range[1] - y_range[0])

        self.helpful_text = self.ax.text(text_xpos, text_ypos, '', ha='center', fontsize=15)
        self.mask_plot = self.ax.imshow(mask, cmap=plt.cm.Greys_r, alpha=0, interpolation='bilinear', extent=extent, zorder=10, aspect='auto', vmin=0, vmax=1)  # , origin='lower')
        self.blitman = BlitManager(self.ax.figure.canvas, [self.mask_plot, self.helpful_text])

    def onselect(self, verts, nbins):
        path = mpl_Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        # -------to highlight the selected region-------------------
        if len(self.ind) > 0:
            mask = np.zeros((nbins, nbins))
            mask.ravel()[self.ind] = 1.
            self.mask_plot.set_data(mask)
            self.mask_plot.set_alpha(self.alpha_other)
            self.helpful_text.set_text('Press ENTER to proceed with this selection or select again')
        else:
            self.mask_plot.set_alpha(0.)
            self.helpful_text.set_text('No data selected; select points by dragging mouse in a loop')
        self.blitman.update()

    def disconnect(self):
        self.helpful_text.set_text('')
        self.mask_plot.set_alpha(0.)
        self.blitman.update()
        self.lasso.disconnect_events()

# ------------------------------------------------------------------
class SelectFromSlider:
    '''
    Object to interactively choose x axis range in a given histogram using the RangeSlider
   '''
    def __init__(self, ax, nbins, data_min, data_max, alpha_other=0.3):
        self.ax = ax
        self.alpha_other = alpha_other
        self.data_min = data_min
        self.data_max = data_max

        onselect = lambda min, max: self.update_span(min, max, nbins)
        self.slider = SpanSelector(ax, onselect, 'horizontal', onmove_callback=onselect, useblit=True, span_stays=True, rectprops=dict(alpha=0))
        self.val = [data_min, data_max]

        mask = np.zeros((nbins, nbins))
        extent = self.data_min, self.data_max, self.ax.get_ylim()[0], self.ax.get_ylim()[1]
        self.mask_plot = self.ax.imshow(mask, cmap=plt.cm.Greys_r, alpha=0, interpolation='bilinear', extent=extent, zorder=10, aspect='auto', vmin=0, vmax=1)

        self.lower_limit_line = ax.axvline(self.data_min, color='k')
        self.upper_limit_line = ax.axvline(self.data_max, color='k')
        self.helpful_text = ax.text(ax.get_xlim()[0] + 0.5 * np.diff(ax.get_xlim())[0], ax.get_ylim()[0] + 0.95 * np.diff(ax.get_ylim())[0], 'Drag cursor to select a range in colors', ha='center', fontsize=args.fontsize)

        # ----------setup the Blit manager and key press event---------------------------
        self.blitman = BlitManager(self.ax.figure.canvas, [self.lower_limit_line, self.upper_limit_line, self.helpful_text])

    def update_span(self, min, max, nbins):
        '''
        Function to update the val passed to a callback by the RangeSlider
        '''
        self.val = [min, max]
        # -------to highlight the selected region-------------------
        if self.val[1] - self.val[0] > 0:
            mask = np.zeros((nbins, nbins))
            mask[:, int((self.val[0] - self.data_min) * nbins/(self.data_max - self.data_min)) : int((self.val[1] - self.data_min) * nbins/(self.data_max - self.data_min))] = 1.
            self.mask_plot.set_data(mask)
            self.mask_plot.set_alpha(self.alpha_other)
            self.helpful_text.set_text('Press ENTER to proceed with this selection or select again')
        else:
            self.mask_plot.set_alpha(0)
            self.helpful_text.set_text('No range selected; select by dragging the cursor')
        # ------update the position of the vertical lines------
        self.lower_limit_line.set_xdata(self.val[0])
        self.upper_limit_line.set_xdata(self.val[1])
        self.blitman.update()

    def disconnect(self):
        self.helpful_text.set_text('')
        self.mask_plot.set_alpha(0)
        self.lower_limit_line.set_xdata(self.data_min)
        self.upper_limit_line.set_xdata(self.data_max)
        self.blitman.update()
        self.slider.disconnect_events()

# -------------------------------------------------------------------------------
def combine_lasso_outputs(args, selection_figname, projection_figname, slider_figname=None):
    '''
    Function to combine the figures produced by lasso selection and display in one frame and write that frame to file
    '''
    myprint('Preparing to combine projection plots based on highlighted selection..', args)
    crop_x1_proj, crop_x2_proj, crop_y1_proj, crop_y2_proj = 150, 3750, 50, 3000

    proj_list_dict = {'dsh': ([10, 1550, 80, 1550], selection_figname, '2D selection')}
    if slider_figname:
        nrow, ncol, figsize = 2, 3, (12, 6)
        proj_list_dict.update({'hist': ([10, 1550, 80, 1550], slider_figname, 'Color selection'), 'dummy': ([0, 0, 0, 0], 'dummy', 'dummy')})
    else:
        nrow, ncol, figsize = 2, 2, (9.7, 8)
    proj_list_dict.update({'x': ([crop_x1_proj, crop_x2_proj, crop_y1_proj, crop_y2_proj], 'proj', 'Los = '), 'y': ([crop_x1_proj, crop_x2_proj, crop_y1_proj, crop_y2_proj], 'proj', 'Los = '), 'z': ([crop_x1_proj, crop_x2_proj, crop_y1_proj, crop_y2_proj], 'proj', 'Los = ')})

    fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
    fig.subplots_adjust(hspace=0, wspace=0, right=1, top=1, bottom=0, left=0)

    for index, thisproj in enumerate(list(proj_list_dict.keys())):
        rowid, colid = int(index / ncol), index % ncol
        axes[rowid][colid].axis('off')

        if proj_list_dict[thisproj][1] == 'proj':
            this_figname = projection_figname.replace('proj_' + args.projection, 'proj_' + thisproj)
            this_label = proj_list_dict[thisproj][2] + thisproj
        else:
            this_figname = proj_list_dict[thisproj][1]
            this_label = proj_list_dict[thisproj][2]

        if not os.path.exists(this_figname): continue
        image = mpimg.imread(this_figname)

        crop_x1, crop_x2, crop_y1, crop_y2 = proj_list_dict[thisproj][0]
        axes[rowid][colid].imshow(image[crop_y1 : crop_y2, crop_x1 : crop_x2], origin='upper')
        fig.text(0.15, 0.15, this_label, color='white', transform=axes[rowid][colid].transAxes, fontsize=args.fontsize, ha='left', va='bottom')

    outfile_rootname = os.path.split(projection_figname)[1].replace('proj_' + args.projection, 'proj_xyz')
    outfile_rootname = os.path.splitext(outfile_rootname)[0]
    outfile_rootname = 'dsh_' + outfile_rootname.replace('projection', 'combined')
    saveplot(fig, args, outfile_rootname, outputdir=os.path.split(selection_figname)[0] + '/')
    plt.show(block=False)

    return fig

# ------------------------------------------------------------------
def make_projections(box_cut, args, identifier, output_selfig, output_slidefig=None):
    '''
    Function to make projection plots once the selection has been finalised
    '''
    # -----make projection plots with the selection-----------
    bounds_dict.update(rad=(0, args.galrad))
    field_to_plot = 'density'
    output_projfig = fig_dir + '%s' % (field_to_plot) + '_boxrad_%.2Fkpc' % (args.galrad) + '_proj_' + args.projection + '_from_' + args.ycolname + '_vs_' + args.xcolname
    if args.selcol: output_projfig += '_colby_' + args.colorcolname
    output_projfig += args.newold_text + '_lasso' + str(identifier) + '_projection.png'

    # ------to make new plots corresponding to the selected region----------
    proj_arr = ['x', 'y', 'z'] if args.combine else [args.projection]

    for thisproj in proj_arr:
        prj = do_plot(box_cut.ds, field_dict[field_to_plot], thisproj, [], box_cut, box.center.in_units(kpc), 2 * args.galrad * kpc, \
                      cmap_dict[field_to_plot], halo_dict[args.halo], unit=projected_unit_dict[field_to_plot], zmin=projected_bounds_dict[field_to_plot][0], \
                      zmax=projected_bounds_dict[field_to_plot][1], weight_field=args.weight if isfield_weighted_dict[field_to_plot] else None, iscolorlog=islog_dict[field_to_plot], \
                      noweight=False)

        this_figname = output_projfig.replace('proj_' + args.projection, 'proj_' + thisproj)
        prj.save(this_figname, mpl_kwargs={'dpi': 500})

    if args.combine: fig = combine_lasso_outputs(args, output_selfig, output_projfig, slider_figname=output_slidefig)

    return prj

# ------------------------------------------------------------------
def on_second_key_press(event, box_cut, ax, args, range_selector, identifier, output_selfig):
    '''
    Function to accept a key press event and display the selected pixels and modify the axis title
    This is based on Raymond's foggie.angular_momentum.lasso_data_selection.ipynb
    '''
    if event.key == 'enter':
        start_time_on_press = time.time()
        x_lower_bound = range_selector.val[0]
        x_upper_bound = range_selector.val[1]

        if x_upper_bound > x_lower_bound:
            # -----save the selection-----------
            output_slidefig = fig_dir + 'datashader_boxrad_%.2Fkpc_%s_vs_%s_colby_%s%s_slider%d.png' % (args.galrad, args.ycolname, args.xcolname, args.colorcolname, args.newold_text, identifier)
            ax.figure.savefig(output_slidefig)  # to save the highlighted image

            # ------to cut out the selected region----------
            if args.debug: myprint('Preparing to cut regions further based on highlighted selection: %s from %.2F to %.2F %s'%(args.colorcolname, x_lower_bound, x_upper_bound, unit_dict[args.colorcol]), args)

            if islog_dict[args.colorcol]: x_lower_bound, x_upper_bound = 10 ** x_lower_bound, 10 ** x_upper_bound

            cut_crit = "((obj[{}] > obj.ds.quan({}, '{}')) & (obj[{}] < obj.ds.quan({}, '{}')))".format(field_dict[args.colorcol], x_lower_bound, unit_dict[args.colorcol], field_dict[args.colorcol], x_upper_bound, unit_dict[args.colorcol])

            box_cut = box_cut.cut_region([cut_crit])

            # ------make projection plots----------
            prj = make_projections(box_cut, args, identifier, output_selfig, output_slidefig=output_slidefig)

        else:
            myprint('No data point selected; therefore, not proceeding any further.', args)

        myprint('Second key press events completed in %s' % (datetime.timedelta(seconds=time.time() - start_time_on_press)), args)

    elif event.key == 'escape':
        myprint('Exiting interactive mode..', args)
        range_selector.disconnect()
    else:
        myprint('You pressed ' + event.key + '; but you need to press ENTER to see your selection or ESC to exit interactive mode', args)

# -------------------------------------------------------------------
def make_histogram(box_cut, args, identifier, output_selfig, ncolbins):
    '''
    Function to prepare histogram with interactive range_selector, once the 2D selection is finalised
    '''
    # ----------to plot the histogram of colors---------------------------
    fig, ax = plt.subplots(1, figsize=(8, 8))
    color_quant = box_cut[field_dict[args.colorcol]]
    if islog_dict[args.colorcol]: color_quant = np.log10(color_quant)

    Y, X = np.histogram(color_quant, bins=ncolbins, normed=False)
    colors = [args.cmap((x - args.cmin) / (args.cmax - args.cmin)) for x in X]

    h = ax.bar(X[:-1], Y, color=colors, width=X[1] - X[0])
    ax.xaxis = make_coordinate_axis(args.colorcol, args.cmin, args.cmax, ax.xaxis, args.fontsize, dsh=False)
    ax.yaxis = make_coordinate_axis('PDF', 0, np.max(Y), ax.yaxis, args.fontsize, dsh=False)

    # ---------setup the range range_selector on the histogram------------------
    range_selector = SelectFromSlider(ax, ncolbins, args.cmin, args.cmax)
    fig.canvas.mpl_connect("key_press_event", lambda event: on_second_key_press(event, box_cut, ax, args, range_selector, identifier, output_selfig))

    return ax

# ------------------------------------------------------------------
def on_first_key_press(event, box, ax, args, nbins, selector, xgrid, ygrid, ncolbins=100, npix_datashader=None):
    '''
    Function to accept a key press event and display the selected pixels and modify the axis title
    This is based on Raymond's foggie.angular_momentum.lasso_data_selection.ipynb
    '''
    if event.key == 'enter':
        start_time_on_press = time.time()
        nselection = len(selector.ind)

        if nselection > 0:
            # -----save the selection-----------
            identifier = random.randint(10, 99)
            output_selfig = fig_dir + 'datashader_boxrad_%.2Fkpc_%s_vs_%s_colby_%s%s_lasso%d.png' % (args.galrad, args.ycolname, args.xcolname, args.colorcolname, args.newold_text, identifier)
            ax.figure.savefig(output_selfig)  # to save the highlighted image
            if args.debug: print('Selected ' + str(nselection) + ' binned pixels:', selector.xys[selector.ind])

            # ------to cut out the selected region----------
            if args.debug: myprint('Preparing to cut regions based on highlighted selection..', args)
            cut_crit = ""
            for index, item in enumerate(selector.ind):
                if npix_datashader is None: # this plot is made using datashader's native mpl support
                    x_lower_bound = xgrid.ravel()[item] - 0.5 * (args.xmax - args.xmin) / nbins
                    x_upper_bound = xgrid.ravel()[item] + 0.5 * (args.xmax - args.xmin) / nbins
                    y_lower_bound = ygrid.ravel()[item] - 0.5 * (args.ymax - args.ymin) / nbins
                    y_upper_bound = ygrid.ravel()[item] + 0.5 * (args.ymax - args.ymin) / nbins
                else:
                    x_lower_bound = convert_from_datashader_frame(xgrid.ravel()[item] - 0.5 * npix_datashader / nbins, args.xmin, args.xmax, npix_datashader)
                    x_upper_bound = convert_from_datashader_frame(xgrid.ravel()[item] + 0.5 * npix_datashader / nbins, args.xmin, args.xmax, npix_datashader)
                    y_lower_bound = convert_from_datashader_frame(ygrid.ravel()[item] - 0.5 * npix_datashader / nbins, args.ymin, args.ymax, npix_datashader)
                    y_upper_bound = convert_from_datashader_frame(ygrid.ravel()[item] + 0.5 * npix_datashader / nbins, args.ymin, args.ymax, npix_datashader)

                if islog_dict[args.xcol] and not args.use_cvs_log: x_lower_bound, x_upper_bound = 10 ** x_lower_bound, 10 ** x_upper_bound
                if islog_dict[args.ycol] and not args.use_cvs_log: y_lower_bound, y_upper_bound = 10 ** y_lower_bound, 10 ** y_upper_bound

                if index: cut_crit += '|'
                cut_crit += "((obj[{}] > obj.ds.quan({}, '{}')) & (obj[{}] < obj.ds.quan({}, '{}')) & (obj[{}] > obj.ds.quan({}, '{}')) & (obj[{}] < obj.ds.quan({}, '{}')))".format(
                    field_dict[args.xcol], x_lower_bound, unit_dict[args.xcol], field_dict[args.xcol], x_upper_bound,
                    unit_dict[args.xcol], field_dict[args.ycol], y_lower_bound, unit_dict[args.ycol],
                    field_dict[args.ycol],
                    y_upper_bound, unit_dict[args.ycol])

            box_cut = box.cut_region([cut_crit])

            if args.selcol: ax_hist = make_histogram(box_cut, args, identifier, output_selfig, ncolbins=ncolbins) # make histogram for selection in color-space
            else: prj = make_projections(box_cut, args, identifier, output_selfig) # OR directly plot projections with all underlying color values
            selector.ind = []
        else:
            myprint('No data point selected; therefore, not proceeding any further.', args)

        myprint('First key press events completed in %s' % (datetime.timedelta(seconds=time.time() - start_time_on_press)), args)

    elif event.key == 'escape':
        myprint('Exiting interactive mode..', args)
        selector.disconnect()

    else:
        myprint('You pressed ' + event.key + '; but you need to press ENTER to see your selection or ESC to exit interactive mode', args)

# -------------------------------------------------------------------------------------------------------------
def setup_interactive(box, ax, args, nbins=20, ncolbins=100, npix_datashader=None):
    '''
    Function to interactively select points from a live plot using LassoSelector
    This is based on Raymond's foggie.angular_momentum.lasso_data_selection.ipynb
    '''
    if npix_datashader is None: x_range, y_range = [args.xmin, args.xmax], [args.ymin, args.ymax]
    else: x_range, y_range = [0, npix_datashader], [0, npix_datashader]

    x = np.linspace(x_range[0], x_range[1], nbins)
    y = np.linspace(y_range[0], y_range[1], nbins)
    xgrid, ygrid = np.meshgrid(x, y)
    xys = np.array([xgrid.ravel(),ygrid.ravel()]).T
    selector = SelectFromCollection(ax, xys, nbins, x_range, y_range)

    fig = plt.gcf()
    fig.canvas.mpl_connect("key_press_event", lambda event: on_first_key_press(event, box, ax, args, nbins, selector, xgrid, ygrid, ncolbins=ncolbins, npix_datashader=npix_datashader))

    selector.helpful_text.set_text('Select points by dragging mouse in a loop')
    selector.blitman.update()

    return fig

# -------------------------------------------------------------------------------------------------------------
def get_color_labels(data_min, data_max, nbins):
    '''
    Function to determine the bin labels of every color category, to make the datashader plot
    This function is still here only to retain backward compatibility with previous versions of datashader which did not
    have matplotlib support; otherwise these functions are redundant with the new mpl support
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
    This function is still here only to retain backward compatibility with previous versions of datashader which did not
    have matplotlib support; otherwise these functions are redundant with the new mpl support
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
    This function is still here only to retain backward compatibility with previous versions of datashader which did not
    have matplotlib support; otherwise these functions are redundant with the new mpl support
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

projected_unit_dict = defaultdict(lambda x: unit_dict[x], density='Msun/pc**2')
cmap_dict = {'density':density_color_map, 'metal':metal_color_map, 'temp':temperature_color_map, 'vrad':velocity_discrete_cmap} # for projection plots, if any
projected_bounds_dict = {'density': (density_proj_min, density_proj_max)}

# -----main code-----------------
if __name__ == '__main__':
    npix_datashader, nselection_bins = 1000, 50

    dummy_args_tuple = parse_args('8508', 'RD0042')  # default simulation to work upon when comand line args not provided
    if type(dummy_args_tuple) is tuple: dummy_args = dummy_args_tuple[0] # if the sim has already been loaded in, in order to compute the box center (via utils.pull_halo_center()), then no need to do it again
    else: dummy_args = dummy_args_tuple
    if dummy_args.xcol == 'radius': dummy_args.xcol == 'rad'
    if not dummy_args.keep: plt.close('all')

    if dummy_args.do_all_sims:
        list_of_sims = get_all_sims(dummy_args) # all snapshots of this particular halo
    else:
        if dummy_args.do_all_halos: halos = get_all_halos(dummy_args)
        else: halos = dummy_args.halo_arr
        list_of_sims = list(itertools.product(halos, dummy_args.output_arr))
    total_snaps = len(list_of_sims)
    if dummy_args.dryrun:
        print('List of the total ' + str(total_snaps) + ' sims =', list_of_sims)
        sys.exit('Exiting dryrun..')
    # parse column names, in case log
    dummy_args.xcolname = 'log_' + dummy_args.xcol if islog_dict[dummy_args.xcol] and not dummy_args.use_cvs_log else dummy_args.xcol
    dummy_args.ycolname = 'log_' + dummy_args.ycol if islog_dict[dummy_args.ycol] and not dummy_args.use_cvs_log else dummy_args.ycol
    if isfield_weighted_dict[dummy_args.xcol] and dummy_args.weight: dummy_args.xcolname += '_wtby_' + dummy_args.weight
    if isfield_weighted_dict[dummy_args.ycol] and dummy_args.weight: dummy_args.ycolname += '_wtby_' + dummy_args.weight
    colorcol_arr = dummy_args.colorcol

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
            if len(list_of_sims) == 1 and not dummy_args.do_all_sims: args = dummy_args_tuple # since parse_args() has already been called and evaluated once, no need to repeat it
            else: args = parse_args(this_sim[0], this_sim[1])

            if type(args) is tuple:
                args, ds, refine_box = args  # if the sim has already been loaded in, in order to compute the box center (via utils.pull_halo_center()), then no need to do it again
                print_mpi('ds ' + str(ds) + ' for halo ' + str(this_sim[0]) + ' was already loaded at some point by utils; using that loaded ds henceforth', args)
            else:
                isdisk_required = np.array(['disk' in item for item in [args.xcol, args.ycol] + args.colorcol]).any()
                ds, refine_box = load_sim(args, region='refine_box', do_filter_particles=True, disk_relative=(isdisk_required or args.diskload) and not args.nodiskload)

            # -------create new fields for angular momentum vectors-----------
            ds.add_field(('gas', 'angular_momentum_phi'), function=phi_angular_momentum, sampling_type='cell', units='degree')
            ds.add_field(('gas', 'angular_momentum_theta'), function=theta_angular_momentum, sampling_type='cell', units='degree')

        except Exception as e:
            print_mpi('Skipping ' + this_sim[1] + ' because ' + str(e), dummy_args)
            continue

        # parse paths and filenames
        fig_dir = args.output_dir + 'figs/' if args.do_all_sims else args.output_dir + 'figs/' + args.output + '/'
        Path(fig_dir).mkdir(parents=True, exist_ok=True)

        if args.fullbox:
            box_width = ds.refine_width  # kpc
            args.galrad = box_width / 2
            box = refine_box
        else:
            box_center = ds.arr(args.halo_center, kpc)
            box_width = args.galrad * 2  # in kpc
            box_width_kpc = ds.arr(box_width, 'kpc')
            box = ds.r[box_center[0] - box_width_kpc / 2.: box_center[0] + box_width_kpc / 2., box_center[1] - box_width_kpc / 2.: box_center[1] + box_width_kpc / 2., box_center[2] - box_width_kpc / 2.: box_center[2] + box_width_kpc / 2., ]

        # ----------to update dicts if parameter not already present--------------
        bounds_dict.update(rad=(0, args.galrad))
        if args.xcol not in field_dict: update_dicts(args.xcol, ds)
        if args.ycol not in field_dict: update_dicts(args.ycol, ds)

        # ----------to determine axes limits--------------
        if args.xmin is None:
            args.xmin = np.log10(bounds_dict[args.xcol][0]) if islog_dict[args.xcol] and not args.use_cvs_log else bounds_dict[args.xcol][0]
        if args.xmax is None:
            args.xmax = np.log10(bounds_dict[args.xcol][1]) if islog_dict[args.xcol] and not args.use_cvs_log else bounds_dict[args.xcol][1]
        if args.ymin is None:
            args.ymin = np.log10(bounds_dict[args.ycol][0]) if islog_dict[args.ycol] and not args.use_cvs_log else bounds_dict[args.ycol][0]
        if args.ymax is None:
            args.ymax = np.log10(bounds_dict[args.ycol][1]) if islog_dict[args.ycol] and not args.use_cvs_log else bounds_dict[args.ycol][1]

        if args.inflow_only:
            box = box.cut_region(["obj[('gas', 'radial_velocity_corrected')] < 0"])
            inflow_outflow_text = '_inflow_only'
        elif args.outflow_only:
            box = box.cut_region(["obj[('gas', 'radial_velocity_corrected')] > 0"])
            inflow_outflow_text = '_outflow_only'
        else:
            inflow_outflow_text = ''

        nofoggie_text = '_nofoggie' if args.nofoggie else ''
        abs_text = '_wabs' if args.overplot_absorbers else ''

        if datashader_ver <= 11 or args.use_old_dsh:
            args.newold_text = '' # for backward compatibility
        else:
            args.newold_text = '_newdsh'

        args.current_redshift = ds.current_redshift
        args.current_time = ds.current_time.in_units('Gyr')
        args.xcolname, args.ycolname = dummy_args.xcolname, dummy_args.ycolname

        for index, thiscolorcol in enumerate(colorcol_arr):
            args.colorcol = thiscolorcol
            args.colorcolname = 'log_' + args.colorcol if islog_dict[args.colorcol] else args.colorcol
            args.colorcol_cat = 'cat_' + args.colorcolname
            # ----------to update dicts if parameter not already present--------------
            if args.colorcol not in field_dict: update_dicts(args.colorcol, ds)

            # ----------to determine axes limits--------------
            if args.cmin is None:
                args.cmin = np.log10(bounds_dict[args.colorcol][0]) if islog_dict[args.colorcol] else bounds_dict[args.colorcol][0]
            if args.cmax is None:
                args.cmax = np.log10(bounds_dict[args.colorcol][1]) if islog_dict[args.colorcol] else bounds_dict[args.colorcol][1]

            if isfield_weighted_dict[args.colorcol] and args.weight: args.colorcolname += '_wtby_' + args.weight

            # ----------to determine colorbar parameters--------------
            if args.cmap is None: args.cmap = colormap_dict[args.colorcol]
            else: args.cmap = plt.get_cmap(args.cmap)
            color_list = args.cmap.colors
            ncolbins = args.ncolbins if args.ncolbins is not None else len(color_list) if len(color_list) <= 10 else 7
            args.color_list = color_list[::int(len(color_list) / ncolbins)]  # truncating color_list in to a length of rougly ncolbins
            args.ncolbins = len(args.color_list)

            print_mpi('Plotting ' + args.xcolname + ' vs ' + args.ycolname + ', color coded by ' + args.colorcolname + ' i.e., plot ' + str(index + 1) + ' of ' + str(len(colorcol_arr)) + '..', args)

            outfile_rootname = 'datashader_boxrad_%.2Fkpc_%s_vs_%s_colby_%s%s%s%s%s.png' % (args.galrad, args.ycolname, args.xcolname, args.colorcolname, inflow_outflow_text, args.newold_text, nofoggie_text, abs_text)
            if args.do_all_sims: outfile_rootname = 'z=*_' + outfile_rootname

            thisfilename = fig_dir + outfile_rootname.replace('*', '%.5F' % (args.current_redshift))

            if not os.path.exists(thisfilename) or args.clobber_plot or args.quick:
                if not os.path.exists(thisfilename):
                    print_mpi(thisfilename + ' plot does not exist. Creating afresh..', args)
                elif args.clobber_plot:
                    print_mpi(thisfilename + ' plot exists but over-writing..', args)

                if args.nofoggie: df = pd.DataFrame(columns=[args.xcolname, args.ycolname, args.colorcolname])
                else: df = get_df_from_ds(box, args)

                paramlist = load_stars_file(args) if args.overplot_stars else None
                abslist = load_absorbers_file(args) if args.overplot_absorbers else None
                if datashader_ver <= 11 or args.use_old_dsh:
                    df, fig = make_datashader_plot(df, thisfilename, args, npix_datashader=npix_datashader, paramlist=paramlist, abslist=abslist)
                else:
                    df, fig = make_datashader_plot_mpl(df, thisfilename, args, paramlist=paramlist, abslist=abslist)
                if args.interactive:
                    myprint('This plot is now in interactive mode..', args)
                    if not (datashader_ver <= 11 or args.use_old_dsh): npix_datashader = None # so that the functions that use datashader's native mpl support are invoked
                    fig = setup_interactive(box, fig.axes[0], args, nbins=nselection_bins, ncolbins=args.ncolbins, npix_datashader=npix_datashader)
            else:
                print_mpi('Skipping colorcol ' + thiscolorcol + ' because plot already exists (use --clobber_plot to over-write) at ' + thisfilename, args)

        print_mpi('This snapshot ' + this_sim[1] + ' completed in %s minutes' % ((time.time() - start_time_this_snapshot) / 60), args)

    comm.Barrier() # wait till all cores reached here and then resume

    if dummy_args.makemovie and dummy_args.do_all_sims:
        print_master('Finished creating snapshots, calling animate_png.py to create movie..', dummy_args)
        code_path = '/'.join(dummy_args.code_path.split('/')[:-3]) + '/'
        if dummy_args.do_all_halos: halos = get_all_halos(args)
        else: halos = dummy_args.halo_arr
        for thishalo in halos:
            args_tuple = parse_args(thishalo, 'RD0020') # RD0020 is inconsequential here, just a place-holder
            if type(args_tuple) is tuple: args = args_tuple[0]  # if the sim has already been loaded in, in order to compute the box center (via utils.pull_halo_center()), then no need to do it again
            else: args = args_tuple
            args.xcolname, args.ycolname = dummy_args.xcolname, dummy_args.ycolname
            fig_dir = args.output_dir + 'figs/'
            for thiscolorcol in colorcol_arr:
                args.colorcol = thiscolorcol
                args.colorcolname = 'log_' + args.colorcol if islog_dict[args.colorcol] else args.colorcol
                outfile_rootname = 'z=*_datashader_boxrad_%.2Fkpc_%s_vs_%s_colby_%s.png' % (args.galrad, args.ycolname, args.xcolname, args.colorcolname)
                subprocess.call(['python ' + code_path + 'animate_png.py --inpath ' + fig_dir + ' --rootname ' + outfile_rootname + ' --delay ' + str(args.delay_frame) + ' --reverse'], shell=True)

    if ncores > 1: print_master('Parallely: time taken for datashading ' + str(total_snaps) + ' snapshots with ' + str(ncores) + ' cores was %s mins' % ((time.time() - start_time) / 60), dummy_args)
    else: print_master('Serially: time taken for datashading ' + str(total_snaps) + ' snapshots with ' + str(ncores) + ' core was %s mins' % ((time.time() - start_time) / 60), dummy_args)
