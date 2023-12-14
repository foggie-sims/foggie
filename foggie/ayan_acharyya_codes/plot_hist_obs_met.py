#!/usr/bin/env python3

"""

    Title :      plot_hist_obs_met
    Notes :      Plot histogram of metallicity distribution for OBSERVED metallicity maps
    Output :     histograms as png files
    Author :     Ayan Acharyya
    Started :    Oct 2023
    Examples :   run plot_hist_obs_met.py --keep --xmin -1.5 --xmax 1 --ymin 0 --ymax 3 --fontsize 20 --halo 8508 --upto_kpc 10 --weight mass --docomoving --use_density_cut --get_native_res --nbins 30 --add_foggie_panel --output DD0538,DD0738,DD0838,DD1038
                 run plot_hist_obs_met.py --keep --xmin -1.5 --xmax 1 --ymin 0 --ymax 3 --fontsize 20 --halo 8508 --upto_kpc 10 --weight mass --docomoving --use_density_cut --get_native_res --nbins 30 --overplot_foggie
                 run plot_hist_obs_met.py --keep --xmin -1.5 --xmax 1 --ymin 0 --ymax 3 --fontsize 20 --halo 8508 --upto_kpc 10 --weight mass --docomoving --use_density_cut --res_arc 0.1 --nbins 20 --overplot_foggie
                 run plot_hist_obs_met.py --keep --fontsize 20 --add_foggie_panel --output DD0538,DD0738,DD0838,DD1038
"""
from header import *
from util import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
start_time = time.time()

# ---------------------------------------
def get_foggie_met(args):
    '''
    Function to plot histogram based on a FOGGIE snapshot
    '''
    if args.upto_kpc is not None: upto_text = '_upto%.1Fckpchinv' % args.upto_kpc if args.docomoving else '_upto%.1Fkpc' % args.upto_kpc
    else: upto_text = '_upto%.1FRe' % args.upto_re
    density_cut_text = '_wdencut' if args.use_density_cut else ''
    if args.get_native_res:
        ncells_text = ''
        res_text = '_res_native'
    else:
        if args.upto_kpc is not None:
            if args.docomoving: args.galrad = args.upto_kpc / (1 + args.current_redshift) / 0.695  # fit within a fixed comoving kpc h^-1, 0.695 is Hubble constant
            else: args.galrad = args.upto_kpc  # fit within a fixed physical kpc
        else:
            args.galrad = args.re * args.upto_re  # kpc

        box_width = args.galrad * 2  # in kpc

        if args.res_arc is not None:
            res_text = '_res%.1farcsec' %args.res_arc
            args.res = get_kpc_from_arc_at_redshift(args.res_arc, args.current_redshift)
            native_res_at_z = 0.27 / (1 + args.current_redshift)  # converting from comoving kpc to physical kpc
            if args.res < native_res_at_z:
                print('Computed resolution %.2f kpc is below native FOGGIE res at z=%.2f, so we set resolution to the native res = %.2f kpc.' % (args.res, args.current_redshift, native_res_at_z))
                args.res = native_res_at_z  # kpc
        else:
            args.res = float(args.res)
            res_text = '_res%.1fkpc' %args.res
            if args.docomoving: args.res = args.res / (1 + args.current_redshift) / 0.695  # converting from comoving kcp h^-1 to physical kpc
        ncells = int(box_width / args.res)
        ncells_text = '_ncells%d' % ncells
    outfilename = args.output_dir + 'txtfiles/' + args.output + '_df_boxrad_%s%s%s.txt' % (upto_text, density_cut_text, ncells_text)
    print('Reading existing file', outfilename)

    df = pd.read_table(outfilename, delim_whitespace=True, comment='#')
    met = df['log_metal'].values if 'log_metal' in df else np.log10(df['metal'].values)
    weights = df[args.weight].values if args.weight is not None else None

    return met, weights, df['radius'], upto_text + density_cut_text + res_text

# ---------------------------------------------------------
def remove_nans(list1, list2):
    '''
    Function to remove NaNs from all lists provided
    Returns shortened lists
    '''
    good_indices = np.array(np.logical_not(np.logical_or(np.isnan(list1), np.isnan(list2))))
    list1 = list1[good_indices]
    list2 = list2[good_indices]
    return list1, list2

# -----------------------------------------------------
def get_dist_map(npix, kpc_per_pix=1):
    '''
    Function to get a map of distance of each from the center
    '''
    center_pix = (npix - 1)/2.
    map_dist = np.array([[np.sqrt((i - center_pix)**2 + (j - center_pix)**2) for j in range(npix)] for i in range(npix)]) * kpc_per_pix # kpc

    return map_dist

# -----------------------------------------------------------------
def plot_grad_hist_map(galaxies, args):
    '''
    Function to plot radial gradient, histogram and 2D map of a given galaxy in a single row
    '''
    ncol = 3
    nrow = len(galaxies)

    zsun_min, zsun_max = -0.5, 0.8  # zsun
    rad_max, units = 10, 'pix' # 'kpc'
    kpc_per_pix = 1.
    pdf_max, nbins = 5, 30

    fig, axes = plt.subplots(nrow, ncol, figsize=(10, 8))
    fig.subplots_adjust(hspace=0.05, wspace=0.3, right=0.97, top=0.95, bottom=0.05, left=0.07)

    for index, thisgal in enumerate(galaxies):
        print('Reading %s which is %d out of %d files..' % (thisgal, index+1, len(galaxies)))
        # loading the data
        if args.add_foggie_panel and ('RD' in thisgal or 'DD' in thisgal):
            thislabel = 'FOGGIE; z= ' + str(foggie_z_dict[thisgal]) # for labeling on the plot
            args.output = thisgal # for reading in FOGGIE files
            args.current_redshift = foggie_z_dict[args.output]
            log_met_zsun_list, weights_list, radius_list, suffix_text = get_foggie_met(args)
            #log_met_zsun_map = log_met_zsun_list
        else:
            thispath = input_path + thisgal + '_zmap.fits'
            thislabel = 'CLEAR; z= ' + str(clear_z_dict[thisgal]) # for labeling on the plot

            log_met_oh_map = fits.open(thispath)[1].data
            log_met_zsun_map = log_met_oh_map - logOH_sun
            weights_list = None
            radius_map = get_dist_map(np.shape(log_met_zsun_map)[0], kpc_per_pix=kpc_per_pix)
            log_met_zsun_list = log_met_zsun_map.flatten()
            radius_list = radius_map.flatten()
            radius_list, log_met_zsun_list = remove_nans(radius_list, log_met_zsun_list)

        ax_row = axes[index] if nrow > 1 else axes

        # -----------plotting radial profile-----------------
        ax = ax_row[0]
        ax.scatter(radius_list, log_met_zsun_list, s=30, color='cornflowerblue')
        linefit, linecov = np.polyfit(radius_list, log_met_zsun_list, 1, cov=True, w=weights_list)
        xarr = np.linspace(0, rad_max, 10)
        ax.plot(xarr, np.poly1d(linefit)(xarr), color='blue', lw=2, ls='solid')
        ax.set_xlim(0, rad_max)
        ax.set_ylim(zsun_min, zsun_max)
        ax.text(ax.get_xlim()[1]*0.9, ax.get_ylim()[1]*0.9, 'Slope = %.2F dex/%s' % (linefit[0], units), ha='right', va='top', fontsize=args.fontsize)
        if index == nrow - 1:
            ax.set_xticklabels(['%.1F' % item for item in ax.get_xticks()], fontsize=args.fontsize)  # last row
            ax.set_xlabel('Galactocentric distance (%s)'%units, fontsize=args.fontsize)
        else:
            ax.set_xticklabels(['' for item in ax.get_xticks()], fontsize=args.fontsize)  # last row
            ax.set_xlabel(']', fontsize=args.fontsize)
        ax.set_yticklabels(['%.1F' % item for item in ax.get_yticks()], fontsize=args.fontsize)  # first column
        ax.set_ylabel(r'Log Metallicity (Z$_{\odot}$)', fontsize=args.fontsize)

        # ---------------plotting histogram----------------------
        ax = ax_row[1]
        ax.hist(log_met_zsun_list, bins=nbins, histtype='step', density=True, lw=1, weights=weights_list, range=(zsun_min, zsun_max), color='cornflowerblue')
        ax.set_xlim(zsun_min, zsun_max)
        ax.set_ylim(0, pdf_max)
        if index == nrow - 1:
            ax.set_xticklabels(['%.1F' % item for item in ax.get_xticks()], fontsize=args.fontsize)  # last row
            ax.set_xlabel(r'Log Metallicity (Z$_{\odot}$)', fontsize=args.fontsize)
        else:
            ax.set_xticklabels(['' for item in ax.get_xticks()], fontsize=args.fontsize)  # last row
            ax.set_xlabel(']', fontsize=args.fontsize)
        ax.set_yticklabels(['%.1F' % item for item in ax.get_yticks()], fontsize=args.fontsize)  # first column
        ax.set_ylabel('Normalised distribution', fontsize=args.fontsize)

        # -------------------plotting map---------------------------------
        ax = ax_row[2]
        p = ax.imshow(log_met_zsun_map, cmap='Blues', extent=(-rad_max/2, rad_max/2, -rad_max/2, rad_max/2), vmin=zsun_min, vmax=zsun_max)
        ax.set_xlim(-rad_max/2, rad_max/2)
        ax.set_ylim(-rad_max/2, rad_max/2)
        if index == nrow - 1:
            ax.set_xticklabels(['%.1F' % item for item in ax.get_xticks()], fontsize=args.fontsize)  # last row
            ax.set_xlabel('Offset (%s)'%units, fontsize=args.fontsize)
        else:
            ax.set_xticklabels(['' for item in ax.get_xticks()], fontsize=args.fontsize)  # last row
            ax.set_xlabel(']', fontsize=args.fontsize)
        ax.set_yticklabels(['%.1F' % item for item in ax.get_yticks()], fontsize=args.fontsize)  # first column
        ax.set_ylabel('Offset (%s)'%units, fontsize=args.fontsize)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(p, cax=cax, orientation='vertical')
        cax.set_ylabel(r'Log Metallicity (Z$_{\odot}$)', fontsize=args.fontsize)

    add_panel_text = '_foggie_panel_' + ','.join(args.output_arr) if args.add_foggie_panel else ''
    if 'suffix_text' not in locals(): suffix_text = ''
    figname = input_path + 'CLEAR_met_grad_hist_map%s.png' %(suffix_text)
    fig.savefig(figname)
    print('Saved images as', figname)

    return fig

# ----------------------------------------------------------------------
def plot_all_hist(galaxies, args):
    '''
    Function to plot only the histograms and save as png, optionally overlaid with FOGGIe histograms
    '''
    nrow, ncol = 2, 3

    fig, axes = plt.subplots(nrow, ncol, figsize=(8, 8), sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.05, wspace=0.05, right=0.95, top=0.95, bottom=0.1, left=0.15)

    if args.xmin is not None and args.xmin > 7: args.xmin = args.xmin - logOH_sun # to convert it from logOH to Zsun units
    if args.xmax is not None and args.xmax > 7: args.xmax = args.xmax - logOH_sun # to convert it from logOH to Zsun units

    for index, thisgal in enumerate(galaxies):
        print('Reading %s which is %d out of %d files..' % (thisgal, index+1, len(galaxies)))
        if args.add_foggie_panel and ('RD' in thisgal or 'DD' in thisgal):
            ax = axes[nrow - 1][ncol - 1] # all FOGGIE overplotted in the last panel
            thislabel = 'FOGGIE; z= ' + str(foggie_z_dict[thisgal]) # for labeling on the plot
            args.output = thisgal # for reading in FOGGIE files
            args.current_redshift = foggie_z_dict[args.output]
            log_met_zsun, weights, radius, suffix_text = get_foggie_met(args)
            color = foggie_color_arr[index - len(galaxies) + len(args.output_arr)]
            yoffset = 0.15 * (index - len(galaxies) + len(args.output_arr))
            lw = 1 if len(args.output_arr) > 1 else 2
        else:
            ax = axes[int(index / ncol)][index % ncol]
            thispath = input_path + thisgal + '_zmap.fits'
            thislabel = 'CLEAR; z= ' + str(clear_z_dict[thisgal]) # for labeling on the plot

            log_met_oh = fits.open(thispath)[1].data.flatten()
            log_met_zsun = log_met_oh - logOH_sun
            weights, color, yoffset, lw = None, 'cornflowerblue', 0, 2

        ax.hist(log_met_zsun, bins=args.nbins, histtype='step', density=True, lw=lw, weights=weights, range=(args.xmin, args.xmax) if args.xmin is not None else None, color=color)

        if args.xmin is not None: ax.set_xlim(args.xmin, args.xmax)
        if args.ymin is not None: ax.set_ylim(args.ymin, args.ymax)

        ax.text(ax.get_xlim()[1] * 0.99, ax.get_ylim()[1] * 0.99 - yoffset, thislabel, fontsize=args.fontsize/1.5, va='top', ha='right', color=color)

        if args.overplot_foggie and clear_foggie_dict[thisgal] is not None: # for plotting corresponding FOGGIE snapshot
            args.output = clear_foggie_dict[thisgal] # for reading in FOGGIE files
            args.current_redshift = foggie_z_dict[args.output]
            thislabel = 'FOGGIE; z= ' + str(foggie_z_dict[args.output]) # for labeling on the plot
            log_met_zsun, weights, suffix_text = get_foggie_met(args)
            color = 'salmon'
            ax.hist(log_met_zsun, bins=args.nbins, histtype='step', density=True, lw=1, weights=weights, range=(args.xmin, args.xmax) if args.xmin is not None else None, color=color)
            ax.text(ax.get_xlim()[1] * 0.99, ax.get_ylim()[1] * 0.99 - 0.17, thislabel, fontsize=args.fontsize/1.5, va='top', ha='right', color=color)

        if int(index / ncol) == nrow - 1: ax.set_xticklabels(['%.1F' % item for item in ax.get_xticks()], fontsize=args.fontsize) # last row
        if index % ncol == 0: ax.set_yticklabels(['%.2F' % item for item in ax.get_yticks()], fontsize=args.fontsize) # first column

    fig.text(0.5, 0.05, r'Log Metallicity (Z$_{\odot}$)', ha='center', va='top', fontsize=args.fontsize)
    fig.text(0.01, 0.5, 'Normalised distribution', ha='left', va='center', rotation='vertical', fontsize=args.fontsize)

    overplot_text = '_foggie_overplotted' if args.overplot_foggie else ''
    add_panel_text = '_foggie_panel_' + ','.join(args.output_arr) if args.add_foggie_panel else ''
    if 'suffix_text' not in locals(): suffix_text = ''
    figname = input_path + 'CLEAR_met_histograms%s%s%s.png' %(overplot_text, add_panel_text, suffix_text)
    fig.savefig(figname)
    print('Saved images as', figname)

    return fig

# -----main code-----------------
if __name__ == '__main__':
    args_tuple = parse_args('8508', 'RD0042')  # default simulation to work upon when comand line args not provided
    if type(args_tuple) is tuple: args = args_tuple[0] # if the sim has already been loaded in, in order to compute the box center (via utils.pull_halo_center()), then no need to do it again
    else: args = args_tuple
    if not args.keep: plt.close('all')

    input_path = HOME + '/Downloads/for_ayan/'
    galaxies = ['GN2_17579', 'GS3_40108', 'GS4_24615', 'GS4_20651', 'GS5_44519']
    if args.add_foggie_panel: galaxies = galaxies + args.output_arr
    foggie_color_arr = ['salmon', 'tan', 'saddlebrown', 'darkorange']
    logOH_sun = 8.69 # Asplund+2009

    foggie_z_dict = {'DD0138':4.3, 'DD0287':2.9, 'RD0020':2.0, 'DD0538':1.84, 'DD0638':1.58, 'DD0738':1.37, 'DD0838':1.19, 'DD1038':0.9, 'RD0030':0.7, 'RD0042':0.0}
    clear_z_dict = {'GS3_40108':1.31, 'GS4_24615':1.32, 'GS4_20651':1.25, 'GN2_17579':1.91, 'GS5_44519':0.93} # got these from Raymond Simons
    clear_foggie_dict = defaultdict(lambda: None, GS3_40108='DD0738', GS4_24615='DD0738', GS4_20651='DD0838', GN2_17579='DD0538', GS5_44519='DD1038') # corresponding FOGGIE snapshots to compare CLEAR galaxies to, based on redshift

    #fig = plot_all_hist(galaxies, args) # plot all histograms
    fig = plot_grad_hist_map(galaxies, args)

    plt.show(block=False)
    print('Completed in %s' % (datetime.timedelta(minutes=(time.time() - start_time) / 60)))

