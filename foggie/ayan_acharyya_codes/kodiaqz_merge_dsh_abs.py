#!/usr/bin/env python3

"""

    Title :      kodiaqz_merge_dsh_abs.py
    Notes :      Make 4 panel plots for KODIAQ-Z paper by plotting absorber catalogue plus merging existing dataframes of different snapshots/halos into a single datashader plot
                 This script is hard-wired for KODIAQ-Z paper, so not a lot of flexibility.
                 This script assumes that all the relevant dataframes already exist, otherwise an error is raised.
    Output :     a 4 panel datashader plot as png file
    Author :     Ayan Acharyya
    Started :    October 2021
    Examples :   run kodiaqz_merge_dsh_abs.py --system ayan_pleiades --do_all_halos --galrad 150 --xcol rad --ycol metal --colorcol vrad --ymin -5 --ymax 1 --output RD0020,RD0018,RD0016
                 run kodiaqz_merge_dsh_abs.py --system ayan_hd --halo 8508 --galrad 150 --xcol rad --ycol metal --colorcol vrad --ymin -5 --ymax 1 --output RD0020
"""
from header import *
from util import *
from datashader_movie import *
import matplotlib.gridspec as gridspec

start_time = time.time()

# -------------------------------------------------------
class SeabornFig2Grid():
    '''
    Class to help "move" seaborn jointplot grids as a subplot in a multi-panel figure
    From stackoverflow solution: https://stackoverflow.com/questions/35042255/how-to-plot-multiple-seaborn-jointplot-in-subplot
    '''
    def __init__(self, seaborngrid, fig,  subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or \
            isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n,m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i,j], self.subgrid[i,j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h= self.sg.ax_joint.get_position().height
        h2= self.sg.ax_marg_x.get_position().height
        r = int(np.round(h/h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r+1,r+1, subplot_spec=self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        #https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure=self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())

# ---------------------------------------------------------------------------------
def hexplot_abs(abslist, axes, args, cmap):
    '''
    Function to make hexplot of absorbers
    '''
    init_len = len(abslist)
    abslist = abslist[(abslist[args.xcolname].between(args.xmin, args.xmax)) & (abslist[args.ycolname].between(args.ymin, args.ymax)) & (abslist[args.colorcolname].between(args.cmin, args.cmax))]

    # -------------to actually plot the simulation data------------
    x, y = abslist[args.xcolname], abslist[args.ycolname]
    axes.ax_joint.hexbin(x, y, alpha=0.7, cmap=cmap, gridsize=(40, 40))

    axes.ax_marg_x = plot_1D_histogram(x, args.xmin, args.xmax, axes.ax_marg_x, vertical=False, type='absorbers')
    axes.ax_marg_y = plot_1D_histogram(y, args.ymin, args.ymax, axes.ax_marg_y, vertical=True, type='absorbers')

    print_mpi('Overplotted ' + str(len(abslist)) + ' of ' + str(init_len) + ', i.e., ' + '%.2F' % (len(abslist) * 100 / init_len) + '% of absorbers inside this box..', args)

    return axes

# -----main code-----------------
if __name__ == '__main__':
    args_tuple = parse_args('8508', 'RD0042')  # default simulation to work upon when comand line args not provided
    if type(args_tuple) is tuple: args = args_tuple[0] # if the sim has already been loaded in, in order to compute the box center (via utils.pull_halo_center()), then no need to do it again
    else: args = args_tuple
    if args.xcol == 'radius': args.xcol == 'rad'
    if not args.keep: plt.close('all')

    if args.do_all_halos: halos = get_all_halos(args)
    else: halos = args.halo_arr
    list_of_sims = list(itertools.product(halos, args.output_arr))
    total_snaps = len(list_of_sims)

    if args.fullbox:
        z_boxrad_dict = {'RD0030':84.64, 'RD0020': 47.96, 'RD0018': 41.11, 'RD0016': 35.97}
        args.galrad = np.max([z_boxrad_dict[output] for output in args.output_arr])
        galrad_text = 'refbox'
    else:
        galrad_text = 'boxrad_%.2Fkpc' % args.galrad

    # parse column names, in case log
    args.xcolname = 'log_' + args.xcol if islog_dict[args.xcol] and not args.use_cvs_log else args.xcol
    args.ycolname = 'log_' + args.ycol if islog_dict[args.ycol] and not args.use_cvs_log else args.ycol
    if isfield_weighted_dict[args.xcol] and args.weight: args.xcolname += '_wtby_' + args.weight
    if isfield_weighted_dict[args.ycol] and args.weight: args.ycolname += '_wtby_' + args.weight
    args.colorcol = args.colorcol[0]
    args.colorcolname = 'log_' + args.colorcol if islog_dict[args.colorcol] else args.colorcol
    if isfield_weighted_dict[args.colorcol] and args.weight: args.colorcolname += '_wtby_' + args.weight
    args.colorcol_cat = 'cat_' + args.colorcolname

    # ----------to determine axes limits--------------
    bounds_dict.update(rad=(0, args.galrad))
    if args.xmin is None:
        args.xmin = np.log10(bounds_dict[args.xcol][0]) if islog_dict[args.xcol] and not args.use_cvs_log else bounds_dict[args.xcol][0]
    if args.xmax is None:
        args.xmax = np.log10(bounds_dict[args.xcol][1]) if islog_dict[args.xcol] and not args.use_cvs_log else bounds_dict[args.xcol][1]
    if args.ymin is None:
        args.ymin = np.log10(bounds_dict[args.ycol][0]) if islog_dict[args.ycol] and not args.use_cvs_log else bounds_dict[args.ycol][0]
    if args.ymax is None:
        args.ymax = np.log10(bounds_dict[args.ycol][1]) if islog_dict[args.ycol] and not args.use_cvs_log else bounds_dict[args.ycol][1]
    if args.cmin is None:
        args.cmin = np.log10(bounds_dict[args.colorcol][0]) if islog_dict[args.colorcol] else bounds_dict[args.colorcol][0]
    if args.cmax is None:
        args.cmax = np.log10(bounds_dict[args.colorcol][1]) if islog_dict[args.colorcol] else bounds_dict[args.colorcol][1]

    # ----------to determine colorbar parameters--------------
    if args.cmap is None:
        args.cmap = colormap_dict[args.colorcol]
    else:
        args.cmap = plt.get_cmap(args.cmap)
    color_list = args.cmap.colors
    ncolbins = args.ncolbins if args.ncolbins is not None else len(color_list)
    args.color_list = color_list[::int(len(color_list) / ncolbins)]  # truncating color_list in to a length of rougly ncolbins
    args.ncolbins = len(args.color_list)
    color_key = get_color_keys(args.cmin, args.cmax, args.color_list)

    # parse paths and filenames
    output_dir = args.output_dir
    fig_dir = output_dir + 'figs/'
    Path(fig_dir).mkdir(parents=True, exist_ok=True)

    halos_text = 'all' if args.do_all_halos else ','.join(args.halo_arr)
    outputs_text = ','.join(args.output_arr)

    args.current_redshift, args.current_time = None, None

    # ----------collating the different dataframes (correpsonding to each snapshot)-------------------------------------
    df_merged = pd.DataFrame()
    abslist = load_absorbers_file(args)

    for index2, this_sim in enumerate(list_of_sims):
        start_time_this_snapshot = time.time()
        halo, output = this_sim[0], this_sim[1]
        print_mpi('Reading dataframe ' + output + ' of halo ' + halo + ' which is ' + str(index2 + 1) + ' out of the total ' + str(total_snaps) + ' snapshots...', args)

        thisboxrad = z_boxrad_dict[output] if args.fullbox else args.galrad
        file = output_dir.replace(args.halo, halo) + 'txtfiles/' + output + '_df_boxrad_%.2Fkpc.txt' % (thisboxrad)

        df = pd.read_table(file, delim_whitespace=True, comment='#')
        df = extract_columns_from_df(df, args)
        df_merged = df_merged.append(df)

        myprint('This snapshot ' + output + ' completed in %s minutes' % ((time.time() - start_time_this_snapshot) / 60), args)

    # -----------process merged dataframe and creating sub dataframe for different panels-------------------------------------------
    df_merged = df_merged[(df_merged[args.xcolname].between(args.xmin, args.xmax)) & (df_merged[args.ycolname].between(args.ymin, args.ymax)) & (
    df_merged[args.colorcolname].between(args.cmin, args.cmax))]
    df_merged[args.colorcol_cat] = categorize_by_quant(df_merged[args.colorcolname], args.cmin, args.cmax, args.ncolbins)
    df_merged[args.colorcol_cat] = df_merged[args.colorcol_cat].astype('category')

    df_inflow = df_merged[df_merged['vrad'] < 0.]
    df_outflow = df_merged[df_merged['vrad'] > 0.]
    abslist_inflow = abslist[abslist['vrad'] < 0.]
    abslist_outflow = abslist[abslist['vrad'] > 0.]

    df_arr = [df_merged[:0], df_outflow, df_merged[:0], df_inflow]
    abs_arr = [abslist_outflow, pd.DataFrame(), abslist_inflow, pd.DataFrame()]

    # -----------creating four-panel datashader plot with merged dataframe-------------------------------------------
    nticks, nrow, ncol = 4, 2, 2
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(nrow, ncol, figure=fig, hspace=0.05, wspace=0.05, right=0.95, top=0.9, bottom=0.15, left=0.15)

    for index in range(nrow * ncol):
        print_mpi('Doing panel ' + str(index + 1) + ' out of ' + str(nrow * ncol) + '..', args)
        df, abs = df_arr[index], abs_arr[index]

        axes = sns.JointGrid(args.xcolname, args.ycolname, df, height=8)
        ax, margx, margy = axes.ax_joint, axes.ax_marg_x, axes.ax_marg_y
        dummy = SeabornFig2Grid(axes, fig, gs[index]) # move this entire set of jointplot axes to the index-th panel of the gridspec figure

        artist = dsshow(df, dsh.Point(args.xcolname, args.ycolname), dsh.mean(args.colorcolname), norm='linear', cmap=list(color_key.values()), x_range=(args.xmin, args.xmax), y_range=(args.ymin, args.ymax), vmin=args.cmin, vmax=args.cmax, aspect='auto', ax=ax)  # , shade_hook=partial(dstf.spread, px=1, shape='square')) # the 40 in alpha_range and `square` in shade_hook are to reproduce original-looking plots as if made with make_datashader_plot()

        if len(abs) > 0:
            if index == 0: overplotted, axes = overplot_stars(abs, axes, args, type='absorbers')
            elif index == 2: axes = hexplot_abs(abs, axes, args, 'Blues')

        if len(df) > 0:
            ax = overplot_binned(df, ax, args)
            margx = plot_1D_histogram(df[args.xcolname], args.xmin, args.xmax, margx, vertical=False)
            margy = plot_1D_histogram(df[args.ycolname], args.ymin, args.ymax, margy, vertical=True)

        xticks = np.linspace(args.xmin, args.xmax, nticks)
        yticks = np.linspace(args.ymin, args.ymax, nticks)
        ax.xaxis.set_ticks(xticks)
        ax.yaxis.set_ticks(yticks)
        ax.xaxis.set_label_text('')
        ax.yaxis.set_label_text('')

        if index % ncol == 0: ax.yaxis.set_ticklabels(['%.1F' % item for item in yticks], fontsize=args.fontsize)
        else: ax.yaxis.set_ticklabels([])
        if int(index / ncol) + 1 == nrow: ax.xaxis.set_ticklabels(['%.1F' % item for item in xticks], fontsize=args.fontsize)
        else: ax.xaxis.set_ticklabels([])

    # ---------to annotate colorbar----------------------
    cax_xpos, cax_ypos, cax_width, cax_height = 0.7, 0.93, 0.25, 0.035
    cax = fig.add_axes([cax_xpos, cax_ypos, cax_width, cax_height])
    plt.colorbar(artist, cax=cax, orientation='horizontal')

    cax.set_xticklabels(['%.0F' % index for index in cax.get_xticks()], fontsize=args.fontsize / 1.2)  # , weight='bold')

    log_text = 'Log ' if islog_dict[args.colorcol] else ''
    fig.text(cax_xpos + cax_width / 2, cax_ypos + cax_height + 0.005, log_text + labels_dict[args.colorcol] + ' (' + unit_dict[args.colorcol] + ')', fontsize=args.fontsize/1.2, ha='center', va='bottom')

    # ---------to annotate and save the figure----------------------
    log_text = 'Log ' if islog_dict[args.xcol] else ''
    fig.text(0.55, 0.05, log_text + labels_dict[args.xcol] + ' (' + unit_dict[args.xcol] + ')', fontsize=args.fontsize, ha='center')

    log_text = 'Log ' if islog_dict[args.ycol] else ''
    fig.text(0.05, 0.52, log_text + labels_dict[args.ycol] + ' (' + unit_dict[args.ycol] + ')', fontsize=args.fontsize, ha='center', rotation='vertical')

    filename = fig_dir + 'kodiaqz_merged_dsh_abs_%s_%s_vs_%s_colby_%s_halos_%s_outputs_%s.png' % (galrad_text, args.ycolname, args.xcolname, args.colorcolname, halos_text, outputs_text)
    plt.savefig(filename, transparent=False)
    myprint('Saved figure ' + filename, args)
    fig.show()

    myprint('Serially: time taken for merging and datashading ' + str(total_snaps) + ' snapshots was %s mins' % ((time.time() - start_time) / 60), args)

