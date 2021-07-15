#!/usr/bin/env python3

"""

    Title :      datashader_movie
    Notes :      Attempt to make datashader plots of 3 given quantities from FOGGIE outputs and make movies
    Output :     datashader plots as png files (which can be later converted to a movie via animate_png.py)
    Author :     Ayan Acharyya
    Started :    July 2021
    Examples :   run datashader_movie.py --system ayan_hd --halo 8508 --galrad 20 --xcol rad --ycol metal --colorcol vrad --do_all_sims --makemovie --delay 0.2

"""
from header import *
from util import *
start_time = time.time()

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
            arr = ds[field_dict[field]].in_units(unit_dict[field]).ndarray_view()
            if islog_dict[field]:
                arr = np.log10(arr)
                field = 'log_' + field
            df[field] = arr

        df.to_csv(outfilename, sep='\t', mode='a', index=None)
    else:
        myprint('Reading from existing file ' + outfilename, args)
        df = pd.read_table(outfilename, delim_whitespace=True, comment='#')

    return df

# -----------------------------------------------------------------------------------
def wrap_axes(df, filename, args):
    '''
    Function to read in raw datashader plot and wrap it in axes using matplotlib AND added x- and y- marginalised histograms using seaborn
    This function is partly based on foggie.render.shade_maps.wrap_axes()
    :return: fig
    '''
    npix_datashader = 1000
    # -----------------to initialise figure---------------------
    axes = sns.JointGrid(xcol, ycol, df, height=8)
    plt.subplots_adjust(hspace=0.05, wspace=0.05, right=0.95, top=0.95, bottom=0.1, left=0.1)
    fig, ax1 = plt.gcf(), axes.ax_joint

    # ------to plot datashader image from file-------------
    img = mpimg.imread(filename)
    ax1.imshow(np.flip(img, 0))
    ax1.set_aspect('auto')

    # ----------to determine axes limits--------------
    x_min, x_max = bounds_dict[args.xcol]
    if islog_dict[args.xcol]: x_min, x_max = np.log10(x_min), np.log10(x_max)
    y_min, y_max = bounds_dict[args.ycol]
    if islog_dict[args.ycol]: y_min, y_max = np.log10(y_min), np.log10(y_max)

    # ----------to filter and bin the dataframe--------------
    df = df[(df[xcol].between(x_min, x_max)) & (df[ycol].between(y_min, y_max))]
    x_bin_size = bin_size_dict[args.xcol]
    x_bins = np.arange(x_min, x_max + x_bin_size, x_bin_size)
    df['binned_cat'] = pd.cut(df[xcol], x_bins)
    if islog_dict[args.ycol]: df[args.ycol] = 10 ** df[ycol] # because otherwise args.ycol is not going to be present in df
    y_binned = df.groupby('binned_cat', as_index=False).agg(np.mean)[args.ycol] # so that the averaging is done in linear space (as opposed to log space, which would be incorrect)
    if islog_dict[args.ycol]: y_binned = np.log10(y_binned)

    # ----------to plot mean binned y vs x profile--------------
    x_bin_centers = x_bins[:-1] + x_bin_size / 2
    x_on_plot = (x_bin_centers - x_min) * npix_datashader / (x_max - x_min) # because we need to stretch the binned x and y into npix_datashader dimensions determined by the datashader plot
    y_on_plot = (y_binned - y_min) * npix_datashader / (y_max - y_min)
    ax1.plot(x_on_plot, y_on_plot, color='black', lw=1)

    # ----------to plot 1D histogram on the top axes--------------
    sns.kdeplot(df[xcol], ax=axes.ax_marg_x, legend=False, color='black', lw=1)
    axes.ax_marg_x.tick_params(axis='x', which='both', top=False)
    axes.ax_marg_x.set_xlim(x_min, x_max)

    # ----------to plot 1D histogram on the right axes--------------
    sns.kdeplot(df[ycol], ax=axes.ax_marg_y, vertical=True, legend=False, color='black', lw=1)
    axes.ax_marg_y.tick_params(axis='y', which='both', right=False)
    axes.ax_marg_y.set_ylim(y_min, y_max)

    # ------to make the x axis-------------
    log_text = 'Log ' if islog_dict[args.xcol] else ''
    delta_x = 100 if x_max - x_min > 100 else 30 if x_max - x_min > 30 else 5 if x_max - x_min > 5 else 1
    ax1.set_xlabel(log_text + labels_dict[args.xcol] + ' (' + unit_dict[args.xcol] + ')', fontsize=args.fontsize)
    ax1.set_xticks(np.arange((x_max - x_min) + 1., step=delta_x) * npix_datashader / (x_max - x_min))
    ax1.set_xticklabels(['%.0F' % index for index in np.arange(x_min, x_max + 1, delta_x)], fontsize=args.fontsize)

    # ------to make the y axis-------------
    log_text = 'Log ' if islog_dict[args.ycol] else ''
    delta_y = 100 if y_max - y_min > 100 else 30 if y_max - y_min > 30 else 5 if y_max - y_min > 5 else 1
    ax1.set_ylabel(log_text + labels_dict[args.ycol] + ' (' + unit_dict[args.ycol] + ')', fontsize=args.fontsize)
    ax1.set_yticks(np.arange((y_max - y_min) + 1., step=delta_y) * npix_datashader / (y_max - y_min))
    ax1.set_yticklabels(['%.0F' % index for index in np.arange(y_min, y_max + 1, delta_y)], fontsize=args.fontsize)

    # ------to make the small colorbar axis on top right-------------
    n = 100000
    dummy_df = pd.DataFrame({'x':np.random.rand(n), 'y': np.random.rand(n)})
    c_min, c_max = bounds_dict[args.colorcol]
    if islog_dict[args.colorcol]:
        arr = [random.uniform(np.log10(c_min), np.log10(c_max)) for i in range(n)]
        dummy_df[args.colorcol] = 10. ** arr
    else:
        dummy_df[args.colorcol] = [random.uniform(c_min, c_max) for i in range(n)]
    colorcol_cat = 'cat_' + args.colorcol
    dummy_df[colorcol_cat] = categorize_by_dict[args.colorcol](dummy_df[args.colorcol])
    dummy_df[colorcol_cat] = dummy_df[colorcol_cat].astype('category')
    color_field_cmap = grab_cmap(dummy_df, 'x', 'y', colorcol_cat, colorkey_dict[args.colorcol])

    ax2 = fig.add_axes([0.7, 0.82, 0.25, 0.06])
    cbar_im = np.flip(color_field_cmap.to_pil(), 1)
    ax2.imshow(cbar_im)
    log_text = 'Log ' if islog_dict[args.colorcol] else ''
    delta_c = 200 if c_max - c_min > 200 else 60 if c_max - c_min > 60 else 10 if c_max - c_min > 10 else 2
    ax2.set_xticks(np.arange((c_max - c_min) + 1., step=delta_c) * np.shape(cbar_im)[1] / (c_max - c_min))
    ax2.set_xticklabels(['%.0F' % index for index in np.arange(c_min, c_max + 1, delta_c)], fontsize=args.fontsize/1.5)#, weight='bold')
    ax2.text(np.shape(cbar_im)[1]/2, 2.5 * np.shape(cbar_im)[0], log_text + labels_dict[args.colorcol] + ' (' + unit_dict[args.colorcol] + ')', fontsize=args.fontsize/1.5, ha='center', va='top')

    for item in ['top', 'bottom', 'left', 'right']: ax2.spines[item].set_color('white')
    ax2.set_yticklabels([])
    ax2.set_yticks([])

    # ---------to annotate and save the figure----------------------
    plt.text(0.033, 0.05, 'z = %.4F' % args.current_redshift , transform=ax1.transAxes, fontsize=args.fontsize)
    plt.savefig(filename, transparent=False)
    myprint('Saved figure ' + filename, args)
    if not args.makemovie: plt.show(block=False)

    return fig

# --------------------------------------------------------------------------------
def make_datashader_plot(ds, outfilename, args):
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

    cvs = dsh.Canvas(plot_width=1000, plot_height=1000, x_range=x_range, y_range=y_range)
    agg = cvs.points(df, xcol, ycol, dsh.count_cat(colorcol_cat))
    img = dstf.spread(dstf.shade(agg, color_key=colorkey_dict[args.colorcol], how='eq_hist', min_alpha=40), shape='square')
    export_image(img, os.path.splitext(outfilename)[0])

    fig = wrap_axes(df, os.path.splitext(outfilename)[0] + '.png', args)

    return df, fig

# -----main code-----------------
if __name__ == '__main__':
    # set variables and dictionaries
    field_dict = {'rad':('gas', 'radius_corrected'), 'density':('gas', 'density'), 'gas_entropy':('gas', 'entropy'), 'stars':('deposit', 'stars_density'), 'metal':('gas', 'metallicity'), 'temp':('gas', 'temperature'), 'dm':('deposit', 'dm_density'), 'vrad':('gas', 'radial_velocity_corrected')}
    unit_dict = {'rad':'kpc', 'density':'Msun/pc**2', 'metal':r'Zsun', 'temp':'K', 'vrad':'km/s', 'ys_age':'Myr', 'ys_mas':'pc*Msun', 'gas_entropy':'keV*cm**3', 'vlos':'km/s'}
    labels_dict = {'rad':'Radius', 'density':'Density', 'metal':'Metallicity', 'temp':'Temperature', 'vrad':'Radial velocity', 'ys_age':'Age', 'ys_mas':'Mass', 'gas_entropy':'Entropy', 'vlos':'LoS velocity'}
    islog_dict = defaultdict(lambda: False, metal=True, density=True, temp=True, gas_entropy=True)
    bin_size_dict = defaultdict(lambda: 1.0, metal=0.1, density=2, temp=1, rad=0.1, vrad=50)
    categorize_by_dict = {'temp':categorize_by_temp, 'metal':categorize_by_metals, 'density':categorize_by_den, 'vrad':categorize_by_outflow_inflow, 'rad':categorize_by_radius}
    colorkey_dict = {'temp':new_phase_color_key, 'metal':new_metals_color_key, 'density': density_color_key, 'vrad': outflow_inflow_color_key, 'rad': radius_color_key}

    dummy_args_tuple = parse_args('8508', 'RD0042')  # default simulation to work upon when comand line args not provided
    if type(dummy_args_tuple) is tuple: dummy_args = dummy_args_tuple[0] # if the sim has already been loaded in, in order to compute the box center (via utils.pull_halo_center()), then no need to do it again
    else: dummy_args = dummy_args_tuple
    if dummy_args.xcol == 'radius': dummy_args.xcol == 'rad'
    if not dummy_args.keep: plt.close('all')

    if dummy_args.do_all_sims: list_of_sims = get_all_sims_for_this_halo(dummy_args) # all snapshots of this particular halo
    else: list_of_sims = [(dummy_args.halo, dummy_args.output)]
    total_snaps = len(list_of_sims)

    # parse column names, in case log
    xcol = 'log_' + dummy_args.xcol if islog_dict[dummy_args.xcol] else dummy_args.xcol
    ycol = 'log_' + dummy_args.ycol if islog_dict[dummy_args.ycol] else dummy_args.ycol
    colorcol = 'log_' + dummy_args.colorcol if islog_dict[dummy_args.colorcol] else dummy_args.colorcol
    colorcol_cat = 'cat_' + colorcol

    # parse paths and filenames
    fig_dir = dummy_args.output_dir + 'figs/' if dummy_args.do_all_sims else dummy_args.output_dir + 'figs/' + dummy_args.output + '/'
    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    outfile_rootname = 'datashader_boxrad_%.2Fkpc_%s_vs_%s_colby_%s.png' % (dummy_args.galrad, ycol, xcol, colorcol)
    if dummy_args.do_all_sims: outfile_rootname = 'z=*_' + outfile_rootname

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
    print_mpi('Operating on snapshots ' + str(core_start + 1) + ' to ' + str(core_end + 1) + 'i.e., ' + str(core_end - core_start + 1) + ' out of ' + str(total_snaps) + ' snapshots', dummy_args)

    for index in range(core_start, core_end + 1):
        start_time_this_snapshot = time.time()
        this_sim = list_of_sims[index]
        print_mpi('Doing snapshot ' + this_sim[1] + ' of halo ' + this_sim[0] + ' which is ' + str(index + 1) + ' out of the total ' + str(core_end - core_start + 1) + ' snapshots...', dummy_args)
        try:
            if dummy_args.do_all_sims: args = parse_args(this_sim[0], this_sim[1])
            else: args = dummy_args_tuple # since parse_args() has already been called and evaluated once, no need to repeat it

            if type(args) is tuple:
                args, ds, refine_box = args  # if the sim has already been loaded in, in order to compute the box center (via utils.pull_halo_center()), then no need to do it again
                print_mpi('ds ' + str(ds) + ' for halo ' + str(this_sim[0]) + ' was already loaded at some point by utils; using that loaded ds henceforth', args)
            else:
                ds, refine_box = load_sim(args, region='refine_box', do_filter_particles=False)
        except (FileNotFoundError, PermissionError) as e:
            print_mpi('Skipping ' + this_sim[1] + ' because ' + str(e), dummy_args)
            continue

        args.current_redshift = ds.current_redshift
        thisfilename = fig_dir + outfile_rootname.replace('*', '%.5F' % (args.current_redshift))

        if not os.path.exists(thisfilename) or args.clobber_plot:
            if not os.path.exists(thisfilename):
                print_mpi(thisfilename + ' plot does not exist. Creating afresh..', args)
            elif args.clobber_plot:
                print_mpi(thisfilename + ' plot exists but over-writing..', args)

            if args.fullbox:
                box_width = ds.refine_width  # kpc
                args.galrad = box.width / 2
            else:
                box_center = ds.arr(args.halo_center, kpc)
                box_width = args.galrad * 2 # in kpc
                box_width_kpc = ds.arr(box_width, 'kpc')
                box = ds.r[box_center[0] - box_width_kpc / 2.: box_center[0] + box_width_kpc / 2., box_center[1] - box_width_kpc / 2.: box_center[1] + box_width_kpc / 2., box_center[2] - box_width_kpc / 2.: box_center[2] + box_width_kpc / 2., ]

            bounds_dict = defaultdict(lambda: None, rad=(0, args.galrad), gas=(1e-29, 1e-22), temp=(2e3, 4e6), metal=(1e-2, 1e1), vrad=(-400, 400))  # in g/cc, range within box; hard-coded for Blizzard RD0038; but should be broadly applicable to other snaps too
            df, fig = make_datashader_plot(box, thisfilename, args)
        else:
            print_mpi('Skipping snapshot because plot already exists (use --clobber_plot to over-write) at ' + thisfilename, args)

        print_mpi('This snapshot ' + this_sim[1] + ' completed in %s minutes' % ((time.time() - start_time_this_snapshot) / 60), args)

    comm.Barrier() # wait till all cores reached here and then resume

    if args.makemovie and args.do_all_sims:
        print_master('Finished creating snapshots, calling animate_png.py to create movie..', args)
        subprocess.call(['python ' + HOME + '/Work/astro/ayan_codes/animate_png.py --inpath ' + fig_dir + ' --rootname ' + outfile_rootname + ' --delay ' + str(args.delay_frame) + ' --reverse'], shell=True)

    if ncores > 1: print_master('Parallely: time taken for filtering ' + str(total_snaps) + ' snapshots with ' + str(ncores) + ' cores was %s mins' % ((time.time() - start_time) / 60), dummy_args)
    else: print_master('Serially: time taken for filtering ' + str(total_snaps) + ' snapshots with ' + str(ncores) + ' core was %s mins' % ((time.time() - start_time) / 60), dummy_args)
