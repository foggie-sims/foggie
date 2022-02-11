#!/usr/bin/env python3

"""

    Title :      compute_MZgrad
    Notes :      Compute mass - metallicity gradient relation for a given FOGGIE galaxy
    Output :     Z gradient plots as png files (also txt file storing all the gradients & mass plus MZgrad plot if this code is run on multiple snapshots)
    Author :     Ayan Acharyya
    Started :    Feb 2022
    Examples :   run compute_MZgrad.py --system ayan_local --halo 8508 --output RD0030 --upto_re 2 --xcol rad_re --keep
                 run compute_MZgrad.py --system ayan_local --halo 8508 --upto_re 2 --xcol rad_re --do_all_sims

"""
from header import *
from util import *
from datashader_movie import *
from uncertainties import ufloat, unumpy
start_time = time.time()

# --------------------------------------------------------------------------------
def get_re(ds, args):
    '''
    Function to determine the effective radius of stellar disk, given a dataset
    Returns the effective radius in kpc
    '''
    #-----initially trimming the dataset approximately to a 30 kpc a side box----------
    box_center = ds.arr(args.halo_center, kpc)
    box_width_kpc = ds.arr(30, 'kpc')
    box = ds.r[box_center[0] - box_width_kpc / 2.: box_center[0] + box_width_kpc / 2., box_center[1] - box_width_kpc / 2.: box_center[1] + box_width_kpc / 2., box_center[2] - box_width_kpc / 2.: box_center[2] + box_width_kpc / 2., ]

    return 4.0 # temporary placeholder; in kpc

# --------------------------------------------------------------------------------------
def bin_data(array, data, bins):
    '''
    Function to bin data based on another array
    '''
    bins_cen = bins[:-1] + np.diff(bins) / 2.
    indices = np.digitize(array, bins)
    binned_data, binned_err = [], []
    for i in range(1, len(bins)):  # assuming all values in array are within limits of bin, hence last cell of binned_data=nan
        thisdata = data[indices == i]
        mean_data, mean_err = np.mean(thisdata), np.std(thisdata)
        binned_data.append(mean_data)
        binned_err.append(mean_err)
    binned_data = np.array(binned_data)
    binned_err = np.array(binned_err)
    return bins_cen, binned_data, binned_err

# ---------------------------------------------------------------------------------
def overplot_binned(df, xcol, ycol, x_bins, ax, is_logscale=False, color='maroon'):
    '''
    Function to overplot binned data on existing plot
    '''
    df['binned_cat'] = pd.cut(df[xcol], x_bins)
    y_binned = df.groupby('binned_cat', as_index=False).agg([(ycol, np.mean)])[ycol]
    y_u_binned = df.groupby('binned_cat', as_index=False).agg([(ycol, np.std)])[ycol]
    if is_logscale: y_binned, y_u_binned = np.log10(y_binned.values), np.log10(y_u_binned.values)

    # ----------to plot mean binned y vs x profile--------------
    x_bin_centers = x_bins[:-1] + np.diff(x_bins) / 2
    ax.errorbar(x_bin_centers, y_binned, c=color, yerr=y_u_binned, lw=1)
    ax.scatter(x_bin_centers, y_binned, c=color, s=60)

    linefit, linecov = np.polyfit(x_bin_centers, y_binned.flatten(), 1, cov=True)#, w=1/y_u_binned.flatten())
    ax.plot(x_bin_centers, np.poly1d(linefit)(x_bin_centers), color=color, lw=1, ls='dashed')
    units = 'dex/re' if 're' in xcol else 'dex/kpc'
    ax.text(0.033, 0.25, 'Slope = %.2F ' % linefit[0] + units, color=color, transform=ax.transAxes, fontsize=args.fontsize)

    return ax

# -----------------------------------------------------------
def plot_gradient(df, args, linefit=None):
    '''
    Function to plot the metallicity profile, along with the fitted gradient if provided
    Saves plot as .png
    '''
    outfile_rootname = 'datashader_log_metal_vs_%s_upto_%.1FRe.png' % (args.xcol, args.upto_re)
    if args.do_all_sims: outfile_rootname = 'z=*_' + outfile_rootname
    filename = args.fig_dir + outfile_rootname.replace('*', '%.5F' % (args.current_redshift))

    # ---------first, plot both cell-by-cell profile first, using datashader---------
    fig, ax = plt.subplots(figsize=(8,8))
    fig.subplots_adjust(hspace=0.05, wspace=0.05, right=0.95, top=0.95, bottom=0.1, left=0.1)
    artist = dsshow(df, dsh.Point(args.xcol, 'log_metal'), dsh.count(), norm='linear', x_range=(0, args.upto_re if 're' in args.xcol else args.galrad), y_range=(args.ylim[0], args.ylim[1]), aspect = 'auto', ax=ax, cmap='Blues_r')#, shade_hook=partial(dstf.spread, px=1, shape='square')) # the 40 in alpha_range and `square` in shade_hook are to reproduce original-looking plots as if made with make_datashader_plot()

    # --------bin the metallicity profile and plot the binned profile-----------
    if 'metal' not in df: df['metal'] = 10 ** (df['log_metal'])
    bin_edges = np.linspace(0, args.upto_re if 're' in args.xcol else args.galrad, 10)
    ax = overplot_binned(df, args.xcol, 'metal', bin_edges, ax, is_logscale=True)

    # ----------plot the fitted metallicity profile---------------
    if linefit is not None:
        fitted_y = np.poly1d(linefit)(bin_edges)
        ax.plot(bin_edges, fitted_y, color='darkblue', lw=2, ls='dashed')
        units = 'dex/re' if 're' in args.xcol else 'dex/kpc'
        plt.text(0.033, 0.2, 'Slope = %.2F ' % linefit[0] + units, color='darkblue', transform=ax.transAxes, fontsize=args.fontsize)

    # ----------tidy up figure-------------
    ax.xaxis = make_coordinate_axis(args.xcol, 0, args.upto_re if 're' in args.xcol else args.galrad, ax.xaxis, args.fontsize, dsh=False, log_scale=False)
    ax.yaxis = make_coordinate_axis('metal', args.ylim[0], args.ylim[1], ax.yaxis, args.fontsize, dsh=False, log_scale=False)

    # ---------annotate and save the figure----------------------
    plt.text(0.033, 0.05, 'z = %.4F' % args.current_redshift, transform=ax.transAxes, fontsize=args.fontsize)
    plt.text(0.033, 0.1, 't = %.3F Gyr' % args.current_time, transform=ax.transAxes, fontsize=args.fontsize)
    plt.savefig(filename, transparent=False)
    myprint('Saved figure ' + filename, args)
    if not args.makemovie: plt.show(block=False)

    return fig

# -------------------------------
def fit_gradient(df, args):
    '''
    Function to linearly fit the (log) metallicity profile out to certain Re, given a dataframe containing metallicity profile
    Returns the fitted gradient with uncertainty
    '''

    linefit, linecov = np.polyfit(df[args.xcol], df['log_metal'], 1, cov=True)
    Zgrad = ufloat(linefit[0], np.sqrt(linecov[0][0]))
    Zcen = ufloat(linefit[1], np.sqrt(linecov[1][1]))
    print('Inferred slope for halo ' + args.halo + ' output ' + args.output + ' is', Zgrad, 'dex/re' if 're' in args.xcol else 'dex/kpc')

    return Zcen, Zgrad

# -------------------------------------------------------------------------------
def get_df_from_ds(ds, args):
    '''
    Function to make a pandas dataframe from the yt dataset, including only the metallicity profile,
    then writes dataframe to file for faster access in future
    This function is somewhat based on foggie.utils.prep_dataframe.prep_dataframe()
    :return: dataframe
    '''
    # -------------read/write pandas df file with ALL fields-------------------
    Path(args.output_dir + 'txtfiles/').mkdir(parents=True, exist_ok=True)  # creating the directory structure, if doesn't exist already
    outfilename = get_correct_tablename(args)

    if not os.path.exists(outfilename) or args.clobber:
        myprint(outfilename + ' does not exist. Creating afresh..', args)

        df = pd.DataFrame()
        fields = ['rad', 'metal'] # only the relevant properties

        for index, field in enumerate(fields):
            myprint('Doing property: ' + field + ', which is ' + str(index + 1) + ' of the ' + str(len(fields)) + ' fields..', args)
            df[field] = ds[field_dict[field]].in_units(unit_dict[field]).ndarray_view()

        df.to_csv(outfilename, sep='\t', index=None)
    else:
        myprint('Reading from existing file ' + outfilename, args)
        df = pd.read_table(outfilename, delim_whitespace=True, comment='#')
        df = df[df['rad'].between(0, args.galrad)] # in case this dataframe has been read in from a file corresponding to a larger chunk of the box

    df['log_metal'] = np.log10(df['metal'])
    df['rad_re'] = df['rad'] / args.re  # normalise by Re
    df = df[[args.xcol, 'log_metal']] # only keeping the columns that are needed to get Z gradient

    return df

# -----main code-----------------
if __name__ == '__main__':
    dummy_args_tuple = parse_args('8508', 'RD0042')  # default simulation to work upon when comand line args not provided
    if type(dummy_args_tuple) is tuple: dummy_args = dummy_args_tuple[0] # if the sim has already been loaded in, in order to compute the box center (via utils.pull_halo_center()), then no need to do it again
    else: dummy_args = dummy_args_tuple
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
                ds, refine_box = load_sim(args, region='refine_box', do_filter_particles=True, disk_relative=False)
        except Exception as e:
            print_mpi('Skipping ' + this_sim[1] + ' because ' + str(e), dummy_args)
            continue

        # parse paths and filenames
        args.fig_dir = args.output_dir + 'figs/' if args.do_all_sims else args.output_dir + 'figs/' + args.output + '/'
        Path(args.fig_dir).mkdir(parents=True, exist_ok=True)

        # extract the required box
        args.re = get_re(ds, args)
        box_center = ds.arr(args.halo_center, kpc)
        args.galrad = args.upto_re * args.re # kpc
        box_width = args.galrad * 2  # in kpc
        box_width_kpc = ds.arr(box_width, 'kpc')
        box = ds.r[box_center[0] - box_width_kpc / 2.: box_center[0] + box_width_kpc / 2., box_center[1] - box_width_kpc / 2.: box_center[1] + box_width_kpc / 2., box_center[2] - box_width_kpc / 2.: box_center[2] + box_width_kpc / 2., ]

        df = get_df_from_ds(box, args) # get dataframe with metallicity profile info

        args.current_redshift = ds.current_redshift
        args.current_time = ds.current_time.in_units('Gyr')
        args.ylim = [-2.2, 1.2] # [-3, 1]

        Zcen, Zgrad = fit_gradient(df, args)
        if not args.noplot: fig = plot_gradient(df, args, linefit=[Zgrad.n, Zcen.n]) # plotting the Z profile, with fit
        print_master('This snapshots completed in %s mins' % ((time.time() - start_time_this_snapshot) / 60), dummy_args)

    if ncores > 1: print_master('Parallely: time taken for datashading ' + str(total_snaps) + ' snapshots with ' + str(ncores) + ' cores was %s mins' % ((time.time() - start_time) / 60), dummy_args)
    else: print_master('Serially: time taken for datashading ' + str(total_snaps) + ' snapshots with ' + str(ncores) + ' core was %s mins' % ((time.time() - start_time) / 60), dummy_args)
