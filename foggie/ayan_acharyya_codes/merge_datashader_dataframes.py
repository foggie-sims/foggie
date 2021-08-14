#!/usr/bin/env python3

"""

    Title :      merge_datashader_dataframes
    Notes :      Merge existing dataframes (potentially of different snapshots/halos) into a single datashader plot of 3 given quantities
                 This script assumes that all the relevant dataframes already exist. If a particular dataframe doesn't exist, it simply skips over that snapshot
    Output :     a single datashader plot as png file
    Author :     Ayan Acharyya
    Started :    August 2021
    Examples :   run merge_datashader_dataframes.py --system ayan_pleiades --do_all_halos --fullbox --xcol rad --ycol metal --colorcol vrad --output RD0020,RD0018,RD0016
                 run merge_datashader_dataframes.py --system ayan_local --halo 8508,4123 --fullbox --xcol rad --ycol metal --colorcol vrad --output RD0030
"""
from header import *
from util import *
from datashader_movie import *

start_time = time.time()

# -----main code-----------------
if __name__ == '__main__':
    args_tuple = parse_args('8508', 'RD0042')  # default simulation to work upon when comand line args not provided
    if type(args_tuple) is tuple: args = args_tuple[0] # if the sim has already been loaded in, in order to compute the box center (via utils.pull_halo_center()), then no need to do it again
    else: args = args_tuple
    if args.xcol == 'radius': args.xcol == 'rad'
    if not args.keep: plt.close('all')

    if args.do_all_sims:
        list_of_sims = get_all_sims(args) # all snapshots of this particular halo
    else:
        if args.do_all_halos: halos = get_all_halos(args)
        else: halos = args.halo_arr
        list_of_sims = list(itertools.product(halos, args.output_arr))
    total_snaps = len(list_of_sims)

    if args.fullbox:
        z_boxrad_dict = {'RD0030':84.64, 'RD0020': 47.96, 'RD0018': 41.11, 'RD0016': 35.97}
        args.galrad = np.max(list(z_boxrad_dict.values()))
        galrad_text = 'refbox'
    else:
        galrad_text = '_boxrad_%.2Fkpc' % args.galrad

    # parse column names, in case log
    args.xcolname = 'log_' + args.xcol if islog_dict[args.xcol] and not args.use_cvs_log else args.xcol
    args.ycolname = 'log_' + args.ycol if islog_dict[args.ycol] and not args.use_cvs_log else args.ycol
    if isfield_weighted_dict[args.xcol] and args.weight: args.xcolname += '_wtby_' + args.weight
    if isfield_weighted_dict[args.ycol] and args.weight: args.ycolname += '_wtby_' + args.weight

    # parse paths and filenames
    output_dir = args.output_dir
    fig_dir = output_dir + 'figs/'
    Path(fig_dir).mkdir(parents=True, exist_ok=True)

    args.current_redshift, args.current_time = None, None
    colorcol_arr = args.colorcol

    halos_text = 'all' if args.do_all_halos else ','.join(args.halo_arr)
    outputs_text = 'all' if args.do_all_sims else ','.join(args.output_arr)
    inflow_outflow_text = '_inflow_only' if args.inflow_only else '_outflow_only' if args.outflow_only else ''

    # ----------collating the different dataframes (correpsonding to each snapshot)-------------------------------------
    for index, thiscolorcol in enumerate(colorcol_arr):
        args.colorcol = thiscolorcol
        args.colorcolname = 'log_' + args.colorcol if islog_dict[args.colorcol] else args.colorcol
        if isfield_weighted_dict[args.colorcol] and args.weight: args.colorcolname += '_wtby_' + args.weight
        args.colorcol_cat = 'cat_' + args.colorcolname
        print_mpi('Plotting ' + args.xcolname + ' vs ' + args.ycolname + ', color coded by ' + args.colorcolname + ' i.e., plot ' + str(index + 1) + ' of ' + str(len(colorcol_arr)) + '..', args)

        thisfilename = fig_dir + 'merged_datashader_%s_%s_vs_%s_colby_%s_halos_%s_outputs_%s%s.png' % (galrad_text, args.ycolname, args.xcolname, args.colorcolname, halos_text, outputs_text, inflow_outflow_text)

        df_merged, paramlist_merged = pd.DataFrame(), pd.DataFrame()
        for index2, this_sim in enumerate(list_of_sims):
            start_time_this_snapshot = time.time()
            halo, output = this_sim[0], this_sim[1]
            print_mpi('Reading dataframe ' + output + ' of halo ' + halo + ' which is ' + str(index2 + 1) + ' out of the total ' + str(total_snaps) + ' snapshots...', args)

            thisboxrad = z_boxrad_dict[output] if args.fullbox else args.galrad
            file = output_dir.replace(args.halo, halo) + 'txtfiles/' + output + '_df_boxrad_' + str(thisboxrad) + 'kpc.txt'

            if os.path.exists(file):
                df = pd.read_table(file, delim_whitespace=True, comment='#')
            else:
                myprint('Cannot find ' + file + '; skipping halo ' + halo + ' snapshot ' + output + '..' , args)
                continue

            df = extract_columns_from_df(df, args)
            df_merged = df_merged.append(df)

            if args.overplot_stars:
                args.output_dir = output_dir.replace(args.halo, halo)
                args.output = output
                paramlist = get_stars_file(args)
                paramlist_merged.append(paramlist)

            myprint('This snapshot ' + output + ' completed in %s minutes' % ((time.time() - start_time_this_snapshot) / 60), args)

        # -----------plotting one datashader plot with the giant merged dataframe-------------------------------------------
        if len(df_merged) > 0:
            if not os.path.exists(thisfilename) or args.clobber_plot:
                if not os.path.exists(thisfilename):
                    print_mpi(thisfilename + ' plot does not exist. Creating afresh..', args)
                elif args.clobber_plot:
                    print_mpi(thisfilename + ' plot exists but over-writing..', args)

                df_merged, fig = make_datashader_plot(df_merged, thisfilename, args, npix_datashader=1000, paramlist=paramlist_merged)
            else:
                myprint('Skipping colorcol ' + thiscolorcol + ' because plot already exists (use --clobber_plot to over-write) at ' + thisfilename, args)
        else:
            myprint('Merged dataframe is empty, skipping plotting step.', args)

    myprint('Serially: time taken for merging and datashading ' + str(total_snaps) + ' snapshots was %s mins' % ((time.time() - start_time) / 60), args)

