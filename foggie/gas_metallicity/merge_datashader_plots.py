#!/usr/bin/env python3

"""

    Title :      merge_datashader_plots
    Notes :      Merge existing datashader plots (potentially of different snapshots/halos) into a single grid plot
                 This script assumes that all the relevant plots already exist. If a particular plot doesn't exist, it simply skips over that snapshot
    Output :     a single datashader plot as png file
    Author :     Ayan Acharyya
    Started :    August 2021
    Examples :   run merge_datashader_plots.py --system ayan_pleiades --do_all_halos --fullbox --xcol rad --ycol metal --colorcol vrad --output RD0016,RD0018,RD0020
                 run merge_datashader_plots.py --system ayan_local --do_all_halos --fullbox --xcol rad --ycol metal --colorcol vrad --output RD0016,RD0018,RD0020 --inflow_only
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

    if args.do_all_halos: halos = get_all_halos(args)
    else: halos = args.halo_arr

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

    crop_x1, crop_x2, crop_y1, crop_y2 = 75, -1, 0, -1
    nrow, ncol, figsize = len(args.output_arr), len(halos), (12, 6)
    total_snaps = len(halos) * len(args.output_arr)

    # ----------collating the different dataframes (correpsonding to each snapshot)-------------------------------------
    for index3, thiscolorcol in enumerate(colorcol_arr):
        args.colorcol = thiscolorcol
        args.colorcolname = 'log_' + args.colorcol if islog_dict[args.colorcol] else args.colorcol
        if isfield_weighted_dict[args.colorcol] and args.weight: args.colorcolname += '_wtby_' + args.weight
        args.colorcol_cat = 'cat_' + args.colorcolname
        print_mpi('Combining ' + args.xcolname + ' vs ' + args.ycolname + ', color coded by ' + args.colorcolname + ' i.e., plot ' + str(index3 + 1) + ' of ' + str(len(colorcol_arr)) + '..', args)

        fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
        fig.subplots_adjust(hspace=0, wspace=0, right=1, top=1, bottom=0, left=0)

        for index1, output in enumerate(args.output_arr):
            for index2, halo in enumerate(halos):
                start_time_this_snapshot = time.time()
                print_mpi('Reading plot ' + output + ' of halo ' + halo + ' which is ' + str(index1 * ncol + index2 + 1) + ' out of the total ' + str(total_snaps) + ' plots...', args)
                axes[index1][index2].axis('off')

                thisboxrad = z_boxrad_dict[output] if args.fullbox else args.galrad
                file = output_dir.replace(args.halo, halo) + 'figs/' + output + '/datashader_boxrad_%.2Fkpc_%s_vs_%s_colby_%s%s.png' % (thisboxrad, args.ycolname, args.xcolname, args.colorcolname, inflow_outflow_text)

                if not os.path.exists(file):
                    myprint('Cannot find ' + file + '; skipping halo ' + halo + ' snapshot ' + output + '..' , args)
                    continue

                image = mpimg.imread(file)
                axes[index1][index2].imshow(image[crop_y1 : crop_y2, crop_x1 if index2 else 0: crop_x2], origin='upper')
                axes[index1][index2].set_aspect('auto')

                myprint('This halo ' + halo + ', snapshot ' + output + ' completed in %s minutes' % ((time.time() - start_time_this_snapshot) / 60), args)

        thisfilename = fig_dir + 'combined_datashader_%s_%s_vs_%s_colby_%s_halos_%s_outputs_%s%s.png' % (galrad_text, args.ycolname, args.xcolname, args.colorcolname, halos_text, outputs_text, inflow_outflow_text)
        fig.savefig(thisfilename, dpi=500)
        myprint('Saved plot as ' + thisfilename, args)
        plt.show(block=False)

    myprint('Serially: time taken for combining ' + str(total_snaps) + ' plots was %s mins' % ((time.time() - start_time) / 60), args)

