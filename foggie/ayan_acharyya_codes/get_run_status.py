#!/usr/bin/env python3

"""

    Title :      get_run_status
    Notes :      Print number of txt and png files existing corresponding to a given halo
    Output :     STDOUT print
    Author :     Ayan Acharyya
    Started :    Nov 2023
    Examples :   run get_run_status.py --system ayan_pleiades --forpaper --halo 8508
"""
from util import *

# -----main code-----------------
if __name__ == '__main__':
    # ------------- arg parse -------------------------------
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='''kill me please''')

    parser.add_argument('--system', metavar='system', type=str, action='store', default='ayan_pleiades', help='Which system are you on? Default is ayan_pleiades')
    parser.add_argument('--pwd', dest='pwd', action='store_true', default=False, help='Just use the current working directory?, default is no')
    parser.add_argument('--foggie_dir', metavar='foggie_dir', type=str, action='store', default=None, help='Specify which directory the dataset lies in, otherwise, by default it will use the args.system variable to determine the FOGGIE data location')
    parser.add_argument('--halo', metavar='halo', type=str, action='store', default='8508', help='which halo?')
    parser.add_argument('--run', metavar='run', type=str, action='store', default='nref11c_nref9f', help='which run?')
    parser.add_argument('--output', metavar='output', type=str, action='store', default='RD0042', help='which output?')
    parser.add_argument('--upto_kpc', metavar='upto_kpc', type=float, action='store', default=10, help='fit metallicity gradient out to what absolute kpc? default is None')
    parser.add_argument('--res_kpc', metavar='res_kpc', type=str, action='store', default=0.7, help='spatial sampling resolution, in kpc, at redshift 0; default is 0.3 kpc')
    parser.add_argument('--res_arc', metavar='res_arc', type=float, action='store', default=None, help='spatial sampling resolution, in arcseconds, to compute the Z statistics; default is None')
    parser.add_argument('--weightby', metavar='weightby', type=str, action='store', default=None, help='gas quantity to weight by; default is None')
    parser.add_argument('--docomoving', dest='docomoving', action='store_true', default=False, help='consider the input upto_kpc as a comoving quantity?, default is no')
    parser.add_argument('--use_density_cut', dest='use_density_cut', action='store_true', default=False, help='impose a density cut to get just the disk?, default is no')
    parser.add_argument('--fit_multiple', dest='fit_multiple', action='store_true', default=False, help='fit one gaussian + one skewed guassian?, default is no')
    parser.add_argument('--islog', dest='islog', action='store_true', default=False, help='set x- and y- scale as log?, default is no')
    parser.add_argument('--weight', metavar='weight', type=str, action='store', default=None, help='column name of quantity to weight the metallicity by; default is None i.e. no weight')
    parser.add_argument('--use_onlyRD', dest='use_onlyRD', action='store_true', default=False, help='Use only the RD snapshots available for a given halo?, default is no')
    parser.add_argument('--use_onlyDD', dest='use_onlyDD', action='store_true', default=False, help='Use only the DD snapshots available for a given halo?, default is no')
    parser.add_argument('--forpaper', dest='forpaper', action='store_true', default=False, help='make plot with certain set panels, specifically for the paper?, default is no')
    parser.add_argument('--nevery', metavar='nevery', type=int, action='store', default=1, help='use every nth snapshot when do_all_sims is specified; default is 1 i.e., all snapshots will be used')
    parser.add_argument('--snap_start', metavar='snap_start', type=int, action='store', default=0, help='index of the DD or RD snapshots to start from, when using --do_all_sims; default is 0')
    parser.add_argument('--snap_stop', metavar='snap_stop', type=int, action='store', default=10000, help='index of the DD or RD snapshots to stop at, when using --do_all_sims; default is 10000')
    parser.add_argument('--plot_onlybinned', dest='plot_onlybinned', action='store_true', default=False, help='plot ONLY the binned plot, without individual pixels?, default is no')

    args = parser.parse_args()
    args.foggie_dir, args.output_dir, args.run_loc, args.code_path, args.trackname, args.haloname, args.spectra_dir, args.infofile = get_run_loc_etc(args)

    # --------assigning additional keyword args-------------
    if args.forpaper:
        args.use_density_cut = True
        args.docomoving = True
        args.fit_multiple = True  # for the Z distribution panel
        args.islog = True  # for the Z distribution panel
        args.weight = 'mass'
        args.upto_kpc = 10
        args.use_onlyDD = True

    # -------determining total snapshots----------------------------
    list_of_sims = get_all_sims_for_this_halo(args)
    n_total_snaps = len(list_of_sims)

    # -------determining txt file name pattern for each snapshot---------------
    if args.upto_kpc is not None: upto_text = '_upto%.1Fckpchinv' % args.upto_kpc if args.docomoving else '_upto%.1Fkpc' % args.upto_kpc
    else: upto_text = '_upto%.1FRe' % args.upto_re
    density_cut_text = '_wdencut' if args.use_density_cut else ''
    snap_txtfiles = args.output_dir + 'txtfiles/*_df_boxrad%s%s.txt' % (upto_text, density_cut_text)
    n_snap_txtfiles = len(glob.glob(snap_txtfiles))

    # -------determining profile png file name pattern for each snapshot---------------
    onlybinned_text = '_onlybinned' if args.plot_onlybinned else ''
    weightby_text = '' if args.weight is None else '_wtby_' + args.weight
    snap_prof_pngfiles = args.output_dir + 'figs/z=*_datashader_log_metal_vs_rad%s%s%s%s.png' % (upto_text, weightby_text, onlybinned_text, density_cut_text)
    n_snap_prof_pngfiles = len(glob.glob(snap_prof_pngfiles))

    # -------determining histogram png file name pattern for each snapshot---------------
    fitmultiple_text = '_fitmultiple' if args.fit_multiple else ''
    islog_text = '_islog' if args.islog else ''
    snap_hist_pngfiles = args.output_dir + 'figs/z=*_log_metal_distribution%s%s%s%s%s.png' % (upto_text, weightby_text, fitmultiple_text, density_cut_text, islog_text)
    n_snap_hist_pngfiles = len(glob.glob(snap_hist_pngfiles))

    # -------determining 3d vs projected png file name pattern for each snapshot---------------
    snap_3dvproj_pngfiles = args.output_dir + 'figs/z=*_%s_projected_vs_3d_Zgrad_den_kpc%s%s.png' % (args.halo, upto_text, weightby_text)
    n_snap_3dvproj_pngfiles = len(glob.glob(snap_3dvproj_pngfiles))

    # -------determining projected evolution png file name pattern for each snapshot---------------
    snap_projev_pngfiles = args.output_dir + 'figs/z=*_%s_projectedZ_Zgrad_den_kpc%s%s.png' % (args.halo, upto_text, weightby_text)
    n_snap_projev_pngfiles = len(glob.glob(snap_projev_pngfiles))

    # -------determining projected evolution png file name pattern for each snapshot---------------
    snap_nonprojev_pngfiles = args.output_dir + 'figs/z=*_%s_nonprojectedZ_Zgrad_den_kpc%s%s.png' % (args.halo, upto_text, weightby_text)
    n_snap_nonprojev_pngfiles = len(glob.glob(snap_nonprojev_pngfiles))

    # -------determining projected prof-hist-map png file name pattern for each snapshot---------------
    snap_projphm_pngfiles = args.output_dir + 'figs/z=*_%s_projectedZ_prof_hist_map*%s%s*.png' % (args.halo, upto_text, weightby_text)
    n_snap_projphm_pngfiles = len(glob.glob(snap_projphm_pngfiles))

    # -------determining non-projected prof-hist-map png file name pattern for each snapshot---------------
    snap_nonprojphm_pngfiles = args.output_dir + 'figs/z=*_%s_nonprojectedZ_prof_hist_map*%s%s.png' % (args.halo, upto_text, weightby_text)
    n_snap_nonprojphm_pngfiles = len(glob.glob(snap_nonprojphm_pngfiles))

    # -------determining combined MZR file name pattern for the halo---------------
    MZR_filename = args.output_dir + 'txtfiles/' + args.halo + '_MZR_xcol_rad%s%s%s.txt' % (upto_text, weightby_text, density_cut_text)
    if os.path.exists(MZR_filename):
        df = pd.read_table(MZR_filename)
        n_MZR_lines = len(df)
        n_unique_MZR_lines = len(pd.unique(df['output']))
        df = df.drop_duplicates(subset='output', keep='last', ignore_index=True)
        grad_col = 'Zgrad_binned_fixedr'
        n_usable_MZR_lines = len(df[~np.isnan(df[grad_col])])
        n_useless_MZR_lines = len(df[np.isnan(df[grad_col])])
    else:
        print('MZR file does not exist.')

    # -------determining combined MZscat file name pattern for the halo---------------
    MZscat_filename = args.output_dir + 'txtfiles/' + args.halo + '_MZscat%s%s%s%s%s.txt' % (upto_text, weightby_text, fitmultiple_text, density_cut_text, islog_text)
    if os.path.exists(MZscat_filename):
        df = pd.read_table(MZscat_filename)
        n_MZscat_lines = len(df)
        n_unique_MZscat_lines = len(pd.unique(df['output']))
        df = df.drop_duplicates(subset='output', keep='last', ignore_index=True)
        sigma_col = 'Zsigma' if 'Zsigma' in df else 'Zvar'
        n_usable_MZscat_lines = len(df[~np.isnan(df[sigma_col])])
        n_useless_MZscat_lines = len(df[np.isnan(df[sigma_col])])
    else:
        print('MZscat file does not exist.')

    # -------printing results---------------------------
    print('Halo %s. Total snaps %d. Txtfiles %d. Gradient pngs %d. Histogram pngs %d. MZR file has %d (%d usable). MZscat file has %d (%d usable). 3D vs projected plots %d. Projected evolution plots %d, Non-projected evolution plots %d, Projected prof-hist-map plots %d, Non-projected prof-hist-map plots %d.' %\
          (args.halo, n_total_snaps, n_snap_txtfiles, n_snap_prof_pngfiles, n_snap_hist_pngfiles, n_unique_MZR_lines, n_usable_MZR_lines, n_unique_MZscat_lines, n_usable_MZscat_lines, n_snap_3dvproj_pngfiles, n_snap_projev_pngfiles, n_snap_nonprojev_pngfiles, n_snap_projphm_pngfiles, n_snap_nonprojphm_pngfiles))