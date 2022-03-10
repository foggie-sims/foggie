#!/usr/bin/env python3

"""

    Title :      plot_MZgrad
    Notes :      Plot mass - metallicity gradient relation for a given FOGGIE galaxy
    Output :     M-Z gradient plots as png files plus, optinally, MZR plot
    Author :     Ayan Acharyya
    Started :    Mar 2022
    Examples :   run plot_MZgrad.py --system ayan_local --halo 5036,8508 --upto_re 3 --xcol rad_re --keep --weight mass --ymin -0.5 --xmin 8.5 --overplot_obs --binby mass --nbins 200 --manga_diag pyqz
                 run plot_MZgrad.py --system ayan_pleiades --halo 8508 --upto_re 3 --xcol rad_re --weight mass --binby mass --nbins 20 --cmap plasma --xmin 8.5 --xmax 11 --ymin 0.3 --overplot_obs --manga_diag n2

"""
from header import *
from util import *
from matplotlib.collections import LineCollection
start_time = time.time()

# ---------------------------------
def load_df(args):
    '''
    Function to load and return the dataframe containing MZGR
    '''
    args.foggie_dir, args.output_dir, args.run_loc, args.code_path, args.trackname, args.haloname, args.spectra_dir, args.infofile = get_run_loc_etc(args)
    grad_filename = args.output_dir + 'txtfiles/' + args.halo + '_MZR_xcol_%s_upto%.1FRe%s.txt' % (args.xcol, args.upto_re, args.weightby_text)

    if os.path.exists(grad_filename):
        print('Trying to read in', grad_filename)
        df = pd.read_table(grad_filename, delim_whitespace=True)

    elif not os.path.exists(grad_filename) and args.xcol == 'rad_re':
        print('Could not find', grad_filename)
        grad_filename = grad_filename.replace('rad_re', 'rad')
        print('Trying to read in', grad_filename, 'instead')
        df = pd.read_table(grad_filename, delim_whitespace=True)
        print('Zgrad is in dex/kpc, will convert it to dex/re')
        df['Zgrad'] *= df['re']
        df['Zgrad_u'] *= df['re']

    elif not os.path.exists(grad_filename) and args.xcol == 'rad':
        print('Could not find', grad_filename)
        grad_filename = grad_filename.replace('rad', 'rad_re')
        print('Trying to read in', grad_filename, 'instead')
        df = pd.read_table(grad_filename, delim_whitespace=True)
        print('Zgrad is in dex/re, will convert it to dex/kpc')
        df['Zgrad'] /= df['re']
        df['Zgrad_u'] /= df['re']

    return df

# -----------------------------------
def overplot_manga(ax, args):
    '''
    Function to overplot the observed MZGR from MaNGA and return the axis handle
    '''
    # ---------read in the manga catalogue-------
    manga_input_filename = HOME + '/models/manga/manga.Pipe3D-v2_4_3_downloaded.fits'
    data = Table.read(manga_input_filename, format='fits')
    df_manga = data.to_pandas()

    # --------trim to required columns---------------
    # options for args.manga_diag are: n2, o3n2, ons, pyqz, t2, m08, t04
    df_manga = df_manga[['mangaid', 'redshift', 're_kpc', 'log_mass', 'alpha_oh_re_fit_' + args.manga_diag, 'e_alpha_oh_re_fit_' + args.manga_diag, 'oh_re_fit_' + args.manga_diag, 'e_oh_re_fit_' + args.manga_diag]]
    df_manga = df_manga.dropna(subset=['alpha_oh_re_fit_' + args.manga_diag])

    sns.kdeplot(df_manga['log_mass'], df_manga['alpha_oh_re_fit_' + args.manga_diag], ax=ax, shade=True, shade_lowest=False, alpha=0.7, n_levels=10, cmap='Greys')
    #ax.scatter(df_manga['log_mass'], df_manga['alpha_oh_re_fit_' + args.manga_diag], c=df_manga['redshift'], cmap='Greys_r', s=10, alpha=0.5)

    return ax, df_manga

# -----------------------------------
def plot_MZGR(args):
    '''
    Function to plot the mass-metallicity gradient relation, based on an input dataframe
    '''

    logbin = True if args.binby == 'mass' else False
    df_master = pd.DataFrame()
    cmap_arr = ['Purples_r', 'Oranges_r', 'Greens_r', 'Reds_r', 'Blues_r']

    # -------------get plot limits-----------------
    if args.xmin is None: args.xmin = 6
    if args.xmax is None: args.xmax = 11
    if args.ymin is None: args.ymin = -3 if args.xcol == 'rad_re' else -0.2
    if args.ymax is None: args.ymax = 0.05
    if args.cmin is None: args.cmin = 0
    if args.cmax is None: args.cmax = 6

    # -------declare figure object-------------
    fig, ax = plt.subplots(1, figsize=(9, 5))
    fig.subplots_adjust(top=0.95, bottom=0.15, left=0.12, right=1.05)

    # ---------plot observations----------------
    if args.overplot_obs:
        ax, df_manga = overplot_manga(ax, args)
        manga_text = '_overplot_manga_' + args.manga_diag
        fig.text(0.15, 0.2, 'MaNGA', ha='left', va='top', color='Grey', fontsize=args.fontsize)
    else:
        df_manga = None
        manga_text = ''

    # --------loop over different FOGGIE halos-------------
    for index, args.halo in enumerate(args.halo_arr):
        df = load_df(args)
        df = df[(np.log10(df['mass']) >= args.xmin) & (np.log10(df['mass']) <= args.xmax)]
        #df = df[(df['Zgrad'] >= args.ymin) & (df['Zgrad'] <= args.ymax)]

        df.drop_duplicates(subset='output', keep='last', ignore_index=True, inplace=True)
        df.sort_values(by='redshift', ascending=False, ignore_index=True, inplace=True)

        if args.binby is not None:
            df[args.binby + '_bins'] = pd.cut(df[args.binby], bins=np.logspace(np.min(np.log10(df[args.binby])), np.max(np.log10(df[args.binby])), args.nbins) if logbin else np.linspace(np.min(df[args.binby]), np.max(df[args.binby]), args.nbins))
            df = df[['redshift', 'mass', 'Zgrad', args.binby + '_bins']].groupby(args.binby + '_bins', as_index=False).agg(np.mean)
            df.dropna(inplace=True)

        # -----plot line with color gradient--------
        points = np.array([np.log10(df['mass']), df['Zgrad']]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(args.cmin, args.cmax)
        lc = LineCollection(segments, cmap=cmap_arr[index], norm=norm)
        lc.set_array(df['redshift'])
        lc.set_linewidth(2)
        plot = ax.add_collection(lc)

        # -------scatter plot------------
        #plot = ax.scatter(np.log10(df['mass']), df['Zgrad'], c=df['redshift'], cmap=cmap_arr[index])

        fig.text(0.15, 0.25 + index * 0.05, halo_dict[args.halo], ha='left', va='top', color=mpl_cm.get_cmap(cmap_arr[index])(0.2), fontsize=args.fontsize)
        df['halo'] = args.halo
        df_master = pd.concat([df_master, df])

    cax = plt.colorbar(plot)
    cax.ax.tick_params(labelsize=args.fontsize)
    cax.set_label('Redshift', fontsize=args.fontsize)

    ax.set_xlim(args.xmin, args.xmax)
    ax.set_ylim(args.ymin, args.ymax)

    ax.set_xticklabels(['%.1F' % item for item in ax.get_xticks()], fontsize=args.fontsize)
    ax.set_yticklabels(['%.2F' % item for item in ax.get_yticks()], fontsize=args.fontsize)

    ax.set_xlabel(r'$\log{(\mathrm{M}_*/\mathrm{M}_\odot)}$', fontsize=args.fontsize)
    ax.set_ylabel(r'$\nabla(\log{\mathrm{Z}}$) (dex/r$_{\mathrm{e}}$)' if args.xcol == 'rad_re' else r'$\Delta Z$ (dex/kpc)', fontsize=args.fontsize)

    binby_text = '' if args.binby is None else '_binby_' + args.binby
    figname = args.output_dir + 'figs/' + ','.join(args.halo_arr) + '_MZGR_xcol_%s_upto%.1FRe%s%s%s.png' % (args.xcol, args.upto_re, args.weightby_text, binby_text, manga_text)
    fig.savefig(figname)
    print('Saved plot as', figname)
    plt.show(block=False)

    return fig, df_master, df_manga


# -----main code-----------------
if __name__ == '__main__':
    args_tuple = parse_args('8508', 'RD0042')  # default simulation to work upon when comand line args not provided
    if type(args_tuple) is tuple: args = args_tuple[0] # if the sim has already been loaded in, in order to compute the box center (via utils.pull_halo_center()), then no need to do it again
    else: args = args_tuple
    if not args.keep: plt.close('all')

    # ---------reading in existing MZgrad txt file------------------
    args.weightby_text = '' if args.weight is None else '_wtby_' + args.weight

    fig, df_binned, df_manga = plot_MZGR(args)

    print('Completed in %s' % (datetime.timedelta(minutes=(time.time() - start_time) / 60)))



