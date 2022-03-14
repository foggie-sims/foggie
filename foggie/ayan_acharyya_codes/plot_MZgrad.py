#!/usr/bin/env python3

"""

    Title :      plot_MZgrad
    Notes :      Plot mass - metallicity gradient relation for a given FOGGIE galaxy
    Output :     M-Z gradient plots as png files plus, optinally, MZR plot
    Author :     Ayan Acharyya
    Started :    Mar 2022
    Examples :   run plot_MZgrad.py --system ayan_local --halo 8508,5036,5016,4123 --upto_re 3 --xcol rad_re --keep --weight mass --ymin -2 --xmin 8.5 --overplot_manga --overplot_clear --binby mass --nbins 200 --zhighlight --use_gasre --overplot_smoothed
                 run plot_MZgrad.py --system ayan_local --halo 8508 --upto_re 3 --xcol rad_re --keep --weight mass --ymin -0.5 --xmin 8.5 --overplot_manga --overplot_clear --overplot_belfiore --overplot_mjngozzi --binby mass --nbins 200 --zhighlight --use_gasre --overplot_smoothed --manga_diag pyqz
                 run plot_MZgrad.py --system ayan_pleiades --halo 8508 --upto_re 3 --xcol rad_re --weight mass --binby mass --nbins 20 --cmap plasma --xmin 8.5 --xmax 11 --ymin 0.3 --overplot_manga --manga_diag n2

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

    df.drop_duplicates(subset='output', keep='last', ignore_index=True, inplace=True)
    df.sort_values(by='redshift', ascending=False, ignore_index=True, inplace=True)

    binned_fit_text = '_binned' if args.use_binnedfit else ''
    which_re = 're_coldgas' if args.use_gasre else 're_stars'
    try:
        df = df[['output', 'redshift', 'time', which_re, 'mass_' + which_re] + [item + binned_fit_text + '_' + which_re for item in ['Zcen', 'Zcen_u', 'Zgrad', 'Zgrad']] + ['Ztotal_' + which_re]]
        df.columns = ['output', 'redshift', 'time', 're', 'mass', 'Zcen', 'Zcen_u', 'Zgrad', 'Zgrad_u', 'Ztotal']
    except:
        pass

    return df

# -----------------------------------
def overplot_mingozzi(ax, paper='M20', color='salmon', diag='PP04'):
    '''
    Function to overplot the observed MZGR from Mingozzi+20 OR Belfiore+17 and return the axis handle
    '''
    print('Overplotting Mingozzi+20 data..')

    input_filename = HOME + '/Desktop/bpt_contsub_contu_rms/newfit/lit_log_mass_Zgrad|r_e_bin.txt'
    df = pd.read_table(input_filename, delim_whitespace=True, comment='#') # grad is in dex/re, mass_bin is in log

    ax.scatter(df['log_mass'], df[diag + '_' + paper], c=color, s=50)
    ax.plot(df['log_mass'], df[diag + '_' + paper], c=color, lw=2)

    return ax

# -----------------------------------
def overplot_clear(ax):
    '''
    Function to overplot the observed MZGR from CLEAR (Simons+21) and return the axis handle
    '''
    print('Overplotting Simons+21 data..')

    input_filename = HOME + '/models/clear/clear_simons_2021.txt'
    df = pd.read_table(input_filename, delim_whitespace=True) # grad is in dex/re, mass_bin is in log
    col_arr = ['r', 'k']

    for index,survey in enumerate(pd.unique(df['survey'])):
        df_sub = df[df['survey'] == survey]

        ax.errorbar(df_sub['mass_bin'], df_sub['grad'], yerr=df_sub['egrad'], c=col_arr[index], ls='none')
        ax.scatter(df_sub['mass_bin'], df_sub['grad'], c=col_arr[index], s=50)
        ax.plot(df_sub['mass_bin'], df_sub['grad'], c=col_arr[index], lw=2)

    return ax

# -----------------------------------
def overplot_manga(ax, args):
    '''
    Function to overplot the observed MZGR from MaNGA and return the axis handle
    '''
    print('Overplotting MaNGA data..')

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

# ---------------------------------------------
def get_multicolored_line(xdata, ydata, colordata, cmap, cmin, cmax, lw=2, ls='solid'):
    '''
    Function to take x,y and z (color) data as three 1D arrays and return a smoothly-multi-colored line object that can be added to an axis object
    '''
    points = np.array([xdata, ydata]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(cmin, cmax)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(colordata)
    lc.set_linewidth(lw)
    lc.set_linestyle(ls)
    return lc

# -----------------------------------
def plot_MZGR(args):
    '''
    Function to plot the mass-metallicity gradient relation, based on an input dataframe
    '''

    logbin = True if args.binby == 'mass' else False
    df_master = pd.DataFrame()
    cmap_arr = ['Purples_r', 'Oranges_r', 'Greens_r', 'Blues_r', 'PuRd_r', 'YlOrBr_r']

    # -------------get plot limits-----------------
    if args.xmin is None: args.xmin = 6
    if args.xmax is None: args.xmax = 11.5
    if args.ymin is None: args.ymin = -2 if args.xcol == 'rad_re' else -0.2
    if args.ymax is None: args.ymax = 0.1
    if args.cmin is None: args.cmin = 0
    if args.cmax is None: args.cmax = 6

    # -------declare figure object-------------
    fig, ax = plt.subplots(1, figsize=(10, 5))
    fig.subplots_adjust(top=0.95, bottom=0.15, left=0.12, right=1.05)

    # ---------plot observations----------------
    obs_text = ''
    if args.overplot_manga:
        ax, df_manga = overplot_manga(ax, args)
        obs_text += '_manga_' + args.manga_diag
        fig.text(0.15, 0.2, 'MaNGA: Pipe3D', ha='left', va='top', color='Grey', fontsize=args.fontsize)
    else:
        df_manga = -99 # bogus value

    if args.overplot_clear:
        ax = overplot_clear(ax)
        obs_text += '_clear'
        fig.text(0.15, 0.25, 'CLEAR: Simons+21', ha='left', va='top', color='k', fontsize=args.fontsize)
        fig.text(0.15, 0.3, 'MaNGA: Belfiore+17', ha='left', va='top', color='r', fontsize=args.fontsize)

    if args.overplot_belfiore:
        ax = overplot_mingozzi(ax, paper='B17', color='darkolivegreen', diag='M08')
        obs_text += '_belfiore_'
        fig.text(0.15, 0.35, 'MaNGA: Belfiore+17', ha='left', va='top', color='darkolivegreen', fontsize=args.fontsize)

    if args.overplot_mingozzi:
        ax = overplot_mingozzi(ax, paper='M20', color='salmon', diag='M08')
        obs_text += '_mingozzi_'
        fig.text(0.15, 0.4, 'MaNGA: Mingozzi+20', ha='left', va='top', color='salmon', fontsize=args.fontsize)

    # --------loop over different FOGGIE halos-------------
    for index, args.halo in enumerate(args.halo_arr[::-1]):
        thisindex = len(args.halo_arr) - index - 1
        df = load_df(args)
        df = df[(np.log10(df['mass']) >= args.xmin) & (np.log10(df['mass']) <= args.xmax)]
        #df = df[(df['Zgrad'] >= args.ymin) & (df['Zgrad'] <= args.ymax)]

        if args.binby is not None:
            df[args.binby + '_bins'] = pd.cut(df[args.binby], bins=np.logspace(np.min(np.log10(df[args.binby])), np.max(np.log10(df[args.binby])), args.nbins) if logbin else np.linspace(np.min(df[args.binby]), np.max(df[args.binby]), args.nbins))
            df = df[['redshift', 'mass', 'Zgrad', args.binby + '_bins']].groupby(args.binby + '_bins', as_index=False).agg(np.mean)
            df.dropna(inplace=True)

        # -----plot line with color gradient--------
        line = get_multicolored_line(np.log10(df['mass']), df['Zgrad'], df['redshift'], cmap_arr[thisindex], args.cmin, args.cmax, lw=1 if args.overplot_smoothed else 2)
        plot = ax.add_collection(line)

        # ------- overplotting redshift-binned scatter plot------------
        if args.zhighlight:
            df['redshift_int'] = np.floor(df['redshift'])
            df_zbin = df.drop_duplicates(subset='redshift_int', keep='last', ignore_index=True)
            dummy = ax.scatter(np.log10(df_zbin['mass']), df_zbin['Zgrad'], c=df_zbin['redshift'], cmap=cmap_arr[thisindex], lw=1, edgecolor='k', s=100, alpha=0.2 if args.overplot_smoothed else 1)
            print('For halo', args.halo, 'highlighted z =', [float('%.1F'%item) for item in df_zbin['redshift'].values])

        # ------- overplotting a boxcar smoothed version of the MZGR------------
        if args.overplot_smoothed:
            line.set_alpha(0.2) # make the actual wiggly line fainter
            npoints = int(len(df)/8)
            if npoints % 2 == 0: npoints += 1
            box = np.ones(npoints) / npoints
            df['Zgrad_smoothed'] = np.convolve(df['Zgrad'], box, mode='same')
            smoothline = get_multicolored_line(np.log10(df['mass']), df['Zgrad_smoothed'], df['redshift'], cmap_arr[thisindex], args.cmin, args.cmax, lw=2)
            plot = ax.add_collection(smoothline)
            print('Boxcar-smoothed plot for halo', args.halo, 'with', npoints, 'points')

        fig.text(0.15, 0.9 - thisindex * 0.05, halo_dict[args.halo], ha='left', va='top', color=mpl_cm.get_cmap(cmap_arr[thisindex])(0.2), fontsize=args.fontsize)
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
    figname = args.output_dir + 'figs/' + ','.join(args.halo_arr) + '_MZGR_xcol_%s_upto%.1FRe%s%s%s.png' % (args.xcol, args.upto_re, args.weightby_text, binby_text, obs_text)
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



