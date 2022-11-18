#!/usr/bin/env python3

"""

    Title :      plot_MZgrad
    Notes :      Plot mass - metallicity gradient relation for a given FOGGIE galaxy
    Output :     M-Z gradient plots as png files plus, optionally, MZR plot
    Author :     Ayan Acharyya
    Started :    Mar 2022
    Examples :   run plot_MZgrad.py --system ayan_local --halo 8508,5036,5016,4123 --upto_re 3 --Zgrad_den rad_re --keep --weight mass --overplot_manga --overplot_clear --binby log_mass --nbins 200 --zhighlight --use_gasre --overplot_smoothed 1500
                 run plot_MZgrad.py --system ayan_local --halo 8508,5036,5016,4123 --upto_kpc 10 --Zgrad_den rad_re --keep --weight mass --overplot_manga --overplot_clear --binby log_mass --nbins 200 --zhighlight --use_gasre --overplot_smoothed 1500
                 run plot_MZgrad.py --system ayan_local --halo 8508 --upto_re 3 --Zgrad_den rad_re --keep --weight mass --overplot_manga --overplot_clear --overplot_belfiore --overplot_mjngozzi --binby log_mass --nbins 200 --zhighlight --use_gasre --overplot_smoothed 1500 --manga_diag pyqz
                 run plot_MZgrad.py --system ayan_pleiades --halo 8508 --upto_re 3 --Zgrad_den rad_re --weight mass --binby log_mass --nbins 20 --cmap plasma --xmax 11 --ymin 0.3 --overplot_manga --manga_diag n2
                 run plot_MZgrad.py --system ayan_local --halo 8508 --Zgrad_den kpc --upto_kpc 10 --keep --weight mass --ycol Zgrad --xcol time --colorcol log_mass --overplot_smoothed 1500 --zhighlight
                 run plot_MZgrad.py --system ayan_local --halo 8508 --Zgrad_den kpc --upto_kpc 10 --keep --weight mass --ycol Zgrad --xcol time --colorcol re --cmax 3 --zhighlight
                 run plot_MZgrad.py --system ayan_local --halo 8508 --Zgrad_den kpc --upto_kpc 10 --keep --weight mass --ycol Ztotal --xcol time --colorcol log_mass --zhighlight --docomoving
                 run plot_MZgrad.py --system ayan_local --halo 8508 --Zgrad_den kpc --upto_kpc 10 --keep --weight mass --ycol Zgrad --xcol log_mass --colorcol time --zhighlight --plot_deviation --zcol log_ssfr
                 run plot_MZgrad.py --system ayan_local --halo 8508 --Zgrad_den kpc --upto_kpc 10 --keep --weight mass --ycol Zgrad --xcol log_mass --colorcol time --zhighlight --plot_timefraction --Zgrad_allowance 0.05 --upto_z 2 --overplot_smoothed 1500
                 run plot_MZgrad.py --system ayan_local --halo 8508 --Zgrad_den kpc --upto_kpc 10 --keep --weight mass --ycol Zgrad --xcol log_mass --colorcol time --zhighlight --plot_timefraction --Zgrad_allowance 0.05 --upto_z 2 --overplot_cadence 50
                 run plot_MZgrad.py --system ayan_local --halo 8508,5016,4123 --Zgrad_den kpc --upto_kpc 10 --keep --weight mass --ycol Zgrad --xcol time --nocolorcoding --zhighlight --overplot_smoothed 1500 --hiderawdata [FOR MOLLY]
                 run plot_MZgrad.py --system ayan_local --halo 8508,5036,5016,4123,2878,2392 --Zgrad_den kpc --upto_kpc 10 --weight mass --glasspaper [FOR MATCHING GLASS PAPER]
"""
from header import *
from util import *
from matplotlib.collections import LineCollection
from matplotlib.colors import is_color_like
start_time = time.time()

# ---------------------------------
def load_df(args):
    '''
    Function to load and return the dataframe containing MZGR
    '''
    args.foggie_dir, args.output_dir, args.run_loc, args.code_path, args.trackname, args.haloname, args.spectra_dir, args.infofile = get_run_loc_etc(args)
    Zgrad_den_text = 'rad' if args.Zgrad_den == 'kpc' else 'rad_re'
    if args.upto_kpc is not None:
        upto_text = '_upto%.1Fckpchinv' % args.upto_kpc if args.docomoving else '_upto%.1Fkpc' % args.upto_kpc
    else:
        upto_text = '_upto%.1FRe' % args.upto_re
    grad_filename = args.output_dir + 'txtfiles/' + args.halo + '_MZR_xcol_%s%s%s.txt' % (Zgrad_den_text, upto_text, args.weightby_text)

    convert_Zgrad_from_dexkpc_to_dexre = False
    convert_Zgrad_from_dexre_to_dexkpc = False

    if os.path.exists(grad_filename):
        print('Trying to read in', grad_filename)
        df = pd.read_table(grad_filename, delim_whitespace=True)

    elif not os.path.exists(grad_filename) and args.Zgrad_den == 're':
        print('Could not find', grad_filename)
        grad_filename = grad_filename.replace('rad_re', 'rad')
        print('Trying to read in', grad_filename, 'instead')
        df = pd.read_table(grad_filename, delim_whitespace=True)
        convert_Zgrad_from_dexkpc_to_dexre = True

    elif not os.path.exists(grad_filename) and args.Zgrad_den == 'kpc':
        print('Could not find', grad_filename)
        grad_filename = grad_filename.replace('rad', 'rad_re')
        print('Trying to read in', grad_filename, 'instead')
        df = pd.read_table(grad_filename, delim_whitespace=True)
        convert_Zgrad_from_dexre_to_dexkpc = True

    df.drop_duplicates(subset='output', keep='last', ignore_index=True, inplace=True)
    df.sort_values(by='redshift', ascending=False, ignore_index=True, inplace=True)

    binned_fit_text = '_binned' if args.use_binnedfit else ''
    which_re = 're_coldgas' if args.use_gasre else 're_stars'
    re_text = 'fixedr' if args.upto_kpc is not None else which_re

    try:
        df = df[['output', 'redshift', 'time', which_re, 'mass_' + re_text] + [item + binned_fit_text + '_' + re_text for item in ['Zcen', 'Zcen_u', 'Zgrad', 'Zgrad']] + ['Ztotal_' + re_text]]
    except KeyError as e:
        if args.upto_kpc is not None:
            print('This is probably an old version of the file, with different column nomenclature. Adjusting accordingly..')
            re_text = which_re
            df = df[['output', 'redshift', 'time', which_re, 'mass_' + re_text] + [item + binned_fit_text + '_' + re_text for item in ['Zcen', 'Zcen_u', 'Zgrad', 'Zgrad']] + ['Ztotal_' + re_text]]
        pass

    df.columns = ['output', 'redshift', 'time', 're', 'mass', 'Zcen', 'Zcen_u', 'Zgrad', 'Zgrad_u', 'Ztotal']
    df['log_mass'] = np.log10(df['mass'])
    df = df.drop('mass', axis=1)

    df['Ztotal'] = np.log10(df['Ztotal']) + 8.69

    if convert_Zgrad_from_dexkpc_to_dexre:
        print('Zgrad is in dex/kpc, converting it to dex/re')
        df['Zgrad'] *= df['re']
        df['Zgrad_u'] *= df['re']

    elif convert_Zgrad_from_dexre_to_dexkpc:
        print('Zgrad is in dex/re, converting it to dex/kpc')
        df['Zgrad'] /= df['re']
        df['Zgrad_u'] /= df['re']

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

# -----------------------------------------------
def plot_zhighlight(df, ax, cmap, args, ycol=None):
    '''
    Function to overplot circles at integer-ish redshifts and return the ax
    '''
    if ycol is None: ycol = args.ycol
    df['redshift_int'] = np.floor(df['redshift'])
    df_zbin = df.drop_duplicates(subset='redshift_int', keep='last', ignore_index=True)
    if is_color_like(cmap): dummy = ax.scatter(df_zbin[args.xcol], df_zbin[ycol], c=cmap, lw=1, edgecolor='k', s=100, alpha=0.5, zorder=20)
    else: dummy = ax.scatter(df_zbin[args.xcol], df_zbin[ycol], c=df_zbin[args.colorcol], cmap=cmap, vmin=args.cmin, vmax=args.cmax, lw=1, edgecolor='k', s=100, alpha=0.2 if (args.overplot_smoothed and 'smoothed' not in ycol) or args.overplot_cadence else 1, zorder=20)
    print('For halo', args.halo, 'highlighted z =', [float('%.1F' % item) for item in df_zbin['redshift'].values], 'with circles')
    return ax

# ----------------------------------
class MyDefaultDict(dict):
    '''
    subclass to modify dict to return the missing key itself
    '''
    __missing__ = lambda self, key: key

# -----------------------------------
def plot_MZGR(args):
    '''
    Function to plot the mass-metallicity gradient relation, based on an input dataframe
    '''

    df_master = pd.DataFrame()
    cmap_arr = ['Purples', 'Oranges', 'Greens', 'Blues', 'PuRd', 'YlOrBr']
    things_that_reduce_with_time = ['redshift', 're'] # whenever this quantities are used as colorcol, the cmap is inverted, so that the darkest color is towards later times

    # -------------get plot limits-----------------
    lim_dict = {'Zgrad': (-0.5, 0.1)  if args.Zgrad_den == 'kpc' else (-2, 0.1), 're': (0, 30), 'log_mass': (8.5, 11.5), 'redshift': (0, 6), 'time': (0, 14), 'sfr': (0, 60), 'log_ssfr': (-11, -8), 'Ztotal': (8, 9), 'log_sfr': (-1, 3)}
    label_dict = MyDefaultDict(Zgrad=r'$\nabla(\log{\mathrm{Z}}$) (dex/r$_{\mathrm{e}}$)' if args.Zgrad_den == 're' else r'$\Delta Z$ (dex/kpc)' if not args.glasspaper else 'Metallicity gradient (dex/kpc)', \
        re='Scale length (kpc)', log_mass=r'$\log{(\mathrm{M}_*/\mathrm{M}_\odot)}$', redshift='Redshift', time='Time (Gyr)', sfr=r'SFR (M$_{\odot}$/yr)', \
        log_ssfr=r'$\log{\, \mathrm{sSFR} (\mathrm{yr}^{-1})}$', Ztotal=r'$\log{(\mathrm{O/H})}$ + 12', log_sfr=r'$\log{(\mathrm{SFR} (\mathrm{M}_{\odot}/yr))}$')

    if args.xmin is None: args.xmin = lim_dict[args.xcol][0]
    if args.xmax is None: args.xmax = lim_dict[args.xcol][1]
    if args.ymin is None: args.ymin = lim_dict[args.ycol][0]
    if args.ymax is None: args.ymax = lim_dict[args.ycol][1]
    if args.cmin is None: args.cmin = lim_dict[args.colorcol][0]
    if args.cmax is None: args.cmax = lim_dict[args.colorcol][1]
    if args.zmin is None: args.zmin = lim_dict[args.zcol][0]
    if args.zmax is None: args.zmax = lim_dict[args.zcol][1]

    # --------------pre-set values to match the GLASS paper plot--------------
    if args.glasspaper:
        args.ycol, args.ymin, args.ymax = 'Zgrad', -0.35, 0.3
        args.xcol, args.xmin, args.xmax = 'redshift', 0, 3.2
        args.colorcol = 'log_ssfr'
        args.nocolorcoding = True
        args.zhighlight = True
        args.overplot_smoothed = None
        args.hiderawdata = False
        args.fontsize = 20

    # -------declare figure object-------------
    fig, ax = plt.subplots(1, figsize=(12, 6))
    fig.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.97 if args.nocolorcoding else 1.05)

    if args.plot_deviation or args.plot_timefraction:
        fig2, ax2 = plt.subplots(1, figsize=(12, 6))
        fig2.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=1.05 if args.plot_deviation else 0.97)

    # ---------plot observations----------------
    obs_text = ''
    if args.overplot_manga and args.Zgrad_den == 're':
        ax, df_manga = overplot_manga(ax, args)
        obs_text += '_manga_' + args.manga_diag
        fig.text(0.15, 0.2, 'MaNGA: Pipe3D', ha='left', va='top', color='Grey', fontsize=args.fontsize)
    else:
        df_manga = -99 # bogus value

    if args.overplot_clear and args.Zgrad_den == 're':
        ax = overplot_clear(ax)
        obs_text += '_clear'
        fig.text(0.15, 0.25, 'CLEAR: Simons+21', ha='left', va='top', color='k', fontsize=args.fontsize)
        fig.text(0.15, 0.3, 'MaNGA: Belfiore+17', ha='left', va='top', color='r', fontsize=args.fontsize)

    if args.overplot_belfiore and args.Zgrad_den == 're':
        ax = overplot_mingozzi(ax, paper='B17', color='darkolivegreen', diag='M08')
        obs_text += '_belfiore_'
        fig.text(0.15, 0.35, 'MaNGA: Belfiore+17', ha='left', va='top', color='darkolivegreen', fontsize=args.fontsize)

    if args.overplot_mingozzi and args.Zgrad_den == 're':
        ax = overplot_mingozzi(ax, paper='M20', color='salmon', diag='M08')
        obs_text += '_mingozzi_'
        fig.text(0.15, 0.4, 'MaNGA: Mingozzi+20', ha='left', va='top', color='salmon', fontsize=args.fontsize)

    if args.glasspaper: # overplot GLASS dtaa point
        ax.plot(3.06, 0.165, marker='*', ms=30, mfc='yellow', mec='red', mew=1)
        fig.text(0.85, 0.94, 'GLASS', ha='left', va='top', color='gold', fontsize=args.fontsize)

    # --------loop over different FOGGIE halos-------------
    for index, args.halo in enumerate(args.halo_arr[::-1]):
        thisindex = len(args.halo_arr) - index - 1
        df = load_df(args)
        # -------- reading in additional dataframes-------
        sfr_filename = args.code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/sfr'
        if os.path.exists(sfr_filename):
            print('Reading SFR history from', sfr_filename)
        else:
            print(sfr_filename, 'not found')
            sfr_filename = sfr_filename.replace(args.run, args.run[:14])
            print('Instead, reading SFR history from', sfr_filename)
        addn_df = pd.read_table(sfr_filename, names=('output', 'redshift', 'sfr'), comment='#', delim_whitespace=True)
        df = df.merge(addn_df[['output', 'sfr']], on='output')
        df['ssfr'] = df['sfr'] / 10**df['log_mass']
        df = df.replace([0, np.inf, -np.inf], np.nan).dropna(axis=0)
        df['log_ssfr'] = np.log10(df['ssfr'])
        df['log_sfr'] = np.log10(df['sfr'])
        df = df.sort_values(args.xcol)

        if args.nocolorcoding:
            args.colorcol, args.cmin, args.cmax = 'dummy', 0, 1
            df['dummy'] = 0.8

        #df = df[(df[args.xcol] >= args.xmin) & (df[args.xcol] <= args.xmax)]
        #df = df[(df[args.ycol] >= args.ymin) & (df[args.ycol] <= args.ymax)]
        df = df[(df[args.colorcol] >= args.cmin) & (df[args.colorcol] <= args.cmax)]

        # ------- plot only the binned plot------------
        if args.binby is not None:
            df[args.binby + '_bins'] = pd.cut(df[args.binby], bins=np.linspace(np.min(df[args.binby]), np.max(df[args.binby]), args.nbins))
            cols_to_bin = [args.colorcol, args.xcol, args.ycol, args.binby + '_bins']
            if args.plot_deviation: cols_to_bin += [args.zcol]
            if 'redshift' not in cols_to_bin: cols_to_bin += ['redshift']
            df = df[cols_to_bin].groupby(args.binby + '_bins', as_index=False).agg(np.mean)
            df.dropna(axis=0, inplace=True)
            df = df.sort_values(args.xcol)

        # -----plot line with color gradient--------
        this_cmap = cmap_arr[thisindex] + '_r' if args.colorcol in things_that_reduce_with_time else cmap_arr[thisindex] # reverse colromap for redshift
        reversed_thiscmap = this_cmap + '_r' if '_r' not in this_cmap else this_cmap[:-2]
        thistextcolor = mpl_cm.get_cmap(this_cmap)(0.2 if args.colorcol == 'redshift' else 0.2 if args.colorcol == 're' else 0.8)
        if not args.hiderawdata: # to hide the squiggly lines (and may be only have the overplotted or z-highlighted version)
            line = get_multicolored_line(df[args.xcol], df[args.ycol], df[args.colorcol], this_cmap, args.cmin, args.cmax, lw=1 if args.overplot_smoothed or args.overplot_cadence or args.glasspaper else 2)
            plot = ax.add_collection(line)

        # ------- overplotting specific snapshot highlights------------
        if args.snaphighlight is not None:
            snaps_to_highlight = [item for item in args.snaphighlight.split(',')]
            df_snaps = df[df['output'].isin(snaps_to_highlight)]
            dummy = ax.scatter(df_snaps[args.xcol], df_snaps[args.ycol], c=df_snaps[args.colorcol], cmap=this_cmap, vmin=args.cmin, vmax=args.cmax, lw=1, edgecolor='k', s=200, alpha=0.2 if args.overplot_smoothed else 1, marker='*', zorder=10)
            print('For halo', args.halo, 'highlighted snapshots =', df_snaps['output'].values, 'with stars')

        # ------- overplotting redshift-binned scatter plot------------
        if args.zhighlight:
            ax = plot_zhighlight(df, ax, this_cmap, args)

        # ------- overplotting a boxcar smoothed version of the MZGR------------
        if args.overplot_smoothed:
            mean_dt = (df['time'].max() - df['time'].min())*1000/len(df) # Myr
            npoints = int(np.round(args.overplot_smoothed/mean_dt))
            if npoints % 2 == 0: npoints += 1
            box = np.ones(npoints) / npoints
            df[args.ycol + '_smoothed'] = np.convolve(df[args.ycol], box, mode='same')
            print('Boxcar-smoothed plot for halo', args.halo, 'with', npoints, 'points, =', npoints * mean_dt, 'Myr')

            if 'line' in locals() and not args.nocolorcoding: line.set_alpha(0.2) # make the actual wiggly line fainter (unless making plots for Molly's talk)
            smoothline = get_multicolored_line(df[args.xcol], df[args.ycol + '_smoothed'], df[args.colorcol], this_cmap, args.cmin, args.cmax, lw=2)
            plot = ax.add_collection(smoothline) ## keep this commented out for making plots for Molly's talk
            if args.nocolorcoding: # for making plots for Molly's talk
                ax = plot_zhighlight(df, ax, this_cmap, args, ycol=args.ycol + '_smoothed')
                smoothline.set_alpha(0.2)

        # ------- overplotting a lower cadence version of the MZGR------------
        if args.overplot_cadence:
            mean_dt = (df['time'].max() - df['time'].min())*1000/len(df) # Myr
            npoints = int(np.round(args.overplot_cadence/mean_dt))
            df_short = df.iloc[::npoints, :]
            print('Overplot for halo', args.halo, 'only every', npoints, 'th data point, i.e. cadence of', npoints * mean_dt, 'Myr')
            if 'line' in locals() and not args.nocolorcoding: line.set_alpha(0.2) # make the actual wiggly line fainter (unless making plots for Molly's talk)

            yfunc = interp1d(df_short[args.xcol], df_short[args.ycol], fill_value='extrapolate') # interpolating the low-cadence data
            cfunc = interp1d(df_short[args.xcol], df_short[args.colorcol], fill_value='extrapolate')
            df[args.ycol + '_interp'] = yfunc(df[args.xcol])
            df[args.colorcol + '_interp'] = cfunc(df[args.xcol])
            interpline = get_multicolored_line(df[args.xcol], df[args.ycol + '_interp'], df[args.colorcol + '_interp'], this_cmap, args.cmin, args.cmax, lw=2)
            plot = ax.add_collection(interpline)

        # ------- making additional plot of deviation in gradient vs other quantities, like SFR------------
        if args.plot_deviation:
            print('Plotting deviation vs', args.colorcol, 'halo', args.halo)
            if args.overplot_smoothed: col_to_subtract = args.ycol + '_smoothed'
            elif args.overplot_cadence: col_to_subtract = args.ycol + '_interp'

            df[args.ycol + '_deviation'] = df[args.ycol] - df[col_to_subtract]
            df = df.sort_values(args.zcol)

            # --------- scatter plot------------
            #ax2.scatter(df[args.zcol], df[args.ycol + '_deviation'], c=thistextcolor, edgecolor='k', lw=0.5, s=50)

            # --------- colored line plot------------
            line2 = get_multicolored_line(df[args.zcol], np.abs(df[args.ycol + '_deviation']), df[args.colorcol], reversed_thiscmap, args.cmin, args.cmax, lw=1)
            plot2 = ax2.add_collection(line2)

            # --------- smoothed colored line plot------------
            line2.set_alpha(0.2) # make the actual wiggly line fainter

            df[args.ycol + '_deviation_smoothed'] = np.convolve(np.abs(df[args.ycol + '_deviation']), box, mode='same')
            smoothline2 = get_multicolored_line(df[args.zcol], df[args.ycol + '_deviation_smoothed'], df[args.colorcol], reversed_thiscmap, args.cmin, args.cmax, lw=2)
            plot2 = ax2.add_collection(smoothline2)

        # ------- making additional plot of deviation in gradient vs other quantities, like SFR------------
        elif args.plot_timefraction:
            print('Plotting time fraction vs', args.colorcol, 'halo', args.halo)
            if args.overplot_smoothed: col_to_subtract = args.ycol + '_smoothed'
            elif args.overplot_cadence: col_to_subtract = args.ycol + '_interp'

            df[args.ycol + '_deviation'] = df[args.ycol] - df[col_to_subtract]
            df = df.sort_values('time')
            args.zcol = 'time'

            # --------- mono-color line plot------------
            plot2 = ax2.plot(df[args.zcol], df[args.ycol + '_deviation'], lw=1, color=thistextcolor)

            if args.zhighlight:
                ax2 = plot_zhighlight(df, ax2, thistextcolor, args, ycol=args.ycol + '_deviation')
                #df_zbin = df.drop_duplicates(subset='redshift_int', keep='last', ignore_index=True)
                #ax2.scatter(df_zbin[args.zcol], df_zbin[args.ycol + '_deviation'], color=thistextcolor, lw=1, edgecolor='k', s=100, alpha=0.5, zorder=20)

            # ---------horizontal lines for cut-off-----
            ax2.axhline(args.Zgrad_allowance, lw=0.5, ls='dashed', color='k')
            ax2.axhline(- args.Zgrad_allowance, lw=0.5, ls='dashed', color='k')
            ax2.axvline(df[df['redshift'] >= args.upto_z]['time'].values[-1], lw=0.5, ls='dashed', color='k')

            ax2.text(lim_dict[args.zcol][1] * 0.95, args.Zgrad_allowance * 1.1, '%.2F dex/%s' % (args.Zgrad_allowance, args.Zgrad_den), c='k', ha='right', va='bottom', fontsize=args.fontsize)
            ax2.text(lim_dict[args.zcol][1] * 0.95, -args.Zgrad_allowance * 1.1, '%.2F dex/%s' % (-args.Zgrad_allowance, args.Zgrad_den), c='k', ha='right', va='top', fontsize=args.fontsize)

            # ---------filled area plot for deviation outside allowance-----
            ax2.fill_between(df[args.zcol], df[args.ycol + '_deviation'], args.Zgrad_allowance, where=(df[args.ycol + '_deviation'] > args.Zgrad_allowance), color=thistextcolor, alpha=0.3)
            ax2.fill_between(df[args.zcol],  df[args.ycol + '_deviation'], -args.Zgrad_allowance, where=(df[args.ycol + '_deviation'] < -args.Zgrad_allowance), color=thistextcolor, alpha=0.3)

            # ---------calculating fration of time spent in filled area-----
            dfsub = df[df['redshift'] >= args.upto_z]
            snaps_outside_allowance = len(dfsub[(dfsub['Zgrad_deviation'] > args.Zgrad_allowance) | (dfsub['Zgrad_deviation'] < -args.Zgrad_allowance)])
            total_snaps = len(dfsub)
            timefraction_outside = snaps_outside_allowance * 100 / total_snaps
            print('Halo', args.halo, 'spends %.2F %%' %timefraction_outside, 'of the time outside +/-', args.Zgrad_allowance, 'dex/kpc deviation upto redshift %.1F' % args.upto_z)

        fig.text(0.85 if args.glasspaper else 0.15, 0.9 - thisindex * 0.05, halo_dict[args.halo], ha='left', va='top', color=thistextcolor, fontsize=args.fontsize)
        if args.plot_deviation or args.plot_timefraction: fig2.text(0.15, 0.9 - thisindex * 0.05, halo_dict[args.halo], ha='left', va='top', color=thistextcolor, fontsize=args.fontsize)
        df['halo'] = args.halo
        df_master = pd.concat([df_master, df])

    # ------- tidying up fig1------------
    if not args.nocolorcoding:
        cax = fig.colorbar(plot)
        cax.ax.tick_params(labelsize=args.fontsize)
        cax.set_label(label_dict[args.colorcol], fontsize=args.fontsize)

    if args.xcol == 'redshift':  ax.set_xlim(args.xmax, args.xmin)
    else: ax.set_xlim(args.xmin, args.xmax)
    ax.set_ylim(args.ymin, args.ymax)

    ax.set_xticklabels(['%.1F' % item for item in ax.get_xticks()], fontsize=args.fontsize)
    ax.set_yticklabels(['%.2F' % item for item in ax.get_yticks()], fontsize=args.fontsize)

    ax.set_xlabel(label_dict[args.xcol], fontsize=args.fontsize)
    ax.set_ylabel(label_dict[args.ycol], fontsize=args.fontsize)

    binby_text = '' if args.binby is None else '_binby_' + args.binby
    if args.upto_kpc is not None:
        upto_text = '_upto%.1Fckpchinv' % args.upto_kpc if args.docomoving else '_upto%.1Fkpc' % args.upto_kpc
    else:
        upto_text = '_upto%.1FRe' % args.upto_re

    figname = args.output_dir + 'figs/' + ','.join(args.halo_arr) + '_%s_vs_%s_colorby_%s_Zgrad_den_%s%s%s%s%s.png' % (args.ycol, args.xcol, args.colorcol, args.Zgrad_den, upto_text, args.weightby_text, binby_text, obs_text)
    fig.savefig(figname, transparent=args.glasspaper)
    print('Saved plot as', figname)

    # ------- tidying up fig2 if any------------
    if args.plot_deviation:
        cax = fig2.colorbar(plot2)
        cax.ax.tick_params(labelsize=args.fontsize)
        cax.set_label(label_dict[args.colorcol], fontsize=args.fontsize)

        ax2.set_xlim(args.zmin, args.zmax)
        ax2.set_ylim(-0.02, 0.12)

        ax2.set_xticklabels(['%.1F' % item for item in ax2.get_xticks()], fontsize=args.fontsize)
        ax2.set_yticklabels(['%.2F' % item for item in ax2.get_yticks()], fontsize=args.fontsize)

        ax2.set_xlabel(label_dict[args.zcol], fontsize=args.fontsize)
        ax2.set_ylabel('Residual ' + label_dict[args.ycol], fontsize=args.fontsize)

        figname = args.output_dir + 'figs/' + ','.join(args.halo_arr) + '_dev_in_%s_vs_%s_colorby_%s_Zgrad_den_%s%s%s%s%s.png' % (args.ycol, args.zcol, args.colorcol, args.Zgrad_den, upto_text, args.weightby_text, binby_text, obs_text)
        fig2.savefig(figname)
        print('Saved plot as', figname)
    elif args.plot_timefraction:
        ax2.set_xlim(lim_dict[args.zcol][0], lim_dict[args.zcol][1])
        ax2.set_ylim(-0.5, 0.5)

        ax2.set_xticklabels(['%.1F' % item for item in ax2.get_xticks()], fontsize=args.fontsize)
        ax2.set_yticklabels(['%.2F' % item for item in ax2.get_yticks()], fontsize=args.fontsize)

        ax2.set_xlabel(label_dict[args.zcol], fontsize=args.fontsize)
        ax2.set_ylabel('Residual ' + label_dict[args.ycol], fontsize=args.fontsize)

        figname = args.output_dir + 'figs/' + ','.join(args.halo_arr) + '_timefrac_outside_%.2F_Zgrad_den_%s%s%s%s%s.png' % (args.Zgrad_allowance, args.Zgrad_den, upto_text, args.weightby_text, binby_text, obs_text)
        fig2.savefig(figname)
        print('Saved plot as', figname)
    else:
        fig2 = None

    plt.show(block=False)
    if 'timefraction_outside' not in locals(): timefraction_outside = -99 # dummy value

    return fig, fig2, df_master, df_manga, timefraction_outside

# -----main code-----------------
if __name__ == '__main__':
    args_tuple = parse_args('8508', 'RD0042')  # default simulation to work upon when comand line args not provided
    if type(args_tuple) is tuple: args = args_tuple[0] # if the sim has already been loaded in, in order to compute the box center (via utils.pull_halo_center()), then no need to do it again
    else: args = args_tuple
    if not args.keep: plt.close('all')

    # ---------reading in existing MZgrad txt file------------------
    args.weightby_text = '' if args.weight is None else '_wtby_' + args.weight
    if args.ycol == 'metal': args.ycol = 'Zgrad' # changing the default ycol to metallicity gradient
    if args.xcol == 'rad': args.xcol = 'log_mass' # changing the default xcol to mass, to make a MZGR plot by default when xcol and ycol aren't specified
    if args.colorcol == ['vrad']: args.colorcol = 'time'
    else: args.colorcol = args.colorcol[0]

    fig, fig2, df_binned, df_manga, tfrac = plot_MZGR(args)

    print('Completed in %s' % (datetime.timedelta(minutes=(time.time() - start_time) / 60)))



