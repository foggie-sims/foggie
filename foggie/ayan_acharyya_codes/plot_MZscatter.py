#!/usr/bin/env python3

"""

    Title :      plot_MZscatter
    Notes :      Plot mass - metallicity scatter relation for a given FOGGIE galaxy
    Output :     M-Z scatter plots as png files
    Author :     Ayan Acharyya
    Started :    Aug 2022
    Examples :   run plot_MZscatter.py --system ayan_local --halo 8508,5036,5016,4123 --upto_re 3 --keep --weight mass --res 0.1 --xcol log_mass --binby log_mass --nbins 200 --zhighlight --use_gasre --overplot_smoothed 1500
                 run plot_MZscatter.py --system ayan_local --halo 8508 --upto_kpc 10 --keep --weight mass --res 0.1 --ycol log_Zvar --xcol log_mass --colorcol time --zhighlight --docomoving --fit_multiple
                 run plot_MZscatter.py --system ayan_local --halo 8508 --upto_kpc 10 --keep --weight mass --ycol log_Zvar --xcol log_mass --forpaper
"""
from header import *
from util import *
from matplotlib.collections import LineCollection
from plot_MZgrad import *
from uncertainties import unumpy
start_time = time.time()

# ---------------------------------
def load_df(args):
    '''
    Function to load and return the dataframe containing MZGR
    '''
    args.foggie_dir, args.output_dir, args.run_loc, args.code_path, args.trackname, args.haloname, args.spectra_dir, args.infofile = get_run_loc_etc(args)
    if args.upto_kpc is not None:
        upto_text = '_upto%.1Fckpchinv' % args.upto_kpc if args.docomoving else '_upto%.1Fkpc' % args.upto_kpc
    else:
        upto_text = '_upto%.1FRe' % args.upto_re

    # ---------reading in dataframe produced by compute_Zscatter.py-----------
    dist_filename = args.output_dir + 'txtfiles/' + args.halo + '_MZscat%s%s%s%s%s.txt' % (upto_text, args.weightby_text, args.fitmultiple_text, args.density_cut_text, args.islog_text)
    df = pd.read_table(dist_filename)
    print('Read in file', dist_filename)
    df.drop_duplicates(subset='output', keep='last', ignore_index=True, inplace=True)
    df.rename(columns={'Zvar':'Zsigma', 'Zvar_u':'Zsigma_u', 'gauss_mean':'Z2_mean', 'Zgauss_mean':'Z2_mean', 'gauss_mean_u':'Z2_mean_u', 'Zgauss_mean_u':'Z2_mean_u', 'gauss_sigma':'Z2_sigma', 'Zgauss_sigma':'Z2_sigma', 'gauss_sigma_u':'Z2_sigma_u', 'Zgauss_sigma_u':'Z2_sigma_u', 'Z2_gamma':'Z2_skew', 'Zgauss_gamma':'Z2_skew', 'Z2_gamma_u':'Z2_skew_u', 'Zgauss_gamma_u':'Z2_skew_u'}, inplace=True) # for backward compatibility

    # ---------reading in dataframe produced by compute_MZgrad.py-----------
    Zgrad_den_text = 'rad' if args.upto_kpc is not None else 'rad_re'
    grad_filename = args.output_dir + 'txtfiles/' + args.halo + '_MZR_xcol_%s%s%s%s.txt' % (Zgrad_den_text, upto_text, args.weightby_text, args.density_cut_text)
    df2 = pd.read_table(grad_filename)
    print('Read in file', grad_filename)
    df2.drop_duplicates(subset='output', keep='last', ignore_index=True, inplace=True)

    # ---------merging both dataframes-----------
    df = df.merge(df2[['output', 'Zcen_fixedr', 'Zgrad_fixedr', 'Zgrad_u_fixedr', 'Zcen_binned_fixedr', 'Zgrad_binned_fixedr', 'Zgrad_u_binned_fixedr', 'Ztotal_fixedr']], on='output')
    cols_to_rename = ['Zcen_fixedr', 'Zgrad_fixedr', 'Zgrad_u_fixedr', 'Zcen_binned_fixedr', 'Zgrad_binned_fixedr', 'Zgrad_u_binned_fixedr']
    df = df.rename(columns=dict(zip(cols_to_rename, [item[:-7] for item in cols_to_rename])))

    df.sort_values(by='redshift', ascending=False, ignore_index=True, inplace=True)

    #if 'res' in df: df = df[df['res'] == -99 if args.get_native_res else float(args.res)]

    cols_to_log = ['Zpeak', 'Z25', 'Z50', 'Z75', 'Zmean', 'Zsigma', 'Ztotal', 'Zcen', 'Zcen_binned', 'Ztotal_fixedr', 'Z2_mean', 'Z2_sigma']
    for thiscol in cols_to_log:
        if thiscol in df:
            if args.islog:
                if thiscol + '_u' in df and (df[thiscol + '_u']!=0).any(): # need to propagate uncertainties properly
                    #df = df[(df[thiscol + '_u'] >= 0) & (np.abs(df[thiscol]/df[thiscol + '_u']).between(1e-1, 1e5))] # remove negative errors and errors that are way too high compared to the measured value; something is wrong there
                    df = df[(np.isnan(df[thiscol])) | ((df[thiscol + '_u'] >= 0) & (df[thiscol] < 1e2))] # remove negative errors and large numbers given it is already in log
                    quant = unumpy.pow(10, unumpy.uarray(df[thiscol].values, df[thiscol + '_u'].values))
                    df.rename(columns={thiscol:'log_' + thiscol, thiscol+'_u':'log_' + thiscol + '_u'}, inplace=True) # column was already in log
                    df[thiscol], df[thiscol + '_u'] = unumpy.nominal_values(quant), unumpy.std_devs(quant)
                else: # no uncertainties available, makes life simpler
                    df.rename(columns={thiscol:'log_' + thiscol}, inplace=True) # column was already in log
                    df[thiscol] = 10**df['log_' + thiscol]
            else:
                if thiscol + '_u' in df and (df[thiscol + '_u']!=0).any(): # need to propagate uncertainties properly
                    df = df[(np.isnan(df[thiscol])) | ((df[thiscol + '_u'] >= 0) & (np.abs(df[thiscol]/df[thiscol + '_u']).between(1e-1, 1e5)))] # remove negative errors and errors that are way too high compared to the measured value; something must be wrong there
                    quant = unumpy.log10(unumpy.uarray(df[thiscol].values, df[thiscol + '_u'].values))
                    df['log_' + thiscol], df['log_' + thiscol + '_u'] = unumpy.nominal_values(quant), unumpy.std_devs(quant)
                else: # no uncertainties available, makes life simpler
                    df['log_' + thiscol] = np.log10(df[thiscol])
        else:
            print(thiscol, 'column not found in dataframe, putting dummy values')
            df['log_' + thiscol] = -99

    df['ZIQR'] = df['Z75'] - df['Z25'] # IQR in linear space; hence performing the subtraction AFTER the Z75 and Z25 columns have been un-logged
    df['log_ZIQR'] = np.log10(df['Z75']) - np.log10(df['Z25']) # log IQR is the width in log-space; hence performing the subtraction AFTER the Z75 and Z25 columns have been converted to log
    df['Zwidth'] = 2.355 * df['Zsigma']
    df['Zwidth_u'] = 2.355 * df['Zsigma_u']
    quant = unumpy.log10(unumpy.uarray(df['Zwidth'].values, df['Zwidth_u'].values))
    df['log_Zwidth'], df['log_Zwidth_u'] = unumpy.nominal_values(quant), unumpy.std_devs(quant)
    for thiscol in ['mass']: df['log_' + thiscol] = np.log10(df[thiscol])

    return df

# -----------------------------------
def plot_MZscatter(args):
    '''
    Function to plot the mass-metallicity scatter relation, based on an input dataframe
    '''

    df_master = pd.DataFrame()
    cmap_arr = ['Purples', 'Oranges', 'Greens', 'Blues', 'PuRd', 'YlOrBr']
    things_that_reduce_with_time = ['redshift', 're'] # whenever this quantities are used as colorcol, the cmap is inverted, so that the darkest color is towards later times

    if args.xmin is None: args.xmin = lim_dict[args.xcol][0]
    if args.xmax is None: args.xmax = lim_dict[args.xcol][1]
    if args.ymin is None: args.ymin = lim_dict[args.ycol][0]
    if args.ymax is None: args.ymax = lim_dict[args.ycol][1]
    if args.cmin is None: args.cmin = lim_dict[args.colorcol][0]
    if args.cmax is None: args.cmax = lim_dict[args.colorcol][1]
    if args.zmin is None: args.zmin = lim_dict[args.zcol][0]
    if args.zmax is None: args.zmax = lim_dict[args.zcol][1]

    # -------declare figure object-------------
    fig, ax = plt.subplots(1, figsize=(12, 6))
    fig.subplots_adjust(top=0.95, bottom=0.15, left=0.1, right=0.97 if args.nocolorcoding else 1.05)

    if args.plot_deviation:
        fig2, ax2 = plt.subplots(1, figsize=(12, 6))
        fig2.subplots_adjust(top=0.95, bottom=0.15, left=0.1, right=1.05)
        args.overplot_smoothed = True

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
        df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
        df['log_ssfr'] = np.log10(df['ssfr'])
        df['log_sfr'] = np.log10(df['sfr'])
        df = df.sort_values(args.xcol)

        #df = df[(df[args.xcol] >= args.xmin) & (df[args.xcol] <= args.xmax)]
        #df = df[(df[args.ycol] >= args.ymin) & (df[args.ycol] <= args.ymax)]
        df = df[(df[args.colorcol] >= args.cmin) & (df[args.colorcol] <= args.cmax)]
        df = df.dropna(subset=[args.xcol, args.ycol, args.colorcol], axis=0)

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
        if args.nocolorcoding:
            ax.plot(df[args.xcol], df[args.ycol], c=thistextcolor, lw=1 if args.overplot_literature else 2, zorder=27 if args.fortalk and not args.plot_timefraction else 2)
        else:
            line = get_multicolored_line(df[args.xcol], df[args.ycol], df[args.colorcol], this_cmap, args.cmin, args.cmax, lw=1 if args.overplot_smoothed else 2)
            plot = ax.add_collection(line)
        if args.overplot_points: ax.scatter(df[args.xcol], df[args.ycol], c=thistextcolor, lw=0.5, s=10)

        # ------- overplotting specific snapshot highlights------------
        if args.snaphighlight is not None:
            snaps_to_highlight = [item for item in args.snaphighlight.split(',')]
            df_snaps = df[df['output'].isin(snaps_to_highlight)]
            if args.nocolorcoding: dummy = ax.scatter(df_snaps[args.xcol], df_snaps[args.ycol], c=thistextcolor, lw=1, edgecolor='gold' if args.fortalk else 'k', s=300, alpha=1, marker='*', zorder=10)
            else: dummy = ax.scatter(df_snaps[args.xcol], df_snaps[args.ycol], c=df_snaps[args.colorcol], cmap=this_cmap, vmin=args.cmin, vmax=args.cmax, lw=1, edgecolor='gold' if args.fortalk else 'k', s=300, alpha=1, marker='*', zorder=10)
            print('For halo', args.halo, 'highlighted snapshots =', df_snaps['output'].values, ' with star-markers\nThese snapshots correspond to times', df_snaps['time'].values, 'Gyr respectively, i.e.,', np.diff(df_snaps['time'].values) * 1000, 'Myr apart')

        # ------- overplotting redshift-binned scatter plot------------
        if args.zhighlight:
            ax = plot_zhighlight(df, ax, thistextcolor if args.nocolorcoding else this_cmap, args)

        # ------- overplotting a boxcar smoothed version of the MZGR------------
        if args.overplot_smoothed:
            mean_dt = (df['time'].max() - df['time'].min())*1000/len(df) # Myr
            npoints = int(np.round(args.overplot_smoothed/mean_dt))
            if npoints % 2 == 0: npoints += 1
            box = np.ones(npoints) / npoints
            df[args.ycol + '_smoothed'] = np.convolve(df[args.ycol], box, mode='same')

            if args.nocolorcoding:
                ax.plot(df[args.xcol], df[args.ycol + '_smoothed'], c=thistextcolor, lw=0.5)
            else:
                smoothline = get_multicolored_line(df[args.xcol], df[args.ycol + '_smoothed'], df[args.colorcol], this_cmap, args.cmin, args.cmax, lw=2)
                plot = ax.add_collection(smoothline)
                line.set_alpha(0.2) # make the actual wiggly line fainter
            print('Boxcar-smoothed plot for halo', args.halo, 'with', npoints, 'points, =', npoints * mean_dt, 'Myr')

        # ------- overplotting a lower cadence version of the MZGR------------
        if args.overplot_cadence:
            mean_dt = (df['time'].max() - df['time'].min())*1000/len(df) # Myr
            npoints = int(np.round(args.overplot_cadence/mean_dt))
            df_short = df.iloc[::npoints, :]
            print('Overplot for halo', args.halo, 'only every', npoints, 'th data point, i.e. cadence of', npoints * mean_dt, 'Myr')

            yfunc = interp1d(df_short[args.xcol], df_short[args.ycol], fill_value='extrapolate') # interpolating the low-cadence data
            cfunc = interp1d(df_short[args.xcol], df_short[args.colorcol], fill_value='extrapolate')
            df[args.ycol + '_interp'] = yfunc(df[args.xcol])
            df[args.colorcol + '_interp'] = cfunc(df[args.xcol])
            if args.nocolorcoding:
                ax.plot(df[args.xcol], df[args.ycol + '_interp'], c=thistextcolor, lw=0.5)
            else:
                newline = get_multicolored_line(df_short[args.xcol], df_short[args.ycol + '_interp'], df_short[args.colorcol + '_interp'], this_cmap, args.cmin, args.cmax, lw=2)
                plot = ax.add_collection(newline)

        # ------- making additional plot of deviation in gradient vs other quantities, like SFR------------
        if args.plot_deviation:
            print('Plotting deviation vs', args.colorcol, 'halo', args.halo)
            df[args.ycol + '_deviation'] = df[args.ycol] - df[args.ycol + '_smoothed']
            df = df.sort_values(args.zcol)

            # --------- scatter plot------------
            #ax2.scatter(df[args.zcol], df[args.ycol + '_deviation'], c=thistextcolor, edgecolor='k', lw=0.5, s=50)

            # --------- colored line plot------------
            if args.nocolorcoding:
                ax.plot(df[args.zcol], df[args.ycol + '_deviation'], c=thistextcolor, lw=1)
            else:
                line2 = get_multicolored_line(df[args.zcol], np.abs(df[args.ycol + '_deviation']), df[args.colorcol], reversed_thiscmap, args.cmin, args.cmax, lw=1)
                plot2 = ax2.add_collection(line2)

                # --------- smoothed colored line plot------------
                line2.set_alpha(0.2) # make the actual wiggly line fainter

            df[args.ycol + '_deviation_smoothed'] = np.convolve(np.abs(df[args.ycol + '_deviation']), box, mode='same')
            if args.nocolorcoding:
                ax.plot(df[args.zcol], df[args.ycol + '_deviation_smoothed'], c=thistextcolor, lw=2)
            else:
                smoothline2 = get_multicolored_line(df[args.zcol], df[args.ycol + '_deviation_smoothed'], df[args.colorcol], reversed_thiscmap, args.cmin, args.cmax, lw=2)
                plot2 = ax2.add_collection(smoothline2)

        fig.text(0.15, 0.9 - thisindex * 0.05, halo_dict[args.halo], ha='left', va='top', color=thistextcolor, fontsize=args.fontsize)
        if args.plot_deviation: fig2.text(0.15, 0.9 - thisindex * 0.05, halo_dict[args.halo], ha='left', va='top', color=thistextcolor, fontsize=args.fontsize)
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

    figname = args.output_dir + 'figs/' + ','.join(args.halo_arr) + '_%s_vs_%s_colorby_%s_res%.2Fkpc%s%s%s%s%s.png' % (args.ycol, args.xcol, args.colorcol, float(args.res), upto_text, args.weightby_text, binby_text, args.density_cut_text, args.islog_text)
    fig.savefig(figname)
    print('Saved plot as', figname)

    # ------- tidying up fig2 if any------------
    if args.plot_deviation:
        cax = fig2.colorbar(plot2)
        cax.ax.tick_params(labelsize=args.fontsize)
        cax.set_label(label_dict[args.colorcol], fontsize=args.fontsize)

        ax2.set_xlim(args.zmin, args.zmax)
        ax2.set_ylim(-0.01, 0.15)

        ax2.set_xticklabels(['%.1F' % item for item in ax2.get_xticks()], fontsize=args.fontsize)
        ax2.set_yticklabels(['%.2F' % item for item in ax2.get_yticks()], fontsize=args.fontsize)

        ax2.set_xlabel(label_dict[args.zcol], fontsize=args.fontsize)
        ax2.set_ylabel('Deviation in ' + label_dict[args.ycol], fontsize=args.fontsize)

        figname = args.output_dir + 'figs/' + ','.join(args.halo_arr) + '_dev_in_%s_vs_%s_colorby_%s_res%.2Fkpc%s%s%s%s%s.png' % (args.ycol, args.zcol, args.colorcol, float(args.res), upto_text, args.weightby_text, binby_text, args.density_cut_text, args.islog_text)
        fig2.savefig(figname)
        print('Saved plot as', figname)
    else:
        fig2 = None

    plt.show(block=False)
    return fig, fig2, df_master


# -------------get plot limits-----------------
lim_dict = {'log_Zpeak': (-2, 0.1), 'log_Z50': (-2, 0.1), 'log_Zmean': (-2, 0.1), 'log_Zvar': (-2, 0.1),
            'Zskew': (-2, 0.1), 're': (0, 30), 'log_mass': (8.5, 11.5), 'redshift': (0, 6), 'time': (0, 14),
            'sfr': (0, 60), 'log_ssfr': (-11, -8), 'Ztotal': (8, 9), 'log_sfr': (-1, 3)}
label_dict = MyDefaultDict(re='Scale length (kpc)', log_mass=r'$\log{(\mathrm{M}_*/\mathrm{M}_\odot)}$', redshift='Redshift', time='Time (Gyr)', sfr=r'SFR (M$_{\odot}$/yr)', \
    log_ssfr=r'$\log{\, \mathrm{sSFR} (\mathrm{yr}^{-1})}$', Ztotal=r'$\log{(\mathrm{O/H})}$ + 12', log_sfr=r'$\log{(\mathrm{SFR} (\mathrm{M}_{\odot}/yr))}$')

# -----main code-----------------
if __name__ == '__main__':
    args_tuple = parse_args('8508', 'RD0042')  # default simulation to work upon when comand line args not provided
    if type(args_tuple) is tuple: args = args_tuple[0] # if the sim has already been loaded in, in order to compute the box center (via utils.pull_halo_center()), then no need to do it again
    else: args = args_tuple
    if not args.keep: plt.close('all')

    # ---------preset values for plotting for paper-------------
    if args.fortalk:
        setup_plots_for_talks()
        args.forpaper = True

    if args.forpaper or args.fortalk:
        args.res = 0.1 # kpc
        args.docomoving = True
        args.fit_multiple = True
        args.nocolorcoding = True
        args.zhighlight = True
        args.get_native_res = True
        args.use_density_cut = True
        args.islog = True


    # ---------reading in existing MZgrad txt file------------------
    args.weightby_text = '' if args.weight is None else '_wtby_' + args.weight
    args.fitmultiple_text = '_fitmultiple' if args.fit_multiple else ''
    args.density_cut_text = '_wdencut' if args.use_density_cut else ''
    args.islog_text = '_islog' if args.islog else ''
    if args.ycol == 'metal': args.ycol = 'log_Zvar' # changing the default ycol to metallicity gradient
    if args.xcol == 'rad': args.xcol = 'log_mass' # changing the default xcol to mass, to make a MZGR plot by default when xcol and ycol aren't specified
    if args.colorcol == ['vrad']: args.colorcol = 'time'
    else: args.colorcol = args.colorcol[0]

    fig, fig2, df_binned = plot_MZscatter(args)

    print('Completed in %s' % (datetime.timedelta(minutes=(time.time() - start_time) / 60)))



