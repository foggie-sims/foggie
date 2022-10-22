#!/usr/bin/env python3

"""

    Title :      plot_Zevolution
    Notes :      Plot evolution of metallicity distribution for a given FOGGIE galaxy
    Output :     Various Z statistics vs time plots as png files
    Author :     Ayan Acharyya
    Started :    Aug 2022
    Examples :   run plot_Zevolution.py --system ayan_local --halo 8508,5036,5016,4123 --upto_re 3 --keep --weight mass --res 0.1 --zhighlight --use_gasre --overplot_smoothed
                 run plot_Zevolution.py --system ayan_local --halo 8508 --upto_kpc 10 --keep --weight mass --res 0.1 --zhighlight --docomoving --fit_multiple
                 run plot_Zevolution.py --system ayan_pleiades --halo 8508 --upto_kpc 10 --keep --weight mass --res 0.1 --zhighlight --docomoving --fit_multiple
                 run plot_Zevolution.py --system ayan_local --halo 8508 --upto_kpc 10 --weight mass --forpaper [FOR THE PAPER]
"""
from header import *
from util import *
from plot_MZscatter import *
import h5py
from scipy.signal import argrelextrema
from scipy.fft import rfft, rfftfreq
start_time = time.time()

# -----------------------------------
def plot_all_stats(df, args):
    '''
    Function to plot the time evolution of Z distribution statistics, based on an input dataframe
    '''
    fig, axes = plt.subplots(6 + 1, figsize=(12, 10), sharex=True)
    fig.subplots_adjust(top=0.98, bottom=0.07, left=0.07, right=0.92, hspace=0.05)

    df = df.sort_values(by='time')
    col_arr = ['darkolivegreen', 'brown', 'black', 'cornflowerblue', 'salmon', 'gold', 'saddlebrown', 'crimson',
               'black', 'darkturquoise', 'lawngreen']
    sfr_col_arr = ['cornflowerblue', 'salmon']  # ['black', 'brown']

    groups = pd.DataFrame({'quantities': [['Z25', 'Z50', 'Z75'], ['Zskew', 'Zmean', 'Zvar'],
                                          ['gauss_mean', 'gauss_sigma'], ['Ztotal', 'Ztotal_fixedr'],
                                          ['Zgrad', 'Zgrad_binned'], ['Zcen', 'Zcen_binned']], \
                           'label': np.hstack([np.tile([r'$\log{(\mathrm{Z}/\mathrm{Z}_\odot)}$'], 4),
                                               [r'$\Delta Z$ (dex/kpc)', r'$\log{(\mathrm{Z}/\mathrm{Z}_\odot)}$']]), \
                           'limits': [(-3, 1), (-3, 1), (-3, 1), (-1, 1), (-0.6, 0), (-3, 1)], \
                           'isalreadylog': np.hstack([np.tile([True], 4), [False, True]])})

    # -----------for first few panels: Z distribution statistics-------------------
    for j in range(len(groups)):
        thisgroup = groups.iloc[j]
        ax = axes[j]
        log_text = 'log_' if thisgroup.isalreadylog else ''
        for i, ycol in enumerate(thisgroup.quantities):
            ax.plot(df['time'], df[log_text + ycol], c=col_arr[i], lw=0.5 if args.overplot_smoothed else 1,
                    alpha=0.3 if args.overplot_smoothed or ycol == 'Zskew' else 1,
                    label=None if args.overplot_smoothed else ycol)

            if args.overplot_smoothed:
                npoints = int(len(df) / 8)
                if npoints % 2 == 0: npoints += 1
                box = np.ones(npoints) / npoints
                df[log_text + ycol + '_smoothed'] = np.convolve(df[log_text + ycol], box, mode='same')
                ax.plot(df['time'], df[log_text + ycol + '_smoothed'], c=col_arr[i], lw=2, label=ycol)

        ax.legend(loc='upper left', fontsize=args.fontsize / 1.5)
        ax.set_ylabel(thisgroup.label, fontsize=args.fontsize)
        ax.set_ylim(thisgroup.limits)
        ax.tick_params(axis='y', labelsize=args.fontsize)

        ax2 = ax.twinx()
        if 'sfr' in df: ax2.plot(df['time'], df['sfr'], c=sfr_col_arr[0], lw=1, alpha=0.2)
        ax2.set_ylim(0, 50)
        ax2.set_yticks([])

        ax3 = ax.twinx()
        if 'log_ssfr' in df: ax3.plot(df['time'], df['log_ssfr'], c=sfr_col_arr[1], lw=1, alpha=0.2)
        ax3.set_ylim(-12, -7)
        ax3.set_yticks([])

    # -----------for last panel first part: SFR-------------------
    if 'sfr' in df: axes[-1].plot(df['time'], df['sfr'], c=sfr_col_arr[0], lw=1)

    axes[-1].set_ylabel(label_dict['sfr'], fontsize=args.fontsize, color=sfr_col_arr[0])
    axes[-1].set_ylim(0, 50)
    axes[-1].tick_params(axis='y', colors=sfr_col_arr[0], labelsize=args.fontsize)

    axes[-1].set_xlabel('Time (Gyr)', fontsize=args.fontsize)
    axes[-1].set_xlim(0, 14)
    #axes[-1].set_xlim(1.5, 2)
    axes[-1].tick_params(axis='x', labelsize=args.fontsize)

    # -----------for last panel second part: sSFR-------------------
    ax2 = axes[-1].twinx()
    if 'log_ssfr' in df: ax2.plot(df['time'], df['log_ssfr'], c=sfr_col_arr[1], lw=1)

    ax2.set_ylabel(label_dict['log_ssfr'], fontsize=args.fontsize, color=sfr_col_arr[1])
    ax2.set_ylim(-12, -7)
    ax2.tick_params(axis='y', colors=sfr_col_arr[1], labelsize=args.fontsize)

    if args.upto_kpc is not None:
        upto_text = '_upto%.1Fckpchinv' % args.upto_kpc if args.docomoving else '_upto%.1Fkpc' % args.upto_kpc
    else:
        upto_text = '_upto%.1FRe' % args.upto_re

    figname = args.output_dir + 'figs/' + ','.join(args.halo_arr) + '_allstats_vs_time_res%.2Fkpc%s%s.pdf' % (float(args.res), upto_text, args.weightby_text)
    fig.savefig(figname)
    print('Saved', figname)
    plt.show(block=False)

    return df, fig

# -----------------------------------------------
def plot_zhighlight(df, ycol, color, ax, args):
    '''
    Function to overplot circles at integer-ish redshifts and return the ax
    '''
    df['redshift_int'] = np.floor(df['redshift'])
    df_zbin = df.drop_duplicates(subset='redshift_int', keep='last', ignore_index=True)
    dummy = ax.scatter(df_zbin['time'], df_zbin[ycol], c=color, lw=1, edgecolor='k', s=100, alpha=0.5, zorder=20)
    print('For halo', args.halo, 'highlighted z =', [float('%.1F' % item) for item in df_zbin['redshift'].values], 'with circles')
    return ax

# -----------------------------------
def plot_time_series(df, args):
    '''
    Function to plot the time evolution of Z distribution statistics, based on an input dataframe
    '''
    fig, axes = plt.subplots(5 if args.forpaper else 7, figsize=(8, 10), sharex=True)
    fig.subplots_adjust(top=0.98, bottom=0.07, left=0.11, right=0.98, hspace=0.05)

    if args.doft:
        fig2, axes2 = plt.subplots(5 if args.forpaper else 7, figsize=(8, 10), sharex=True)
        fig2.subplots_adjust(top=0.98, bottom=0.07, left=0.11, right=0.98, hspace=0.05)
        freq_arr = rfftfreq(len(df), np.median(np.diff(df['time'].values))) # Gyr^-1

    df = df.sort_values(by='time')
    df['ZIQR_rel'] = df['ZIQR'] / df['Z50'] # IQR relative to the median Z

    col_arr = ['brown', 'darkolivegreen', 'black', 'cornflowerblue', 'salmon', 'gold', 'saddlebrown', 'crimson',
               'black', 'darkturquoise', 'lawngreen']
    sfr_col_arr = ['black']

    groups = pd.DataFrame({'quantities': [['Zgrad', 'Zgrad_binned'], ['Z50', 'ZIQR'], ['Zmean', 'Zvar']], \
                           'legend': [['All cells fitted', 'Radial bins fitted'], ['Median Z', 'IQR'], ['Mean Z (fit)', 'Variance (fit)']], \
                           'label': np.hstack([r'$\Delta Z$ (dex/kpc)', np.tile([r'Z/Z$_\odot$'], 2)]), \
                           'limits': [(-0.5, 0.1), (1e-3, 8), (1e-4, 2)], \
                           'isalreadylog': np.hstack([np.tile([False], 3)])})

    # -----------for first few panels: Z distribution statistics-------------------
    for j in range(len(groups)):
        thisgroup = groups.iloc[j]
        ax = axes[j]
        if args.doft: ax2 = axes2[j]
        log_text = 'log_' if thisgroup.isalreadylog else ''
        for i, ycol in enumerate(thisgroup.quantities):
            ax.plot(df['time'], df[log_text + ycol], c=col_arr[i], lw=1, label=thisgroup.legend[i])
            if args.zhighlight: ax = plot_zhighlight(df,log_text + ycol, col_arr[i], ax, args)
            # -------for the FT plot--------------
            if args.doft:
                yft_arr = np.abs(rfft(df[ycol].values))
                ax2.plot(freq_arr, np.log10(yft_arr), c=col_arr[i], lw=0.5 if args.overplot_smoothed else 1, label=thisgroup.legend[i], alpha=0.3 if args.overplot_smoothed else 1)

                if args.overplot_smoothed:
                    npoints = int(len(df) / 100)
                    if npoints % 2 == 0: npoints += 1
                    box = np.ones(npoints) / npoints
                    yft_arr_smoothed = np.convolve(yft_arr, box, mode='same')
                    ax2.plot(freq_arr, np.log10(yft_arr_smoothed), c=col_arr[i], lw=2)


        ax.legend(loc='upper left', fontsize=args.fontsize / 1.5)
        ax.set_ylabel(thisgroup.label, fontsize=args.fontsize)
        ax.set_ylim(thisgroup.limits)
        ax.tick_params(axis='y', labelsize=args.fontsize)

    axiscount = len(groups)
    
    # -----------for SFR panel-------------------
    ycol = 'sfr'
    thisax = axes[axiscount]
    if args.doft: thisax2 = axes2[axiscount]
    if ycol in df:
        thisax.plot(df['time'], df[ycol], c=col_arr[0], lw=1, label='SFR')
        if args.zhighlight: thisax = plot_zhighlight(df, ycol, col_arr[0], thisax, args)
        # -------for the FT plot--------------
        if args.doft:
            yft_arr = np.abs(rfft(df[ycol].values))
            thisax2.plot(freq_arr, np.log10(yft_arr), c=col_arr[0], lw=1, label='SFR')

    thisax.set_ylabel(label_dict[ycol], fontsize=args.fontsize)
    thisax.set_ylim(0, 50)
    thisax.tick_params(axis='y', labelsize=args.fontsize)
    thisax.legend(loc='upper left', fontsize=args.fontsize / 1.5)
    axiscount += 1 

    if not args.forpaper:
        # -----------for metal production panel-------------------
        thisax = axes[axiscount]
        if 'log_metal_produced' in df:
            thisax.plot(df['time'], df['log_metal_produced'], c=col_arr[0], lw=1, label='Metal mass produced')
            if args.zhighlight: thisax = plot_zhighlight(df, 'log_metal_produced', col_arr[0], thisax, args)

        thisax.set_ylabel(label_dict['log_mass'], fontsize=args.fontsize)
        thisax.set_ylim(4, 7.2)
        thisax.tick_params(axis='y', labelsize=args.fontsize)
        thisax.legend(loc='upper left', fontsize=args.fontsize / 1.5)
        axiscount += 1

        # -----------for metal ejection panel-------------------
        thisax = axes[axiscount]
        if 'log_metal_flux' in df:
            thisax.plot(df['time'], df['log_metal_flux'], c=col_arr[0], lw=1, label='Metal flux ejected')
            if args.zhighlight: thisax = plot_zhighlight(df, 'log_metal_flux', col_arr[0], thisax, args)

        thisax.set_ylabel(r'$\log{(\mathrm{M}_{\odot}/\mathrm{yr})}$', fontsize=args.fontsize)
        thisax.set_ylim(-5, 0)
        thisax.tick_params(axis='y', labelsize=args.fontsize)
        thisax.legend(loc='upper left', fontsize=args.fontsize / 1.5)
        axiscount += 1

    # -----------for merger history panel-------------------
    thisax = axes[axiscount]
    rvir_df = pd.read_hdf(args.code_path + '/halo_infos/00' + args.halo + '/nref11c_nref9f/rvir_masses.hdf5', key='all_data') ## args.run is hard-coded here, for noe
    df2 = df[['output', 'redshift', 'time']].merge(rvir_df[['redshift', 'radius', 'stars_mass']], on='redshift')

    filename = args.code_path + 'satellites/Tempest_satorbits.hdf5'
    f = h5py.File(filename, 'r')

    ## ----set up the colorbar parameters------
    cmap_name, log_mass_min, log_mass_max = 'plasma', 9, 10
    norm = mplcolors.Normalize(vmin=log_mass_min, vmax=log_mass_max)
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_name)

    for index, thissat in enumerate(f.keys()):
        #print('Doing', thissat, 'which is', index+1, 'out of', len(f.keys()), 'satellites..')
        thisdf = pd.DataFrame({'time':np.array(f[thissat]['Time(Gyr)'][()]).flatten(), 'dist':np.array(f[thissat]['Dist(R200)'][()]).flatten()})
        thisdf = df2.merge(thisdf, on='time', how='inner') # need to merge with main df in order to get rvir info corresponding to each DD output
        thisdf['dist'] = thisdf['dist'] * thisdf['radius'] # converting Distance from R200 units to kpc units

        this_log_mass = f[thissat]['log10(MaxMass)'][()]
        this_tpeak = f[thissat]['Tpeak(Gyr)'][()]

        host_tpeak = df2.iloc[(df2['time']-this_tpeak).abs().argsort()]['time'].values[0] # find tpeak in host df that is closest to this_tpeak
        host_mass_tpeak = df2[df2['time'] == host_tpeak]['stars_mass'].values[0] # find host mass at closest time stamp to this_tpeak
        thisdf['log_rel_mass'] = this_log_mass - np.log10(host_mass_tpeak) # log(A/B) = log(A) - log(B)

        minima_indices = argrelextrema(thisdf['dist'].values, np.less)[0]
        thisdf = thisdf[thisdf.index.isin(minima_indices)]

        if len(thisdf) > 0:
            thisax.scatter(thisdf['time'], thisdf['log_rel_mass'], s=2e3/thisdf['dist'], color=cmap.to_rgba(this_log_mass), lw=0.5, edgecolor='k')
    f.close()

    thisax.set_ylabel(r'$\log{(\mathrm{M}_{\mathrm{sat,t}_{\mathrm{peak}}}/\mathrm{M}_{\mathrm{host,t}_{\mathrm{peak}}})}$', fontsize=args.fontsize)
    thisax.set_ylim(-2, 0.5)
    thisax.tick_params(axis='y', labelsize=args.fontsize)
    thisax.text(13.8, 0, 'Larger markers indicate closer passes', color='k', ha='right', va='bottom', fontsize=args.fontsize / 1.5)

    ## ------set up the colorbar axis--------------
    dummy_ax = fig.add_axes([0.2, 0.2, 0.5, 0.5]) # dummy axis, dimensions do not matter
    img = dummy_ax.imshow(np.array([[log_mass_min, log_mass_max]]), cmap=cmap_name, vmin=log_mass_min, vmax=log_mass_max)
    dummy_ax.set_visible(False)
    cax = fig.add_axes([0.13, 0.08, 0.01, 0.15]) # this is the dimension of the cbar axis [left, bottom, width, height]
    plt.colorbar(img, orientation='vertical', cax=cax)
    cax.set_ylabel(r'$\log{(\mathrm{M/M}_\odot)}$', fontsize=args.fontsize / 1.5)

    # -----------for x axis-------------------
    thisax.set_xlabel('Time (Gyr)', fontsize=args.fontsize)
    thisax.set_xlim(0, 14)
    thisax.tick_params(axis='x', labelsize=args.fontsize)

    # -----------for all y axes of the FT plot-----------
    if args.doft:
        for fax in axes2:
            fax.legend(loc='upper left', fontsize=args.fontsize / 1.5)
            fax.set_ylabel('log(power spec)', fontsize=args.fontsize)
            # fax.set_ylim(thisgroup.limits)
            fax.tick_params(axis='y', labelsize=args.fontsize)
        thisax = axes2[4]
        thisax.set_xlabel(r'Frequency (Gyr$^{-1}$)', fontsize=args.fontsize)
        thisax.set_xlim(0.08/2, 190/2)
        thisax.tick_params(axis='x', labelsize=args.fontsize)


    if args.upto_kpc is not None:
        upto_text = '_upto%.1Fckpchinv' % args.upto_kpc if args.docomoving else '_upto%.1Fkpc' % args.upto_kpc
    else:
        upto_text = '_upto%.1FRe' % args.upto_re

    figname = args.output_dir + 'figs/' + ','.join(args.halo_arr) + '_timeseries_res%.2Fkpc%s%s.pdf' % (float(args.res), upto_text, args.weightby_text)
    fig.savefig(figname)
    print('Saved', figname)
    if args.doft:
        figname2 = figname.replace('timeseries', 'fourier')
        fig2.savefig(figname2)
        print('Saved', figname2)

    plt.show(block=False)

    return df, fig

# -----main code-----------------
if __name__ == '__main__':
    args_tuple = parse_args('8508', 'RD0042')  # default simulation to work upon when comand line args not provided
    if type(args_tuple) is tuple: args = args_tuple[0] # if the sim has already been loaded in, in order to compute the box center (via utils.pull_halo_center()), then no need to do it again
    else: args = args_tuple
    if not args.keep: plt.close('all')

    # ---------preset values for plotting for paper-------------
    if args.forpaper:
        args.res = 0.1 # kpc
        args.zhighlight = True
        args.docomoving = True
        args.fit_multiple = True

    # ---------reading in existing MZgrad txt file------------------
    args.weightby_text = '' if args.weight is None else '_wtby_' + args.weight
    args.fitmultiple_text = '_fitmultiple' if args.fit_multiple else ''
    if args.upto_kpc is not None:
        upto_text = '_upto%.1Fckpchinv' % args.upto_kpc if args.docomoving else '_upto%.1Fkpc' % args.upto_kpc
    else:
        upto_text = '_upto%.1FRe' % args.upto_re
    output_filename = args.output_dir + 'txtfiles/' + args.halo + '_timeseries%s%s%s.txt' % (upto_text, args.weightby_text, args.fitmultiple_text)

    if not os.path.exists(output_filename) or args.clobber:
        print(output_filename, 'does not exist, merging dataframes to create it..')

        cmap_arr = ['Purples', 'Oranges', 'Greens', 'Blues', 'PuRd', 'YlOrBr']
        things_that_reduce_with_time = ['redshift', 're'] # whenever this quantities are used as colorcol, the cmap is inverted, so that the darkest color is towards later times

        df = load_df(args)

        # -------- reading in and merging dataframe with SFR info-------
        sfr_filename = args.code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/sfr'
        if os.path.exists(sfr_filename):
            print('Reading SFR history from', sfr_filename)
            sfr_df = pd.read_table(sfr_filename, names=('output', 'redshift', 'sfr'), comment='#', delim_whitespace=True)
            df = df.merge(sfr_df[['output', 'sfr']], on='output')
            df['ssfr'] = df['sfr'] / 10 ** df['log_mass']
            df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
            df['log_ssfr'] = np.log10(df['ssfr'])
            df['log_sfr'] = np.log10(df['sfr'])
        else:
            print('Did not find', sfr_filename, ', therefore will not plot the SFR-related panels')

        # -------- reading in and merging dataframe with metal flux info-------
        flux_df = pd.DataFrame(columns=('output', 'log_metal_flux'))
        flux_file_path = args.output_dir + 'txtfiles/'
        flux_files = glob.glob(flux_file_path + '*_rad%.1Fkpc_nchunks%d_fluxes_mass.hdf5'%(args.galrad, args.nchunks))

        if len(flux_files) > 0:
            for i, thisfile in enumerate(flux_files):
                print('Extracting metal flux from file', thisfile, 'which is', i+1, 'out of', len(flux_files), 'files..')
                output = os.path.split(thisfile)[-1][:6]
                thisdf = pd.read_hdf(thisfile, 'all_data')
                net_metal_flux = thisdf['net_metal_flux'][np.where(thisdf['radius'] >= args.upto_kpc)[0][0]] # msun/yr
                flux_df.loc[len(flux_df)] = [output, np.log10(net_metal_flux)] # log(msun/yr)

            df = df.merge(flux_df[['output', 'log_metal_flux']], on='output')
        else:
            print('Did not find any metal flux file, therefore will not plot the metal flux panels')

        # -------- reading in and merging dataframe with SFR info-------
        production_df = pd.DataFrame(columns=('output', 'log_metal_produced'))
        production_files = glob.glob(flux_file_path + '*_rad%.1Fkpc_nchunks%d_metal_sink_source.txt'%(args.galrad, args.nchunks))

        if len(production_files) > 0:
            for i, thisfile in enumerate(production_files):
                print('Extracting metal production from file', thisfile, 'which is', i+1, 'out of', len(flux_files), 'files..')
                output = os.path.split(thisfile)[-1][:6]
                thisdf = pd.read_table(thisfile, delim_whitespace=True, comment='#')
                metal_produced = thisdf['metal_produced'][:np.where(thisdf['radius'] >= args.upto_kpc)[0][0]+1].sum() # msun
                production_df.loc[len(production_df)] = [output, np.log10(metal_produced)] # log(msun)

            df = df.merge(production_df[['output', 'log_metal_produced']], on='output')
        else:
            print('Did not find any metal production file, therefore will not plot the metal production panels')

        df = df.sort_values('time')
        df.to_csv(output_filename, sep='\t', index=None)
        print('Saved merged dataframe as', output_filename)
    else:
        print('Reading from existing file', output_filename)
        df = pd.read_table(output_filename, delim_whitespace=True, comment='#')

    if not args.forpaper: df, fig1 = plot_all_stats(df, args)
    df, fig2 = plot_time_series(df, args)

    print('Completed in %s' % (datetime.timedelta(minutes=(time.time() - start_time) / 60)))
