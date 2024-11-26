#!/usr/bin/env python3

"""

    Title :      compute_MZgrad
    Notes :      Compute mass - metallicity gradient relation for a given FOGGIE galaxy
    Output :     txt file storing all the gradients & mass plus, optionally, Z profile plots
    Author :     Ayan Acharyya
    Started :    Feb 2022
    Examples :   run compute_MZgrad.py --system ayan_local --halo 8508 --output RD0030 --upto_re 3 --xcol rad_re --keep --weight mass
                 run compute_MZgrad.py --system ayan_local --halo 8508 --output RD0030 --upto_kpc 10 --xcol rad --keep --weight mass --notextonplot
                 run compute_MZgrad.py --system ayan_pleiades --halo 8508 --upto_re 3 --xcol rad_re --do_all_sims --weight mass --write_file --noplot
                 run compute_MZgrad.py --system ayan_pleiades --halo 8508 --upto_kpc 10 --xcol rad --do_all_sims --weight mass --write_file --noplot
                 run compute_MZgrad.py --system ayan_local --halo 8508 --output RD0030 --upto_kpc 10 --xcol rad --keep --weight mass --plot_onlybinned --forproposal
                 run compute_MZgrad.py --system ayan_local --halo 8508 --output RD0030 --upto_kpc 10 --xcol rad --keep --forpaper
                 run compute_MZgrad.py --system ayan_hd --halo 8508 --output RD0030 --upto_kpc 10 --xcol rad --plot_stellar --write_file --keep --forpaper

"""
from header import *
from util import *
from datashader_movie import *
from uncertainties import ufloat, unumpy
from yt.utilities.physical_ratios import metallicity_sun
start_time = time.time()

# -----------------------------------------------------------------------------
def get_disk_stellar_mass(args):
    '''
    Function to get the disk stellar mass for a given output, which is defined as the stellar mass contained within args.galrad, which can either be a fixed absolute size in kpc OR = args.upto_re*Re
    '''
    mass_filename = args.code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/masses_z-less-2.hdf5'
    upto_radius = args.galrad # should this be something else? 2*Re may be?

    if os.path.exists(mass_filename):
        print('Reading in', mass_filename)
        alldata = pd.read_hdf(mass_filename, key='all_data')
        thisdata = alldata[alldata['snapshot'] == args.output]

        if len(thisdata) == 0: # snapshot not found in masses less than z=2 file, so try greater than z=2 file
            mass_filename = mass_filename.replace('less', 'gtr')
            print('Could not find spanshot in previous file, now reading in', mass_filename)
            alldata = pd.read_hdf(mass_filename, key='all_data')
            thisdata = alldata[alldata['snapshot'] == args.output]

            if len(thisdata) == 0: # snapshot still not found in file
                print('Snapshot not found in either file. Returning bogus mass')
                return -999

        thisshell = thisdata[thisdata['radius'] <= upto_radius]
        if len(thisshell) == 0: # the smallest shell available in the mass profile is larger than the necessary radius within which we need the stellar mass
            if thisdata['radius'].iloc[0] <= 1.5*args.galrad:
                print('Smallest shell available in the mass profile is larger than args.galrad, taking the mass in the smallest shell as galaxy stellar mass')
                mstar = thisdata['stars_mass'].iloc[0] # assigning the mass of the smallest shell as the stellar mass
            else:
                print('Smallest shell avialable in mass profile is too small compared to args.galrad. Returning bogus mass')
                return -999
        else:
            mstar = thisshell['stars_mass'].values[-1] # radius is in kpc, mass in Msun
    else:
        print('File not found:', mass_filename)
        mstar = -999

    print('Stellar mass for halo ' + args.halo + ' output ' + args.output + ' (z=%.1F' %(args.current_redshift) + ') within ' + str(upto_radius) + ' kpc is %.2E' %(mstar))
    return mstar

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
def fit_binned(df, xcol, ycol, x_bins, ax=None, fit_inlog=False, color='darkorange', weightcol=None):
    '''
    Function to overplot binned data on existing plot
    '''
    df['binned_cat'] = pd.cut(df[xcol], x_bins)

    if weightcol is not None:
        agg_func = lambda x: np.sum(x * df.loc[x.index, weightcol]) / np.sum(df.loc[x.index, weightcol]) # function to get weighted mean
        agg_u_func = lambda x: np.sqrt(((np.sum(df.loc[x.index, weightcol] * x**2) / np.sum(df.loc[x.index, weightcol])) - (np.sum(x * df.loc[x.index, weightcol]) / np.sum(df.loc[x.index, weightcol]))**2) * (np.sum(df.loc[x.index, weightcol]**2)) / (np.sum(df.loc[x.index, weightcol])**2 - np.sum(df.loc[x.index, weightcol]**2))) # eq 6 of http://seismo.berkeley.edu/~kirchner/Toolkits/Toolkit_12.pdf
        #agg_u_func = np.std
        #agg_u_func = lambda x: np.std(x)/np.sqrt(len(x))
    else:
        agg_func, agg_u_func = np.mean, np.std

    y_binned = df.groupby('binned_cat', as_index=False).agg([(ycol, agg_func)])[ycol].values.flatten()
    y_u_binned = df.groupby('binned_cat', as_index=False).agg([(ycol, agg_u_func)])[ycol].values.flatten()
    x_bin_centers = x_bins[:-1] + np.diff(x_bins) / 2

    if fit_inlog:
        quant = unumpy.log10(unumpy.uarray(y_binned, y_u_binned)) # for correct propagation of errors
        y_binned, y_u_binned = unumpy.nominal_values(quant), unumpy.std_devs(quant)

    # getting rid of potential nan values
    indices = np.array(np.logical_not(np.logical_or(np.isnan(x_bin_centers), np.isnan(y_binned))))
    x_bin_centers = x_bin_centers[indices]
    y_binned = y_binned[indices]
    y_u_binned = y_u_binned[indices]

    # ----------to plot mean binned y vs x profile--------------
    try:
        linefit, linecov = np.polyfit(x_bin_centers, y_binned, 1, cov=True, w=None if args.noweight_forfit else 1 / (y_u_binned) ** 2)
    except Exception as e:
        myprint('Faced error:' + e + '; Therefore trying to fit without providing weights', args)
        linefit, linecov = np.polyfit(x_bin_centers, y_binned, 1, cov=True)
        pass
    y_fitted = np.poly1d(linefit)(x_bin_centers)

    Zgrad = ufloat(linefit[0], np.sqrt(linecov[0][0]))
    Zcen = ufloat(linefit[1], np.sqrt(linecov[1][1]))

    if fit_inlog and args.plotlog: y_binned, y_u_binned, y_fitted = 10**y_binned, 10**y_u_binned, 10**y_fitted ##

    print('Upon radially binning: Inferred slope for halo ' + args.halo + ' output ' + args.output + ' is', Zgrad, 'dex/re' if 're' in args.xcol else 'dex/kpc')

    if ax is not None:
        ax.errorbar(x_bin_centers, y_binned, c=color, yerr=y_u_binned, lw=2, ls='none', zorder=1)
        ax.scatter(x_bin_centers, y_binned, c=color, s=150, lw=1, ec='black', zorder=10)
        ax.plot(x_bin_centers, y_fitted, color=color, lw=2.5, ls='dashed')
        units = 'dex/re' if 're' in xcol else 'dex/kpc'
        if not (args.notextonplot or args.forproposal): ax.text(0.033, 0.05, r'Slope = %.2F $\pm$ %.2F ' % (Zgrad.n, Zgrad.s) + units, color=color, transform=ax.transAxes, fontsize=args.fontsize/1.5 if args.narrowfig else args.fontsize, va='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='white'))
        return ax
    else:
        return Zcen, Zgrad

# -----------------------------------------------------------
def plot_gradient(df, args, linefit=None):
    '''
    Function to plot the metallicity profile, along with the fitted gradient if provided
    Saves plot as .png
    '''
    onlybinned_text = '_onlybinned' if args.plot_onlybinned else ''
    weightby_text = '' if args.weight is None else '_wtby_' + args.weight
    density_cut_text = '_wdencut' if args.use_density_cut else ''
    if args.upto_kpc is not None:
        upto_text = '_upto%.1Fckpchinv' % args.upto_kpc if args.docomoving else '_upto%.1Fkpc' % args.upto_kpc
    else:
        upto_text = '_upto%.1FRe' % args.upto_re

    outfile_rootname = '%s_datashader_log_metal_vs_%s%s%s%s%s.png' % (args.output, args.xcol,upto_text, weightby_text, onlybinned_text, density_cut_text)
    if args.do_all_sims: outfile_rootname = 'z=*_' + outfile_rootname[len(args.output)+1:]
    filename = args.fig_dir + outfile_rootname.replace('*', '%.5F' % (args.current_redshift))

    # ---------first, plot both cell-by-cell profile first, using datashader---------
    if (args.forproposal and args.output != 'RD0042'):
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.subplots_adjust(right=0.95, top=0.95, bottom=0.2, left=0.17)
    elif args.narrowfig:
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.subplots_adjust(right=0.95, top=0.95, bottom=0.2, left=0.17)
    else:
        fig, ax = plt.subplots(figsize=(8, 8))
        fig.subplots_adjust(hspace=0.05, wspace=0.05, right=0.95, top=0.95, bottom=0.1, left=0.17)
    if not args.plot_onlybinned: artist = dsshow(df, dsh.Point(args.xcol, 'log_metal'), dsh.count(), norm='linear', x_range=(0, args.galrad / args.re if 're' in args.xcol else args.galrad), y_range=(args.ymin, args.ymax), aspect = 'auto', ax=ax, cmap='cividis', shade_hook=partial(dstf.spread, px=1, shape='square')) # the 40 in alpha_range and `square` in shade_hook are to reproduce original-looking plots as if made with make_datashader_plot()

    # --------bin the metallicity profile and plot the binned profile-----------
    ax = fit_binned(df, args.xcol, 'metal', args.bin_edges, ax=ax, fit_inlog=True, weightcol=args.weight)

    # ----------plot the fitted metallicity profile---------------
    if not (args.plot_onlybinned or args.forpaper):
        color = 'limegreen'
        if linefit is not None:
            fitted_y = np.poly1d(linefit)(args.bin_edges)
            ax.plot(args.bin_edges, fitted_y, color=color, lw=3, ls='solid')
            units = 'dex/re' if 're' in args.xcol else 'dex/kpc'
            if not args.notextonplot: plt.text(0.033, 0.15, 'Slope = %.2F ' % linefit[0] + units, color=color, transform=ax.transAxes, va='center', fontsize=args.fontsize/1.5 if args.narrowfig else args.fontsize, bbox=dict(facecolor='white', alpha=0.8, edgecolor='w'))

    # ----------tidy up figure-------------
    ax.set_xlim(0, args.upto_re if 're' in args.xcol else np.ceil(args.upto_kpc /0.695) if args.forappendix else args.galrad if args.forpaper else args.upto_kpc)
    delta_y = (args.ymax - args.ymin) / 50
    ax.set_ylim(args.ymin - delta_y, args.ymax) # the small offset between the actual limits and intended tick labels is to ensure that tick labels do not reach the very edge of the plot
    if args.forproposal: ax.set_yscale('log')

    ax.set_xlabel('Radius (kpc)', fontsize=args.fontsize)
    ax.set_ylabel(r'Metallicity (Z$_{\odot}$)' if args.forproposal else r'Log Metallicity (Z$_{\odot}$)', fontsize=args.fontsize)

    ax.set_xticks(np.arange(0, np.ceil(args.upto_kpc /0.695) if args.forappendix else np.min([np.ceil(args.upto_kpc /0.695), args.galrad]), 4 if args.forappendix else 2)) # for nice, round number tick marks
    if args.forpaper: ax.set_yticks(np.linspace(-1.5, 0.5, 5))
    else: ax.set_yticks(np.linspace(args.ymin, args.ymax, 5))

    ax.set_xticklabels(['%.1F' % item for item in ax.get_xticks()], fontsize=args.fontsize)
    ax.set_yticklabels(['%.2F' % item for item in ax.get_yticks()], fontsize=args.fontsize)

    if args.fortalk:
        #mplcyberpunk.add_glow_effects()
        try: mplcyberpunk.make_lines_glow()
        except: pass
        try: mplcyberpunk.make_scatter_glow()
        except: pass

    # ---------annotate and save the figure----------------------
    if not (args.forproposal and args.output == 'RD0042') and not args.forpaper:
        plt.text(0.033, 0.25, 'z = %.2F' % args.current_redshift, transform=ax.transAxes, fontsize=args.fontsize/1.5 if args.narrowfig else args.fontsize)
        plt.text(0.033, 0.15 if args.forproposal else 0.3, 't = %.1F Gyr' % args.current_time, transform=ax.transAxes, fontsize=args.fontsize/1.5 if args.narrowfig else args.fontsize)
    plt.savefig(filename, transparent=args.fortalk)
    myprint('Saved figure ' + filename, args)
    if not args.makemovie: plt.show(block=False)

    return fig

# -------------------------------
def fit_gradient(df, args, weight=None):
    '''
    Function to linearly fit the (log) metallicity profile out to certain Re, given a dataframe containing metallicity profile
    Returns the fitted gradient with uncertainty
    '''

    if weight is None: linefit, linecov = np.polyfit(df[args.xcol], df['log_metal'], 1, cov=True)
    else: linefit, linecov = np.polyfit(df[args.xcol], df['log_metal'], 1, cov=True, w=df[weight])
    Zgrad = ufloat(linefit[0], np.sqrt(linecov[0][0]))
    Zcen = ufloat(linefit[1], np.sqrt(linecov[1][1]))
    print('Inferred slope for halo ' + args.halo + ' output ' + args.output + ' is', Zgrad, 'dex/re' if 're' in args.xcol else 'dex/kpc')

    return Zcen, Zgrad

# -----------------------------------------------------------
def plot_stellar_metallicity_profile(box, args):
    '''
    Function to plot the stellar metallicity profile, in bins of stellar ages, along with fitting
    Saves plot as .png
    Returns figure handle
    '''
    print(f'Plotting stellar metallicity profiles for {args.output}..')
    age_bins = np.linspace(0, 14, 14+1)
    cmap_arr = ['Purples_r', 'Oranges_r', 'Greens_r', 'Blues_r', 'PuRd_r', 'Greys_r', 'Reds_r', 'YlGnBu_r', 'RdPu_r', 'YlGn_r', 'YlOrBr_r', 'cividis', 'plasma', 'viridis']

    # ----------determining file names--------------
    density_cut_text = '_wdencut' if args.use_density_cut else ''
    if args.upto_kpc is not None:
        upto_text = '_upto%.1Fckpchinv' % args.upto_kpc if args.docomoving else '_upto%.1Fkpc' % args.upto_kpc
    else:
        upto_text = '_upto%.1FRe' % args.upto_re
    outfilename = args.output_dir + 'txtfiles/%s_stellar_metallicity%s%s.txt' % (args.output, upto_text, density_cut_text)
    figfile_rootname = '%s_datashader_log_stellar_metal_vs_%s%s%s.png' % (args.output, args.xcol,upto_text, density_cut_text)
    if args.do_all_sims: figfile_rootname = 'z=*_' + figfile_rootname[len(args.output)+1:]
    filename = args.fig_dir + figfile_rootname.replace('*', '%.5F' % (args.current_redshift))

    # -------reading in file----------------------
    if not os.path.exists(filename) or args.clobber_plot:
        start_time2 = time.time()
        if not os.path.exists(outfilename) or args.clobber:
            print(f'{outfilename} not found, making new one..')
            age = box[('stars', 'age')].in_units('Gyr').ndarray_view()
            metallicity = box[('stars', 'metallicity_fraction')].in_units('Zsun').ndarray_view()
            distance = box[('stars', 'radius_corrected')].in_units('kpc').ndarray_view()
            mass = box[('stars', 'particle_mass')].in_units('Msun').ndarray_view()
            df = pd.DataFrame({'age': age, 'metal': metallicity, 'mass': mass, 'rad': distance})

            df.to_csv(outfilename, sep='\t', index=None)
            print(f'Written stellar metallicity df at {outfilename}')
            print('Finished making in %s' % (datetime.timedelta(minutes=(time.time() - start_time2) / 60)))
        else:
            print(f'Reading stellar metallicity from existing {outfilename} (might take a couple minutes)..')
            df = pd.read_table(outfilename)
            print('..finished reading in %s' % (datetime.timedelta(minutes=(time.time() - start_time2) / 60)))

        bin_labels = age_bins[:-1] + np.diff(age_bins)/2
        df['age_bin'] = pd.cut(df['age'], bins=age_bins, labels=bin_labels)
        df['log_metal'] = np.log10(df['metal'])
        age_bin_centers = np.unique(df['age_bin'])

        fig, ax = plt.subplots(figsize=(12, 6))
        fig.subplots_adjust(right=0.98, top=0.95, bottom=0.15, left=0.12)

        # --------looping over age bins-----------
        Zgrad_arr, Zcen_arr = [], []
        for index, this_age in enumerate(age_bin_centers):
            print(f'Doing age bin {index + 1} of {len(age_bin_centers)}..')
            df_sub = df[df['age_bin'] == this_age]
            artist = dsshow(df_sub, dsh.Point(args.xcol, 'log_metal'), dsh.count(), norm='linear', x_range=(0, args.galrad / args.re if 're' in args.xcol else args.galrad), y_range=(args.ymin, args.ymax), aspect = 'auto', ax=ax, cmap=cmap_arr[index])#, shade_hook=partial(dstf.spread, px=1, shape='square')) # the 40 in alpha_range and `square` in shade_hook are to reproduce original-looking plots as if made with make_datashader_plot()

            # --------radially bin the profile in metallicity space-----------
            df_sub['binned_cat'] = pd.cut(df_sub[args.xcol], args.bin_edges)

            stats = np.array([[group.agg(np.mean)['metal'], group.agg(np.std)['metal'], group.agg(np.mean)[args.xcol], group.agg(np.sum)['mass']] for key, group in df_sub.groupby('binned_cat')])
            y_binned = stats[:, 0]
            y_u_binned = stats[:, 1]
            x_bin_centers = stats[:, 2]
            mass_binned = stats[:, 3]

            # --------fit the binned profile in log metallicity space-----------
            quant = unumpy.log10(unumpy.uarray(y_binned, y_u_binned)) # for correct propagation of errors
            y_binned, y_u_binned = unumpy.nominal_values(quant), unumpy.std_devs(quant)

            # getting rid of potential nan values
            indices = np.array(np.logical_not(np.logical_or(np.isnan(x_bin_centers), np.isnan(y_binned))))
            x_bin_centers = x_bin_centers[indices]
            y_binned = y_binned[indices]
            y_u_binned = y_u_binned[indices]
            mass_binned = mass_binned[indices]

            # --------fit the binned profile-----------
            try:
                linefit, linecov = np.polyfit(x_bin_centers, y_binned, 1, cov=True, w=mass_binned)
            except ValueError:
                print(f'Could not radially fit age bin {this_age} for snapshot {args.output}, so skipping this age bin..')
                Zgrad_arr.append(ufloat(np.nan, np.nan))
                Zcen_arr.append(ufloat(np.nan, np.nan))
                continue
            y_fitted = np.poly1d(linefit)(x_bin_centers)

            Zgrad = ufloat(linefit[0], np.sqrt(linecov[0][0]))
            Zgrad_arr.append(Zgrad)
            Zcen = ufloat(linefit[1], np.sqrt(linecov[1][1]))
            Zcen_arr.append(Zcen)
            print(f'Radially binned stellar Z slope for ages {this_age - np.diff(age_bins)[0]/2}-{this_age + np.diff(age_bins)[0]/2} Gyr  is {Zgrad} dex/kpc')

            # --------plot the fitted profile-----------
            color = mpl_cm.get_cmap(cmap_arr[index])(0.2)
            ax.errorbar(x_bin_centers, y_binned, color=color, yerr=y_u_binned, lw=1, ls='none', zorder=1)
            ax.scatter(x_bin_centers, y_binned, color=color, s=100, lw=1, ec='black', zorder=5)
            ax.plot(x_bin_centers, y_fitted, color=color, lw=1, ls='dashed')
            if not (args.notextonplot or args.forproposal): ax.text(0.99 - int(index / 7) * 0.5, 0.02 + index * 0.07 - int(index / 7) * 7 * 0.07, r'[%.1F-%.1F] Gyr: Slope = %.2F $\pm$ %.2F dex/kpc' % (this_age - np.diff(age_bins)[0]/2, this_age + np.diff(age_bins)[0]/2, Zgrad.n, Zgrad.s), color=color, transform=ax.transAxes, fontsize=args.fontsize/1.5, va='bottom', ha='right', bbox=dict(facecolor='white', alpha=0.8, lw=0.1))

        # ----------tidy up figure-------------
        ax.set_xlim(0, args.upto_re if 're' in args.xcol else np.ceil(args.upto_kpc /0.695) if args.forappendix else args.galrad if args.forpaper else args.upto_kpc)
        delta_y = (args.ymax - args.ymin) / 50
        ax.set_ylim(args.ymin - delta_y, args.ymax) # the small offset between the actual limits and intended tick labels is to ensure that tick labels do not reach the very edge of the plot

        ax.set_xlabel('Radius (kpc)', fontsize=args.fontsize)
        ax.set_ylabel(r'Log Stellar Metallicity (Z$_{\odot}$)', fontsize=args.fontsize)

        ax.set_xticks(np.arange(0, np.ceil(args.upto_kpc /0.695) if args.forappendix else np.min([np.ceil(args.upto_kpc /0.695), args.galrad]), 4 if args.forappendix else 2)) # for nice, round number tick marks
        ax.set_yticks(np.linspace(args.ymin, args.ymax, 5))

        ax.set_xticklabels(['%.1F' % item for item in ax.get_xticks()], fontsize=args.fontsize)
        ax.set_yticklabels(['%.2F' % item for item in ax.get_yticks()], fontsize=args.fontsize)

        if args.fortalk:
            try: mplcyberpunk.make_lines_glow()
            except: pass
            try: mplcyberpunk.make_scatter_glow()
            except: pass

        # ---------annotate and save the figure----------------------
        if not (args.forproposal and args.output == 'RD0042') and not args.forpaper:
            plt.text(0.033, 0.25, 'z = %.2F' % args.current_redshift, transform=ax.transAxes, fontsize=args.fontsize/1.5 if args.narrowfig else args.fontsize)
            plt.text(0.033, 0.15 if args.forproposal else 0.3, 't = %.1F Gyr' % args.current_time, transform=ax.transAxes, fontsize=args.fontsize/1.5 if args.narrowfig else args.fontsize)
        fig.savefig(filename, transparent=args.fortalk)
        myprint('Saved figure ' + filename, args)
        plt.show(block=False)

        # ----------write out gradients as new df-------------
        new_df = pd.DataFrame({'age_bin': age_bin_centers, 'Zgrad': unumpy.nominal_values(Zgrad_arr), 'Zgrad_u': unumpy.std_devs(Zgrad_arr), 'Zcen': unumpy.nominal_values(Zcen_arr), 'Zcen_u': unumpy.std_devs(Zcen_arr)})
        new_df['output'] = args.output
        new_df['redshift'] = args.current_redshift
        new_df['time'] = args.current_time
        new_df['halo'] = args.halo
        new_df = new_df[['halo', 'output', 'redshift', 'time', 'age_bin', 'Zgrad', 'Zgrad_u', 'Zcen', 'Zcen_u']]

        if args.write_file:
            outfilename2 = args.output_dir + 'txtfiles/%s_stellar_metallicity_gradient_vs_age%s%s.txt' % (args.halo, upto_text, density_cut_text)
            if not os.path.isfile(outfilename2):
                new_df.to_csv(outfilename2, sep='\t', index=None, header='column_names')
                print(f'Written stellar metallicity gradients df at {outfilename2}')
            else:
                new_df.to_csv(outfilename2, sep='\t', mode='a', index=False, header=False)
                print('Appended stellar metallicity gradients to file', outfilename2)
    else:
        print(f'Figure for {args.output} already exists, so skipping it')
        df, new_df, fig = None, None, None

    return df, new_df, fig

# -------------------------------------------------------------------------------
def get_df_from_ds(box, args, outfilename=None):
    '''
    Function to make a pandas dataframe from the yt dataset, including only the metallicity profile,
    then writes dataframe to file for faster access in future
    This function is somewhat based on foggie.utils.prep_dataframe.prep_dataframe()
    :return: dataframe
    '''
    # -------------read/write pandas df file with ALL fields-------------------
    Path(args.output_dir + 'txtfiles/').mkdir(parents=True, exist_ok=True)  # creating the directory structure, if doesn't exist already
    if outfilename is None: outfilename = get_correct_tablename(args)

    if not os.path.exists(outfilename) or args.clobber:
        myprint(outfilename + ' does not exist. Creating afresh..', args)

        if args.use_density_cut:
            rho_cut = get_density_cut(args.current_time)  # based on Cassi's CGM-ISM density cut-off
            box = box.cut_region(['obj["gas", "density"] > %.1E' % rho_cut])
            print('Imposing a density criteria to get ISM above density', rho_cut, 'g/cm^3')

        df = pd.DataFrame()
        fields = ['rad', 'metal'] # only the relevant properties
        if args.weight is not None: fields += [args.weight]

        for index, field in enumerate(fields):
            myprint('Doing property: ' + field + ', which is ' + str(index + 1) + ' of the ' + str(len(fields)) + ' fields..', args)
            df[field] = box[field_dict[field]].in_units(unit_dict[field]).ndarray_view()

        df.to_csv(outfilename, sep='\t', index=None)
    else:
        myprint('Reading from existing file ' + outfilename, args)
        try:
            df = pd.read_table(outfilename, delim_whitespace=True, comment='#')
        except pd.errors.EmptyDataError:
            print('File existed, but it was empty, so making new file afresh..')
            dummy_args = copy.deepcopy(args)
            dummy_args.clobber = True
            df = get_df_from_ds(box, dummy_args, outfilename=outfilename)

    df = df[df['rad'].between(0, args.galrad)]  # in case this dataframe has been read in from a file corresponding to a larger chunk of the box
    df['log_metal'] = np.log10(df['metal'])
    df['rad_re'] = df['rad'] / args.re  # normalise by Re
    cols_to_extract = [args.xcol, 'log_metal']
    if args.weight is not None: cols_to_extract += [args.weight]
    df = df[cols_to_extract] # only keeping the columns that are needed to get Z gradient
    if 'metal' not in df: df['metal'] = 10 ** (df['log_metal'])

    return df

# -----main code-----------------
if __name__ == '__main__':
    dummy_args_tuple = parse_args('8508', 'RD0042')  # default simulation to work upon when comand line args not provided
    if type(dummy_args_tuple) is tuple: dummy_args = dummy_args_tuple[0] # if the sim has already been loaded in, in order to compute the box center (via utils.pull_halo_center()), then no need to do it again
    else: dummy_args = dummy_args_tuple
    if not dummy_args.keep: plt.close('all')

    if dummy_args.do_all_sims: list_of_sims = get_all_sims_for_this_halo(dummy_args) # all snapshots of this particular halo
    else: list_of_sims = list(itertools.product([dummy_args.halo], dummy_args.output_arr))
    total_snaps = len(list_of_sims)

    if dummy_args.forpaper:
        dummy_args.docomoving = True
        dummy_args.use_density_cut = True
        dummy_args.weight = 'mass'

    # -------set up dataframe and filename to store/write gradients in to--------
    cols_in_df = ['output', 'redshift', 'time']
    cols_to_add = ['mass', 'Zcen', 'Zcen_u', 'Zgrad', 'Zgrad_u', 'Zcen_binned', 'Zcen_u_binned', 'Zgrad_binned', 'Zgrad_u_binned', 'Ztotal']
    if dummy_args.upto_kpc is not None: cols_in_df = np.hstack([cols_in_df, ['re_stars', 're_coldgas'], [item + '_fixedr' for item in cols_to_add]])
    else: cols_in_df = np.hstack([cols_in_df, ['re_stars', 're_coldgas'], [item + '_re_stars' for item in cols_to_add], [item + '_re_coldgas' for item in cols_to_add]])

    df_grad = pd.DataFrame(columns=cols_in_df)
    weightby_text = '' if dummy_args.weight is None else '_wtby_' + dummy_args.weight
    density_cut_text = '_wdencut' if dummy_args.use_density_cut else ''
    if dummy_args.upto_kpc is not None:
        upto_text = '_upto%.1Fckpchinv' % dummy_args.upto_kpc if dummy_args.docomoving else '_upto%.1Fkpc' % dummy_args.upto_kpc
    else:
        upto_text = '_upto%.1FRe' % dummy_args.upto_re
    grad_filename = dummy_args.output_dir + 'txtfiles/' + dummy_args.halo + '_MZR_xcol_%s%s%s%s.txt' % (dummy_args.xcol, upto_text, weightby_text, density_cut_text)
    if dummy_args.write_file and dummy_args.clobber and os.path.isfile(grad_filename): subprocess.call(['rm ' + grad_filename], shell=True)

    if os.path.isfile(grad_filename) and not dummy_args.clobber and dummy_args.write_file: # if gradfile already exists
        existing_df_grad = pd.read_table(grad_filename)
        outputs_existing_on_file = pd.unique(existing_df_grad['output'])

    if dummy_args.dryrun:
        print('List of the total ' + str(total_snaps) + ' sims =', list_of_sims)
        sys.exit('Exiting dryrun..')
    # parse column names, in case log

    # --------------read in the cold gas profile file ONCE for a given halo-------------
    if dummy_args.upto_kpc is None:
        gasprofile = get_gas_profile(dummy_args)
    else:
        print('Not reading in cold gas profile because any re calculation is not needed')
        gasprofile = None

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
        this_sim = list_of_sims[index]
        if 'outputs_existing_on_file' in locals() and this_sim[1] in outputs_existing_on_file:
            print_mpi('Skipping ' + this_sim[1] + ' because it already exists in file', dummy_args)
            continue # skip if this output has already been done and saved on file

        start_time_this_snapshot = time.time()
        this_df_grad = pd.DataFrame(columns=cols_in_df)
        print_mpi('Doing snapshot ' + this_sim[1] + ' of halo ' + this_sim[0] + ' which is ' + str(index + 1 - core_start) + ' out of the total ' + str(core_end - core_start + 1) + ' snapshots...', dummy_args)
        halos_df_name = dummy_args.code_path + 'halo_infos/00' + this_sim[0] + '/' + dummy_args.run + '/'
        halos_df_name += 'halo_cen_smoothed' if dummy_args.use_cen_smoothed else 'halo_c_v'
        try:
            if len(list_of_sims) == 1 and not dummy_args.do_all_sims: args = dummy_args_tuple # since parse_args() has already been called and evaluated once, no need to repeat it
            else: args = parse_args(this_sim[0], this_sim[1])

            if type(args) is tuple:
                args, ds, refine_box = args  # if the sim has already been loaded in, in order to compute the box center (via utils.pull_halo_center()), then no need to do it again
                print_mpi('ds ' + str(ds) + ' for halo ' + str(this_sim[0]) + ' was already loaded at some point by utils; using that loaded ds henceforth', args)
            else:
                isdisk_required = np.array(['disk' in item for item in [args.xcol, args.ycol] + args.colorcol]).any()
                ds, refine_box = load_sim(args, region='refine_box', do_filter_particles=True, disk_relative=False, halo_c_v_name=halos_df_name)
        except Exception as e:
            print_mpi('Skipping ' + this_sim[1] + ' because ' + str(e), dummy_args)
            continue

        # parse paths and filenames
        args.fig_dir = args.output_dir + 'figs/' if args.do_all_sims or len(list_of_sims) > 1 else args.output_dir + 'figs/' + args.output + '/'
        Path(args.fig_dir).mkdir(parents=True, exist_ok=True)
        args.plotlog = False
        if args.fortalk:
            setup_plots_for_talks()
            args.forpaper = True
        if args.forpaper:
            args.docomoving = True
            args.use_density_cut = True
            args.weight = 'mass'
        if args.forproposal:
            args.plotlog = True

        args.current_redshift = ds.current_redshift
        args.current_time = ds.current_time.in_units('Gyr').tolist()
        if args.ymin is None:
            args.ymin = -2.5 if args.plot_stellar else -1.5 if args.forpaper else -2.2
        if args.ymax is None:
            args.ymax = 0.7 if args.plot_stellar else 0.6 if args.forpaper else 1.2

        re_from_stars = get_re_from_stars(ds, args) if args.upto_kpc is None else np.nan # kpc
        re_from_coldgas = get_re_from_coldgas(args, gasprofile=gasprofile) if args.upto_kpc is None else np.nan # kpc
        thisrow = [args.output, args.current_redshift, args.current_time, re_from_stars, re_from_coldgas] # row corresponding to this snapshot to append to df

        if args.upto_kpc is not None:
            method_arr = ['']
            upto_radius_arr = [args.upto_kpc]
        else:
            method_arr = ['stars', 'coldgas'] # it is important that stars and coldgas appear in this sequence
            upto_radius_arr = np.array([re_from_stars, re_from_coldgas])

        for this_upto_radius in upto_radius_arr:
            if this_upto_radius > 0:
                if args.upto_kpc is not None:
                    args.re = np.nan
                    if args.docomoving: args.galrad = this_upto_radius / (1 + args.current_redshift) / 0.695 # fit within a fixed comoving kpc h^-1, 0.695 is Hubble constant
                    else: args.galrad = this_upto_radius # fit within a fixed physical kpc
                else:
                    args.re = this_upto_radius
                    args.galrad = args.re * args.upto_re  # kpc
                args.bin_edges = np.linspace(0, args.galrad / args.re if 're' in args.xcol else args.galrad, 10)

                # extract the required box
                box_center = ds.halo_center_kpc
                box = ds.sphere(box_center, ds.arr(args.galrad, 'kpc'))

                df = get_df_from_ds(box, args) # get dataframe with metallicity profile info
                if len(df) == 0:
                    print_mpi('Skipping ' + this_sim[1] + ' because empty dataframe', dummy_args)
                    thisrow += (np.ones(10) * np.nan).tolist()
                    continue

                # ------for stellar metallicity---------
                if args.plot_stellar:
                    df_stellar_prof, df_stellar_grad, fig = plot_stellar_metallicity_profile(box, args)
                    thisrow += (np.ones(10) * np.nan).tolist() # dummy values
                else:
                    # ------for gas metallicity---------
                    Zcen, Zgrad = fit_gradient(df, args, weight=args.weight)
                    Zcen_binned, Zgrad_binned = fit_binned(df, args.xcol, 'metal', args.bin_edges, ax=None, fit_inlog=True, weightcol=args.weight)

                    if not args.noplot: fig = plot_gradient(df, args, linefit=[Zgrad.n, Zcen.n]) # plotting the Z profile, with fit

                    mstar = get_disk_stellar_mass(args) # Msun

                    df['metal_mass'] = df['mass'] * df['metal'] * metallicity_sun
                    Ztotal = (df['metal_mass'].sum()/df['mass'].sum())/metallicity_sun # in Zsun
                    Ztotal = np.log10(Ztotal) # in log Zsun

                    thisrow += [mstar, Zcen.n, Zcen.s, Zgrad.n, Zgrad.s, Zcen_binned.n, Zcen_binned.s, Zgrad_binned.n, Zgrad_binned.s, Ztotal]
            else:
                thisrow += (np.ones(10)*np.nan).tolist()


        this_df_grad.loc[len(this_df_grad)] = thisrow
        df_grad = pd.concat([df_grad, this_df_grad])
        if args.write_file:
            if not os.path.isfile(grad_filename):
                this_df_grad.to_csv(grad_filename, sep='\t', index=None, header='column_names')
                print('Wrote to gradient file', grad_filename)
            else:
                this_df_grad.to_csv(grad_filename, sep='\t', index=None, mode='a', header=False)
                print('Appended to gradient file', grad_filename)

        print_mpi('This snapshots completed in %s mins' % ((time.time() - start_time_this_snapshot) / 60), dummy_args)

    if ncores > 1: print_master('Parallely: time taken for ' + str(total_snaps) + ' snapshots with ' + str(ncores) + ' cores was %s mins' % ((time.time() - start_time) / 60), dummy_args)
    else: print_master('Serially: time taken for ' + str(total_snaps) + ' snapshots with ' + str(ncores) + ' core was %s mins' % ((time.time() - start_time) / 60), dummy_args)
