'''
    Filename: resolved_metallicity_plots.py
    Author: Ayan
    Created: 6-12-24
    Last modified: 7-22-24 by Cassi
    This file works with fogghorn_analysis.py to make the set of plots for resolved gas phase metallicity.
    If you add a new function to this scripts, then please also add the function name to the appropriate list at the end of fogghorn/header.py

'''

from foggie.fogghorn.header import *
from foggie.fogghorn.util import *

# --------------------------------------------------------------------------------------------------------------------
def gas_metallicity_resolved_MZR(ds, region, args, output_filename):
    '''
    Plots a spatially resolved gas metallicity vs gas mass relation.
    Returns nothing. Saves output as png file
    '''
    df = get_df_from_ds(region, args)

    # --------- Setting up the figure ---------
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.subplots_adjust(hspace=0.05, wspace=0.05, right=0.95, top=0.95, bottom=0.1, left=0.15)

    # Ayan will add stuff here

    # ---------annotate and save the figure----------------------
    plt.text(0.97, 0.95, 'z = %.2F' % ds.current_redshift, ha='right', transform=ax.transAxes, fontsize=args.fontsize)
    plt.savefig(output_filename)
    print('Saved figure ' + output_filename)
    plt.close()

# --------------------------------------------------------------------------------------------------------------------
def gas_metallicity_histogram(ds, region, args, output_filename):
    '''
    Plots a histogram of the gas metallicity (No Gaussian fits, for now).
    Returns nothing. Saves output as png file
    '''
    df = get_df_from_ds(region, args)

    # --------- Plotting the histogram ---------
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.subplots_adjust(hspace=0.05, wspace=0.05, right=0.95, top=0.95, bottom=0.1, left=0.15)

    color = 'salmon'
    p = plt.hist(df['log_metal'], bins=args.nbins, histtype='step', lw=2, density=True, ec=color, weights=df[args.weight] if args.weight is not None else None)

    # ---------- Adding vertical lines for percentile -------------
    percentiles = weighted_quantile(df['log_metal'], [0.25, 0.50, 0.75], weight=df[args.weight] if args.weight is not None else None)
    for thispercentile in np.atleast_1d(percentiles): ax.axvline(thispercentile, lw=1, ls='solid', color='maroon')

    # ---------- Tidy up figure-------------
    plt.legend(loc='upper right', bbox_to_anchor=(1, 0.75), fontsize=args.fontsize)
    ax.set_xlim(-1.5, 2.0)
    ax.set_ylim(0, 2.5)

    ax.set_xticks(np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 5))
    ax.set_yticks(np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 6))

    ax.set_xlabel(r'Log Metallicity (Z$_{\odot}$)', fontsize=args.fontsize)
    ax.set_ylabel('Normalised distribution', fontsize=args.fontsize)
    ax.set_xticklabels(['%.1F' % item for item in ax.get_xticks()], fontsize=args.fontsize)
    ax.set_yticklabels(['%.2F' % item for item in ax.get_yticks()], fontsize=args.fontsize)

    # ---------annotate and save the figure----------------------
    plt.text(0.97, 0.95, 'z = %.2F' % ds.current_redshift, ha='right', transform=ax.transAxes, fontsize=args.fontsize)
    plt.savefig(output_filename)
    print('Saved figure ' + output_filename)
    plt.close()

# ---------------------------------------------------------------------------------
def bin_fit_radial_profile(df, xcol, ycol, x_bins, ax, args, color='darkorange'):
    '''
    Function to overplot binned data on existing plot of radial profile of gas metallicity
    '''
    from uncertainties import ufloat, unumpy # this import statement is not in header.py because it is rarely used by the other functions

    df['binned_cat'] = pd.cut(df[xcol], x_bins)

    if args.weight is not None:
        agg_func = lambda x: np.sum(x * df.loc[x.index, args.weight]) / np.sum(df.loc[x.index, args.weight]) # function to get weighted mean
        agg_u_func = lambda x: np.sqrt(((np.sum(df.loc[x.index, args.weight] * x**2) / np.sum(df.loc[x.index, args.weight])) - (np.sum(x * df.loc[x.index, args.weight]) / np.sum(df.loc[x.index, args.weight]))**2) * (np.sum(df.loc[x.index, args.weight]**2)) / (np.sum(df.loc[x.index, args.weight])**2 - np.sum(df.loc[x.index, args.weight]**2))) # eq 6 of http://seismo.berkeley.edu/~kirchner/Toolkits/Toolkit_12.pdf
    else:
        agg_func, agg_u_func = np.mean, np.std

    y_binned = df.groupby('binned_cat', as_index=False).agg([(ycol, agg_func)])[ycol].values.flatten()
    y_u_binned = df.groupby('binned_cat', as_index=False).agg([(ycol, agg_u_func)])[ycol].values.flatten()
    x_bin_centers = x_bins[:-1] + np.diff(x_bins) / 2

    # --------- For correct propagation of errors, given that the actual fitting will be in log-space ----------
    quant = unumpy.log10(unumpy.uarray(y_binned, y_u_binned))
    y_binned, y_u_binned = unumpy.nominal_values(quant), unumpy.std_devs(quant)

    # getting rid of potential nan values
    indices = np.array(np.logical_not(np.logical_or(np.isnan(x_bin_centers), np.isnan(y_binned))))
    x_bin_centers = x_bin_centers[indices]
    y_binned = y_binned[indices]
    y_u_binned = y_u_binned[indices]

    # ---------- Plot mean binned y vs x profile--------------
    linefit, linecov = np.polyfit(x_bin_centers, y_binned, 1, cov=True)
    y_fitted = np.poly1d(linefit)(x_bin_centers)

    Zgrad = ufloat(linefit[0], np.sqrt(linecov[0][0]))
    Zcen = ufloat(linefit[1], np.sqrt(linecov[1][1]))

    ax.errorbar(x_bin_centers, y_binned, c=color, yerr=y_u_binned, lw=2, ls='none', zorder=1)
    ax.scatter(x_bin_centers, y_binned, c=color, s=150, lw=1, ec='black', zorder=10)
    ax.plot(x_bin_centers, y_fitted, color=color, lw=2.5, ls='dashed')
    ax.text(0.033, 0.2, 'Slope = %.2F ' % linefit[0] + 'dex/kpc', color=color, transform=ax.transAxes, fontsize=args.fontsize, va='center', bbox=dict(facecolor='k', alpha=0.6, edgecolor='k'))
    return ax, Zcen, Zgrad

# --------------------------------------------------------------------------------------------------------------------
def gas_metallicity_radial_profile(ds, region, args, output_filename):
    '''
    Plots a radial profile of the gas metallicity, overplotted with the radially binned profile and the fit to the binned profile.
    Returns nothing. Saves output as png file
    '''
    ylim = [-2.2, 1.2]
    if args.upto_kpc is not None:
        if args.docomoving: galrad = args.upto_kpc / (1 + ds.current_redshift) / 0.695  # include stuff within a fixed comoving kpc h^-1, 0.695 is Hubble constant
        else: galrad = args.upto_kpc  # include stuff within a fixed physical kpc
        region = ds.sphere(ds.halo_center_kpc, ds.arr(galrad, 'kpc')) # if a args.upto_kpc is specified, then the analysis 'region' will be restricted up to that
    else:
        galrad = ds.refine_width / 2.

    df = get_df_from_ds(region, args)

    # --------- First, plot both cell-by-cell profile first, using datashader---------
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.subplots_adjust(hspace=0.05, wspace=0.05, right=0.95, top=0.95, bottom=0.1, left=0.17)
    artist = dsshow(df, dsh.Point('rad', 'log_metal'), dsh.count(), norm='linear',x_range=(0, galrad), y_range=(ylim[0], ylim[1]), aspect='auto', ax=ax, cmap='Blues_r')

    # -------- Next, bin the metallicity profile and overplot the binned profile-----------
    bin_edges = np.linspace(0, galrad, 10)
    ax, Zcen, Zgrad = bin_fit_radial_profile(df, 'rad', 'metal', bin_edges, ax, args)
    linefit = [Zgrad.n, Zcen.n]

    # ---------- Then, plot the fitted metallicity profile---------------
    color = 'limegreen'
    fitted_y = np.poly1d(linefit)(bin_edges)
    ax.plot(bin_edges, fitted_y, color=color, lw=3, ls='solid')
    plt.text(0.033, 0.3, 'Slope = %.2F ' % linefit[0] + 'dex/kpc', color=color, transform=ax.transAxes, va='center', fontsize=args.fontsize,bbox=dict(facecolor='k', alpha=0.6, edgecolor='k'))

    # ---------- Tidy up figure-------------
    ax.set_xlim(0, args.upto_kpc)
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_yscale('log')

    ax.set_xlabel('Radius (kpc)', fontsize=args.fontsize)
    ax.set_ylabel(r'Log Metallicity (Z$_{\odot}$)', fontsize=args.fontsize)

    ax.set_xticks(np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 6))
    ax.set_yticks(np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 5))

    ax.set_xticklabels(['%.1F' % item for item in ax.get_xticks()], fontsize=args.fontsize)
    ax.set_yticklabels(['%.2F' % item for item in ax.get_yticks()], fontsize=args.fontsize)

    # --------- Annotate and save the figure----------------------
    plt.text(0.033, 0.05, 'z = %.2F' % ds.current_redshift, transform=ax.transAxes, fontsize=args.fontsize)
    plt.savefig(output_filename)
    print('Saved figure ' + output_filename)
    plt.close()


