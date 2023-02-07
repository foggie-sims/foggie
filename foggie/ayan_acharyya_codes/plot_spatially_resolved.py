#!/usr/bin/env python3

"""

    Title :      plot_spatially_resolved.py
    Notes :      Plot spatially resolved relations, profiles at a given resolution, for a given FOGGIE galaxy
    Output :     spatially resolved plots as png
    Author :     Ayan Acharyya
    Started :    Jan 2023
    Examples :   run plot_spatially_resolved.py --system ayan_local --halo 8508 --output RD0030 --upto_kpc 10 --res 0.1 --plot_map --weight mass --docomoving --proj x --nbins 100 --fit_multiple --forproposal
                 run plot_spatially_resolved.py --system ayan_pleiades --halo 8508 --upto_kpc 10 --res 0.1 --do_all_sims --weight mass --use_gasre

"""
from header import *
from util import *
from compute_Zscatter import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
start_time = time.time()

# -----------------------------------------------------
def annotate_axes(xlabel, ylabel, ax, args):
    '''
    Function to annotate x and y axes
    Returns the axis handle
    '''
    if args.forproposal: # scale bar inset instead of axis ticks
        length = 5  # kpc
        if args.plot_Z: # sufficient if only the Z panel has the scale info
            ax.text(0.08, 0.07, '%d kpc' % length, color='white', ha='left', va='bottom', transform=ax.transAxes, fontsize=args.fontsize)
            ax.axhline(-9, 0.05, 0.05 + length / np.diff(ax.get_xlim())[0], c='white', lw=2)  # converting length from kpc units to plot units
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
    else:
        ax.set_xticks(np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 5))
        ax.set_yticks(np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 5))
        ax.set_xlabel(xlabel, fontsize=args.fontsize)
        ax.set_ylabel(ylabel, fontsize=args.fontsize)
        ax.set_xticklabels(['%.1F' % item for item in ax.get_xticks()], fontsize=args.fontsize)
        ax.set_yticklabels(['%.1F' % item for item in ax.get_yticks()], fontsize=args.fontsize)

    return ax

# ---------------------------------------------------------------------------------
def make_colorbar_axis(label, artist, fig, fontsize):
    '''
    Function to make the colorbar axis
    '''
    cax_xpos, cax_ypos, cax_width, cax_height = 0.7, 0.835, 0.25, 0.035
    cax = fig.add_axes([cax_xpos, cax_ypos, cax_width, cax_height])
    plt.colorbar(artist, cax=cax, orientation='horizontal')

    cax.set_xticklabels(['%.0F' % index for index in cax.get_xticks()], fontsize=fontsize)
    fig.text(cax_xpos + cax_width / 2, cax_ypos + cax_height + 0.005, label, fontsize=fontsize, ha='center', va='bottom')

    return fig, cax

# -----------------------------------------------------
def saveplot(fig, name, args):
    '''
    Function to save a figure given a name and print a statement
    Returns the figure handle
    '''
    outfile_rootname = '%s_%s%s%s.png' % (args.output, name, args.res_text, args.upto_text)
    if args.do_all_sims: outfile_rootname = 'z=*_' + outfile_rootname[len(args.output) + 1:]
    figname = args.fig_dir + outfile_rootname.replace('*', '%.5F' % (args.current_redshift))
    plt.savefig(figname, transparent=False)
    myprint('Saved figure ' + figname, args)

    return fig

# ----------------------------------------------------
def plot_map_from_frb(map, args, cmap='viridis', label=None, makelog=True, name='', clim=None):
    '''
    Function to plot projection plot from the given 2D array as input
    '''
    sns.set_style('ticks')  # instead of darkgrid, so that there are no grids overlaid on the projections
    fig, ax = plt.subplots(figsize=(8, 7))
    if args.forproposal: fig.subplots_adjust(right=0.98, top=0.98, bottom=0.15, left=0.02)
    else: fig.subplots_adjust(right=0.85, top=0.98, bottom=0.02, left=0.17)

    proj = ax.imshow(map, cmap=cmap, norm=LogNorm() if makelog else None, extent=[-args.galrad, args.galrad, -args.galrad, args.galrad], vmin=clim[0] if clim is not None else None, vmax=clim[1] if clim is not None else None)
    divider = make_axes_locatable(ax)

    if args.forproposal:
        cax = divider.append_axes('bottom', size='5%', pad=0.05)
        cbar = plt.colorbar(proj, cax=cax, orientation='horizontal')
    else:
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = plt.colorbar(proj, cax=cax, orientation='vertical')

    ax = annotate_axes(r'x (kpc)', r'y (kpc)', ax, args)

    cbar.ax.tick_params(labelsize=args.fontsize)
    if label is not None: cbar.set_label(label, fontsize=args.fontsize)

    if not args.forproposal:
        plt.text(0.97, 0.95, 'z = %.2F' % args.current_redshift, ha='right', transform=ax.transAxes, fontsize=args.fontsize)
        plt.text(0.97, 0.9, 't = %.1F Gyr' % args.current_time, ha='right', transform=ax.transAxes, fontsize=args.fontsize)

    fig = saveplot(fig, 'map_%s_%s'%(name, args.projection), args)
    plt.show(block=False)

    return fig

# ----------------------------------------------------
def plot_ks_relation(frb, args):
    '''
    Function to plot spatially resolved KS relation
    Requires FRB object as input
    '''
    plt.style.use('seaborn-whitegrid') # instead of ticks, so that grids are overlaid on the plot
    sigma_star_lim = (-1.0, 4.0)
    sigma_gas_lim = (-2.5, 2.5)
    sigma_sfr_lim = (-4.0, 0.5)

    # ----- getting all maps ------------
    map_sigma_star = np.array(frb['deposit', 'stars_density'].in_units('Msun/pc**2')) # stellar mass surface density in Msun/pc^2
    map_sigma_gas = np.array(frb['gas', 'density'].in_units('Msun/pc**2')) # gas mass surface density in Msun/pc^2
    map_sigma_star_young = np.array(frb['deposit', 'young_stars_density'].in_units('Msun/kpc**2')) # young stars mass surface density in Msun/kpc^2
    map_sigma_sfr = map_sigma_star_young / 10e6 # dividing young stars mass by 10 Myr to get SFR surface density in Msun/yr/kpc^2

    # ----- plotting surface maps ----------
    if args.plot_map:
        cmap = density_color_map
        fig_star_map = plot_map_from_frb(map_sigma_star, args, cmap=cmap, label=r'$\Sigma_{\mathrm{star}} (\mathrm{M}_{\odot}/\mathrm{pc}^2)$', name='sigma_star', clim=(10**sigma_star_lim[0], 10**sigma_star_lim[1]))
        fig_gas_map = plot_map_from_frb(map_sigma_gas, args, cmap=cmap, label=r'$\Sigma_{\mathrm{gas}} (\mathrm{M}_{\odot}/\mathrm{pc}^2)$', name='sigma_gas', clim=(10**sigma_gas_lim[0], 10**sigma_gas_lim[1]))
        fig_sfr_map = plot_map_from_frb(map_sigma_sfr, args, cmap=cmap, label=r'$\Sigma_{\mathrm{SFR}} (\mathrm{M}_{\odot}/\mathrm{yr}/\mathrm{kpc}^2)$', name='sigma_sfr', clim=(10**sigma_sfr_lim[0], 10**sigma_sfr_lim[1]))
    else:
        fig_star_map, fig_gas_map, fig_sfr_map = None, None, None

    # ----- plotting KS relation ------------
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.subplots_adjust(right=0.95, top=0.9, bottom=0.12, left=0.15)

    xdata = np.log10(map_sigma_gas).flatten()
    ydata = np.log10(map_sigma_sfr).flatten()

    xdata = np.ma.compressed(np.ma.masked_array(xdata, ~np.isfinite(ydata)))
    ydata = np.ma.compressed(np.ma.masked_array(ydata, ~np.isfinite(ydata)))
    ax.scatter(xdata, ydata, s=100, lw=0)

    ax.set_xlim(sigma_gas_lim)
    ax.set_ylim(sigma_sfr_lim)

    # ------ fittingthe relation and overplotting ---------
    linefit, linecov = np.polyfit(xdata, ydata, 1, cov=True)
    print('At %.1F kpc resolution, %d out of %d pixels are valid, and KS fit =' % (args.res, len(ydata), len(map_sigma_gas)**2), linefit)
    xarr = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 10)
    ax.plot(xarr, np.poly1d(linefit)(xarr), color='b', lw=2, ls='solid', label=r'Fitted slope = %.1F $\pm$ %.1F' % (linefit[0], np.sqrt(linecov[0][0])))
    ax.plot(xarr, np.poly1d([1.4, -4])(xarr), color='b', lw=2, ls='dashed', label=r'KS relation') # from literature: https://ned.ipac.caltech.edu/level5/March15/Kennicutt/Kennicutt6.html
    ax.legend(loc='lower right', fontsize=args.fontsize)

    ax = annotate_axes(r'$\log{\, \Sigma_{\mathrm{gas}} (\mathrm{M}_{\odot}/\mathrm{pc}^2)}$', r'$\log{\, \Sigma_{\mathrm{SFR}} (\mathrm{M}_{\odot}/\mathrm{yr}/\mathrm{kpc}^2)}$', ax, args)

    plt.text(0.97, 0.35, 'z = %.2F' % args.current_redshift, ha='right', transform=ax.transAxes, fontsize=args.fontsize)
    plt.text(0.97, 0.3, 't = %.1F Gyr' % args.current_time, ha='right', transform=ax.transAxes, fontsize=args.fontsize)

    fig = saveplot(fig, 'KSrelation_%s'%(args.projection), args)
    plt.show(block=False)

    return fig, fig_star_map, fig_gas_map, fig_sfr_map

# -----------------------------------------------------
def get_dist_map(args):
    '''
    Function to get a map of distance of each from the center
    '''
    center_pix = (args.ncells - 1)/2.
    kpc_per_pix = 2 * args.galrad / args.ncells
    map_dist = np.array([[np.sqrt((i - center_pix)**2 + (j - center_pix)**2) for j in range(args.ncells)] for i in range(args.ncells)]) * kpc_per_pix # kpc

    return map_dist

# ----------------------------------------------------
def plot_metallicity(frb, args):
    '''
    Function to plot spatially resolved metallicity map, histogram and radial profile at a given resolution
    Requires FRB object as input
    '''
    x_lim = (0, args.galrad)
    Z_lim = (-2.0, 2.0)

    # ----- getting maps ------------
    map_gas_mass = frb['gas', 'mass']
    map_metal_mass = frb['gas', 'metal_mass']
    map_Z = np.array((map_metal_mass / map_gas_mass).in_units('Zsun')) # now in Zsun units
    map_dist = get_dist_map(args)

    if args.weight is not None:
        map_weights = np.array(frb['gas', args.weight])
        map_Z = len(map_weights)**2 * map_Z * map_weights / np.sum(map_weights)

    # ----- plotting surface maps ----------
    if args.plot_map:
        #fig_dist_map = plot_map_from_frb(map_dist, args, cmap='Blues_r', label=r'Radius (kpc)', makelog=False, name='dist', clim=x_lim)
        fig_Z_map = plot_map_from_frb(map_Z, args, cmap=old_metal_color_map, label=r'Z/Z$_{\odot}$', name='metallicity', clim=(10**Z_lim[0], 10**Z_lim[1]))
    else:
        fig_Z_map, fig_dist_map = None, None

    # ----- plotting Z profile ------------
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.subplots_adjust(right=0.95, top=0.9, bottom=0.12, left=0.15)
    '''
    xdata = map_dist.flatten()
    ydata = np.log10(map_Z).flatten() # in log(Z/Zsun) units

    if args.weight is not None: wdata = map_weights.flatten()

    ax.scatter(xdata, ydata, s=30, lw=0)

    ax.set_xlim(x_lim)
    ax.set_ylim(Z_lim)

    # ------ fittingthe relation and overplotting ---------
    linefit, linecov = np.polyfit(xdata, ydata, 1, cov=True)
    print('At %.1F kpc resolution, Z profile fit =' % args.res, linefit)
    xarr = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 10)
    ax.plot(xarr, np.poly1d(linefit)(xarr), color='b', lw=2, ls='solid')

    ax = annotate_axes(r'Radius (kpc)', r'$\log{\, \mathrm{Z/Z}_{\odot}}$', ax, args)

    plt.text(0.97, 0.9, r'Fitted slope = %.2F dex/kpc' % (linefit[0]), ha='right', transform=ax.transAxes, fontsize=args.fontsize)
    plt.text(0.03, 0.17, 'z = %.2F' % args.current_redshift, ha='left', transform=ax.transAxes, fontsize=args.fontsize)
    plt.text(0.03, 0.1, 't = %.1F Gyr' % args.current_time, ha='left', transform=ax.transAxes, fontsize=args.fontsize)

    fig = saveplot(fig, 'Zprofile_%s'%(args.projection), args)
    plt.show(block=False)
    '''
    # ------------ plotting Z distribution ------------
    fig2, ax = plt.subplots(figsize=(8, 7))
    fig2.subplots_adjust(right=0.95, top=0.9, bottom=0.12, left=0.15)
    '''
    ydata = 10 ** ydata # back to Zsun units
    args.xmax = 2 #10**Z_lim[1] # fit_distribution (which is invoked after a few lines) requires args.xmax to be = Z_max (in linear spaec)

    if args.weight is None: p = plt.hist(ydata, bins=args.nbins, histtype='step', lw=2, ec='salmon', density=True)
    else: p = plt.hist(ydata, bins=args.nbins, histtype='step', lw=2, density=True, range=(0, args.xmax), ec='salmon', weights=wdata)

    ax.set_xlim(0, args.xmax)
    ax.set_ylim(0, 3.5)

    # ------ fittingthe relation and overplotting ---------
    fit, _, Z25, Z50, Z75, _, Zmean, Zvar, _, _, gauss_mean, _ = fit_distribution(ydata, args, weights=wdata)
    xvals = p[1][:-1] + np.diff(p[1])
    if args.fit_multiple:
        ax.plot(xvals, multiple_gauss(xvals, *fit), c='k', lw=2, label='Total fit')
        ax.plot(xvals, gauss(xvals, fit[:3]), c='k', lw=2, ls='--', label=None if args.annotate_profile else 'Regular Gaussian')
        ax.plot(xvals, skewed_gauss(xvals, fit[3:]), c='k', lw=2, ls='dotted', label=None if args.annotate_profile else 'Skewed Gaussian')
    else:
        ax.plot(xvals, fit.best_fit, c='k', lw=2, label='Fit')

    # ----------adding vertical lines-------------
    if args.fit_multiple:
        ax.axvline(gauss_mean.n, lw=2, ls='dashed', color='k')
        ax.axvline(Zmean.n, lw=2, ls='dotted', color='k')
    else:
        ax.axvline(Zmean.n, lw=2, ls='dotted', color='k')

    ax.axvline(Z25.n, lw=2.5, ls='solid', color='salmon')
    ax.axvline(Z50.n, lw=2.5, ls='solid', color='salmon')
    ax.axvline(Z75.n, lw=2.5, ls='solid', color='salmon')

    ax = annotate_axes(r'Metallicity (Z$_{\odot}$)', 'Normalised distribution', ax, args)

    plt.text(0.97, 0.9, 'z = %.2F' % args.current_redshift, ha='right', transform=ax.transAxes, fontsize=args.fontsize)
    plt.text(0.97, 0.83, 't = %.1F Gyr' % args.current_time, ha='right', transform=ax.transAxes, fontsize=args.fontsize)

    plt.text(0.97, 0.7, r'Mean = %.2F Z$\odot$' % Zmean.n, ha='right', transform=ax.transAxes, fontsize=args.fontsize)
    plt.text(0.97, 0.63, r'Sigma = %.2F Z$\odot$' % Zvar.n, ha='right', transform=ax.transAxes, fontsize=args.fontsize)

    fig2 = saveplot(fig2, 'Zdistribution_%s'%(args.projection), args)
    plt.show(block=False)
    '''
    return fig, fig2, fig_Z_map

# ----------------------------------------------------
def plot_kinematics(frb, args):
    '''
    Function to plot spatially resolved velocity and velocity dispersion mapa, and radial profiles at a given resolution
    Requires FRB object as input
    '''
    x_lim = (0, args.galrad)
    vel_lim = (-400, 400)
    vdisp_lim = (0, 100)

    # ----- getting maps ------------
    map_gas_vel = np.array(frb['gas', 'velocity_los'].in_units('km/s'))
    map_gas_vdisp = None
    map_dist = get_dist_map(args)

    # ----- plotting surface maps ----------
    if args.plot_map:
        fig_vel_map = plot_map_from_frb(map_gas_vel, args, cmap=velocity_discrete_cmap, label=r'LoS velocity (km/s)', makelog=False, name='vel', clim=vel_lim)
        fig_vdisp_map = plot_map_from_frb(map_gas_vdisp, args, cmap='Blues', label=r'LoS velocity dispersion (km/s)', makelog=False, name='vel', clim=vdisp_lim)
    else:
        fig_vel_map, fig_vdisp_map = None, None

    # ----- plotting vel profile ------------
    fig_vel_profile, ax = plt.subplots(figsize=(8, 7))
    fig_vel_profile.subplots_adjust(right=0.95, top=0.9, bottom=0.12, left=0.15)

    xdata = map_dist.flatten()
    ydata = map_gas_vel.flatten()

    ax.scatter(xdata, ydata, s=30, lw=0)

    ax.set_xlim(x_lim)
    ax.set_ylim(vel_lim)

    ax = annotate_axes(r'Radius (kpc)', r'LoS velocity (km/s)', ax, args)

    plt.text(0.97, 0.35, 'z = %.2F' % args.current_redshift, ha='right', transform=ax.transAxes, fontsize=args.fontsize)
    plt.text(0.97, 0.3, 't = %.1F Gyr' % args.current_time, ha='right', transform=ax.transAxes, fontsize=args.fontsize)

    fig_vel_profile = saveplot(fig_vel_profile, 'vel_profile_%s'%(args.projection), args)
    plt.show(block=False)

    # ----- plotting vel dispersion profile ------------
    fig_vdisp_profile, ax = plt.subplots(figsize=(8, 7))
    fig_vdisp_profile.subplots_adjust(right=0.95, top=0.9, bottom=0.12, left=0.15)

    ydata = map_gas_vdisp.flatten()

    ax.scatter(xdata, ydata, s=30, lw=0)

    ax.set_xlim(x_lim)
    ax.set_ylim(vdisp_lim)

    ax = annotate_axes(r'Radius (kpc)', r'LoS velocity dispersion (km/s)', ax, args)

    plt.text(0.97, 0.35, 'z = %.2F' % args.current_redshift, ha='right', transform=ax.transAxes, fontsize=args.fontsize)
    plt.text(0.97, 0.3, 't = %.1F Gyr' % args.current_time, ha='right', transform=ax.transAxes, fontsize=args.fontsize)

    fig_vdisp_profile = saveplot(fig_vdisp_profile, 'vdisp_profile_%s'%(args.projection), args)
    plt.show(block=False)

    return fig_vel_profile, fig_vel_map, fig_vdisp_profile, fig_vdisp_map

# ----------------------------------------------------
def plot_cellmass(box, args):
    '''
    Function to plot intrinsic cell mass map and profile \
    Requires YTRegion object as input
    '''
    sns.set_style('ticks')  # instead of darkgrid, so that there are no grids overlaid on the projections
    proj_dict = {'x': ('y', 'z'), 'y': ('z', 'x'), 'z': ('x', 'y')}
    cm_lim = (-1, 7) # log Msun
    Z_lim = (0.01, 4)

    # ------------------- plotting 2D map with yt -----------------------
    rho_cut = get_density_cut(args.current_time)  # based on Cassi's CGM-ISM density cut-off
    if args.plot_map:
        thisfield = ('gas', 'density')
        ad = box.ds.all_data()
        cgm = ad.cut_region(['obj["gas", "density"] < %.1E' % rho_cut])
        ism = ad.cut_region(['obj["gas", "density"] > %.1E' % rho_cut])

        prj = yt.SlicePlot(ds, args.projection, thisfield, center=box.center, data_source=ism, width=2 * args.galrad * kpc, fontsize=args.fontsize)
        prj.set_cmap(thisfield, cmap='Blues')
        prj.annotate_timestamp(corner='lower_right', redshift=True, draw_inset_box=True)
        outfile_rootname = '%s_%s%s.png' % (args.output, 'ism_slice', args.upto_text)
        if args.do_all_sims: outfile_rootname = 'z=*_' + outfile_rootname[len(args.output) + 1:]
        figname = args.fig_dir + outfile_rootname.replace('*', '%.5F' % (args.current_redshift))
        prj.save(figname)

        prj = yt.SlicePlot(ds, args.projection, thisfield, center=box.center, data_source=cgm, width=2 * args.galrad * kpc, fontsize=args.fontsize)
        prj.set_cmap(thisfield, cmap='Reds')
        prj.annotate_timestamp(corner='lower_right', redshift=True, draw_inset_box=True)
        prj.save(figname.replace('ism', 'cgm'))

    # ----- making df with all relevant cells in box ------------
    x = box['gas', 'x'].in_units('kpc') - box.center.in_units('kpc')[0]
    y = box['gas', 'y'].in_units('kpc') - box.center.in_units('kpc')[1]
    z = box['gas', 'z'].in_units('kpc') - box.center.in_units('kpc')[2]
    rad = box['gas', 'radius_corrected'].in_units('kpc')
    cm = box['gas', 'cell_mass'].in_units('Msun')
    den = box['gas', 'density'].in_units('g/cm**3')
    Z = box['gas', 'metallicity'].in_units('Zsun')
    df = pd.DataFrame({'pos_x': x, 'pos_y': y, 'pos_z': z, 'radius': rad, 'cell_mass': cm, 'density': den, 'metallicity':Z})
    df['log_density'] = np.log10(df['density'])
    df['log_cell_mass'] = np.log10(df['cell_mass'])
    df['log_metallicity'] = np.log10(df['metallicity'])
    df = df[df['radius'] <= np.sqrt(2) * args.galrad]

    # ------ making new column to categorise ISM vs CGM -------------------
    log_rho_cut = np.log10(rho_cut)
    df['cat'] = df['log_density'].apply(lambda x: 'ism' if x > log_rho_cut else 'cgm')
    df['cat'] = df['cat'].astype('category')
    color_key = dict(ism='cornflowerblue', cgm='salmon') # 2 color-codings, one for ISM and one for CGM
    '''
    # ------------------- plotting 2D map with datshader -----------------------
    if args.plot_map:
        fig_den_map, ax1 = plt.subplots(figsize=(8, 8))
        fig_den_map.subplots_adjust(right=0.95, top=0.9, bottom=0.12, left=0.17)
    
        artist = dsshow(df, dsh.Point('pos_' + proj_dict[args.projection][0], 'pos_' + proj_dict[args.projection][1]), dsh.by('cat', dsh.mean('density')), norm='log', color_key=color_key, x_range=(-args.galrad, args.galrad), y_range=(-args.galrad, args.galrad), aspect = 'auto', ax=ax1, shade_hook=partial(dstf.spread, shape='square', px=6)) # to make the main datashader plot
    
        ax1 = annotate_axes(proj_dict[args.projection][0] + ' (kpc)', proj_dict[args.projection][1] + ' (kpc)', ax1, args) # to annotate coordinate axes
        #fig_den_map, ax2 = make_colorbar_axis(r'Log gas density (g/cm$^3$)', artist, fig_den_map, args.fontsize) # to make colorbar axis
    
        plt.text(0.1, 0.15, 'z = %.2F' % args.current_redshift, ha='left', transform=ax1.transAxes, fontsize=args.fontsize)
        plt.text(0.1, 0.1, 't = %.1F Gyr' % args.current_time, ha='left', transform=ax1.transAxes, fontsize=args.fontsize)
        fig_den_map = saveplot(fig_den_map, 'map_density', args)
    '''
    # ----- plotting cell mass radial profile with datashader ------------
    plt.style.use('seaborn-whitegrid') # instead of ticks, so that grids are overlaid on the plot
    fig_cm_profile, ax1 = plt.subplots(figsize=(8, 8))
    fig_cm_profile.subplots_adjust(right=0.95, top=0.9, bottom=0.12, left=0.17)

    artist = dsshow(df, dsh.Point('radius', 'log_cell_mass'), dsh.count_cat('cat'), norm='linear', color_key=color_key, x_range=(0, args.galrad), y_range=cm_lim, aspect = 'auto', ax=ax1, shade_hook=partial(dstf.spread, shape='circle', px=2)) # to make the main datashader plot

    ax1 = annotate_axes('Radius (kpc)', r'Log cell mass (M$_{\odot}$)', ax1, args) # to annotate axes

    plt.text(0.1, 0.15, 'z = %.2F' % args.current_redshift, ha='left', transform=ax1.transAxes, fontsize=args.fontsize)
    plt.text(0.1, 0.1, 't = %.1F Gyr' % args.current_time, ha='left', transform=ax1.transAxes, fontsize=args.fontsize)
    fig_cm_profile = saveplot(fig_cm_profile, 'cm_profile', args)
    plt.show(block=False)

    # ----- plotting cell mass vs density color coded by metallicity with datashader ------------
    from datashader_movie import categorize_by_quant, get_color_keys, get_color_labels
    ncolbins = 7
    df['cat2'] = categorize_by_quant(df['metallicity'], Z_lim[0], Z_lim[1], ncolbins)
    df['cat2'] = df['cat2'].astype('category')
    args.cmap = mpl_cm.get_cmap(args.cmap)
    color_list = args.cmap.colors
    color_list = color_list[::int(len(color_list) / ncolbins)]  # truncating color_list in to a length of rougly ncolbins

    color_key = get_color_keys(Z_lim[0], Z_lim[1], color_list)

    fig_cm_profile2, ax2 = plt.subplots(figsize=(8, 8))
    fig_cm_profile2.subplots_adjust(right=0.95, top=0.9, bottom=0.12, left=0.17)

    artist = dsshow(df, dsh.Point('log_density', 'log_cell_mass'), dsh.mean('metallicity'), norm='linear', cmap=args.cmap, y_range=cm_lim, vmin=Z_lim[0], vmax=Z_lim[1], aspect = 'auto', ax=ax2, shade_hook=partial(dstf.spread, shape='circle', px=2)) # to make the main datashader plot

    ax2 = annotate_axes(r'Log density (gm/cm$^3$)', r'Log cell mass (M$_{\odot}$)', ax2, args) # to annotate axes

    plt.text(0.1, 0.15, 'z = %.2F' % args.current_redshift, ha='left', transform=ax2.transAxes, fontsize=args.fontsize)
    plt.text(0.1, 0.1, 't = %.1F Gyr' % args.current_time, ha='left', transform=ax2.transAxes, fontsize=args.fontsize)
    fig_cm_profile2 = saveplot(fig_cm_profile2, 'cm_vs_density_colorby_Z', args)
    plt.show(block=False)

    return df, fig_cm_profile2, fig_cm_profile

# -----main code-----------------
if __name__ == '__main__':
    dummy_args_tuple = parse_args('8508', 'RD0042')  # default simulation to work upon when comand line args not provided
    if type(dummy_args_tuple) is tuple:
        dummy_args = dummy_args_tuple[0]  # if the sim has already been loaded in, in order to compute the box center (via utils.pull_halo_center()), then no need to do it again
    else:
        dummy_args = dummy_args_tuple
    if not dummy_args.keep: plt.close('all')

    if dummy_args.do_all_sims:
        list_of_sims = get_all_sims_for_this_halo(dummy_args)  # all snapshots of this particular halo
    else:
        list_of_sims = list(itertools.product([dummy_args.halo], dummy_args.output_arr))
    total_snaps = len(list_of_sims)

    # -------set up dataframe and filename to store/write gradients in to--------
    weightby_text = '' if dummy_args.weight is None else '_wtby_' + dummy_args.weight
    if dummy_args.upto_kpc is not None:
        upto_text = '_upto%.1Fckpchinv' % dummy_args.upto_kpc if dummy_args.docomoving else '_upto%.1Fkpc' % dummy_args.upto_kpc
    else:
        upto_text = '_upto%.1FRe' % dummy_args.upto_re

    # --------domain decomposition; for mpi parallelisation-------------
    comm = MPI.COMM_WORLD
    ncores = comm.size
    rank = comm.rank
    print_master('Total number of MPI ranks = ' + str(ncores) + '. Starting at: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()), dummy_args)
    comm.Barrier()  # wait till all cores reached here and then resume

    split_at_cpu = total_snaps - ncores * int(total_snaps / ncores)
    nper_cpu1 = int(total_snaps / ncores)
    nper_cpu2 = nper_cpu1 + 1
    if rank < split_at_cpu:
        core_start = rank * nper_cpu2
        core_end = (rank + 1) * nper_cpu2 - 1
    else:
        core_start = split_at_cpu * nper_cpu2 + (rank - split_at_cpu) * nper_cpu1
        core_end = split_at_cpu * nper_cpu2 + (rank - split_at_cpu + 1) * nper_cpu1 - 1

    print_mpi('Operating on snapshots ' + str(core_start + 1) + ' to ' + str(core_end + 1) + ', i.e., ' + str(core_end - core_start + 1) + ' out of ' + str(total_snaps) + ' snapshots', dummy_args)

    # ----------------- looping over snapshots ---------------------------------------------
    for index in range(core_start + dummy_args.start_index, core_end + 1):
        start_time_this_snapshot = time.time()
        # ----------- reading in snapshot along with refinebox -------------------
        this_sim = list_of_sims[index]
        print_mpi('Doing snapshot ' + this_sim[1] + ' of halo ' + this_sim[0] + ' which is ' + str(index + 1 - core_start) + ' out of the total ' + str(core_end - core_start + 1) + ' snapshots...', dummy_args)
        halos_df_name = dummy_args.code_path + 'halo_infos/00' + this_sim[0] + '/' + dummy_args.run + '/' + 'halo_cen_smoothed'
        try:
            if len(list_of_sims) == 1 and not dummy_args.do_all_sims:
                args = dummy_args_tuple  # since parse_args() has already been called and evaluated once, no need to repeat it
            else:
                args = parse_args(this_sim[0], this_sim[1])

            if type(args) is tuple:
                args, ds, refine_box = args  # if the sim has already been loaded in, in order to compute the box center (via utils.pull_halo_center()), then no need to do it again
                print_mpi('ds ' + str(ds) + ' for halo ' + str(this_sim[0]) + ' was already loaded at some point by utils; using that loaded ds henceforth', args)
            else:
                ds, refine_box = load_sim(args, region='refine_box', do_filter_particles=True, disk_relative=False, halo_c_v_name=halos_df_name)
        except Exception as e:
            print_mpi('Skipping ' + this_sim[1] + ' because ' + str(e), dummy_args)
            continue

        # ---------- determine certain args parameters ---------------
        args.fig_dir = args.output_dir + 'figs/' if args.do_all_sims else args.output_dir + 'figs/' + args.output + '/'
        Path(args.fig_dir).mkdir(parents=True, exist_ok=True)

        args.current_redshift = ds.current_redshift
        args.current_time = ds.current_time.in_units('Gyr').v

        args.weight_text, args.upto_text = weightby_text, upto_text

        if args.upto_kpc is not None:
            if args.docomoving: args.galrad = args.upto_kpc / (1 + args.current_redshift) / 0.695  # fit within a fixed comoving kpc h^-1, 0.695 is Hubble constant
            else: args.galrad = args.upto_kpc  # fit within a fixed physical kpc
        else:
            args.galrad = args.re * args.upto_re  # kpc

        if args.galrad > 0:
            # extract the required box
            box_center = ds.halo_center_kpc
            box_width_kpc = 2 * args.galrad * kpc  # in kpc
            box = ds.r[box_center[0] - box_width_kpc / 2.: box_center[0] + box_width_kpc / 2., box_center[1] - box_width_kpc / 2.: box_center[1] + box_width_kpc / 2., box_center[2] - box_width_kpc / 2.: box_center[2] + box_width_kpc / 2.]

            dummy_field = ('deposit', 'young_stars_density') # dummy field just to create the FRB; any field can be extracted from the FRB thereafter

            # ---------- creating FRB from refinebox, based on desired resolution ---------------
            for args.res in args.res_arr:
                args.res_text = '_res%.1Fkpc' % args.res
                args.ncells = int(box_width_kpc / args.res)
                dummy_proj = ds.proj(dummy_field, args.projection, center=box_center, data_source=box)
                frb = dummy_proj.to_frb(box_width_kpc, args.ncells, center=box_center)

                # ---------- call various plotting routines with the frb ------------
                if args.plot_ks: fig_ks, fig_star_map, fig_gas_map, fig_sfr_map = plot_ks_relation(frb, args)
                if args.plot_Z: fig_Zprofile, fig_Zdistribution, fig_Z_map = plot_metallicity(frb, args)
                if args.plot_vel: fig_vel_profile, fig_vel_map, fig_vdisp_profile, fig_vdisp_map = plot_kinematics(frb, args)
                if args.plot_cm: df, fig_den_map, fig_cm_profile = plot_cellmass(box, args) # this is a intrinsic simulation quantity, not an observable, hence FRB not applicable

        print_mpi('This snapshots completed in %s mins' % ((time.time() - start_time_this_snapshot) / 60), dummy_args)


    if ncores > 1: print_master('Parallely: time taken for ' + str(total_snaps) + ' snapshots with ' + str(ncores) + ' cores was %s mins' % ((time.time() - start_time) / 60), dummy_args)
    else: print_master('Serially: time taken for ' + str(total_snaps) + ' snapshots with ' + str(ncores) + ' core was %s mins' % ((time.time() - start_time) / 60), dummy_args)
