##!/usr/bin/env python3

"""

    Title :      investigate_metallicity
    Notes :      Very crude script, to diagnose what's going on with metallicity maps upon co-adding HII regions along the LoS
    Output :     metallicity plots as png files
    Author :     Ayan Acharyya
    Started :    June 2021
    Example :    run investigate_metallicity.py --system ayan_local --halo 4123 --output RD0020 --mergeHII 0.04 --galrad 2 --base_spatial_res 0.06 --z 0.25 --obs_wave_range 0.8,0.85 --saveplot --keep --which_df H2_summed --Zout D16 --nooutliers --lookup_met 1287 --iscolorlog --islog --plotstyle hist --correlation

"""
from header import *
from util import *
from make_ideal_datacube import *
from foggie.galaxy_mocks.mock_ifu.lookup_flux import read_photoionisation_grid
import pickle

start_time = time.time()

# ---------------------------------------------
def load_HIIlist(args, linelist):
    '''
    Function to load the list of HII regions and then perform shift, incline, filter, etc. with the position coordinates
    '''
    paramlist = get_HII_list(args)
    paramlist = shift_ref_frame(paramlist, args)
    paramlist = incline(paramlist, args)  # coordinate transformation by args.inclination degrees in the YZ plane (keeping X fixed)
    paramlist = get_grid_coord(paramlist, args)

    paramlist = filter_h2list(paramlist, args, linelist)

    g = int(np.ceil(args.galrad * 2 / args.base_spatial_res))
    paramlist['cell_index'] = paramlist[xcol] + paramlist[ycol] * g

    return paramlist

# ---------------------------------------------
def get_operations_to_merge(df, countcol='cell_index', weightcol='Q_H0', cols_to_mean = [], cols_to_wtmean = ['lognII', 'age', 'logU', 'Zin', 'Zout_D16', 'Zout_PPN2', 'Zout_KD02', 'mass'], cols_to_sum = ['Q_H0', 'H6562', 'NII6584', 'SII6717', 'SII6730', 'OII3727', 'OII3729']):
    '''
    Function to gather the list of operations to perform on various columns while grouping several rows
    '''
    weighted_mean = lambda x: np.average(x, weights=df.loc[x.index, weightcol])
    sum_operations = {item: sum for item in cols_to_sum}
    mean_operations = {item: np.mean for item in cols_to_mean}
    weighted_mean_operations = {item: weighted_mean for item in cols_to_wtmean}
    all_operations = {**sum_operations, **mean_operations, **weighted_mean_operations, **{countcol:'count'}}
    return all_operations

# ---------------------------------------------
def makegridplot(photgrid, xcol, ycol, zcol, args, loopcol='logU'):
    '''
    Function to plot (crude)3D grid maps to visualise the photoionisation grid
    '''
    cmap_arr = ['Purples_r', 'Blues_r', 'Greens_r', 'Oranges_r', 'YlOrBr_r', 'PuRd_r']
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    if args.iscolorlog:
        photgrid['log_' + zcol] = np.log10(photgrid[zcol])
        zcol = 'log_' + zcol

    ax.set_xlabel(xcol), ax.set_ylabel(ycol), ax.set_zlabel(zcol)
    for index, thisloop in enumerate(np.sort(pd.unique(photgrid[loopcol]))):
        thislabel = loopcol + ' = ' + str(thisloop)
        myprint('Surface plotting for ' + zcol + ': ' + thislabel + '..', args)
        photgrid_slice = photgrid[photgrid[loopcol] == thisloop]
        data = photgrid_slice[zcol]
        data[np.isnan(data)] = np.nanmean(data) # to replace NaN values with an overall mean, otherwise 3D plot does not plot anything at all even if there's a single NaN
        #data[~np.isfinite(data)] = np.mean(data[np.isfinite(data)]) # to replace NaN values with an overall mean, otherwise 3D plot does not plot anything at all even if there's a single NaN
        ax.plot_trisurf(photgrid_slice[xcol], photgrid_slice[ycol], data, cmap=cmap_arr[index], linewidth=0.5, label=thislabel)
    #ax.legend()

    if args.saveplot:
        nooutlier_text = '_no_outlier' if args.nooutliers else ''
        fig_output_dir = os.path.split(args.idealcube_filename)[0].replace('/fits/', '/figs/') + '/'
        rootname = os.path.splitext(os.path.split(args.idealcube_filename)[1])[0].replace('ideal_ifu', '3D_grid')
        figname = fig_output_dir + rootname + '_' + xcol + '_vs_' + ycol + '_vs_' + zcol + '_colby_' + loopcol + '_at_age=' + str(args.age_slice) + 'Myr' + nooutlier_text
        pickle.dump(fig, open(figname, 'wb'))
        myprint('Saved 3D plot as pickle at ' + figname, args)

    plt.show(block=False)

    return ax

# ---------------------------------------------
def makeplot(df_orig, colorcol, args, xcol=None, ycol=None):
    '''
    Function to plot (crude) metallicity maps and scatter plots
    '''
    df = df_orig.copy()
    if colorcol in df.columns: df = df.sort_values(by=colorcol, ascending=True) # to make color-coded scatter points a bit easier to visualise, by setting the sequence of overplotting of different points
    if xcol is None: xcol = 'pos_' + projection_dict[args.projection][0] + '_grid'
    if ycol is None: ycol = 'pos_' + projection_dict[args.projection][1] + '_grid'

    if args.swap_axes: # swap x and y axes
        temp = xcol
        xcol = ycol
        ycol = temp

    myprint('Plotting ' + xcol + ' vs ' + ycol + ' colored by ' + colorcol + '..', args)

    cmap_dict = defaultdict(lambda: metal_color_map, Z_ratio_upon_sum= 'coolwarm', Z_ratio_upon_diag='coolwarm', H6562= density_color_map, count_h2= 'Blues', age='Blues', logU='Blues', lognII='Blues', logq= 'Blues', logP_k= 'Blues', frequency='Blues', Q_H0='Blues', mass='Blues')
    cmax_dict = defaultdict(lambda: 2.2, count_h2= 1.7, Z_ratio_upon_sum= 2,  Z_ratio_upon_diag= 2, H6562= -15, age= 10, logU= -1, lognII= 10, logq= 8.5, logP_k= 9, Q_H0=53, mass=6.0)
    cmin_dict = defaultdict(lambda: -0.6, count_h2= 0, Z_ratio_upon_sum= -2, Z_ratio_upon_diag= -2, H6562= -18, age= 0, logU= -4, lognII= 6, logq= 6.5, logP_k= 4, Q_H0=47, mass=2.9)
    hist_color = 'darkblue'

    if args.islog:
        df[xcol] = np.log10(df[xcol])
        df[ycol] = np.log10(df[ycol])
    if args.iscolorlog:
        df[colorcol] = np.log10(df[colorcol])

    df = df.replace([-np.inf, np.inf], np.nan).dropna()#subset=[xcol, ycol, colorcol], how="all")

    if args.plotstyle == 'map':
        fig, ax = plt.subplots()
        gridsize = int(np.ceil(args.galrad * 2 / args.base_spatial_res))
        colordata = np.zeros((gridsize, gridsize)) * np.NaN # declare empty array of NaN values; can't initialise with 0 as 0 is a meaningful number here
        for index, row in df.iterrows(): colordata[int(row[xcol]), int(row[ycol])] = row[colorcol]
        p = plt.imshow(np.transpose(colordata), origin='lower', cmap=cmap_dict[colorcol], vmin=cmin_dict[colorcol], vmax=cmax_dict[colorcol])
        cb = plt.colorbar(p).ax

    else:
        x, y = df[xcol], df[ycol]
        axes = sns.JointGrid(xcol, ycol, df, height=7)
        plt.subplots_adjust(hspace=0.1, wspace=0.1, right=0.85, top=0.95, bottom=0.1, left=0.1)
        ax, fig = axes.ax_joint, plt.gcf()

        sns.kdeplot(x, ax=axes.ax_marg_x, legend=False, color=hist_color, lw=1)
        sns.kdeplot(y, ax=axes.ax_marg_y, vertical=True, legend=False, color=hist_color, lw=1)
        axes.ax_marg_x.tick_params(axis='x', which='both', top=False)
        axes.ax_marg_y.tick_params(axis='y', which='both', right=False)

        if 'hex' in args.plotstyle: p = ax.hexbin(x, y, alpha=0.9, cmap=cmap_dict[colorcol], gridsize=(20, 20))
        elif 'contour' in args.plotstyle: p = sns.kdeplot(x, y, ax=ax, shade=False, shade_lowest=False, alpha=1, n_levels=5, palette=cmap_dict[colorcol])
        elif 'hist' in args.plotstyle: p = sns.histplot(x=x, y=y, ax=ax, alpha=0.9, palette=cmap_dict[colorcol], cbar=True)
        elif 'scatter' in args.plotstyle : p = ax.scatter(x, y, s=30, c=df[colorcol], cmap=cmap_dict[colorcol], vmin=cmin_dict[colorcol], vmax=cmax_dict[colorcol], ec='none')
        else: print('Could not find indicated plot style. Please re-try by setting --plotstyle to one of the following: scatter, density OR contour')

        if 'hex' in args.plotstyle or 'scatter' in args.plotstyle: cb = plt.colorbar(p).ax
        elif 'contour' in args.plotstyle or 'hist' in args.plotstyle: cb = p.figure.axes[-1]

        # get the current positions of the joint ax and the ax for the marginal x
        pos_joint_ax = ax.get_position()
        pos_marg_x_ax = axes.ax_marg_x.get_position()
        # reposition the joint ax so it has the same width as the marginal x ax
        ax.set_position([pos_joint_ax.x0, pos_joint_ax.y0, pos_marg_x_ax.width, pos_joint_ax.height])
        # reposition the colorbar using new x positions and y positions of the joint ax
        cb.set_position([.83, pos_joint_ax.y0, .07, pos_joint_ax.height])

        if args.correlation:
            coef_arr = stats.pearsonr(x,y)[0] #np.corrcoef(x, y)[0][1]) # Pearson product-moment correlation coefficients
            ax.text(ax.get_xlim()[-1] * 0.95, ax.get_ylim()[-1] * 0.85, r'$\rho$ = ' + str(round(coef_arr, 2)), color=hist_color, ha='right', fontsize=args.fontsize)

    ax.set_xlabel('log ' + xcol if args.islog else xcol, fontsize=args.fontsize)
    ax.set_ylabel('log ' + ycol if args.islog else ycol, fontsize=args.fontsize)
    if 'cb' in locals(): cb.set_ylabel('log ' + colorcol if args.iscolorlog else colorcol, fontsize=args.fontsize)
    fig.text(0.1, 0.9, args.which_df, va='top', ha='left', c='k', fontsize=args.fontsize, transform=ax.transAxes)

    if ('Zin' in xcol or 'Zout' in xcol) and ('Zin' in ycol or 'Zout' in ycol): # for a Z vs Z comparison plot
        Zlim = [0.01, 5.5]
        if args.islog:
            Zlim = ax.get_ylim() if args.swap_axes else ax.get_xlim()
        else:
            ax.set_xlim(Zlim)
            ax.set_ylim(Zlim)
        ax.plot(Zlim, Zlim, c='k', lw=1)

    if args.saveplot:
        if args.islog:
            xcol = 'log_' + xcol
            ycol = 'log_' + ycol
        if args.iscolorlog:
            colorcol = 'log_' + colorcol
        fig_output_dir = os.path.split(args.idealcube_filename)[0].replace('/fits/', '/figs/') + '/'
        rootname = os.path.splitext(os.path.split(args.idealcube_filename)[1])[0].replace('ideal_ifu', args.which_df)
        saveplot(fig, args, rootname + '_' + args.plotstyle + '_' + xcol + '_vs_' + ycol + '_colby_' + colorcol, outputdir=fig_output_dir)

    plt.show(block=False)
    return ax

# ---------------------------------------------
def get_metallicity(index, paramlist, ifunc_LND, ifunc_RGI, linelist, quant1, quant2, quant3, quant4):
    '''
    Function to lookup photoionisation grid to assign fluxes to a single HII region based on its parameters and then re-compute the metallicity based on those fluxes
    Requires the interpolation function (based on the photoionisation grid) as an input); this is to make the sure the interp function is computed only once and NOT every time this function is called
    '''
    thisrow = paramlist.loc[index]
    coord = np.vstack([thisrow[quant1], thisrow[quant2], thisrow[quant3], thisrow[quant4]]).transpose()
    print('HII region', index, 'has parameters', quant1, quant2, quant3, quant4, '=', coord)

    print('\nBased on LND interpolation it has fluxes..')
    for ind, label in enumerate(linelist['label']):
        thisflux = (10 ** ifunc_LND[ind](coord)) * thisrow['mass'] / mappings_starparticle_mass
        thisrow[label] = thisflux
        print(label, 'flux =', thisflux, 'ergs/s')
    Zout_LND = get_D16_metallicity(thisrow)
    print('Using these fluxes as input to D16, the output metallicity is', Zout_LND, 'Z/Zun =', np.log10(Zout_LND), 'in log units')

    print('\nBased on RGI interpolation it has fluxes..')
    for ind, label in enumerate(linelist['label']):
        thisflux = (10 ** ifunc_RGI[ind](coord)) * thisrow['mass'] / mappings_starparticle_mass
        thisrow[label] = thisflux
        print(label, 'flux =', thisflux, 'ergs/s')
    Zout_RGI = get_D16_metallicity(thisrow)
    print('Using these fluxes as input to D16, the output metallicity is', Zout_RGI, 'Z/Zun =', np.log10(Zout_RGI), 'in log units')

    return Zout_LND, Zout_RGI

# -----main code-----------------
if __name__ == '__main__':
    # parse args and load in sim
    args = parse_args('8508', 'RD0042')
    if type(args) is tuple:
        args, ds, refine_box = args  # if the sim has already been loaded in, in order to compute the box center (via utils.pull_halo_center()), then no need to do it again
        myprint('ds ' + str(ds) + ' for halo ' + str(args.halo) + ' was already loaded at some point by utils; using that loaded ds henceforth', args)
    if not args.keep: plt.close('all')

    xcol = 'pos_' + projection_dict[args.projection][0] + '_grid'
    ycol = 'pos_' + projection_dict[args.projection][1] + '_grid'

    relevant_lines = ['H6562', 'NII6584', 'SII6717', 'SII6730', 'OII3727', 'OII3729']
    linelist = read_linelist(mappings_lab_dir + 'targetlines.txt')  # list of emission lines
    linelist_sub = linelist[linelist['label'].isin(relevant_lines)]
    instrument = telescope(args) # declare an instrument
    ifu = idealcube(args, instrument, linelist) # declare a cube object

    # -----read in main HII list-----------------
    paramlist = load_HIIlist(args, ifu.linelist)

    paramsub = paramlist[['cell_index', xcol, ycol, 'Zin', 'lognII', 'age', 'logU', 'Q_H0', 'mass'] + relevant_lines]
    paramsub['Zout_D16'] = get_D16_metallicity(paramsub)
    paramsub['Zout_PPN2'] = get_PPN2_metallicity(paramsub)
    paramsub['Zout_KD02'] = get_KD02_metallicity(paramsub)

    # -----read in Lisa's photionisation grid------------
    lisa_grid_file = HOME + '/Desktop/Lisa_UV_diag/Mappings_P_pp_ratio_grid.txt'
    lisa_grid = pd.read_table(lisa_grid_file, delim_whitespace=True, comment="#")
    logOHsol = 8.72 # from Kewley+19a (pressure paper)
    lisa_grid['Zin'] = 10 ** (lisa_grid['Z'] - logOHsol)
    lisa_grid.rename(columns={'log(q)': 'logq', 'log(P/k)': 'logP_k', '[NII]6584/Halpha': 'N2Ha', '[NII]6584/[SII]6717,31': 'N2S2', '[NII]6584/[OII]3727,29':'N2O2'}, inplace=True)

    # -------computing the D16 and PPN2 metallicities-------
    lisa_grid['Zout_D16'] = get_D16_metallicity_from_ratios(lisa_grid)
    lisa_grid['Zout_PPN2'] = get_PPN2_metallicity_from_ratios(lisa_grid)
    lisa_grid['Zout_KD02'] = get_KD02_metallicity_from_ratios(lisa_grid)

    # -----read in model photoionisation grid-----------------
    mappings_grid_file = 'totalspec' + mmg.outtag + '.txt'
    mappings_starparticle_mass = mmg.mappings_starparticle_mass

    photgrid = read_photoionisation_grid(mappings_lab_dir + mappings_grid_file)
    photgrid.rename(columns={'Z': 'Zin', 'KD02': 'Zout_KD02', 'D16': 'Zout_D16'}, inplace=True)
    photgrid[relevant_lines] = photgrid[relevant_lines].replace(0, 1e-5) # to get rid of flux == 0 values, otherwise they lead to math exceptions later
    photgrid['Zout_PPN2'] = get_PPN2_metallicity(photgrid)

    choice_arr = [1, 3, 2, 4]  # choose the sequence of variables to interpolate in 4D
    quantity_dict = {1: mmg.Z_arr, 2: mmg.age_arr, 3: mmg.lognII_arr, 4: mmg.logU_arr}
    quant_dict = {1: 'Zin', 2: 'age', 3: 'lognII', 4: 'logU'}  # has to correspond to the same sequence as quantity_dict etc.

    quant1, quant2, quant3, quant4 = [quant_dict[item] for item in choice_arr]
    quantity1, quantity2, quantity3, quantity4 = [quantity_dict[item] for item in choice_arr]

    # -----removing outliers as per D16-----------------
    if args.nooutliers:
        lpok_min, lpok_max = 5.2, 6.7  # from Dopita+16
        #lpok_min, lpok_max = 5, 7  # close approximation of above; just as an experiment
        lognII_min, lognII_max = lpok_min -4 +6, lpok_max -4 + 6
        lisa_grid_initial_length, photgrid_initial_length, paramsub_initial_length = len(lisa_grid), len(photgrid), len(paramsub)

        lisa_grid = lisa_grid[lisa_grid['logP_k'].between(lpok_min, lpok_max)]
        print('Upon removing outliers beyond log(P/k)=', lpok_min, 'and', lpok_max, 'Lisas 3D grid loses %.2F %% grid cells'%((1 - len(lisa_grid) / lisa_grid_initial_length) * 100))

        photgrid = photgrid[photgrid['lognII'].between(lognII_min, lognII_max)]
        print('Upon removing outliers beyond lognII=', lognII_min, 'and', lognII_max, 'my 4D grid loses %.2F %% grid cells'%((1 - len(photgrid) / photgrid_initial_length) * 100))

        paramsub = paramsub[paramsub['lognII'].between(lognII_min, lognII_max)]
        print('Upon removing outliers beyond lognII=', lognII_min, 'and', lognII_max, 'HII list loses %.2F %% particles'%((1 - len(paramsub) / paramsub_initial_length) * 100))

    # -----group by HII along LoS-----------------
    all_operations = get_operations_to_merge(paramsub)
    paramsum = paramsub.groupby([ xcol, ycol, 'cell_index'], as_index=False).agg(all_operations)
    myprint('Lengths before and after summing %d %d \n'%(len(paramsub), len(paramsum)), args)

    paramsum['Zout_D16_after_sum'] = get_D16_metallicity(paramsum)
    paramsum['Zout_PPN2_after_sum'] = get_PPN2_metallicity(paramsum)
    paramsum['Zout_KD02_after_sum'] = get_KD02_metallicity(paramsum)
    paramsum = paramsum.rename(columns={'cell_index': 'count_h2'})
    paramsum['Z_ratio_upon_diag'] = paramsum['Zout_' + args.Zout] / paramsum['Zin']
    paramsum['Z_ratio_upon_sum'] = paramsum['Zout_' + args.Zout + '_after_sum'] / paramsum['Zout_' + args.Zout]
    for col in relevant_lines: paramsum[col] /= 4 * np.pi * (ifu.distance * Mpc_to_cm)**2 # converting from ergs/s to ergs/s/cm^2 for the summed df
    paramsum = paramsum.sort_values(by='Z_ratio_upon_sum', ascending=False).reset_index(drop=True)

    # -----removing outliers as per D16-----------------
    if args.nooutliers:
        paramsum_initial_length = len(paramsum)
        paramsum = paramsum[paramsum['lognII'].between(lognII_min, lognII_max)]
        print('Upon removing outliers beyond lognII=', lognII_min, 'and', lognII_max, 'summed HII list loses %.2F %% particles'%((1 - len(paramsum) / paramsum_initial_length) * 100))
    '''
    # ----------temporary----------------
    Zout_highcut = 10**0.8
    paramsub_initial_length = len(paramsub)
    paramsub = paramsub[paramsub['Zout_' + args.diag] <= Zout_highcut]  ## tmeporary
    print('Upon removing outliers beyond Zout_%s >'%args.diag, Zout_highcut, 'HII list loses %.2F %% particles' % ((1 - len(paramsub) / paramsub_initial_length) * 100))
    '''
    # -----plot stuff-----------------
    if not args.noplot:
        if args.which_df == 'H2_all':
            # ---------Zin vs Zout for HII regions-----------
            if args.plotstyle == 'scatter': # i.e., color-coded by a 3rd variable
                if args.iscolorlog:
                    ax_Zout_vs_Zin = makeplot(paramsub, 'mass', args, xcol='Zin', ycol='Zout_' + args.Zout)
                    ax_Zout_vs_Zin = makeplot(paramsub, 'Q_H0', args, xcol='Zin', ycol='Zout_' + args.Zout)
                else:
                    ax_Zout_vs_Zin = makeplot(paramsub, 'age', args, xcol='Zin', ycol='Zout_' + args.Zout)
                    ax_Zout_vs_Zin = makeplot(paramsub, 'logU', args, xcol='Zin', ycol='Zout_' + args.Zout)
                    ax_Zout_vs_Zin = makeplot(paramsub, 'lognII', args, xcol='Zin', ycol='Zout_' + args.Zout)
            else: # color-coded by density of points
                ax_Zout_vs_Zin = makeplot(paramsub, 'frequency', args, xcol='Zin', ycol='Zout_' + args.Zout)

        elif args.which_df == 'H2_summed':
            if args.plotstyle == 'map':
                args.islog, args.iscolorlog = False, True
                # ---------2D maps-----------
                ax_Zin = makeplot(paramsum, 'Zin', args)
                ax_Zout = makeplot(paramsum, 'Zout_' + args.Zout, args)
                ax_Z_ratio_upon_sum = makeplot(paramsum, 'Z_ratio_upon_diag', args)
                ax_Zout_after_sum = makeplot(paramsum, 'Zout_' + args.Zout + '_after_sum', args)
                ax_Z_ratio_upon_sum = makeplot(paramsum, 'Z_ratio_upon_sum', args)
                ax_count = makeplot(paramsum, 'count_h2', args)
                ax_mass = makeplot(paramsum, 'mass', args)
                ax_Q_H0 = makeplot(paramsum, 'Q_H0', args)
                ax_Halpha = makeplot(paramsum, 'H6562', args)
            elif args.plotstyle == 'scatter': # i.e., color-coded by a 3rd variable
                '''
                # ---------relative Z discrapancy vs individual parameters-----------
                ax_logU_vs_Z_ratio_upon_sum = makeplot(paramsum, 'count_h2', args, xcol='logU', ycol='Z_ratio_upon_sum')
                ax_lognII_vs_Z_ratio_upon_sum = makeplot(paramsum, 'count_h2', args, xcol='lognII', ycol='Z_ratio_upon_sum')
                ax_age_vs_Z_ratio_upon_sum = makeplot(paramsum, 'count_h2', args, xcol='age', ycol='Z_ratio_upon_sum')
                ax_Zin_vs_Z_ratio_upon_sum = makeplot(paramsum, 'count_h2', args, xcol='Zin', ycol='Z_ratio_upon_sum')
                '''
                # ---------Zout vs Zout for SUMMED HII regions-----------
                ax_Zout_after_sum_vs_Zout = makeplot(paramsum, 'age', args, xcol='Zout_' + args.Zout, ycol='Zout_' + args.Zout + '_after_sum')
                ax_Zout_after_sum_vs_Zout = makeplot(paramsum, 'logU', args, xcol='Zout_' + args.Zout, ycol='Zout_' + args.Zout + '_after_sum')
                ax_Zout_after_sum_vs_Zout = makeplot(paramsum, 'lognII', args, xcol='Zout_' + args.Zout, ycol='Zout_' + args.Zout + '_after_sum')
                ax_Zout_after_sum_vs_Zout = makeplot(paramsum, 'count_h2', args, xcol='Zout_' + args.Zout, ycol='Zout_' + args.Zout + '_after_sum')
            else: # color-coded by density of points
                ax_Zout_after_sum_vs_Zout = makeplot(paramsum, 'frequency', args, xcol='Zout_' + args.Zout, ycol='Zout_' + args.Zout + '_after_sum')

        elif args.which_df == '4D_grid':
            # ---------Zin vs Zout for my 4D photoionisation grid-----------
            if args.plotstyle == 'scatter': # i.e., color-coded by a 3rd variable
                ax_Zout_vs_Zin_grid = makeplot(photgrid, 'age', args, xcol='Zin', ycol='Zout_' + args.Zout)
                ax_Zout_vs_Zin_grid = makeplot(photgrid, 'logU', args, xcol='Zin', ycol='Zout_' + args.Zout)
                ax_Zout_vs_Zin_grid = makeplot(photgrid, 'lognII', args, xcol='Zin', ycol='Zout_' + args.Zout)
            else: # color-coded by density of points
                ax_Zout_vs_Zin_grid = makeplot(photgrid, 'frequency', args, xcol='Zin', ycol='Zout_' + args.Zout)

            if args.plot3d:
                # ---------3D plots for my 4D photoionisation grid-----------
                ax_Zin_vs_lognII_vs_N2S2 = makegridplot(photgrid[photgrid['age'] == args.age_slice], 'Zin', 'lognII', 'N2S2', args, loopcol='logU')
                ax_Zin_vs_lognII_vs_N2Ha = makegridplot(photgrid[photgrid['age'] == args.age_slice], 'Zin', 'lognII', 'N2Ha', args, loopcol='logU')

        elif args.which_df == '3D_grid':
            # ---------Zin vs Zout for Lisa's 3D photoionisation grid-----------
            if args.plotstyle == 'scatter': # i.e., color-coded by a 3rd variable
                ax_Zout_vs_Zin_grid = makeplot(lisa_grid, 'logq', args, xcol='Zin', ycol='Zout_' + args.Zout)
                ax_Zout_vs_Zin_grid = makeplot(lisa_grid, 'logP_k', args, xcol='Zin', ycol='Zout_' + args.Zout)
            else: # color-coded by density of points
                ax_Zout_vs_Zin_grid = makeplot(lisa_grid, 'frequency', args, xcol='Zin', ycol='Zout_' + args.Zout)

        else:
            print('Cannot identify indicated dataframe. Re-try by setting --which_df to one of these: H2_all, H2_summed, 4D_grid OR 3D-grid.')

    # -----look up metallicity of a given HII region from the grid-----------------
    if args.lookup_metallicity is not None:
        ifunc_LND, ifunc_RGI = [], []
        myprint('Interpolating 4D in the sequence: ' + quant1 + ', ' + quant2 + ', ' + quant3 + ', ' + quant4, args)
        for label in linelist_sub['label']:
            l = np.reshape(np.array(np.log10(photgrid[label])), (len(quantity1), len(quantity2), len(quantity3), len(quantity4)))
            iff_RGI = RGI((quantity1, quantity2, quantity3, quantity4), l)
            ifunc_RGI.append(iff_RGI)

            iff_LND = LND(np.array(photgrid[[quant1, quant2, quant3, quant4]]), np.log10(photgrid[label]))
            ifunc_LND.append(iff_LND)

        Zout_LND, Zout_RGI = get_metallicity(args.lookup_metallicity, paramsub, ifunc_LND, ifunc_RGI, linelist_sub, quant1, quant2, quant3, quant4)

    myprint('Complete in %s minutes' % ((time.time() - start_time) / 60), args)
