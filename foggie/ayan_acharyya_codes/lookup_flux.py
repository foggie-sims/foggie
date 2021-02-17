##!/usr/bin/env python3

"""

    Title :      lookup_flux
    Notes :      To compute (and optionally plot) fluxes of HII regions for various emission lines (and output to an ASCII file) using MAPPINGS photoionisation model grid
    Output :     One pandas dataframe as a txt file
    Author :     Ayan Acharyya
    Started :    January 2021
    Example :    run lookup_flux.py --system ayan_local --halo 8508 --output RD0042 --plot_phase_space --plot_fluxgrid --plot_metgrad --plot_Zin_Zout # optional plotting arguments

"""
from header import *
import make_mappings_grid as mmg
reload(mmg)

# ------------------------------------------------------------------------------------------------------------------
def find_nearest(array, value):
    '''
    Function to find element in array nearest to a given value
    '''

    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

# ---------------------------------------------------------------------------------------------
def getOmega(radius, scale_length = 4.):
    '''
    Function to derive Omega for each HII region, given its galactocentric diatance
    There are 11 uniformly spaced points in log-scale, from Om=0.05 to )m = 5.0, both inclusive,
    to account for radially varying Omega from 5 at r=0 to 0.05 at r > 2R_e
    log Omega = max[log(5) - r/R_e, log(0.05)].
    Hence, Omega = 5 at r = 0, Omega = 0.5 at r = R_e, and Omega = 0.05 at r >= 2 R_e
    '''

    Omega = 10**(np.max([np.log10(5) - radius/scale_length, np.log10(0.05)])) # radius and scale_length in kpc
    return Omega

# --------------------------------------------------------
def calc_n2(Om, r, Q_H0):
    '''
    Function to calculate HII region density
    '''

    return (np.sqrt(3 * Q_H0 * (1 + Om) / (4 * np.pi * mmg.alpha_B * r ** 3)))

# -----------------------------------------------
def calc_U(Om, nII, Q_H0):
    '''
    Function to calculate volume averaged ionisation parameter
    '''

    return mmg.func(nII, np.log10(Q_H0)) * ((1 + Om) ** (4 / 3.) - (4. / 3. + Om) * (Om ** (1 / 3.)))

# --------------------------------------------------------
def calc_Z(r, logOHcen, logOHgrad, logOHsun):
    '''
    Function to "paint" metallicity gradient
    '''

    Z = 10 ** (logOHcen - logOHsun + logOHgrad * r)  # assuming central met logOHcen and gradient logOHgrad dex/kpc, logOHsun = 8.77 (Dopita 16)
    return Z

# ----------------------------------------------------------
def remove_star(indices, list_of_var):
    '''
    Function to delete a certain HII region from all lists
    '''

    new_list_of_var = []
    for list in list_of_var:
        list = np.delete(list, indices, 0)
        new_list_of_var.append(list)
    return new_list_of_var

# -----------------------------------------------------------------
def poly(x, R, k):
    '''
    Function to use KD02 R23 diagnostic for the upper Z branch
    '''

    return np.abs(np.poly1d(k)(x) - np.log10(R))


# --------------------------------------------------------------------------
def get_KD02_metallicity(photgrid):
    '''
    Function to compute KD02 metallicity
    '''

    log_ratio = np.log10(np.divide(photgrid['NII6584'], (photgrid['OII3727'] + photgrid['OII3729'])))
    logOH = 1.54020 + 1.26602 * log_ratio + 0.167977 * log_ratio ** 2
    Z = 10 ** logOH  # converting to Z (in units of Z_sol) from log(O/H) + 12
    return Z

# ---------------------------------------------------------------------------------
def get_D16_metallicity(photgrid):
    '''
    Function to compute D16 metallicity
    '''

    log_ratio = np.log10(np.divide(photgrid['NII6584'], (photgrid['SII6730'] + photgrid['SII6717']))) + 0.264 * np.log10(np.divide(photgrid['NII6584'], photgrid['H6562']))
    logOH = log_ratio + 0.45 * (log_ratio + 0.3) ** 5  # + 8.77
    Z = 10 ** logOH  # converting to Z (in units of Z_sol) from log(O/H) + 12
    return Z

# ------------------------------------------------------------------------
def read_photoionisation_grid(gridfilename):
    '''
    Function to read in photoionisation model grid
    '''

    photgrid = pd.read_table(gridfilename, comment='#', delim_whitespace=True)
    if photgrid['age'].max() > 100.: photgrid['age'] /= 1e6 # convert age to units of Myr if already not so; if it is already in units of Myr then it is unlikely to have ages > 100 Myr
    photgrid['lognII'] = np.log10(photgrid['nII'])
    photgrid['logU'] = np.log10(photgrid['<U>'])

    # --------------add metallicity columns for diagnostic purposes-----------
    photgrid['KD02'] = get_KD02_metallicity(photgrid)
    photgrid['D16'] = get_D16_metallicity(photgrid)
    return photgrid

# ------------------------------------------------------------------
def saveplot(fig, args, plot_suffix):
    '''
    Function to save plots with a consistent nomenclature
    '''

    outplotname = args.output_dir + 'figs/' + args.output + args.mergeHII_text + args.without_outlier + plot_suffix + '.png'
    fig.savefig(outplotname)
    print('Saved plot as', outplotname)

# ---------------------------------------------------------------
def makeplot_phase_space(paramlist, args, plot_suffix=''):
    '''
    Function to plot the P-r phase space of HII regions
    '''

    fig = plt.figure()

    paramlist['P'] = 10 ** paramlist['log(P/k)'] * 1.38e-16  # converting P/k in CGS units to pressure in dyne/cm^2
    paramlist['logr'], paramlist['logP'] = np.log10(paramlist['r']), np.log10(paramlist['P'])

    marker_model, color_model = 'o', 'gray' # paramlist['logQ'].values #
    plt.scatter(paramlist['r'], paramlist['P'], c=color_model, lw=0, alpha=0.3, marker=marker_model)

    plt.xlabel('HII region size [pc]', fontsize=args.fontsize)
    plt.ylabel('HII region pressure [dyne/cm^2]', fontsize=args.fontsize)
    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(1e-12, 1e-7)
    plt.hlines([1e-11, 1e-9], 0.6, 25., color='k', linestyles='dashed') # plotting range of Fig 2 of  Tremblin+2014
    plt.vlines([0.6, 25.], 1e-11, 1e-9, color='k', linestyles='dashed')
    ax.text(1.8, 10**-8.8, 'Plot window of Tremblin+14, Fig 2', color='k', fontsize=args.fontsize)
    if isinstance(color_model, (list, tuple, np.ndarray)):
        cb = plt.colorbar()
        cb.set_label('Luminosity [log(s^-1)]', fontsize=args.fontsize)

    # ------------for plotting Tremblin+14 Fig 2 data--------
    if args.plot_obsv_phase_space:
        marker_obs, color_obs = '*', 'green'
        tremblin = pd.read_csv(HOME + '/Dropbox/papers/enzo_paper/AA_Working/Tremblin2014_Fig2_Dataset.csv', comment='#', names=['r', 'P'], header=None)
        ax.scatter(tremblin['r'], tremblin['P'], c=color_obs, marker=marker_obs, lw=0, alpha=0.7, s=200)

    if args.saveplot: saveplot(fig, args, plot_suffix)
    plt.show(block=False)
    return fig

# --------------------------------------------------------------
def makeplot_fluxgrid(paramlist, photgrid, args, quant1='Z', quant2='nII', quant3='age', quant4='logU', plot_suffix=''):
    '''
    Function to plot grid of model flux ratios, overlaid with interpolated fluxes: for diagnostic purposes
    '''

    if args.xratio is None: args.xratio = 'NII6584/H6562'
    if args.yratio is None: args.yratio = 'NII6584/SII6717,SII6730'

    if '/' in args.xratio: photgrid[args.xratio] = np.array([photgrid[item] for item in args.xratio.split('/')[0].split(',')]).sum(axis=0) / np.array([photgrid[item] for item in args.xratio.split('/')[1].split(',')]).sum(axis=0)
    if '/' in args.yratio: photgrid[args.yratio] = np.array([photgrid[item] for item in args.yratio.split('/')[0].split(',')]).sum(axis=0) / np.array([photgrid[item] for item in args.yratio.split('/')[1].split(',')]).sum(axis=0)

    fig = plt.figure(figsize=(8, 8))
    ax_fluxgrid = plt.gca()

    arr1 = np.unique(photgrid[quant1])
    arr2 = np.unique(photgrid[quant2])
    arr3 = np.unique(photgrid[quant3])
    arr4 = pd.unique(photgrid[quant4])
    ls_dict = {0:'-.', 1:'--', 2:':', 3:'-', 4:'-', 5:'--'}

    for index,thisquant4 in enumerate(arr4):
        subset = photgrid[photgrid[quant4] == thisquant4]

        x = np.reshape(np.log10(subset[args.xratio].values), (len(arr3), len(arr2), len(arr1)))
        y = np.reshape(np.log10(subset[args.yratio].values), (len(arr3), len(arr2), len(arr1)))
        lim = np.shape(x)[0]
        ls = ls_dict[index]

        for i in range(0, lim):
            for k in range(0, np.shape(x)[2]):
                plt.plot(x[i, :, k], y[i, :, k], c='red', lw=0.5, ls=ls)
                if args.annotate and i == lim - 1: plt.annotate(quant1 + '=' + str(arr1[k]), xy=(x[i, -1, k], y[i, -1, k]), xytext=(x[i, -1, k] - 0.1, y[i, -1, k] + 0.1),color='red',fontsize=10,arrowprops=dict(arrowstyle="->", connectionstyle="arc3",color='red'))
                if args.pause: plt.pause(1)
            for k in range(0, np.shape(x)[1]):
                plt.plot(x[i, k, :], y[i, k, :], c='blue', lw=0.5, ls=ls)
                if args.annotate and i == lim - 1: plt.annotate(quant2 + '=' + str(arr2[k]),xy=(x[i, k, 0], y[i, k, 0]),xytext=(x[i, k, 0] - 0.2, y[i, k, 0] - 0.1),color='blue',fontsize=10,arrowprops=dict(arrowstyle="->", connectionstyle="arc3",color='blue'))
                if args.pause: plt.pause(1)
            if args.annotate: plt.annotate(quant3 + '=' + str(arr3[i]), xy=(x[i, 0, -1], y[i, 0, -1]), xytext=(x[i, 0, -1] + 0.3, y[i, 0, -1] - 0.2), color='black', fontsize=10,arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color='black'))

        for i in range(0, np.shape(x)[1]):
            for j in range(0, np.shape(x)[2]):
                plt.plot(x[:, i, j], y[:, i, j], c='black', lw=0.5, ls=ls)
                if args.pause: plt.pause(1)

    plt.xlabel('log('+args.xratio+')')
    plt.ylabel('log('+args.yratio+')')
    if '/' in args.xratio: plt.xlim(-3.5, 0.5)
    if '/' in args.yratio: plt.ylim(-1, 2)

    if '/' in args.xratio: paramlist[args.xratio] = np.array([paramlist[item] for item in args.xratio.split('/')[0].split(',')]).sum(axis=0) / np.array([paramlist[item] for item in args.xratio.split('/')[1].split(',')]).sum(axis=0)
    if '/' in args.yratio: paramlist[args.yratio] = np.array([paramlist[item] for item in args.yratio.split('/')[0].split(',')]).sum(axis=0) / np.array([paramlist[item] for item in args.yratio.split('/')[1].split(',')]).sum(axis=0)
    ax_fluxgrid.scatter(np.log10(paramlist[args.xratio]), np.log10(paramlist[args.yratio]), c='k', lw=0, alpha=0.3)

    if args.saveplot: saveplot(fig, args, plot_suffix + '_' + args.xratio.replace('/', '|') + '_' + args.yratio.replace('/', '|'))
    plt.show(block=False)
    return ax_fluxgrid

# --------------------------------------------------------------------
def makeplot_metgrad(paramlist, Zoutcol, args, plot_suffix=''):
    '''
    Function to plot metallicity gradient given a list of HII regions
    '''

    fig = plt.figure()

    plt.scatter(paramlist['radial_dist'], np.log10(paramlist['Zin']), c='k', lw=0, alpha=0.5)
    plt.scatter(paramlist['radial_dist'], np.log10(paramlist[Zoutcol]), c=paramlist['logU'] + np.log10(3e10), lw=0, alpha=0.2)
    linefit = np.polyfit(paramlist['radial_dist'], np.log10(paramlist[Zoutcol]), 1, cov=False)
    x_arr = np.arange(args.galrad)
    plt.plot(x_arr, np.poly1d(linefit)(x_arr), c='b', label='Fitted gradient=' + str('%.4F' % linefit[0]))

    plt.xlabel('Galactocentric distance (kpc)', fontsize=args.fontsize)
    plt.ylabel(r'$\log{(Z/Z_{\bigodot})}$', fontsize=args.fontsize)
    plt.legend(loc='lower left', fontsize=args.fontsize)
    ax = plt.gca()
    ax.tick_params(axis='both', labelsize=args.fontsize)
    cb = plt.colorbar()
    cb.set_label('log(U) (cms^-1)', fontsize=args.fontsize)
    plt.xlim(-1, args.galrad) # kpc
    if args.saveplot: saveplot(fig, args, plot_suffix)
    plt.show(block=False)
    return ax

# --------------------------------------------------------------------
def makeplot_Zin_Zout(paramlist, Zoutcol, args, plot_suffix=''):
    '''
    Function to plot input vs output metallicities
    '''

    fig = plt.figure()

    plt.scatter(np.log10(paramlist['Zin']), np.log10(paramlist[Zoutcol]), c=paramlist['nII'].values, lw=0, alpha=0.5)
    plt.plot(np.log10(paramlist['Zin']), np.log10(paramlist['Zin']), c='k')  # 1:1 line

    plt.xlabel('Zin', fontsize=args.fontsize)
    plt.ylabel('Zout', fontsize=args.fontsize)
    ax = plt.gca()
    ax.tick_params(axis='both', labelsize=args.fontsize)
    cb = plt.colorbar()
    cb.set_label('density log(cm^-3)', fontsize=args.fontsize)
    if args.saveplot: saveplot(fig, args, plot_suffix)
    plt.show(block=False)
    return fig

# ---------------------------------------------------------
def lookup_grid(paramlist, args):
    '''
    Function to deal with input/output dataframe to do the whole computation
    '''

    start_time = time.time()

    # -------------------reading in external models-----------------------
    photgrid = read_photoionisation_grid(mappings_lab_dir + mappings_grid_file) # reading HII model grid file
    linelist = mmg.read_linelist(mappings_lab_dir + 'targetlines.txt') # reading list of emission lines to be extracted from the models

    # -------------------calculating two new quantities for HII regions-----------------------
    paramlist['radial_dist'] = np.sqrt((paramlist['pos_x'] - args.halo_center[0]) ** 2 + (paramlist['pos_y'] - args.halo_center[1]) ** 2)  # kpc
    print('Deb226: min, max hii region distance in kpc', np.min(paramlist['radial_dist']), np.max(paramlist['radial_dist']), '\n') #
    paramlist['logQ'] = np.log10(paramlist['Q_H0'])
    paramlist.rename(columns={'gas_metal':'Zin'}, inplace=True)

    # ----------------to plot args.xratio vs args.yratio to check if lookup-interpolation is done right--------------
    choice_arr = [1, 3, 2, 4]  # choose the sequence of variables to interpolate in 4D
    quantity_dict = {1: mmg.Z_arr, 2: mmg.age_arr, 3: mmg.lognII_arr, 4: mmg.logU_arr}
    quantitya_dict = {1: 'Zin', 2: 'age', 3: 'lognII', 4: 'logU'}  # has to correspond to the same sequence as quantity_dict etc.
    quant_dict = {1: 'Z', 2: 'age', 3: 'lognII', 4: 'logU'}  # has to correspond to the same sequence as quantitya_dict etc.
    quant1, quant2, quant3, quant4 = [quant_dict[item] for item in choice_arr]

   # -----------------reading grid files onto RGI-------------------------------------------------------------------
    quantity1, quantity2, quantity3, quantity4 = [quantity_dict[item] for item in choice_arr]
    quantity1a, quantity2a, quantity3a, quantity4a = [quantitya_dict[item] for item in choice_arr] # has to correspond to the same sequence as quantity1 etc.
    print('Interpolating 4D in the sequence:', quantity1a, quantity2a, quantity3a, quantity4a)

    ifunc = []
    for label in linelist['label']:
        try:
            if args.use_RGI:
                l = np.reshape(np.log10(photgrid[label]), (len(quantity1), len(quantity2), len(quantity3), len(quantity4)))
                iff = RGI((quantity1, quantity2, quantity3, quantity4), l)
            else:
                iff = LND(np.array(photgrid[[quant1, quant2, quant3, quant4]]), np.log10(photgrid[label]))
            ifunc.append(iff)
        except KeyError:
            linelist = linelist[~(linelist['label'] == label)].reset_index(drop=True) # discarding label from linelist if it is not present in photgrid[fluxes]
            pass

    # ----------looping over diag_arr and Om_ar to lookup fluxes-------------------------------------
    for diag in args.diag_arr:
        if diag == 'R23': logOHsun = 8.93  # KD02
        else: logOHsun = 8.77  # Dopita 16

        # --------------creating output directories---------------------------------
        path = args.output_dir + 'txtfiles/' + args.output + '_emission_list' + '_' + diag + mmg.outtag
        subprocess.call(['mkdir -p ' + path], shell=True)

        # ----------looping over Om_arr to lookup fluxes-------------------------------------
        for Om in args.Om_arr:
            fout = path + '/emission_list_Om' + str(Om) + args.mergeHII_text + '.txt'
            if not os.path.exists(fout) or args.clobber:
                paramlist['r'] *= 3.06e16  # convert to m from pc; because the following computations are in SI units
                paramlist['nII'] = calc_n2(Om, paramlist['r'], paramlist['Q_H0'])
                paramlist['lognII'] = np.log10(paramlist['nII'])
                paramlist['<U>'] = calc_U(Om, paramlist['nII'], paramlist['Q_H0'])
                paramlist['logU'] = np.log10(paramlist['<U>'])
                paramlist['r_Strom'] = paramlist['r'] / ((1 + Om) ** (1 / 3.)) # in pc
                paramlist['r_i'] = paramlist['r_Strom']* (Om ** (1 / 3.)) # in pc
                paramlist['log(P/k)'] = np.log10(paramlist['nII']) + mmg.ltemp - 6. # log(P/k) is in CGS units now
                paramlist['r'] /= 3.06e16  # convert to pc from m
                paramlist['r_Strom'] /= 3.06e16  # convert to pc from m
                coord = np.vstack([paramlist[quantity1a], paramlist[quantity2a], paramlist[quantity3a], paramlist[quantity4a]]).transpose()

                # ---------compute fluxes by interpolating and scaling to star particle mass (MAPPINGS model mass for each star = 1000 Msun)---
                for ind in range(len(linelist)): paramlist[linelist.loc[ind, 'label']] = (10**ifunc[ind](coord)) * paramlist['mass']/mappings_starparticle_mass

                # ---to discard outliers that are beyond the parameters used to calibrate the diagnostic---
                if args.nooutliers and diag == 'D16':
                    n = len(paramlist)
                    paramlist = paramlist[paramlist['nII'].between(10**(5.2-4+6), 10**(6.7-4+6))].reset_index(drop=True) #D16 models have 5.2 < lpok < 6.7
                    print('Discarding outliers: For Om=', Om, ':', len(paramlist), 'out of', n, 'H2Rs ae retained as per D16 criteria')

                # --------to calculate Zout based on different diagnostics---------------------------------------
                paramlist['Zout_D16'] = 10 ** (np.log10(np.divide(paramlist['NII6584'], (paramlist['SII6717'] + paramlist['SII6730']))) + 0.264 * np.log10(np.divide(paramlist['NII6584'], paramlist['H6562'])))  # D16 method
                paramlist['Zout_KD02'] = 1.54020 + 1.26602 * np.log10(np.divide(paramlist['NII6584'], (paramlist['OII3727'] + paramlist['OII3729']))) + 0.167977 * np.log10(np.divide(paramlist['NII6584'],(paramlist['OII3727'] + paramlist['OII3729']))) ** 2  # KD02 method

                # ------------------writing dataframe to file--------------------------------------------------------------
                header = 'Units for the following columns: \n\
                pos_x, pos_y, pos_z: kpc \n\
                vel_x, vel_y, vel_z: km/s \n\
                age: Myr \n\
                mass: Msun \n\
                gas_pressure in a cell: g/cm^3 \n\
                Q_H0: ionisation photon flux from star particle, photons/s \n\
                logQ: log(photons/s) \n\
                r_stall: stalled HII region radius, pc \n\
                r_inst: instantaneous HII region radius, pc \n\
                r: assigned HII region radius, pc \n\
                <U>: volumne averaged ionsation parameter, dimensionless \n\
                nII: HII region number density per m^3\n\
                log(P/k): HII region pressure, SI units\n\
                Z: metallicity of ambient gas, Zsun units\n\
                fluxes: ergs/s/cm^2\n'

                np.savetxt(fout, [], header=header, comments='#')
                paramlist.to_csv(fout, sep='\t', mode='a', index=None)
                print('Parameter list saved at', fout)
            else:
                print('Reading existing file from', fout)
                paramlist = pd.read_table(fout, delim_whitespace=True, comment='#')

            # ---------------choosing which metallicity indicator to work with for the rest of the diagnostic plots-------------------
            Zoutcol = 'Zout_' + diag
            paramlist = paramlist[np.isfinite(paramlist[Zoutcol])]

            # ---------various diagnostic/debugging plots---------------------------
            if args.plot_fluxgrid:
                ax_fluxgrid = makeplot_fluxgrid(paramlist, photgrid, args, quant1=quant1, quant2=quant2, quant3=quant3, quant4=quant4, plot_suffix='_' + diag + ',Om=' + str(Om) + '_grid')
            if args.plot_metgrad:
                ax_metgrad = makeplot_metgrad(paramlist, Zoutcol, args, plot_suffix='_' + diag + ',Om=' + str(Om) + '_metallicity_gradient')
            if args.plot_Zin_Zout:
                fig_Zinout = makeplot_Zin_Zout(paramlist, Zoutcol, args, plot_suffix='_' + diag + ',Om=' + str(Om) + '_Zin_vs_Zout')
            if args.plot_phase_space:
                fig_phase_space = makeplot_phase_space(paramlist, args, plot_suffix='_' + diag + ',Om=' + str(Om) + '_P_vs_r_colorby_logQ')

    print('Done in %s minutes' % ((time.time() - start_time) / 60))
    return paramlist

# ---------to use the MAPPINGS model grid generated by make_mappings_grid.py---------
mappings_grid_file = 'totalspec' + mmg.outtag + '.txt'  # name of MAPPINGS grid file to be used
mappings_starparticle_mass = mmg.mappings_starparticle_mass
'''
# ---------to use a custom photoionisation grid (must be in the same format as the MAPPINGS grid)-------------------
mappings_grid_file = 'totalspec_sph_logT4.0_MADtemp_ion_lum_from_age_Z0.05,2.0_age0.0,5.0_lnII6.0,11.0_lU-4.0,-1.0_4D.txt'
mappings_starparticle_mass = 300.
'''

# -----------------------------------------------------------------------------
if __name__ == '__main__':
    args = parse_args('8508', 'RD0042')
    if not args.keep: plt.close('all')

    infilename = args.output_dir + 'txtfiles/' + args.output + '_radius_list' + args.mergeHII_text + '.txt'
    paramlist = pd.read_table(infilename, delim_whitespace=True, comment='#') # reading in the list of HII region parameters
    paramlist = lookup_grid(paramlist, args)
