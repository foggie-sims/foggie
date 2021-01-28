# -----python code to compute fluxes of HII regions for given metallicity gradient--------
# ----------by Ayan, last modified Feb 2019--------------------
# -----  example usage:
# ------ ipython> run lookup.py --file DD0600_lgf --Om 0.5 --diag D16 --logOHgrad -0.1 --plot_metgrad --write_file --saveplot --mergeHII 0.04


##!/usr/bin/env python3

""""

    Title :      lookup_flux
    Notes :      To compute (and optionally plot) fluxes of HII regions for various emission lines (and output to an ASCII file) using MAPPINGS photoionisation model grid
    Author:      Ayan Acharyya
    Started  :   January 2021
    Example :    run <scriptname>.py --system ayan_local --halo 8508 --output RD0042

"""
from header import *
start_time = time.time()

# -------------function to find element in array nearest to a given value---------------------
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

# -------------function to derive Omega for each HII region, given its galactocentric diatance---------------------
def getOmega(radius, scale_length = 4.):
    # there are 11 uniformly spaced points in log-scale, from Om=0.05 to )m = 5.0, both inclusive,
    # to account for radially varying Omega from 5 at r=0 to 0.05 at r > 2R_e
    # log Omega = max[log(5) - r/R_e, log(0.05)]
    # Hence, Omega = 5 at r = 0, Omega = 0.5 at r = R_e, and Omega = 0.05 at r >= 2 R_e
    Omega = 10**(np.max([np.log10(5) - radius/scale_length, np.log10(0.05)])) # radius and scale_length in kpc
    return Omega

# ----------function to plot the P-r phase space---------------------
def plot_phase_space(paramlist, outplotname, args):
    if args.fontsize is not None: fs = int(args.fontsize)
    else: fs = 15 # ticklabel fontsize

    paramlist['P'] = 10 ** paramlist['log(P/k)'] * 1.38e-16  # converting P/k in CGS units to pressure in dyne/cm^2
    paramlist['logr'], paramlist['logP'] = np.log10(paramlist['r']), np.log10(paramlist['P'])

    # --------for hexplot---------------
    if args.plothex:
        axes = sns.JointGrid('logr', 'logP', paramlist, height=8)
        sns.kdeplot(paramlist['logr'].values, ax=axes.ax_marg_x, legend=False, color='r', linestyle='solid')
        sns.kdeplot(paramlist['logP'].values, ax=axes.ax_marg_y, vertical=True, legend=False, color='r', linestyle='solid')
        axes.ax_joint.hexbin(paramlist['logr'].values, paramlist['logP'].values, alpha=0.7, cmap='Reds', gridsize=(40, 40))
        axes.set_axis_labels('HII region size [pc]', 'HII region pressure [dyne/cm^2]', fontsize=fs)
        fig = plt.gcf()
        if args.saveplot:
            fig.savefig(os.path.splitext(outplotname)[0]+'_hexplot' + os.path.splitext(outplotname)[1])
            print 'Saved plot at', os.path.splitext(outplotname)[0]+'_hexplot' + os.path.splitext(outplotname)[1]
        plt.show(block=False)

    # --------for scatter plot-----------
    marker_model, color_model = 'o', 'gray' # paramlist['logQ'].values #
    fig = plt.figure()
    plt.scatter(paramlist['r'], paramlist['P'], c=color_model, lw=0, alpha=0.3, marker=marker_model)
    plt.xlabel('HII region size [pc]', fontsize=fs)
    plt.ylabel('HII region pressure [dyne/cm^2]', fontsize=fs)
    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(1e-12, 1e-7)
    plt.hlines([1e-11, 1e-9], 0.6, 25., color='k', linestyles='dashed') # plotting range of Fig 2 of  Tremblin+2014
    plt.vlines([0.6, 25.], 1e-11, 1e-9, color='k', linestyles='dashed')
    ax.text(1.8, 10**-8.8, 'Plot window of Tremblin+14, Fig 2', color='k', fontsize=fs)
    if isinstance(color_model, (list, tuple, np.ndarray)):
        cb = plt.colorbar()
        cb.set_label('Luminosity [log(s^-1)]', fontsize=fs)  # 'dist(kpc)')#)'H6562 luminosity log(ergs/s)'#

    # ------------for plotting Tremblin+14 Fig 2 data--------
    if args.plot_obs:
        marker_obs, color_obs = '*', 'green'
        tremblin = pd.read_csv(HOME + '/Dropbox/papers/enzo_paper/AA_Working/Tremblin2014_Fig2_Dataset.csv', comment='#', names=['r', 'P'], header=None)
        ax.scatter(tremblin['r'], tremblin['P'], c=color_obs, marker=marker_obs, lw=0, alpha=0.7, s=200)

    if args.saveplot:
        fig.savefig(outplotname)
        print 'Saved plot at', outplotname
    plt.show(block=False)

# ------------function to plot met grad------------------------
def plot_metgrad(paramlist, Zoutcol, logOHgrad, logOHcen, logOHsun, outplotname, args):
    if args.fontsize is not None: fs = int(args.fontsize)
    else: fs = 15 # ticklabel fontsize

    fig = plt.figure()
    plt.plot(np.arange(args.galsize / 2), np.poly1d((logOHgrad, logOHcen))(np.arange(args.galsize / 2)) - logOHsun,
             c='brown',
             ls='dotted', label='Target gradient=' + str('%.3F' % logOHgrad))  #

    plt.scatter(paramlist['distance'], np.log10(paramlist['Zin']), c='k', lw=0, alpha=0.5)
    #plt.scatter(paramlist['distance'], np.log10(paramlist[Zoutcol]), c=np.log10(paramlist['nII']) - 6, lw=0, alpha=0.2)
    plt.scatter(paramlist['distance'], np.log10(paramlist[Zoutcol]), c=paramlist['logU'] + np.log10(3e10), lw=0, alpha=0.2)
    linefit = np.polyfit(paramlist['distance'], np.log10(paramlist[Zoutcol]), 1, cov=False)
    x_arr = np.arange(args.galsize / 2)
    plt.plot(x_arr, np.poly1d(linefit)(x_arr), c='b',
             label='Fitted gradient=' + str('%.4F' % linefit[0]) + ', offset=' + \
                   str('%.1F' % ((linefit[0] - logOHgrad) * 100 / logOHgrad)) + '%')

    plt.xlabel('Galactocentric distance (kpc)', fontsize=fs)
    plt.ylabel(r'$\log{(Z/Z_{\bigodot})}$', fontsize=fs)
    plt.legend(loc='lower left', fontsize=fs)
    ax = plt.gca()
    ax.tick_params(axis='both', labelsize=fs)
    cb = plt.colorbar()
    #cb.set_label('Density log(cm^-3)', fontsize=fs)  # 'H6562 luminosity log(ergs/s)')#
    cb.set_label('log(U) (cms^-1)', fontsize=fs)  # 'H6562 luminosity log(ergs/s)')#
    # if not args.diag == 'R23': plt.ylim(ylim_dict[logOHgrad],0.2)
    plt.xlim(-1, 12)  # kpc
    if logOHgrad > 0: plt.ylim(logOHcen - logOHsun - 1., logOHgrad * 12.)
    if args.saveplot:
        fig.savefig(outplotname)
        print 'Saved plot at', outplotname
    plt.show(block=False)

# -----------Function to check if float------------------------------
def isfloat(str):
    try:
        float(str)
    except ValueError:
        return False
    return True

# -----------------------------------------------
def num(s):
    if s[-1].isdigit():
        return str(format(float(s), '0.2e'))
    else:
        return str(format(float(s[:-1]), '0.2e'))

# --------------------------------------------------------
def calc_n2(Om, r, Q_H0):
    return (np.sqrt(3 * Q_H0 * (1 + Om) / (4 * np.pi * alpha_B * r ** 3)))

# -----------------------------------------------
def calc_U(Om, nII, Q_H0):
    return r.func(nII, np.log10(Q_H0)) * ((1 + Om) ** (4 / 3.) - (4. / 3. + Om) * (Om ** (1 / 3.)))

# ---------function to paint metallicity gradient-----------------------------------------------
def calc_Z(r, logOHcen, logOHgrad, logOHsun):
    Z = 10 ** (logOHcen - logOHsun + logOHgrad * r)  # assuming central met logOHcen and gradient logOHgrad dex/kpc, logOHsun = 8.77 (Dopita 16)
    return Z

# -------------function to delete a certain H2R from all lists--------------------------
def remove_star(indices, list_of_var):
    new_list_of_var = []
    for list in list_of_var:
        list = np.delete(list, indices, 0)
        new_list_of_var.append(list)
    return new_list_of_var

# -------------function to use KD02 R23 diagnostic for the upper Z branch--------------------------
def poly(x, R, k):
    return np.abs(np.poly1d(k)(x) - np.log10(R))

# --------------function to deal with input/output dataframe to do the whole computation-----------
def lookup_full_df(paramlist, args):
    start_time = time.time()
    # -----------------------assigning default values to args-----------------------
    if args.file is not None: fn = args.file
    else: fn = 'DD0600_lgf'

    if args.diag is not None: diag_arr = [ar for ar in args.diag.split(',')]
    else: diag_arr = ['D16'] #['KD02', 'D16']

    if args.Om is not None:
        if type(args.Om) is str: Om_arr = [float(ar) for ar in args.Om.split(',')]
        elif type(args.Om) is float: Om_arr = [args.Om]
        else: Om_arr = args.Om # already in a list
    else: Om_arr = [0.05, 0.5, 5.0] #np.round(10**np.linspace(np.log10(0.05), np.log10(5), 10+1), 2) #[0.05, 0.5, 5.0] # expanded Omega array to 11 uniformly spaced points in log-scale, to account for radially varying Omega from 5 at r=0 to 0.05 at r > 2R_e

    if args.logOHgrad is not None:
        if type(args.logOHgrad) is str:
            logOHgrad_arr = [float(ar) for ar in args.logOHgrad.split(',')]
        elif type(args.logOHgrad) is float:
            logOHgrad_arr = [args.logOHgrad]
        else:
            logOHgrad_arr = args.logOHgrad  # already in a list
    else: logOHgrad_arr = [-0.2, -0.15, 0.15, 0.2] #[0.01, 0.025, 0.05, 0.07, 0.1, 0.15, 0.2, -0.2, -0.15, -0.1, -0.07, -0.05, -0.025, -0.01]

    if args.fontsize is not None: fs = int(args.fontsize)
    else: fs = 15 # ticklabel fontsize

    if args.outpath is not None: outpath = args.outpath
    else: outpath = HOME+'/Desktop/bpt_contsub_contu_rms/'

    if args.mergeHII is not None: args.mergeHII = float(args.mergeHII) # kpc, within which if two HII regions are they'll be treated as merged

    args.center = 0.5*1310.72022072 # kpc
    args.galsize = 30. # kpc


    args.galsize_z = 0.3 # kpc in z direction
    args.center_z = 0.500191325895*1310.72022072 # kpc units, from Goldbaum simulations in cell units

    if not args.keep: plt.close('all')
    allstars_text = '_allstars' if args.allstars else ''

    # --------------------------------calculating two new columns-----------------------
    paramlist['distance'] = np.sqrt((paramlist['x'] - args.center) ** 2 + (paramlist['y'] - args.center) ** 2)  # kpc
    paramlist['logQ'] = np.log10(paramlist['Q_H0'])
    print'min, max h2r distance in kpc', np.min(paramlist['distance']), np.max(paramlist['distance']), '\n'  #
    mergeHII_text = '_mergeHII='+str(args.mergeHII)+'kpc' if args.mergeHII is not None else ''
    wo = '_no_outlier' if args.nooutliers else ''

    # ---------to plot the original simulation luminosity map------------------
    if args.plot_lummap:
        # ---figure for age map-----
        fig = plt.figure()
        plt.scatter(x, y, c=paramlist['Q_H0'], s=5, lw=0)
        cb = plt.colorbar()
        cb.set_label('Q_H0', fontsize=fs)
        plt.ylabel('y (kpc)', fontsize=fs)
        plt.xlabel('x (kpc)', fontsize=fs)
        plt.xlim(args.center - args.galsize/2, args.center + args.galsize/2)
        plt.ylim(args.center - args.galsize/2, args.center + args.galsize/2)
        ax = plt.gca()
        ax.set_xticklabels(['%.2F' % (i - args.center) for i in list(ax.get_xticks())], fontsize=fs)
        ax.set_yticklabels(['%.2F' % (i - args.center) for i in list(ax.get_yticks())], fontsize=fs)
        ax.tick_params(axis='both', labelsize=fs)
        xdiff = np.diff(ax.get_xlim())[0]
        ydiff = np.diff(ax.get_ylim())[0]
        ax.text(ax.get_xlim()[-1] - 0.1 * xdiff, ax.get_ylim()[-1] - 0.1 * ydiff, fn, color='k', ha='right', va='center',fontsize=fs)
        if args.saveplot:
            outplotname = outpath + fn + allstars_text+'_QH0_map.eps'
            fig.savefig(outplotname)
            print 'Saved figure at', outplotname

        if args.allstars:
            # -------figure for age CDF------
            factor = 1e52 # just a factor for cleaner plot
            age_turn = 5 # Myr
            Q_H0 = np.array([x for _, x in sorted(zip(paramlist['age'], paramlist['Q_H0']), key=lambda pair: pair[0])]) / factor
            age = np.sort(paramlist['age'])
            Q_sum = np.cumsum(Q_H0)
            Q_tot = Q_sum[-1]
            Q_turn = Q_sum[np.where(age >= age_turn)[0][0]]
            fig = plt.figure()
            plt.plot(age, Q_sum, c='k')
            plt.axhline(Q_tot, c='k', ls='dotted', label='Total ionising luminosity')
            plt.axhline(Q_turn, c='k', ls='dashed', label='%.1F%% of the total at %d Myr'%(Q_turn*100./Q_tot, age_turn))
            plt.axvline(age_turn, c='k', ls='dashed')
            plt.xlabel(r'Age $<$ t Myr', fontsize=fs)
            plt.ylabel(r'Cumulative $Q_{H0}$ (10$^{\mathrm{%d}}$ ergs/s)'%np.log10(factor), fontsize=fs)
            plt.xlim(0,10)
            plt.ylim(0,7)
            plt.legend(loc='lower right', fontsize=fs)
            ax = plt.gca()
            ax.set_xticklabels(ax.get_xticks(), fontsize=fs)
            ax.set_yticklabels(ax.get_yticks(), fontsize=fs)
            if args.saveplot:
                outplotname = outpath + fn + allstars_text+'_QH0_cumulative.eps'
                fig.savefig(outplotname)
                print 'Saved plot at', outplotname

        plt.show(block=False)
    if args.allstars: sys.exit()

    # ------------Reading pre-defined line list-----------------------------
    linelist = pd.read_table(HOME+'/Mappings/lab/targetlines.txt', comment='#', delim_whitespace=True, skiprows=3, names=('wave', 'label', 'wave_vacuum'))
    linelist = linelist.sort_values(by=('wave')).reset_index(drop=True)

    # -----------------reading HII grid files-------------------------------------------------------------------
    '''
    r.outtag = '_sph_logT4.0_MADtemp_Z0.05,5.0_age0.0,5.0_lnII5.0,12.0_lU-4.0,-1.0_4D' # for testing only
    r.lognII_arr = np.linspace(5., 12., 6) # for testing only
    r.Z_arr = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0]) # for testing only
    '''
    s = pd.read_table(HOME+'/Mappings/lab/totalspec' + r.outtag + '.txt', comment='#', delim_whitespace=True)
    if s['age'].max() > 100.: s['age'] /= 1e6 # convert age to units of Myr if already not so; if it is already in units of Myr then it is unlikely to have ages > 100 Myr
    s['lognII'] = np.log10(s['nII'])
    s['logU'] = np.log10(s['<U>'])

    # ----------------to plot Halpha/Q0 vs logU to check SFR calibration---------------
    if args.plot_HIIgrid:
        Q_input = 1e49  # s^-1 ; ionising photon rate used as input to MAPPINGS models
        Q_to_Ha = 1.37e-12  # units of ergs; factor to convert ionising photon rate (s^-1) to Ha luminosity (ergs/s)
        all_nII = np.log10(pd.unique(s['nII']))
        nII_col_dict = {all_nII[0]:'b', all_nII[1]:'cyan', all_nII[2]:'g', all_nII[3]:'y', all_nII[4]:'orange', all_nII[5]:'r'}
        age_col_dict = {0.:'b', 1.:'cyan', 2.:'g', 3:'y', 4:'orange', 5:'r'}

        for Z in pd.unique(s['Z']):
            print 'Plotting for Z=', Z
            subset = s[s['Z'] == Z]
            fig = plt.figure()
            ax = plt.gca()
            groups = subset.groupby(by=('age', 'nII')).groups
            for key, values in groups.iteritems():
                subsubset = subset.ix[values][['Z', 'age', 'nII', '<U>', 'logQ0', 'lqin', 'H6562']]
                thisage = key[0]/1e6
                thisnII = np.log10(key[1])
                plt.plot(subsubset['lqin'].values, subsubset['H6562'].values/(Q_input*Q_to_Ha), alpha=0.5, c=age_col_dict[thisage])
            ax.text(0.9, 0.9, 'Z/Zsol = '+str(Z), ha='right', va='top', color='k', fontsize=20, transform=ax.transAxes)
            plt.xlabel('log(U)')
            plt.ylabel('Halpha (ergs s^-1) / (Input Q0 (s^-1) x 1.37 x 10^-12)')
            # plt.ylim(0, 0.12) # for previous model grid, with input bolometric luminosity
            fig.savefig('/Users/acharyya/Desktop/bpt_contsub_contu_rms/Ha_by_Q_vs_logU_Z'+str(Z)+'.eps')
        plt.show(block=False)
    # ----------------to plot xratio vs yratio to check if lookup-interpolation is done right--------------
    choice_arr = [1, 3, 2, 4]  # choose the sequence of variables to interpolate in 4D
    quantity_dict = {1: r.Z_arr, 2: r.age_arr, 3: r.lognII_arr, 4: r.logU_arr}
    quantitya_dict = {1: 'Zin', 2: 'age', 3: 'lognII', 4: 'logU'}  # has to correspond to the same sequence as quantity_dict etc.
    quant_dict = {1: 'Z', 2: 'age', 3: 'lognII', 4: 'logU'}  # has to correspond to the same sequence as quantitya_dict etc.
    quant1, quant2, quant3, quant4 = [quant_dict[item] for item in choice_arr]

    if args.plot_fluxgrid:
        xratio = 'NII6584/H6562' if args.xratio is None else args.xratio
        #xratio = 'OIII5007/OII3727,OII3729' if args.xratio is None else args.xratio
        yratio = 'NII6584/SII6717,SII6730' if args.yratio is None else args.yratio
        #yratio = 'OIII5007/HBeta' if args.yratio is None else args.yratio

        if '/' in xratio: s[xratio] = np.array([s[item] for item in xratio.split('/')[0].split(',')]).sum(axis=0) / np.array([s[item] for item in xratio.split('/')[1].split(',')]).sum(axis=0)
        if '/' in yratio: s[yratio] = np.array([s[item] for item in yratio.split('/')[0].split(',')]).sum(axis=0) / np.array([s[item] for item in yratio.split('/')[1].split(',')]).sum(axis=0)

        fig = plt.figure(figsize=(8, 8))
        ax_fluxgrid = plt.gca()

        arr1 = np.unique(s[quant1])
        arr2 = np.unique(s[quant2])
        arr3 = np.unique(s[quant3])
        arr4 = pd.unique(s[quant4])
        ls_dict = {0:'-.', 1:'--', 2:':', 3:'-', 4:'-', 5:'--'}

        for index,thisquant4 in enumerate(arr4):
            subset = s[s[quant4] == thisquant4]

            x = np.reshape(np.log10(subset[xratio].values), (len(arr3), len(arr2), len(arr1)))
            y = np.reshape(np.log10(subset[yratio].values), (len(arr3), len(arr2), len(arr1)))
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

        plt.xlabel('log('+xratio+')')
        plt.ylabel('log('+yratio+')')
        if '/' in xratio: plt.xlim(-3.5, 0.5)
        if '/' in yratio: plt.ylim(-1, 2)
        plt.show(block=False)
    # -----------------reading grid files onto RGI-------------------------------------------------------------------
    quantity1, quantity2, quantity3, quantity4 = [quantity_dict[item] for item in choice_arr]
    quantity1a, quantity2a, quantity3a, quantity4a = [quantitya_dict[item] for item in choice_arr] # has to correspond to the same sequence as quantity1 etc.
    #print 'Interpolating 4D in the sequence:', quantity1a, quantity2a, quantity3a, quantity4a

    ifunc = []
    for label in linelist['label']:
        try:
            if args.use_RGI:
                l = np.reshape(np.log10(s[label]), (len(quantity1), len(quantity2), len(quantity3), len(quantity4)))
                iff = RGI((quantity1, quantity2, quantity3, quantity4), l)
            else:
                iff = LND(np.array(s[[quant1, quant2, quant3, quant4]]), np.log10(s[label]))
            ifunc.append(iff)
        except KeyError:
            linelist = linelist[~(linelist['label'] == label)].reset_index(drop=True) # discarding label from linelist if it is not present in s[fluxes]
            pass

    # ----------looping over diag_arr, logOHgrad_arr and Om_ar to lookup fluxes-------------------------------------
    for diag in diag_arr:
        if diag == 'R23': logOHsun = 8.93  # KD02
        else: logOHsun = 8.77  # Dopita 16

        # ----------looping over logOHgrad_arr to lookup fluxes-------------------------------------
        for logOHgrad in logOHgrad_arr:
            if logOHgrad > 0.1:
                logOHcen = np.round(logOHsun - logOHgrad*5, 2) # pivoting metallicity value = solar at 5 kpc, for positive gradients
                print 'Tying metallicity value to solar at 5 kpc, for positive gradient of ' + str(logOHgrad) + ' such that logOHcen = ' + str(logOHcen)
            elif logOHgrad > 0:
                logOHcen = np.round(logOHsun - logOHgrad*10, 2) # pivoting metallicity value = solar at 10 kpc, for positive gradients
                print 'Tying metallicity value to solar at 10 kpc, for positive gradient of ' + str(logOHgrad) + ' such that logOHcen = ' + str(logOHcen)
            elif logOHgrad < -0.1:
                logOHcen = 9.0
                print 'Tying central metallicity value to ' + str(logOHcen) + ', for too negative gradient of ' + str(logOHgrad)
            else:
                if diag == 'R23': logOHcen = 9.5  # to use upper branch of R23 diag of KD02
                else: logOHcen = logOHsun  # 9.2 #(Ho 15)

            # --------------create output directories---------------------------------
            path = HOME+'/models/emissionlist' + '_Zgrad' + str(logOHcen) + ',' + str(logOHgrad) + r.outtag
            subprocess.call(['mkdir -p ' + path], shell=True)

            # -------to plot histogram of derived Z------------------------------
            if args.plot_hist:
                print 'outtag =', r.outtag
                if diag == 'D16':
                    log_ratio = np.log10(np.divide(s['NII6584'],(s['SII6730']+s['SII6717']))) + 0.264*np.log10(np.divide(s['NII6584'],s['H6562']))
                    logOHobj_map = log_ratio + 0.45*(log_ratio + 0.3)**5 # + 8.77
                elif diag == 'KD02':
                    log_ratio = np.log10(np.divide(s['NII6584'], (s['OII3727'] + s['OII3729'])))
                    logOHobj_map = 1.54020 + 1.26602 *log_ratio  + 0.167977 *log_ratio ** 2
                print 'log_ratio med, min', np.median(log_ratio), np.min(log_ratio)  #
                print 'logOHobj_map before conversion med, min', np.median(logOHobj_map), np.min(logOHobj_map) #

                Z_list = 10**(logOHobj_map) #converting to Z (in units of Z_sol) from log(O/H) + 12
                print 'Z_list after conversion med, mean, min',np.median(Z_list), np.mean(Z_list), np.min(Z_list) #

                plt.figure()
                plt.hist(Z_list, 100, range =(-1,6)) #
                plt.title('Z_map for grids') #

                plt.xlabel('Z/Z_sol') #
                if args.saveplot:
                    outplotname = outpath + 'Z_map for grids.eps'
                    fig.savefig(outplotname)
                    print 'Saved plot at', outplotname
                plt.show(block=False) #

            # ----------looping over Om_arr to lookup fluxes-------------------------------------
            if args.mergeOmega:
                paramlist_Om_merged = pd.DataFrame()
                fout_Om_merged = path + '/emissionlist_' + fn + '_Om-99' + mergeHII_text + '.txt'  # Om = -99 implies radially varying Omega

            for Om in Om_arr:
                if Om < 0: Om = int(Om)  # Om = -99 implies radially variable Omega according to lookup.py/getOmega(radius)
                fout = path + '/emissionlist_' + fn + '_Om' + str(Om) + mergeHII_text + '.txt'
                if not os.path.exists(fout) or args.clobber:
                    paramlist['r'] *= 3.06e16 # convert to m from pc
                    '''
                    if args.logOHgrad is not None: paramlist['Zin'] = calc_Z(paramlist['distance'], logOHcen, logOHgrad, logOHsun)
                    else: paramlist = paramlist.rename(columns={'Z_gas':'Zin'}) # not painting in metallicity gradient but using existing column
                    '''
                    paramlist['Zin'] = calc_Z(paramlist['distance'], logOHcen, logOHgrad, logOHsun)
                    paramlist['nII'] = calc_n2(Om, paramlist['r'], paramlist['Q_H0'])
                    paramlist['lognII'] = np.log10(paramlist['nII'])
                    paramlist['<U>'] = calc_U(Om, paramlist['nII'], paramlist['Q_H0'])
                    paramlist['logU'] = np.log10(paramlist['<U>'])
                    paramlist['r_Strom'] = paramlist['r'] / ((1 + Om) ** (1 / 3.))
                    paramlist['r_i'] = paramlist['r_Strom']* (Om ** (1 / 3.))
                    paramlist['log(P/k)'] = np.log10(paramlist['nII']) + ltemp - 6. # log(P/k) is in CGS units now
                    paramlist['r'] /= 3.06e16 # convert to pc from m
                    paramlist['r_Strom'] /= 3.06e16 # convert to pc from m
                    coord = np.vstack([paramlist[quantity1a], paramlist[quantity2a], paramlist[quantity3a], paramlist[quantity4a]]).transpose()
                    # ---------compute fluxes by interpolating and scaling to star particle mass (MAPPINGS model mass for each star = 300 Msun)---
                    for ind in range(len(linelist)): paramlist[linelist.loc[ind, 'label']] = (10**ifunc[ind](coord)) * paramlist['mass']/300.
                    '''
                    #pdb.set_trace() #
                    flag_arr1, flag_arr2 = [], []
                    np.set_printoptions(formatter={'float_kind': lambda x: "%.2f" % x})
                    for hii in range(len(coord)):
                        np.set_printoptions(formatter={'float_kind': lambda x: "%.2f" % x})
                        flag1 = np.array([(quantity1[0] <= coord[hii][0] <= quantity1[-1]),
                                      (quantity2[0] <= coord[hii][1] <= quantity2[-1]),
                                      (quantity3[0] <= coord[hii][2] <= quantity3[-1]),
                                      (quantity4[0] <= coord[hii][3] <= quantity4[-1])]).all()
                        flag_arr1.append(flag1)

                        nearest_grid = np.array((quantity1[(np.abs(quantity1 - coord[hii][0])).argmin()],
                                                 quantity2[(np.abs(quantity2 - coord[hii][1])).argmin()],
                                                 quantity3[(np.abs(quantity3 - coord[hii][2])).argmin()],
                                                 quantity4[(np.abs(quantity4 - coord[hii][3])).argmin()]))

                        grid_row = s[(s[quant1] == nearest_grid[0]) & (s[quant2] == nearest_grid[1]) & (s[quant3] == nearest_grid[2]) & (s[quant4] == nearest_grid[3])]
                        if '/' in xratio:
                            interp_NII = ifunc[linelist[linelist['label'] == 'NII6584'].index[0]](coord[hii])[0]
                            interp_Ha = ifunc[linelist[linelist['label'] == 'H6562'].index[0]](coord[hii])[0]
                            interp_SIIa = ifunc[linelist[linelist['label'] == 'SII6717'].index[0]](coord[hii])[0]
                            interp_SIIb = ifunc[linelist[linelist['label'] == 'SII6730'].index[0]](coord[hii])[0]
                            interp_log_NII_Ha = np.log10(10**(interp_NII) / 10**(interp_Ha))
                            interp_log_NII_SII = np.log10(10**(interp_NII) / (10**(interp_SIIa) + 10**(interp_SIIb)))

                            flag2 = np.isfinite(interp_NII)
                            flag_arr2.append(flag2)

                            nearest_NII = ifunc[linelist[linelist['label'] == 'NII6584'].index[0]](nearest_grid)[0]
                            nearest_Ha = ifunc[linelist[linelist['label'] == 'H6562'].index[0]](nearest_grid)[0]
                            nearest_SIIa = ifunc[linelist[linelist['label'] == 'SII6717'].index[0]](nearest_grid)[0]
                            nearest_SIIb = ifunc[linelist[linelist['label'] == 'SII6730'].index[0]](nearest_grid)[0]
                            nearest_log_NII_Ha = np.log10(10**(nearest_NII) / 10**(nearest_Ha))
                            nearest_log_NII_SII = np.log10(10**(nearest_NII) / (10**(nearest_SIIa) + 10**(nearest_SIIb)))

                            grid_NII = grid_row['NII6584'].values[0]
                            grid_Ha = grid_row['H6562'].values[0]
                            grid_SIIa = grid_row['SII6717'].values[0]
                            grid_SIIb = grid_row['SII6730'].values[0]
                            grid_log_NII_Ha = np.log10(grid_NII / grid_Ha)
                            grid_log_NII_SII = np.log10(grid_NII / (grid_SIIa + grid_SIIb))
                            print 'Deb340:', hii, coord[hii], np.array([interp_log_NII_Ha, interp_log_NII_SII]), nearest_grid, np.array([nearest_log_NII_Ha, nearest_log_NII_SII]), np.array([grid_log_NII_Ha, grid_log_NII_SII]), flag1, flag2
                        else:
                            interp_log_xflux = ifunc[linelist[linelist['label'] == xratio].index[0]](coord[hii])[0]
                            interp_log_yflux = ifunc[linelist[linelist['label'] == yratio].index[0]](coord[hii])[0]

                            flag2 = np.isfinite(interp_log_xflux)
                            flag_arr2.append(flag2)

                            nearest_log_xflux = ifunc[linelist[linelist['label'] == xratio].index[0]](nearest_grid)[0]
                            nearest_log_yflux = ifunc[linelist[linelist['label'] == yratio].index[0]](nearest_grid)[0]

                            grid_log_xflux = np.log10(grid_row[xratio].values[0])
                            grid_log_yflux = np.log10(grid_row[yratio].values[0])

                            print 'Deb340:', hii, coord[hii], np.array([interp_log_xflux, interp_log_yflux]), nearest_grid, np.array([nearest_log_xflux, nearest_log_yflux]), np.array([grid_log_xflux, grid_log_yflux]), flag1, flag2
                    print 'Deb352:', np.array(flag_arr1).all(), np.array(flag_arr2).all()
                    '''
                    # ---to discard outliers that are beyond the parameters used to calibrate the diagnostic---
                    if args.nooutliers and diag == 'D16':
                        n = len(paramlist)
                        paramlist = paramlist[paramlist['nII'].between(10**(5.2-4+6), 10**(6.7-4+6))].reset_index(drop=True) #D16 models have 5.2 < lpok < 6.7
                        print 'Discarding outliers: For logOHgrad=', logOHgrad, ',Om=', Om, ':', len(paramlist), 'out of', n, 'H2Rs meet D16 criteria'

                    # --------to calculate Zout based on different diagnostics---------------------------------------
                    #pdb.set_trace() #
                    paramlist['Zout_D16'] = 10 ** (np.log10(np.divide(paramlist['NII6584'], (paramlist['SII6717'] + paramlist['SII6730']))) + 0.264 * np.log10(np.divide(paramlist['NII6584'], paramlist['H6562'])))  # D16 method
                    paramlist['Zout_KD02'] = 1.54020 + 1.26602 * np.log10(np.divide(paramlist['NII6584'], (paramlist['OII3727'] + paramlist['OII3729']))) + 0.167977 * np.log10(np.divide(paramlist['NII6584'],(paramlist['OII3727'] + paramlist['OII3729']))) ** 2  # KD02 method
                    '''
                    Zout= []
                    for k in range(len(paramlist)):
                        ratio = (paramlist.loc[k, 'OII3727'] + paramlist.loc[k, 'OII3729'] + paramlist.loc[k, 'OIII4363']) / paramlist.loc[k, 'HBeta']
                        this_Zout = 10 ** (op.fminbound(poly, 7., 9., args=(ratio,[-0.996645, 32.6686, -401.868, 2199.09, -4516.46])) - 8.93)  # k parameters from Table 3 of KD02 for q=8e7
                        Zout.append(this_Zout)
                    paramlist['Zout_R23'] = np.array(Zout)
                    '''
                    # ------------------writing dataframe to file--------------------------------------------------------------
                    if args.write_file:
                        header = 'Units for the following columns: \n\
                        x, y, z: kpc \n\
                        vel_x: km/s \n\
                        vel_y: km/s \n\
                        vel_z: km/s \n\
                        age: Myr \n\
                        mass: Msun \n\
                        gas_P in a cell: N/m^2 \n\
                        Q_H0: photons/s \n\
                        logQ: log(photons/s) \n\
                        r_stall: pc \n\
                        r_inst: pc \n\
                        r: pc \n\
                        <U>: volumne averaged, dimensionless \n\
                        nII: HII region number density per m^3\n\
                        log(P/k): CGS units \n\
                        Z: metallicity in Zsun units \n\
                        fluxes: ergs/s/cm^2'

                        np.savetxt(fout, [], header=header, comments='#')
                        paramlist.to_csv(fout, sep='\t', mode='a', index=None)
                        print 'Parameter list saved at', fout
                else:
                    print 'Reading existing file from', fout
                    paramlist = pd.read_table(fout, delim_whitespace=True, comment='#')
                # ---------to plot line flux grid---------------------------
                if args.plot_fluxgrid:
                    if '/' in xratio: paramlist[xratio] = np.array([paramlist[item] for item in xratio.split('/')[0].split(',')]).sum(axis=0) / np.array([paramlist[item] for item in xratio.split('/')[1].split(',')]).sum(axis=0)
                    if '/' in yratio: paramlist[yratio] = np.array([paramlist[item] for item in yratio.split('/')[0].split(',')]).sum(axis=0) / np.array([paramlist[item] for item in yratio.split('/')[1].split(',')]).sum(axis=0)
                    ax_fluxgrid.scatter(np.log10(paramlist[xratio]), np.log10(paramlist[yratio]), c='k', lw=0, alpha=0.3)
                    plt.show(block=False)

                # ---------------choosing which metallicity indicator to work with, based on command line option-------------------
                Zoutcol = 'Zout_' + diag
                paramlist = paramlist[np.isfinite(paramlist[Zoutcol])]

                # ---to check final distribution of Z after looking up the grid---
                if args.plot_metgrad:
                    outplotname = outpath + fn + '_Zgrad_' + diag + '_col_density_logOHgrad=' + str(logOHgrad) + ',Om=' + str(Om) + wo + mergeHII_text + '.eps'
                    plot_metgrad(paramlist, Zoutcol, logOHgrad, logOHcen, logOHsun, outplotname, args)

                # -----to plot Zin vs Zout--------
                if args.plot_Zinout:
                    fig = plt.figure()
                    plt.scatter(np.log10(paramlist['Zin']), np.log10(paramlist[Zoutcol]), c=paramlist['nII'].values, lw=0, alpha=0.5)
                    plt.plot(np.log10(paramlist['Zin']), np.log10(paramlist['Zin']), c='k') #1:1 line
                    plt.xlabel('Zin', fontsize=fs)
                    plt.ylabel('Zout', fontsize=fs)
                    ax = plt.gca()
                    ax.tick_params(axis='both', labelsize=fs)
                    cb = plt.colorbar()
                    cb.set_label('density log(cm^-3)', fontsize=fs)#'dist(kpc)')#)'H6562 luminosity log(ergs/s)'#
                    xdiff = np.diff(ax.get_xlim())[0]
                    ydiff = np.diff(ax.get_ylim())[0]
                    ax.text(ax.get_xlim()[-1]-0.1*xdiff, ax.get_ylim()[-1]-0.1*ydiff, 'Input gradient = %.3F'%logOHgrad, color='k', ha='right', va='center', fontsize=fs)
                    #plt.title('logOHgrad='+str(logOHgrad)+', Om='+str(Om))
                    if args.saveplot:
                        outplotname = outpath+fn+'_Zout_'+diag+'_vs_Zin_col_nII_logOHgrad='+str(logOHgrad)+',Om='+str(Om)+wo+'.eps'
                        fig.savefig(outplotname)
                        print 'Saved plot at', outplotname
                    plt.show(block=False)

                # --------to plot pressure-size phase space------------
                if args.plot_phase_space:
                    outplotname = outpath+fn+'_P_vs_r_col_logQ_logOHgrad='+str(logOHgrad)+',Om='+str(Om)+wo+'.eps'
                    plot_phase_space(paramlist, outplotname, args)

                # ---------------to merge different Omega files to create one file with approriately radially binned variable Omega-------------------
                if args.mergeOmega:
                    paramlist['Omega_orig'] = Om
                    paramlist['Omega_variable'] = paramlist.apply(lambda row: getOmega(row.distance), axis=1)
                    paramlist['Omega_bin'] = paramlist.apply(lambda row: find_nearest(Om_arr, row.Omega_variable), axis=1)
                    paramlist_sub = paramlist[paramlist['Omega_orig'] == paramlist['Omega_bin']]
                    paramlist_Om_merged = paramlist_Om_merged.append(paramlist_sub)

            # -------------Omega loop ends here----------------------------
            if args.mergeOmega and args.write_file:
                if not os.path.exists(fout_Om_merged) or args.clobber:
                    header = 'Units for the following columns: \n\
                    x, y, z: kpc \n\
                    vel_x: km/s \n\
                    vel_y: km/s \n\
                    vel_z: km/s \n\
                    age: Myr \n\
                    mass: Msun \n\
                    gas_P in a cell: N/m^2 \n\
                    Q_H0: photons/s \n\
                    logQ: log(photons/s) \n\
                    r_stall: pc \n\
                    r_inst: pc \n\
                    r: pc \n\
                    <U>: volumne averaged, dimensionless \n\
                    nII: HII region number density per m^3\n\
                    log(P/k): SI units \n\
                    Z: metallicity in Zsun units \n\
                    fluxes: ergs/s/cm^2'

                    np.savetxt(fout_Om_merged, [], header=header, comments='#')
                    paramlist_Om_merged.to_csv(fout_Om_merged, sep='\t', mode='a', index=None)
                    print 'Omega merged parameter list saved at', fout_Om_merged, '\n'
                else:
                    print 'Reading existing Omega merged file from', fout_Om_merged, '\n'
                    paramlist_Om_merged = pd.read_table(fout_Om_merged, delim_whitespace=True, comment='#')

            # ---to check final distribution of Z after merging Omega---
            if args.plot_metgrad_mergeOmega:
                outplotname = outpath + fn + '_Zgrad_' + diag + '_col_density_logOHgrad=' + str(logOHgrad) + ',Om=-99' + wo + mergeHII_text + '.eps'
                plot_metgrad(paramlist_Om_merged, Zoutcol, logOHgrad, logOHcen, logOHsun, outplotname, args)

            # --------to plot final pressure-size phase space after merging Omega------------
            if args.plot_phase_space_mergeOmega:
                outplotname = outpath + fn + '_P_vs_r_col_logQ_logOHgrad=' + str(logOHgrad) + ',Om=-99' + wo + '.pdf'
                plot_phase_space(paramlist_Om_merged, outplotname, args)

    # ------------------------------------------------------------------------------------------------
    if not args.write_file: print 'Text files not saved.'
    print('Done in %s minutes' % ((time.time() - start_time) / 60))

    if args.mergeOmega: return paramlist, paramlist_Om_merged
    else: return paramlist

# ---------------defining constants-----------------------------------------
ylim_dict = {-0.1:-1.2, -0.05:-0.9, -0.025:-0.5, -0.01:-0.02}
#alpha_B = 3.46e-19  # m^3/s OR 3.46e-13 cc/s, Krumholz & Matzner (2009) for 7e3 K
alpha_B = 2.59e-19  # m^3/s OR 2.59e-13 cc/s, for Te = 1e4 K, referee quoted this values
c = 3e8  # m/s
ltemp = 4.  # assumed 1e4 K temp

# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # -------------------arguments parsed-------------------------------------------------------
    parser = ap.ArgumentParser(description="dummy")
    parser.add_argument('--write_file', dest='write_file', action='store_true')
    parser.set_defaults(write_file=False)
    parser.add_argument('--plot_metgrad', dest='plot_metgrad', action='store_true')
    parser.set_defaults(plot_metgrad=False)
    parser.add_argument('--plot_metgrad_mergeOmega', dest='plot_metgrad_mergeOmega', action='store_true')
    parser.set_defaults(plot_metgrad_mergeOmega=False)
    parser.add_argument('--plot_Zinout', dest='plot_Zinout', action='store_true')
    parser.set_defaults(plot_Zinout=False)
    parser.add_argument('--plot_hist', dest='plot_hist', action='store_true')
    parser.set_defaults(plot_hist=False)
    parser.add_argument('--plot_lummap', dest='plot_lummap', action='store_true')
    parser.set_defaults(plot_lummap=False)
    parser.add_argument('--plot_HIIgrid', dest='plot_HIIgrid', action='store_true')
    parser.set_defaults(plot_HIIgrid=False)
    parser.add_argument('--plot_fluxgrid', dest='plot_fluxgrid', action='store_true')
    parser.set_defaults(plot_fluxgrid=False)
    parser.add_argument('--plot_phase_space', dest='plot_phase_space', action='store_true')
    parser.set_defaults(plot_phase_space=False)
    parser.add_argument('--plot_phase_space_mergeOmega', dest='plot_phase_space_mergeOmega', action='store_true')
    parser.set_defaults(plot_phase_space_mergeOmega=False)
    parser.add_argument('--plothex', dest='plothex', action='store_true')
    parser.set_defaults(plothex=False)
    parser.add_argument('--plot_obs', dest='plot_obs', action='store_true')
    parser.set_defaults(plot_obs=False)
    parser.add_argument('--allstars', dest='allstars', action='store_true')
    parser.set_defaults(allstars=False)
    parser.add_argument('--nooutliers', dest='nooutliers', action='store_true')
    parser.set_defaults(nooutliers=False)
    parser.add_argument('--keep', dest='keep', action='store_true')
    parser.set_defaults(keep=False)
    parser.add_argument('--saveplot', dest='saveplot', action='store_true')
    parser.set_defaults(saveplot=False)
    parser.add_argument('--clobber', dest='clobber', action='store_true')
    parser.set_defaults(clobber=False)
    parser.add_argument('--annotate', dest='annotate', action='store_true')
    parser.set_defaults(annotate=False)
    parser.add_argument('--pause', dest='pause', action='store_true')
    parser.set_defaults(pause=False)
    parser.add_argument('--use_RGI', dest='use_RGI', action='store_true')
    parser.set_defaults(use_RGI=False)
    parser.add_argument('--mergeOmega', dest='mergeOmega', action='store_true')
    parser.set_defaults(mergeOmega=False)

    parser.add_argument("--file")
    parser.add_argument("--diag")
    parser.add_argument("--Om")
    parser.add_argument("--logOHgrad")
    parser.add_argument("--fontsize")
    parser.add_argument("--outpath")
    parser.add_argument("--mergeHII")
    parser.add_argument("--xratio")
    parser.add_argument("--yratio")
    args, leftovers = parser.parse_known_args()

    allstars_text = '_allstars' if args.allstars else ''
    mergeHII_text = '_mergeHII='+str(args.mergeHII)+'kpc' if args.mergeHII is not None else ''
    if args.file is not None: fn = args.file
    else: fn = 'DD0600_lgf'

    # --------------reading in the parameter list-----------------------------------------------------------
    paramlist = pd.read_table(HOME+'/models/rad_list/rad_list_newton' + allstars_text + '_' + fn + mergeHII_text, delim_whitespace=True, comment='#')
    paramlist = lookup_full_df(paramlist, args)
