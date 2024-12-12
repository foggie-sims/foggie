#!/usr/bin/env python3

"""

    Title :      make_table_fitted_quant
    Notes :      Combine the fitted gradient, mean and width of histograms for all snapshots and halos into one big machine readable table for the paper
    Output :     One big ASCII table + one small latex table to go in the paper as sample
    Author :     Ayan Acharyya
    Started :    Aug 2023
    Examples :   run make_table_fitted_quant.py --system ayan_local --halo 8508,5036,5016,4123,2878,2392 --Zgrad_den kpc --upto_kpc 10 --forpaper
"""
from header import *
from util import *
from plot_MZscatter import load_df
from uncertainties import unumpy
start_time = time.time()


# ------------------------------------------------------------------------------
def insert_line_in_file(line, pos, filename, output=None):
    '''
    Function to inserting a line in a file, given the filename and what and where to insert
    '''
    f = open(filename, 'r')
    contents = f.readlines()
    f.close()

    if pos == -1: pos = len(contents)  # to append to end of file
    contents.insert(pos, line)

    if output is None: output = filename
    f = open(output, 'w')
    contents = ''.join(contents)
    f.write(contents)
    f.close()
    return

# ----------------------------------------------------------------
def get_header(df):
    '''
    Function to obtain the header for a given table, based on the column names
    Returns: header as string AND modifies the column names in input df according to a dictionary inside this function
    '''
    header_dict = {'halo': 'Name of the FOGGEI halo', 'output': 'Name of the snapshot', 'redshift': 'Redshift of snapshot', 'time': 'Cosmological time of the snapshot (Gyr)', \
                   'log_mass': 'Log of stellar mass (in Msun)', 'log_Ztotal': 'Log of total metallicity (in Zsun)', \
                   'Zgrad': 'Fitted gradient of the radial profile, with uncertainty (in dex/kpc)', 'log_Z50': 'Log of the median of the metallicity distribution (in Zsun)', \
                   'log_ZIQR': 'Log of inter-quartile range of the distribution (75th - 25th percentile) (in Zsun)', \
                   'log_Zmean': 'Log of fitted center of the skewed Gaussian, with uncertainty (in Zsun)', 'log_Zwidth': 'Log of fitted width (FWHM) of the skewed Gaussian, with uncertainty (in Zsun)'}

    header = ''
    for thiscol in df.columns: header += '# ' + header_dict[thiscol] + '\n'

    return header


# ----------------------------------------------------------------
def make_latex_table(df, tabname, args):
    '''
    Function to minimise the given larger master table into a small latex table for the paper
    Returns: saves .tex file at the destination given by outtabname
    '''
    column_dict = {'halo':'Halo', 'output':'Output', 'redshift':r'$z$', 'time':'Time (Gyr)', 'log_mass':r'M$_{\star}$/M$_{\odot}$', 'log_Ztotal':'$\log Z_{\rm total}$/Z$_{\odot}$', 'Zgrad':r'$\nabla Z$ (dex/kpc)', 'log_Z50':'$\log Z_{\rm median}$/Z$_{\odot}$', 'log_ZIQR':'$\log Z_{\rm IQR}$/Z$_{\odot}$', 'log_Zmean':'Z$_{\rm cen}$/Z$_{\odot}$', 'log_Zwidth':'Z$_{\rm width}$/Z$_{\odot}$'}
    redshift_arr = [0, 1, 2]
    columns_with_unc = ['log_Zmean', 'log_Zwidth', 'Zgrad']
    decimal_dict = defaultdict(lambda: 2, redshift=1, Zgrad=3)

    tex_df = pd.DataFrame(columns = df.columns)
    for args.halo in args.halo_arr:
        for thisredshift in redshift_arr:
            thishalo = halo_dict[args.halo]
            thisrow = df[(df['halo'] == thishalo) & (df['redshift'].between(thisredshift - 0.01, thisredshift + 0.01))].iloc[0:1]
            tex_df = tex_df.append(thisrow)

    for thiscol in tex_df.columns:
        try:
            if thiscol in columns_with_unc: # special treatment for columns with +/-
                tex_df[thiscol] = [('$%.' + str(decimal_dict[thiscol]) + 'f\pm%.' + str(decimal_dict[thiscol]) + 'f$') % (item.n, item.s) for item in tex_df[thiscol].values]
            elif thiscol == 'log_Ztotal' or thiscol == 'log_Z50':
                tex_df[thiscol] = tex_df[thiscol].map(lambda x: '$%.2f$' % x if x < 0 else '$\phantom{-}%.2f$' % x)
            else:
                tex_df[thiscol] = tex_df[thiscol].map(('${:,.' + str(decimal_dict[thiscol]) + 'f}$').format)
        except ValueError: # columns that do not have numbers
            continue

    tex_df = tex_df.rename(columns=column_dict) # change column names to nice ones
    tex_df.drop(labels='Output', axis=1, inplace=True) # paper doesn't need DD names

    tex_df.to_latex(tabname, index=None, escape=False)
    print('Saved latex table at', tabname)

    return tex_df

# -----main code-----------------
if __name__ == '__main__':
    args_tuple = parse_args('8508', 'RD0042')  # default simulation to work upon when comand line args not provided
    if type(args_tuple) is tuple: args = args_tuple[0] # if the sim has already been loaded in, in order to compute the box center (via utils.pull_halo_center()), then no need to do it again
    else: args = args_tuple
    if not args.keep: plt.close('all')

    # ---------preset values for plotting for paper-------------
    if args.forpaper:
        args.weight = 'mass'
        args.fit_multiple = True
        args.use_density_cut = True
        args.islog = True
        args.docomoving = True
        args.get_native_res = True

    args.weightby_text = '' if args.weight is None else '_wtby_' + args.weight
    args.fitmultiple_text = '_fitmultiple' if args.fit_multiple else ''
    args.density_cut_text = '_wdencut' if args.use_density_cut else ''
    args.islog_text = '_islog' if args.islog else ''

    if args.upto_kpc is not None: upto_text = '_upto%.1Fckpchinv' % args.upto_kpc if args.docomoving else '_upto%.1Fkpc' % args.upto_kpc
    else: upto_text = '_upto%.1FRe' % args.upto_re

    # ---------setting up master dataframe------------------------
    #cols_in_df = ['halo', 'output', 'redshift', 'time', 'log_mass', 'log_Ztotal',  'log_Z50', 'log_ZIQR', 'log_Zmean', 'log_Zmean_u', 'log_Zwidth', 'log_Zwidth_u', 'Zgrad', 'Zgrad_u']
    cols_in_df = ['halo', 'output', 'redshift', 'time', 'log_mass', 'log_Ztotal',  'log_Z50', 'log_ZIQR', 'Zgrad_binned', 'Zgrad_u_binned'] # removed fitted center and width for paper
    master_df = pd.DataFrame(columns=cols_in_df)
    tex_path = HOME + '/Documents/writings/papers/FOGGIE_Zgrad/Tables/'
    master_filename = tex_path + 'master_table_Zpaper%s%s%s%s%s.txt' % (upto_text, args.weightby_text, args.fitmultiple_text, args.density_cut_text, args.islog_text)

    # --------loop over different FOGGIE halos-------------
    for index, args.halo in enumerate(args.halo_arr):
        # -------- loading dataframe-------
        try:
            df = load_df(args)
        except FileNotFoundError as e:
            pass

        # -------- reading in and merging dataframe with SFR info-------
        sfr_filename = args.code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/sfr'
        if os.path.exists(sfr_filename):
            print('Reading SFR history from', sfr_filename)
            sfr_df = pd.read_table(sfr_filename, names=('output', 'redshift', 'sfr'), comment='#', delim_whitespace=True)
            df = df.merge(sfr_df[['output', 'sfr']], on='output')
        else:
            print('Did not find', sfr_filename, ', therefore will not plot the SFR-related panels')
        df = df.sort_values('time')

        # ---------filter out only necessary columns--------------
        df = df[cols_in_df[1:]]
        df.drop_duplicates(subset='output', keep='last', ignore_index=True, inplace=True)
        df['halo'] = halo_dict[args.halo]
        master_df = master_df.append(df)

    # ---------collate and save master df------------
    master_df = master_df[cols_in_df]
    master_df = master_df.rename(columns={'Zgrad_binned': 'Zgrad', 'Zgrad_u_binned': 'Zgrad_u'})
    cols_with_u = ['Zgrad', 'log_Zmean', 'log_Zwidth']
    for thiscol in cols_with_u:
        if thiscol in master_df:
            master_df[thiscol] = unumpy.uarray(master_df[thiscol].values, master_df[thiscol + '_u'].values)
            master_df.drop(labels=thiscol+'_u', axis=1, inplace=True)

    header = get_header(master_df)
    master_df.to_csv(master_filename, sep='\t', index=None, header=True)
    insert_line_in_file(header, 0, master_filename)
    print('Wrote to master ASCII file', master_filename)

    # ---------make highlight table for paper------------
    tex_tabname = tex_path + os.path.splitext(os.path.split(master_filename)[1])[0].replace('master', 'small') + '.tex'
    tex_df = make_latex_table(master_df, tex_tabname, args)

    print('Completed in %s' % (datetime.timedelta(minutes=(time.time() - start_time) / 60)))
