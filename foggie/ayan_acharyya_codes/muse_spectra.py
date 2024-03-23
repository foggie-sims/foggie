#!/usr/bin/env python3

"""

    Title :      muse_spectra.py
    Notes :      just trying something out, for Anshu
    Author :     Ayan Acharyya
    Started :    Nov 2023
    Examples :   run muse_spectra.py --input_dir /Users/acharyya/models/muse/
                 run muse_spectra.py --input_dir /Users/acharyya/models/muse/ --plot_vlines_at 6150,6420 --plot_interv_at_z 2.5,2.8
                 run muse_spectra.py --plot_vlines_at 6150,6420 --plot_interv_at_z 2.5,2.8

"""

import numpy as np
from matplotlib import pyplot as plt
plt.style.use('seaborn-whitegrid')
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 1
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import time, datetime, argparse, os
start_time = time.time()

# -----------------------------------------------
def load_intervening_linelist(args):
    '''
    Function to load intervening linelist
    Returns dataframe containing linelist
    '''
    color_arr = ['darkorchid', 'crimson', 'hotpink', 'darkmagenta'] # colors of vertical lines to plot for each redshift value of intervening absorber

    df = pd.read_table(args.input_dir + args.interv_linelist_file, delim_whitespace=True, comment='#')
    df['type'] = 'INTERV'

    if args.plot_interv_at_z is not None:
        full_df = pd.DataFrame()
        for index, thisz in enumerate([float(item) for item in args.plot_interv_at_z.split(',')]):
            print('Adding intervening lines to the list for redshift', thisz)
            df['redshift'] = thisz
            df['obswave'] = df['restwave'] * (1 + thisz)
            df['color'] = color_arr[index]
            full_df = full_df.append(df)
    else:
        full_df = None

    return full_df

# -----------------------------------------------
def load_linelist(path):
    '''
    Function to load and clean linelist
    Returns dataframe containing linelist
    '''
    df = pd.read_table(path, delim_whitespace=True, comment='%', names=['restwave', 'label1', 'label2', 'dummy1', 'dummy2', 'color', 'dummy3', 'type', 'notes'])
    df['label'] = df['label1'] + df['label2'].astype(str)
    df['color'] = df['type'].map(line_dict)
    df = df[['label', 'restwave', 'type', 'color']]
    df = df.sort_values(by='restwave')

    return df

# -----------------------------------------------
def plot_spectra(df, args, linelist=None, include_line_types=None, interv_linelist=None):
    '''
    Function to load and plot spectra in multiple panels
    Returns array of figure handles
    '''
    df = df.sort_values(by=args.wavecol)
    if linelist is not None:
        if include_line_types is not None: linelist = linelist[linelist['type'].isin(include_line_types)]
        linelist['obswave'] = linelist['restwave'] * (1 + args.re dshift)
        if args.plot_interv_at_z is not None: linelist = linelist.append(interv_linelist) # appending the intervening lines, so they al can be plotted

    if args.plot_vlines_at is not None: args.vlines_to_plot = np.array([float(item) for item in args.plot_vlines_at.split(',')])
    else: args.vlines_to_plot = np.array([])

    wave_span = df[args.wavecol].values[-1] - df[args.wavecol].values[0] + args.owave
    nrow_needed = int(np.ceil(wave_span / args.dwave))
    nfigs_needed = int(np.ceil(nrow_needed / args.nrow_max))

    this_end = df[args.wavecol].values[0] + args.owave
    fig_arr = []
    output_file = args.output_dir + os.path.splitext(args.spectra_file)[0] + '_plot.pdf'
    pdf = PdfPages(output_file)

    for fig_index in range(nfigs_needed):
        print('Making %d of %d figures' %(fig_index + 1, nfigs_needed))
        fig, axes = plt.subplots(args.nrow_max, 1, figsize=(8, 9))
        fig.subplots_adjust(top=0.93, bottom=0.07, left=0.1, right=0.98, hspace=0.5)

        for axis_index, ax in enumerate(axes):
            this_start = this_end - args.owave
            this_end = this_start + args.dwave + args.owave
            dfsub = df[df[args.wavecol].between(this_start, this_end)]

            ax.plot(dfsub[args.wavecol], dfsub[args.fluxcol], c='cornflowerblue', lw=1)
            ax.plot(dfsub[args.wavecol], dfsub[args.fluxucol], c='lightsteelblue', lw=0.5)
            vlines = args.vlines_to_plot[np.where(np.logical_and(args.vlines_to_plot >= this_start, args.vlines_to_plot <= this_end))[0]]
            for this_vline in vlines: ax.axvline(this_vline, lw=2, ls='dashed', c='magenta')

            if linelist is not None:
                thislist = linelist[linelist['obswave'].between(this_start, this_end)].reset_index(drop=True)
                for thisline in thislist.iloc:
                    ax.axvline(thisline['obswave'], c=thisline['color'], lw=1, ls='dashed')
                    ax.text(thisline['obswave'] - 1, 68, thisline['label'], color=thisline['color'], va='top', ha='right', rotation=90, fontsize=args.fontsize/2.5)

            ax.set_ylim(0, 70)
            ax.set_xlim(this_start, this_end)
            ax.tick_params(axis='both', labelsize=args.fontsize/2)

            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())
            ax2.tick_params(axis='both', labelsize=args.fontsize/2)
            ax2.set_xticklabels(['%d' % item for item in ax2.get_xticks() / (1 + args.redshift)])

        fig.text(0.5, 0.02, r'Observed Wavelength ($\AA$)', ha='center', fontsize=args.fontsize)
        fig.text(0.5, 0.96, r'Rest-frame Wavelength ($\AA$)', ha='center', fontsize=args.fontsize)
        fig.text(0.02, 0.5, r'Flux (units?)', va='center', rotation='vertical', fontsize=args.fontsize)

        fig_arr.append(fig)
        pdf.savefig(fig)
        plt.show(block=False)

    pdf.close()
    print('Saved figure in', output_file)
    return fig_arr, linelist


# ------------ global dictionaries -------------------------------------------------
source_dict = {1: 'muse_udf_1D_spec_lyc.csv', 2: 'muse_udf_1D_spec_background_source.csv'}
redshift_dict = {1: 3.08, 2: 3.43}
line_dict = {'EMISSION': 'saddlebrown', 'FINESTR': 'salmon', 'ISM': 'darkkhaki', 'PHOTOSPHERE': 'teal', 'WIND': 'darkgreen'}

# -----main code-----------------
if __name__ == '__main__':
    # ------------- arg parse -------------------------------
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='''kill me please''')

    parser.add_argument('--input_dir', metavar='input_dir', type=str, action='store', default='/Users/acharyya/models/muse/', help='Which directory is the spectra in? Default is Downloads')
    parser.add_argument('--output_dir', metavar='output_dir', type=str, action='store', default=None, help='Which directory is the spectra in? Default is None, in which case the input_dir will be used as output_dir')
    parser.add_argument('--spectra_file', metavar='spectra_file', type=str, action='store', default=None, help='Which file is the spectra in? Default is None, in which case it will be determined from the dictionary')
    parser.add_argument('--linelist_file', metavar='linelist_file', type=str, action='store', default='stacked_linelist', help='Which file is the linelist in? Default is Jane\'s stacked list')
    parser.add_argument('--interv_linelist_file', metavar='interv_linelist_file', type=str, action='store', default='intervening_linelist', help='Which file is the intervening linelist in? Default is None')
    parser.add_argument('--redshift', metavar='redshift', type=float, action='store', default=None, help='redshift of the source; default is None in which case it will be determined from the dictionary')
    parser.add_argument('--plot_interv_at_z', metavar='plot_interv_at_z', type=str, action='store', default=None, help='redshift/s of the intervening absorber; default is 3')
    parser.add_argument('--source', metavar='source', type=str, action='store', default=1, help='which source? default is 1')
    parser.add_argument('--wavecol', metavar='wavecol', type=str, action='store', default='wave', help='column name of wavelength; default is wave')
    parser.add_argument('--fluxcol', metavar='fluxcol', type=str, action='store', default='spectra', help='column name of flux; default is spectra')
    parser.add_argument('--fluxucol', metavar='fluxucol', type=str, action='store', default='noise', help='column name of flux uncertainty; default is noise')
    parser.add_argument('--fontsize', metavar='fontsize', type=int, action='store', default=20, help='tick label args.fontsize; default is 20')
    parser.add_argument('--nrow_max', metavar='nrow_max', type=int, action='store', default=5, help='maximum number of rows/panels in each page of figure; default is 5')
    parser.add_argument('--dwave', metavar='dwave', type=int, action='store', default=310, help='wavelength width of each row, in Angstrom; default is 310 Angstrom')
    parser.add_argument('--owave', metavar='owave', type=int, action='store', default=10, help='wavelength overlap between each row, in Angstrom; default is 10 Angstrom')
    parser.add_argument('--keep', dest='keep', action='store_true', default=False, help='keep previous plots opened?, default is no')
    parser.add_argument('--plot_vlines_at', metavar='plot_vlines_at', type=str, action='store', default=None, help='plot a bunch of vertical lines at specific observed wavelengths, in Angstrom? default is No')

    args = parser.parse_args()

    # --------------- defaults ----------------------------------------------
    if not args.keep: plt.close('all')
    if args.spectra_file is None: args.spectra_file = source_dict[args.source]
    if args.redshift is None: args.redshift = redshift_dict[args.source]
    if args.output_dir is None: args.output_dir = args.input_dir

    # ---------------- load spectra and linelists ------------------------
    df = pd.read_table(args.input_dir + args.spectra_file, delim_whitespace=True)
    ll = load_linelist(args.input_dir + args.linelist_file)
    interv_ll = load_intervening_linelist(args)

    # ---------------- plot spectra ------------------------
    fig, linelist = plot_spectra(df, args, linelist=ll, include_line_types = ['EMISSION', 'ISM'], interv_linelist=interv_ll)

    print('Completed in %s' % (datetime.timedelta(minutes=(time.time() - start_time) / 60)))
