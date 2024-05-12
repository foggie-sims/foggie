#!/usr/bin/env python3

"""

    Title :      plot_spherical_electron_density
    Notes :      Plot spherical electron density as as function of SFR and stellar mass for ALL halos and snapshots in one plot
    Output :     One plot as png file
    Author :     Ayan Acharyya
    Started :    May 2024
    Examples :   run plot_spherical_electron_density.py --system ayan_pleiades --upto_kpc 50 --docomoving --do_all_halos
                 run plot_spherical_electron_density.py --system ayan_local --do_all_halos --upto_kpc 50 --docomoving --nbins 100 --nocolorcoding
"""
from header import *
from util import *
plt.rcParams['axes.linewidth'] = 1
from datetime import datetime, timedelta

start_time = datetime.now()

# ----------------------------------------------------
def discrete_cmap(N, base_cmap=None):
    '''
    Create an N-bin discrete colormap from the specified input map
    '''

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

# -----main code-----------------
if __name__ == '__main__':
    args_tuple = parse_args('8508', 'RD0042')  # default simulation to work upon when comand line args not provided
    if type(args_tuple) is tuple: args, ds, refine_box = args_tuple # if the sim has already been loaded in, in order to compute the box center (via utils.pull_halo_center()), then no need to do it again
    else: args = args_tuple
    if not args.keep: plt.close('all')

    # --------determing filenames-----------------
    if args.do_all_halos: args.halo_arr = ['8508', '5036', '5016', '4123', '2878', '2392']
    args.weightby_text = '_wtby_' + args.weight if args.weight is not None else ''
    args.upto_text = '_upto%.1Fckpchinv' % args.upto_kpc if args.docomoving else '_upto%.1Fkpc' % args.upto_kpc
    args.nbins_text = '_nbins%d' % args.nbins
    figname = args.output_dir + 'figs/' + ','.join(args.halo_arr) + '_el_density_vs_sfr_mass%s%s%s.png' % (args.upto_text, args.nbins_text, args.weightby_text)
    outfilename = args.output_dir + 'txtfiles/' + ','.join(args.halo_arr) + '_spherical_el_density_evolution_combined%s%s%s.txt' % (args.upto_text, args.nbins_text, args.weightby_text)

    # ---------------loading the combined dataframe file---------------------------
    if not os.path.exists(outfilename) or args.clobber:
        print('The following combined file does not exists, so making new..\n%s' %outfilename)
        df = pd.DataFrame()

        # --------loop over different FOGGIE halos-------------
        for index, args.halo in enumerate(args.halo_arr):
            args.foggie_dir, args.output_dir, args.run_loc, args.code_path, args.trackname, args.haloname, args.spectra_dir, args.infofile = get_run_loc_etc(args)
            filename = args.output_dir + 'txtfiles/' + args.halo + '_spherical_el_density_evolution%s%s%s.txt' % (args.upto_text, args.nbins_text, args.weightby_text)
            print('For halo %d out of %d, reading file %s...' %(index + 1, len(args.halo_arr), filename))

            df_halo = pd.read_table(filename, delim_whitespace=True, comment='#')
            df_halo = df_halo.drop_duplicates(subset='output', keep='last')
            df_halo['halo'] = args.halo
            df = df.append(df_halo)

        df.to_csv(outfilename, sep='\t', index=None)
        print('\nSaved master dataframe as', outfilename)
    else:
        print('Reading existing combined file..\n%s' %outfilename)
        df = pd.read_table(outfilename, delim_whitespace=True)

    # --------declaring plot parameters and dicts-----------------------
    args.fontsize, args.fontfactor, pointsize = 15, 1.5, 20
    quant_arr = ['central', 'alpha', 'break_rad', 'beta']
    label_dict = {'central':r'Central ED log (1/cm$^3$)', 'alpha':'Inner exponent', 'break_rad':'Break radius (kpc)', 'beta':'Outer exponent'}
    lim_dict = {'central': (-3, -1), 'alpha': (-2, 0.1), 'break_rad': (0, 15), 'beta': (-0.05, 0.05)}
    color = 'purple'

    binby_col, color_col, cmap = 'redshift', 'bin_color', 'magma'
    groupby_col = binby_col + '_bin'
    bins = [0, 0.02, 0.05, 0.1, 0.5, 3.0, 6.0] # non-linear redshift bins
    all_colors = []
    for val in np.linspace(0., 0.8, len(bins) - 1):
        rgb = plt.cm.get_cmap(cmap)(val, bytes=True)[:-1]
        all_colors.append("#{0:02x}{1:02x}{2:02x}".format(rgb[0], rgb[1], rgb[2]))
    all_bins = [pd.Interval(bins[item], bins[item + 1], closed='left') for item in range(len(bins) - 1)]
    col_dict = dict(zip(all_bins, all_colors))

    # ---------------setting up the figure---------------------------
    fig, axes = plt.subplots(len(quant_arr), 2, sharex='col', sharey='row', figsize=(4, 6))
    ax_sfr, ax_mass = axes[:, 0], axes[:, 1]
    fig.subplots_adjust(top=0.95, bottom=0.1, left=0.2, right=0.95)

    # --------binning by redshift-----------------------
    df[groupby_col] = pd.cut(df[binby_col], bins, right=False)
    df[color_col] = df[groupby_col].map(col_dict)

    # --------loop over different parameters and plot them-------------
    for ind, quant in enumerate(quant_arr):
        ax_sfr[ind].scatter(df['sfr'], df[quant], c=color if args.nocolorcoding else df[color_col], s=pointsize, lw=0)
        ax_sfr[ind].errorbar(df['sfr'], df[quant], c=color if args.nocolorcoding else df[color_col], yerr=df[quant + '_u'], lw=1, ls='none', zorder=1)

        ax_mass[ind].scatter(df['log_mstar'], df[quant], c=color if args.nocolorcoding else df[color_col], s=pointsize, lw=0)
        ax_mass[ind].errorbar(df['log_mstar'], df[quant], c=color if args.nocolorcoding else df[color_col], yerr=df[quant + '_u'], lw=1, ls='none', zorder=1)

        ax_sfr[ind].set_ylim(lim_dict[quant][0], lim_dict[quant][1])
        ax_sfr[ind].set_yticklabels(['%.2F' % item for item in ax_sfr[ind].get_yticks()], fontsize=args.fontsize / args.fontfactor)
        ax_sfr[ind].set_ylabel(label_dict[quant], fontsize=args.fontsize / args.fontfactor)

    # ------------annotating axes labels------------------
    for ind, this_bin in enumerate(all_bins): ax_mass[1].text(8.6, lim_dict[quant_arr[1]][0] + 0.2 + ind * 0.2, '%.2f < z <= %.2f' % (this_bin.left, this_bin.right) , c=col_dict[this_bin], fontsize = args.fontsize / args.fontfactor**2, ha='left', va='top')

    ax_sfr[-1].set_xlim(0, 15)
    ax_mass[-1].set_xlim(8.5, 11.5)

    ax_sfr[-1].set_xticklabels(['%.1F' % item for item in ax_sfr[-1].get_xticks()], fontsize=args.fontsize / args.fontfactor)
    ax_mass[-1].set_xticklabels(['%.1F' % item for item in ax_mass[-1].get_xticks()], fontsize=args.fontsize / args.fontfactor)

    ax_sfr[-1].set_xlabel(r'SFR (M$_{\odot}$/yr)', fontsize=args.fontsize / args.fontfactor)
    ax_mass[-1].set_xlabel(r'$\log{(\mathrm{M}_* / \mathrm{M}_{\odot})}$', fontsize=args.fontsize / args.fontfactor)

    # ------------saving the figure------------------
    fig.savefig(figname, transparent=args.glasspaper or args.fortalk)
    print('\nSaved plot as', figname)

    plt.show(block=False)
    print('Everything completed in %s' % timedelta(seconds=(datetime.now() - start_time).seconds))

