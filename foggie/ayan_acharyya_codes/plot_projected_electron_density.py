#!/usr/bin/env python3

"""

    Title :      plot_projected_electron_density
    Notes :      Plot projected electron density as as function of SFR and stellar mass for ALL halos and snapshots in one plot
    Output :     One plot as png file
    Author :     Ayan Acharyya
    Started :    May 2024
    Examples :   run plot_projected_electron_density.py --system ayan_pleiades --upto_kpc 10 --res 0.2 --docomoving --do_all_halos
                 run plot_projected_electron_density.py --system ayan_hd --do_all_halos --upto_kpc 10 --res 0.2 --docomoving --nbins 100 --nocolorcoding
"""
from header import *
from util import *
plt.rcParams['axes.linewidth'] = 1
from datetime import datetime, timedelta

start_time = datetime.now()

# -----main code-----------------
if __name__ == '__main__':
    args_tuple = parse_args('8508', 'RD0042')  # default simulation to work upon when comand line args not provided
    if type(args_tuple) is tuple: args, ds, refine_box = args_tuple # if the sim has already been loaded in, in order to compute the box center (via utils.pull_halo_center()), then no need to do it again
    else: args = args_tuple
    if not args.keep: plt.close('all')

    # --------determing filenames and make figure-----------------
    args.fontsize, args.fontfactor, pointsize, col_quant = 15, 1.2, 50, 'time'
    if args.do_all_halos: args.halo_arr = ['8508', '5036', '5016', '4123', '2878', '2392']

    args.weightby_text = '_wtby_' + args.weight if args.weight is not None else ''
    args.upto_text = '_upto%.1Fckpchinv' % args.upto_kpc if args.docomoving else '_upto%.1Fkpc' % args.upto_kpc
    args.res_text = '_res%.1Fckpchinv' % float(args.res) if args.docomoving else '_res%.1Fkpc' % float(args.res)
    args.nbins_text = '_nbins%d' % args.nbins
    figname = args.output_dir + 'figs/' + ','.join(args.halo_arr) + '_el_density_vs_sfr_mass%s%s%s%s.png' % (args.upto_text, args.res_text, args.nbins_text, args.weightby_text)
    outfilename = args.output_dir + 'txtfiles/' + ','.join(args.halo_arr) + '_projected_el_density_evolution_combined%s%s%s%s.txt' % (args.upto_text, args.res_text, args.nbins_text, args.weightby_text)

    ax_arr = ['x', 'y', 'z']
    quant_arr = ['ed_peak', 'ed_width']
    label_dict = {'ed_peak':r'Peak ED distribution log (1/cm$^2$)', 'ed_width':r'Width of ED distribution log (1/cm$^2$)', 'time':'Time (Gyr)'}
    lim_dict = {'ed_peak': (18.5, 20.5), 'ed_width': (0.2, 1.4), 'time': (0, 14)}
    col_arr = ['rebeccapurple', 'chocolate', 'darkgreen', 'darkblue', 'crimson', 'darkkhaki']
    cmap_dict = {'8508':'Purples', '4123':'Oranges', '5036':'Greens', '5016':'Blues', '2878':'PuRd', '2392':'Greys'}

    fig, axes = plt.subplots(len(quant_arr), 2, sharex='col', sharey='row', figsize=(12, 7))
    ax_sfr, ax_mass = axes[:, 0], axes[:, 1]
    fig.subplots_adjust(top=0.95 if args.nocolorcoding else 0.88, bottom=0.12 if args.nocolorcoding else 0.08, left=0.12, right=0.97)

    df_master = pd.DataFrame()

    # --------loop over different FOGGIE halos-------------
    for index, args.halo in enumerate(args.halo_arr):
        args.foggie_dir, args.output_dir, args.run_loc, args.code_path, args.trackname, args.haloname, args.spectra_dir, args.infofile = get_run_loc_etc(args)
        filename = args.output_dir + 'txtfiles/' + args.halo + '_projected_el_density_evolution%s%s%s%s.txt' % (args.upto_text, args.res_text, args.nbins_text, args.weightby_text)
        print('For halo %d out of %d, reading file %s...' %(index + 1, len(args.halo_arr), filename))

        df = pd.read_table(filename, delim_whitespace=True, comment='#')
        df = df.drop_duplicates(subset='output', keep='last')
        df['halo'] = args.halo

        for ind, quant in enumerate(quant_arr):
            df[quant] = df[np.hstack([quant + '_' + ax] for ax in ['x', 'y', 'z'])].mean(axis=1)
            df = df[~df[['sfr', 'log_mstar', quant]].isin([np.nan, np.inf, -np.inf]).any(1)]

            p = ax_sfr[ind].scatter(df['sfr'], df[quant], c=col_arr[index] if args.nocolorcoding else df[col_quant], cmap=None if args.nocolorcoding else cmap_dict[args.halo], s=pointsize, lw=0)
            p = ax_mass[ind].scatter(df['log_mstar'], df[quant], c=col_arr[index] if args.nocolorcoding else df[col_quant], cmap=None if args.nocolorcoding else cmap_dict[args.halo], s=pointsize, lw=0)

            ax_sfr[ind].set_ylim(lim_dict[quant][0], lim_dict[quant][1]) # the small offset between the actual limits and intended tick labels is to ensure that tick labels do not reach the very edge of the plot
            ax_sfr[ind].set_yticklabels(['%.2F' % item for item in ax_sfr[ind].get_yticks()], fontsize=args.fontsize)
            ax_sfr[ind].set_ylabel(label_dict[quant], fontsize=args.fontsize / args.fontfactor)

        df_master = df_master.append(df[np.hstack((['halo', 'output', 'redshift', 'time', 'sfr', 'log_mstar'], quant_arr))])

    ax_sfr[-1].set_xlim(0, 15)
    ax_mass[-1].set_xlim(8.5, 11)

    ax_sfr[-1].set_xticklabels(['%.1F' % item for item in ax_sfr[-1].get_xticks()], fontsize=args.fontsize)
    ax_mass[-1].set_xticklabels(['%.1F' % item for item in ax_mass[-1].get_xticks()], fontsize=args.fontsize)

    ax_sfr[-1].set_xlabel(r'SFR (M$_{\odot}$/yr)', fontsize=args.fontsize)
    ax_mass[-1].set_xlabel(r'$\log{(\mathrm{M}_* / \mathrm{M}_{\odot})}$', fontsize=args.fontsize)

    # ---------making the colorbar axis once, that will correspond to all panels--------------
    if not args.nocolorcoding:
        cax_xpos, cax_ypos, cax_width, cax_height = 0.12, 0.93, 0.85, 0.02
        cax = fig.add_axes([cax_xpos, cax_ypos, cax_width, cax_height])
        plt.colorbar(p, cax=cax, orientation='horizontal')

        cax.set_xticklabels(['%.1F' % index for index in cax.get_xticks()], fontsize=args.fontsize / args.fontfactor)
        fig.text(cax_xpos + cax_width / 2, cax_ypos + cax_height + 0.005, label_dict[col_quant], ha='center', va='bottom', fontsize=args.fontsize)

    fig.savefig(figname, transparent=args.glasspaper or args.fortalk)
    print('\nSaved plot as', figname)

    df_master.to_csv(outfilename, sep='\t', index=None)
    print('\nSaved master dataframe as', outfilename)

    plt.show(block=False)
    print('Everything completed in %s' % timedelta(seconds=(datetime.now() - start_time).seconds))

