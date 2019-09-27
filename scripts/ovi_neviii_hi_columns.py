'''
November 2018
Molly Peeples

calculates OVI and NeVIII columns (medians, percentiles, ratios);
largely for comparison to CASBAH  data 

'''

from __future__ import print_function

import numpy as np
import yt
import sys
import os
import argparse

import pickle


from foggie.utils.get_refine_box import get_refine_box
from foggie.utils.get_proper_box_size import get_proper_box_size


def parse_args():
    '''
    Parse command line arguments.  Returns args object.
    '''
    parser = argparse.ArgumentParser(description="makes a bunch of plots")

    parser.add_argument('--output', metavar='output', type=str, action='store',
                        help='which output? default is RD0020')
    parser.set_defaults(output="RD0020")

    parser.add_argument('--calc', dest='calc', action='store_true')
    parser.add_argument('--no-calc', dest='calc', action='store_false', help="default is no calc")
    parser.set_defaults(calc=False)

    parser.add_argument('--compile', dest='compile', action='store_true')
    parser.add_argument('--no-compile', dest='compile', action='store_false', help="default is not compiling the big table")
    parser.set_defaults(compile=False)

    parser.add_argument('--plot', dest='plot', action='store_true')
    parser.add_argument('--no-plot', dest='plot', action='store_false', help="default is no plotting")
    parser.set_defaults(plot=False)

    args = parser.parse_args()
    return args


def calc_cddf(output):
    import trident
    # load the simulation
    outputds = output + '/' + output
    forced_ds = yt.load(outputds)

    track_name = "/nobackupp2/mpeeples/halo_008508/orig/nref11n_nref10f_orig/halo_track"
    output_dir = "/nobackupp2/mpeeples/halo_008508/orig/nref11n_nref10f_orig/ovi_neviii/"
    ## output_dir = "/Users/molly/Dropbox/foggie-collab/plots/halo_008508/natural/nref11/spectra/"
    ## os.chdir(output_dir)
    track = Table.read(track_name, format='ascii')
    track.sort('col1')
    zsnap = forced_ds.get_parameter('CosmologyCurrentRedshift')

    proper_box_size = get_proper_box_size(forced_ds)
    forced_box, forced_c, width = get_refine_box(forced_ds, zsnap, track)
    width = width * proper_box_size
    print("width = ", width, "kpc")

    resolution = (1048,1048)
    radii = np.zeros(resolution)
    indices = np.indices(resolution)
    for x in range(0, 1048):
        for y in range(0, 1048):
            radii[x,y] = (width / 1048) * np.sqrt((524-x)**2 + (524-y)**2)
    big_radii = np.concatenate((radii, radii, radii), axis=None)
    pkl_name = output_dir + 'radii_' + output + '_physicalkpc.pkl'
    print("saving to ", pkl_name)
    pickle.dump(big_radii, open( pkl_name, "wb" ) )

    trident.add_ion_fields(forced_ds, ions=['O VI', 'Ne VIII'])

    ions = ['H_p0_number_density', 'O_p5_number_density', 'Ne_p7_number_density']
    fields = []
    for ion in ions:
        field = ('gas', ion)
        fields.append(field)

    dp_forced_x = yt.ProjectionPlot(forced_ds, 'x', fields, center=forced_c, \
                        width=(width,"kpc"), data_source=forced_box)
    dp_forced_y = yt.ProjectionPlot(forced_ds, 'y', fields, center=forced_c, \
                        width=(width,"kpc"), data_source=forced_box)
    dp_forced_z = yt.ProjectionPlot(forced_ds, 'z', fields, center=forced_c, \
                        width=(width,"kpc"), data_source=forced_box)


    for ion in ions:
        colr = np.array([])
        print("trying ",ion)

        frb = dp_forced_x.data_source.to_frb((width,'kpc'), resolution)
        forced = np.array(np.log10(frb[ion]))
        colr = np.append(colr, forced.ravel(), axis=None)

        frb = dp_forced_y.data_source.to_frb((width,'kpc'), resolution)
        forced = np.array(np.log10(frb[ion]))
        colr = np.append(colr, forced.ravel(), axis=None)

        frb = dp_forced_z.data_source.to_frb((width,'kpc'), resolution)
        forced = np.array(np.log10(frb[ion]))
        colr = np.append(colr, forced.ravel(), axis=None)

        colr[colr == -np.inf] = 1


        pkl_name = output_dir + ion + '_nref10f_' + output + '_column_densities.pkl'
        print("saving to ", pkl_name)
        pickle.dump(colr, open( pkl_name, "wb" ) )

def compile_columns():
    from astropy.table import Table
    from astropy.io import ascii
    min_o6 = 12.75
    outputs = ascii.read('outputs.txt')
    data = Table(names=('DD', 'redshift',
        'ovi10', 'ovi25', 'ovi34', 'ovi50', 'ovi68', 'ovi75', 'ovi90',
        'neviii10', 'neviii25', 'neviii34', 'neviii50', 'neviii68', 'neviii75', 'neviii90',
        'neviiilim10', 'neviiilim25', 'neviiilim34', 'neviiilim50', 'neviiilim68', 'neviiilim75', 'neviiilim90',
        'ratio10', 'ratio25', 'ratio34','ratio50', 'ratio68',' ratio75', 'ratio90',
        'ratiolim10', 'ratiolim25', 'ratiolim34', 'ratiolim50', 'ratiolim68', 'ratiolim75', 'ratiolim90'))
    data['DD'] = data['DD'].astype('str')

    for i, dd in enumerate(outputs['dd']):
        redshift = outputs['redshift'][i]
        print('trying ', dd)
        try:
            ion = 'O_p5_number_density'
            # print("trying ",ion)
            pkl_name = ion + '_nref10f_' + dd + '_column_densities.pkl'
            print("opening ", pkl_name)
            try:
                o6 = pickle.load(open( pkl_name, "rb" ) ,encoding='latin1')
            except:
                o6 = pickle.load(open( pkl_name, "rb" ))
            ion = 'Ne_p7_number_density'
            # print("trying ",ion)
            pkl_name = ion + '_nref10f_' + dd + '_column_densities.pkl'
            print("opening ", pkl_name)
            try:
                ne8 = pickle.load(open( pkl_name, "rb"), encoding='latin1' )
            except:
                ne8 = pickle.load(open( pkl_name, "rb"))
            ratio = np.power(10.0, ne8) / np.power(10.0, o6)
            #print(np.percentile(ratio, 10), np.percentile(ratio, 50), np.percentile(ratio, 90))
            ind = np.where(o6 > min_o6)
            ratiolim = np.power(10.0, ne8[ind]) / np.power(10.0, o6[ind])
            #print(np.percentile(ratiolim, 10), np.percentile(ratiolim, 50), np.percentile(ratiolim, 90))
            ne8lim = ne8[ind]
            data.add_row([dd, redshift,
                  np.percentile(o6, 10), np.percentile(o6, 25), np.percentile(o6, 34), np.percentile(o6, 50),
                  np.percentile(o6, 68),  np.percentile(o6, 75), np.percentile(o6, 90),
                  np.percentile(ne8, 10), np.percentile(ne8, 25), np.percentile(ne8, 34), np.percentile(ne8, 50),
                  np.percentile(ne8, 68),  np.percentile(ne8, 75), np.percentile(ne8, 90),
                  np.percentile(ne8lim, 10), np.percentile(ne8lim, 25), np.percentile(ne8lim, 34), np.percentile(ne8lim, 50),
                  np.percentile(ne8lim, 68),  np.percentile(ne8lim, 75), np.percentile(ne8lim, 90),
                  np.percentile(ratio, 10), np.percentile(ratio, 25), np.percentile(ratio, 34), np.percentile(ratio, 50),
                  np.percentile(ratio, 68),  np.percentile(ratio, 75), np.percentile(ratio, 90),
                  np.percentile(ratiolim, 10), np.percentile(ratiolim, 25), np.percentile(ratiolim, 34), np.percentile(ratiolim, 50),
                  np.percentile(ratiolim, 68),  np.percentile(ratiolim, 75), np.percentile(ratiolim, 90)])

            data.write('nref10f_o6_ne8.dat', format='ascii.basic')
        except:
            print('output ', dd, ' does not seem to have any pkls, so sad')

def plot_columns():
    from astropy.io import ascii
    data = ascii.read('nref10f_o6_ne8.dat')
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rcParams['font.family'] = 'stixgeneral'
    mpl.rcParams['font.size'] = 26.


    fig = plt.figure(figsize=(16,8))
    ax = fig.add_subplot(111)
    ax.fill_between(data['redshift'], np.log10(data['ratiolim10']), np.log10(data['ratiolim90']), facecolor='darkorange', alpha=0.2)
    ax.fill_between(data['redshift'], np.log10(data['ratiolim25']), np.log10(data['ratiolim75']), facecolor='darkorange', alpha=0.2)
    ax.step(data['redshift'], np.log10(data['ratiolim50']), lw=5, color='darkorange')
    ax.text(0.5, 0.25, 'min log N(OVI) = 12.75', fontsize=34)
    plt.xlabel(r'redshift', fontsize=34)
    plt.ylabel('log [N(NeVIII)/N(OVI)]', fontsize=34)
    fig.tight_layout()
    fig.savefig('redshift_ratiolim.png')

    fig = plt.figure(figsize=(16,8))
    ax = fig.add_subplot(111)
    ax.fill_between(data['redshift'], np.log10(data['ratiolim10']), np.log10(data['ratiolim90']), facecolor='darkorange', alpha=0.2)
    ax.fill_between(data['redshift'], np.log10(data['ratiolim25']), np.log10(data['ratiolim75']), facecolor='darkorange', alpha=0.2)
    ax.step(data['redshift'], np.log10(data['ratiolim50']), lw=5, color='darkorange')
    ax.step(data['redshift'], np.log10(data['ratio10']), color='#4daf4a', ls=':')
    ax.step(data['redshift'], np.log10(data['ratio90']), color='#4daf4a', ls=':')
    ax.step(data['redshift'], np.log10(data['ratio50']), lw=5, color='#4daf4a')
    ax.text(0.5, 0.45, 'min log N(OVI) = 12.75', color='darkorange',fontsize=34)
    plt.xlabel(r'redshift', fontsize=34)
    plt.ylabel('log [N(NeVIII)/N(OVI)]', fontsize=34)
    fig.tight_layout()
    fig.savefig('redshift_ratioboth.png')

    fig = plt.figure(figsize=(16,8))
    ax = fig.add_subplot(111)
    ax.fill_between(data['redshift'], data['ovi10'], data['ovi90'], facecolor='darkorange', alpha=0.2)
    ax.fill_between(data['redshift'], data['ovi25'], data['ovi75'], facecolor='darkorange', alpha=0.2)
    ax.step(data['redshift'], data['ovi50'], lw=5, color='darkorange', label='O VI ')
    ax.fill_between(data['redshift'], data['neviii10'], data['neviii90'], facecolor='#4daf4a', alpha=0.2)
    ax.fill_between(data['redshift'], data['neviii25'], data['neviii75'], facecolor='#4daf4a', alpha=0.2)
    ax.step(data['redshift'], data['neviii50'], lw=5, color='#4daf4a', label='Ne VIII')
    # ax.text(0.5, 0.45, 'min log N(OVI) = 12.75', color='darkorange',fontsize=34)
    plt.legend(loc='upper left')
    plt.xlabel(r'redshift', fontsize=34)
    plt.ylabel('log column density', fontsize=34)
    fig.tight_layout()
    fig.savefig('redshift_o6ne8.png')


if __name__ == "__main__":
    args = parse_args()
    print(args)
    if args.calc:
        calc_cddf(args.output)
    if args.compile:
        compile_columns()
    if args.plot:
        plot_columns()
    sys.exit("~~~*~*~*~*~*~all done!!!! yay column densities!")
