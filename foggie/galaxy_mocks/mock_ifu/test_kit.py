##!/usr/bin/env python3

"""

    Title :      test_kit
    Notes :      Cluster of random functions for debugging; not well documented
    Author :     Ayan Acharyya
    Started :    April 2021
    Examdfe :    run test_kit.py --system ayan_local --halo 5036 --output RD0030 --mergeHII 0.04 --base_spatial_res 0.04 --z 0.25 --base_wave_range 0.64,0.68 --obs_wave_range 0.8,0.85

"""
from header import *
from util import *
from make_ideal_datacube import *
import foggie.galaxy_mocks.mock_ifu.make_mappings_grid as mmg
plt.style.use('seaborn')

# -----------------------------------------------------------------------------
def getscatterplot(df, quant1, quant2, colby=None):
    fig = plt.figure()
    plt.scatter(df[quant1], df[quant2], c=df[colby] if colby is not None else 'b')
    plt.xlabel(quant1)
    plt.ylabel(quant2)
    if colby is not None:
        cb=plt.colorbar()
        cb.set_label(colby)
    fig.text(0.6, 0.8, halo_dict[args.halo] + '; ' + args.output)
    plt.show(block=False)
    return fig

# -----------------------------------------------------------------------------
def getposplot(emlist, quant, is_col_log=False):
    '''
    Function to make scatter plot of y vs z coordinate of particles (in kpc, wrt halo center) and color code by column of choice
    '''
    df = emlist.copy()
    fig = plt.figure()
    if is_col_log: df[quant] = np.log10(df[quant])
    plt.scatter(df['pos_y_cen'], df['pos_z_cen'], c=df[quant])
    plt.xlim(-args.galrad, args.galrad)
    plt.ylim(-args.galrad, args.galrad)
    cb=plt.colorbar()
    cb.set_label(quant)
    fig.text(0.6, 0.8, halo_dict[args.halo] + '; ' + args.output)
    plt.show(block=False)
    return fig

# -----------------------------------------------------------------------------
def gethistplot(emlist, emlist_cutout, quant='Zin', quant_label='Gas metallicity Z/Zsun', is_col_log=False):
    df = emlist.copy()
    df_cutout = emlist_cutout.copy()
    fig = plt.figure()
    if is_col_log:
        df[quant] = np.log10(df[quant])
        df_cutout[quant] = np.log10(df_cutout[quant])
    a = plt.hist(df[quant], bins=40)
    a = plt.hist(df_cutout[quant], bins=40)
    plt.ylabel('Number of HII regions')
    plt.xlabel(quant_label)
    fig.text(0.6, 0.8, halo_dict[args.halo] + '; ' + args.output)
    plt.show(block=False)
    return fig

# -----------------------------------------------------------------------------
def getprojplot(args):
    ds, refine_box = load_sim(args, region='refine_box')
    prj = yt.ProjectionPlot(ds, 'x', ('deposit', 'stars_density'), center=ds.halo_center_kpc, data_source=refine_box, width=2 * args.galrad * kpc, weight_field=None)
    prj.set_unit(('deposit', 'stars_density'), 'Msun/pc**2')
    prj.set_zlim(('deposit', 'stars_density'), zmin=density_proj_max, zmax=density_proj_max)
    prj.set_cmap(('deposit', 'stars_density'), plt.cm.Greys_r)
    prj.annotate_text((0.6, 0.9), halo_dict[args.halo] + '; ' + args.output, coord_system='axis')
    prj.save(name=args.output_dir + 'figs/' + '%s_%s' % (args.output, 'stars'), suffix='png', mpl_kwargs={'dpi': 500})
    return prj

# -----------------------------------------------------------------------------
if __name__ == '__main__':
    start_time = time.time()
    quant, quant_label = 'Zin', 'Gas metallicity Z/Zsun'

    dummy_args = parse_args('8508', 'RD0042')
    if type(dummy_args) is tuple: dummy_args = dummy_args[0] # if the sim has already been loaded in, in order to compute the box center (via utils.pull_halo_center()), then no need to do it again
    if not dummy_args.keep: plt.close('all')
    
    if dummy_args.do_all_sims:
        list_of_sims = get_all_sims(dummy_args) # all snapshots of this particular halo
    else:
        if dummy_args.do_all_halos: halos = get_all_halos(dummy_args)
        else: halos = dummy_args.halo_arr
        list_of_sims = list(itertools.product(halos, dummy_args.output_arr))

    for index, this_sim in enumerate(list_of_sims):
        myprint('Doing halo ' + this_sim[0] + ' snapshot ' + this_sim[1] + ', which is ' + str(index + 1) + ' out of ' + str(len(list_of_sims)) + '..', dummy_args)
        if len(list_of_sims) == 1: args = dummy_args_tuple # since parse_args() has already been called and evaluated once, no need to repeat it
        else: args = parse_args(this_sim[0], this_sim[1])

        if type(args) is tuple:
            args, ds, refine_box = args  # if the sim has already been loaded in, in order to compute the box center (via utils.pull_halo_center()), then no need to do it again
            myprint('ds ' + str(ds) + ' for halo ' + str(this_sim[0]) + ' was already loaded at some point by utils; using that loaded ds henceforth', args)

        args.diag = args.diag_arr[0]
        args.Om = args.Om_arr[0]

        emlist = get_HII_list(args)
        emlist = shift_ref_frame(emlist, args)
        emlist = incline(emlist, args)
        emlist['r_cen'] = np.sqrt(emlist['pos_x_cen']**2 + emlist['pos_y_cen']**2 + emlist['pos_z_cen']**2) # kpc
        emlist = emlist.sort_values(by='r_cen')
        emlist_cutout = get_grid_coord(emlist, args)

        if args.plot_hist: fig = gethistplot(emlist, emlist_cutout, quant=quant, quant_label=quant_label)
        if args.saveplot: fig.savefig(args.output_dir + 'figs/' + args.output + '_' + quant + '_histogram.png')

    print('Complete in %s minutes' % ((time.time() - start_time) / 60))
    