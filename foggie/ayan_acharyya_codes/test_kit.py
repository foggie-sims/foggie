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
import make_mappings_grid as mmg

# -----------------------------------------------------------------------------
def getplot(emlist, quant, is_col_log=False):
    '''
    Function to make scatter plot of y vs z coordinate of particles (in kpc, wrt halo center) and color code by column of choice
    '''
    df = emlist.copy()
    plt.figure()
    if is_col_log: df[quant] = np.log10(df[quant])
    plt.scatter(df['pos_y_cen'], df['pos_z_cen'], c=df[quant])
    plt.xlim(-args.galrad, args.galrad)
    plt.ylim(-args.galrad, args.galrad)
    cb=plt.colorbar()
    cb.set_label(quant)
    plt.show(block=False)

# -----------------------------------------------------------------------------
def getproj(args):
    ds, refine_box = load_sim(args, region='refine_box')
    prj = yt.ProjectionPlot(ds, 'x', ('deposit', 'stars_density'), center=ds.halo_center_kpc, data_source=refine_box, width=2 * args.galrad * kpc, weight_field=None)
    prj.set_unit(('deposit', 'stars_density'), 'Msun/pc**2')
    prj.set_zlim(('deposit', 'stars_density'), zmin=density_proj_max, zmax=density_proj_max)
    prj.set_cmap(('deposit', 'stars_density'), plt.cm.Greys_r)
    prj.save(name=args.output_dir + 'figs/' + '%s_%s' % (args.output, 'stars'), suffix='png', mpl_kwargs={'dpi': 500})

# -----------------------------------------------------------------------------
if __name__ == '__main__':
    start_time = time.time()

    args = parse_args('8508', 'RD0042')
    if not args.keep: plt.close('all')

    args.diag = args.diag_arr[0]
    args.Om = args.Om_arr[0]

    emlist = get_HII_list(args)
    emlist = shift_ref_frame(emlist, args)
    emlist = incline(emlist, args)
    emlist_cutout = get_grid_coord(emlist, args)

    print('Complete in %s minutes' % ((time.time() - start_time) / 60))
    