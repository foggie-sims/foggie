import yt
import foggie
from foggie.utils.get_run_loc_etc import get_run_loc_etc
from foggie.utils.get_refine_box import get_refine_box
from foggie.utils.get_halo_center import get_halo_center
from foggie.utils.get_proper_box_size import get_proper_box_size
from foggie.utils.foggie_load import *
from foggie.satellites.for_paper.central_projection_plots import make_projection_plots
from foggie.utils.consistency import *
from scipy.spatial import geometric_slerp
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
plt.ioff()
from numpy import *
import argparse


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='''identify satellites in FOGGIE simulations''')
    parser.add_argument('-system', '--system', metavar='system', type=str, action='store', \
                        help='Which system are you on? Default is Jase')
    parser.set_defaults(system="saje")

    parser.add_argument('-simname', '--simname', default=None, help='Simulation to be analyzed.')

    parser.add_argument('-simdir', '--simdir', default='/nobackupp2/mpeeples', help='simulation output directory')

    parser.add_argument('-haloname', '--haloname', default='halo_008508', help='halo_name')

    parser.add_argument('-df_mtype', '--df_mtype', default='particles', help='frame mtype')

    parser.add_argument('-df_rad', '--df_rad', default=3, type = int, help='frame radius')

    parser.add_argument('--run', metavar='run', type=str, action='store',
                        help='which run? default is natural')
    parser.set_defaults(run="nref11c_nref9f")

    parser.add_argument('--halo', metavar='halo', type=str, action='store',
                        help='which halo? default is 8508 (Tempest)')
    parser.set_defaults(halo="8508")

    parser.add_argument('--pwd', dest='pwd', action='store_true',
                        help='just use the pwd?, default is no')
    parser.set_defaults(pwd=False)

    parser.add_argument('--output', metavar='output', type=str, action='store',
                        help='which output? default is RD0020')
    parser.set_defaults(output="RD0027")

    parser.add_argument('--output_list', nargs='+', help='<Required> Set flag', default = [''], required=False)

    parser.add_argument('--save_dir', metavar='save_dir', type=str, action='store',
                        help='directory to save products')
    parser.set_defaults(save_dir="/nobackupp2/rcsimons/foggie")

    parser.add_argument('--use_halo_c_v', dest='use_halo_c_v', action='store_true',
                        help='just use the pwd?, default is no')
    parser.set_defaults(use_halo_c_v=False)

    parser.add_argument('--run_parallel', dest='run_parallel', action='store_true',
                        help='just use the pwd?, default is no')
    parser.set_defaults(run_paralell=False)
    
    parser.add_argument('-dd_min', '--dd_min', default=-99, type = int, help='frame radius')    
    parser.add_argument('-dd_max', '--dd_max', default=-99, type = int, help='frame radius')
    parser.add_argument('-cores', '--cores', default=1, type = int, help='frame radius')
    



    args = parser.parse_args()
    return args



def find_disk_Lhat(ds, args):
    sp = ds.sphere(ds.halo_center_kpc, (args.df_rad, 'kpc'))

    if args.df_mtype == 'particles':
        bulk_vel = sp.quantities.bulk_velocity(use_gas=False,use_particles=True,particle_type='all').to('km/s')
    elif args.df_mtype == 'all':
        bulk_vel = sp.quantities.bulk_velocity().to('km/s')

    sp = sp.cut_region(["(obj['temperature'] > {}) & (obj['temperature'] < {})".format(0, 1.e4)])

    sp.set_field_parameter('bulk_velocity', bulk_vel)
    results = {}
    results['L'] = []
    for i in ['x', 'y', 'z']:
        L_gas     = sp.quantities.total_quantity(('gas', 'angular_momentum_%s'%i))
        results['L'].append(float(L_gas.to('cm**2*g/s').value))
    results['L'] = np.array(results['L'])
    disk_Lhat = results['L']/np.sqrt(np.sum(results['L']**2.))
    return disk_Lhat, bulk_vel



def radial_profile(L_dic, ds, bulk_vel, disk_Lhat, low_temp, high_temp, mtype, is_gas, args, sp_rad = (250., 'kpc'), rprofbins = np.arange(0, 250, 0.5)*kpc):

    print (mtype)
    if is_gas:
        #gas w/temperature cut
        sp_use = ds.sphere(ds.halo_center_kpc, sp_rad)
        sp_use.set_field_parameter('bulk_velocity', bulk_vel)
        sp_use.set_field_parameter('normal', disk_Lhat)

        sp_use = sp_use.cut_region(["(obj['temperature'] > {}) & (obj['temperature'] < {})".format(low_temp, high_temp)])
        rname = ('index', 'radius')
        xname = ('gas', 'angular_momentum_x')
        yname = ('gas', 'angular_momentum_y')
        zname = ('gas', 'angular_momentum_z')
        mname = ('gas', 'cell_mass')
        vrname = ('gas', 'radial_velocity')
        ctname = ('index', 'cylindrical_theta')
        crname = ('index', 'cylindrical_radius')
        czname = ('index', 'cylindrical_z')
        metalname = ('gas', 'metallicity')

    else:
        #particles
        #sp_use = sp_sm
        sp_use = ds.sphere(ds.halo_center_kpc, sp_rad)
        sp_use.set_field_parameter('bulk_velocity', bulk_vel)
        sp_use.set_field_parameter('normal', disk_Lhat)
        rname = (mtype, 'particle_radius')
        xname = (mtype, 'particle_relative_angular_momentum_x')
        yname = (mtype, 'particle_relative_angular_momentum_y')
        zname = (mtype, 'particle_relative_angular_momentum_z')
        mname = (mtype, 'particle_mass')
        vrname = (mtype, 'particle_radial_velocity')
        ctname = (mtype, 'particle_position_cylindrical_theta')
        crname = (mtype, 'particle_position_cylindrical_radius')
        czname = (mtype, 'particle_position_cylindrical_z')

    Li       = yt.create_profile(sp_use, [rname], fields=[xname, yname, zname, mname], 
                                 weight_field=None, accumulation=False, override_bins = {rname:rprofbins})

    L_dic[mtype] = {}
    L_dic[mtype]['rprof'] = {}

    L_dic[mtype]['rprof']['r']    = Li.x.to('kpc')
    L_dic[mtype]['rprof']['Lx']   = Li[xname].to('g*cm**2/s')
    L_dic[mtype]['rprof']['Ly']   = Li[yname].to('g*cm**2/s')
    L_dic[mtype]['rprof']['Lz']   = Li[zname].to('g*cm**2/s')
    L_dic[mtype]['rprof']['mass'] = Li[mname].to('Msun')


    Lx       = sp_use[xname].to('g*cm**2/s')
    Ly       = sp_use[yname].to('g*cm**2/s')
    Lz       = sp_use[zname].to('g*cm**2/s')
    R        = sp_use[rname].to('kpc')
    vr       = sp_use[vrname].to('km/s')
    mass     = sp_use[mname].to('Msun')
    cr       = sp_use[crname].to('kpc')
    cz       = sp_use[czname].to('kpc')

    if is_gas:
        metallicity = sp_use[metalname].to('Zsun')
        metalbins = np.array([0, 0.02, 1, np.inf])


    Ltot = np.sqrt(Lx**2. + Ly**2. + Lz**2.)        
    thel = np.arctan2(Ly,Lx)*180./pi
    phil = np.arccos(Lz/Ltot)*180./pi

    xvar, xmn, xmx               = thel, -180, 180
    yvar, ymn, ymx               = phil,    0, 180
    rvar, rmn, rmd, rmx          = R, 0, 20, 250
    vrvar                        = vr
    crvar                        = cr
    czvar                        = cz

    vrbins   = np.array([-np.inf, -250, -100, 0, 100, 250, np.inf])

    nbins  = 200
    nrbins = 100

    #crbins   = np.linspace(crmn, crmx, ncrbins)
    crbins    = concatenate((np.linspace(0, 4, 4), np.linspace(4, 30, 4), np.linspace(30, 100, 4)))

    rbins    = concatenate((np.linspace(rmn, rmd, 20), np.linspace(rmd, rmx, 10)))
    czbins   = np.array([-30, -10, -2, 2, 10, 30])
    thelbins = np.linspace(xmn, xmx, nbins)
    philbins = np.linspace(ymn, ymx, nbins)

    for hst_type in ['r_dist', 'c_dist']:
        L_dic[mtype][hst_type] = {}

        if is_gas:
            if hst_type == 'r_dist':
                varlist     = (rvar, vrvar, metallicity, xvar, yvar)
                varnamelist = [rname, vrname, metalname, 'thel', 'phil']
                binlist     = (rbins, vrbins, metalbins, thelbins, philbins)
            elif hst_type == 'c_dist':
                varlist     = ( crvar, czvar, vrvar, metallicity, xvar, yvar)
                varnamelist = [crname, czname, vrname, metalname, 'thel', 'phil']
                binlist     = (crbins, czbins, vrbins, metalbins, thelbins, philbins)
        else:
            if hst_type == 'r_dist':
                varlist     = (rvar, vrvar,   xvar, yvar)
                varnamelist = [rname, vrname, 'thel', 'phil']
                binlist     = (rbins, vrbins,  thelbins, philbins)
    
            elif hst_type == 'c_dist':
                varlist     = ( crvar, czvar,   vrvar,  xvar, yvar)
                varnamelist = [crname, czname, vrname,'thel', 'phil']
                binlist     = (crbins, czbins, vrbins,thelbins, philbins)

        for (weights, weight_name) in [(Ltot, 'L_hst'), (mass, 'M_hst')]:
            hst = np.histogramdd(varlist, bins = binlist, weights = weights)
            L_dic[mtype][hst_type][weight_name]    = hst[0]

        L_dic[mtype][hst_type]['hst_bins']  = hst[1]
        L_dic[mtype][hst_type]['variables'] = varnamelist

    return L_dic




def run_process(args, output,  sp_rad = (250., 'kpc'), save = True): 
        args.output = output
        fname = '%s/angular_momentum/profiles/%s/Lprof_%s_%s'%(args.save_dir, args.halo, args.halo, args.output)
        print (fname)
        if os.path.exists(fname+'.npz'): return
        print ('continuing for :',fname)
        ds, _ = load_sim(args)
        disk_Lhat, bulk_vel = find_disk_Lhat(ds, args)

        mass_types = [(0.,    1.5e4, 'cold',   True),
                      (1.5e4, 1.e5,  'warm',   True),
                      (1.e5,  1.e6,  'warmhot',True),
                      (1.e6,  1.e10, 'hot',    True),
                      (-1.,  -1.,    'stars',   False),
                      (-1.,  -1.,    'young_stars',  False),
                      (-1.,  -1.,    'dm',       False)]


        L_dic = {}

        L_dic['props'] = {}
        L_dic['props']['bulk_velocity'] = bulk_vel
        L_dic['props']['center'] = ds.halo_center_kpc
        L_dic['props']['sphere_radius'] = sp_rad
        L_dic['props']['disk_Lhat'] = disk_Lhat
        L_dic['props']['frame_radius'] = (args.df_rad, 'kpc')
        L_dic['props']['frame_mtype']  = args.df_mtype
        for (low_temp, high_temp, mtype, is_gas) in mass_types:
            ds, _ = load_sim(args)
            L_dic = radial_profile(L_dic, ds, bulk_vel, disk_Lhat, low_temp, high_temp, mtype, is_gas, args, sp_rad = sp_rad)
            ds.index.clear_all_data()
        if save:
            np.savez_compressed(fname, a = L_dic)
            print ('saved L dictionary to %s.npz...'%fname)

        return


if __name__ == '__main__':
        print ('hi')
        args = parse_args()
        if len(args.output_list) == 0: run_list = np.arange(args.dd_min, args.dd_max, 1)
        else: run_list = np.array(args.output_list)
        #if args.run_parallel: Parallel(n_jobs = args.cores, backend = 'multiprocessing')(delayed(run_process)(args = args, output = output) for output in ['DD%.4i'%i for i in run_list)
        #else: run_process(args = args)
        print (run_list)










