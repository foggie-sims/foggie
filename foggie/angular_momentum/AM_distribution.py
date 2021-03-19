import yt
import foggie
from foggie.utils.get_run_loc_etc import get_run_loc_etc
from foggie.utils.get_refine_box import get_refine_box
from foggie.utils.get_halo_center import get_halo_center
from foggie.utils.get_proper_box_size import get_proper_box_size
from foggie.utils.foggie_load import *
from foggie.satellites.for_paper.central_projection_plots import make_projection_plots
import argparse
from foggie.utils.consistency import *
from numpy import *
from scipy.spatial import geometric_slerp
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
plt.ioff()

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='''identify satellites in FOGGIE simulations''')
    parser.add_argument('-system', '--system', metavar='system', type=str, action='store', \
                        help='Which system are you on? Default is Jase')
    parser.set_defaults(system="saje")

    parser.add_argument('-simname', '--simname', default=None, help='Simulation to be analyzed.')

    parser.add_argument('-simdir', '--simdir', default='/nobackupp2/mpeeples', help='simulation output directory')

    parser.add_argument('-haloname', '--haloname', default='halo_008508', help='halo_name')

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


    parser.add_argument('--save_dir', metavar='save_dir', type=str, action='store',
                        help='directory to save products')
    parser.set_defaults(save_dir="/nobackupp2/rcsimons/foggie")

    parser.add_argument('--use_halo_c_v', dest='use_halo_c_v', action='store_true',
                        help='just use the pwd?, default is no')
    parser.set_defaults(use_halo_c_v=False)

    args = parser.parse_args()
    return args



def find_disk_Lhat(ds):

    sp = ds.sphere(ds.halo_center_kpc, (3, 'kpc'))

    bulk_vel = sp.quantities.bulk_velocity().to('km/s')
    sp.set_field_parameter('bulk_velocity', bulk_vel)

    results = {}
    results['L'] = []

    for i in ['x', 'y', 'z']:
        L_gas     = sp.quantities.total_quantity(('gas', 'angular_momentum_%s'%i))

        results['L'].append(float(L_gas.to('cm**2*g/s').value))

    results['L'] = np.array(results['L'])

    disk_Lhat = results['L']/np.sqrt(np.sum(results['L']**2.))


    return disk_Lhat, bulk_vel



def radial_profile(ds, bulk_vel, mass_types, args, save = True, sp_rad = (100., 'kpc'), rbins = np.arange(0, 100, 0.5)*kpc):
    sp_sm = ds.sphere(ds.halo_center_kpc, sp_rad)
    sp_sm.set_field_parameter('bulk_velocity', bulk_vel)

    L_dic = {}


    for (low_temp, high_temp, mtype, clr) in mass_types:
        print (mtype)
        if low_temp < 0.:
            #particles
            sp_use = sp_sm
            rname = (mtype, 'particle_radius')
            xname = (mtype, 'particle_angular_momentum_x')
            yname = (mtype, 'particle_angular_momentum_y')
            zname = (mtype, 'particle_angular_momentum_z')
            mname = (mtype, 'particle_mass')
        else:
            #gas w/temperature cut
            sp_use = sp_sm.cut_region(["(obj['temperature'] > {}) & (obj['temperature'] < {})".format(low_temp, high_temp)])

            rname = ('index', 'radius')
            xname = ('gas', 'angular_momentum_x')
            yname = ('gas', 'angular_momentum_y')
            zname = ('gas', 'angular_momentum_z')
            mname = ('gas', 'cell_mass')

        Li       = yt.create_profile(sp_use, [rname], fields=[xname, yname, zname, mname], 
                                     weight_field=None, accumulation=False, override_bins = {rname:rbins})

        L_dic[mtype] = {}
        L_dic[mtype]['rprof'] = {}
        L_dic[mtype]['adist'] = {}

        L_dic[mtype]['rprof']['r']    = Li.x.to('kpc')
        L_dic[mtype]['rprof']['Lx']   = Li[xname].to('g*cm**2/s')
        L_dic[mtype]['rprof']['Ly']   = Li[yname].to('g*cm**2/s')
        L_dic[mtype]['rprof']['Lz']   = Li[zname].to('g*cm**2/s')
        L_dic[mtype]['rprof']['mass'] = Li[mname].to('Msun')

        Lx    = sp_use[xname].to('g*cm**2/s')
        Ly    = sp_use[yname].to('g*cm**2/s')
        Lz    = sp_use[zname].to('g*cm**2/s')
        R     = sp_use[rname].to('kpc')
        mass     = sp_use[mname].to('Msun')

        Ltot = np.sqrt(Lx**2. + Ly**2. + Lz**2.)
        
        L_dic[mtype]['adist']['r']       = R
        L_dic[mtype]['adist']['phil']    = np.arccos(Lz/Ltot)*180./pi
        L_dic[mtype]['adist']['thel']    = np.arctan2(Ly,Lx)*180./pi
        L_dic[mtype]['adist']['ltot']    = Ltot
        L_dic[mtype]['adist']['mass']    = mass

    if save:
        fname = '%s/angular_momentum/profiles/%s/Lprof_%s_%s.npy'%(args.save_dir, args.halo, args.halo, args.output)
        np.save(fname, L_dic)
        print ('saved L dictionary to %s...'%fname)
    return L_dic


if __name__ == '__main__':
    args = parse_args()
    ds, refine_box = load_sim(args)



    mass_types = [(0., 1.5e4, 'cold', 'darkblue'),
                 (1.5e4, 1.e5, 'warm', 'blue'),
                 (1.e5, 1.e6, 'warmhot', 'red'),
                 (1.e6, 1.e10, 'hot', 'darkred'), 
                 (-1., -1., 'stars', 'goldenrod'),
                 (-1., -1., 'dm', 'black')]

    disk_Lhat, bulk_vel = find_disk_Lhat(ds)
    L_dic               = radial_profile(ds, bulk_vel, mass_types, args)













    '''
    make_projection_plots(ds = refine_box.ds, center = ds.halo_center_kpc,
                          refine_box = refine_box, x_width = 20.*kpc, 
                          fig_dir = '/Users/rsimons/Desktop', haloname = args.halo, name = args.run, 
                          fig_end = 'projection', do = ['temp'], axes = ['x', 'y', 'z'], is_central = True, add_arrow = False)

    
    sp_sm = ds.sphere(ds.halo_center_kpc, (15., 'kpc'))

    Lx = sp_sm[('gas', 'angular_momentum_x')]
    Ly = sp_sm[('gas', 'angular_momentum_y')]
    Lz = sp_sm[('gas', 'angular_momentum_z')]
    M_gas = sp_sm[('gas', 'cell_mass')]
    R_gas = sp_sm[('gas', 'radius_corrected')]
    T_gas = sp_sm[('gas', 'temperature')]
    L_L = Lx * L[0] + Ly * L[1] + Lz * L[2]


    fig, ax = plt.subplots(1,1, figsize = (10, 10))


    gd = abs(L_L) < 2.e68
    #ax.hist2d(R_gas[gd], L_L[gd], bins = 200,norm = matplotlib.colors.LogNorm(), weights = M_gas[gd])
    ax.hist2d(R_gas, np.log10(T_gas), bins = 200, norm = matplotlib.colors.LogNorm(),weights = L_L)

    #ax.set_ylim(-2.e67, 2.e67)

    #ax.set_yscale('symlog', linthreshy=1.e66)
    fig.savefig('/Users/rsimons/Desktop/test.png', dpi = 300)


    if False:
        field = ('gas', 'density')
        cmap = density_color_map

        unit = 'Msun/pc**2'
        cmap.set_bad('k')
        unit = 'Msun/pc**2'
        zmin = density_proj_min 
        zmax = density_proj_max
        weight_field = None

        np.random.seed(1)
        E = np.random.randn(3)
        E -= E.dot(L) * L / np.linalg.norm(L)**2
        E/=np.sqrt(np.sum(E**2.))


        slerp_vec_all = {}
        north_vec_all = {}

        march = [L, E, -L, -E, L, E]


        Nslerps = 40
        for ii in np.arange(4):
            x = march[ii]
            y = march[ii+1]
            slerp_vec_all[ii] = geometric_slerp(x, y, np.linspace(1./Nslerps, 1., Nslerps))
            north_vec_all[ii] = [y for i in np.arange(Nslerps-1)]
            north_vec_all[ii].append(march[ii+2])

        slerp_vecs = np.concatenate((slerp_vec_all[0], slerp_vec_all[1], slerp_vec_all[2], slerp_vec_all[3]))        
        north_vecs = np.concatenate((north_vec_all[0], north_vec_all[1], north_vec_all[2], north_vec_all[3]))        



        for d, (vec, north) in enumerate(zip(slerp_vecs, north_vecs)):
            prj = yt.OffAxisProjectionPlot(ds, vec, field, center = ds.halo_center_kpc, width=(20, 'kpc'),
                                           north_vector = north, data_source = sp_sm)
            prj.set_unit(field, unit)
            prj.set_zlim(field, zmin = zmin, zmax =  zmax)
            prj.set_cmap(field, cmap)
            prj.save('/user/rsimons/foggie/angular_momentum/movies/new_%i.png'%d)


    '''













