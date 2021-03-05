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
    parser.set_defaults(save_dir="~")

    parser.add_argument('--use_halo_c_v', dest='use_halo_c_v', action='store_true',
                        help='just use the pwd?, default is no')
    parser.set_defaults(use_halo_c_v=False)

    args = parser.parse_args()
    return args




if __name__ == '__main__':
    args = parse_args()
    ds, refine_box = load_sim(args)

    '''
    make_projection_plots(ds = refine_box.ds, center = ds.halo_center_kpc,
                          refine_box = refine_box, x_width = 20.*kpc, 
                          fig_dir = '/Users/rsimons/Desktop', haloname = args.halo, name = args.run, 
                          fig_end = 'projection', do = ['temp'], axes = ['x', 'y', 'z'], is_central = True, add_arrow = False)
    '''
    sp = ds.sphere(ds.halo_center_kpc, (5, 'kpc'))
    sp.set_field_parameter('bulk_velocity', ds.halo_velocity_kms)
    results = {}
    results['L'] = []

    for i in ['x', 'y', 'z']:
        L_gas     = sp.quantities.total_quantity(('gas', 'angular_momentum_%s'%i))
        print (i, L_gas)
        results['L'].append(float(L_gas.to('cm**2*g/s').value))

    results['L'] = np.array(results['L'])

    L = results['L']/np.sqrt(np.sum(results['L']**2.))
    
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















