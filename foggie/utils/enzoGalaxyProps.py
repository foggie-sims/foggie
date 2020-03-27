'''Filename: enzoGalaxyProps.py
Author: Raymond Simons (rsimons@stsci.edu)
Determine the center and calculate galaxy properties from an Enzo simulation.
Dependencies:'''
import sys
import os
import glob
import yt
import numpy as np
from numpy import *
import astropy
from astropy.cosmology import Planck13 as cosmo
import os, sys, argparse




def find_center(dd, ds, cen_pos = 10.e3, bin_width = 4.e3, del_pos = 20):
    '''
    Determine galaxy center, given the refined box
    returns: max_ndens_arr
    '''
    # Use the center of the refine box as a prior for the center of the galaxy

    small_refine_box = refine_box

    


    return center

def find_center_noprior(dd, ds, cen_pos = 10.e3, bin_width = 4.e3, del_pos = 20):
    '''
    crude determination of galaxy center
    returns: max_ndens_arr
    '''

    stars_pos_x = dd['stars', 'particle_position_x']
    stars_pos_y = dd['stars', 'particle_position_y']
    stars_pos_z = dd['stars', 'particle_position_z']

    star_pos = [stars_pos_x.value, stars_pos_y.value, stars_pos_z.value]

    min_pos = cen_pos - bin_width
    max_pos = cen_pos + bin_width
    bins = [arange(min_pos,max_pos,del_pos), arange(min_pos,max_pos,del_pos), arange(min_pos,max_pos,del_pos)]



    H, edges = histogramdd(star_pos, bins = bins)
    max_ndens_index = unravel_index(H.argmax(), H.shape)

    max_ndens_loc = array([(edges[0][max_ndens_index[0]] + edges[0][max_ndens_index[0]+1])/2.,
                           (edges[1][max_ndens_index[1]] + edges[1][max_ndens_index[1]+1])/2.,
                           (edges[2][max_ndens_index[2]] + edges[2][max_ndens_index[2]+1])/2.])

    max_ndens_arr = ds.arr([max_ndens_loc[0], max_ndens_loc[1], max_ndens_loc[2]], units)



    #end of First pass
    print('\tDone with coarse pass searching for center, moving to fine pass')


    bin_width = 40
    del_pos = 0.5

    min_pos_x = float(max_ndens_arr[0]) - bin_width
    max_pos_x = float(max_ndens_arr[0]) + bin_width

    min_pos_y = float(max_ndens_arr[1]) - bin_width
    max_pos_y = float(max_ndens_arr[1]) + bin_width

    min_pos_z = float(max_ndens_arr[2]) - bin_width
    max_pos_z = float(max_ndens_arr[2]) + bin_width


    bins = [arange(min_pos_x,max_pos_x,del_pos), arange(min_pos_y,max_pos_y,del_pos), arange(min_pos_z,max_pos_z,del_pos)]

    H, edges = histogramdd(star_pos, bins = bins)
    max_ndens_index = unravel_index(H.argmax(), H.shape)

    max_ndens_loc = array([(edges[0][max_ndens_index[0]] + edges[0][max_ndens_index[0]+1])/2.,
                           (edges[1][max_ndens_index[1]] + edges[1][max_ndens_index[1]+1])/2.,
                           (edges[2][max_ndens_index[2]] + edges[2][max_ndens_index[2]+1])/2.])

    max_ndens_arr = ds.arr([max_ndens_loc[0], max_ndens_loc[1], max_ndens_loc[2]], units)


    return max_ndens_arr

def find_rvirial(dd, ds, center, start_rad = 0, delta_rad_coarse = 20, delta_rad_fine = 1, rad_units = 'kpc'):
    from foggie.utils.get_rvir import find_rvir
    res = find_rvir(ds, halo_center = ds.halo_center_kpc, do_fig = False, figdir = '')
    return res['rvir'].to('kpc')

def find_hist_center(positions, masses):
    '''
    Find the center of a particle distribution by interactively refining
    a mass weighted histogram
    '''
    pos = np.array(positions)
    masses = np.array(masses)
    if len(pos) == 0:
        return None
    mass_current = masses
    old_center = np.array([0,0,0])
    refined_pos = pos.copy()
    refined_mas = mass_current.copy()
    refined_dist = 1e20
    nbins=3
    center = None

    dist = lambda x,y:np.sqrt(np.sum((x-y)**2.0))
    dist2 = lambda x,y:np.sqrt(np.sum((x-y)**2.0,axis=1))

    j=0
    while len(refined_pos)>1e1 or j==0:
        table,bins=np.histogramdd(refined_pos, bins=nbins, weights=refined_mas)
        bin_size = min((np.max(bins,axis=1)-np.min(bins,axis=1))/nbins)
        centeridx = np.where(table==table.max())
        le = np.array([bins[0][centeridx[0][0]],
                       bins[1][centeridx[1][0]],
                       bins[2][centeridx[2][0]]])
        re = np.array([bins[0][centeridx[0][0]+1],
                       bins[1][centeridx[1][0]+1],
                       bins[2][centeridx[2][0]+1]])
        center = 0.5*(le+re)
        refined_dist = dist(old_center,center)
        old_center = center.copy()
        idx = dist2(refined_pos,center)<bin_size
        refined_pos = refined_pos[idx]
        refined_mas = refined_mas[idx]
        j+=1

    return center

def find_shapes(center, pos, ds, nrad=10, rmax=None):
    '''
    Find the shape of the given particle distribution at nrad different
    radii, spanning from 0.1*rmax to rmax.
    rmax = max(r(pos)) if not given.
    '''

    print('Starting shape calculation')

    units = center.units
    center = center.value

    try:
        pos = np.array([pos[:,0] - center[0],
                        pos[:,1] - center[1],
                        pos[:,2] - center[2]]).transpose()
        pos = ds.arr(pos, units)
        pos = pos.in_units(units).value
        r = np.sqrt(pos[:,0]**2 + pos[:,1]**2 + pos[:,2]**2)
    except IndexError: # no stars found
        pos = np.array([])

    if len(pos) > 1:
        if not rmax: rmax = r.max()
        radii = np.linspace(0.1*rmax, rmax, nrad)
    else:
        radii = np.array([])

    c_to_a = np.empty(radii.size)
    b_to_a = np.empty(radii.size)
    axes = []

    for i,r in enumerate(radii):
        # get shapes
        try:
            axis_out = axis_ratios(pos, r, axes_out=True, fix_volume = False)
            c_to_a[i] = axis_out[0][0]
            b_to_a[i] = axis_out[0][1]
            axes.append(axis_out[1])
        except UnboundLocalError:
            print( 'Not enough particles to find shapes at r = %g in snapshot %s'%(r, ds.parameter_filename ))
            b_to_a[i] = c_to_a[i] = None
            axes.append([])

    return radii, c_to_a, b_to_a, axes

def L_crossing(x, y, z, vx, vy, vz, weight, center):
    x, y, z = x-center[0], y-center[1],z-center[2]
    cx, cy, cz = y*vz - z*vy, z*vx - x*vz, x*vy - y*vx
    lx, ly, lz = [np.sum(l * weight) for l in [cx, cy, cz]]
    L = np.array([lx, ly, lz])
    L /= np.sqrt(np.sum(L*L))
    return L

def find_galaxyprops(galaxy_props, ds, hc_sphere, max_ndens_arr):

        print( 'Determining stellar and gas mass...')
        # Get total stellar mass
        stars_mass = hc_sphere[('stars', 'particle_mass')].in_units('Msun')
        stars_total_mass = stars_mass.sum().value[()]
        galaxy_props['stars_total_mass'] = np.append(galaxy_props['stars_total_mass'], stars_total_mass)

        # Get total mass of gas
        gas_mass = hc_sphere[('gas', 'cell_mass')].in_units('Msun')
        gas_total_mass = gas_mass.sum().value[()]
        galaxy_props['gas_total_mass'] = np.append(galaxy_props['gas_total_mass'],
                                                   gas_total_mass)
        print( '\tlog Mgas/Msun = ', log10(gas_total_mass))
        print( '\tlog M*/Msun = ', log10(stars_total_mass))


        print( 'Determining location of max stellar density...')
        # Get max density of stars (value, location)
        stars_maxdens = hc_sphere.quantities.max_location(('deposit', 'stars_cic'))
        stars_maxdens_val = stars_maxdens[0].in_units('Msun/kpc**3').value[()]

        print( stars_maxdens)
        #difference bt yt-3.2.3 and yt-3.3dev: stars_maxdens has different # elements; this works for both
        stars_maxdens_loc = np.array([stars_maxdens[-3].in_units('kpc').value[()],
                                      stars_maxdens[-2].in_units('kpc').value[()],
                                      stars_maxdens[-1].in_units('kpc').value[()]])
        galaxy_props['stars_maxdens'].append((stars_maxdens_val, stars_maxdens_loc))
        print( '\t Max Stellar Density = ', stars_maxdens_loc)



        print( 'Determining location of max gas density...')
        # Get max density of gas
        gas_maxdens = hc_sphere.quantities.max_location(('gas', 'density'))
        gas_maxdens_val = gas_maxdens[0].in_units('Msun/kpc**3').value[()]
        gas_maxdens_loc = np.array([gas_maxdens[-3].in_units('kpc').value[()],
                                    gas_maxdens[-2].in_units('kpc').value[()],
                                    gas_maxdens[-1].in_units('kpc').value[()]])
        galaxy_props['gas_maxdens'].append((gas_maxdens_val, gas_maxdens_loc))
        print( '\t Max Gas Density = ', stars_maxdens_loc)



        print( 'Determining refined histogram center of stars...')
        #---Need to Check these--#
        # Get refined histogram center of stars
        stars_pos_x = hc_sphere[('stars', 'particle_position_x')].in_units('kpc')
        stars_pos_y = hc_sphere[('stars', 'particle_position_y')].in_units('kpc')
        stars_pos_z = hc_sphere[('stars', 'particle_position_z')].in_units('kpc')

        stars_pos = np.array([stars_pos_x, stars_pos_y, stars_pos_z]).transpose()
        stars_hist_center = find_hist_center(stars_pos, stars_mass)
        galaxy_props['stars_hist_center'].append(stars_hist_center)
        print( '\t Refined histogram center of stars = ', stars_hist_center)


        print( 'Computing stellar density profile...')
        # Get stellar density profile
        sc_sphere_r = 0.1
        ssphere_r = sc_sphere_r*hc_sphere.radius
        while ssphere_r < ds.index.get_smallest_dx():
                ssphere_r = 2.0*ssphere_r
        sc_sphere =  ds.sphere(max_ndens_arr, ssphere_r)

        try:
                p_plot = yt.ProfilePlot(sc_sphere, 'radius', 'stars_mass', n_bins=50, weight_field=None, accumulation=True)
                p_plot.set_unit('radius', 'kpc')
                p_plot.set_unit('stars_mass', 'Msun')
                p = p_plot.profiles[0]

                radii, smass = p.x.value, p['stars_mass'].value
                rhalf = radii[smass >= 0.5*smass.max()][0]
        except (IndexError, ValueError): # not enough stars found
                radii, smass = None, None
                rhalf = None
        galaxy_props['stars_rhalf'] = np.append(galaxy_props['stars_rhalf'], rhalf)
        galaxy_props['stars_mass_profile'].append((radii, smass))

        print( '\tStars half-light radius = ', rhalf)


        print( 'Determining center of mass within 15 kpc of the galaxy...')
        # Get center of mass of stars
        gal_sphere = ds.sphere(max_ndens_arr, (15, 'kpc'))
        stars_pos_x = gal_sphere[('stars', 'particle_position_x')].in_units('kpc')
        stars_pos_y = gal_sphere[('stars', 'particle_position_y')].in_units('kpc')
        stars_pos_z = gal_sphere[('stars', 'particle_position_z')].in_units('kpc')
        gal_stars_mass = gal_sphere[('stars', 'particle_mass')].in_units('Msun')
        gal_total_mass = gal_stars_mass.sum().value[()]


        stars_com = np.array([np.dot(stars_pos_x, gal_stars_mass)/gal_total_mass,
                              np.dot(stars_pos_y, gal_stars_mass)/gal_total_mass,
                              np.dot(stars_pos_z, gal_stars_mass)/gal_total_mass])
        galaxy_props['stars_com'].append(stars_com)
        print( '\tCenter of mass = ', stars_com)


        print( 'Setting stars center...')
        # Define center of stars
        center = 'maxndens'
        if center == 'max_dens':
                stars_center = stars_maxdens_loc
        elif center == 'com':
                stars_center = stars_com
        elif center == 'maxndens':
                stars_center = max_ndens_arr
        else:
                stars_center = stars_hist_center

        stars_center = ds.arr(stars_center, 'kpc')
        galaxy_props['stars_center'].append(stars_hist_center)
        print( '\tStars Center = ', stars_center)




        # Get angular momentum of stars
        try:
                x, y, z = [sc_sphere[('stars', 'particle_position_%s'%s)] for s in 'xyz']
                vx, vy, vz = [sc_sphere[('stars', 'particle_velocity_%s'%s)] for s in 'xyz']
                mass = sc_sphere[('stars', 'particle_mass')]
                try:
                        metals = sc_sphere[('stars', 'particle_metallicity1')]
                        stars_L = L_crossing(x, y, z, vx, vy, vz, mass*metals, sc_sphere.center)
                except:
                        stars_L = L_crossing(x, y, z, vx, vy, vz, mass, sc_sphere.center)

        except IndexError: # no stars found
                stars_L = [None, None, None]
                print("No stars exception")

        galaxy_props['stars_L'].append(stars_L)
        del(sc_sphere)


        # Get angular momentum of gas
        gas_center = ds.arr(gas_maxdens_loc, 'kpc')
        gc_sphere =  ds.sphere(gas_center, ssphere_r)
        x, y, z = [gc_sphere[('gas', '%s'%s)] for s in 'xyz']
        cell_volume = gc_sphere[('gas', 'cell_volume')]

        try:
                #for VELA runs
                vx, vy, vz = [gc_sphere[('gas', 'momentum_%s'%s)] for s in 'xyz'] # momentum density
                metals = gc_sphere[('gas', 'metal_ia_density')] + gc_sphere[('gas', 'metal_ii_density')]
                gas_L = L_crossing(x, y, z, vx, vy, vz, metals*cell_volume**2, gc_sphere.center)
        except:
                #for enzo runs
                density=gc_sphere[('gas', 'density')]
                vx, vy, vz = [gc_sphere[('gas', 'velocity_%s'%s)] for s in 'xyz']
                metals=gc_sphere[('gas', 'metal_density')]
                gas_L = L_crossing(x, y, z, density*vx, density*vy, density*vz, metals*cell_volume**2, gc_sphere.center)

        galaxy_props['gas_L'].append(gas_L)
        del(gc_sphere)


        return galaxy_props



def parse():
    '''
    Parse command line arguments
    '''
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='''\
                                Generate the cameras to use in Sunrise and make projection plots
                                of the data for some of these cameras. Then export the data within
                                the fov to a FITS file in a format that Sunrise understands.
                                ''')

    parser.add_argument('sim_name', nargs='?', default=None, help='Snapshot files to be analyzed.')

    parser.add_argument('snap_name', nargs='?', default=None, help='Snapshot files to be analyzed.')


    args = vars(parser.parse_args())
    return args


if __name__=="__main__":
    #snaps = np.sort(np.asarray(glob.glob("RD????/RD????")))  #ENZO format a list of snapshots in separate directories
    #snaps = np.sort(np.asarray(glob.glob("~/Dropbox/rcs_foggie/data/halo_008508/nref11n_nref10f_selfshield_z6/RD????/RD????")))

    #snaps = np.asarray(['/Users/rsimons/Dropbox/rcs_foggie/data/halo_008508/nref11n_nref10f_selfshield_z6/RD0018/RD0018'])
    form='ENZO'

    args = parse()
    simname = args['sim_name']

    #snaps = np.sort(np.asarray(glob.glob("/nobackupp2/mpeeples/halo_008508/nref11n_selfshield_z15/%s/%s"%(args['snap_name'], args['snap_name']))))
    snaps = np.sort(np.asarray(glob.glob("/nobackupp2/mpeeples/halo_008508/%s/%s/%s"%(args['sim_name'], args['snap_name'], args['snap_name']))))



    assert snaps.shape[0] > 0

    print("Calculating Galaxy Props for "+form+": ", snaps)

    abssnap = os.path.abspath(snaps[0])
    dirname = os.path.dirname(os.path.dirname(abssnap))
    #simname = os.path.basename(dirname) #assumes directory name for simulation name

    print( "Simulation name:  ", simname)
    '''
    particle_headers = []
    particle_data = []
    stars_data = []
    new_snapfiles = []
    for sn in snaps:
        aname=os.path.basename(sn)
        adir=os.path.abspath(os.path.dirname(sn))
        snap_dir = os.path.join(adir,simname+'_'+aname+'_sunrise')
        yt_fig_dir = snap_dir+'/yt_projections'
        print( "Sunrise directory: ", snap_dir)
        if not os.path.lexists(snap_dir):
            print ("Creating Sunrise directory:", snap_dir)
            os.mkdir(snap_dir)
        if not os.path.lexists(yt_fig_dir):
            print ("Creating YT figure directory:", yt_fig_dir)
            os.mkdir(yt_fig_dir)


        new_snapfiles.append(os.path.abspath(sn))
    new_snapfiles = np.asarray(new_snapfiles)
    '''

    new_snapfiles = np.asarray(snaps)


    galaxy_props = {}
    fields = ['scale', 'stars_total_mass', 'stars_com', 'stars_maxdens', 'stars_maxndens', 'stars_hist_center',
              'stars_rhalf', 'stars_mass_profile', 'stars_L','gas_total_mass', 'gas_maxdens', 'gas_L', 'rvir',
              'Mvir_dm', 'stars_center','snap_files']
    for field in fields:
        if field in ['scale', 'stars_total_mass', 'stars_rhalf', 'gas_total_mass' ]:
            galaxy_props[field] = np.array([])
        else:
            galaxy_props[field] = []



    def _stars(pfilter, data):
        return data[(pfilter.filtered_type, "particle_type")] == 2

    #this gets dark matter particles in zoom region only
    def _darkmatter(pfilter, data):
        return data[(pfilter.filtered_type, "particle_type")] == 4

    yt.add_particle_filter("stars",function=_stars, filtered_type='all',requires=["particle_type"])
    yt.add_particle_filter("darkmatter",function=_darkmatter, filtered_type='all',requires=["particle_type"])

    ts = yt.DatasetSeries(new_snapfiles)


    for ds,snap_dir in zip(reversed(ts),np.flipud(new_snapfiles)):
        print( "Getting galaxy props: ",  snap_dir)

        ds.add_particle_filter('stars')
        ds.add_particle_filter('darkmatter')

        dd = ds.all_data()
        ds.domain_right_edge = ds.arr(ds.domain_right_edge,'code_length')
        ds.domain_left_edge  = ds.arr(ds.domain_left_edge,'code_length')

        try:
            print('Loading data...')
            stars_pos_x = dd['stars', 'particle_position_x'].in_units('kpc')
            print('Loaded.')
            assert stars_pos_x.shape[0] > 5
        except AttributeError:
            print("No star particles found, skipping: ", snap_dir)
            continue


        scale = round(1.0/(ds.current_redshift+1.0),3)
        galaxy_props['scale'] = np.append(galaxy_props['scale'], scale)

        galaxy_props['snap_files'] = np.append(galaxy_props['snap_files'],snap_dir)


        print( 'Determining center...')
        max_ndens_arr = find_center(dd, ds, cen_pos = ds.domain_center.in_units('kpc')[0].value[()])
        print( '\tCenter = ', max_ndens_arr)
        sys.stdout.flush()

        #Generate Sphere Selection
        print( 'Determining virial radius...')
        rvir = find_rvirial(dd, ds, max_ndens_arr)
        print( '\tRvir = ', rvir)
        sys.stdout.flush()

        hc_sphere = ds.sphere(max_ndens_arr, rvir)


        galaxy_props['stars_maxndens'].append(max_ndens_arr.value)
        galaxy_props['rvir'] = np.append(galaxy_props['rvir'], rvir.value[()])
        galaxy_props['Mvir_dm'] = np.append(galaxy_props['Mvir_dm'], hc_sphere[('darkmatter', 'particle_mass')].in_units('Msun').sum().value[()])


        #Find Galaxy Properties
        galaxy_props = find_galaxyprops(galaxy_props, ds, hc_sphere, max_ndens_arr)


        del (hc_sphere)
        sys.stdout.flush()




    # Save galaxy props file
    galprops_outdir = '/nobackupp2/rcsimons/foggie_momentum/galprops'
    galaxy_props_file = galprops_outdir + '/' + simname + '_' + args['snap_name'] + '_galprops.npy'


    print( '\nSuccessfully computed galaxy properties')
    print( 'Saving galaxy properties to ', galaxy_props_file)

    np.save(galaxy_props_file, galaxy_props)
