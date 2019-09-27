import numpy as np
from numpy import *
import os, sys, argparse
import glob
import astropy
from astropy.io import fits
import astropy
from astropy.cosmology import Planck15 as cosmo
from joblib import Parallel, delayed
import scipy
from scipy.interpolate import UnivariateSpline
import yt


#This file will be used to store the profile of the momentum

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

    parser.add_argument('-run_parallel', '--run_parallel', default=False, help='Run parallel')

    parser.add_argument('-simname', '--simname', default='nref11n_nref10f', help='Simulation to be analyzed.')

    parser.add_argument('-snapname', '--snapname', default=None, help='Snapshot files to be analyzed.')

    parser.add_argument('-haloname', '--haloname', default='halo_008508', help='halo_name')
    parser.add_argument('-on_system', '--on_system', default='local', help='System being used (pfe or local)')
    parser.add_argument('-ddmin', '--ddmin', default=906, help='halo_name')
    parser.add_argument('-ddmax', '--ddmax', default=907, help='halo_name')
    parser.add_argument('-n_jobs', '--n_jobs', default=3, help='number of jobs')



    args = vars(parser.parse_args())
    return args


def recenter_2(amom):
    print 'Recentering...'
    #amom.cen_x, amom.cen_y, amom.cen_z = yt.YTArray(galprops['stars_center'][0], 'kpc')
    #amom.cen_x, amom.cen_y, amom.cen_z = yt.YTArray([ds.quan(0.4922914505, 'code_length'), ds.quan(0.482047080994, 'code_length'), ds.quan(0.504963874817, 'code_length')]).to('kpc')
    #amom.cen_x, 




    amom.stars_x   = amom.stars_x_box  - amom.cen_x
    amom.stars_y   = amom.stars_y_box  - amom.cen_y
    amom.stars_z   = amom.stars_z_box  - amom.cen_z
    amom.stars_pos = array([amom.stars_x, amom.stars_y, amom.stars_z])
    amom.stars_pos_mag = sqrt(amom.stars_x**2.  + amom.stars_y**2.  + amom.stars_z**2.)


    amom.dark_x   = amom.dark_x_box  - amom.cen_x
    amom.dark_y   = amom.dark_y_box  - amom.cen_y
    amom.dark_z   = amom.dark_z_box  - amom.cen_z
    amom.dark_pos = array([amom.dark_x, amom.dark_y, amom.dark_z])
    amom.dark_pos_mag = sqrt(amom.dark_x**2.  + amom.dark_y**2.  + amom.dark_z**2.)


    #Determine the mass-weighted velocity of the stars in the inner 1 kpc

    stars_inner_1kpc = where(amom.stars_pos_mag < 1)[0]
    print len(stars_inner_1kpc)
    amom.cen_vx = np.average(amom.stars_vx_box[stars_inner_1kpc], weights = amom.star_mass[stars_inner_1kpc])
    amom.cen_vy = np.average(amom.stars_vy_box[stars_inner_1kpc], weights = amom.star_mass[stars_inner_1kpc])
    amom.cen_vz = np.average(amom.stars_vz_box[stars_inner_1kpc], weights = amom.star_mass[stars_inner_1kpc])


    amom.stars_vx  = amom.stars_vx_box - amom.cen_vx
    amom.stars_vy  = amom.stars_vy_box - amom.cen_vy
    amom.stars_vz  = amom.stars_vz_box - amom.cen_vz
    amom.stars_vel = array([amom.stars_vx, amom.stars_vy, amom.stars_vz])
    amom.stars_vel_mag = sqrt(amom.stars_vx**2. + amom.stars_vy**2. + amom.stars_vz**2.)



    amom.dark_vx  = amom.dark_vx_box - amom.cen_vx
    amom.dark_vy  = amom.dark_vy_box - amom.cen_vy
    amom.dark_vz  = amom.dark_vz_box - amom.cen_vz
    amom.dark_vel = array([amom.dark_vx, amom.dark_vy, amom.dark_vz])
    amom.dark_vel_mag = sqrt(amom.dark_vx**2. + amom.dark_vy**2. + amom.dark_vz**2.)

    return amom

class momentum_obj():
    def __init__(self, simname, aname, snapfile, fits_name):
        self.ds = yt.load(snapfile)
        self.simname = simname
        self.aname = aname
        self.snapfile = snapfile
        self.fits_name = fits_name

    def load(self):
        dd = self.ds.all_data()

        def _stars(pfilter, data):
            return data[(pfilter.filtered_type, "particle_type")] == 2

        # these are only the must refine dark matter particles
        def _darkmatter(pfilter, data):
            return data[(pfilter.filtered_type, "particle_type")] == 4

        yt.add_particle_filter("stars",function=_stars, filtered_type='all',requires=["particle_type"])
        yt.add_particle_filter("darkmatter",function=_darkmatter, filtered_type='all',requires=["particle_type"])
        self.ds.add_particle_filter('stars')
        self.ds.add_particle_filter('darkmatter')



        try:
            print 'Loading stars particle indices...'
            self.stars_id = dd['stars', 'particle_index']
            assert self.stars_id.shape > 5
        except AttributeError,AssertionError:
            print "No star particles found, skipping: ", self.ds._file_amr
            return



        #self.stars_metallicity1 = dd['stars', 'particle_metallicity1']
        #self.stars_metallicity2 = dd['stars', 'particle_metallicity2']
        
        print 'Loading star velocities...'
        self.stars_vx_box = dd['stars', 'particle_velocity_x'].in_units('km/s')
        self.stars_vy_box = dd['stars', 'particle_velocity_y'].in_units('km/s')
        self.stars_vz_box = dd['stars', 'particle_velocity_z'].in_units('km/s')

        print 'Loading star positions...'
        self.stars_x_box = dd['stars', 'particle_position_x'].in_units('kpc')
        self.stars_y_box = dd['stars', 'particle_position_y'].in_units('kpc')
        self.stars_z_box = dd['stars', 'particle_position_z'].in_units('kpc')

        print 'Loading star mass...'
        self.star_mass = dd['stars', 'particle_mass'].in_units('Msun')

        print 'Loading star age...'
        self.star_creation_time = dd['stars', 'creation_time'].in_units('yr')
        self.star_age = self.ds.arr(cosmo.age(self.ds.current_redshift).value, 'Gyr').in_units('yr') - self.star_creation_time




        print 'Loading dark matter particle indices...'
        self.dark_id = dd['darkmatter', 'particle_index']
        
        print 'Loading dark matter velocities...'
        self.dark_vx_box = dd['darkmatter', 'particle_velocity_x'].in_units('km/s')
        self.dark_vy_box = dd['darkmatter', 'particle_velocity_y'].in_units('km/s')
        self.dark_vz_box = dd['darkmatter', 'particle_velocity_z'].in_units('km/s')

        print 'Loading dark matter positions...'
        self.dark_x_box = dd['darkmatter', 'particle_position_x'].in_units('kpc')
        self.dark_y_box = dd['darkmatter', 'particle_position_y'].in_units('kpc')
        self.dark_z_box = dd['darkmatter', 'particle_position_z'].in_units('kpc')

        print 'Loading dark matter mass...'
        self.dark_mass = dd['darkmatter', 'particle_mass'].in_units('Msun')

        print 'Loading dark matter age...'
        self.dark_creation_time = dd['darkmatter', 'creation_time'].in_units('yr')
        self.dark_age = self.ds.arr(cosmo.age(self.ds.current_redshift).value, 'Gyr').in_units('yr') - self.dark_creation_time


        if False:
            print 'Loading gas velocity...'
            self.gas_vx = dd['gas', 'velocity_x'].in_units('km/s')
            self.gas_vy = dd['gas', 'velocity_y'].in_units('km/s')
            self.gas_vz = dd['gas', 'velocity_z'].in_units('km/s')
            
            print 'Loading gas cell position...'
            self.gas_x = dd['gas', 'x'].in_units('kpc')
            self.gas_y = dd['gas', 'y'].in_units('kpc')
            self.gas_z = dd['gas', 'z'].in_units('kpc')

            print 'Loading gas temperature...'
            self.gas_temp = dd['gas', 'temperature']

            print 'Loading gas cell mass...'
            self.gas_mass = dd['gas', 'cell_mass']







        print 'Finished loading...'
        return 1


    def recenter(self, galprops):
        print 'Recentering...'
        self.cen_x, self.cen_y, self.cen_z = yt.YTArray(galprops['stars_center'][0], 'kpc')
        self.stars_x   = self.stars_x_box  - self.cen_x
        self.stars_y   = self.stars_y_box  - self.cen_y
        self.stars_z   = self.stars_z_box  - self.cen_z
        self.stars_pos = array([self.stars_x, self.stars_y, self.stars_z])
        self.stars_pos_mag = sqrt(self.stars_x**2.  + self.stars_y**2.  + self.stars_z**2.)


        self.dark_x   = self.dark_x_box  - self.cen_x
        self.dark_y   = self.dark_y_box  - self.cen_y
        self.dark_z   = self.dark_z_box  - self.cen_z
        self.dark_pos = array([self.dark_x, self.dark_y, self.dark_z])
        self.dark_pos_mag = sqrt(self.dark_x**2.  + self.dark_y**2.  + self.dark_z**2.)


        #Determine the mass-weighted velocity of the stars in the inner 1 kpc

        stars_inner_1kpc = where(self.stars_pos_mag < 1)
        self.cen_vx = np.average(self.stars_vx_box[stars_inner_1kpc], weights = self.star_mass[stars_inner_1kpc])
        self.cen_vy = np.average(self.stars_vy_box[stars_inner_1kpc], weights = self.star_mass[stars_inner_1kpc])
        self.cen_vz = np.average(self.stars_vz_box[stars_inner_1kpc], weights = self.star_mass[stars_inner_1kpc])


        self.stars_vx  = self.stars_vx_box - self.cen_vx
        self.stars_vy  = self.stars_vy_box - self.cen_vy
        self.stars_vz  = self.stars_vz_box - self.cen_vz
        self.stars_vel = array([self.stars_vx, self.stars_vy, self.stars_vz])
        self.stars_vel_mag = sqrt(self.stars_vx**2. + self.stars_vy**2. + self.stars_vz**2.)



        self.dark_vx  = self.dark_vx_box - self.cen_vx
        self.dark_vy  = self.dark_vy_box - self.cen_vy
        self.dark_vz  = self.dark_vz_box - self.cen_vz
        self.dark_vel = array([self.dark_vx, self.dark_vy, self.dark_vz])
        self.dark_vel_mag = sqrt(self.dark_vx**2. + self.dark_vy**2. + self.dark_vz**2.)








    def calc_angular_momentum(self, ptype = 'stars'):
        print 'Calculating angular momentum for type: %s...'%ptype

        #Calculate momentum for stars
        if ptype == 'stars':
            self.stars_jx = self.stars_vz * self.stars_y - self.stars_z * self.stars_vy
            self.stars_jy = self.stars_vx * self.stars_z - self.stars_x * self.stars_vz
            self.stars_jz = self.stars_vy * self.stars_x - self.stars_y * self.stars_vx
            self.stars_j  = array([self.stars_jx, self.stars_jy, self.stars_jz])
            self.stars_j_mag  = sqrt(self.stars_jx**2. + self.stars_jy**2. + self.stars_jz**2.)

        if ptype  =='darkmatter':
            self.dark_jx = self.dark_vz * self.dark_y - self.dark_z * self.dark_vy
            self.dark_jy = self.dark_vx * self.dark_z - self.dark_x * self.dark_vz
            self.dark_jz = self.dark_vy * self.dark_x - self.dark_y * self.dark_vx
            self.dark_j  = array([self.dark_jx, self.dark_jy, self.dark_jz])
            self.dark_j_mag  = sqrt(self.dark_jx**2. + self.dark_jy**2. + self.dark_jz**2.)


        if ptype == 'gas':
            #Calculate angular momentum for gas
            self.gas_jx = self.gas_vz * self.gas_y - self.gas_z * self.gas_vy
            self.gas_jy = self.gas_vx * self.gas_z - self.gas_x * self.gas_vz
            self.gas_jz = self.gas_vy * self.gas_x - self.gas_y * self.gas_vx
            self.gas_j  = array([self.gas_jx, self.gas_jy, self.gas_jz])
            self.gas_j_mag  = sqrt(self.gas_jx**2. + self.gas_jy**2. + self.gas_jz**2.)

        return self

    def measure_potential(self, r_min = 0.1,  r_step1 = 0.2, r_cen1 = 5, r_step2 = 1,  r_cen2 = 15, r_step3 = 5, r_max = 200.):

        print 'Measuring the potential...'
        center = self.ds.arr([self.cen_x, self.cen_y, self.cen_z], 'kpc')

        rad_steps = concatenate((arange(r_min,  r_cen1, r_step1), 
                                 arange(r_cen1, r_cen2, r_step2),
                                 arange(r_cen2, r_max,  r_step3)))
        self.mass_profile = zeros((2,len(rad_steps)))

        for i in arange(0,len(rad_steps)):
            print i, rad_steps[i], len(rad_steps)
            try:
                gc_sphere =  self.ds.sphere(center, self.ds.arr(rad_steps[i],'kpc'))
                baryon_mass, particle_mass = gc_sphere.quantities.total_quantity(["cell_mass", "particle_mass"])
                self.mass_profile[0,i] = rad_steps[i]
                self.mass_profile[1,i] = baryon_mass + particle_mass
            except:
                print '\tsomething broken in measure_potential..'
                self.mass_profile[0,i] = 0.
                self.mass_profile[1,i] = 0.


        self.spl = UnivariateSpline(self.mass_profile[0,:], self.mass_profile[1,:])


    def measure_circularity(self, use_self = False):
        print 'Calculating circularity...'

        G = yt.units.G.to('kpc**3*Msun**-1*s**-2')
        #internal_mass_gas   = self.ds.arr(self.spl(self.gas_pos_mag),'g').in_units('Msun')


        #self.vcirc_gas      = self.ds.arr(sqrt(G*internal_mass_gas/(self.gas_pos_mag)),'kpc/s').in_units('km/s')
        #self.jcirc_gas      = self.vcirc_gas * self.gas_pos_mag

        internal_mass_stars = self.ds.arr(self.spl(self.stars_pos_mag),'g').in_units('Msun')
        self.vcirc_stars    = self.ds.arr(sqrt(G*internal_mass_stars/(self.stars_pos_mag)),'kpc/s').in_units('km/s')
        self.jcirc_stars    = self.vcirc_stars * self.stars_pos_mag

        internal_mass_dark = self.ds.arr(self.spl(self.dark_pos_mag),'g').in_units('Msun')
        self.vcirc_dark    = self.ds.arr(sqrt(G*internal_mass_dark/(self.dark_pos_mag)),'kpc/s').in_units('km/s')
        self.jcirc_dark    = self.vcirc_dark * self.dark_pos_mag


        self.L_mag          = sqrt(self.L_disk[0]**2.+self.L_disk[1]**2.+self.L_disk[2]**2.)
        self.L_mag_fixed    = sqrt(self.L_disk_fixed[0]**2.+self.L_disk_fixed[1]**2.+self.L_disk_fixed[2]**2.)
 

        costheta_stars            = np.dot(self.L_disk, self.stars_j)/(self.stars_j_mag*self.L_mag)
        costheta_stars_fixed      = np.dot(self.L_disk_fixed, self.stars_j)/(self.stars_j_mag*self.L_mag_fixed)
 
        self.jz_stars       = costheta_stars*self.stars_j_mag
        self.jz_stars_fixed       = costheta_stars_fixed*self.stars_j_mag
 
        self.epsilon_stars  = self.jz_stars/self.jcirc_stars
        self.epsilon_stars_fixed  = self.jz_stars_fixed/self.jcirc_stars





        costheta_dark            = np.dot(self.L_disk, self.dark_j)/(self.dark_j_mag*self.L_mag)
        costheta_dark_fixed      = np.dot(self.L_disk_fixed, self.dark_j)/(self.dark_j_mag*self.L_mag_fixed)
 
        self.jz_dark       = costheta_dark*self.dark_j_mag
        self.jz_dark_fixed       = costheta_dark_fixed*self.dark_j_mag
 
        self.epsilon_dark  = self.jz_dark/self.jcirc_dark
        self.epsilon_dark_fixed  = self.jz_dark_fixed/self.jcirc_dark




        #costheta_gas        = np.dot(self.L_disk, self.gas_j)/(self.gas_j_mag*self.L_mag)
        #self.jz_gas         = costheta_gas*self.gas_j_mag
        #self.epsilon_gas    = self.jz_gas/self.jcirc_gas
        #costheta_gas   = np.dot(self.L_disk, self.gas_pos)/(self.gas_pos_mag*self.L_mag)
        #self.zz_gas    = self.ds.arr(costheta_gas * self.gas_pos_mag, 'kpc')
        #self.rr_gas    = sqrt(self.gas_pos_mag**2. - self.zz_gas**2.)

        #costheta_stars = np.dot(self.L_disk, self.stars_pos)/(self.stars_pos_mag*self.L_mag)
        #self.zz_stars  = self.ds.arr(costheta_stars * self.stars_pos_mag, 'kpc')
        #self.rr_stars  = sqrt(self.stars_pos_mag**2. - self.zz_stars**2.)



    def gas_momentum_heatmap(self):
        print 'Measuring gas momentum profiles...'

        cold_gas_zz = where((abs(self.rr_gas) < 30) & (self.gas_temp < 1.e4))
       
        eps_min = -2.5
        eps_max = 2.5
        min_z   = -10
        max_z   = 10
        min_r   = 0
        max_r   = 30
        min_rad = 0
        max_rad = 100.
        bins_n  = 200


        '''
        cold_gas_zz = where((abs(self.rr_gas) < max_r) & (self.gas_temp < 1.e4))
        weights = self.gas_mass[cold_gas_zz]

        self.cg_zz_heatmap, self.cg_zz_xedges, self.cg_zz_yedges = np.histogram2d(self.epsilon_gas[cold_gas_zz], self.zz_gas[cold_gas_zz], 
                                                                   bins=[linspace(eps_min,eps_max,bins_n), linspace(min_z,max_z,bins_n)], 
                                                                   weights = weights)


        cold_gas_rr = where((abs(self.zz_gas) < (max_z-min_z)/2.) & (self.gas_temp < 1.e4))
        weights = self.gas_mass[cold_gas_rr]
        print min_r, max_r
        self.cg_rr_heatmap, self.cg_rr_xedges, self.cg_rr_yedges = np.histogram2d(self.epsilon_gas[cold_gas_rr], self.rr_gas[cold_gas_rr], 
                                                                   bins=[linspace(eps_min,eps_max,bins_n), linspace(min_r,max_r,bins_n)], 
                                                                   weights = weights)
        print self.cg_rr_xedges.min()


        cold_gas = where(self.gas_temp < 1.e4)
        weights = self.gas_mass[cold_gas]
        self.cg_rad_heatmap, self.cg_rad_xedges, self.cg_rad_yedges = np.histogram2d(self.epsilon_gas[cold_gas], self.gas_pos_mag[cold_gas], 
                                                                   bins=[linspace(eps_min,eps_max,bins_n), linspace(min_rad,max_rad,bins_n)], 
                                                                   weights = weights)
        '''
    def write_fits(self):
        print '\tGenerating fits for %s...'%self.aname
        master_hdulist = []
        prihdr = fits.Header()
        prihdr['COMMENT'] = "Storing the momentum measurements in this FITS file."
        prihdr['simname'] = self.simname
        prihdr['scale'] = self.aname.strip('a')
        prihdr['snapfile'] = self.snapfile

        prihdu = fits.PrimaryHDU(header=prihdr)    
        master_hdulist.append(prihdu)

        colhdr = fits.Header()

        if False:
            master_hdulist.append(fits.ImageHDU(data = self.L_disk                                                             , header = colhdr, name = 'net_angmomentum'))  
            master_hdulist.append(fits.ImageHDU(data = self.L_disk_fixed                                                       , header = colhdr, name = 'net_angmomentum_fixed'))  


        master_hdulist.append(fits.ImageHDU(data = self.stars_id                                                           , header = colhdr, name = 'stars_id'))
        master_hdulist.append(fits.ImageHDU(data = self.dark_id                                                            , header = colhdr, name = 'dark_id'))


        master_hdulist.append(fits.ImageHDU(data = np.stack((self.stars_x_box , self.stars_y_box , self.stars_z_box))           , header = colhdr, name = 'stars_box_position'))
        master_hdulist.append(fits.ImageHDU(data = np.stack((self.stars_vx_box , self.stars_vy_box , self.stars_vz_box))        , header = colhdr, name = 'stars_box_velocity'))
        master_hdulist.append(fits.ImageHDU(data = np.stack((self.stars_x , self.stars_y , self.stars_z))                       , header = colhdr, name = 'stars_gal_position'))
        master_hdulist.append(fits.ImageHDU(data = np.stack((self.stars_vx , self.stars_vy , self.stars_vz))                    , header = colhdr, name = 'stars_gal_velocity'))


        master_hdulist.append(fits.ImageHDU(data = np.stack((self.dark_x_box ,  self.dark_y_box ,  self.dark_z_box))             , header = colhdr, name = 'dark_box_position'))
        master_hdulist.append(fits.ImageHDU(data = np.stack((self.dark_vx_box , self.dark_vy_box , self.dark_vz_box))            , header = colhdr, name = 'dark_box_velocity'))
        master_hdulist.append(fits.ImageHDU(data = np.stack((self.dark_x ,      self.dark_y ,      self.dark_z))                 , header = colhdr, name = 'dark_gal_position'))
        master_hdulist.append(fits.ImageHDU(data = np.stack((self.dark_vx ,     self.dark_vy ,     self.dark_vz))                , header = colhdr, name = 'dark_gal_velocity'))




        if False:
            master_hdulist.append(fits.ImageHDU(data = np.stack((self.stars_jx, self.stars_jy, self.stars_jz))                 , header = colhdr, name = 'stars_gal_angmomentum'))
            master_hdulist.append(fits.ImageHDU(data = np.stack((self.dark_jx,  self.dark_jy,  self.dark_jz))                  , header = colhdr, name = 'dark_gal_angmomentum'))



            master_hdulist.append(fits.ImageHDU(data = self.epsilon_stars                                                      , header = colhdr, name = 'stars_epsilon'))
            master_hdulist.append(fits.ImageHDU(data = self.epsilon_stars_fixed                                                , header = colhdr, name = 'stars_epsilon_fixed'))

            master_hdulist.append(fits.ImageHDU(data = self.epsilon_dark                                                       , header = colhdr, name = 'dark_epsilon'))
            master_hdulist.append(fits.ImageHDU(data = self.epsilon_dark_fixed                                                 , header = colhdr, name = 'dark_epsilon_fixed'))


        master_hdulist.append(fits.ImageHDU(data = self.star_mass                                                           , header = colhdr, name = 'star_mass'))
        master_hdulist.append(fits.ImageHDU(data = self.star_age                                                            , header = colhdr, name = 'star_age'))

        master_hdulist.append(fits.ImageHDU(data = self.dark_mass                                                           , header = colhdr, name = 'dark_mass'))
        master_hdulist.append(fits.ImageHDU(data = self.dark_age                                                            , header = colhdr, name = 'dark_age'))

        if False:
            master_hdulist.append(fits.ImageHDU(data = self.mass_profile                                                        , header = colhdr, name = 'mass_profile'))



        if False:
            # save gas info
            master_hdulist.append(fits.ImageHDU(data = np.stack((self.cg_zz_xedges , self.cg_zz_yedges))        , header = colhdr, name = 'gas_zz_epsilon_edges'))
            master_hdulist.append(fits.ImageHDU(data = self.cg_zz_heatmap                                       , header = colhdr, name = 'gas_zz_epsilon'))


            master_hdulist.append(fits.ImageHDU(data = np.stack((self.cg_rr_xedges , self.cg_rr_yedges))        , header = colhdr, name = 'gas_rr_epsilon_edges'))
            master_hdulist.append(fits.ImageHDU(data = self.cg_rr_heatmap                                       , header = colhdr, name = 'gas_rr_epsilon'))


            master_hdulist.append(fits.ImageHDU(data = np.stack((self.cg_rad_xedges , self.cg_rad_yedges))     , header = colhdr, name = 'gas_rad_epsilon_edges'))
            master_hdulist.append(fits.ImageHDU(data = self.cg_rad_heatmap                                     , header = colhdr, name = 'gas_rad_epsilon'))

            master_hdulist.append(fits.ImageHDU(data = np.stack((self.gas_x , self.gas_y , self.gas_z))        , header = colhdr, name = 'gas_xyz_position'))
            master_hdulist.append(fits.ImageHDU(data = np.stack((self.rr_gas, self.zz_gas))                                , header = colhdr, name = 'gas_cylindrical_position'))
            master_hdulist.append(fits.ImageHDU(data = np.stack((self.gas_jx, self.gas_jy, self.gas_jz))       , header = colhdr, name = 'gas_momentum'))
            master_hdulist.append(fits.ImageHDU(data = self.epsilon_gas                                                    , header = colhdr, name = 'gas_epsilon'))
            #master_hdulist.append(fits.ImageHDU(data = self.gas_temp                                                       , header = colhdr, name = 'gas_temperature'))
            #master_hdulist.append(fits.ImageHDU(data = self.gas_mass                                                       , header = colhdr, name = 'gas_mass'))
        

        print '\tSaving to ' + self.fits_name
        thdulist = fits.HDUList(master_hdulist)
        thdulist.writeto(self.fits_name, clobber = True)

        return master_hdulist



def run_measure_momentum(haloname, simname, snapname, galprops, on_system = 'pfe'):
    if on_system == 'pfe':
        snaps = np.sort(np.asarray(glob.glob("/nobackupp2/mpeeples/%s/orig/%s/%s/%s"%(haloname, simname, snapname, snapname))))
        out_dir = '/nobackupp2/rcsimons/foggie_momentum/momentum_fits'
    else:
        print "/Volumes/gdrive/foggie/%s/nref11n/%s/%s/%s"%(haloname, simname, snapname, snapname)
        snaps = np.sort(np.asarray(glob.glob("/Volumes/gdrive/foggie/%s/nref11n/%s/%s/%s"%(haloname, simname, snapname, snapname))))
        out_dir = '/Users/rsimons/Dropbox/rcs_foggie/outputs'
        #snaps = np.sort(np.asarray(glob.glob("/Users/rsimons/Dropbox/rcs_foggie/data/%s/%s/%s/%s"%(haloname, simname, snapname, snapname))))
        #out_dir = '/Users/rsimons/Dropbox/rcs_foggie/outputs'



    assert os.path.lexists(snaps[0])

    assert os.path.lexists(out_dir)

    new_snapfiles = np.asarray(snaps)


    ts = yt.DatasetSeries(new_snapfiles)
    for ds,snapfile in zip(reversed(ts),np.flipud(new_snapfiles)):
    
        ad = ds.all_data()


        print 'Creating momentum fits file for '+ snapfile
        aname = snapfile.split('/')[-1]
        fits_name = out_dir+'/'+simname+'_'+aname+'_momentum.fits'

        print 'fits name : ', fits_name


        print 'Generating angular momentum object...'
        amom = momentum_obj(simname, aname, snapfile, fits_name)

        amom.load()


        amom.recenter(galprops)


        amom.L_disk = galprops['gas_L'][0]
        amom.L_disk_fixed = [-0.37085436,  0.14802026,  0.91681898]



        amom.calc_angular_momentum(ptype = 'stars')
        amom.calc_angular_momentum(ptype = 'darkmatter')

        amom.measure_potential()
        amom.measure_circularity()
        #amom.gas_momentum_heatmap()
        amom.write_fits()




        return amom


if __name__ == "__main__":

    #args = parse()

    #simname = args['simname']
    #snapname = args['snapname']
    #haloname = args['haloname']
    #run_parallel = args['run_parallel']
    #on_system = args['on_system']


    ddmin = 906
    ddmax = 908
    run_parallel = False
    haloname = 'halo_008508'
    simname = 'nref11n_nref10f'
    on_system = 'local'

    '''
    if on_system == 'pfe':
        galprops_outdir = '/nobackupp2/rcsimons/foggie_momentum/galprops'
        galaxy_props_file = galprops_outdir + '/'  + simname + '_' + snapname + '_galprops.npy'
    else:
        galprops_outdir = '/Users/rsimons/Dropbox/rcs_foggie/outputs'
        galaxy_props_file = galprops_outdir + '/temp_galprops.npy'

    galprops = np.load(galaxy_props_file)[()]
    '''
    #print haloname, simname, snapname, run_parallel

    #ddmin, ddmax = int(args['ddmin']), int(args['ddmax'])
    #snapnames = ['DD%.4i'%i for i in arange(ddmin, ddmax)]

    #snapnames = ['DD0906']
    #snapnames = ['DD0907']
    snapnames = ['DD0956']




    '''
    if run_parallel:
        n_jobs = int(args['n_jobs'])
        if (simname is not None) & (haloname is not None):
            Parallel(n_jobs = n_jobs, backend = 'threading')(delayed(run_measure_momentum)(haloname = haloname, simname = simname, snapname = snapname, galprops = galprops, on_system = on_system) for snapname in snapnames)
        else:
            print 'run_all_parallel set to True, but no simname or haloname provided.'
    else:
        for snapname in snapnames:
            amom = run_measure_momentum(haloname = haloname, simname = simname, snapname = snapname, galprops = galprops, on_system = on_system)
    '''

    if True:
        # Test run_measure_momentum in ipython
        for s, snapname in enumerate(snapnames):
            if on_system == 'pfe':
                snaps = np.sort(np.asarray(glob.glob("/nobackupp2/mpeeples/%s/%s/%s/%s"%(haloname, simname, snapname, snapname))))
                out_dir = '/nobackupp2/rcsimons/foggie_momentum/momentum_fits'
            else:
                print "/Volumes/gdrive/foggie/%s/nref11n/%s/%s/%s"%(haloname, simname, snapname, snapname)
                
                snaps = np.sort(np.asarray(glob.glob("/Users/rsimons/Dropbox/rcs_foggie/data/%s/%s/%s"%(haloname, snapname, snapname))))
                out_dir = '/Users/rsimons/Dropbox/rcs_foggie/outputs'
                #snaps = np.sort(np.asarray(glob.glob("/Users/rsimons/Dropbox/rcs_foggie/data/%s/%s/%s/%s"%(haloname, simname, snapname, snapname))))
                #out_dir = '/Users/rsimons/Dropbox/rcs_foggie/outputs'



            assert os.path.lexists(snaps[0])



            assert os.path.lexists(out_dir)

            new_snapfiles = np.asarray(snaps)


            ts = yt.DatasetSeries(new_snapfiles)
            for ds,snapfile in zip(reversed(ts),np.flipud(new_snapfiles)):
            
                ad = ds.all_data()


                print 'Creating momentum fits file for '+ snapfile
                aname = snapfile.split('/')[-1]
                fits_name = out_dir+'/'+simname+'_'+aname+'_momentum.fits'

                print 'fits name : ', fits_name


                print 'Generating angular momentum object...'
                amom = momentum_obj(simname, aname, snapfile, fits_name)

                amom.load()

                
                #amom.recenter(galprops)
                if snapname == 'DD0906': cen = yt.YTArray([ds.quan(0.4922914505, 'code_length'), ds.quan(0.482047080994, 'code_length'), ds.quan(0.504963874817, 'code_length')]).to('kpc')
                if snapname == 'DD0907': cen = yt.YTArray([ds.quan(0.492289543152, 'code_length'), ds.quan(0.48203754425, 'code_length'), ds.quan(0.504967689514, 'code_length')]).to('kpc')
                if snapname == 'DD0956': cen = yt.YTArray([ds.quan(0.49216556549072266, 'code_length'), ds.quan(0.48153591156005865, 'code_length'), ds.quan(0.5051298141479492, 'code_length')]).to('kpc')

                amom.cen_x, amom.cen_y, amom.cen_z = cen[0], cen[1], cen[2]
                amom = recenter_2(amom)
                
                #amom.L_disk = galprops['gas_L'][0]
                #amom.L_disk_fixed = [-0.37085436,  0.14802026,  0.91681898]

                #amom.calc_angular_momentum(ptype = 'stars')
                #amom.calc_angular_momentum(ptype = 'darkmatter')

                #amom.measure_potential()
                #amom.measure_circularity()
                #amom.gas_momentum_heatmap()
                amom.write_fits()













