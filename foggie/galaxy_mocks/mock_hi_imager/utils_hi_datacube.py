import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import unyt as u
import scipy
from scipy import stats
from foggie.clumps.clump_finder.utils_clump_finder import read_virial_mass_file
from functools import partial

from foggie.galaxy_mocks.mock_hi_imager.HICubeHeader import * #Constants

class TqdmProgressBar:
    '''
    Basic display bar for progress
    '''
    def __init__(self, title, maxval,position):
        from tqdm import tqdm

        self._pbar = tqdm(leave=True, total=maxval, desc=title, position=position)
        self.i = 0

    def update(self, i=None):
        if i is None:
            i = self.i + 1
        n = i - self.i
        self.i = i
        self._pbar.update(n)

    def finish(self):
        self._pbar.close()


def generate_cut_mask(ds,source_cut,ifu_shape,pixel_res_kpc):
    '''
    This function generates a projected mask for a cut region following the same logic as generate_spectra.
    This is used for masking the disk out in the projected images for distinguishing CGM and disk.
    '''
    print("Loading data...")
    
    nx,ny,nspec = ifu_shape
    import gc
    cut_mask = np.zeros((nx,ny)).astype(bool)
    print("SHAPE OF CUT_MASK IS",np.shape(cut_mask))

    if args.memory_chunks>0:
        pos_x_projected = source_cut['gas','pos_x_projected']
        pos_y_projected = source_cut['gas','pos_y_projected']
        pos_dx = source_cut['gas','dx']

        nCells = np.size(pos_dx)
        chunk_size = int(np.ceil(nCells / args.memory_chunks))

        for cc in range(args.memory_chunks):
            start = cc * chunk_size
            end = min((cc+1)*chunk_size+1,nCells)

            nChunkCells = end - start 

            print("Calculating projection parameters...")
            min_x_idx =  np.round( (pos_x_projected[start:end] - pos_dx[start:end]/2.).in_units('kpc') / pixel_res_kpc + nx / 2.).astype(int)
            max_x_idx =  np.round( (pos_x_projected[start:end] + pos_dx[start:end]/2.).in_units('kpc') / pixel_res_kpc + nx / 2.).astype(int)

            min_y_idx =  np.round( (pos_y_projected[start:end] - pos_dx[start:end]/2.).in_units('kpc') / pixel_res_kpc + ny / 2.).astype(int)
            max_y_idx =  np.round( (pos_y_projected[start:end] + pos_dx[start:end]/2.).in_units('kpc') / pixel_res_kpc + ny / 2.).astype(int)

            mask = ( (max_x_idx>=0) & (max_y_idx>=0) & (min_x_idx < nx) & (min_y_idx < ny) )

            min_x_idx = min_x_idx[mask]
            max_x_idx = max_x_idx[mask]
            min_y_idx = min_y_idx[mask]
            max_y_idx = max_y_idx[mask]

            for i in range(0,np.size(min_x_idx)):
                #Can paralllelize!
                x0 = min_x_idx[i]
                x1 = max_x_idx[i]
                y0 = min_y_idx[i]
                y1 = max_y_idx[i]

                cut_mask[x0:x1, y0:y1] = True


            gc.collect()
    return cut_mask




def _v_disp(field,data):
    #Velocity dispersion of a single cell of gas
    return np.sqrt(3*kb* np.divide( data['gas','temperature'], data['gas','mass']))


def get_inclination_rot_arr(inclination,position_angle):
    '''
    Calculate the rotation matrix for converting from the original coordinate system
    into a new basis defined by the inclination and position angle.
    Modified from foggie_load's definition for disk orientation
    '''
    inclination = inclination * np.pi/180. #convert to radians
    position_angle = position_angle * np.pi/180. #convert to radians
    z = np.array([np.cos(np.pi/2. - inclination)*np.cos(position_angle),np.cos(np.pi/2. - inclination)*np.sin(position_angle),np.sin(np.pi/2. - inclination)])
    
    np.random.seed(99)
    x = np.random.randn(3)            # take a random vector
    x = np.array( [np.cos(position_angle)*np.sin(np.pi/2.-inclination), np.sin(position_angle)*np.sin(np.pi/2.-inclination), np.cos(np.pi/2.-inclination)] )

    x -= x.dot(z) * z       # make it orthogonal to L
    x /= np.linalg.norm(x)            # normalize it
    y = np.cross(z, x)           # cross product with L

    xhat = np.array([1,0,0])
    yhat = np.array([0,1,0])
    zhat = np.array([0,0,1])
    transArr0 = np.array([[xhat.dot(x), xhat.dot(y), xhat.dot(z)],
                          [yhat.dot(x), yhat.dot(y), yhat.dot(z)],
                          [zhat.dot(x), zhat.dot(y), zhat.dot(z)]])
    
    return np.linalg.inv(transArr0) , z


def _pseudo_pos_x_projected(field,data,observer_distance,rot_arr):
    '''
    Psuedo-function for adding a projected x position field
    '''
    old_x = data['gas','x_disk'].in_units('kpc')
    old_y = data['gas','y_disk'].in_units('kpc')
    old_z = data['gas','z_disk'].in_units('kpc')

    
    return rot_arr[0][0]*old_x+rot_arr[0][1]*old_y+rot_arr[0][2]*old_z


def _pseudo_pos_y_projected(field,data,observer_distance,rot_arr):
    '''
    Psuedo-function for adding a projected y position field
    '''
    old_x = data['gas','x_disk'].in_units('kpc')
    old_y = data['gas','y_disk'].in_units('kpc')
    old_z = data['gas','z_disk'].in_units('kpc') 

    
    return rot_arr[1][0]*old_x+rot_arr[1][1]*old_y+rot_arr[1][2]*old_z

def _pseudo_pos_z_projected(field,data,observer_distance,rot_arr):
    '''
    Psuedo-function for adding a projected z position field
    Corresponds to distance from the observer.
    '''
    old_x = data['gas','x_disk'].in_units('kpc')
    old_y = data['gas','y_disk'].in_units('kpc')
    old_z = data['gas','z_disk'].in_units('kpc') 
    
    return rot_arr[2][0]*old_x+rot_arr[2][1]*old_y+rot_arr[2][2]*old_z + observer_distance.in_units('kpc')

def _pseudo_v_doppler(field,data,Lhat):
    '''
    Psuedo-function for adding the projected velocity field.
    Accounts for hubble flow based on distance to observer.
    '''
    hubble_flow = H0 * data['gas','pos_z_projected'].in_units('Mpc')
    projected_velocity = data['gas','vx_disk'] * Lhat[0] + data['gas','vy_disk'] * Lhat[1] + data['gas','vz_disk'] * Lhat[2]
    projected_velocity = projected_velocity + hubble_flow #check_sign

    return projected_velocity

def _v_doppler_density_weighted(field,data):
    '''
    Field for density weighted projected velocities. Used for making simple moment maps and finding maximum projected velocities.
    '''
    return np.multiply(data['gas','v_doppler'],data['gas','H_p0_number_density'])


def gaussian_high_pass(spatial_frequencies, min_spatial_freq):
    min_freq_sigma = min_spatial_freq / np.sqrt(2.0 * np.log(2.0)) #make min spatial freq half the FWHM of the gaussian
    return 1. - np.exp(-np.power(spatial_frequencies,2) / (2*(min_freq_sigma)**2))

def gaussian_low_pass(spatial_frequencies, max_spatial_freq):
    return np.exp(-np.power(spatial_frequencies,2) / (2*(max_spatial_freq)**2))

def butterworth_high_pass(spatial_frequencies, min_spatial_freq,n):
    return 1. / (1. + np.power(spatial_frequencies / min_spatial_freq, 2*n))


def enforce_hermitian_symmetry(freq_image):
    """Ensures that Fourier coefficients satisfy Hermitian symmetry."""
    flipped = np.conj(np.roll(np.roll(
        np.flipud(np.fliplr(freq_image)), 
        1, axis=0), 
        1, axis=1))
    freq_image = (freq_image + flipped) / 2
    return freq_image


import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def apply_gaussian_smoothing(ifu, instrument, args, for_highpass = False):
    #Apply pointspread function to every pixel
    if for_highpass:
        lmbda =  21.106114054160/100./1000. # in km
        max_observable_angle = lmbda /  instrument.min_baseline  #1/rad
        min_spatial_freq = max_observable_angle * rad_to_arcsec # 1/rad
        sigma = min_spatial_freq / args.base_spatial_res
        print("FOR HPF SIGMA=",sigma)
        ifu_units = ifu[0,0,0].units
        ifu = scipy.ndimage.gaussian_filter(ifu, sigma=sigma,axes=[0,1])
        return ifu * ifu_units
    try:
        ifu_units = ifu[0,0,0].units
    except:
        ifu_units = 1


    sigma = (instrument.obs_spatial_res/args.base_spatial_res) / 2 / np.sqrt(2*np.log(2)) #Convert full width of beam to sigma
    #ifu = scipy.ndimage.gaussian_filter(ifu, sigma=instrument.obs_spatial_res/args.base_spatial_res , axes=[0,1])
    ifu = scipy.ndimage.gaussian_filter(ifu, sigma=sigma , axes=[0,1])

    return ifu * ifu_units

def make_moment_map_figures(args,ifu,output):
 #Calculate some ideal moment maps...
        instrument = radio_telescope(args)
        dv = instrument.obs_spec_res * u.km/u.s
        ifu_velocities = np.linspace( -0.5*dv * instrument.obs_channels, 0.5*dv * instrument.obs_channels, instrument.obs_channels)
      #  print("ifu_velocities=",ifu_velocities)
       # print("dv=",dv)
        nx,ny,nz = np.shape(ifu)
        nu = np.linspace(args.base_freq_range[1].in_units('1/s'),args.base_freq_range[0].in_units('1/s'),nz)
        dnu = ( np.max(nu)-np.min(nu) ) / np.size(nu)
        Iv = ifu * dnu/dv

        m0_map_ideal = np.sum(Iv,axis=2) * dv
        m1_map_ideal = np.divide( np.sum(np.multiply(Iv,ifu_velocities[np.newaxis,np.newaxis,:]),axis=2 ) , m0_map_ideal ) * dv
        m1_map_ideal[np.isnan(m1_map_ideal)] = 0

        m2_term = ifu_velocities[np.newaxis,np.newaxis,:] - m1_map_ideal[:,:,np.newaxis]
        print("m2_term=",np.min(m2_term),np.max(m2_term))
        m2_map_ideal = np.sqrt( np.divide(np.sum( np.multiply(Iv, np.power(m2_term,2.)),axis=2 ) , m0_map_ideal) * dv ).in_units('km/s')
        m2_map_ideal[np.isnan(m2_map_ideal)] = 0
        
         
        nBins = 100
        logNHI_min = 17.5
        logNHI_max = 22.
        sigma_min = np.log10(3)
        sigma_max = 2
        m0_v_m2_histogram,tmp,tmp,tmp =  stats.binned_statistic_2d(np.log10(m0_map_ideal).flatten(), np.log10(m2_map_ideal).flatten(), np.ones(np.size(m0_map_ideal)), statistic = 'sum', bins = [nBins,nBins], range=[[logNHI_min,logNHI_max],[sigma_min,sigma_max]] )
        plt.figure()
        plt.imshow(np.flipud(np.rot90(m0_v_m2_histogram)),cmap='inferno',norm=LogNorm())
        plt.xlabel("log(N$_{\\rm HI}$/cm$^{-2}$)")
        plt.ylabel("$\\sigma$ [km/s]")
        xtks = [18,19,20,21,22]
        ytks = [5,10,20,30,50,70,100]
        xtk=[]
        ytk=[]
        nx,ny = np.shape(m0_v_m2_histogram)
        for i in range(len(xtks)):
            xtk.append((xtks[i] - logNHI_min) / (logNHI_max - logNHI_min) * nx)
            xtks[i] = str(xtks[i])

        for i in range(len(ytks)):
            ytk.append(ny - (np.log10(ytks[i]) - sigma_min ) / (sigma_max  -  sigma_min) * ny)
            ytks[i] = str(ytks[i])

        plt.xticks(xtk,xtks)
        plt.yticks(ytk,ytks)
        plt.colorbar()

        plt.savefig(output+"_m0_v_m2.png",bbox_inches='tight')
        plt.close()

        hf=h5py.File(output+"_MomentMaps.h5",'w')
        hf.create_dataset("m0_map",data=m0_map_ideal)
        hf.create_dataset("m1_map",data=m1_map_ideal)
        hf.create_dataset("m2_map",data=m2_map_ideal)


        hf.close()




from PIL import Image
import os

def create_gif_from_pngs(png_folder, output_gif, duration=500):
    """
    Compile a series of .png files into a .gif.

    Parameters:
        png_folder (str): Path to the folder containing .png files.
        output_gif (str): Path to save the output .gif file.
        duration (int): Duration for each frame in milliseconds. Default is 500ms.
    """
    # Get a sorted list of .png files in the folder
    png_files = sorted([f for f in os.listdir(png_folder) if f.endswith('.png')])
    
    if not png_files:
        raise ValueError("No .png files found in the specified folder.")
    
    # Open the images and compile them into a list
    images = [Image.open(os.path.join(png_folder, file)) for file in png_files]
    
    # Save as .gif
    images[0].save(
        output_gif,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )
    print(f"GIF saved at {output_gif}")



def LoadSofiaMask(filedir):
    with fits.open(filedir) as hdul:
        #hdul.info()  # Show HDU list
        header = hdul[0].header
        mask = hdul[0].data  # This is a NumPy array

    #nspec,nx,ny = np.shape(mask)
    print("MASK LOADED FROM SOFIA")
    return np.transpose(mask, (1,2,0))


def FindMaxProjectedVelocity(args,ds,source_cut,Rvir=None):
    import yt
    if Rvir is None:
        try:
            Rvir = read_virial_mass_file(args.halo,args.snapshot,args.run,args.code_dir,key='radius')
        except:
            print("Warning, could not read virial mass file...Setting to 250 kpc")
            Rvir = 250.
   


    observer_distance = (args.z * c / H0).in_units("kpc")
    rot_arr, Lhat = get_inclination_rot_arr(args.inclination,args.position_angle)

    north_vector=ds.z_unit_disk


    print("Observer distance=",observer_distance)
    _pos_x_projected = partial(_pseudo_pos_x_projected, observer_distance = observer_distance, rot_arr = rot_arr )
    _pos_y_projected = partial(_pseudo_pos_y_projected, observer_distance = observer_distance, rot_arr = rot_arr )
    _pos_z_projected = partial(_pseudo_pos_z_projected, observer_distance = observer_distance, rot_arr = rot_arr )

    ds.add_field(
        ('gas', 'pos_x_projected'),
          function=_pos_x_projected,
          sampling_type='cell',
          force_override=True
    )

    ds.add_field(
        ('gas', 'pos_y_projected'),
          function=_pos_y_projected,
          sampling_type='cell',
          force_override=True
    )

    ds.add_field(
        ('gas', 'pos_z_projected'),
          function=_pos_z_projected,
          sampling_type='cell',
          force_override=True
    )

    print("Source_cut projected_z=",np.min(source_cut['gas','pos_z_projected'].in_units('kpc')),np.max(source_cut['gas','pos_z_projected'].in_units('kpc')))

    _v_doppler = partial(_pseudo_v_doppler, Lhat=Lhat )

    ds.add_field(
        ('gas', 'v_doppler'),
          function=_v_doppler,
          sampling_type='cell',
          force_override=True
    )


    center = ds.halo_center_code
    W = [args.fov_kpc,args.fov_kpc,Rvir*2.]  * ds.units.kpc
    res_r = 0.274 * ds.units.kpc
    N = int(np.round(args.fov_kpc/res_r))



    ds.add_field(
    ('gas', 'v_doppler_density_weighted'),
        function=_v_doppler_density_weighted,
        sampling_type='cell',
        force_override=True,
        units='1/(cm**2*s)',
    )

    #V doppler should already be in the disk frame...

    Lhat = Lhat @ np.linalg.inv(ds.disk_rot_arr) #switch this back to the simulation frame for use with yt...
 #R_total @ np.array([0,0,1])
    density_projection = yt.off_axis_projection(source_cut, center,Lhat,W,N,("gas","H_p0_number_density"), method='integrate',north_vector=north_vector).in_units("1/cm**2")

    v_doppler_projection = yt.off_axis_projection(source_cut, center,Lhat,W,N,("gas","v_doppler_density_weighted"), method='integrate', north_vector=north_vector)

    v_doppler_projection = np.divide( v_doppler_projection , density_projection )#h0 density weighted map
    v_doppler_projection = v_doppler_projection.in_units('km/s')

    v_doppler_projection=v_doppler_projection - (args.z * c) # remove the average hubble flow
    v_doppler_projection = v_doppler_projection.in_units('km/s')
    
    #return r_plot,density_radial_profile

    mask = (density_projection < 1e16)
    v_doppler_projection[mask]=0

    return np.max(np.abs(v_doppler_projection.in_units('km/s').v)) * u.km/u.s #* np.sin(np.radians(args.disk_inclination)) #in km/s

