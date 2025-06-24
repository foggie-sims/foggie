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
    x -= x.dot(z) * z       # make it orthogonal to L
    x /= np.linalg.norm(x)            # normalize it
    y = np.cross(z, x)           # cross product with L

    # Calculate the rotation matrix for converting from original coordinate system
    # into this new basis
    xhat = np.array([1,0,0])
    yhat = np.array([0,1,0])
    zhat = np.array([0,0,1])
    transArr0 = np.array([[xhat.dot(x), xhat.dot(y), xhat.dot(z)],
                          [yhat.dot(x), yhat.dot(y), yhat.dot(z)],
                          [zhat.dot(x), zhat.dot(y), zhat.dot(z)]])
    
    return np.linalg.inv(transArr0) , z


def _pseudo_pos_x_projected(field,data,observer_distance,rot_arr):
    '''
    Psuedo-function for adding a pojrected x position field
    '''
    old_x = data['gas','x_disk'].in_units('kpc')
    old_y = data['gas','y_disk'].in_units('kpc')
    old_z = data['gas','z_disk'].in_units('kpc')

    
    return rot_arr[0][0]*old_x+rot_arr[0][1]*old_y+rot_arr[0][2]*old_z


def _pseudo_pos_y_projected(field,data,observer_distance,rot_arr):
    '''
    Psuedo-function for adding a pojrected y position field
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




def make_simple_moment_maps(args,ds,source_cut,L,north_vector,filename,max_r=150,Rvir=None,radial_averaging_stat='mean'):
    import yt
    if Rvir is None:
        try:
            Rvir = read_virial_mass_file(args.halo,args.snapshot,args.run,args.code_dir,key='radius')
        except:
            print("Warning, could not read virial mass file...Setting to 250 kpc")
            Rvir = 250.
   

    center = ds.halo_center_code
    W = [2*max_r,2*max_r,Rvir*2.]  * ds.units.kpc
    res_r = 0.274 * ds.units.kpc
    N = int(np.round(2*max_r/res_r))



    ds.add_field(
    ('gas', 'v_doppler_density_weighted'),
        function=_v_doppler_density_weighted,
        sampling_type='cell',
        force_override=True,
        units='1/(cm**2*s)',
    )

    #V doppler should already be in the disk frame...

    L = L @ np.linalg.inv(ds.disk_rot_arr) #switch this back to the simulation frame for use with yt...
 #R_total @ np.array([0,0,1])
    density_projection = yt.off_axis_projection(source_cut, center,L,W,N,("gas","H_p0_number_density"), method='integrate',north_vector=north_vector).in_units("1/cm**2")

    
    v_doppler_projection = yt.off_axis_projection(source_cut, center,L,W,N,("gas","v_doppler_density_weighted"), method='integrate', north_vector=north_vector)

    v_doppler_projection = np.divide( v_doppler_projection , density_projection )#h0 density weighted map
    v_doppler_projection = v_doppler_projection.in_units('km/s')

    v_doppler_projection=v_doppler_projection - (args.z * c) # remove the average hubble flow
    v_doppler_projection = v_doppler_projection.in_units('km/s')
    print(v_doppler_projection)
    

    pixel_indices = np.indices(np.shape(density_projection))
    print("SHAPE OF DENSITY PROJECTION=",np.shape(density_projection))
    pixel_size = res_r.in_units('kpc').v # kpc
    print("PIXEL_SIZE=",pixel_size)
    print("MAX_R=",max_r)
    pixel_radii = np.sqrt(((pixel_indices[0,:,:] * pixel_size) - max_r.in_units('kpc').v)**2 + ((pixel_indices[1,:,:] * pixel_size) - max_r.in_units('kpc').v)**2)

    nbins_pix = np.max(np.shape(pixel_radii)).astype(int)
    density_radial_profile,tmp,tmp =  stats.binned_statistic(pixel_radii.flatten(),density_projection.flatten(),statistic=radial_averaging_stat,bins=nbins_pix)

    r_plot = np.linspace(0,max_r.in_units('kpc').v,nbins_pix)
    plt.figure(figsize=(8,6))
    plt.plot(r_plot,density_radial_profile,'k',lw=3,)
    plt.xlabel("Radius [kpc]")
    plt.ylabel("log(N$_{\\rm HI}$/cm$^{-2}$)")
    plt.yscale('log')
    plt.xscale('log')
    plt.xlim([0.8e1,1.2e2])
    plt.ylim([1e16,1e25])

    if filename is None:
        filename="./ideal_moment_maps/misc/NHI_Proj_i"+str(int(args.inclination)).zfill(2)+"_pa"+str(np.round(args.position_angle).astype(int)).zfill(3)

    plt.savefig(filename+"_density_radial_profile_log.png",bbox_inches='tight')
    plt.close()
    
    hf = h5py.File(filename+"_density_radial_profile_log.h5", 'w')
    hf.create_dataset('density_radial_profile', data=density_radial_profile)
    hf.create_dataset('pixel_radii', data=pixel_radii)
    hf.close()

    #return r_plot,density_radial_profile


    mask = (density_projection < 1e16)
    density_projection[mask]=np.nan
    v_doppler_projection[mask]=np.nan


    labelsize = 18
    ticksize = 16
    titlesize = 20
    plt.figure(figsize=(16,6))
    plt.subplot(121)
    plt.imshow(density_projection,norm=LogNorm(vmin=1e14,vmax=1e22),cmap='viridis')
    xtk=[-40,-20,0,20,40]
    xtks=[]
    for tk in xtk:
        xtks.append(str(tk))
    xtk = np.array(xtk)
    xtk = xtk-np.min(xtk)
    xtk = xtk / np.max(xtk) * np.shape(density_projection)[0]
    plt.xticks(xtk,xtks,fontsize=ticksize)
    plt.yticks(xtk,xtks,fontsize=ticksize)
    plt.xlabel("$\\Delta$ x [kpc]",fontsize=labelsize)
    plt.ylabel("$\\Delta$ y [kpc]",fontsize=labelsize)
    cbar=plt.colorbar()
    cbar.set_label("log(N$_{\\rm HI}$/cm$^{-2}$)", fontsize=labelsize)
    cbar.ax.tick_params(labelsize=ticksize)
    plt.title("inclination="+str(np.round(args.inclination).astype(int))+", position angle="+str(np.round(args.position_angle).astype(int)),fontsize=titlesize)

    plt.subplot(122)
    plt.imshow(v_doppler_projection,vmin=-150,vmax=150,cmap='RdYlBu_r')
    xtk=[-40,-20,0,20,40]
    xtks=[]
    for tk in xtk:
        xtks.append(str(tk))
    xtk = np.array(xtk)
    xtk = xtk-np.min(xtk)
    xtk = xtk / np.max(xtk) * np.shape(density_projection)[0]
    plt.xticks(xtk,xtks,fontsize=ticksize)
    plt.yticks(xtk,xtks,fontsize=ticksize)
    plt.xlabel("$\\Delta$ x [kpc]",fontsize=labelsize)
   # plt.ylabel("$\\Delta$ y [kpc]",fontsize=labelsize)
    cbar=plt.colorbar()
    cbar.set_label("$\\Delta$v [km s$^{-1}$]", fontsize=labelsize)
    cbar.ax.tick_params(labelsize=ticksize)


    if filename is not None: plt.savefig(filename+".png",bbox_inches='tight')
    else: plt.savefig("./ideal_moment_maps/misc/moment_map_i"+str(int(args.inclination)).zfill(2)+"_pa"+str(np.round(args.position_angle).astype(int)).zfill(3)+".png",bbox_inches="tight")
    plt.close()

    hf = h5py.File(filename+"_simple_moment_maps.h5",'w')
    hf.create_dataset("density_projection", data = density_projection)
    hf.create_dataset("v_doppler_projection", data = v_doppler_projection)
    hf.create_dataset("max_r", data = max_r.in_units('kpc').v)
    hf.close()


    return r_plot,density_radial_profile, density_projection, v_doppler_projection



def generate_cut_mask(ds,source_cut,ifu_shape,pixel_res_kpc,args):
    '''
    This function generates a projected mask for a cut region following the same logic as generate_spectra.
    This is used for masking the disk out in the projected images for distinguishing CGM and disk.
    '''
    print("Loading data...")
    
    nx,ny,nspec = ifu_shape
    import gc
   # ifu = np.zeros(ifu_shape) * u.erg*u.g/u.km/u.s/u.s/u.statC/u.statC
    cut_mask = np.zeros((nx,ny)).astype(bool)
    print("SHAPE OF CUT_MASK IS",np.shape(cut_mask))

    if args.memory_chunks<=0: args.memory_chunks = 1

    pos_x_projected = source_cut['gas','pos_x_projected']
    pos_y_projected = source_cut['gas','pos_y_projected']
    pos_dx = source_cut['gas','dx']

    nCells = np.size(pos_dx)
    chunk_size = int(np.ceil(nCells / args.memory_chunks))

    for cc in range(args.memory_chunks):
        start = cc * chunk_size
        end = min((cc+1)*chunk_size+1,nCells)


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
            x0 = min_x_idx[i]
            x1 = max_x_idx[i]
            y0 = min_y_idx[i]
            y1 = max_y_idx[i]

            cut_mask[x0:x1, y0:y1] = True


        gc.collect()
    return cut_mask




def gaussian_high_pass(spatial_frequencies, min_spatial_freq):
    min_freq_sigma = min_spatial_freq / np.sqrt(2.0 * np.log(2.0)) #make min spatial freq the FWHM of the gaussian
    return 1. - np.exp(-np.power(spatial_frequencies,2) / (2*(min_freq_sigma)**2))
def gaussian_low_pass(spatial_frequencies, max_spatial_freq):
    return np.exp(-np.power(spatial_frequencies,2) / (2*(max_spatial_freq)**2))
def butterworth_high_pass(spatial_frequencies, min_spatial_freq,n):
    return 1. / (1. + np.power(spatial_frequencies / min_spatial_freq, 2*n))


def enforce_hermitian_symmetry(freq_image):
    """ Ensures that Fourier coefficients satisfy Hermitian symmetry. """
    freq_image = (freq_image + np.conj(np.flipud(np.fliplr(freq_image)))) / 2
    return freq_image


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
