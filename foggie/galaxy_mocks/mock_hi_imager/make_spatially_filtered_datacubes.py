import numpy as np
import h5py
from functools import partial
from joblib import Parallel, delayed
import multiprocessing
import time
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import scipy

from foggie.galaxy_mocks.mock_hi_imager.HICubeHeader import * #Constants
from foggie.galaxy_mocks.mock_hi_imager.utils_hi_datacube import apply_gaussian_smoothing

from foggie.galaxy_mocks.mock_hi_imager.utils_hi_datacube import gaussian_high_pass
from foggie.galaxy_mocks.mock_hi_imager.utils_hi_datacube import butterworth_high_pass
from foggie.galaxy_mocks.mock_hi_imager.utils_hi_datacube import enforce_hermitian_symmetry
from foggie.galaxy_mocks.mock_hi_imager.utils_hi_datacube import TqdmProgressBar

from foggie.galaxy_mocks.mock_hi_imager.custom_clean import custom_clean as cy_custom_clean
from foggie.galaxy_mocks.mock_hi_imager.custom_clean import complex_clean as cy_complex_clean
from foggie.galaxy_mocks.mock_hi_imager.custom_clean import masked_clean as cy_masked_clean
from foggie.galaxy_mocks.mock_hi_imager.custom_clean import complex_masked_clean as cy_complex_masked_clean


'''
Functions to spatially filter mock hi cubes generated in make_mock_datacubes.py

These functions take each slice into fourier space, apply a high pass filter (HPF) to cut out diffuse signal, and then run the CLEAN algorithm to recover as much signal as possible.
Cleaning is unfortunately still necesarry even when using a Gaussian HPF, as the spatial frequency cuts are so tight.

The core functionality of the CLEANing is in a cython code (custom_clean.pyx) and therefore may need to be compiled on your system.

Author: Cameron Trapp
Last updated 06/17/2025
'''


def get_beam_model(image_shape,instrument,args,spatial_frequencies,min_spatial_freq_lmbda,order=2):
    '''
    Function to get the beam model for a given instrument and image shape.
    Creates the beam by applying the instrument's gaussian smoothing to a point source image, and then applying a high pass filter to remove low frequencies.
    Normalized such that the max value is 1
    '''
    try:
        nx,ny,nz = image_shape
    except:
        nx,ny = image_shape

    

    point_source_image = np.zeros((nx,ny))
    point_source_image[nx//2,ny//2] = 1.

    #apply initial gaussian smoothing
    point_source_image = apply_gaussian_smoothing(point_source_image,instrument,args)
    #filter out low frequencies
    uv = np.fft.fftshift( np.fft.fft2(point_source_image) )
    ux,uy = np.shape(uv)

    if args.high_pass_filter_type is None: #This will introduce ringing artifacts you probably don't want...
        mask = (spatial_frequencies >= min_spatial_freq_lmbda).astype(int)
    elif args.high_pass_filter_type=='gaussian': #Works okay, introduces ringing artifacts and cuts off a lot of diffuse emission
        mask = gaussian_high_pass(spatial_frequencies, min_spatial_freq_lmbda)
    elif args.high_pass_filter_type=='butterworth': #Doesn't really work? Maybe needs a higher order (>10??)
        mask = butterworth_high_pass(spatial_frequencies, min_spatial_freq_lmbda,order)

    uv = np.multiply(uv,mask)

    #uv = enforce_hermitian_symmetry(uv)

    beam = np.fft.ifft2( np.fft.ifftshift(uv) )
    beam_norm = np.max(np.abs(beam))
    #print("Beam norm=",beam_norm)
    beam = beam / beam_norm
    return beam

def get_clean_beam(image_shape,args,instrument):
    '''
    Function to get the clean beam (simple gaussian) for a given instrument and image shape.
    This is the beam that each clean component is convolved with to generate the cleaned image.
    '''
    try:
        nx,ny,nz = image_shape
    except:
        nx,ny = image_shape


    sigma = float(instrument.obs_spatial_res) / float(args.base_spatial_res) / 2 / np.sqrt(2*np.log(2)) 
    x = (np.arange(nx) - float(nx-1)/2.).astype(float)
    y = (np.arange(ny) - float(ny-1)/2.).astype(float)
    X, Y = np.meshgrid(x, y,  indexing='ij')

    positions = np.sqrt(np.power(X,2) + np.power(Y,2)) #Redo this...
    beam= 1./(2.*np.pi*sigma**2) * np.exp(-np.power(positions,2) / (2*(sigma)**2)) 

    return beam / np.abs(np.max(beam))






def _filter_and_clean_slice(args,instrument,max_residual_limit,beam,mask,clean_beam,ifu_slice,i,nu,clean_mask=None):
        '''
        Pseudo-Function to spatially filter and spectral slice and run the CLEAN algorithm in parallel.
        Runs a simple Hogbom CLEAN algorithm in image space.
        Called by filter_spatial_frequencies_slicewise below.
        The bulk of the computation is done in the cython functions in custom_clean.pyx
        '''
        clean_complex = True
        if clean_mask is not None:
            clean_mask = np.array(np.where(clean_mask))
            n_masked_pixels = np.shape(clean_mask)[1]


        nx,ny = np.shape(ifu_slice)
        new_slice = np.zeros((2*nx,2*ny)) #Make twice as large to easily move the beam models around
        new_slice[nx//2:nx//2+nx,ny//2:ny//2+ny] = ifu_slice.value

        uv = np.fft.fftshift( np.fft.fft2(new_slice) ) #Take into fourier space
        ux,uy = np.shape(uv)

        sampling_frequency = 1./(args.base_spatial_res * arcsec_to_rad) #in 1/rad
        x = (np.arange(ux) - (ux-1)/2) * sampling_frequency / ux 
        y = (np.arange(uy) - (uy-1)/2) * sampling_frequency / uy
        X, Y = np.meshgrid(x, y,  indexing='ij')

        spatial_frequencies = np.sqrt(np.power(X,2) + np.power(Y,2)) #The abs spatial frequency of each point in fourier space

        lmbda = c / nu[i] #wavelength in km
        max_observable_angle = lmbda / instrument.min_baseline #1/rad
        min_spatial_freq_lmbda = 1. / max_observable_angle # 1/rad

        if args.high_pass_filter_type is None: #This will introduce ringing artifacts you probably don't want...
            mask = (spatial_frequencies >= min_spatial_freq_lmbda).astype(int)
        elif args.high_pass_filter_type=='gaussian': #Works okay, introduces ringing artifacts and cuts off a lot of diffuse emission
            mask = gaussian_high_pass(spatial_frequencies, min_spatial_freq_lmbda)
        elif args.high_pass_filter_type=='butterworth': #Doesn't really work? Maybe needs a higher order (>10??)
            mask = butterworth_high_pass(spatial_frequencies, min_spatial_freq_lmbda,order=2)

        beam = get_beam_model(np.shape(new_slice),instrument,args,spatial_frequencies,min_spatial_freq_lmbda,order=2) #Get the dirty beam from an ideal point source


        uv = np.multiply(uv,mask) #Apply the spatial filter to the image
        
       # uv = enforce_hermitian_symmetry(uv)

        filtered_image = np.fft.ifft2( np.fft.ifftshift(uv) ) #Gives the dirty image

        if not clean_complex:
           dirty_slice =  (np.abs(filtered_image) * np.exp(1j * np.angle(filtered_image)) ).real
           beam = ( np.abs(beam) * np.exp(1j * np.angle(beam)) ).real
        else:
            dirty_slice = filtered_image

        save_tmp_slices = True #make arg? Saves cleaned slices in ./tmp in case of crash or having to restart as some survey parameters take a long time to clean
        try:
            #Load previously cleaned slice from temp files
            hf_cs = h5py.File("./outputs/tmp_slices/tmp_cleaned_slice_"+args.halo+args.mock_suffix+"_"+args.survey+"_slice"+str(i)+".h5",'r')
            cleaned_slice = hf_cs["cleaned_slice"][...]
            residuals = hf_cs["residuals"][...]
            hf_cs.close()
            cleaned_slice_with_residuals = cleaned_slice + residuals
            print("Slice",i,"loaded cleaned slice from file")
        except:
            #Call the cython code to run CLEAN on the slice
            if args.max_clean_iterations is None:
                args.max_clean_iterations = int(ux*uy)
            tclean = time.time()
            if clean_complex:
                print("cleaning complex...")
                if clean_mask is not None:
                    cleaned_slice, residuals = cy_complex_masked_clean(dirty_slice[ux//4:3*ux//4,uy//4:3*uy//4], beam, clean_beam,clean_mask,n_masked_pixels, max_residual_limit, i, args.clean_gain, args.max_clean_iterations)
                else:
                    cleaned_slice, residuals = cy_complex_clean(dirty_slice[ux//4:3*ux//4,uy//4:3*uy//4], beam, clean_beam, max_residual_limit, i, args.clean_gain, args.max_clean_iterations)
                    #cleaned_slice, residuals = complex_clean(dirty_slice[ux//4:3*ux//4,uy//4:3*uy//4], beam, args,instrument, clean_beam, max_residual_limit=max_residual_limit,plot_beam_figures=True,tid=i)
            elif clean_mask is not None:
                cleaned_slice, residuals = cy_masked_clean(dirty_slice[ux//4:3*ux//4,uy//4:3*uy//4], beam, clean_beam,clean_mask,n_masked_pixels, max_residual_limit, i, args.clean_gain, args.max_clean_iterations)
            print("Slice",i,"ran clean in",time.time()-tclean,"seconds")
            cleaned_slice_with_residuals = cleaned_slice + residuals
            if save_tmp_slices:
                hf_cs = h5py.File("./outputs/tmp_slices/tmp_cleaned_slice_"+args.halo+args.mock_suffix+"_"+args.survey+"_slice"+str(i)+".h5",'w')
                hf_cs.create_dataset("cleaned_slice",data=cleaned_slice)
                hf_cs.create_dataset("residuals",data=residuals)
                hf_cs.close()

        #Ensure final image is real, accounting for phase
        if clean_complex:
           dirty_image =  (np.abs(dirty_slice) * np.exp(1j * np.angle(dirty_slice)) ).real
           clean_image =  (np.abs(cleaned_slice) * np.exp(1j * np.angle(cleaned_slice)) ).real
           clean_image_with_residuals =  (np.abs(cleaned_slice_with_residuals) * np.exp(1j * np.angle(cleaned_slice_with_residuals)) ).real

        else:
            dirty_image = dirty_slice.real
            clean_image = cleaned_slice.real #np.abs(cleaned_slice) * np.exp(1j * np.abs(cleaned_slice.imag))
            clean_image_with_residuals = cleaned_slice_with_residuals.real#np.abs(cleaned_slice_with_residuals) * np.exp(1j * np.abs(cleaned_slice_with_residuals.imag))

        if False: #For diagnostics if needed
            plt.figure()
            vmin = np.max(np.abs(dirty_image))/1e6
            if vmin == 0: vmin = 0.1
            plt.imshow(np.abs(dirty_image),norm=LogNorm(vmin=vmin))
            plt.colorbar()
            plt.savefig("./slices/dirty_slice_"+str(i)+".png")
            plt.close()

            plt.figure()
            vmin = np.max(np.abs(clean_image))/1e6
            if vmin == 0: vmin = 0.1
            plt.imshow(np.abs(clean_image),norm=LogNorm(vmin=vmin))
            plt.colorbar()
            plt.savefig("./slices/cleaned_slice_"+str(i)+".png")
            plt.close()

            plt.figure()
            vmin = np.max(np.abs(clean_image_with_residuals))/1e6
            if vmin == 0: vmin = 0.1
            plt.imshow(np.abs(clean_image_with_residuals),norm=LogNorm(vmin=vmin))
            plt.colorbar()
            plt.savefig("./slices/cleaned_slice_with_residuals"+str(i)+".png")
            plt.close()

        print("Slice",i,"filtered and cleaned!")

        return clean_image.real, clean_image_with_residuals.real



def filter_spatial_frequencies_slicewise(ifu, noise_flux, instrument, args, order=2, clean_mask=None):
    '''
    This is the core function of the spatial filtering and cleaning process.
    Takes an IFU cube, applies a spatial filter to each slice, and then runs the CLEAN algorithm on each slice.
    This ideally should be run in parallel, and primarily sets up/calls _filter_and_clean_slice to do the actual filtering and cleaning.
    This function is called by make_mock_datacubes.py to filter and clean the IFU cube.
    '''
    ifu = ifu + noise_flux

    '''
    Quick Check to make sure the noise flux is not changing dramatically with the filtering
    Sets the noise flux to clean to based on the filtered noise level.
    '''
    nx,ny,nz=np.shape(ifu)
    nu = np.linspace(instrument.obs_freq_range[0].in_units('1/s'),instrument.obs_freq_range[0].in_units('1/s'),nz)
    
    uv = np.fft.fftshift( np.fft.fft2(noise_flux[:,:,0].value) )


    ux,uy = np.shape(uv)
    sampling_frequency = 1./(args.base_spatial_res * arcsec_to_rad) #in 1/rad
    x = (np.arange(ux) - (ux-1)/2) * sampling_frequency / ux ###Something wrong with this?? Probably okay, maybe a factor of 2?
    y = (np.arange(uy) - (uy-1)/2) * sampling_frequency / uy
    X, Y = np.meshgrid(x, y,  indexing='ij')

    spatial_frequencies = np.sqrt(np.power(X,2) + np.power(Y,2)) #Redo this...

    lmbda = c / np.mean(nu) #wavelength in km
    max_observable_angle = lmbda / instrument.min_baseline #1/rad
    min_spatial_freq_lmbda = 1. / max_observable_angle # 1/rad

    if args.high_pass_filter_type is None: #This will introduce ringing artifacts you probably don't want...
        mask = (spatial_frequencies >= min_spatial_freq_lmbda).astype(int)
    elif args.high_pass_filter_type=='gaussian': #Works okay, introduces ringing artifacts and cuts off a lot of diffuse emission
        mask = gaussian_high_pass(spatial_frequencies, min_spatial_freq_lmbda)
    elif args.high_pass_filter_type=='butterworth': #Doesn't really work? Maybe needs a higher order (>10??)
        mask = butterworth_high_pass(spatial_frequencies, min_spatial_freq_lmbda,order=2)

    beam = get_beam_model(np.shape(uv),instrument,args,spatial_frequencies,min_spatial_freq_lmbda,order=2)

    uv = np.multiply(uv,mask)    
  #  uv = enforce_hermitian_symmetry(uv)

    filtered_noise = np.fft.ifft2( np.fft.ifftshift(uv) )

    filtered_noise_level = np.std(filtered_noise)

    print("Noise level was=",f"{np.std(noise_flux):e}")
    print("Filtered noise level=",f"{filtered_noise_level:e}")
    max_residual_limit = args.clean_sigma*filtered_noise_level #args.sigma_noise_level * filtered_noise_level # Clean to the noise limit of the filtered data, not the input noise cube
    '''
    End of Noise Test
    '''

    nx,ny,nz = np.shape(ifu)

    ifu_units = ifu[0,0,0].units
    
    mask=None

    nu = np.linspace(instrument.obs_freq_range[0].in_units('1/s'),instrument.obs_freq_range[0].in_units('1/s'),nz)

    ifu_with_residuals = np.copy(ifu)
    
    if args.nthreads is None: args.nthreads = multiprocessing.cpu_count()-1
    elif args.nthreads > multiprocessing.cpu_count()-1: args.nthreads = multiprocessing.cpu_count()-1

    if args.nthreads>1:   
        #Run _filter_and_clean_slice in parallel 
        print("Filtering and cleaning in parallel...")
        ux=nx
        uy=ny

        clean_beam = get_clean_beam([2*nx,2*ny,nz],args,instrument) #Define the clean beam (gaussian) with 2x fov for easy fitting to final image

        filter_and_clean_slice = partial(_filter_and_clean_slice, args=args,instrument=instrument,max_residual_limit=max_residual_limit,beam=None,mask=None,clean_beam=clean_beam,nu=nu,clean_mask=clean_mask)
        print("Working with",args.nthreads,"threads...")
        tParr=time.time()
        if clean_mask is not None: results = Parallel(n_jobs=args.nthreads)(delayed(filter_and_clean_slice)(ifu_slice=ifu[:,:,itr],i=itr,clean_mask=clean_mask[:,:,itr]) for itr in range(0,nz))#nz))
        else: results = Parallel(n_jobs=args.nthreads)(delayed(filter_and_clean_slice)(ifu_slice=ifu[:,:,itr],i=itr) for itr  in range(0,nz))#nz))

        results=np.array(results)
        clean_image = np.transpose(results[:,0,:,:], (1,2,0))
        clean_image_with_residuals = np.transpose(results[:,1,:,:], (1,2,0))

        print("Time to clean in parallel=",time.time()-tParr)

        ifu = np.array(clean_image)
        ifu_with_residuals = np.array(clean_image_with_residuals)
    else:
      #Not tested recently. Not recommended to run this linearly if it can be avoided...
      pbar = TqdmProgressBar("Filtering spatial frequencies",nz,position=0)
      for i in range(nz):
        uv = np.fft.fftshift( np.fft.fft2(ifu[:,:,i]) )


        if True:
            ux,uy = np.shape(uv)
            print("nx,ny,nz=",nx,ny,nz)
            print("ux,uy=",ux,uy)
            sampling_frequency = 1./(args.base_spatial_res * arcsec_to_rad) #in 1/rad
            x = (np.arange(ux) - (ux-1)/2) * sampling_frequency / ux ###Something wrong with this?? Probably okay, maybe a factor of 2?
            y = (np.arange(uy) - (uy-1)/2) * sampling_frequency / uy
            X, Y = np.meshgrid(x, y,  indexing='ij')

            spatial_frequencies = np.sqrt(np.power(X,2) + np.power(Y,2)) #Redo this...
            #print("Shape of spatial frequencies=",np.shape(spatial_frequencies))
            print("spatial frequencies range from",np.min(spatial_frequencies),np.max(spatial_frequencies))
            print("min nonzero spatial frequency",np.min(spatial_frequencies[np.where(spatial_frequencies>0)]))
            print("min insturment spatial frequency=",instrument.min_spatial_freq)

            lmbda = c / nu[i] #wavelength in km
            max_observable_angle = lmbda / instrument.min_baseline #1/rad
            min_spatial_freq_lmbda = 1. / max_observable_angle # 1/rad

            if args.high_pass_filter_type is None: #This will introduce ringing artifacts you probably don't want...
                mask = (spatial_frequencies >= min_spatial_freq_lmbda).astype(int)
            elif args.high_pass_filter_type=='gaussian': #Works okay, introduces ringing artifacts and cuts off a lot of diffuse emission
                mask = gaussian_high_pass(spatial_frequencies, min_spatial_freq_lmbda)
            elif args.high_pass_filter_type=='butterworth': #Doesn't really work? Maybe needs a higher order (>10??)
                mask = butterworth_high_pass(spatial_frequencies, min_spatial_freq_lmbda,order)

            print("Max freq =", 1. / (instrument.obs_spatial_res*arcsec_to_rad))
            #lp_mask = gaussian_low_pass(spatial_frequencies, 1. / (instrument.obs_spatial_res*arcsec_to_rad))
            print("Shape of mask=",np.shape(mask))
            print("Size of UV plane=",np.size(uv))
            print("Size of mask=",np.size(np.where(mask>0.5)[0]))


        uv = np.multiply(uv,mask)



        filtered_image = np.fft.ifft2( np.fft.ifftshift(uv) )
        if np.max(np.abs(filtered_image.imag)) > 1e-10:
            print(f"Max imaginary component: {np.max(np.abs(filtered_image.imag))}")
    
        dirty_slice = ( np.abs(filtered_image) * np.exp(1j * np.angle(filtered_image)) )#.real

        beam = get_beam_model(np.shape(ifu),instrument,args,spatial_frequencies,min_spatial_freq_lmbda,order=order)

        plot_beam_figures=False
        if i==0: plot_beam_figures=True
        cleaned_slice,cleaned_slice_with_residuals = custom_clean(dirty_slice,beam,args,instrument,max_residual_limit = max_residual_limit,plot_beam_figures=plot_beam_figures)
 
        ifu[:,:,i] = np.abs(cleaned_slice) * np.exp(1j * np.abs(cleaned_slice.imag))
        ifu_with_residuals[:,:,i] = np.abs(cleaned_slice_with_residuals) * np.exp(1j * np.abs(cleaned_slice_with_residuals.imag))


        plt.figure()
        plt.imshow(np.abs(dirty_slice),norm=LogNorm(vmin=np.min(np.abs(dirty_slice[np.abs(dirty_slice>0)])) / 10))
        plt.savefig("./slices/dirty_slice_"+str(i)+".png")
        plt.close()
        plt.figure()
        plt.imshow(np.abs(cleaned_slice),norm=LogNorm(vmin=np.min(np.abs(cleaned_slice[np.abs(cleaned_slice>0)])) / 10))
        plt.savefig("./slices/cleaned_slice_"+str(i)+".png")
        plt.close()
        pbar.update(i)
      pbar.update(nz)
      pbar.finish()



    print("Cleaned ifu=",ifu[0,0,0])
    print("residual ifu=",ifu_with_residuals[0,0,0])
    return  ifu*ifu_units, ifu_with_residuals*ifu_units#, unfiltered_ifu




#The Functions below aren't used anymore, but are kept for reference and possible future use.
def custom_clean(dirty_slice,beam,args,instrument,max_residual_limit = None, max_iterations=None,plot_beam_figures=False,tid=0,clean_beam=None):
    '''
    This is a largely defunct function for the CLEAN algorithm used for visualization purposes.
    Use the cythonized version in custom_clean.py instead.
    '''

    itr=0
    residual = np.copy(dirty_slice)
    clean_map = np.copy(dirty_slice)*0.0
    nx,ny = np.shape(dirty_slice)
    

    if max_residual_limit is None:
        max_residual_limit = np.min(np.abs(dirty_slice[np.abs(dirty_slice)>0])) * 10. ##Change? Especially when noise model added in!!

    if clean_beam is None: clean_beam = get_clean_beam(np.shape(beam),args,instrument)

    if tid==0 and plot_beam_figures:
        plt.figure()
        vmin = np.min(np.abs(beam[np.abs(beam>0)])) / 10.
        #vmin = np.max(np.abs(beam))/1e6
        plt.imshow(np.abs(beam),norm=LogNorm(vmin=vmin))
        plt.grid(False)  # Ensures explicit gridlines are off
        plt.savefig("dirty_beam.png")
        plt.close()

        plt.figure()
        vmin = np.min(np.abs(beam[np.abs(beam>0)])) / 10.
        #vmin = np.max(np.abs(beam))/1e6
        plt.imshow(np.abs(beam),norm=LogNorm(vmin=vmin))
        plt.grid(False)  # Ensures explicit gridlines are off
        plt.xlim([nx//6,2*nx//6])
        plt.ylim([ny//6,2*ny//6])
        plt.savefig("dirty_beam_zoom.png")
        plt.close()

        plt.figure()
        vmin = np.max(clean_beam)/10000.
        plt.imshow(np.abs(clean_beam),norm=LogNorm(vmin=vmin))
        plt.grid(False)  # Ensures explicit gridlines are off
        plt.savefig("clean_beam.png")
        plt.close()


    if max_iterations is None:
        max_iterations = int(nx*ny)

    convolved_beam = np.copy(dirty_slice)


    vmax = np.max(np.abs(residual))
    vmin = vmax*1e-6
    last_residuals = np.sum(np.abs(residual))
    while itr<max_iterations:
        max_residual = np.max(residual.real)
        if max_residual_limit is not None and max_residual <= max_residual_limit: break

        max_idx = np.argmax(residual.real)
        max_idx = np.unravel_index(max_idx,residual.shape)

        xshift = max_idx[0] - nx//2
        yshift = max_idx[1] - ny//2

        b_min_x = int(nx//2-xshift-1)
        b_max_x = int(3*nx//2-xshift-1)
        b_min_y = int(ny//2-yshift-1)
        b_max_y = int(3*ny//2-yshift-1)

        t1=time.time()

#
        residual = residual - args.clean_gain * residual[max_idx[0],max_idx[1]] * beam[b_min_x:b_max_x,b_min_y:b_max_y]


        clean_map[max_idx[0],max_idx[1]] = clean_map[max_idx[0],max_idx[1]] + args.clean_gain * residual[max_idx[0],max_idx[1]]


        if False:#itr%100==0 or np.sum(np.abs(next_residual))>last_residuals) and plot_beam_figures:
            plt.figure(figsize=(20.*1.33,8))
            plt.subplot(141)
            residual_plot = ( np.abs(residual) * np.exp(1j * np.angle(residual)) ).real
            plt.imshow(np.abs(residual),norm=LogNorm(vmin=vmin,vmax=vmax))
            plt.colorbar()
            plt.xticks([])
            plt.yticks([])
            plt.subplot(142)
            plt.imshow(np.abs(convolved_beam),norm=LogNorm())
            plt.colorbar()
            plt.xticks([])
            plt.yticks([])
            plt.subplot(143)
            plt.imshow(np.abs(clean_map),norm=LogNorm(vmin=vmin,vmax=vmax))
            plt.colorbar()
            plt.xticks([])
            plt.yticks([])
            plt.subplot(144)
            plt.imshow(np.abs(clean_map+residual),norm=LogNorm(vmin=vmin,vmax=vmax))
            plt.colorbar()
            plt.xticks([])
            plt.yticks([])
            plt.savefig("./clean/residuals"+str(itr)+".png",bbox_inches='tight')
            plt.close()

        itr+=1


    clean_map = scipy.ndimage.gaussian_filter(clean_map, sigma=instrument.obs_spatial_res/args.base_spatial_res / 2 / np.sqrt(2*np.log(2))  , axes=[0,1])

    return clean_map, clean_map + residual


def complex_clean(dirty_slice,beam,args,instrument,clean_beam,max_residual_limit = None, max_iterations=None,plot_beam_figures=False,tid=0,):
    '''
    This is a largely defunct function for the CLEAN algorithm used for visualization purposes.
    Use the cythonized version in custom_clean.py instead.
    '''
    itr=0
    residual = np.copy(dirty_slice)
    nx,ny = np.shape(dirty_slice)
    clean_map = np.copy(dirty_slice) * 0


    if tid == 0 and plot_beam_figures:
        plt.figure()
        vmin = np.min(np.abs(beam[np.abs(beam>0)])) / 10.
        #vmin = np.max(np.abs(beam))/1e6
        plt.imshow(np.abs(beam),norm=LogNorm(vmin=vmin))
        plt.grid(False)  # Ensures explicit gridlines are off
        plt.savefig("dirty_beam.png")
        plt.close()

        plt.figure()
        vmin = np.min(np.abs(beam[np.abs(beam>0)])) / 10.
        #vmin = np.max(np.abs(beam))/1e6
        plt.imshow(np.abs(beam),norm=LogNorm(vmin=vmin))
        plt.grid(False)  # Ensures explicit gridlines are off
        plt.xlim([nx//4,3*nx//4])
        plt.ylim([ny//4,3*ny//4])
        plt.savefig("dirty_beam_zoom.png")
        plt.close()

        plt.figure()
        vmin = np.max(clean_beam)/10000.
        plt.imshow(np.abs(clean_beam),norm=LogNorm(vmin=vmin))
        plt.grid(False)  # Ensures explicit gridlines are off
        plt.savefig("clean_beam.png")
        plt.close()


    if max_iterations is None:
        max_iterations = int(nx*ny)

    convolved_beam = np.copy(dirty_slice)


    vmax = np.max(np.abs(residual))
    vmin = vmax*1e-6
    while itr<max_iterations:

        real_residual = (np.abs(residual) * np.exp(1j * np.angle(residual)) ).real

        max_residual = np.max(real_residual)
        if max_residual_limit is not None and max_residual <= max_residual_limit: break
        max_idx = np.argmax(real_residual)
        #max_idx = np.argmax(residual)
        max_idx = np.unravel_index(max_idx,residual.shape)
        if max_residual_limit is not None and max_residual <= max_residual_limit: break

        xshift = max_idx[0] - nx//2
        yshift = max_idx[1] - ny//2


        b_min_x = int(nx//2-xshift)
        b_max_x = int(3*nx//2-xshift)
        b_min_y = int(ny//2-yshift)
        b_max_y = int(3*ny//2-yshift)

        if tid==0: print('max_residual=',f"{residual[max_idx[0],max_idx[1]]:e}","at itr=",itr,"for limit",f"{max_residual_limit:e}")

        clean_map = clean_map + args.clean_gain * max_residual * clean_beam[b_min_x:b_max_x,b_min_y:b_max_y]#residual[max_idx[0],max_idx[1]]

        residual = residual - args.clean_gain * max_residual * beam[b_min_x:b_max_x,b_min_y:b_max_y]


        if (tid+1)%args.nthreads==0 and (itr%25==0):
            convolved_beam = args.clean_gain * residual[max_idx[0],max_idx[1]] * beam[b_min_x:b_max_x,b_min_y:b_max_y]
            print("SIZE OF RESIDUAL=",np.shape(residual))
            #smoothed_map = scipy.ndimage.gaussian_filter(clean_map, sigma=instrument.obs_spatial_res/args.base_spatial_res , axes=[0,1])
            plt.figure(figsize=(20.*1.33,8))
            plt.subplot(141)
            plt.imshow(np.abs(residual),norm=LogNorm(vmin=vmin,vmax=vmax))
            plt.scatter(max_idx[1],max_idx[0],marker='x',color='red',s=5)
            plt.colorbar()
            plt.xticks([])
            plt.yticks([])
            plt.subplot(142)
            plt.imshow(np.abs(convolved_beam),norm=LogNorm())
            plt.scatter(max_idx[1],max_idx[0],marker='x',color='red',s=5)
            plt.colorbar()
            plt.xticks([])
            plt.yticks([])
            plt.subplot(143)
            plt.imshow(np.abs(clean_map),norm=LogNorm(vmin=vmin,vmax=vmax))
            plt.colorbar()
            plt.xticks([])
            plt.yticks([])
            plt.subplot(144)
            plt.imshow(np.abs(np.add(clean_map,residual)),norm=LogNorm(vmin=vmin,vmax=vmax))
            plt.colorbar()
            plt.xticks([])
            plt.yticks([])
            plt.savefig("./clean/residuals_slice"+str(tid)+"_"+str(itr)+".png",bbox_inches='tight')
            plt.close()

        itr+=1

    clean_map = scipy.ndimage.gaussian_filter(clean_map, sigma=instrument.obs_spatial_res/args.base_spatial_res / 2 / np.sqrt(2*np.log(2)) , axes=[0,1])
    return clean_map, clean_map + residual
