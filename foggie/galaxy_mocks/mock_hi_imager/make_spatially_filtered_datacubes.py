
import numpy as np
import h5py
from functools import partial
from joblib import Parallel, delayed
import multiprocessing
import time
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import scipy
import sys

from foggie.galaxy_mocks.mock_hi_imager.HICubeHeader import * #Constants
from foggie.galaxy_mocks.mock_hi_imager.utils_hi_datacube import apply_gaussian_smoothing

from foggie.galaxy_mocks.mock_hi_imager.utils_hi_datacube import gaussian_high_pass
from foggie.galaxy_mocks.mock_hi_imager.utils_hi_datacube import butterworth_high_pass
from foggie.galaxy_mocks.mock_hi_imager.utils_hi_datacube import enforce_hermitian_symmetry
from foggie.galaxy_mocks.mock_hi_imager.utils_hi_datacube import TqdmProgressBar

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

    uv = enforce_hermitian_symmetry(uv)


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

    beam2 = np.zeros((nx,ny))
    beam2[nx//2,ny//2] = 1.
    beam2= scipy.ndimage.gaussian_filter(beam2, sigma=sigma , axes=[0,1])

    beam2= beam2 / np.max(beam2)
    test_beam = beam/np.abs(np.max(beam))
    print("\n\nIntegral of beam2=",np.sum(beam2))
    print("Integral of Clean Beam=",np.sum(test_beam))
    
    return beam2


def _filter_and_clean_slice(args,instrument,max_residual_limit,beam,mask,clean_beam,ifu_slice,i,clean_mask=None,noise_slice=None):
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


        pre_filt_slice_flux = np.sum(ifu_slice)

        nx,ny = np.shape(ifu_slice)
        new_slice = np.zeros((2*nx,2*ny))
        new_slice[nx//2:nx//2+nx,ny//2:ny//2+ny] = ifu_slice.value

        uv = np.fft.fftshift( np.fft.fft2(new_slice) )

        ux,uy = np.shape(uv)

        sampling_frequency = 1./(args.base_spatial_res * arcsec_to_rad) #in 1/rad
        x = (np.arange(ux) - (ux-1)/2) * sampling_frequency / ux 
        y = (np.arange(uy) - (uy-1)/2) * sampling_frequency / uy
        X, Y = np.meshgrid(x, y,  indexing='ij')

        spatial_frequencies = np.sqrt(np.power(X,2) + np.power(Y,2)) 
        nz = args.obs_channels
        nu = np.linspace(args.base_freq_range[1].in_units('1/s'),args.base_freq_range[0].in_units('1/s'),nz)
        lmbda = c / nu[i] #wavelength in km
        max_observable_angle = lmbda.in_units('km').v  / instrument.min_baseline #1/rad
        min_spatial_freq_lmbda = 1. / max_observable_angle # 1/rad
        print("MIN SPATIAL FREQ LAMBDA=",min_spatial_freq_lmbda)
        print("Max spatial frequency in image=",np.max(spatial_frequencies)/2.)
        if args.high_pass_filter_type is None: #This will introduce ringing artifacts you probably don't want...
            mask = (spatial_frequencies >= min_spatial_freq_lmbda).astype(int)
        elif args.high_pass_filter_type=='gaussian': #Works okay, introduces ringing artifacts and cuts off a lot of diffuse emission
            mask = gaussian_high_pass(spatial_frequencies, min_spatial_freq_lmbda)
        elif args.high_pass_filter_type=='butterworth': #Doesn't really work? Maybe needs a higher order (>10??)
            mask = butterworth_high_pass(spatial_frequencies, min_spatial_freq_lmbda,order=2)

        beam = get_beam_model(np.shape(new_slice),instrument,args,spatial_frequencies,min_spatial_freq_lmbda,order=2)

        uv = np.multiply(uv,mask)

        uv = enforce_hermitian_symmetry(uv)

        filtered_image = np.fft.ifft2( np.fft.ifftshift(uv) )

        if False:
            if i==200:
                plt.figure()
                plt.imshow(filtered_image[nx//2:nx//2+nx,ny//2:ny//2+ny].real,norm=LogNorm())
                plt.colorbar()
                plt.show()

            filtered_image[nx//2:nx//2+nx,ny//2:ny//2+ny] = filtered_image[nx//2:nx//2+nx,ny//2:ny//2+ny] + noise_slice
            if i==200:
                plt.figure()
                plt.imshow(filtered_image[nx//2:nx//2+nx,ny//2:ny//2+ny].real,norm=LogNorm())
                plt.colorbar()
                plt.show()
        filt_slice_flux = np.sum(filtered_image)
        #dirty_slice = filtered_image
        if not clean_complex:
           dirty_slice =  (np.abs(filtered_image) * np.exp(1j * np.angle(filtered_image)) ).real
           beam = ( np.abs(beam) * np.exp(1j * np.angle(beam)) ).real
        else:
            dirty_slice = filtered_image
            dirty_slice.imag=0
            beam.imag=0



        residual_correction= 1.0


#        if True:
        save_tmp_slices = False #make arg?
        try:
            #Load cleaned slice from temp files
            hf_cs = h5py.File("./outputs/tmp_slices/tmp_cleaned_slice_"+args.halo+args.mock_suffix+"_"+args.survey+"_slice"+str(i)+".h5",'r')
            cleaned_slice = hf_cs["cleaned_slice"][...]
            residuals = hf_cs["residuals"][...]
            hf_cs.close()
            cleaned_slice_with_residuals = cleaned_slice + residuals
            print("Slice",i,"loaded cleaned slice from file")
        except:
            if args.max_iterations is None:
                args.max_iterations = int(ux*uy)
            tclean = time.time()
            if clean_complex:
                  print("cleaning complex...")
                  if clean_mask is not None:
                    print("\n\nDoing masked clean!\n\n")
                    cleaned_slice, residuals = cy_complex_masked_clean(dirty_slice[ux//4:3*ux//4,uy//4:3*uy//4], beam, clean_beam,clean_mask,n_masked_pixels, max_residual_limit, i, args.clean_gain, args.max_iterations)
                  else:
                    cleaned_slice, residuals, finished = cy_complex_clean(dirty_slice[ux//4:3*ux//4,uy//4:3*uy//4], beam, clean_beam, max_residual_limit, i, args.clean_gain, args.max_iterations)
            elif clean_mask is not None:
                cleaned_slice, residuals = cy_masked_clean(dirty_slice[ux//4:3*ux//4,uy//4:3*uy//4], beam, clean_beam,clean_mask,n_masked_pixels, max_residual_limit, i, args.clean_gain, args.max_iterations)
            print("Slice",i,"ran clean in",time.time()-tclean,"seconds")
            clean_comp_power = np.sum(cleaned_slice.real)
            residual_power = np.sum(residuals.real)
            cleaned_slice = cleaned_slice
            cleaned_slice_with_residuals = cleaned_slice + residual_correction*residuals
            cleaned_and_residual_power = np.sum(cleaned_slice_with_residuals.real)
            if save_tmp_slices:
                hf_cs = h5py.File("./outputs/tmp_slices/tmp_cleaned_slice_"+args.halo+args.mock_suffix+"_"+args.survey+"_slice"+str(i)+".h5",'w')
                hf_cs.create_dataset("cleaned_slice",data=cleaned_slice)
                hf_cs.create_dataset("residuals",data=residuals)
                hf_cs.close()

        if clean_complex:
           dirty_image = dirty_slice.real
           clean_image = cleaned_slice.real
           clean_image_with_residuals = cleaned_slice_with_residuals.real
        else:
            dirty_image = dirty_slice.real
            clean_image = cleaned_slice.real 
            clean_image_with_residuals = cleaned_slice_with_residuals.real

        if False:#i%args.nthreads==0:
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

        post_clean_slice_flux = np.sum(clean_image_with_residuals.real)
        #print("For slice",i,": pre-filter flux=",f"{pre_filt_slice_flux:e}","filt flux=",f"{filt_slice_flux:e}","post-clean flux=",f"{post_clean_slice_flux:e}")
        #print("For slice",i,"Fraction of flux retained after filtering and cleaning=",f"{post_clean_slice_flux/pre_filt_slice_flux:e}")
        #print("For slice",i,"clean component power=",f"{clean_comp_power:e}","residual power=",f"{residual_power:e}","combined=",f"{cleaned_and_residual_power:e}")

        return clean_image.real, clean_image_with_residuals.real#, np.abs(dirty_slice[ux//4:3*ux//4,uy//4:3*uy//4])



def filter_spatial_frequencies_slicewise(ifu, noise_flux, instrument, args, order=2, max_residual_limit = 1, clean_mask=None):
    '''
    This is the core function of the spatial filtering and cleaning process.
    Takes an IFU cube, applies a spatial filter to each slice, and then runs the CLEAN algorithm on each slice.
    This ideally should be run in parallel, and primarily sets up/calls _filter_and_clean_slice to do the actual filtering and cleaning.
    This function is called by make_mock_datacubes.py to filter and clean the IFU cube.
    '''
    ideal_ifu=np.copy(ifu)


    nx,ny,nz=np.shape(ifu)
    nu = np.linspace(instrument.obs_freq_range[0].in_units('1/s'),instrument.obs_freq_range[0].in_units('1/s'),nz)
    
    uv = np.fft.fftshift( np.fft.fft2(noise_flux[:,:,0].value) )

    ####fourier_image =  np.abs(uv)

    ux,uy = np.shape(uv)
    sampling_frequency = 1./(args.base_spatial_res * arcsec_to_rad) #in 1/rad
    x = (np.arange(ux) - (ux-1)/2) * sampling_frequency / ux ###Something wrong with this?? Probably okay, maybe a factor of 2?
    y = (np.arange(uy) - (uy-1)/2) * sampling_frequency / uy
    X, Y = np.meshgrid(x, y,  indexing='ij')

    spatial_frequencies = np.sqrt(np.power(X,2) + np.power(Y,2)) #Redo this...


    lmbda = (c / np.mean(nu))#wavelength in km
    max_observable_angle = lmbda.in_units('km').v  / instrument.min_baseline #1/rad
    min_spatial_freq_lmbda = 1. / max_observable_angle # 1/rad

    for i in range(nz):
        filt_ideal_ifu = np.copy(ideal_ifu[:,:,i])*0
        ideal_uv = np.fft.fftshift( np.fft.fft2(ideal_ifu[:,:,i]) )
        mask = gaussian_high_pass(spatial_frequencies, min_spatial_freq_lmbda)
        ideal_uv = np.multiply(ideal_uv,mask)
    #transform back to image space
        filt_ideal_ifu = np.fft.ifft2( np.fft.ifftshift(ideal_uv) ).real
        #print("IFU FLUX =",f"{np.sum(ideal_ifu[:,:,i]):e}")
        #print("Ideal IFU FLUX after filtering=",f"{np.sum(filt_ideal_ifu):e}")
        if np.sum(ideal_ifu[:,:,i]) < np.sum(filt_ideal_ifu):
            print("\n\n\nWARNING, FILTERING INCREASED THE TOTAL FLUX?!!! Check the beam model and filtering code, something is wrong!\n\n\n")
    


    if args.high_pass_filter_type is None: #This will introduce ringing artifacts you probably don't want...
        mask = (spatial_frequencies >= min_spatial_freq_lmbda).astype(int)
    elif args.high_pass_filter_type=='gaussian': #Works okay, introduces ringing artifacts and cuts off a lot of diffuse emission
        mask = gaussian_high_pass(spatial_frequencies, min_spatial_freq_lmbda)
    elif args.high_pass_filter_type=='butterworth': #Doesn't really work? Maybe needs a higher order (>10??)
        mask = butterworth_high_pass(spatial_frequencies, min_spatial_freq_lmbda,order=2)

    beam = get_beam_model(np.shape(uv),instrument,args,spatial_frequencies,min_spatial_freq_lmbda,order=2)

    uv = np.multiply(uv,mask)    
    uv = enforce_hermitian_symmetry(uv)

    filtered_noise = np.fft.ifft2( np.fft.ifftshift(uv) )

    filtered_noise_level = np.std(filtered_noise)


    #Calcualte filtered noise
    filtered_noise_flux = np.copy(noise_flux)*0
    for i in range(0,nz):
        uv = np.fft.fftshift( np.fft.fft2(noise_flux[:,:,i].value) )
        lmbda = (c / nu[i])#wavelength in km
        max_observable_angle = lmbda.in_units('km').v  / instrument.min_baseline #1/rad
        min_spatial_freq_lmbda = 1. / max_observable_angle # 1/rad
        if args.high_pass_filter_type is None: #This will introduce ringing artifacts you probably don't want...
            mask = (spatial_frequencies >= min_spatial_freq_lmbda).astype(int)
        elif args.high_pass_filter_type=='gaussian': #Works okay, introduces ringing artifacts and cuts off a lot of diffuse emission
            mask = gaussian_high_pass(spatial_frequencies, min_spatial_freq_lmbda)
        elif args.high_pass_filter_type=='butterworth': #Doesn't really work? Maybe needs a higher order (>10??)
            mask = butterworth_high_pass(spatial_frequencies, min_spatial_freq_lmbda,order=2)
        uv = np.multiply(uv,mask)
        filtered_noise_flux[:,:,i] = np.fft.ifft2( np.fft.ifftshift(uv) )

        if not args.do_noiseless_clean:
            ifu[:,:,i] = ifu[:,:,i] + noise_flux[:,:,i].value * np.std(noise_flux[:,:,i]) / np.std(filtered_noise_flux[:,:,i])



        filtered_noise_flux[:,:,i] = filtered_noise_flux[:,:,i] * np.std(noise_flux[:,:,i]) / np.std(filtered_noise_flux[:,:,i]) #Scale the filtered noise to have the same noise level as the input noise cube, since filtering can change the noise level


    print("Max Residual limit was=",f"{max_residual_limit:e}")
    print("Noise level was=",f"{np.std(noise_flux):e}")
    print("Filtered noise level=",f"{filtered_noise_level:e}")
    noise_level = np.std(noise_flux).in_units('s/cm**2').v
    max_residual_limit = args.clean_sigma*noise_level#filtered_noise_level #args.sigma_noise_level * filtered_noise_level # Clean to the noise limit of the filtered data, not the input noise cube

    nx,ny,nz = np.shape(ifu)

    ifu_units = ifu[0,0,0].units

    #ifu = ifu * 1e24
    
    mask=None

    nu = np.linspace(instrument.obs_freq_range[0].in_units('1/s'),instrument.obs_freq_range[0].in_units('1/s'),nz)
    slice_frequencies = np.zeros((nz,1000))

   # dirty_image = np.copy(ifu)
    #fourier_image = np.copy(ifu)
    #filtered_fourier_image = np.copy(fourier_image)
    ifu_with_residuals = np.copy(ifu)
    

    if args.nthreads is None: args.nthreads = multiprocessing.cpu_count()-1
    elif args.nthreads > multiprocessing.cpu_count()-1: args.nthreads = multiprocessing.cpu_count()-1

    if True:    
        print("Filtering and cleaning in parallel...")
        ux=nx
        uy=ny

        clean_beam = get_clean_beam([2*nx,2*ny,nz],args,instrument)

        filter_and_clean_slice = partial(_filter_and_clean_slice, args=args,instrument=instrument,max_residual_limit=max_residual_limit,beam=None,mask=None,clean_beam=clean_beam,clean_mask=clean_mask)
        print("Working with",args.nthreads,"threads...")
        tParr=time.time()
        if clean_mask is not None: results = Parallel(n_jobs=args.nthreads)(delayed(filter_and_clean_slice)(ifu_slice=ifu[:,:,itr],i=itr,clean_mask=clean_mask[:,:,itr],noise_slice = filtered_noise_flux[:,:,i]) for itr in range(0,nz))#nz))
        else: results = Parallel(n_jobs=args.nthreads)(delayed(filter_and_clean_slice)(ifu_slice=ifu[:,:,itr],i=itr,noise_slice = filtered_noise_flux[:,:,i]) for itr  in range(0,nz))#nz))

        print(np.shape(results))
        results=np.array(results)
        clean_image = np.transpose(results[:,0,:,:], (1,2,0))
        #dirty_image = np.transpose(results[:,2,:,:], (1,2,0))
        clean_image_with_residuals = np.transpose(results[:,1,:,:], (1,2,0))

        print(np.shape(clean_image))
        print("Time to clean in parallel=",time.time()-tParr)

        ifu = np.array(clean_image)

        ifu_with_residuals = np.array(clean_image_with_residuals)
        if args.do_noiseless_clean:
            print("Adding noise flux AFTER cleaning...")
            ifu_with_residuals = ifu_with_residuals + filtered_noise_flux #noise_flux * filtered_noise_level / np.std(noise_flux) 

    print("Cleaned ifu=",ifu[0,0,0])
    print("residual ifu=",ifu_with_residuals[0,0,0])
    return  ifu*ifu_units, ifu_with_residuals*ifu_units#, unfiltered_ifu