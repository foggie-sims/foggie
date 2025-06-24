import scipy
import numpy as np
import unyt as u
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from foggie.galaxy_mocks.mock_hi_imager.HICubeHeader import * #Constants
from foggie.galaxy_mocks.mock_hi_imager.radio_telescopes import radio_telescope
from foggie.galaxy_mocks.mock_hi_imager.make_spatially_filtered_datacubes import filter_spatial_frequencies_slicewise
from foggie.galaxy_mocks.mock_hi_imager.utils_hi_datacube import apply_gaussian_smoothing


'''
Functions used to convert ideal HI datacubes generated in make_ideal_datacubes.py to mock datacubes.
Logic starts at get_mock_hi_datacube() and calls various functions in this file to create the smoothed+noisy cubes.

If desired, functions in make_spatially_filtered_datacubes.py will be called to spatially filter the cubes to mimic interferometry.

Author: Cameron Trapp
Last updated 06/17/2025
'''




def apply_primary_beam_correction(ifu, instrument, args):
    '''
    Accounts for the primary beam drop off based on the survey parameters
    Primary beam is modeled as a Gaussian with a FWHM defined by the instrument.
    This FWHM typically sets the FOV in observation.
    if instrument.primary_beam_FWHM_deg is not defined, it's assumed to be negligible (i.e. a mosaic survey)
    '''
    primary_beam = None
    if instrument.primary_beam_FWHM_deg is not None:
        nx,ny,ns = np.shape(ifu)
        y,x = np.indices((nx,ny))
        pixel_radii_deg = np.sqrt( np.power((x-nx//2), 2) + np.power((y-ny//2), 2) ) * args.base_spatial_res / 3600. #in degrees
        pb_sigma = instrument.primary_beam_FWHM_deg / (2.0 * np.sqrt(2.0 * np.log(2.0))) #in degrees
        primary_beam = np.exp(-np.power(pixel_radii_deg, 2) / (2*(pb_sigma)**2)) # peaks at one
        ifu = np.multiply(ifu,primary_beam[:,:,np.newaxis])

    return ifu,primary_beam


def add_noise(ifu, args):
    '''
    Adds a gaussian noise profile to the image.
    Assumes that the noise is uncorrleated with the signal and keeps the noise cube separate from the ifu cube
    Noise will need to be renormalized on smoothing.
    The noise level is set by args.min_column_density divided by args.sigma_noise_level
    '''
    nx,ny,nz=np.shape(ifu)
    nu = np.linspace(args.base_freq_range[1].in_units('1/s'),args.base_freq_range[0].in_units('1/s'),nz)
    dnu = ( np.max(nu)-np.min(nu) ) / np.size(nu)
    if args.min_column_density>0:
        noise_level = args.min_column_density / u.cm / u.cm / dnu / args.sigma_noise_level# 3. /3. # / nz 3-sigma detection?

    N = np.size(ifu)

    try:
        #If restarting from a previous crash/run will ensure you are using the same noise profile.
        hf_nc = h5py.File("./outputs/tmp_slices/tmp_noise_cube_"+args.halo+args.mock_suffix+"_"+args.survey+".h5",'r')
        noise_flux = hf_nc['noise_flux'][...] * noise_level.units
        hf_nc.close()
        print("Using noise flux from previous run!")
    except:
        noise_flux = np.random.normal(0.0,noise_level.value, N).reshape(np.shape(ifu)) #Gaussian
        hf_nc = h5py.File("./outputs/tmp_slices/tmp_noise_cube_"+args.halo+args.mock_suffix+"_"+args.survey+".h5",'w')
        hf_nc.create_dataset('noise_flux', data=noise_flux)
        hf_nc.close()
        noise_flux = noise_flux * noise_level.units

    print("Noise level is:",f"{noise_level*dnu:e}","(",f"{noise_level:e}",")")
    print("Mean noise flux is:",f"{np.mean(noise_flux*dnu):e}","(",f"{np.mean(noise_flux):e}",")")
    print("Stdv noise flux is:",f"{np.std(noise_flux*dnu):e}","(",f"{np.std(noise_flux):e}",")")

    return noise_flux, noise_level





def get_mock_hi_datacube(args, ds, ifu, clean_mask):
    '''
    This is the core function for the creating the mock datacubes.
    This calls the functions in this file to add noise and smooth the cube.
    Additionally, this calls functions in make_spatially_filtered_datacubes.py to spatially filter the data and run the CLEAN algorithm if desired
    '''
    instrument = radio_telescope(args) #load instrument parameters

    ### Add Noise to the Image ###
    if args.min_column_density>0:
        noise_flux, noise_level = add_noise(ifu,args)

        plt.figure()
        nz = args.base_channels
        nu = np.linspace(args.base_freq_range[1].in_units('1/s'),args.base_freq_range[0].in_units('1/s'),nz)
        dnu = ( np.max(nu)-np.min(nu) ) / np.size(nu)
        image = np.sum(ifu+noise_flux,axis=2) * dnu
        vmin = args.min_column_density/(50.)
        vmax = 1e22
        #print('vmin=',vmin)
        plt.imshow(image,norm=LogNorm(vmin=vmin,vmax=vmax),cmap='inferno')
        plt.colorbar()
        plt.savefig(args.output+"_NoisyImage.png")
        plt.close()

        noise_std = np.std(noise_flux)
        noise_flux = apply_gaussian_smoothing(noise_flux,instrument,args)
        print("Noise flux after smoothing=",f"{np.std(noise_flux):e}")
        print("Renormalizing noise flux...")
        noise_flux = noise_flux / np.std(noise_flux) * noise_std 
        print("Smoothed noise flux after renormalizing=",f"{np.std(noise_flux):e}")

    ### Apply the Primary Beam Correction ###
    ifu, primary_beam = apply_primary_beam_correction(ifu,instrument,args)

    ### Apply Gaussian Smoothing to mimic the PSF ###
    ifu = apply_gaussian_smoothing(ifu,instrument,args)

    #Make some basic plots
    if args.min_column_density>0:
        plt.figure()
        nz = args.base_channels
        nu = np.linspace(args.base_freq_range[1].in_units('1/s'),args.base_freq_range[0].in_units('1/s'),nz)
        dnu = ( np.max(nu)-np.min(nu) ) / np.size(nu)
        image = np.sum(ifu+noise_flux,axis=2) * dnu
        plt.imshow(image,norm=LogNorm(vmin=vmin,vmax=vmax),cmap='inferno')
        plt.colorbar()
        plt.savefig(args.output+"_NoisyImageSmoothed.png")
        plt.close()



        plt.figure()
        image = ifu+noise_flux
        #image[image<3*noise_level] = 0
        image = np.sum(image,axis=2)*dnu
        #image[image<3.*args.min_column_density] = 0
        image[image<5*instrument.integrated_channels_for_noise*noise_level*dnu] = 0
        plt.imshow(image,norm=LogNorm(vmin=vmin,vmax=vmax),cmap='inferno')
        plt.colorbar()
        plt.savefig(args.output+"_NoisyImageSmoothed_with5SigmaColDensCut.png")
        plt.close()


    #If your min_basline is 0, you aren't missing any short baseline signal!
    if instrument.min_baseline <= 0: args.skip_spatial_filtering = True

    ### Apply Spatial Filtering to account for missing diffuse signal in interferometry ###
    if not args.skip_spatial_filtering:
        clean_image, clean_image_with_residuals = filter_spatial_frequencies_slicewise(ifu,noise_flux,instrument,args,clean_mask=clean_mask)

        #Return the clean components only, clean components with residuals, and the unfiltered mock cube
        return clean_image, clean_image_with_residuals, ifu + noise_flux
    else:
        #Return only the unfiltered mock cube
        return  ifu*0, ifu*0, ifu + noise_flux
    


