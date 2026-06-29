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
    primary_beam = None
    if instrument.primary_beam_FWHM_deg is not None:
        nx,ny,ns = np.shape(ifu)
        y,x = np.indices((nx,ny))
        pixel_radii_deg = np.sqrt( np.power((x-nx//2), 2) + np.power((y-ny//2), 2) ) * args.base_spatial_res / 3600. #in degrees
        pb_sigma = instrument.primary_beam_FWHM_deg / (2.0 * np.sqrt(2.0 * np.log(2.0))) #in degrees
        primary_beam = np.exp(-np.power(pixel_radii_deg, 2) / (2*(pb_sigma)**2)) # peaks at one
        ifu = np.multiply(ifu,primary_beam[:,:,np.newaxis])

    return ifu,primary_beam


def convert_flux_to_column_density(ifu,instrument,args):
    #From Draine Pg. 72
    nx,ny,nz=np.shape(ifu)
    nu = np.linspace(instrument.obs_freq_range[0].in_units('1/s'),instrument.obs_freq_range[0].in_units('1/s'),nz)
    dnu = ( np.max(nu)-np.min(nu) ) / np.size(nu)
    observer_distance = (args.z * c / H0).in_units("m")
    integrated_flux = np.sum(ifu,axis=2)*dnu *  np.pi*np.power(observer_distance,2) / instrument.area.in_units('m**2')

    
    species_mass,gamma_ul, A_ul, Elevels, Glevels, n_u_fraction, n_l_fraction = load_line_properties("HI_21cm")
    nu_ul = (Elevels[1]-Elevels[0])/h #in Hz

    prefactor = 3./16./np.pi * A_ul * h * nu_ul
 
    print("ifu=",ifu[0,0,0])
    print("integrated_flux=",integrated_flux[0],"(should be in eV/m^2/s)")
    print("dnu=",dnu)
    print("nu_ul = ",nu_ul)
    print("A_ul=",A_ul)
    print("h=",h)
    print('prefactor=',prefactor)
    N_HI = integrated_flux / prefactor

    return N_HI.in_units('1/cm**2')

def convert_column_density_to_flux(ifu, column_density, instrument, args):
    nx,ny,nz=np.shape(ifu)
    nu = np.linspace(instrument.obs_freq_range[1].in_units('1/s'),instrument.obs_freq_range[0].in_units('1/s'),nz)
    dnu = ( np.max(nu)-np.min(nu) ) / np.size(nu)

    species_mass,gamma_ul, A_ul, Elevels, Glevels, n_u_fraction, n_l_fraction = load_line_properties("HI_21cm")
    nu_ul = (Elevels[1]-Elevels[0])/h #in Hz

    prefactor = 3./16./np.pi * A_ul * h * nu_ul

    observer_distance = (args.z * c / H0).in_units("m")


    return prefactor * column_density / dnu * instrument.area.in_units('m**2') / (np.pi*np.power(observer_distance,2))
    
       
def add_noise(ifu, instrument, args):
    nx,ny,nz=np.shape(ifu)
    nu = np.linspace(args.base_freq_range[1].in_units('1/s'),args.base_freq_range[0].in_units('1/s'),nz)
    dnu = ( np.max(nu)-np.min(nu) ) / np.size(nu)
    #print("dnu=",dnu)
    if args.min_column_density>0:
        noise_level = args.min_column_density / u.cm / u.cm / dnu / args.sigma_noise_level# 3. /3. # / nz 3-sigma detection?

   # print("Noise level=",noise_level)
   # print("IFU = ",np.max(ifu))
    N = np.size(ifu)
    #noise_flux = np.random.normal(0.0, noise_level.value, (N,2)) 
    #noise_flux = np.linalg.norm(noise_flux,axis=1).reshape(np.shape(ifu)) * noise_level.units #rician distribution

    try:
        #Have the same noise profile if restarted
        hf_nc = h5py.File("./outputs/tmp_slices/tmp_noise_cube_"+args.halo+args.mock_suffix+"_"+args.survey+".h5",'r')
       #hf_nc = h5py.File("/Volumes/wde4tb/foggie_hi_images/outputs/tmp_slices/tmp_noise_cube_"+args.halo+args.mock_suffix+"_"+args.survey+".h5",'r')
        noise_flux = hf_nc['noise_flux'][...] * noise_level.units
        hf_nc.close()
        print("Using noise flux from previous run!")
        if np.shape(noise_flux)!=np.shape(ifu):
            print("Noise cube shape does not match IFU shape! Regenerating noise cube...")
            raise Exception
    except:
        noise_flux = np.random.normal(0.0,noise_level.value, N).reshape(np.shape(ifu)) #Gaussian
        hf_nc = h5py.File("./outputs/tmp_slices/tmp_noise_cube_"+args.halo+args.mock_suffix+"_"+args.survey+".h5",'w')
        hf_nc.create_dataset('noise_flux', data=noise_flux)
        hf_nc.close()
        noise_flux = noise_flux * noise_level.units
   # print("noise flux=",noise_flux)
   # print(np.min(noise_flux),np.max(noise_flux))

    #noise_flux = np.random.normal(0.0,noise_level.value, N).reshape(np.shape(ifu)) * noise_level.units

    print("Noise level is:",f"{noise_level*dnu:e}","(",f"{noise_level:e}",")")
    print("Mean noise flux is:",f"{np.mean(noise_flux*dnu):e}","(",f"{np.mean(noise_flux):e}",")")
    print("Stdv noise flux is:",f"{np.std(noise_flux*dnu):e}","(",f"{np.std(noise_flux):e}",")")

    



    return noise_flux, noise_level

    #flux_counts, random_noise, absolute_noise = convert_incident_flux_to_counts(ifu,instrument,args)

   # noisy_counts = flux_counts + random_noise

    #noisy_ifu  = convert_counts_to_flux(noisy_counts, instrument, args)
    #noise_flux = convert_counts_to_flux(random_noise, instrument, args)
    #noisy_ifu[noisy_ifu<0]=0
   # return noisy_ifu, np.std(noise_flux)





def get_mock_hi_datacube(args, ds, ifu, clean_mask):
    ##Start here!
    instrument = radio_telescope(args)
    max_residual_limit = None#1e-24
    if args.min_column_density>0:
        noise_flux, noise_level = add_noise(ifu,instrument,args)
        max_residual_limit = noise_level#*instrument.integrated_channels_for_noise #3sigma 
        plt.figure()

        nz = args.obs_channels
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

    ifu, primary_beam = apply_primary_beam_correction(ifu,instrument,args)
    ifu = apply_gaussian_smoothing(ifu,instrument,args)


    if args.min_column_density>0:
        plt.figure()
        nz = args.obs_channels
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

    if instrument.min_baseline <= 0: args.skip_spatial_filtering = True

    unfiltered_image = ifu + noise_flux
    if not args.skip_spatial_filtering:

        hf = h5py.File(args.output+"_AllImages.h5",'w')
        hf.create_dataset('ideal_ifu',data=ifu+noise_flux)
        hf.close()


        if args.clean_wideband: return filter_spatial_frequencies_wideband(ifu,noise_flux,instrument,args,max_residual_limit=max_residual_limit,clean_mask=clean_mask)
      
        clean_image, clean_image_with_residuals = filter_spatial_frequencies_slicewise(ifu,noise_flux,instrument,args,max_residual_limit=max_residual_limit,clean_mask=clean_mask)
        return clean_image, clean_image_with_residuals, unfiltered_image
    else:
        return  ifu*0, ifu*0, unfiltered_image
    



