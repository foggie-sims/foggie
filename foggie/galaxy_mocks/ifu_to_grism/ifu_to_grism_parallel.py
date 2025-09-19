from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck15 as cosmo
from scipy import interpolate
from scipy import stats
from astropy.convolution import Gaussian2DKernel, convolve_fft
from matplotlib.colors import LogNorm

from joblib import Parallel, delayed
import multiprocessing
#from multiprocessing import Pool

from functools import partial

from ifu_to_grism_argparser import parse_args
from load_instrument import load_instrument

from load_sky_backgrounds import load_sky_backgrounds


''' 
    Convert from an idealized ifu to an idealized grism image.
    If the bandwidth defined by lambda_max and lambda_min is greater than
    the bandwidth in the IFU fits file, a background continuum will be added to
    each pixel with signal.
    
    args are parsed as follows:
    REQUIRED:
    --ifu_dir: The directory/filename of the ifu fits file.
    
    OPTIONAL:
    These parameters are for defining the instrument and imaging parameters/conditions
    --instrument: The instrument you wish to model (Default is WFC3).
    --filter: The filter you wish to model (Default is G102).
    --dfilter: The direct imaging filter to use in the header. Currently doesn't change functionality (Default is F105W).
    --nBeams: How many beams to model [1-4]. 1 Models only the 1st order beam, 2 adds the 0th order, 3 adds the 2nd order, and 4 adds the -1st order.
    
    --exposure: The exposure time in seconds (Default is 300).
    --effective_exposure: The effective exposure time used for the header and SNR calculations (Default is exposure).
    --snr: Target signal to noise ratio (Default is 10.)
    
    --pEarthshine: The percentage strength of the earthshine background to add to sky background (0.0-1.0 recommended, Default is 0.5)
    --pZodiac: The percentage strength of the zodiacal light to add to sky background (0.1-1.0 recommended, default is 0.5)
    
    These parameters are for adjusting how the IFU should be fit into the grism image (ie if it is much smaller)
    --ifu_offset_x: Pixel offset to add to ifu image when placing in GRISM image space. Default is 0.
    --ifu_offset_y: Pixel offset to add to ifu image when placing in GRISM image space. Default is 0.
    --ifu_wavelength_offset: Wavelength offsFactor to boost the flux of the simualated ifu. Default is 1.0et for ifu data in Angstrom. For testing purposes. Default is 0.
    --ifu_signal_boost_factor: Factor to boost the flux of the simualated ifu. For testing purposes. Default is 1.0
    
    These parameters are for overriding instrumental norms:
    --lambda_min: The minimum wavelength in Angstroms for the output spectra (Defaults to instrument/filter).
    --lambda_max: The maximum wavelength in Angstroms for the output spectra (Defaults to instrument/filter).
    
    These parameters are for defining a custom/ideal instrument and will not work if instrument is not set to "Custom" (Not implemented):
    --dispersion: How many Angstroms per pixel for dispersion (This will not do anything if instrument is defined).
    --xoffset: How offset is the zeroth order beam from the direct image in pixels? (This will not do anything if instrument is defined).
    --yoffset: How offset is the zeroth order beam from the direct image in pixels? (This will not do anything if instrument is defined).

   
    --plot_all: Plot additional diagnostic plots. Default is False.
    --output: Output director/filename for plots and new fits files (Default is grism_test).
'''
    
xaspect_ratio=1
fig_x = 8.
fig_y = 5.5

rad_to_arcsec = 180./np.pi * 3600.

IFU_TO_GRISM_DIR = "."

class TqdmProgressBar:
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
        
        


    


def PlotMomentMaps(moment0,delta_lambda,args,figsize=(fig_x,fig_y)):
    '''
    Simple function to visualize the input ifu_cube
    '''
    output_suffix="_moment0"
    filetype=".png"
    print("Plotting to ",args.output+output_suffix+filetype)

    to_plot = moment0 * delta_lambda
    fig = plt.figure()
    cm = plt.cm.inferno
    cm.set_bad('k')
    print("to_plot=",np.max(to_plot))
    
    plt.imshow(np.rot90(to_plot),cmap=cm,norm=LogNorm())
    plt.colorbar()
    
    fig.savefig(args.output+output_suffix+filetype, dpi = 300)



def pad_spectra_with_continuum(ifu_cube, ifu_wavelength, args, flat_continuum=None):
    '''
    Function to pad the bandwith of the idealized ifu_cube with some continuum background.
    Only adds continuum to pixels with signal.
    Currently only allows for adding a flat continuum.
    '''
    lambda_min = float(args.lambda_min)
    lambda_max = float(args.lambda_max)
    
    nspec,nx,ny = np.shape(ifu_cube)
    ifu_lambda_min = np.min(ifu_wavelength)
    ifu_lambda_max = np.max(ifu_wavelength)
    
    ifu_res = (ifu_lambda_max - ifu_lambda_min) / float(nspec)
    
    nspec_new = int(np.round((lambda_max - lambda_min) / float(ifu_res)))
    
    new_ifu_cube = np.zeros((nspec_new , nx, ny))
    new_ifu_wavelength = np.linspace(lambda_min,lambda_max,nspec_new)
    
    left_index = int(np.round(  (ifu_lambda_min - lambda_min) / float(ifu_res) ))
    right_index = nspec_new-int(np.round( (lambda_max - ifu_lambda_max) / float(ifu_res) )) 

    
    ifu_cube[np.isnan(ifu_cube)]=0
    if flat_continuum is None: flat_continuum = np.min(ifu_cube[ifu_cube>0])
    m0map = np.sum(ifu_cube, axis=0)
    continuum_mask =(m0map>0) #only add continuum to pixels with signal
    
    new_ifu_cube[left_index:right_index,:,:] = ifu_cube

    new_ifu_cube[:,continuum_mask] += flat_continuum
    
    
    return new_ifu_cube, new_ifu_wavelength
    

def iterate_ifu_pixel(thread_id,Nthreads,beams,ifu_wavelength,datacube,grism_shape,pbar_title=None):
    '''
    Each thread calculates the dispersion track for all wavelengths in their assigned pixels for each beam.
    Returns the summed contribution to the final grism image for those pixels
    '''

    grism_nx,grism_ny = grism_shape
    ifu_nspec,ifu_nx,ifu_ny = np.shape(datacube)


    Npixels = ifu_nx*ifu_ny
    Npixels_per_thread = np.ceil(Npixels / Nthreads)
    
    start_pixel = int(thread_id*Npixels_per_thread)
    end_pixel = int((thread_id+1)*Npixels_per_thread)
    if end_pixel>Npixels: end_pixel = int(Npixels)
    
    output = np.zeros((grism_nx,grism_ny))
    
    #print("Thread",thread_id,": starting at",start_pixel,"and ending at",end_pixel)
    
    if pbar_title is not None and thread_id==0:
        pbar = TqdmProgressBar("Thread "+str(thread_id)+": "+pbar_title,(end_pixel-start_pixel+1)*len(beams),position=thread_id)
     
    beam_itr = -1
    for beam in beams:
        beam_itr+=1
        print("Beam wavelength range:",beam.sensitivity_wavelength[0],"-",beam.sensitivity_wavelength[-1])
        interp = interpolate.interp1d(beam.sensitivity_wavelength, beam.sensitivity_response, bounds_error = False, fill_value = 0.)
        sensitivity_interpolate = interp(ifu_wavelength)
        for ij in range(start_pixel, end_pixel):
            if pbar_title is not None and thread_id==0:
                if (ij-start_pixel)%1==0: pbar.update(ij-start_pixel+beam_itr*(end_pixel-start_pixel+1))
        #else:
        #    if (ij-start_pixel)%100==0: print("Thread",thread_id,":",ij-start_pixel,"/",end_pixel-start_pixel)
            i = ij % ifu_nx
            j = int(np.floor(ij / ifu_ny))
            spec_flux = datacube[:,i,j] # erg/cm^2/s/Angstrom
            if np.max(np.abs(spec_flux))==0: continue
        #for beam in beams:
            i_px = i+instrument.xrange[0]
            j_px = j+instrument.yrange[0]
            
            trace = beam.inv_disp_lambda(i_px,j_px,ifu_wavelength) #Gives beam trace parameter for each wavelength
            disp_x = beam.disp_x(i_px,j_px,trace) #Gives x dispersion for each wavelength
            disp_y = beam.disp_y(i_px,j_px,disp_x,trace) #Gives y dispersion for each wavelength
                
                
            #sensitivity = beam.sensitivity_response
            #sensitivity_wavelength = beam.sensitivity_wavelength
            #
            #interp = interpolate.interp1d(beam.sensitivity_wavelength, beam.sensitivity_response, bounds_error = False, fill_value = 0.)
            #sensitivity_interpolate = interp(ifu_wavelength)

#            measured_flux = np.divide(counts,sensitivity_interpolate)  * instrument.gain / args.effective_exposure #convert back to erg/cm^2/s/Angstrom, No dependence on exposure time?
            #measured_flux = counts
            measured_flux = np.multiply(spec_flux, sensitivity_interpolate) * args.effective_exposure/instrument.gain #Counts

            #start_x = beam.xoffset+i+ifu_offset_x
            #start_y = beam.yoffset+j+ifu_offset_y
            start_x = i+ifu_offset_x
            start_y = j+ifu_offset_y
            if (start_y <=0) | (start_y>=grism_ny-1): continue
                
            #spec_flux = ifu_cube[lambda_mask,i,j] #TODO: CHECK IF DIMMING ACCOUNTED FOR PROPERLY IN AYAN'S CODE / dimming_factor
                                
            xpixel = start_x + disp_x
            ypixel = start_y + disp_y
                
                
            min_x = int(np.min(xpixel))
            max_x = int(np.max(xpixel))
            
            min_y = int(np.min(ypixel))
            max_y = int(np.max(ypixel))


            if (max_x>=0 and min_x<=grism_nx) and (max_y>0 and min_y<=grism_ny):
                #Rebin spectral data with above dispersions
                if min_x<0: min_x=0
                if min_y<0: min_y=0
                
                if max_x>=grism_nx: max_x=grism_nx-1
                if max_y>=grism_ny: max_y=grism_ny-1
                
                bin_nx = int(max_x - min_x + 1)
                bin_ny = int(max_y - min_y + 1)

                binned_trace,xedge,yedge,binnum = stats.binned_statistic_2d(xpixel,ypixel,measured_flux,statistic='sum', bins=(bin_nx,bin_ny),range=((min_x,max_x),(min_y,max_y)))
                if (np.max(binned_trace)>0): print("binned_trace=",np.max(binned_trace))

                output[min_x:max_x+1,min_y:max_y+1] += binned_trace
                


    if pbar_title is not None and thread_id==0:
        pbar.update((end_pixel-start_pixel+1)*len(beams))
        pbar.finish()
    return output


def iterate_ifu_pixel_2(thread_id,Nthreads,beams,ifu_wavelength,datacube,grism_shape,pbar_title=None):
    '''
    Each thread calculates the dispersion track for all wavelengths in their assigned pixels for each beam.
    Returns the summed contribution to the final grism image for those pixels
    '''

    grism_nx,grism_ny = grism_shape
    ifu_nspec,ifu_nx,ifu_ny = np.shape(datacube)


    Npixels = ifu_nx*ifu_ny
    Npixels_per_thread = np.ceil(Npixels / Nthreads)
    
    start_pixel = int(thread_id*Npixels_per_thread)
    end_pixel = int((thread_id+1)*Npixels_per_thread)
    if end_pixel>Npixels: end_pixel = int(Npixels)
    
    output = np.zeros((grism_nx,grism_ny))
    
    #print("Thread",thread_id,": starting at",start_pixel,"and ending at",end_pixel)
    
    if pbar_title is not None and thread_id==0:
        pbar = TqdmProgressBar("Thread "+str(thread_id)+": "+pbar_title,(end_pixel-start_pixel+1)*len(beams),position=thread_id)
     
    beam_itr = -1
    for beam in beams:
        beam_itr+=1
        #print("Beam wavelength range:",beam.sensitivity_wavelength[0],"-",beam.sensitivity_wavelength[-1])
        interp = interpolate.interp1d(beam.sensitivity_wavelength, beam.sensitivity_response, bounds_error = False, fill_value = 0.)
        sensitivity_interpolate = interp(ifu_wavelength)
        for ij in range(start_pixel, end_pixel):
            if pbar_title is not None and thread_id==0:
                if (ij-start_pixel)%1==0: pbar.update(ij-start_pixel+beam_itr*(end_pixel-start_pixel+1))
        #else:
        #    if (ij-start_pixel)%100==0: print("Thread",thread_id,":",ij-start_pixel,"/",end_pixel-start_pixel)
            i = ij % ifu_nx
            j = int(np.floor(ij / ifu_ny))
            spec_flux = datacube[:,i,j] # erg/cm^2/s/Angstrom
            if np.max(np.abs(spec_flux))==0: continue
            #spec_flux = np.multiply(spec_flux,sensitivity_interpolate)*args.effective_exposure/instrument.gain #Counts
            print("Looking at pixel",i,j,"with flux",np.sum(spec_flux))
        #for beam in beams:
            i_px = i+instrument.xrange[0]
            j_px = j+instrument.yrange[0]


            #trace ranges from 500-
            min_dx = int(0)
            max_dx = int(instrument.xrange[1] - instrument.xrange[0])
            n_dx = int(max_dx - min_dx + 1)
            n_dx = int(n_dx*n_dx)
            dx_bin = np.linspace(min_dx,max_dx,n_dx)
            if beam_itr==3: #-1 order
                dx_bin = np.linspace(-max_dx,min_dx,n_dx) #-1st order beam
            
            dy_bin = beam.disp_y(i_px,j_px,dx_bin) #Gives y dispersion for each trace bin
            trace_bin = np.sqrt(np.multiply(dx_bin,dx_bin) + np.multiply(dy_bin,dy_bin)) #Gives beam trace parameter for each bin


            interp_dx = interpolate.interp1d(trace_bin, dx_bin, bounds_error = False, fill_value = 0.)

      

            trace = beam.inv_disp_lambda(i_px,j_px,ifu_wavelength) #Gives beam trace parameter for each wavelength



            disp_x = interp_dx(trace) #Gives x dispersion for each wavelength
            disp_y = beam.disp_y(i_px,j_px,disp_x) #Gives y dispersion for each wavelength
                
            if beam_itr==3:
                print("FOR BEAM",beam_itr," dx_bin=",dx_bin)
                print("FOR BEAM",beam_itr," dy_bin=",dy_bin)
                print("FOR BEAM",beam_itr," trace_bin=",trace_bin)
                print("FOR BEAM",beam_itr,"trace=",trace)
                print("FOR BEAM",beam_itr,"disp_x=",disp_x)
                print("FOR BEAM",beam_itr,"disp_y=",disp_y)



            #sensitivity = beam.sensitivity_response
            #sensitivity_wavelength = beam.sensitivity_wavelength
            #
            #interp = interpolate.interp1d(beam.sensitivity_wavelength, beam.sensitivity_response, bounds_error = False, fill_value = 0.)
            #sensitivity_interpolate = interp(ifu_wavelength)

#            measured_flux = np.divide(counts,sensitivity_interpolate)  * instrument.gain / args.effective_exposure #convert back to erg/cm^2/s/Angstrom, No dependence on exposure time?
            #measured_flux = counts
            measured_flux = np.multiply(spec_flux, sensitivity_interpolate) * args.effective_exposure/instrument.gain #Counts

            start_x = beam.xoffset+i+ifu_offset_x
            start_y = beam.yoffset+j+ifu_offset_y
                                
            if (start_y <=0) | (start_y>=grism_ny-1): continue
                
            #spec_flux = ifu_cube[lambda_mask,i,j] #TODO: CHECK IF DIMMING ACCOUNTED FOR PROPERLY IN AYAN'S CODE / dimming_factor
                                
            xpixel = start_x + disp_x
            ypixel = start_y + disp_y
                
                
            min_x = int(np.min(xpixel))
            max_x = int(np.max(xpixel))
            
            min_y = int(np.min(ypixel))
            max_y = int(np.max(ypixel))


            if (max_x>=0 and min_x<=grism_nx) and (max_y>0 and min_y<=grism_ny):
                #Rebin spectral data with above dispersions
                if min_x<0: min_x=0
                if min_y<0: min_y=0
                
                if max_x>=grism_nx: max_x=grism_nx-1
                if max_y>=grism_ny: max_y=grism_ny-1
                
                bin_nx = int(max_x - min_x + 1)
                bin_ny = int(max_y - min_y + 1)

                binned_trace,xedge,yedge,binnum = stats.binned_statistic_2d(xpixel,ypixel,measured_flux,statistic='sum', bins=(bin_nx,bin_ny),range=((min_x,max_x),(min_y,max_y)))
                if (np.max(binned_trace)>0): print("binned_trace=",np.max(binned_trace))

                output[min_x:max_x+1,min_y:max_y+1] += binned_trace
                


    if pbar_title is not None and thread_id==0:
        pbar.update((end_pixel-start_pixel+1)*len(beams))
        pbar.finish()
    return output




def map_ifu_to_grism(args,instrument):
    '''
    Main function to convert from an ifu to grism, where args is read in by parse args
    and instrument is of the _instrument class defined in load_instrument.py,
    which contains relevant information for how each beam (0th, 1st, 2nd, and -1th)
    is traced and dispersed.
    
    
    '''

    
    print("Loading ",args.ifu_dir)
    data = fits.open(args.ifu_dir)
    
    lambda_min = args.lambda_min
    lambda_max = args.lambda_max
    bw_lambda = lambda_max - lambda_min
    
    dispersion = args.dispersion
    
    header = data[0].header
    z = header['redshift']
    
    print("Redshift=",z)
    #filt_lambdas = data[0].header['lambdas'] * 1.e6/(1+z) ##TODO-Check units and redshift
    
    ifu_cube = data[0].data
    if True:
        ifu_cube = np.rot90(ifu_cube,axes=(2,1))
    ifu_wavelength = data[1].data + args.ifu_wavelength_offset
    print("ifu wavelength ranges from:",np.min(ifu_wavelength),"-",np.max(ifu_wavelength))
    dimming_factor = (1.+z)**4     
    lambda_mask = (ifu_wavelength > lambda_min) & (ifu_wavelength < lambda_max)
    ifu_wavelength = ifu_wavelength[lambda_mask]
    
    ifu_lambda_min = np.min(ifu_wavelength)
    ifu_lambda_max = np.max(ifu_wavelength)
    
    if (ifu_lambda_min>lambda_min) | (ifu_lambda_max<lambda_max):
        ifu_cube, ifu_wavelength = pad_spectra_with_continuum(ifu_cube,ifu_wavelength,args,flat_continuum=0)
        ifu_lambda_min = np.min(ifu_wavelength)
        ifu_lambda_max = np.max(ifu_wavelength)
        lambda_mask = (ifu_wavelength >= lambda_min) & (ifu_wavelength <= lambda_max)
        ifu_wavelength = ifu_wavelength[lambda_mask]
    
    ifu_cube *= args.ifu_signal_boost_factor
    #sky_ifu = load_sky_backgrounds(ifu_wavelength,args.pEarthshine,args.pZodiacal)
    print("ifu=",np.max(ifu_cube))
    
    CROTA2 = 0.0 #Shouldn't matter
    CD1_1 = header['CDELT1'] * np.cos(CROTA2) #Scale and rotation for mapping platescale
    ifu_arc_pix = cosmo.arcsec_per_kpc_proper(z).value*CD1_1
    #grism_nx = int(ifu_nx + np.round(2*lambda_max / dispersion))
    #grism_ny = int(ifu_ny)
    
    grism_nx = int(instrument.xrange[1]-instrument.xrange[0])
    grism_ny = int(instrument.yrange[1]-instrument.yrange[0])
        
    m0 = np.zeros((grism_nx, grism_ny))
    m0[ifu_offset_x:np.shape(ifu_cube)[1]+ifu_offset_x , ifu_offset_y:np.shape(ifu_cube)[2]+ifu_offset_y] = np.sum(ifu_cube,axis=0)
    print("m0=",np.max(m0))
    PlotMomentMaps(m0,header['CDELT1'],args)


    
    
    
    

    
    ifu_nspec, ifu_nx, ifu_ny = np.shape(ifu_cube)
    
        

   # global xaspect_ratio
   # xaspect_ratio = float(grism_nx) / float(ifu_nx)

    x = np.arange(0, ifu_nx, 1)
    y = np.arange(0, ifu_ny, 1)

    xx, yy = np.meshgrid(x, y)
    xx = xx.astype('int')
    yy = yy.astype('int')
    
    
    #1st, 0th, 2nd, -1st order beams
    beams = [instrument.beam_a, instrument.beam_b, instrument.beam_c, instrument.beam_e]
    #beams = [instrument.beam_a,instrument.beam_b,instrument.beam_c]
    beam_names = ["1st order beam","0th order beam","2nd order beam","-1st order beam"]
    #beam_names = ["1st order beam","0th order beam","2nd order beam"]

    num_cores = multiprocessing.cpu_count()-1
    
    grism = np.zeros((grism_nx,grism_ny))
    #sky_background = np.copy(grism)

    #print("Adding sky background...")
    #ifu_cube = ifu_cube + sky_background    


    #Convert ifu to counts, determine Poisson noise
   # if args.snr>0:    
   #     random_noise,absolute_noise = calculate_noise_flux(ifu_cube,ifu_wavelength,instrument,args)
   # else:
   #     random_noise=0*ifu_cube
    #    absolute_noise=0*ifu_cube
    ####
    #ifu_cube = np.add(ifu_cube,random_noise)
    _iterate_ifu_pixel_spec_flux = partial(iterate_ifu_pixel,datacube=ifu_cube[lambda_mask,:,:],Nthreads=num_cores,beams=[beams[ii] for ii in args.beams],ifu_wavelength=ifu_wavelength,grism_shape=[grism_nx,grism_ny],pbar_title="Making Grism Image")
    #_iterate_ifu_pixel_spec_flux = partial(iterate_ifu_pixel_2,datacube=ifu_cube[lambda_mask,:,:],Nthreads=num_cores,beams=beams[0:args.nBeams],ifu_wavelength=ifu_wavelength,grism_shape=[grism_nx,grism_ny],pbar_title="Making Grism Image")


    if True:
        x= Parallel(n_jobs=num_cores)(delayed(_iterate_ifu_pixel_spec_flux)(i) for i in range(0,num_cores))
        grism += np.sum(x,0) #with poisson noise
        print("grism->",np.max(grism))

            
    if args.snr>0:
        #sky_ifu = np.ones(np.shape(ifu_cube)) * sky_ifu[:,np.newaxis,np.newaxis] * (instrument.pixel_size_arcsec)**2 #background flux (ergs/s/cm^2/A)
        #print("sky_background=",np.max(sky_ifu))

        #_iterate_ifu_pixel_spec_flux = partial(iterate_ifu_pixel,datacube=sky_ifu[lambda_mask,:,:],Nthreads=num_cores,beams=beams[0:args.nBeams],ifu_wavelength=ifu_wavelength,grism_shape=[grism_nx,grism_ny],pbar_title="Making Sky Background")
        #x= Parallel(n_jobs=num_cores)(delayed(_iterate_ifu_pixel_spec_flux)(i) for i in range(0,num_cores))
        #sky_background += np.sum(x,0) #with poisson noise
        #print("sky_grism->",np.max(sky_background))

        print("Calculating instrument background noise...") ##TODO REPLACE WITH FILE LOADED FROM STSCI
        instrument_background = (instrument.dark_current + instrument.thermal_background + instrument.zodiacal_background+instrument.earthshine) * args.effective_exposure #/ instrument.gain #electrons/s * s
        #instrument_background = (instrument.dark_current + instrument.thermal_background) * args.effective_exposure / instrument.gain #electrons/s * s


    #instrument_err = instrument_background / args.snr
    #instrument_noise =  np.random.poisson(lam=np.power(instrument_err,2), size=np.shape(grism)) - np.power(instrument_err,2)
                

    #grism_noise += instrument_noise

    #return grism, m0, ifu_wavelength, instrument_noise, instrument_background, sky_background
    else:
        instrument_background = 0
        #sky_background = 0*grism
    return grism, m0, ifu_wavelength, instrument_background




def apply_spatial_smoothing(grism,args,kernel_size=1):
    '''
    Apply simple gaussian smoothing
    '''
    kern = Gaussian2DKernel(kernel_size) #TODO-Figure out how to properly model the spatial smoothing and line smoothing separately/together
    return convolve_fft(grism, kern)
        
def add_noise_basic(grism,args,snr=100000.):
    '''
    Add some gaussian noise
    '''
    np.random.seed(0)
    return grism+np.random.normal(0, np.nanmax(grism)/snr, grism.shape) 

 
def calculate_noise_from_counts(counts, args):
    absolute_noise = counts / args.snr
    print("Max counts=",np.max(absolute_noise))
    if np.max(absolute_noise) <= 1: print("Warning: Absolute Noise <= 1. Target SNR may be too high for the given exposure time.")
    random_noise = np.random.poisson(lam=np.power(absolute_noise,2), size=np.shape(counts)) - np.power(absolute_noise,2)
    return random_noise,absolute_noise
    
def calculate_noise_flux(ifu,wavelengths,instrument,args):
    print("Calculating wavelength dependent Poisson noise distribution...")
    dlambda = (np.max(wavelengths)-np.min(wavelengths))/np.size(wavelengths) ##angstroms
    planck = 6.626e-27  # ergs.sec Planck's constant
    c = 3e5 #km/s


    noise = np.zeros((np.shape(ifu)))
    err = np.copy(ifu)
    ns,nx,ny = np.shape(ifu)
    wavelength_mat = np.tile(wavelengths,(nx,ny,1))
    wavelength_mat = np.transpose(wavelength_mat,(2,0,1))

    flux_density_to_counts = wavelength_mat*1e-10*np.pi * (instrument.radius*1e2)**2 * args.effective_exposure * instrument.electrons_per_photon * dlambda / (planck * (c * 1e3)) #/ (wavelength * 1e-10))
    flux = np.multiply(ifu , flux_density_to_counts)


    absolute_noise = flux / args.snr
    print(np.max(absolute_noise))
    if np.max(absolute_noise) <= 1: print("Warning: Absolute Noise <= 1. Target SNR may be too high for the given exposure time.")
    random_noise = np.random.poisson(lam=np.power(absolute_noise,2), size=np.shape(ifu)) - np.power(absolute_noise,2)
    
    return np.divide(random_noise,flux_density_to_counts),np.divide(absolute_noise,flux_density_to_counts)
        
def convert_incident_flux_to_counts(ifu,wavelengths,sky_background,beam,instrument,args):
    print("Converting incident flux to electron counts...")
    dlambda = (np.max(wavelengths)-np.min(wavelengths))/np.size(wavelengths) ##angstroms
    planck = 6.626e-27  # ergs.sec Planck's constant
    c = 3e5 #km/s



    ns,nx,ny = np.shape(ifu)
    wavelength_mat = np.tile(wavelengths,(nx,ny,1))
    wavelength_mat = np.transpose(wavelength_mat,(2,0,1))

    #flux_density_to_counts = wavelength_mat*1e-10*np.pi * (instrument.radius*1e2)**2 * args.effective_exposure * instrument.electrons_per_photon * dlambda / (planck * (c * 1e3)) #/ (wavelength * 1e-10))
    #flux = np.multiply(ifu , flux_density_to_counts) # in units of counts
    
    interp = interpolate.interp1d(beam.sensitivity_wavelength, beam.sensitivity_response, bounds_error = False, fill_value = 0.)
    sensitivity_interpolate = interp(wavelengths)  
    
   
    #flux = np.multiply(ifu,sensitivity_interpolate)*args.effective_exposure/instrument.gain #in units of counts
    flux = ifu * sensitivity_interpolate[:,np.newaxis,np.newaxis] * args.effective_exposure/instrument.gain
        
    random_noise = np.ones(np.shape(flux)) * sky_background[:,np.newaxis,np.newaxis] * (instrument.pixel_size_arcsec)**2 #background flux (ergs/s/cm^2/A)
    random_noise = random_noise * sensitivity_interpolate[:,np.newaxis,np.newaxis] * args.effective_exposure/instrument.gain #background photon count
    if args.snr>0:
        print("Calculating wavelength dependent Poisson noise distribution...")
        absolute_noise = (flux+random_noise) / args.snr
        random_noise = np.random.poisson(lam=np.power(absolute_noise,2), size=np.shape(ifu)) - np.power(absolute_noise,2)
       
        return flux, random_noise, absolute_noise
    else:
       return flux, flux*0.,flux*0.
    
    
    


                
    
def MakeGrismPlot(grism,args,output_suffix="",filetype=".png",figsize=(8.,5.5),plot_log=True,title=""):
    '''
    Makes a simple image from the grism outputs.
    '''
    print("Plotting to ",args.output+output_suffix+filetype)

    grism[np.isnan(grism)]=0
    if plot_log and np.min(grism)<0:
        grism = grism-np.min(grism)
        
    #grism+=1e-20

    fig = plt.figure(figsize=figsize)
    cm = plt.cm.inferno
    cm.set_bad('k')
        
    print(np.min(grism))
    print(np.max(grism))
    
    vmax=np.max(grism)
    vmin=np.min(grism[grism>0])/10.
    if vmin==0: vmin=1e-6

    plt.title(title+" [Counts]")
    if plot_log:
        plt.imshow(grism,cmap=cm,norm=LogNorm(vmin=vmin,vmax=vmax))
        plt.colorbar()
        fig.savefig(args.output+output_suffix+filetype, dpi = 300)
    else:
        #import warnings
       # warnings.warn("Issue with plotting"+args.output+output_suffix+filetype+". Plotting in linear scale. Check file")
        plt.imshow(grism,cmap=cm)
        plt.colorbar()
        fig.savefig(args.output+output_suffix+filetype, dpi = 300)
    
def WriteFitsFile(grism,absolute_noise,args,output_suffix=""):
    '''
    Save the grism image to a FITS file with headers appropriate for use with grizli
    '''
    
    filename = args.output + output_suffix + '.fits'
    data = fits.open(args.ifu_dir)
    ifu_header = data[0].header
    
    
    grism_header = fits.Header({'CRPIX1': 1, 
                               'CRVAL1': ifu_header['CRVAL1'], 
                               'CDELT1': ifu_header['CDELT1'], 
                               'CTYPE1': ifu_header['CTYPE1'], 
                               'CRPIX2': 1, 
                               'CRVAL2': ifu_header['CRVAL2'], 
                               'CDELT2': ifu_header['CDELT2'], 
                               'CTYPE2': ifu_header['CTYPE2'], 
                               'INSTRUME': args.instrument,
                               'FILTER': args.filter,
                               'DFILTER': args.dfilter,
                              # 'CONFFILE': ,\ #optional
                               'EXPTIME': args.exposure,
                               'EFFEXPTM': args.effective_exposure,
                              # 'MDRIZSKY': ,\ #optional
                              # 'CPDIS1': ,\ #optional
                             #  'MODULE': ,\ #optional
                             #  'ISCUTOUT': ,\ #optional
                               #TODO-Do similar exposure time and target snr as in ayans code, add target_snr, electros_per_photon to header
                               'SIM': ifu_header['simulation'], 
                               'NSNAP': ifu_header['snapshot'], 
                               'ZDIAG': ifu_header['metallicity_diagnostic'], 
                               'OMEGA': ifu_header['Omega'], 
                               'CUT-KPC': ifu_header['cutout_from_sim(kpc)'], 
                               'INC-DEG': ifu_header['inclination(deg)'], 
                               'REDSHIFT': ifu_header['redshift'], 
                               'DIST-MPC': ifu_header['distance(Mpc)'], 
                               'SPTLRES':ifu_header['base_spatial_res(kpc)'], 
                               'SPECRES': ifu_header['base_spec_res(km/s)'], 
                               'BOXSIZE': ifu_header['box_size'], 
                               'PIXSIZE': ifu_header['pixel_size(kpc)'], 
                               'RESTWAVE':ifu_header['rest_wave_range(A)'], 
                               'LABELS': ifu_header['labels'], 
                               'LAMBDAS': ifu_header['lambdas'], 
                               'DISPERS': args.dispersion, 
                                })
    
    
    grism_uncertainty = absolute_noise#np.zeros((np.shape(grism))) #Uncertainty for each pixel ##TODO-How to determine uncertainties?
    grism_dq = np.zeros((np.shape(grism))) # data quality of pixels. 0=good See https://www.stsci.edu/hst/instrumentation/acs/data-analysis/dq-flag-definitions
    
    
    grism_hdu = fits.PrimaryHDU(grism, grism_header)
    sci_hdu = fits.ImageHDU(grism,name="SCI")
    err_hdu = fits.ImageHDU(grism_uncertainty, name="ERR")
    dq_hdu = fits.ImageHDU(grism_dq, name="DQ")
    
    
    hdulist = fits.HDUList([grism_hdu,sci_hdu,err_hdu,dq_hdu])
    hdulist.writeto(filename, overwrite=True)
    print("GRISM spectrum written to:",filename)



args = parse_args() #Get user arguments (ifu_to_grism_argparser.py)
args, instrument = load_instrument(args) #Set instrument defaults if not specified by user (ifu_to_grism_argparser.py)

ifu_offset_x = args.ifu_offset_x
ifu_offset_y = args.ifu_offset_y


print("Mapping IFU to grism data...")
grism, m0, ifu_wavelength, instrument_background = map_ifu_to_grism(args,instrument)
MakeGrismPlot(np.rot90(grism),args,"_ideal",figsize=(fig_x*xaspect_ratio,fig_y),title="Ideal Grism")
WriteFitsFile(grism,grism*0,args,"_ideal")
WriteFitsFile(m0,grism*0,args,"_direct_image")


print("Applying spatial smoothing to grism...")
grism = apply_spatial_smoothing(grism,args)
MakeGrismPlot(np.rot90(grism),args,"_smoothingOnly",figsize=(fig_x*xaspect_ratio,fig_y),title="Smoothed Grism")
WriteFitsFile(grism,grism*0,args,"_smoothingOnly")
    
    
grism = grism + instrument_background
MakeGrismPlot(np.rot90(grism),args,"_smooth_with_background",figsize=(fig_x*xaspect_ratio,fig_y))

if args.snr>0:
    print("Adding Some Noise...")
    random_noise,absolute_noise = calculate_noise_from_counts(grism,args)
    #grism = add_noise_basic(grism,args)
    grism=grism+random_noise
    MakeGrismPlot(np.rot90(grism),args,"_mock_raw",figsize=(fig_x*xaspect_ratio,fig_y),title="Raw Mock Grism")
    WriteFitsFile(grism,absolute_noise,args,"_mock_raw")

    grism = grism - instrument_background
    MakeGrismPlot(np.rot90(grism),args,"_mock_background_subtracted",figsize=(fig_x*xaspect_ratio,fig_y),title="Mock Grism")
    WriteFitsFile(grism,absolute_noise,args,"_mock_background_subtracted")

