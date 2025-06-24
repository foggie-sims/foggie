
import numpy as np
import unyt as u
from functools import partial

from foggie.galaxy_mocks.mock_hi_imager.HICubeHeader import * #Constants

from foggie.galaxy_mocks.mock_hi_imager.radio_telescopes import radio_telescope

from foggie.galaxy_mocks.mock_hi_imager.line_properties import load_line_properties
from foggie.galaxy_mocks.mock_hi_imager.line_properties import _Emission_HI_21cm

from foggie.galaxy_mocks.mock_hi_imager.utils_hi_datacube import _pseudo_pos_x_projected
from foggie.galaxy_mocks.mock_hi_imager.utils_hi_datacube import _pseudo_pos_y_projected
from foggie.galaxy_mocks.mock_hi_imager.utils_hi_datacube import _pseudo_pos_z_projected
from foggie.galaxy_mocks.mock_hi_imager.utils_hi_datacube import _pseudo_v_doppler


from foggie.galaxy_mocks.mock_hi_imager.utils_hi_datacube import get_inclination_rot_arr

from foggie.galaxy_mocks.mock_hi_imager.utils_hi_datacube import make_simple_moment_maps
from foggie.galaxy_mocks.mock_hi_imager.utils_hi_datacube import generate_cut_mask

'''
Functions used to generate ideal HI datacubes. Logic starts at get_ideal_hi_datacube(), but the core functionality is in generate_spectra

Author: Cameron Trapp
Last updated 06/17/2025
'''




def generate_spectra(ds,source_cut,args,ifu_shape,pixel_res_kpc):
    '''
    This is the core function for creating the ideal datacube.
    For each cell it calculates the line profile following Draine's "Physics of the Interstellar and Intergalactic Medium"(Gaussian Core + Damping wings).
    It then projects the line profile onto the IFU grid, taking into account the pixel resolution and the position of the cell in the IFU.
    
    This projection is controlled by the inclination, position angle, pixel resolution, and field of view in the arguments/instrument definition.

    For large fields of view this can cause memory issues, so args.memory_chunks can be set to run this in chunks.


    Arguments are:
        ds: yt datasource object
        source_cut: cut region of (likely want to use refine_box)
        args: system arguments
        ifu_shape: shape of the desired ifu cube
        pixel_res_kpc: the resolution of each pixel in kpc. Should oversample the psf by ~4
    '''
    print("Loading data...")
    species_mass,gamma_ul, A_ul, Elevels, Glevels, n_u_fraction, n_l_fraction = load_line_properties("HI_21cm")

    temp = source_cut['gas','temperature']
    v_doppler = source_cut['gas','v_doppler']
    
    
    nx,ny,nspec = ifu_shape

    ifu = np.zeros(ifu_shape) *u.s/u.cm/u.cm#* u.erg*u.g*u.km/u.s/u.s/u.statC/u.statC

    print("Calculating frequencies...")
    nu = np.zeros((nspec))
    print("SPECTRAL RANGE=",args.base_freq_range)
    nu[:] = np.linspace(args.base_freq_range[0].in_units('1/s'),args.base_freq_range[1].in_units('1/s'),nspec)
    nu = nu / u.s

    dopplerBeta = v_doppler  / c

    nu_ul = (Elevels[1]-Elevels[0])/h #in Hz
    f_lu = Glevels[1] / Glevels[0] * A_ul * (m_e * c**3.)/(8.*np.pi**2.*e**2.*nu_ul**2.) #unitless

    doppler_freq_shift = nu_ul * (np.sqrt( np.divide(-dopplerBeta+1 , dopplerBeta+1) ) - 1)

    del v_doppler; del dopplerBeta

    if args.memory_chunks<=0:
        args.memory_chunks=1 #No chunking

    import gc
    nCells = np.size(temp)
    chunk_size = int(np.ceil(nCells / args.memory_chunks))

    pos_x_projected = source_cut['gas','pos_x_projected']
    pos_y_projected = source_cut['gas','pos_y_projected']
    pos_dx = source_cut['gas','dx']
    emission_power = source_cut['gas','Emission_HI_21cm']


    for cc in range(args.memory_chunks):
        start = cc * chunk_size
        end = min((cc+1)*chunk_size+1,nCells)

        nChunkCells = end - start 
        v = np.zeros((nChunkCells , nspec)) * u.cm / u.s
        b = np.zeros((nChunkCells)) * u.cm / u.s
        coreFreqs = np.zeros((nChunkCells , nspec),dtype=bool)
        wingFreqs = np.zeros((nChunkCells , nspec),dtype=bool)
        transition_v = np.zeros((nChunkCells)) * u.cm / u.s

        print("Determing core and wing freqs in chunk",cc)
        v_num1 = np.divide(nu , nu_ul+doppler_freq_shift[start:end,np.newaxis])
        v_num1 = np.multiply(v_num1, v_num1)
        v_numerator = 1 - v_num1
        v_denominator = 1 + v_num1
        v[:,:] = np.divide( v_numerator , v_denominator) * c #in cm/s

        lamda_ul = c / nu_ul 

        b[:]=12.90*np.sqrt(temp[start:end].in_units('K')/ds.units.K*np.power(10.,-4) / (species_mass / amu))*1000.*100. * u.cm / u.s #cm/s

        prefactor = np.sqrt(np.pi)  * (e**2)/(m_e*c) * f_lu*lamda_ul# / b 

        z = np.sqrt(10.31+np.log(7616 / (gamma_ul*lamda_ul)*b.in_units('km/s').v)) #Some transition specific constant
        transition_v[:] = np.abs(b*z)



        coreFreqs[:,:] = np.abs(v) <= transition_v[:,np.newaxis]  #shape of ncells, npsec
        wingFreqs[:,:] =  np.abs(v) >  transition_v[:,np.newaxis] 

        del transition_v; 

        print("Creating line profile...")
        print('intializing sigma...')
        sigma = np.zeros(np.shape(v)) * u.cm * u.cm
        print('initializing b_2d...')
        b_2d = np.zeros((np.shape(v))) * u.cm / u.s
        b_2d[:,:] = b[:,np.newaxis].in_units('cm/s')

        print("Calculating core chunk",cc)
        v_chunk = v[coreFreqs]
        b_chunk = b_2d[coreFreqs].in_units('cm/s')
        exponential_term = np.exp(-np.divide( np.power(v_chunk,2) , np.power(b_chunk,2)))
        del v_chunk
        prefactor_term = np.divide(prefactor,b_chunk) 
        del b_chunk

        sigma[coreFreqs]= np.multiply( prefactor_term , exponential_term )
        del prefactor_term;del exponential_term
 
        print("Calculating wing chunk",cc)
        v_chunk = v[wingFreqs]
        sigma[wingFreqs] = np.divide( prefactor* (1/(4*np.power(np.pi,1.5)) * gamma_ul*lamda_ul) , np.power(v_chunk,2)) #Was commented out? double check


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

        sigma = sigma[mask,:]

        #emission_power = source_cut['gas','Emission_HI_21cm'][mask]
        chunk_power = emission_power[start:end][mask]

        prefactor = m_e * c / np.pi / (e**2) / f_lu
        emission_term = np.multiply(chunk_power[:,np.newaxis] , sigma)

        for i in range(0,np.size(min_x_idx)):
            #Can paralllelize!
            x0 = min_x_idx[i]
            x1 = max_x_idx[i]
            y0 = min_y_idx[i]
            y1 = max_y_idx[i]

            ifu[x0:x1 , y0:y1, :] = ifu[x0:x1 , y0:y1, :] + prefactor * emission_term[i,:][np.newaxis,np.newaxis,:]#Power per Hz per m^2


        gc.collect()
    return ifu


def get_ideal_hi_datacube(args,ds, source_cut,moment_map_filename=None):
    '''
    This function primarily sets up the needed yt fields, as well as defines the instrument and observer parameters.
    The actual ideal datacube is created in generate_spectra.
    This function can be overridden to instead make simple moment_maps or to generate a mask for a projected cut region.

    Arguments are:
        ds: yt datasource object
        source_cut: cut region of (likely want to use refine_box)
        args: system arguments
        moment_map_filename: name of the output for simple moment maps
    '''

    print("Generating ideal HI datacube...")

    print("Defining instrument...")
    instrument = radio_telescope(args)

    if args.base_freq_range  is None: args.base_freq_range = instrument.obs_freq_range  #Not sure why these are different? May have to oversample initially?
    if args.base_channels    is None: args.base_channels = instrument.obs_channels
    if args.set_fov_auto:
        args.fov_kpc = (5. * observer_distance  / instrument.min_spatial_freq).in_units('kpc') #5 times the minimum spatial frequency in the uv plane
    else:
        args.fov_kpc = args.fov_kpc * u.kpc
    if args.set_res_auto:
        args.base_spatial_res = instrument.obs_spatial_res / 4. #resolve the psf by a factor of 4 in image space, units of arcseconds
    else:
        if args.base_spatial_res is None: args.base_spatial_res = instrument.obs_spatial_res #/ 10. #Oversample the psf?


    if args.primary_beam_FWHM_deg is None: args.primary_beam_FWHM_deg = instrument.primary_beam_FWHM_deg

    print("Adding fields...")
    nz = args.base_channels
    nu = np.linspace(args.base_freq_range[1].in_units('1/s'),args.base_freq_range[0].in_units('1/s'),nz)

    ds.add_field(
        ('gas', 'Emission_HI_21cm'),
          function = _Emission_HI_21cm,
          sampling_type='cell',
          force_override=True
    )

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

    print("Source_cut v_doppler=",np.min(source_cut['gas','v_doppler'].in_units('km/s')),np.max(source_cut['gas','v_doppler'].in_units('km/s')))

    nspec = args.base_channels
    pixel_res_kpc = (args.base_spatial_res * arcsec_to_rad * observer_distance).in_units('kpc')
    nx = np.round( args.fov_kpc * u.kpc / pixel_res_kpc ).astype(int)
    ny = np.round( args.fov_kpc * u.kpc / pixel_res_kpc ).astype(int)

    if nx%2==1:
        '''This is important for making sure the clean algorithm converges for our beam model'''
        nx+=1
        print("Adding single pixel to force odd dimensions...")
    if ny%2==1:
        ny+=1

    print("Pixel res in kpc = ",pixel_res_kpc)
    print("Image shape=",nx,ny)


    print("Generating spectrum...")
    print("freq_range=",args.base_freq_range)


    #Two additional options for this function for some basic functionality. Should probably be moved to separate functions...
    if args.make_simple_moment_maps: return make_simple_moment_maps(args,ds,source_cut,Lhat,north_vector,moment_map_filename,max_r=args.fov_kpc/2.,Rvir=500.,radial_averaging_stat='mean')
    if args.make_disk_cut_mask: return generate_cut_mask(ds,source_cut,[nx,ny,nspec],pixel_res_kpc,args)    


    if not args.skip_full_datacube: return generate_spectra(ds,source_cut,args,[nx,ny,nspec],pixel_res_kpc) #The main calculation
    return None #This option really just exists if you want to just add all the projection fields


