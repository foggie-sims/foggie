
import numpy as np
import unyt as u
import matplotlib.pyplot as plt
from functools import partial

from foggie.galaxy_mocks.mock_hi_imager.HICubeHeader import * #Constants

from foggie.galaxy_mocks.mock_hi_imager.radio_telescopes import radio_telescope

from foggie.galaxy_mocks.mock_hi_imager.line_properties import load_line_properties
from foggie.galaxy_mocks.mock_hi_imager.line_properties import _Emission_HI_21cm

from foggie.galaxy_mocks.mock_hi_imager.utils_hi_datacube import _pseudo_pos_x_projected
from foggie.galaxy_mocks.mock_hi_imager.utils_hi_datacube import _pseudo_pos_y_projected
from foggie.galaxy_mocks.mock_hi_imager.utils_hi_datacube import _pseudo_pos_z_projected
from foggie.galaxy_mocks.mock_hi_imager.utils_hi_datacube import _pseudo_v_doppler
from foggie.galaxy_mocks.mock_hi_imager.utils_hi_datacube import _v_disp


from foggie.galaxy_mocks.mock_hi_imager.utils_hi_datacube import get_inclination_rot_arr

from foggie.galaxy_mocks.mock_hi_imager.utils_hi_datacube import make_moment_map_figures
from foggie.galaxy_mocks.mock_hi_imager.utils_hi_datacube import generate_cut_mask
 
 
'''
Functions used to generate ideal HI datacubes. Logic starts at get_ideal_hi_datacube(), but the core functionality is in generate_spectra

Author: Cameron Trapp
Last updated 06/17/2025
'''


def generate_spectra(args,ds,source_cut,spectralRange,ifu_shape,pixel_res_kpc,observer_distance):
    #Defines line profile with a gaussian core and damping wings. Following Draine "Physics of the Interstellar and Intergalactic Medium"
    print("Loading data...")
    species_mass,gamma_ul, A_ul, Elevels, Glevels, n_u_fraction, n_l_fraction = load_line_properties("HI_21cm")

    temp = source_cut['gas','temperature']
    v_doppler = source_cut['gas','v_doppler']
    
    
    nx,ny,nspec = ifu_shape

    ifu = np.zeros(ifu_shape) *u.s/u.cm/u.cm#* u.erg*u.g*u.km/u.s/u.s/u.statC/u.statC

    print("Calculating frequencies...")
    nu = np.zeros((nspec))
    print("SPECTRAL RANGE=",spectralRange)
    nu[:] = np.linspace(spectralRange[0].in_units('1/s'),spectralRange[1].in_units('1/s'),nspec)
    nu = nu / u.s


    dopplerBeta = v_doppler  / c

    nu_ul = (Elevels[1]-Elevels[0])/h #in Hz
    f_lu = Glevels[1] / Glevels[0] * A_ul * (m_e * c**3.)/(8.*np.pi**2.*e**2.*nu_ul**2.) #unitless

    doppler_freq_shift = nu_ul * (np.sqrt( np.divide(-dopplerBeta+1 , dopplerBeta+1) ) - 1)


    del v_doppler; del dopplerBeta

    import gc
    if args.memory_chunks>0:
        nCells = np.size(temp)
        chunk_size = int(np.ceil(nCells / args.memory_chunks))


        pos_x_projected = source_cut['gas','pos_x_projected']
        pos_y_projected = source_cut['gas','pos_y_projected']
        pos_z_projected = source_cut['gas','pos_z_projected']
        pos_dx = source_cut['gas','dx']
        emission_power = source_cut['gas','Emission_HI_21cm']


        for cc in range(args.memory_chunks):
            start = cc * chunk_size
            end = min((cc+1)*chunk_size+1,nCells)

            nChunkCells = end - start
            if nChunkCells<=0:
                continue 
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
           # b6 = b.in_units('km/s') / (u.km/u.s)
    
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
            distance_factor = np.power( np.divide(observer_distance.in_units('kpc') , pos_z_projected[start:end][mask].in_units('kpc')) , 2.) #account for slight differences in distance (Should be ~1)

            prefactor = m_e * c / np.pi / (e**2) / f_lu
            emission_term = np.multiply(chunk_power[:,np.newaxis] , sigma)
            emission_term = np.multiply(emission_term, distance_factor[:,np.newaxis])

            for i in range(0,np.size(min_x_idx)):
                #Can paralllelize!
                x0 = min_x_idx[i]
                x1 = max_x_idx[i]
                y0 = min_y_idx[i]
                y1 = max_y_idx[i]

                ifu[x0:x1 , y0:y1, :] = ifu[x0:x1 , y0:y1, :] + prefactor * emission_term[i,:][np.newaxis,np.newaxis,:]#Power per Hz per m^2


            gc.collect()
    return ifu



def get_ideal_hi_datacube(args,ds, source_cut, linelist=['HI_21cm'],moment_map_filename=None):
    print("Generating ideal HI datacube...")
    instrument = radio_telescope(args)

    if args.base_freq_range is None: args.base_freq_range = instrument.obs_freq_range  #Not sure why these are different? May have to oversample initially?

    print("Adding fields...")
    print(args.base_freq_range)
    nz = args.obs_channels
    nu = np.linspace(args.base_freq_range[1].in_units('1/s'),args.base_freq_range[0].in_units('1/s'),nz)
    dnu = ( np.max(nu)-np.min(nu) ) / np.size(nu)
    #Emission_HI_21cm = partial(_Emission_HI_21cm, dnu=dnu)

    ds.add_field(
        ('gas', 'Emission_HI_21cm'),
          function = _Emission_HI_21cm,
          sampling_type='cell',
          force_override=True
    )

    ds.add_field(
        ('gas', 'v_disp'),
          function=_v_disp,
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


    print("Defining instrument...")
    instrument = radio_telescope(args)

    if args.base_freq_range  is None: args.base_freq_range = instrument.obs_freq_range  #Not sure why these are different? May have to oversample initially?
    if args.base_spatial_res is None: args.base_spatial_res = instrument.obs_spatial_res #/ 10. #Oversample the psf?
    if args.set_fov_auto:
        args.fov_kpc = (5. * observer_distance  / instrument.min_spatial_freq).in_units('kpc') #5 times the minimum spatial frequency in the uv plane
    else:
        try: args.fov_kcp = args.fov_kpc.in_units('kpc').v * u.kpc
        except: args.fov_kpc = args.fov_kpc * u.kpc
    if args.set_res_auto:
        args.base_spatial_res = instrument.obs_spatial_res / 4. #resolve the psf by a factor of 4 in image space, units of arcseconds
        print("FOV set to:",args.fov_kpc)


    if args.primary_beam_FWHM_deg is None: args.primary_beam_FWHM_deg = instrument.primary_beam_FWHM_deg


    nspec = args.obs_channels
    pixel_res_kpc = (args.base_spatial_res * arcsec_to_rad * observer_distance).in_units('kpc')
    nx = np.round( args.fov_kpc / pixel_res_kpc ).astype(int)
    ny = np.round( args.fov_kpc / pixel_res_kpc ).astype(int)

    if nx%2==1:
        nx+=1
        print("Adding single pixel to force odd dimensions...")
    if ny%2==1:
        ny+=1

    print("Pixel res in kpc = ",pixel_res_kpc)
    print("Image shape=",nx,ny)
    print("Generating spectrum...")


    print("freq_range=",args.base_freq_range)
    print("args.fov_kpc=",args.fov_kpc)
    if args.make_simple_moment_maps: return make_simple_moment_maps(args,ds,source_cut,Lhat,north_vector,moment_map_filename,max_r=args.fov_kpc/2.,Rvir=500.,radial_averaging_stat='mean')
    if args.make_disk_cut_mask:
        return generate_cut_mask(ds,source_cut,[nx,ny,nspec],pixel_res_kpc)    
    if args.make_cell_mass_projection:
        return generate_cell_mass_projection(ds,source_cut,[nx,ny,nspec],pixel_res_kpc)
    if not args.skip_full_datacube: return generate_spectra(args,ds,source_cut,args.base_freq_range,[nx,ny,nspec],pixel_res_kpc,observer_distance)

    return None

def make_simple_moment_maps(args,ds,source_cut,L,north_vector,filename,max_r=150,Rvir=None,radial_averaging_stat='mean'):
    import yt
    if Rvir is None:
        try:
            Rvir = read_virial_mass_file(args.halo,args.snapshot,args.run,args.code_dir,key='radius')
        except:
            print("Warning, could not read virial mass file...Setting to 250 kpc")
            Rvir = 250.
   

    center = ds.halo_center_code
    W = [2*max_r,2*max_r,Rvir*2.  * ds.units.kpc]
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
    


    p = yt.OffAxisProjectionPlot(ds, L,("gas","v_doppler"),center=center,width=[2*max_r.in_units('kpc').v,'kpc'],depth=[Rvir*2.,'kpc'], weight_field=('gas','H_p0_number_density'),max_level=11,moment=2, method='integrate', north_vector=north_vector,data_source=source_cut)
    moment2_projection = p.frb['gas','v_doppler'].in_units('km/s')
    print("moment2_projection=",moment2_projection)
    p2 = yt.OffAxisProjectionPlot(ds, L,("gas","H_p0_number_density"),center=center,width=[2*max_r.in_units('kpc').v,'kpc'],depth=[Rvir*2.,'kpc'],max_level=11,moment=1, method='integrate',weight_field=None, north_vector=north_vector,data_source=source_cut)
    moment2_density_projection = p2.frb['gas','H_p0_number_density'].in_units('1/cm**2')

    pixel_indices = np.indices(np.shape(density_projection))
    print("SHAPE OF DENSITY PROJECTION=",np.shape(density_projection))
    pixel_size = res_r.in_units('kpc').v # kpc
    print("PIXEL_SIZE=",pixel_size)
    print("MAX_R=",max_r)
    pixel_radii = np.sqrt( np.power(pixel_indices[0,:,:] * pixel_size - max_r.in_units('kpc').v , 2.) + np.power( pixel_indices[1,:,:] * pixel_size - max_r.in_units('kpc').v , 2.) )

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


    hf = h5py.File(filename+"_simple_moment_maps.h5",'w')
    hf.create_dataset("density_projection", data = density_projection)
    hf.create_dataset("v_doppler_projection", data = v_doppler_projection)
    hf.create_dataset("moment2_projection", data = moment2_projection)
    hf.create_dataset("moment2_density_projection", data = moment2_density_projection)
    hf.create_dataset("max_r", data = max_r.in_units('kpc').v)
    hf.close()

    mask = (density_projection < 1e16)
    density_projection[mask]=np.nan
    v_doppler_projection[mask]=np.nan
    
    mask2 = (moment2_density_projection < 1e16)
    moment2_projection[mask2]=np.nan
    #moment2_projection[mask]=np.nan

    labelsize = 18
    ticksize = 16
    titlesize = 20
    plt.figure(figsize=(22,6))
    plt.subplot(131)
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

    plt.subplot(132)
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

    plt.subplot(133)
    plt.imshow(np.rot90(moment2_projection),vmin=0,vmax=100,cmap='inferno')
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
    cbar.set_label("$\\sigma_{v}$ [km s$^{-1}$]", fontsize=labelsize)
    cbar.ax.tick_params(labelsize=ticksize)

    if filename is not None: plt.savefig(filename+".png",bbox_inches='tight')
    else: plt.savefig("./ideal_moment_maps/misc/moment_map_i"+str(int(args.inclination)).zfill(2)+"_pa"+str(np.round(args.position_angle).astype(int)).zfill(3)+".png",bbox_inches="tight")
    plt.close()




    return r_plot,density_radial_profile, density_projection, v_doppler_projection


