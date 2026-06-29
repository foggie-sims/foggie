import numpy as np
import scipy
import argparse
from argparse import Namespace
from scipy.special import erf
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
#from foggie.galaxy_mocks.mock_ifu.make_ideal_datacube import gauss
import os 
from foggie.utils.foggie_load import foggie_load

from foggie.utils.consistency import *
from foggie.clumps.clump_finder import *
from foggie.clumps.clump_finder.utils_clump_finder import read_virial_mass_file

from foggie.galaxy_mocks.mock_hi_imager.HICubeHeader import * #Constants
from foggie.galaxy_mocks.mock_hi_imager.hi_datacube_arg_parser import parse_args
from foggie.galaxy_mocks.mock_hi_imager.radio_telescopes import radio_telescope

from foggie.galaxy_mocks.mock_hi_imager.utils_hi_datacube import FindMaxProjectedVelocity

from foggie.galaxy_mocks.mock_hi_imager.make_ideal_datacubes import get_ideal_hi_datacube
from foggie.galaxy_mocks.mock_hi_imager.make_mock_datacubes import get_mock_hi_datacube

from foggie.galaxy_mocks.mock_hi_imager.make_mock_datacubes import apply_gaussian_smoothing
from foggie.galaxy_mocks.mock_hi_imager.make_mock_datacubes import apply_primary_beam_correction


#from astropy import units as u
import unyt as u
import time

from functools import partial
from joblib import Parallel, delayed
import multiprocessing

from astropy.io import fits


import sys


'''
Functions to make synthetic HI datacubes targeting specific pre- or user-defined radio surveys.

The core functions are in make_ideal_datacubes.py, make_mock_datacubes.py, and make_spatially_filtered_datacubes.py

make_ideal_datacubes.py calculates the emission profiles for the given projection and creates an ideal noiseless datacube at the given pixel resolution
make_mock_datacubes.py adds gaussian noise and applies gaussian smoothing to make the typical mock datacube
make_spatially_filtered_datacubes.py takes this one step further and simulates the effects of interferometry by filtering out diffuse components of the image

Basic Example usage:
    python make_hi_datacube.py --output outputs/Tempest_MhongooseLR_20Mpc_i40 --z .004666666666 --halo 008508 --clean_sigma 0.5  --mock_suffix _NHI1e18_RD0042_GaussianHPF --memory_chunks 1000 --survey MHONGOOSE_LR  --min_column_density 1e18   --high_pass_filter_type gaussian  --nthreads 15 --clean_gain 0.1 --set_res_auto 1 --fov_kpc 350 --inclination 40 

    The args are parsed as follows:
    
    IO Arguments:
    --halo: Which halo should be analyzed. Default is 008508 (Tempest)
    --snapshot: Which snapshot should be analyzed? Default is RD0042
    --run: What refinement run should be analyzed? Default is nref11c_nref9f  
    --survey: 'What instrument/survey do you want to simulate?' Options are MHONGOOSE_LR, MHONGOOSE_HR, THINGS. This can be in part or completely overwritten by subsequent arguments
    --system: Set the system to get data paths from get_run_loc_etc if not None Overrides --code_dir and --data_dir. Default is None. 
    --data_dir: Where are the simulation outputs?
    --code_dir: Where is the foggie analysis directory?
    --output: Where to save datacubes and plots. Needs directory and prefix: e.g. outputs/Tempest_MhonghooseHR
    --mock_suffix: Suffix to append to mock datacubes (e.g. _NHI1e18_GaussianHPF). Default is None.
    --pwd: Use pwd arguments in get_run_loc_etc. Default is False.
    --forcepath: Use forcepath in get_run_loc_etc. Default is False.

    --nthreads: How many threads for parellel calculations. Defaults to number of avaialbe cpus


    Instrument Overrides:
    --obs_freq_range: What is the observed frequency range in microns. If None will set automatically based on survey/simulation data.
    --obs_spec_res: What is the observed spectral resolution in km/s. If None will default to instrument.
    --obs_spatial_res: Spatial resolution in arcseconds. Defaults to instrument.
    --obs_channels: Number of spectral channels. A different way to set spectral res. Defaults to instrument.
    --base_freq_range: Not implemented. To oversample spectral axis
    --base_spatial_res: What is the spatial resolution of your pixels. Defaults to 4x the observed resolution


    Observational Parameters:
    --inclination: Observed inclination in degrees. Defaults to 45.
    --position_angle: Observed position_angle (rotation) in degrees. Defaults to 0.
    --z: Redshift. Used to set distance. Defaults to 0.002
    --fov_kpc: Dimension of the image in kpc. Defaults to 250 kpc.
    --max_projected_velocity: Alternative way to set the bandwidth. Defaults to the maximum for all gas in fov above a certain column density.

    --set_fov_auto: Automatically set the field of view to resolve the missing short baselines by a factor of 5.
    --set_res_auto: Set the pixel resolution to automatically oversample the observe spatial resolution.

    Spatial Filtering/Clean Parameters:
    --high_pass_filter_type: Type of high pass filter to apply to datacube. Options are "gaussian" or "butterworth". Default is a simple cut (infintely sharp filter).
    --clean_gain: What is the loop gain for the clean algorithm? Recommended 0.1-0.5. May be able to do much higher...
    --max_iterations: What is the maximum number of iterations for the clean algorithm? Default is None.
    --clean_mask_filedir: What is your clean mask?"
    --do_noiseless_clean: Do cleaning on the noiseless datacube and then add appropriate noise. Default is False."
    --clean_sigma: To what noise level do you want to clean to? Default is 3 sigma.

    
    This pipeline can also be run in a few different configurations. The following parameters allow you to either make simple moment maps,
    skip generating more expensive datacubes, or make projection masks for filtering out disks/satellites/clumps.

    Alternative Run Options:
    --make_simple_moment_maps: Do you want to create and save simple moment maps for these projections? Default is False.'
    --skip_full_datacube: Skip making the full datacube to save time. Default is false
    --skip_mock_datacube: Only make the ideal datacube

    --make_disk_cut_mask: Create a projected mask for the disk:
    --make_cell_mass_projection: Do you want to make a projection of the HI weighted cell mass at each pixel? Default is False
    --make_clump_cut_mask: Do you want to make a mask for the clump projection? Default is False"
    --clump_cut_file: What is the clump cut file to use?
    --clump_cut_suffix: What is the suffix to append to the clump cutoff file? Default is ''
    --n_clumps_to_cut: How many clumps do you want to cut? Default is 1"





Author: Cameron Trapp
Last updated 06/12/2026
'''

if __name__ == "__main__":
    args = parse_args()

    if args.log_min_column_density>0:
        args.min_column_density = np.power(10.,args.log_min_column_density)

    print("Redshift is z=",args.z)
    print("observer_distance =" ,(args.z * c / H0).in_units("kpc"))

    t0 = time.time()
    
    if args.system is not None:
        from foggie.utils.get_run_loc_etc import get_run_loc_etc
        data_dir, output_dir_default, run_loc, code_dir, trackname, haloname, spectra_dir, infofile = get_run_loc_etc(args)
        if args.data_dir is None:
            args.data_dir = data_dir
        if args.code_dir is None:
            args.code_dir = code_dir



    halo_id = args.halo #008508
    snapshot = args.snapshot #RD0042
    nref = args.run #nref11c_nref9f

    gal_name="unknown"
    if halo_id=="008508":gal_name="Tempest"
    if halo_id=="005036":gal_name="Maelstrom"
    if halo_id=="005016":gal_name="Squall"
    if halo_id=="004123":gal_name="Blizzard"
    if halo_id=="002392":gal_name="Hurricane"
    if halo_id=="002878":gal_name="Cyclone"


    snap_name = args.data_dir + "halo_"+halo_id+"/"+nref+"/"+snapshot+"/"+snapshot
    trackname = args.code_dir+"halo_tracks/"+halo_id+"/nref11n_selfshield_15/halo_track_200kpc_nref9"
    halo_c_v_name = args.code_dir+"halo_infos/"+halo_id+"/"+nref+"/halo_c_v"

    #particle_type_for_angmom = 'young_stars' ##Currently the default
    particle_type_for_angmom = 'gas' #Should be defined by gas with Temps below 1e4 K

    catalog_dir = args.code_dir + 'halo_infos/' + halo_id + '/'+nref+'/'
    smooth_AM_name = catalog_dir + 'AM_direction_smoothed'
    #smooth_AM_name = None

    ds, refine_box = foggie_load(snap_name, trackfile_name=trackname, halo_c_v_name=halo_c_v_name, do_filter_particles=True,disk_relative=True,particle_type_for_angmom=particle_type_for_angmom,smooth_AM_name = smooth_AM_name)

    fov_sphere_radius = (args.fov_kpc / 2.) * np.sqrt(2.)
    if fov_sphere_radius > np.max(refine_box['gas','radius_corrected']):
        sphere = ds.sphere(center=ds.halo_center_kpc, radius=(fov_sphere_radius, 'kpc'))
        print("\n \n Using sphere with radius",fov_sphere_radius)
        source_cut = sphere
    else:
        source_cut = refine_box

    if args.mask_disk or args.make_disk_cut_mask or args.make_simple_moment_maps:
        from foggie.clumps.clump_finder.clump_finder import clump_finder
        from foggie.clumps.clump_finder.clump_finder_argparser import get_default_args
        from foggie.clumps.clump_finder.utils_clump_finder import mask_clump
        from foggie.clumps.clump_finder.utils_clump_finder import load_clump

        print("Loading disk clump...")
        run_disk_finder = False
        #clump_file = args.output + gal_name
        clump_file = "./outputs/"+gal_name+"_"+args.snapshot+"_"+args.run+"_"
        try:
            hf_test = h5py.File(clump_file+"_Disk.h5",'r')
            hf_test.close()
        except:
             run_disk_finder = True

        if run_disk_finder:
            cf_args = get_default_args()
            cf_args.identify_disk = True
            cf_args.max_disk_void_size = np.round( (5. / 0.274)**3. ).astype(int)
            cf_args.max_disk_hole_size = int(np.round(7./0.274)) #~36
            cf_args.closing_iterations = 1
            cf_args.n_dilation_iterations = 10 #Dilate the disk mask 10 times by 1 cell each (at nref 11 -> ~2.74 kpc, each shell was a width of 0.274 kpc)
            cf_args.n_cells_per_dilation = 1
            cf_args.output = clump_file
            clump_finder(cf_args,ds,refine_box)

        hf_disk = h5py.File(clump_file+"_Disk.h5",'r')
        cell_ids_to_mask = np.array(hf_disk['cell_ids'])
        hf_disk.close()
        #for i in range(5):
        #   hf_shell = h5py.File(clump_file+"_DiskDilationShell_n"+str(i)+".h5",'r')
        #    cell_ids_to_mask = np.append(cell_ids_to_mask, np.array(hf_shell['cell_ids']))
        #    hf_shell.close()

        if args.mask_disk:
            source_cut = mask_clump(ds,clump_file+"_Disk.h5",source_cut=source_cut,clump_cell_ids=cell_ids_to_mask)
        else:
            source_cut = load_clump(ds, clump_file+"_Disk.h5",source_cut=source_cut)

        print("Disk loaded!")

    #source_cut = sphere

    args.max_projected_velocity = FindMaxProjectedVelocity(args,ds,source_cut)

    print("Max projected velocity is",args.max_projected_velocity,'km/s')


    if args.make_disk_cut_mask:
        print("Making disk cut mask...")
        disk_cut_mask = get_ideal_hi_datacube(args, ds, source_cut)
        instrument = radio_telescope(args)

        disk_cut_mask_smoothed = apply_gaussian_smoothing(disk_cut_mask.astype(float),instrument,args)
        disk_cut_mask_smoothed[disk_cut_mask_smoothed>0]=1
        disk_cut_mask_smoothed=disk_cut_mask_smoothed.astype(bool)
        hf = h5py.File(args.output+"_DiskCutMask.h5",'w')
        hf.create_dataset('disk_cut_mask',data=disk_cut_mask)
        hf.create_dataset('disk_cut_mask_smoothed',data=disk_cut_mask_smoothed)

        hf.close()
        sys.exit()

    if args.make_cell_mass_projection:
        cell_mass_projection = get_ideal_hi_datacube(args, ds, source_cut)
        instrument = radio_telescope(args)

        cell_mass_projection_smoothed = apply_gaussian_smoothing(cell_mass_projection.astype(float),instrument,args)
        hf = h5py.File(args.output+"_CellMassProjection.h5",'w')
        hf.create_dataset('cell_mass_projection',data=cell_mass_projection)
        hf.create_dataset('cell_mass_projection_smoothed',data=cell_mass_projection_smoothed)

        hf.close()

        plt.figure()
        plt.imshow(cell_mass_projection_smoothed,norm=LogNorm())
        plt.colorbar()
        plt.savefig(args.output+"_CellMassProjectionSmoothed.png")
        plt.close()
        plt.figure()
        plt.imshow(cell_mass_projection,norm=LogNorm())
        plt.colorbar()
        plt.savefig(args.output+"_CellMassProjection.png")
        plt.close()

        sys.exit()

    if args.make_clump_cut_mask:
        from foggie.clumps.clump_finder.clump_finder import clump_finder
        from foggie.clumps.clump_finder.clump_finder_argparser import get_default_args
        from foggie.clumps.clump_finder.utils_clump_finder import mask_clump
        from foggie.clumps.clump_finder.utils_clump_finder import load_clump

        print("Loading clump to mask...")
        run_disk_finder = False
        #clump_file = args.output + gal_name
        clump_cut = load_clump(ds, args.clump_cut_file+"0.h5",source_cut=source_cut)
        if args.n_clumps_to_cut>1:
            for i in range(1,args.n_clumps_to_cut):
                clump_cut = clump_cut + load_clump(ds, args.clump_cut_file+str(int(i))+".h5",source_cut=source_cut)
        source_cut = clump_cut

        print("Clump loaded!")

        print("Making disk cut mask...")
        clump_cut_mask = get_ideal_hi_datacube(args, ds, source_cut)
        instrument = radio_telescope(args)

        clump_cut_mask_smoothed = apply_gaussian_smoothing(clump_cut_mask.astype(float),instrument,args)
        clump_cut_mask_smoothed[clump_cut_mask_smoothed>0]=1
        clump_cut_mask_smoothed=clump_cut_mask_smoothed.astype(bool)
        hf = h5py.File(args.output+args.clump_cut_suffix+".h5",'w')
        hf.create_dataset('clump_cut_mask',data=clump_cut_mask)
        hf.create_dataset('clump_cut_mask_smoothed',data=clump_cut_mask_smoothed)
        hf.close()
        plt.figure()
        plt.imshow(np.sum(clump_cut_mask,axis=2))
        plt.savefig(args.output+"_SatCutMask.png")
        plt.close()

        plt.figure()
        plt.imshow(np.sum(clump_cut_mask_smoothed,axis=2))
        plt.savefig(args.output+"_SatCutMaskSmoothed.png")
        plt.close()

        sys.exit()    


    if args.output is None and not args.skip_full_datacube:
        ifu = get_ideal_hi_datacube(args, ds, source_cut)
    elif not args.skip_full_datacube:
        if args.force_ideal_ifu:
            ifu = get_ideal_hi_datacube(args, ds, source_cut)
            hf = h5py.File(args.output+".h5", "w")
            dset = hf.create_dataset("ifu", data=ifu.value)
            dset.attrs["unit"] = str(ifu.units)
            hf.close()
        else:
            try:
                hf = h5py.File(args.output+".h5", "r")
                ifu = hf["ifu"][:] * u.Unit(hf["ifu"].attrs["unit"])
                print('Ifu loaded as',ifu)
                hf.close()
                args.skip_full_datacube = True
                get_ideal_hi_datacube(args,ds,source_cut)   
            except:
                print("Could not load existing ifu at",args.output+".h5 - will create a new one!")
                ifu = get_ideal_hi_datacube(args, ds, source_cut)
                hf = h5py.File(args.output+".h5", "w")
                dset = hf.create_dataset("ifu", data=ifu.value)
                dset.attrs["unit"] = str(ifu.units)
                hf.close()

    #make_moment_map_figures(args,ifu,args.output+"_ideal")

    if args.mock_suffix is not None:
        args.output = args.output+args.mock_suffix
       
    if args.make_simple_moment_maps:
      radial_profiles = []
      radii = []
      for inc in [60]:
        for pa in  np.linspace(0,355,72):
            args.inclination = inc
            args.position_angle = pa
            if True: source_cut = ds.sphere(center=ds.halo_center_kpc, radius=(500, 'kpc'))

            pixel_radii, radial_profile, dens_proj, vdopp_proj = get_ideal_hi_datacube(args, ds, source_cut)#,moment_map_filename = './ideal_moment_maps/i20/RD0038/moment_map_i'+str(inc).zfill(2)+'_pa'+str(int(pa)).zfill(3))
            radial_profiles.append(radial_profile)
            radii.append(pixel_radii)
      # create_gif_from_pngs("./ideal_moment_maps/i20/RD0038","Tempest_Doppler_Vel_i20.gif",duration = 150)
   
      radial_profiles = np.array(radial_profiles)
      mean_NHI = np.mean(radial_profiles,axis=0)
      min_NHI = np.min(radial_profiles,axis=0)
      max_NHI = np.max(radial_profiles,axis=0)
      std_NHI = np.std(radial_profiles,axis=0)
      plt.figure()
      plt.plot(radii[0],mean_NHI,'k',lw=3,label='Mean')
      plt.plot(radii[0], np.power(10.,20.93-1.67*np.log10(radii[0]/10.)),'r--',lw=2,label='Power Law')
      plt.fill_between(radii[0],min_NHI,max_NHI,alpha=0.5,label='Range',color='k')
      plt.xlabel("Radius [kpc]")
      plt.ylabel("log(N$_{\\rm HI}$/cm$^{-2}$)")
      plt.yscale('log') 
      plt.xscale('log')
      plt.xlim([0.8e1,1.2e2])
      plt.ylim([1e16,1e25])
      plt.legend()
      plt.savefig("./ideal_moment_maps/"+args.halo+"_NHI_radial_profile_min_max.png",bbox_inches='tight')
      plt.close()

      plt.figure()
      plt.plot(radii[0],mean_NHI,'k',lw=3,label='Mean')
      plt.fill_between(radii[0],mean_NHI-std_NHI,mean_NHI+std_NHI,alpha=0.5,label='Stdv.',color='k')
      plt.plot(radii[0], np.power(10.,20.93-1.67*np.log10(radii[0]/10.)),'r--',lw=2,label='Power Law')
      plt.xlabel("Radius [kpc]")
      plt.ylabel("log(N$_{\\rm HI}$/cm$^{-2}$)")
      plt.yscale('log') 
      plt.xscale('log')
      plt.xlim([0.8e1,1.2e2])
      plt.ylim([1e16,1e25])
      plt.legend()
      plt.savefig("./ideal_moment_maps/"+args.halo+"_NHI_radial_profile_stdv.png",bbox_inches='tight')
      plt.close()

      hf=h5py.File("./ideal_moment_maps/"+args.halo+"_NHI_radial_profile.h5",'w')
      hf.create_dataset('radii',data=radii[0])
      hf.create_dataset('mean_NHI',data=mean_NHI)
      hf.create_dataset('min_NHI',data=min_NHI)
      hf.create_dataset('max_NHI',data=max_NHI)
      hf.create_dataset('std_NHI',data=std_NHI)
      hf.create_dataset('all_curves',data=radial_profiles)
      hf.close()
      '''
      for itr in range(0,120):
        pa = 360.*itr/120
        inc = 45 + 20*np.sin(2.*np.pi*itr/40.)
        args.inclination = inc
        args.position_angle = pa
        ifu = get_ideal_hi_datacube(args, ds, source_cut,moment_map_filename = "./ideal_moment_maps/multi_inc/RD0038/moment_map_itr"+str(itr).zfill(3))
      create_gif_from_pngs("./ideal_moment_maps/multi_inc/RD0038","Tempest_Doppler_Vel_wobbles.gif",duration = 150)

      '''



    print("Ideal IFU")
    print(np.max(ifu))
    print(np.shape(ifu))
    plt.figure()
    nz = args.obs_channels
    nu = np.linspace(args.base_freq_range[1].in_units('1/s'),args.base_freq_range[0].in_units('1/s'),nz)
    nx,ny,nz = np.shape(ifu)
    dnu = ( np.max(nu)-np.min(nu) ) / np.size(nu)
    image  = np.sum(ifu,axis=2)*dnu
    ideal_vmin = np.min(image[image>0]) / 10.
    ideal_vmax = np.max(image)
    plt.imshow(image,norm=LogNorm(vmin=ideal_vmin),cmap='inferno')
    plt.colorbar()
    plt.savefig(args.output+"_IdealImage.png")
    plt.close()

    plt.figure()
    nz = args.obs_channels
    nu = np.linspace(args.base_freq_range[1].in_units('1/s'),args.base_freq_range[0].in_units('1/s'),nz)
    nx,ny,nz = np.shape(ifu)
    dnu = ( np.max(nu)-np.min(nu) ) / np.size(nu)
    image  = np.sum(ifu,axis=2)*dnu
    image[image<args.min_column_density] = ideal_vmin
    ideal_vmin = np.min(image[image>0]) / 10.
    ideal_vmax = np.max(image)

    image[image<args.min_column_density] = ideal_vmin

    plt.imshow(image,norm=LogNorm(vmin=ideal_vmin),cmap='inferno')
    plt.colorbar()
    plt.savefig(args.output+"_IdealImage_WithColDensCutoff.png")
    plt.close()

    if args.skip_mock_datacube:
        print("Time to run=",time.time()-t0)
        sys.exit()

        sys.exit(0)

    clean_mask = None
    if args.use_clean_mask:
        clean_mask = ((ifu*dnu*args.obs_channels>=args.min_column_density))
        if args.clean_mask_filedir is not None:
            clean_mask = LoadSofiaMask(args.clean_mask_filedir+"_filtered_mask.fits")

        hf.close()

    if args.test_calibrator_clean:
        mifu = np.max(ifu)
        uifu = ifu.units
        ifu = np.zeros((np.shape(ifu))) * uifu
        ifu[nx//2+nx//6,ny//2-ny//10,:] = mifu #Point source test!



    clean_image, clean_image_with_residuals, unfiltered_image = get_mock_hi_datacube(args, ds, ifu,clean_mask)

    instrument = radio_telescope(args)

    print("Mock IFU")
    print(np.max(clean_image))
    print(np.shape(clean_image))
    plt.figure()
    image  = np.sum(clean_image,axis=2)*dnu
    vmin = args.min_column_density/(50.)
    vmax = 1e22    #image[image>0]=np.max(image)
    plt.imshow(image,norm=LogNorm(vmin=vmin,vmax=vmax),cmap='inferno')
    plt.colorbar()
    plt.savefig(args.output+"_FilteredImage.png")
    plt.close()

    plt.figure()
    image  = np.sum(clean_image_with_residuals,axis=2)*dnu
    #image[image>0]=np.max(image)
    plt.imshow(image,norm=LogNorm(vmin=vmin,vmax=vmax),cmap='inferno')
    plt.colorbar()
    plt.savefig(args.output+"_FilteredImageWithResiduals.png")
    plt.close()

    plt.figure()
    noise_level = args.min_column_density / u.cm / u.cm / dnu / args.sigma_noise_level #5-sigma detection
    image  = np.sum(clean_image_with_residuals,axis=2)*dnu
    image[image<args.sigma_noise_level * noise_level*dnu * instrument.integrated_channels_for_noise] = 0
    plt.imshow(image,norm=LogNorm(vmin=vmin,vmax=vmax),cmap='inferno')
    plt.colorbar()
    plt.savefig(args.output+"_FilteredImageWithResiduals_with"+str(args.sigma_noise_level)+"SigmaColDensCut.png")
    plt.close()

    observer_distance = (args.z * c / H0).in_units("kpc")


    print("Saving to hdf5...")
    hf = h5py.File(args.output+"_AllImages.h5",'w')
    hf.create_dataset('ideal_ifu',data=ifu)
    hf.create_dataset('mock_ifu',data=clean_image)
    hf.create_dataset('mock_ifu_with_residuals',data=clean_image_with_residuals)
    hf.create_dataset('unfiltered_mock_ifu',data=unfiltered_image)
    #hf.create_dataset('dirty_ifu',data=dirty_image)
    hf.create_dataset('nu',data=nu)
    hf.create_dataset('obs_spatial_res_arcseconds',data=instrument.obs_spatial_res)
    hf.create_dataset('obs_spec_res_kmps',data=instrument.obs_spec_res)
    hf.create_dataset('observer_distance',data=observer_distance.in_units('kpc').v)
    hf.create_dataset('noise_level',data=noise_level)
    tmp, primary_beam = apply_primary_beam_correction(ifu,instrument,args)
    if primary_beam is not None: hf.create_dataset('primary_beam',data=primary_beam)

    hf.create_dataset('dnu',data=dnu)
    hf.create_dataset('fov_kpc',data=args.fov_kpc.in_units('kpc').v)
    hf.create_dataset('spec_res_kms',data=instrument.spec_res_kms)
    hf.close()


    if True:
        #Delete all temporary slices in /tmp_slices
        os.system("rm -rf " + "./outputs/tmp_slices")
        os.system("mkdir " + " ./outputs/tmp_slices")

    print("Time to run=",time.time()-t0)
