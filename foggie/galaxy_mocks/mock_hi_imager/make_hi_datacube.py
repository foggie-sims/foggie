import numpy as np
import unyt as u
import h5py
import time
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from foggie.utils.foggie_load import foggie_load
from foggie.utils.consistency import *
from foggie.clumps.clump_finder import *

from foggie.galaxy_mocks.mock_hi_imager.HICubeHeader import * #Constants
from foggie.galaxy_mocks.mock_hi_imager.hi_datacube_arg_parser import parse_args
from foggie.galaxy_mocks.mock_hi_imager.radio_telescopes import radio_telescope

from foggie.galaxy_mocks.mock_hi_imager.utils_hi_datacube import FindMaxProjectedVelocity

from foggie.galaxy_mocks.mock_hi_imager.make_ideal_datacubes import get_ideal_hi_datacube
from foggie.galaxy_mocks.mock_hi_imager.make_mock_datacubes import get_mock_hi_datacube

from foggie.galaxy_mocks.mock_hi_imager.make_mock_datacubes import apply_gaussian_smoothing
from foggie.galaxy_mocks.mock_hi_imager.make_mock_datacubes import apply_primary_beam_correction

'''
Functions to make synthetic HI datacubes targeting specific pre- or user-defined radio surveys.

The core functions are in make_ideal_datacubes.py, make_mock_datacubes.py, and make_spatially_filtered_datacubes.py

make_ideal_datacubes.py calculates the emission profiles for the given projection and creates an ideal noiseless datacube at the given pixel resolution
make_mock_datacubes.py adds gaussian noise and applies gaussian smoothing to make the typical mock datacube
make_spatially_filtered_datacubes.py takes this one step further and simulates the effects of interferometry by filtering out diffuse components of the image


Author: Cameron Trapp
Last updated 06/17/2025
'''

if __name__ == "__main__":
    ### Load the data and do some book keeping ###
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

    ds, refine_box = foggie_load(snap_name, trackname, halo_c_v_name=halo_c_v_name, do_filter_particles=True,disk_relative=True,particle_type_for_angmom=particle_type_for_angmom,smooth_AM_name = smooth_AM_name)

    fov_sphere_radius = (args.fov_kpc / 2.) * np.sqrt(2.)
    if fov_sphere_radius > np.max(refine_box['gas','radius_corrected']):
        sphere = ds.sphere(center=ds.halo_center_kpc, radius=(fov_sphere_radius, 'kpc'))
        print("\n \n Using sphere with radius",fov_sphere_radius)
        source_cut = sphere
    else:
        source_cut = refine_box


    ### Calculate the disk mask if needed ###
    if args.mask_disk or args.make_disk_cut_mask:
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
        for i in range(5):
            hf_shell = h5py.File(clump_file+"_DiskDilationShell_n"+str(i)+".h5",'r')
            cell_ids_to_mask = np.append(cell_ids_to_mask, np.array(hf_shell['cell_ids']))
            hf_shell.close()

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
        plt.figure()
        plt.imshow(disk_cut_mask)
        plt.savefig(args.output+"_DiskCutMask.png")
        plt.close()

        plt.figure()
        plt.imshow(disk_cut_mask_smoothed)
        plt.savefig(args.output+"_DiskCutMaskSmoothed.png")
        plt.close()

        sys.exit()

    #### Create the ideal datacube ####
    if not args.skip_full_datacube:
        if args.force_ideal_ifu:
            ifu = get_ideal_hi_datacube(args, ds, source_cut)
            hf = h5py.File(args.output+".h5", "w")
            dset = hf.create_dataset("ifu", data=ifu.value)
            dset.attrs["unit"] = str(ifu.units)
            hf.close()
        else:
            try:
                ### Try to load if already calculated
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


    if args.mock_suffix is not None:
        args.output = args.output+args.mock_suffix
       
    if args.make_simple_moment_maps:
      ###If making simple moment maps save and exit here!
      radial_profiles = []
      radii = []
      for inc in [20]:
        for pa in  np.linspace(0,355,72):
            args.inclination = inc
            args.position_angle = pa
            source_cut = ds.sphere(center=ds.halo_center_kpc, radius=(500, 'kpc'))

            pixel_radii, radial_profile, dens_proj, vdopp_proj = get_ideal_hi_datacube(args, ds, source_cut)#,moment_map_filename = './ideal_moment_maps/i20/RD0038/moment_map_i'+str(inc).zfill(2)+'_pa'+str(int(pa)).zfill(3))
            radial_profiles.append(radial_profile)
            radii.append(pixel_radii)
   
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

    ### Make some basic figures
    print("Ideal IFU")
    print(np.max(ifu))
    print(np.shape(ifu))
    plt.figure()
    nz = args.base_channels
    nu = np.linspace(args.base_freq_range[1].in_units('1/s'),args.base_freq_range[0].in_units('1/s'),nz)
    nx,ny,nz = np.shape(ifu)
    dnu = ( np.max(nu)-np.min(nu) ) / np.size(nu)
    image  = np.sum(ifu,axis=2)*dnu
    ideal_vmin = np.min(image[image>0]) / 10.
    ideal_vmax = np.max(image)
    plt.imshow(image,norm=LogNorm(vmin=ideal_vmin),cmap='inferno')
    plt.colorbar()
    plt.savefig(args.output+"_IdealImage.png")
    #plt.show()
    plt.close()

    plt.figure()
    nz = args.base_channels
    nu = np.linspace(args.base_freq_range[1].in_units('1/s'),args.base_freq_range[0].in_units('1/s'),nz)
    nx,ny,nz = np.shape(ifu)
    dnu = ( np.max(nu)-np.min(nu) ) / np.size(nu)
    image  = np.sum(ifu,axis=2)*dnu
    #clean_mask = np.where(image>=args.min_column_density)
    image[image<args.min_column_density] = ideal_vmin
    ideal_vmin = np.min(image[image>0]) / 10.
    ideal_vmax = np.max(image)

    image[image<args.min_column_density] = ideal_vmin

    plt.imshow(image,norm=LogNorm(vmin=ideal_vmin),cmap='inferno')
    plt.colorbar()
    plt.savefig(args.output+"_IdealImage_WithColDensCutoff.png")
    #plt.show()
    plt.close()

    if args.skip_mock_datacube:
        print("Time to run=",time.time()-t0)
        sys.exit()

        sys.exit(0)


    #### Make the Mock Datacubes ####
    clean_mask = None
    if args.use_clean_mask:
        clean_mask = ((ifu*dnu*args.base_channels>=args.min_column_density))

    if args.test_calibrator_clean:
        mifu = np.max(ifu)
        uifu = ifu.units
        ifu = np.zeros((np.shape(ifu))) * uifu
        ifu[nx//2+nx//6,ny//2-ny//10,:] = mifu #Point source test!


    clean_image, clean_image_with_residuals, unfiltered_image = get_mock_hi_datacube(args, ds, ifu,clean_mask)

    #### Make some plots and save everything ####

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

    #hf.create_dataset('fourier_image',data=fourier_image)
    #hf.create_dataset('filtered_fourier_image',data=filtered_fourier_image)
   # hf.create_dataset('mock_column_densities',data=mock_column_densities)
  #  hf.create_dataset('ideal_column_densities',data=ideal_column_densities)
    #hf.create_dataset('args',data=args)
    #hf.create_dataset('instrument',data=instrument)
   # hf.create_dataset('actual_column_densities',data=density_projection)
    hf.close()


    if True:
        #Delete all temporary slices in /tmp_slices
        os.system("rm -rf " + "./outputs/tmp_slices")
        os.system("mkdir " + " ./outputs/tmp_slices")

    print("Time to run=",time.time()-t0)
