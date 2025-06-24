from astropy.io import fits
import numpy as np
import h5py
import unyt as u

from foggie.galaxy_mocks.mock_hi_imager.HICubeHeader import * #Constants

import argparse


def parse_args():
    '''Parse command line arguments. Returns args object.'''
    parser = argparse.ArgumentParser()
    # Optional arguments:
    parser.add_argument('--halo', metavar='halo', type=str, action='store', \
                        help='Which halo id? Default is 008508 (Tempest).')
    parser.set_defaults(halo="008508")
    
    parser.add_argument('--input_dir', metavar='input_dir', type=str, action='store', \
                        help='Which input directory? Default is /Users/ctrapp/Documents/foggie_analysis/analysis_tools/synthetic_HI_imager/outputs/.')
    parser.set_defaults(input_dir="/Users/ctrapp/Documents/foggie_analysis/analysis_tools/synthetic_HI_imager/outputs/")

    parser.add_argument('--input_survey', metavar='input_survey', type=str, action='store', \
                        help='Which survey? Default is Grid_MhongooseHR_NHI5e18_bmin29_RD0042.')
    parser.set_defaults(input_survey="Grid_MhongooseHR_NHI5e18_bmin29_RD0042")

    parser.add_argument('--image_type', metavar='image_type', type=str, action='store', \
                        help="Which image do you want to use? Options are 'smoothed','filtered', 'noisy', or 'ideal'. Default is smoothed.")
    parser.set_defaults(image_type='smoothed')

    parser.add_argument('--bminstr', metavar='bminstr', type=str, action='store', \
                        help="Which bmin string? Default is '' (empty string).")
    parser.set_defaults(bminstr='')

    parser.add_argument('--renormalize_noise', metavar='renormalize_noise', type=bool, action='store', \
                        help="Set the noise average to 0? Default is False.")
    parser.set_defaults(renormalize_noise=False)

    args = parser.parse_args()
    return args


args = parse_args()



if args.halo == "008508":
    gal_name = "tempest";GalName="Tempest"
elif args.halo == "005036":
    gal_name = "maelstrom";GalName="Maelstrom"
elif args.halo == "005016":
    gal_name = "squall";GalName="Squall"
elif args.halo == "004123":
    gal_name = "blizzard";GalName="Blizzard"
elif args.halo == "002392":
    gal_name = "hurricane";GalName="Hurricane"
elif args.halo == "002878":
    gal_name = "cyclone";GalName="Cyclone"

if args.bminstr == "-1": args.bminstr = ''


input_filebase = args.input_dir#"/Users/ctrapp/Documents/foggie_analysis/analysis_tools/synthetic_HI_imager/outputs/"
input_survey = args.input_survey#"Grid_MhongooseHR_NHI5e18_bmin29_RD0042"

print(input_filebase + GalName + input_survey +args.bminstr+"_AllImages.h5")
hf = h5py.File(input_filebase + GalName + input_survey +args.bminstr+"_AllImages.h5",'r')
print(hf.keys())

output_filename = input_filebase+GalName+input_survey+args.bminstr+"_"+args.image_type+".fits"



if args.image_type == "filtered":
    new_datacube = hf['mock_ifu_with_residuals'][...]
elif args.image_type == "smoothed":
    new_datacube = hf['unfiltered_mock_ifu'][...]
elif args.image_type == "noisy":
    new_datacube = hf['ideal_ifu'][...]
    nu = hf['nu'][...]
    dnu = hf['dnu'][...]
    noise_level = hf['noise_level'][...]

    nx,ny,nz=np.shape(new_datacube)

    N = np.size(new_datacube)

    noise_flux = np.random.normal(0.0,noise_level, N).reshape(np.shape(new_datacube)) #Gaussian
    new_datacube = new_datacube + noise_flux
elif args.image_type == "ideal":
    new_datacube = hf['ideal_ifu'][...]


if args.renormalize_noise:
    #Renormalize the noise to have a mean of 0
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    nx,ny,nz = np.shape(new_datacube)
    print("Shape is",nx,ny,nz)
    new_datacube = new_datacube - 1*np.mean(new_datacube[0:int(nx/4),0:int(ny/4),:])
    plt.figure()
    plt.imshow(np.sum(new_datacube, axis=2), cmap='inferno',norm=LogNorm())
    plt.show()

nx,ny,nspec = np.shape(new_datacube)

#Get Observational Parameters
fov_kpc = hf['fov_kpc'][...]
observer_distance = hf['observer_distance'][...] * u.kpc#in kpc
z = observer_distance * H0 / c

obs_spatial_res_arcseconds = hf['obs_spatial_res_arcseconds'][...]
dnu = hf['dnu'][...]

print(hf.keys())
dnu_kmps = hf['spec_res_kms'][...]
dnu_mps = dnu_kmps * 1000.

hf.close()



input_filename = "/Users/ctrapp/Documents/foggie_analysis/analysis_tools/tilted_ring_fits/NGC_2403_NA_CUBE_THINGS.fits"
with fits.open(input_filename) as hdul:
    hdul.info()  # Show HDU list
    header = hdul[0].header
    data = hdul[0].data  # This is a NumPy array


# Step 4: Save to a new FITS file
#output_filename = "/Users/ctrapp/Documents/foggie_analysis/analysis_tools/tilted_ring_fits/"+gal_name+"_NHI18_unfiltered_mock_ifu.fits"

bmaj = obs_spatial_res_arcseconds / 3600.#0.001666666666666666
bmin = obs_spatial_res_arcseconds / 3600.#0.001666666666666666 

image_array = (new_datacube * dnu)


new_image = np.zeros((1,nspec,ny,nx))
for ks in range(0,nspec):
    new_image[0,ks,:,:] = image_array[:,:,ks]

observer_distance = (z * c / H0).in_units("kpc")

fov_deg = fov_kpc / observer_distance.in_units('kpc').v * 180./np.pi

print("Fov_kpc=",fov_kpc)
print("fov_deg=",fov_deg)

header['NAXIS1'] = nx
header['NAXIS2'] = ny
header['NAXIS3'] = nspec
          
header['CRPIX1'] = int(nx/2)
header['CDELT1'] = -fov_deg / nx
header['CUNIT1'] = 'DEGREE            '

header['CRPIX2'] = int(ny/2)
header['CDELT2'] = fov_deg / ny
header['CUNIT2'] = 'DEGREE            '

header['CRVAL3'] = 0
header['CRPIX3'] = int(nspec/2)
header['CDELT3'] = -dnu_mps
header['CUNIT3'] = 'M/S               ' 

header['BMAJ'] = bmaj                                                  
header['BMIN'] = bmin    
header['OBJECT'] = gal_name
header['OBSERVER'] = 'ctrapp  '

hdu = fits.PrimaryHDU(data=new_image, header=header)
hdu.writeto(output_filename, overwrite=True)

print("SHAPE OF IMAGE =",nx,ny)
print("bmaj=",bmaj)
print("bmaj in arsec=",bmaj*3600)
print("fov_kpc=",fov_kpc)
print("arcsec per pixel=",fov_deg / nx * 3600)
print("kpc per pixel=",fov_kpc / nx)

print(f"\nSaved modified FITS file as {output_filename}")