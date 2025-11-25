import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import unyt as u

from make_hi_datacube_v2 import load_line_properties
from astropy.io import fits
import sys

c = 3e5 * u.km / u.s  # km/s
H0 = 70. * u.km / u.s / u.Mpc  # km/s/Mpc Hubble's constant
h = 4.135667696e-15 * u.eV * u.s #eV * s
Mpc_to_m = 3.08e22
Mpc_to_cm = Mpc_to_m * 100
kpc_to_cm = Mpc_to_cm / 1000
kb = 8.617333262e-5 * u.eV / u.K
m_e = 9.1094*np.power(10.,-28.) * u.g #grams
e = 4.8032*np.power(10.,-10.) * u.statC #cm^(3/2) * g^(1/2) * s^(-1)
amu = 1.6735575*np.power(10.,-24) * u.g

arcsec_to_rad = 1./60./60. * np.pi/180.

#For each galaxy survey combination
## Load the ideal image, sum to get ~mass
## For each image (noisy, smoothed, filtered)
### Load the ifu
### Load the mask
### Project the masked image to get column density
### Calculate the sum of the masked image to get ~mass
### Return projected column density map, mass ratio

#Then, save image as .png and hdf5 with mass information
#Compile all Mhongoose HR images into a 4x6 plot for all galaxies (repeat for all surveys just for fun)
#Put mass ratios in table

####EDIT TO READ FROM FILE IN FUTURE RUNS
species_mass,gamma_ul, A_ul, Elevels, Glevels, n_u_fraction, n_l_fraction = load_line_properties("HI_21cm")
HI_21cm_freq = (Elevels[1]-Elevels[0])/h #in Hz
HI_21cm_lmbda_m = 3.0*np.power(10,8)*u.m/u.s / HI_21cm_freq #in m
THINGS_BW_freq = 1.56e6 / u.s
obs_freq_range = [HI_21cm_freq - THINGS_BW_freq/2., HI_21cm_freq + THINGS_BW_freq/2.]
nchannels=128

nu = np.linspace(obs_freq_range[1].in_units('1/s'),obs_freq_range[0].in_units('1/s'),nchannels)
dnu = ( np.max(nu)-np.min(nu) ) / np.size(nu)
##########################

sigma = 5.


def LoadFromFitsFile(filedir):
    with fits.open(filedir) as hdul:
        ifu = hdul[0].data

    #tmp,nspec,nx,ny = np.shape(ifu)
    return np.transpose(ifu[0],(1,2,0))


def LoadSofiaMask(filedir):
    with fits.open(filedir) as hdul:
        #hdul.info()  # Show HDU list
        header = hdul[0].header
        mask = hdul[0].data  # This is a NumPy array

    #nspec,nx,ny = np.shape(mask)
    return np.transpose(mask, (1,2,0))


    
def CalculateMomentMaps(ifu,dnu,moment=None,spec_res_kms=5.2/2.):
    m0_map = np.sum(ifu, axis=2) * dnu

    nx,ny,nspec = np.shape(ifu)
    bw_kms = nspec * spec_res_kms
    velocities = np.linspace(-bw_kms/2. , bw_kms/2., nspec)
    
    if moment==0:
        return m0_map

    m1_map = np.divide( np.sum(np.multiply(ifu, velocities[np.newaxis,np.newaxis,:]),axis=2) , m0_map ) * dnu
    if moment==1:
        return m1_map

    m2_term = np.subtract( velocities[np.newaxis,np.newaxis,:] , m1_map[:,:,np.newaxis] )
    m2_term = np.multiply(m2_term,m2_term)
   # plt.figure()
    #plt.imshow(np.sqrt( np.divide(np.sum( np.multiply(ifu, m2_term),axis=2) , m0_map) * dnu),vmin=0,vmax=150)
   # plt.colorbar()
    #plt.show()
    m2_map = np.sqrt( np.divide(np.sum( np.multiply(ifu, m2_term),axis=2) , m0_map) * dnu) 
    m2_map[np.isnan(m2_map)] = 0
        
    
    if moment==2:
        return m2_map

    return m0_map, m1_map, m2_map

def ApplyPrimaryBeamCorrection(ifu, primary_beam):
    if primary_beam is None:
        return ifu

    nx,ny,nspec = np.shape(ifu)
    ifu = np.divide(ifu, primary_beam[:,:,np.newaxis])
    #for ks in range(0,nspec):
    #    ifu[:,:,ks] = np.divide(ifu[:,:,ks], primary_beam)
        
    return ifu
    

inputDir = "/Users/ctrapp/Documents/foggie_analysis/analysis_tools/synthetic_HI_imager/aitoff_runs/tmp/"
GalName = sys.argv[3]
Survey = '_MhongooseLR_10Mpc_NHI1e18_RD0042'
DiskSuffix  = '_MhongooseLR_10Mpc'
filebase = inputDir+GalName+Survey
inputFile = filebase+"_GaussianHPF_AllImages.h5"
noisyFitsFile = filebase+"_noisy.fits"
          
diskFile = inputDir+GalName+DiskSuffix+"_DiskCutMask.h5"
hf_disk = h5py.File(diskFile,'r')
print(hf_disk.keys())
disk_mask = hf_disk['disk_cut_mask_smoothed'][...]
hf_disk.close()
disk_masses=[]
cgm_masses=[]

print("Shape of disk mask=",np.shape(disk_mask))

          
hf = h5py.File(inputFile,'r')
print(hf.keys())
primary_beam = None
try:
    primary_beam = hf['primary_beam'][...] #FAST doesnt have this
except:
    print("Could not load primary beam...")
ideal_ifu = hf['ideal_ifu'][...]
dnu = hf['dnu'][...]
spec_res_kms = hf['spec_res_kms'][...]
print("spec_res_kms set to",spec_res_kms)
ideal_map = np.sum(ideal_ifu,axis=2) * dnu
print("Shape of ideal ifu (summed on 2)=",np.shape(ideal_ifu))
ideal_disk_mass = np.sum(ideal_map[disk_mask])
ideal_cgm_mass = np.sum(ideal_map)-ideal_disk_mass

          
smoothed_ifu = hf['unfiltered_mock_ifu'][...]
smoothed_mask = LoadSofiaMask(filebase+"_smoothed_mask.fits")
smoothed_ifu[smoothed_mask<=0] = 0
smoothed_ifu = ApplyPrimaryBeamCorrection(smoothed_ifu,primary_beam)
smoothed_map = np.sum(smoothed_ifu,axis=2) * dnu
smoothed_disk_mass = np.sum(smoothed_map[disk_mask])
smoothed_cgm_mass = np.sum(smoothed_map)-smoothed_disk_mass


filtered_ifu = hf['mock_ifu_with_residuals'][...]
filtered_mask = LoadSofiaMask(filebase+"_filtered_mask.fits")
filtered_ifu[filtered_mask<=0] = 0

filtered_ifu = ApplyPrimaryBeamCorrection(filtered_ifu,primary_beam)
filtered_map = np.sum(filtered_ifu,axis=2) * dnu
filtered_map = filtered_map * np.max(smoothed_map)/np.max(filtered_map)
filtered_disk_mass = np.sum(filtered_map[disk_mask])
filtered_cgm_mass = np.sum(filtered_map)-filtered_disk_mass

          
hf.close()



output_filename = "/Users/ctrapp/Documents/foggie_analysis/analysis_tools/synthetic_HI_imager/aitoff_runs/"
output_filename += GalName+"MhongooseLR_10Mpc_NHI1e18_RD0042_i"+sys.argv[1]+"_pa"+sys.argv[2]

hfo = h5py.File(output_filename+".h5",'w')
hfo.create_dataset('ideal_map', data=ideal_map)
hfo.create_dataset('smoothed_map', data=smoothed_map)
hfo.create_dataset('filtered_map', data=filtered_map)
hfo.create_dataset('ideal_disk_mass', data=ideal_disk_mass)
hfo.create_dataset('smoothed_disk_mass', data=smoothed_disk_mass)
hfo.create_dataset('filtered_disk_mass', data=filtered_disk_mass)
hfo.create_dataset('ideal_cgm_mass', data=ideal_cgm_mass)
hfo.create_dataset('smoothed_cgm_mass', data=smoothed_cgm_mass)
hfo.create_dataset('filtered_cgm_mass', data=filtered_cgm_mass)
hfo.close()

plt.figure()
plt.imshow(np.rot90(ideal_map), norm=LogNorm(vmin=1e14, vmax=1e22), cmap='inferno')
plt.colorbar()
plt.title("Ideal Map")
plt.savefig(output_filename+"_ideal_map.png", dpi=300, bbox_inches='tight')
plt.close()
 
plt.figure()
plt.imshow(np.rot90(filtered_map), norm=LogNorm(vmin=1e14, vmax=1e22), cmap='inferno')
plt.colorbar()
plt.title("Filtered Map")
plt.savefig(output_filename+"_filtered_map.png", dpi=300, bbox_inches='tight')
plt.close()
 
