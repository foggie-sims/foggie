import numpy as np
from scipy import interpolate

root_directory = "./"

def load_sky_backgrounds(wavelengths , earth_shine_percentage=0.5, zodiacal_percentage = 0.5):
    '''
    Interpolate sky backgrounds from earthshine and zodial light. From https://hst-docs.stsci.edu/wfc3ihb/chapter-9-wfc3-exposure-time-calculation/9-7-sky-background
    The earth shine is the background light from the earth and ranges from 0% to 100%
    The zodiacal light is background light from dust illuminated by the sun and ranges from magnitudes of 22.1 (at latitude ~0) to 23.3 (at latitude ~90)
    '''


    filedir = root_directory + "conf/sky_backgrounds.txt"
    
    print("Reading sky background flux from",filedir)
    data = np.loadtxt(filedir)
    
    background_wavelength = data[:,0]
    earth_shine = data[:,1] #light from earth. Ranges from 0% to 100% depending on position of HST
    zodiacal_light = data[:,2] #Sunlight reflected by dust. Ranges from ~10% to 100% depending on orbit. See link above for info
    
    
    background_flux = earth_shine * earth_shine_percentage + zodiacal_light * zodiacal_percentage
    
    interp = interpolate.interp1d(background_wavelength, background_flux, bounds_error = False, fill_value = 0.)
    background_interpolate = interp(wavelengths)  
    
    print("Sky backgrounds are:",background_interpolate)
    
    return background_interpolate
