import numpy as np
import sys
import os
import matplotlib as mpl
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})
mpl.rcParams['font.family'] = 'serif'
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import AxesGrid
import matplotlib.image as mpimg
from yt.units import kpc
from astropy.table import Table
from astropy.io import fits
from astropy.utils.data import download_file
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.convolution import Gaussian2DKernel
from astropy.modeling.models import Lorentz1D
from astropy.convolution import convolve_fft
from astropy.constants import c as speedoflight
import pickle
from functools import partial
import datashader as dshader
import datashader.transfer_functions as tf
from datashader import reductions
from datashader.utils import export_image
import pandas as pd
import trident
from foggie.utils.get_halo_center import get_halo_center
from foggie.utils.get_refine_box import get_refine_box
from foggie.utils.get_proper_box_size import get_proper_box_size
from foggie.utils.consistency import *
from foggie.utils.enzoGalaxyProps import find_rvirial
import foggie.utils.get_halo_info as ghi
import yt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator)
import csv
from collections import OrderedDict as odict
mpl.rcParams['axes.linewidth']=1
mpl.rcParams['axes.edgecolor']='k'
from astropy.io import fits
from ccdproc import rebin as ccdrebin
from foggie.utils.foggie_load import foggie_load
from astropy.cosmology import FlatLambdaCDM
from yt.analysis_modules.star_analysis.api import StarFormationRate
from foggie.utils.get_run_loc_etc import get_run_loc_etc
import argparse



def parse_args():
    '''Parse command line arguments. Returns args object.
    NOTE: Need to move command-line argument parsing to separate file.'''

    parser = argparse.ArgumentParser()
    # Optional arguments:
    parser.add_argument('--halo', metavar='halo', type=str, action='store', \
                        help='Which halo? Default is 8508 (Tempest)')
    parser.set_defaults(halo='8508')

    parser.add_argument('--run', metavar='run', type=str, action='store', \
                        help='Which run? Default is nref11c_nref9f')
    parser.set_defaults(run='nref11c_nref9f')

    parser.add_argument('--output', metavar='output', type=str, action='store', \
                        help='Which output? Default is RD0032')
    parser.set_defaults(output='RD0032')

    parser.add_argument('--system', metavar='system', type=str, action='store', \
                        help='Which system are you on? Default is ramona')
    parser.set_defaults(system='ramona')

    parser.add_argument('--pwd', dest='pwd', action='store_true', \
                        help='Just use the working directory? Default is no')
    parser.set_defaults(pwd=False)

    parser.add_argument('--instrument', metavar='instrument', type=str, action='store', \
                        help='Which instrument? Default is COS-G130M. Choose KCWI or COS-G130M')
    parser.set_defaults(instrument='COS-G130M')

    parser.add_argument('--line_list', metavar='line_list', type=str, action='store',
                        help="Which lines? Default is ['H', 'C', 'Si','N', 'O', 'Mg']")
    parser.set_defaults(line_list=['H', 'C', 'Si','N', 'O', 'Mg'])

    parser.add_argument('--steps', metavar='steps', type=int, action='store',
                        help="How many steps? Default is 0")
    parser.set_defaults(steps=0)

    parser.add_argument('--stepsize', metavar='stepsize', type=int, action='store',
                        help="What stepsize? Default is 0")
    parser.set_defaults(stepsize=0)

    parser.add_argument('--make_spectra', dest='make_spectra', action='store_true',
                        help="Do you want to create spectra? Default is True")
    parser.set_defaults(make_spectra=True)

    parser.add_argument('--make_fits', dest='make_fits', action='store_true',
                        help="Do you want to create a fits file? Default is True")
    parser.set_defaults(make_fits=True)

    parser.add_argument('--convolve_with_instrument_psf', dest='convolve_with_instrument_psf', action='store_true',
                        help="Do you want to convolve the final output with an instrument psf? Default is True")
    parser.set_defaults(convolve_with_instrument_psf=True)

    parser.add_argument('--make_plots', dest='make_plots', action='store_true',
                        help="Do you want to create projection plots with arrows and plots of all spectra? Default is False")
    parser.set_defaults(make_plots=False)

    parser.add_argument('--physical_properties_in_header', dest='physical_properties_in_header', action='store_true',
                        help="Do you want to put the physical properties of the halo into the fits header? Default is True")
    parser.set_defaults(physical_properties_in_header=True)

    parser.add_argument('--custom_wl', dest='custom_wl', action='store_true', \
                        help="Do you want to define your own start and end wavelength? Default is False")
    parser.set_defaults(custom_wl=False)

    parser.add_argument('--custom_startwl', metavar='custom_startwl', type=float, action='store',
                        help="If custom_wl=True, define your start wavelength here. Default is 0.")
    parser.set_defaults(custom_startwl=0.)

    parser.add_argument('--custom_endwl', metavar='custom_endwl', type=float, action='store',
                        help="If custom_wl=True, define your end wavelength here. Default is 0.")
    parser.set_defaults(custom_endwl=0.)

    args = parser.parse_args()
    return args

def make_IFU(args, speedoflight=speedoflight):
    halo = args.halo
    sim = args.run
    snap = args.output
    instrument = args.instrument
    line_list = args.line_list
    steps = args.steps
    stepsize = args.stepsize
    make_spectra = args.make_spectra
    make_fits = args.make_fits
    make_plots = args.make_plots
    physical_properties_in_header = args.physical_properties_in_header
    convolve_with_instrument_psf = args.convolve_with_instrument_psf
    foggie_dir, output_dir, run_loc, code_path, trackname, haloname, spectra_dir, infofile = get_run_loc_etc(args)
    output_dir = output_dir+"mockobservations/"
    if not (os.path.exists(output_dir)): os.system('mkdir -p ' + output_dir)
    print(halo)
    fn = foggie_dir+'halo_00'+halo+'/'+sim+'/'+snap+'/'+snap
    #track_name = "/Users/raugustin/foggie/foggie/halo_tracks/008508/nref11n_selfshield_15/halo_track_600kpc_nref8"
    track_name = trackname
    os.chdir(output_dir)
    instrument = args.instrument
    line_list = args.line_list
    if instrument == 'KCWI':
        if args.custom_wl==True:
            startwl = args.custom_startwl
            endwl = args.custom_endwl
        else:
            startwl = 3500
            endwl = 5600
        ##### THE FOLLOWING NEEDS TO BE UPDATED WITH A PROPER LSF KERNEL FILE
        lsf = 80 # km/s
        speedoflight = speedoflight.value / 1000. # speedoflight is in m/s and converted to km /s
        dl_lsf = lsf/speedoflight * (startwl+endwl)/2.
        wlres = 0.28 #Angstrom per pixel BH1,2,3 0.125, BM 0.28, BL 0.625
        lsfwidth  = dl_lsf/wlres
        ##### THE ABOVE NEEDS TO BE UPDATED WITH A PROPER LSF KERNEL FILE
        psf = 0.8 #in units arcsec
        instrument_pixelsize = 0.4 #arcsec

    elif instrument == 'COS-G130M':
        if args.custom_wl==True:
            startwl = args.custom_startwl
            endwl = args.custom_endwl
        else:
            startwl = 1132
            endwl = 1433

    fullds, region = foggie_load(fn,track_name)
    zsnap = fullds.get_parameter('CosmologyCurrentRedshift')
    properwidth = fullds.refine_width # in kpc
    smallestcell = fullds.index.get_smallest_dx().in_units('code_length') # in code_length
    smallestcell = fullds.index.get_smallest_dx().in_units('kpc') # in kpc
    leftedge = region.left_edge
    rightedge = region.right_edge
    boxcenter = region.center
    center, halo_velocity = get_halo_center(fullds, boxcenter)

    if stepsize == 0:
        stepsize = smallestcell
    if steps == 0:
        steps= int(properwidth/stepsize)
    xc, yc, zc = center
    xl, yl, zl = leftedge
    xr, yr, zr = rightedge
    startwl =  startwl * (1+zsnap)
    endwl =  endwl * (1+zsnap)

    if make_plots == True:
        px = yt.ProjectionPlot(fullds, 'x', 'density', center=center, width=properwidth)
        py = yt.ProjectionPlot(fullds, 'y', 'density', center=center, width=properwidth)
        pz = yt.ProjectionPlot(fullds, 'z', 'density', center=center, width=properwidth)

    if make_spectra == True:
        i = 0 # use this if run breaks down at some point for restart
        while i < steps:
            print(i)
            """ # use this if run breaks down at some point for restart
            if i == 25:
                j = 424
            else: j = 0
            """
            j = 0
            while j < steps:
                startx = xl
                starty = yl
                startz = zl
                stepsize = stepsize.in_units('code_length')
                ray_start = [startx + i * stepsize, yl, startz + j * stepsize]
                ray_end = [startx + i * stepsize, yr, startz + j * stepsize]
                print(ray_start)
                print(ray_end)
                ray = trident.make_simple_ray(fullds,start_position=ray_start,end_position=ray_end,data_filename="ray.h5",lines=line_list,ftype='gas',redshift=zsnap)
                if make_plots == True:
                    px.annotate_ray(ray, arrow=True)
                    py.annotate_ray(ray, arrow=True)
                    pz.annotate_ray(ray, arrow=True)
                    if instrument == 'COS-G130M':
                        sg = trident.SpectrumGenerator(lambda_min=startwl, lambda_max=endwl, dlambda=0.01)
                        sg.make_spectrum(ray, lines=line_list)
                        sg.save_spectrum(str(i)+'_'+str(j)+'_'+'spec_raw.txt')
                        sg.plot_spectrum(str(i)+'_'+str(j)+'_'+'raw_'+str(snap)+'.png')
                        sg.add_milky_way_foreground()
                        sg.plot_spectrum(str(i)+'_'+str(j)+'_'+'MW_'+str(snap)+'.png')
                        sg.apply_lsf(filename="avg_COS_G130M.txt")
                        sg.plot_spectrum(str(i)+'_'+str(j)+'_'+'MW+LSF_'+str(snap)+'.png')
                        sg.add_gaussian_noise(30)
                        sg.plot_spectrum(str(i)+'_'+str(j)+'_'+'MW+LSF+noise_'+str(snap)+'.png')
                        sg.save_spectrum(str(i)+'_'+str(j)+'_'+'spec_final.txt')
                    if instrument == 'KCWI':
                        sg = trident.SpectrumGenerator(lambda_min=startwl, lambda_max=endwl, dlambda=wlres)
                        sg.make_spectrum(ray, lines=line_list)
                        sg.save_spectrum(str(i)+'_'+str(j)+'_'+'spec_raw.txt')
                        sgi = sg
                        sgi.apply_lsf(function='boxcar',width=3) # THIS NEEDS TO BE UPDATED WITH A PROPER LSF KERNEL FILE
                        sgi.save_spectrum(str(i)+'_'+str(j)+'_'+'spec_finali.txt')
                        sg.add_milky_way_foreground() # WE ALSO NEED AN ATMOSPHERE MODEL!
                        sg.apply_lsf(function='boxcar',width=3) # THIS NEEDS TO BE UPDATED WITH A PROPER LSF KERNEL FILE
                        sg.add_gaussian_noise(30)
                        sg.save_spectrum(str(i)+'_'+str(j)+'_'+'spec_final.txt')
                        trident.plot_spectrum([sg.lambda_field, sgi.lambda_field],[sg.flux_field, sgi.flux_field],lambda_limits=[startwl,endwl], stagger=0, step=[False, True],label=['Observed','Ideal'], filename=str(i)+'_'+str(j)+'_'+'ideal_and_obs'+str(zsnap)+'.png')
                else:
                    if instrument == 'COS-G130M':
                        sg = trident.SpectrumGenerator(lambda_min=startwl, lambda_max=endwl, dlambda=0.01)
                        sg.make_spectrum(ray, lines=line_list)
                        sg.save_spectrum(str(i)+'_'+str(j)+'_'+'spec_raw.txt')
                        sg.add_milky_way_foreground()
                        sg.apply_lsf(filename="avg_COS_G130M.txt")
                        sg.add_gaussian_noise(30)
                        sg.save_spectrum(str(i)+'_'+str(j)+'_'+'spec_final.txt')
                    if instrument == 'KCWI':
                        sg = trident.SpectrumGenerator(lambda_min=startwl, lambda_max=endwl, dlambda=wlres)
                        sg.make_spectrum(ray, lines=line_list)
                        sg.save_spectrum(str(i)+'_'+str(j)+'_'+'spec_raw.txt')
                        sg.add_milky_way_foreground()
                        sg.apply_lsf(function='boxcar',width=3) # THIS NEEDS TO BE UPDATED WITH A PROPER LSF KERNEL FILE
                        sg.add_gaussian_noise(30)
                        sg.save_spectrum(str(i)+'_'+str(j)+'_'+'spec_final.txt')
                j+=1
            i += 1
    if make_plots == True:
        px.save(halo+'_'+sim+'_'+snap+'_'+'projection_x_annotated.png')
        py.save(halo+'_'+sim+'_'+snap+'_'+'projection_y_annotated.png')
        pz.save(halo+'_'+sim+'_'+snap+'_'+'projection_z_annotated.png')


    if make_fits == True:

        wllen = len(np.loadtxt('0_0_spec_final.txt')[:,2])

        isteps = steps #can be changed if not square
        jsteps = steps #can be changed if not square
        data_raw = np.zeros((wllen,isteps,jsteps))
        errors_raw = np.zeros((wllen,isteps,jsteps))
        data_final = np.zeros((wllen,isteps,jsteps))
        errors_final = np.zeros((wllen,isteps,jsteps))
        grid = np.zeros((steps,steps,2))

        #### convert stepsize to kpc
        stepsize = fullds.quan(stepsize, 'code_length').in_units('kpc')
        for i in range(isteps):
            print(i)
            for j in range(jsteps):
                file_raw = np.loadtxt(str(i)+'_'+str(j)+'_'+'spec_raw.txt')
                file_final = np.loadtxt(str(i)+'_'+str(j)+'_'+'spec_final.txt')
                flux_raw = file_raw[:,2]
                flux_raw_error = file_raw[:,3]
                flux_final = file_final[:,2]
                flux_final_error = file_final[:,3]
                data_raw[:,i,j]=flux_raw
                errors_raw[:,i,j]=flux_raw_error
                data_final[:,i,j]=flux_final
                errors_final[:,i,j]=flux_final_error
        hdr0 = fits.Header()
        hdr1 = fits.Header()
        hdr2 = fits.Header()
        hdr3 = fits.Header()
        hdr4 = fits.Header()

        for hdr in [hdr0,hdr1,hdr2,hdr3,hdr4]:
            if hdr == hdr0:
                hdr['OBSERVER'] = 'Horst'
                hdr['INSTR'] = instrument
                hdr['HALO'] = halo
                hdr['SIM_OUT'] = sim
                hdr['SNAPSHOT'] = snap
                hdr['REDSHIFT'] = zsnap

                if physical_properties_in_header == True:

                    totalgasmass = sum(region['gas', 'matter_mass']).in_units('g')
                    totalgasmass_in_msol = totalgasmass.in_units('Msun')
                    totalstellarmass = sum(region['stars', 'particle_mass']).in_units('g')
                    totalstellarmass_in_msol = totalstellarmass.in_units('Msun')
                    totaldmmass = sum(region['dm', 'particle_mass']).in_units('g')
                    totaldmmass_in_msol = totaldmmass.in_units('Msun')
                    hdr['STMASS'] = (totalstellarmass, 'in units g')
                    hdr['STMASS'] = (totalstellarmass_in_msol, 'in units M_sol')
                    hdr['GASMASS'] = (totalgasmass, 'in units g')
                    hdr['GASMASS'] = (totalgasmass_in_msol, 'in units M_sol')
                    hdr['DMMASS'] = (totaldmmass, 'in units g')
                    hdr['DMMASS'] = (totaldmmass_in_msol, 'in units M_sol')

                    sfr = StarFormationRate(ds, data_source=region).Msol_yr[-1]
                    hdr['SFR'] = (sfr.item(), 'in units M_sol/yr')

            elif hdr in [hdr1,hdr2,hdr3,hdr4]:
                hdr['NAXIS'] = 3
                hdr['NAXIS1'] = isteps
                hdr['NAXIS2'] = jsteps
                hdr['NAXIS3'] = len(flux_raw)
                hdr['CRPIX1'] = int(isteps/2.)
                hdr['CRPIX2'] = int(jsteps/2.)
                hdr['CD1_1'] = stepsize.in_units('kpc').item()
                hdr['CD1_2'] = 0.
                hdr['CD2_1'] = 0.
                hdr['CD2_2'] = stepsize.in_units('kpc').item()
                hdr['CUNIT1'] = 'kpc'
                hdr['CUNIT2'] = 'kpc'
                hdr['CRVAL1'] = 0.
                hdr['CRVAL2'] = 0.
                hdr['CTYPE3'] = 'AWAV    '
                hdr['CUNIT3'] = 'Angstrom'
                hdr['CD3_3'] = wlres
                hdr['CRPIX3'] = 1.
                hdr['CRVAL3'] = startwl
        primary_hdu = fits.PrimaryHDU(header = hdr0)
        data_raw_hdu = fits.ImageHDU(data_raw, header = hdr1)
        errors_raw_hdu = fits.ImageHDU(errors_raw, header = hdr2)
        data_final_hdu = fits.ImageHDU(data_final, header = hdr3)
        errors_final_hdu = fits.ImageHDU(errors_final, header = hdr4)
        hdulist = fits.HDUList([primary_hdu, data_raw_hdu, errors_raw_hdu, data_final_hdu, errors_final_hdu])
        fitsfile = halo+'_'+sim+'_'+snap+'_'+'foggie_ifu'+'_isteps'+str(isteps)+'_jsteps'+str(jsteps)+'_stepsize'+str(stepsize.value)+'.fits'
        hdulist.writeto(fitsfile)
    if convolve_with_instrument_psf == True:

        hdulist = fits.open(fitsfile)
        hdr0 = hdulist[0].header
        hdr1 = hdulist[1].header
        hdr2 = hdulist[2].header
        hdr3 = hdulist[3].header
        hdr4 = hdulist[4].header
        observeddata = hdulist[3].data
        observederror = hdulist[4].data


        ### going from physical coordinates to sky coordinates
        cosmo = FlatLambdaCDM(H0=69.5, Om0=0.285, Tcmb0=2.725)
        kpcperarcmin = cosmo.kpc_proper_per_arcmin(zsnap)
        arcminperdegree = 60.
        hdr3['CD1_1'] = stepsize.in_units('kpc').item()
        hdr3['CD2_2'] = stepsize.in_units('kpc').item()
        hdr4['CD1_1'] = stepsize.in_units('kpc').item()
        hdr4['CD2_2'] = stepsize.in_units('kpc').item()
        hdr3['CD1_1'] = hdr3['CD1_1']*(1./kpcperarcmin)*(1./arcminperdegree)
        hdr3['CD2_2'] = hdr3['CD2_2']*(1./kpcperarcmin)*(1./arcminperdegree)
        hdr4['CD1_1'] = hdr4['CD1_1']*(1./kpcperarcmin)*(1./arcminperdegree)
        hdr4['CD2_2'] = hdr4['CD2_2']*(1./kpcperarcmin)*(1./arcminperdegree)
        hdr3['CUNIT1'] = 'deg'
        hdr3['CUNIT2'] = 'deg'
        hdr4['CUNIT1'] = 'deg'
        hdr4['CUNIT2'] = 'deg'

        cdelt = hdr3['CD1_1']
        ### convolving with psf
        telescope_resolution = psf*u.arcsecond
        sigma = telescope_resolution.to('deg')/2./(cdelt*u.deg)
        psfconv = Gaussian2DKernel(sigma)


        convolveddata=observeddata

        for i in range(len(observeddata[:,0,0])):
            slice = observeddata[i,:,:]
            convolveddata[i,:,:] = convolve_fft(slice, psfconv, boundary='wrap')


        ### rebin
        instrument_pixelsize = instrument_pixelsize/60/60 # from arcsec to deg
        oldpixelsize = cdelt #deg
        newpixelsize = instrument_pixelsize #deg
        rebin = oldpixelsize/newpixelsize
        oldpixelnumber = len(convolveddata[0,0,:])
        newpixelnumber = int(oldpixelnumber*rebin)
        rebinneddata = np.zeros((len(convolveddata[:,0,0]),newpixelnumber,newpixelnumber))
        for i in range(len(convolveddata[:,0,0])):
            slice = convolveddata[i,:,:]
            rebinneddata[i,:,:] = ccdrebin(slice, (newpixelnumber,newpixelnumber))

        ### change central pixel and pixelsize for wcs
        hdr3['CRPIX1'] = int(newpixelnumber/2.)
        hdr3['CRPIX2'] = int(newpixelnumber/2.)
        hdr4['CRPIX1'] = int(newpixelnumber/2.)
        hdr4['CRPIX2'] = int(newpixelnumber/2.)
        hdr3['CD1_1'] = instrument_pixelsize
        hdr3['CD2_2'] = instrument_pixelsize
        hdr4['CD1_1'] = instrument_pixelsize
        hdr4['CD2_2'] = instrument_pixelsize

        ### add new info to primary Header
        hdr0['INSTR'] = 'KCWI'
        primary_hdu = fits.PrimaryHDU(header = hdr0)
        data_hdu = fits.ImageHDU(rebinneddata, header = hdr3)
        error_hdu = 'TBD'
        hdulist = fits.HDUList([primary_hdu, data_hdu])
        newfitsfile = 'mock_'+instrument+'_obs'+fitsfile
        hdulist.writeto(newfitsfile)


################################################################################################################################################################################################################################
################################################################################################################################################################################################################################
################################################################################################################################################################################################################################
################################################################################################################################################################################################################################
################################################################################################################################################################################################################################

args = parse_args()
make_IFU(args)
print('you just created a new fits file \(^o^)/')
