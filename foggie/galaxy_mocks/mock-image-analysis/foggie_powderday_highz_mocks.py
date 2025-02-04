import numpy as np
import scipy.ndimage
import scipy as sp
import h5py
import hyperion
from hyperion.model import ModelOutput
from hyperion.util.constants import pc
import make_color_image
import astropy
import astropy.io.fits as fits
import astropy.io.ascii as ascii
import astropy.cosmology
import astropy.nddata
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy import wcs
from astropy import units as u
from astropy import constants
from PIL import Image
from astropy.convolution import *
from astropy.stats import gaussian_fwhm_to_sigma
import glob
import os
import matplotlib
import matplotlib.pyplot as pyplot
import photutils
from photutils import aperture_photometry
from photutils import CircularAperture


fogh = 0.695
fogcos = astropy.cosmology.FlatLambdaCDM(H0=100.0*fogh,Om0=0.285,Ob0=0.0461)

def fluxes_from_rtout(rtout_file,camera_distance_pc=1.0e7,redshift_override=None,
    halo_c_v='/Users/gsnyder/Dropbox/Projects/PythonCode/foggie/foggie/halo_infos/004123/nref11c_nref9f/halo_c_v',
    this_halo='4123',this_dd='0500'):


    mo=ModelOutput(rtout_file)
    imo=mo.get_image(distance=camera_distance_pc*pc,units='MJy/sr') #surface brightness is independent of distance
    #distance needs to be big -- Mpc scales, to assert small angle approximation



    nu = np.float64(imo.nu)
    wav= np.float64(imo.wav)

    wav_m = constants.c.value/nu

    #assert wav is in microns
    for w,wm in zip(wav,wav_m):
        assert (wm*1e6==w)

    halocv = ascii.read(halo_c_v,data_start=1,names=['yyy','redshift','name','xc','yc','zc','xv','yv','zv','xxx'],delimiter='|')

    if this_halo is None:
        this_halo = rtout_file.split('_')[0]

    if this_dd is None:
        this_dd = rtout_file.split('_')[1][0:4]


    halo_i = halocv['name']=='DD'+this_dd

    try:
        halo_redshift=halocv['redshift'][halo_i].value[0]
    except:
        print('failure matching halo redshift to snap number')
        return None

    if redshift_override is not None:
        use_redshift=redshift_override
    else:
        use_redshift = halo_redshift




    sb_image = np.asarray(imo.val[0,:,:,:])


    pix_area_sr=imo.pix_area_sr
    pix_area_pc=(camera_distance_pc**2)*(pix_area_sr)

    pix_side_pc=pix_area_pc**0.5

    #method 1, use r**2 and distmod
    njy_image_at_camera = 1.0e15 * sb_image * pix_area_sr

    njy_image_at_10pc = njy_image_at_camera*(camera_distance_pc/10.0)**2

    try:
        absmags=-2.5*np.log10( (1.0e-9)*np.sum(njy_image_at_10pc,axis=(0,1)) / 3631.0 )
    except:
        return None


    mags_via_distmod = absmags + fogcos.distmod(use_redshift).value

    #method 2, use ADD and surface brightness dimming

    pix_side_arcsec = (pix_side_pc/1e3)/(fogcos.kpc_proper_per_arcmin(use_redshift).value/60)

    pix_area_arcsec_z = pix_side_arcsec**2

    sq_arcsec_per_sr = 42545170296.0





    sb_factor = (1.0 + use_redshift)**-4   ## matches relative scaling when using "distmod"

    to_njy_per_pix = (sb_factor)*(1.0e6)*(1.0e9)*pix_area_arcsec_z/sq_arcsec_per_sr

    njy_image = sb_image*to_njy_per_pix

    try:
        mags=-2.5*np.log10( (1.0e-9)*np.sum(njy_image,axis=(0,1)) / 3631.0 )
    except:
        return None

    output={}
    output['mjy_per_sr']=sb_image
    output['njy_per_pix_z']=njy_image
    output['wav_um']=wav
    output['halo_redshift']=halo_redshift
    output['redshift_override']=redshift_override

    output['pix_area_sr']=pix_area_sr
    output['pix_side_pc']=pix_side_pc
    output['pix_side_arcsec']=pix_side_arcsec
    output['mags']=mags
    output['mags_via_distmod']=mags_via_distmod

    output['hdus']=fits.HDUList()

    for i,w in enumerate(wav):
        fits_hdu_i = fits.PrimaryHDU(output['njy_per_pix_z'][:,:,i])

        wav_str='{:3.1f}'.format(w)

        fits_hdu_i.header['EXTNAME']='MockData_Pristine_'+wav_str
        fits_hdu_i.header['SIM']='foggie_'+os.path.basename(os.path.dirname(halo_c_v))
        fits_hdu_i.header['SNAP']='DD'+this_dd
        fits_hdu_i.header['HALO']=this_halo
        fits_hdu_i.header['redshift']=use_redshift
        fits_hdu_i.header['origz']=halo_redshift
        fits_hdu_i.header['wave']=(w,'micron')
        fits_hdu_i.header['pixscale']=(pix_side_arcsec,'arcsec')
        fits_hdu_i.header['pix_pc']=(pix_side_pc,'parsec, pristine image')
        fits_hdu_i.header['BUNIT']=('nanojanskies','final units')
        fits_hdu_i.header['ABMAG']=(output['mags'][i],'pristine image')

        output['hdus'].append(fits_hdu_i)

    return output




def convolve_with_fwhm_and_rebin(in_hdu, fwhm_arcsec=0.10, desired_pix_side_arcsec=None):

    #load image data and metadata
    image_in=in_hdu.data
    header_in=in_hdu.header

    extlabel=header_in['EXTNAME'].split('_')[-1]

    pixel_size_arcsec=header_in['pixscale']

    sigma_arcsec=fwhm_arcsec*gaussian_fwhm_to_sigma
    sigma_pixels=sigma_arcsec/pixel_size_arcsec

    image_out=sp.ndimage.filters.gaussian_filter(image_in,sigma_pixels,mode='nearest')


    if desired_pix_side_arcsec is not None:
        np_orig=image_out.shape[0]

        np_new=np_orig*pixel_size_arcsec/desired_pix_side_arcsec
        np_new_int=np.int64(np_new)

        orig_fov = np_orig*pixel_size_arcsec
        new_fov = np_new_int*desired_pix_side_arcsec
        diff = (orig_fov-new_fov)/2.0

        box_arcsec=(diff,diff,orig_fov-diff,orig_fov-diff)
        box=(diff/pixel_size_arcsec,diff/pixel_size_arcsec,(orig_fov-diff)/pixel_size_arcsec,(orig_fov-diff)/pixel_size_arcsec)

        #multiply by pixel scale ratio squared in order to preserve total flux
        rebinned_image=Image.fromarray(image_out).resize(size=(np_new_int, np_new_int),box=box)*(desired_pix_side_arcsec/pixel_size_arcsec)**2

        final_image=rebinned_image
        out_pix_arcsec = desired_pix_side_arcsec
    else:
        final_image=image_out
        out_pix_arcsec = pixel_size_arcsec

    hdu_out = fits.ImageHDU(final_image,header=header_in)
    hdu_out.header['FWHM']=(fwhm_arcsec,'arcsec')
    hdu_out.header['SIGMA']=(sigma_arcsec,'arcsec')
    hdu_out.header['EXTNAME']='MockData_PSF_'+extlabel
    hdu_out.header['PIXSCALE']=(out_pix_arcsec,'arcsec')
    hdu_out.header['PIXORIG']=(pixel_size_arcsec,'arcsec, pristine image')
    hdu_out.header['IN_EXT']=header_in['EXTNAME']

    return hdu_out



def add_simple_noise_extractedsn(in_hdu,radius_arcsec=0.5,extractedsn=300):


    image_in=in_hdu.data
    header_in=in_hdu.header

    #this is the approximate equation for sb magnitude input limit
    #sigma_njy=(2.0**(-0.5))*((1.0e9)*(3631.0/5.0)*10.0**(-0.4*sb_maglim))*header_in['PIXSIZE']*(3.0*header_in['FWHM'])

    #get flux in aperture
    #can we guarantee that the galaxy is in the middle?
    npix=image_in.shape[0]
    ci=np.float32(npix)/2

    radius_pixels = radius_arcsec/in_hdu.header['PIXSIZE']

    positions = [(ci, ci),]
    aperture = CircularAperture(positions, r=radius_pixels)
    phot_table = aperture_photometry(image_in, aperture)
    flux_aperture=phot_table['aperture_sum'][0]

    #get npix in aperture
    area_pixels=np.pi*radius_pixels**2



    #convert to pixel noise level
    sigma_njy=flux_aperture/(extractedsn*(area_pixels)**0.5)



    noise_image = sigma_njy*np.random.randn(npix,npix)

    image_out=image_in + noise_image

    hdu_out = fits.ImageHDU(image_out,header=header_in)
    hdu_out.header['EXTNAME']='MockImage_SN'
    hdu_out.header['EXTSN']=(extractedsn,'rough extracted S/N ratio')
    hdu_out.header['APERFLUX']=flux_aperture
    hdu_out.header['APERRAD']=radius_pixels
    hdu_out.header['RMSNOISE']=(sigma_njy,'nanojanskies')

    return hdu_out


def combo_movie(halo='4123',pix_sigma=0.2,outdir='mock_movie',
                background_folder='/Users/gsnyder/Dropbox/Projects/PythonCode/mock-surveys/mocks_from_publicdata/outputs/TNG100-1_7_6_xyz/'):

    if not os.path.lexists('mock_movie'):
        os.makedirs('mock_movie')

    #ff pixel size is 0.03 arcsec
    pixsize=0.03

    #choose filters
    # f814w, f150w, f356w ?

    #load full field images
    bffh=fits.open(os.path.join(background_folder,'TNG100-1_xyz_F814W.fits'))[0]
    gffh=fits.open(os.path.join(background_folder,'TNG100-1_xyz_F150W.fits'))[0]
    rffh=fits.open(os.path.join(background_folder,'TNG100-1_xyz_F356W.fits'))[0]

    #convolve with PSF
    psf_fwhm_b=0.065
    psf_fwhm_g=0.065
    psf_fwhm_r=0.130

    if not (os.path.lexists('mock_movie/tmp_b_ff.fits') and os.path.lexists('mock_movie/tmp_g_ff.fits') and os.path.lexists('mock_movie/tmp_r_ff.fits')):
        print('convolving full field image')
        bffh_psf=convolve_with_fwhm_and_rebin(bffh,fwhm_arcsec=psf_fwhm_b)
        gffh_psf=convolve_with_fwhm_and_rebin(gffh,fwhm_arcsec=psf_fwhm_g)
        rffh_psf=convolve_with_fwhm_and_rebin(rffh,fwhm_arcsec=psf_fwhm_r)
        print('saving full field image')
        bffh_psf.writeto('mock_movie/tmp_b_ff.fits')  #this adds a primary HDU to the thing
        gffh_psf.writeto('mock_movie/tmp_g_ff.fits')
        rffh_psf.writeto('mock_movie/tmp_r_ff.fits')
    else:
        print('reloading full field image')
        bffh_psf=fits.open('mock_movie/tmp_b_ff.fits')[1]
        gffh_psf=fits.open('mock_movie/tmp_g_ff.fits')[1]
        rffh_psf=fits.open('mock_movie/tmp_r_ff.fits')[1]

    #add noise

    npix=bffh.data.shape[0]

    sigma_njy = 0.05 #???

    b_noise_image = sigma_njy*np.random.randn(npix,npix)
    g_noise_image = sigma_njy*np.random.randn(npix,npix)
    r_noise_image = sigma_njy*np.random.randn(npix,npix)



    cutout_pos=SkyCoord(0.0*u.deg,0.0*u.deg)
    #get appropriate cutouts from full frames -- both 100x and 1000x zoom
    cutout_size_zoom=1200
    zc=cutout_size_zoom/2
    cutout_size_zoomzoom=400
    zzc=cutout_size_zoomzoom/2

    ff_wcs=wcs.WCS(bffh.header)


    print(bffh_psf.data.shape)
    print(cutout_pos)
    print(cutout_size_zoom)

    bcut_z=Cutout2D(bffh_psf.data,cutout_pos,cutout_size_zoom,wcs=ff_wcs,mode='strict')
    gcut_z=Cutout2D(gffh_psf.data,cutout_pos,cutout_size_zoom,wcs=ff_wcs,mode='strict')
    rcut_z=Cutout2D(rffh_psf.data,cutout_pos,cutout_size_zoom,wcs=ff_wcs,mode='strict')
    bcutn_z=Cutout2D(b_noise_image,cutout_pos,cutout_size_zoom,wcs=ff_wcs,mode='strict')
    gcutn_z=Cutout2D(g_noise_image,cutout_pos,cutout_size_zoom,wcs=ff_wcs,mode='strict')
    rcutn_z=Cutout2D(r_noise_image,cutout_pos,cutout_size_zoom,wcs=ff_wcs,mode='strict')

    bcut_zz=Cutout2D(bffh_psf.data,cutout_pos,cutout_size_zoomzoom,wcs=ff_wcs,mode='strict')
    gcut_zz=Cutout2D(gffh_psf.data,cutout_pos,cutout_size_zoomzoom,wcs=ff_wcs,mode='strict')
    rcut_zz=Cutout2D(rffh_psf.data,cutout_pos,cutout_size_zoomzoom,wcs=ff_wcs,mode='strict')
    bcutn_zz=Cutout2D(b_noise_image,cutout_pos,cutout_size_zoomzoom,wcs=ff_wcs,mode='strict')
    gcutn_zz=Cutout2D(g_noise_image,cutout_pos,cutout_size_zoomzoom,wcs=ff_wcs,mode='strict')
    rcutn_zz=Cutout2D(r_noise_image,cutout_pos,cutout_size_zoomzoom,wcs=ff_wcs,mode='strict')

    print(gcut_z.data.shape)

    print(gcutn_zz.data.shape)

    #save empty versions as FITS and PNG

    rt_files = np.sort(np.asarray(glob.glob('rtfiles/*.rtout')))
    for rtf in rt_files:
        this_dd=os.path.basename(rtf).split('.')[0]

        #load individual stamps
        this_output = fluxes_from_rtout(rtf,this_halo='4123',this_dd=this_dd)

        #convolve and rebin w/ same parameters
        bstamp=convolve_with_fwhm_and_rebin(this_output['hdus'][0],fwhm_arcsec=psf_fwhm_b,desired_pix_side_arcsec=pixsize)
        gstamp=convolve_with_fwhm_and_rebin(this_output['hdus'][1],fwhm_arcsec=psf_fwhm_g,desired_pix_side_arcsec=pixsize)
        rstamp=convolve_with_fwhm_and_rebin(this_output['hdus'][2],fwhm_arcsec=psf_fwhm_r,desired_pix_side_arcsec=pixsize)

        print(rtf, gstamp.data.shape)

        #add stamp at center of cutout
        b_z_frame = astropy.nddata.add_array(bcut_z.data + bcutn_z.data , bstamp.data, (zc,zc) )
        g_z_frame = astropy.nddata.add_array(gcut_z.data + gcutn_z.data , gstamp.data, (zc,zc) )
        r_z_frame = astropy.nddata.add_array(rcut_z.data + rcutn_z.data , rstamp.data, (zc,zc) )

        b_zz_frame = astropy.nddata.add_array(bcut_zz.data + bcutn_zz.data , bstamp.data, (zzc,zzc) )
        g_zz_frame = astropy.nddata.add_array(gcut_zz.data + gcutn_zz.data , gstamp.data, (zzc,zzc) )
        r_zz_frame = astropy.nddata.add_array(rcut_zz.data + rcutn_zz.data , rstamp.data, (zzc,zzc) )

        #save FITS and PNG of each sub-frame at both 100x and 1000x zoom
        frame_file_z='mock_movie/z_'+this_dd+'.png'
        frame_file_zz='mock_movie/zz_'+this_dd+'.png'

        #probably need some rotation somwhere?

        f=pyplot.figure(figsize=(4.0,4.0),dpi=200)
        pyplot.subplots_adjust(wspace=0.0,hspace=0.0,top=1.0,right=1.0,left=0.00,bottom=0.00)
        ax=f.add_subplot(1,1,1)
        ax.set_xticks([]) ; ax.set_yticks([])
        rgbthing_z = make_color_image.make_interactive_nasa(2.5*b_z_frame,g_z_frame,r_z_frame,10.0,8.0)
        ax.imshow(rgbthing_z,interpolation='nearest',aspect='auto',origin='lower')
        f.savefig(frame_file_z,dpi=200)
        pyplot.close(f)

        f=pyplot.figure(figsize=(4.0,4.0),dpi=200)
        pyplot.subplots_adjust(wspace=0.0,hspace=0.0,top=1.0,right=1.0,left=0.00,bottom=0.00)
        ax=f.add_subplot(1,1,1)
        ax.set_xticks([]) ; ax.set_yticks([])
        rgbthing_zz = make_color_image.make_interactive_nasa(2.5*b_zz_frame,g_zz_frame,r_zz_frame,10.0,8.0)
        ax.imshow(rgbthing_zz,interpolation='nearest',aspect='auto',origin='lower')
        f.savefig(frame_file_zz,dpi=200)
        pyplot.close(f)


    #save a few full frames as FITS and PNG?


    return
