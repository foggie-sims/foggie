from astropy.io import fits
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from astropy.cosmology import Planck15 as cosmo
from scipy import interpolate
from astropy.convolution import Gaussian2DKernel, convolve_fft
import glob
plt.ioff()
plt.close('all')
plt.rcParams['text.usetex'] = True


grism_fl = '../outputs/grism_5016_DD0970.fits'
grism_fl = '../outputs/grism_8508_DD0487_secondrun.fits'
grism_fl = '../outputs/mcrx_firstrun.fits'
#grism_fl = '../outputs/grism_secondrun.fits'
#grism_fl = '../outputs/grism_noLyabsorbtion.fits'

grism_fl  = '../outputs/mcrx_highres.fits'


halo = '5016'
#DDname = 'DD0706'
#DDname = 'DD1035'
#DDname = 'DD1347'
fls = glob.glob('../outputs/*mcrx*.fits')
fls = ['mcrx_5016_DD1118.fits']
labels = False
DDnames = [fl[-11:-5] for fl in fls] 

DDnames = [(True, 'DD1347', [2]), (False, 'DD0697', [0]), (False, 'DD1118', [0])]
#DDnames = ['DD0697', 'DD1118']



#cams = [2]#, 2]#, 1, 2]
scale_ang_per_pix  = 46
scale_arc_per_pix = 0.13
lam_min = 8000.
lam_max = 17000.

##labels = whether to annotate line names
##DDname = which snapshot
##cams = which cams to run, saved in the existing fits file structure ---- REPLACE 
for (labels, DDname, cams) in DDnames:
    grism_fl  = '../outputs/mcrx_%s_%s.fits'%(halo, DDname)

    data = fits.open(grism_fl)

    #filters = data['FILTERS']

    grism_x_sz = int(1.2*(lam_max - lam_min)/scale_ang_per_pix)
    grism_y_sz = 25

    #DD_time = np.load('/Users/rsimons/Dropbox/rcs_foggie/outputs/DD_time_new.npy', allow_pickle = True)[()]


    #if 'grism' in grism_fl: z = data[1].header['redshift'] 
    #else: z = 0
    for cam in cams:
        np.random.seed(0)
        grism = np.zeros((grism_y_sz, grism_x_sz))

        if 'grism' in grism_fl: 
            camname = 'CAMERA%i-BROADBAND'%cam
            z = data[1].header['redshift'] 
            filt_lam = (data['FILTERS'].data['lambda_eff'] * 1.e6)/(1 + z)
            dimming_factor = 1.
        else: 
            camname = 'CAMERA%i'%cam
            if DDname == 'DD1347': z = 0.686
            elif DDname == 'DD0970': z = 1.132
            elif DDname == 'DD1035': z = 1.038
            elif DDname == 'DD0706': z = 1.642

            if DDname == 'DD0711': z =  1.63
            elif DDname == 'DD1355': z =  0.679
            elif DDname == 'DD0847': z =  1.34
            elif DDname == 'DD1349': z =  0.684
            elif DDname == 'DD1360': z =  0.674
            elif DDname == 'DD0697': z =  1.664
            elif DDname == 'DD1303': z =  0.728
            elif DDname == 'DD0731': z =  1.582
            elif DDname == 'DD0712': z =  1.627
            elif DDname == 'DD0706': z =  1.642
            elif DDname == 'DD0716': z =  1.618
            elif DDname == 'DD1351': z =  0.682
            elif DDname == 'DD0630': z =  1.847
            elif DDname == 'DD1346': z =  0.687
            elif DDname == 'DD0970': z =  1.132
            elif DDname == 'DD1365': z =  0.669
            elif DDname == 'DD1035': z =  1.038
            elif DDname == 'DD0705': z =  1.644
            elif DDname == 'DD1118': z =  0.93
            elif DDname == 'DD1364': z =  0.67
            elif DDname == 'DD1361': z =  0.673
            elif DDname == 'DD1302': z =  0.729
            elif DDname == 'DD1358': z =  0.676
            elif DDname == 'DD0518': z =  2.228
            elif DDname == 'DD0707': z =  1.64
            elif DDname == 'DD1366': z =  0.669
            elif DDname == 'DD0696': z =  1.667
            elif DDname == 'DD1316': z =  0.715
            elif DDname == 'DD1347': z =  0.686
            elif DDname == 'DD1301': z =  0.73
            else:
                print ('no redshift known')
                continue


            lambda_observed = data['LAMBDA'].data['lambda'] * 1.e10 * (1.+z) #in ang
            dimming_factor = (1.+z)**4.
            lam_use = (lambda_observed > lam_min) & (lambda_observed < lam_max)

            lambda_observed = lambda_observed[lam_use]

        cube = data[camname]
        cube_xsize = cube.data.shape[1]
        cube_ysize = cube.data.shape[2]

        cube_arc_pix = cosmo.arcsec_per_kpc_proper(z).value*cube.header['CD1_1']

        center_start_ypix = int(grism_y_sz/2.)
        center_start_xpix = int(grism_x_sz/10.)
        scale_up = 10.
        x = np.arange(-cube_xsize/2., cube_xsize/2., 1) * scale_up*cube_arc_pix/scale_arc_per_pix
        y = np.arange(-cube_xsize/2., cube_xsize/2., 1) * scale_up*cube_arc_pix/scale_arc_per_pix

        xx, yy = np.meshgrid(x, y)
        xx = xx.astype('int')
        yy = yy.astype('int')

        bad_pix = np.arange(97,103)

        for i in np.arange(cube_xsize):
            if i%50. == 0: print (i, '/', cube_xsize)
            for j in np.arange(cube_ysize):
                start_x = center_start_xpix + xx[i,j]
                start_y = center_start_ypix + yy[i,j]
                #if (i in bad_pix) & (j in bad_pix): continue
                if (start_y <= 0) | (start_y >= grism_y_sz - 1): continue
                spec_flux   = cube.data[lam_use,i,j]/dimming_factor
                interp = interpolate.interp1d(lambda_observed, spec_flux, bounds_error = False, fill_value = 0.)
                x_interpolate = lambda_observed[0] + scale_ang_per_pix * np.arange(grism_x_sz - start_x)
                y_interpolate = interp(x_interpolate)

                if start_x < 0: y_interpolate = y_interpolate[abs(start_x):]
                grism[start_y, max(start_x, 0):] += y_interpolate











        fig, ax = plt.subplots(1,1, figsize = (7, 1.4))
        interp_ticks = interpolate.interp1d((np.arange(grism_x_sz) - center_start_xpix) * scale_ang_per_pix/1.e4 + lambda_observed[0]/1.e4, np.arange(grism_x_sz))
        mark_ticks = np.arange(lam_min/1.e4, lam_max/1.e4, 0.1)
        xticks = interp_ticks(mark_ticks)
        ax.set_yticks([])
        ax.set_xticks(xticks)

        ax.set_xlabel(r'observed wavelength ($\mu$m)', fontsize = 14)
        ax.set_xticklabels(['%.1f'%m for m in mark_ticks])
        if labels: ax.annotate('HST/WFC3 G102 + G141', (0.99, 0.03), ha = 'right', va = 'bottom', fontsize = 12, xycoords = 'axes fraction', color = 'white', fontweight = 'bold')
        ax.annotate('z = %.1f'%z, (0.99, 0.98), ha = 'right', va = 'top', fontsize = 14, xycoords = 'axes fraction', color = 'white', fontweight = 'bold')



        lines = [(r'[OII]', 3727, 'right'), (r'H$\beta$', 4862, 'right'), ('[OIII]', 5007, 'left'), (r'H$\alpha$ + [NII]', 6563, 'right'), (r'[SII]', 6717, 'left'), (r'[SIII]', 9530, 'center')]
        for l, (name, wave, x_adj) in enumerate(lines):
            try: obs_tick= interp_ticks(wave * (1+(z))/(1.e4))
            except: continue
            if x_adj == 'right': sgn = -1
            elif x_adj == 'left': sgn = +1
            elif x_adj == 'center': sgn = 0
            if name == '[OIII]': sgn*=5.
            if (wave * (1+(z))/(1.e4) < 0.8) | (wave * (1+(z))/(1.e4) > 1.7): continue
            ax.annotate(name, xytext = (obs_tick + sgn * 1, 5), xy = (obs_tick, 10),  ha = x_adj, va = 'bottom', fontsize = 10, xycoords = 'data', color = 'white', fontweight = 'bold', arrowprops=dict(arrowstyle="-", color = "white"))
            


            #ax.axvline(obs_tick, color = 'white', alpha = 0.3, ymin = 0.7, ymax = 1.0)



        xbar = center_start_xpix/2.
        ybar0 = center_start_ypix - 1./scale_arc_per_pix/2.
        ybar1 = center_start_ypix + 1./scale_arc_per_pix/2.
        ax.plot([xbar, xbar], [ybar0, ybar1], 'w-')
        if labels: ax.annotate('1"', (xbar-2, ybar1), ha = 'right', va = 'bottom', color = 'white', fontweight = 'bold', fontsize = 12)

        cm = plt.cm.viridis
        cm.set_bad('k')
        if DDname == 'DD1118':
            ax.annotate('Main Galaxy', (0.45, 0.5), xycoords = 'axes fraction', ha = 'right', va = 'center', color = 'white', fontweight = 'bold')
            ax.annotate('Satellite Galaxy', (0.45, 0.2),  xycoords = 'axes fraction', ha = 'right', va = 'center', color = 'white', fontweight = 'bold')


            vmx_fact = 1.8
        elif DDname =='DD0697':
            vmx_fact = 1.8
        else:
            vmx_fact = 1.2


        ax.imshow(grism, cmap = cm)
        fig.subplots_adjust(top = 1.0, left = 0.0, right = 1.0, bottom = 0.35)
        fig.savefig('../figures/2D_spectra/idealized/%s_cam%i_nonoise_noconv.png'%(grism_fl.split('/')[-1].replace('.fits', ''), cam), dpi  = 300)


        kern = Gaussian2DKernel(1.)
        grism = convolve_fft(grism, kern)
        ax.imshow(grism, cmap = cm)
        fig.subplots_adjust(top = 1.0, left = 0.0, right = 1.0, bottom = 0.35)
        fig.savefig('../figures/2D_spectra/noisefree/%s_cam%i_nonoise.png'%(grism_fl.split('/')[-1].replace('.fits', ''), cam), dpi  = 300)
        grism+=np.random.normal(0, nanmax(grism[int(grism_y_sz/2.), :])/50., grism.shape)

        spec = grism[int(grism_y_sz/2.), :]
        srt = sort(spec)
        ax.imshow(grism, cmap = cm, vmin = srt[int(0.1 * len(srt))], vmax = vmx_fact*srt[int(0.98 * len(srt))])
        fig.savefig('../figures/2D_spectra/realistic/%s_cam%i.png'%(grism_fl.split('/')[-1].replace('.fits', ''), cam), dpi  = 300)



        plt.close('all')














        fig2, ax2 = plt.subplots(1,1, figsize = (8, 5))

        if 'grism' in grism_fl: 
            camname = 'CAMERA%i-BROADBAND'%cam
            filt_lam = (data['FILTERS'].data['lambda_eff'] * 1.e6)/(1 + data[1].header['redshift'] )

        else: 
            camname = 'CAMERA%i'%cam
            filt_lam = (data[4].data['lambda'] * 1.e6)#/(1 + z)


        gd = np.arange(len(filt_lam))#~np.isnan(filt_lam)

        if False:
            xmn, xmx = 99, 100

            filt_dat =  np.mean(np.mean(data[camname].data[:,xmn:xmx, xmn:xmx], axis = 1), axis = 1)
            ax2.plot(filt_lam[gd], filt_dat[gd], 'r-', label = 'CAMERA%i-BROADBAND, [99,99]'%cam)
            xmn, xmx = 100, 101
            filt_dat =  np.mean(np.mean(data[camname].data[:,xmn:xmx, xmn:xmx], axis = 1), axis = 1)
            ax2.plot(filt_lam[gd], filt_dat[gd], linestyle = '-', color = 'darkblue', label = 'CAMERA%i-BROADBAND, [100,100]'%cam, alpha = 0.6)

        else:
            #plot single spaxel

            spxs = [(int(cube_xsize/2.),     int(cube_xsize/2.)    , 'blue'), 
                    (int(cube_xsize/2. + 1), int(cube_xsize/2. + 1), 'red'),
                    (int(cube_xsize/2. + 2), int(cube_xsize/2. + 2), 'green')]
            for (x, y, clr) in spxs:
                spec = data[camname].data[:,x, y]
                ax2.plot(filt_lam[gd], spec[gd], '-', color = clr, label = 'CAMERA%i-BROADBAND, [%i,%i]'%(cam, x, y), alpha = 0.6)





        ax2.set_xlim(0.1, 1.5)
        ax2.legend(loc = 1)
        ax2.set_xlabel(r'rest wavelength ($\mu$m)', fontsize = 15)
        ax2.set_ylabel('F$_{\lambda}$ (%s)'%data[camname].header['IMUNIT'].replace('^2', '$^2$'))

        ax22 = ax2.twiny()
        mmin = round(ceil(0.1 * (1+z)/0.2) * 0.2, 1) 
        mmax = round(ceil(1.5 * (1+z)/0.2) * 0.2, 1)

        mark_ticks = np.arange(mmin, mmax, 0.2)
        ax22.set_xticks(mark_ticks/(1 + z))
        ax22.set_xticklabels(['%.1f'%m for m in mark_ticks])

        ax22.set_xlim(ax2.get_xlim())
        ax22.set_xlabel(r'observed wavelength ($\mu$m), z = %.1f'%(z), fontsize = 15, labelpad = 12)

        fig2.tight_layout()
        fig2.savefig('../figures/1D_spectra/1D_%s_cam%i.png'%(grism_fl.split('/')[-1].replace('.fits', ''), cam), dpi  = 300)





































