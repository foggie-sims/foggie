import argparse


def parse_args():
    '''Parse command line arguments. Returns args object.
    NOTE: Need to move command-line argument parsing to separate file.'''

    parser = argparse.ArgumentParser()
    # Optional arguments:
    parser.add_argument('--ifu_dir', metavar='ifu_dir', type=str, action='store',required=True, \
                        help='What is the ifu directory? (Required)')
    parser.set_defaults(ifu_dir=None)
    
    
    parser.add_argument('--instrument', metavar='instrument', type=str, action='store', \
                        help='Which instrument you are modeling. Default is WFC3.')
    parser.set_defaults(instrument='WFC3')
    
    parser.add_argument('--filter', metavar='filter', type=str, action='store', \
                        help='Which filter you are modeling. Default is G102.')
    parser.set_defaults(filter='G102')
    
    parser.add_argument('--dfilter', metavar='dfilter', type=str, action='store', \
                        help='Which direct filter you are modeling. Default is F105W.')
    parser.set_defaults(dfilter='F105W')
    
    
    parser.add_argument('--nBeams', metavar='nBeams', type=int, action='store', \
                        help='How many beams do you want to model [1-4]. 1 only models 1st order, 2 adds 0th order, 3 adds 2nd order, 4 adds -1st order.')
    parser.set_defaults(nBeams=3)  
    
    parser.add_argument('--beams', metavar='beams', nargs='+', type=int, action='store', \
                        help='Which beams do you want? 0=1st Order, 1=0th Order, 2=2nd Order, 3=-1st Order. Default is 0 3.')
    parser.set_defaults(beams=[0,3])  
    

    parser.add_argument('--exposure', metavar='exposure', type=float, action='store', \
                        help='What exposure time do you want to model in seconds? Default is 300s')
    parser.set_defaults(exposure=300)
    
    parser.add_argument('--effective_exposure', metavar='effective_exposure', type=float, action='store', \
                        help='What effective exposure time do you want to model in seconds? Default is exposure.')
    parser.set_defaults(effective_exposure=None)
    
    parser.add_argument('--snr', metavar='snr', type=float, action='store', \
                        help='What snr to want to target? Default is 10.')
    parser.set_defaults(snr=10.)
    
    parser.add_argument('--pEarthshine', metavar='pEarthshine', type=float, action='store', \
                        help='Percentage strength of the background earthshine for defining sky background. (Default is 0.5)')
    parser.set_defaults(pEarthshine=0.5)    
    
    parser.add_argument('--pZodiacal', metavar='pZodiacal', type=float, action='store', \
                        help='Percentage strength of the zodiacal background for defining sky background. (Default is 0.5)')
    parser.set_defaults(pZodiacal=0.5)    
    
    
    parser.add_argument('--lambda_min', metavar='lambda_min', type=float, action='store', \
                        help='Minimum wavelength for ouput grism spectra in Angstroms. Default to provided instrument+filter.')
    parser.set_defaults(lambda_min=-1)

    parser.add_argument('--lambda_max', metavar='lambda_max', type=float, action='store', \
                        help='Maximum wavelength for ouput grism spectra in Angstroms. Default to provided instrument+filter.')
    parser.set_defaults(lambda_max=-1)
    
    parser.add_argument('--dispersion', metavar='dispersion', type=float, action='store', \
                        help='How many Angstroms per pixel in the grism data. Default to provided instrument+filter.')
    parser.set_defaults(dispersion=-1)
    
#    parser.add_argument('--scale_arc_per_pix', metavar='scale_arc_per_pix', type=str, action='store', \
#                        help='How many arcseconds per pixel in the grism data. Default is 0.13')
#    parser.set_defaults(scale_arc_per_pix='.13')

    parser.add_argument('--xoffset', metavar='xoffset', type=float, action='store', \
                        help='What is the zeroeth order x-offset in pixels from the direct image? Default to provided instrument+filter.')
    parser.set_defaults(xoffset=None)
    
    parser.add_argument('--yoffset', metavar='yoffset', type=float, action='store', \
                        help='What is the first order y-offset in pixels from the direct image? Default to provided instrument+filter.')
    parser.set_defaults(yoffset=None)

    
    parser.add_argument('--output', metavar='output', type=str, action='store', \
                        help='Output directory for plots and data. Default is grism_test')
    parser.set_defaults(output='outputs//grism_test')
    

  


    parser.add_argument('--ifu_offset_x', metavar='ifu_offset_x', type=int, action='store', \
                        help='Pixel offset to add to ifu image when placing in GRISM image space. Default is 0.')
    parser.set_defaults(ifu_offset_x=0)   


    parser.add_argument('--ifu_offset_y', metavar='ifu_offset_y', type=int, action='store', \
                        help='Pixel offset to add to ifu image when placing in GRISM image space. Default is 0.')
    parser.set_defaults(ifu_offset_y=0)   
    
    parser.add_argument('--ifu_wavelength_offset', metavar='ifu_wavelength_offset', type=int, action='store', \
                        help='Wavelength offset for ifu data in Angstrom. For testing purposes. Default is 0.')
    parser.set_defaults(ifu_wavelength_offset=0)   
    
    parser.add_argument('--plot_all', metavar='plot_all', type=bool, action='store', \
                        help='Plot additional diagnostic plots. Default is False.')
    parser.set_defaults(plot_all=False)   

    parser.add_argument('--ifu_signal_boost_factor', metavar='ifu_signal_boost_factor', type=float, action='store', \
                        help='Factor to boost the flux of the simualated ifu. Default is 1.0')
    parser.set_defaults(ifu_signal_boost_factor=1.0)   

    args = parser.parse_args()
    
    
    return args

