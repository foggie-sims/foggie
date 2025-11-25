import unyt as u
import numpy as np

from foggie.galaxy_mocks.mock_hi_imager.line_properties import load_line_properties

from foggie.galaxy_mocks.mock_hi_imager.HICubeHeader import * #Constants


class radio_telescope(object):
    def __init__(self, args):
        self.name = args.survey
        self.get_instrument_properties(args)

    def get_instrument_properties(self, args):
        '''
        Function to set attributes of the given survey either based on pre-saved list or the user inputs
        For the list of pre-saved instrument attributes intrument_dict:
        name: must be in lowercase
        obs_freq_range: in Hz
        obs_spec_res: in km/s
        obs_spatial_res: in arcseconds
        '''

        species_mass,gamma_ul, A_ul, Elevels, Glevels, n_u_fraction, n_l_fraction = load_line_properties("HI_21cm")

        #https://science.nrao.edu/facilities/vla/docs/manuals/oss/performance/resolution
        #in arcseconds
        VLA_max_angular_scales = {'4':  {'A': 800, 'B' : 2200, 'C': 20000, 'D': 20000 },
                                  'P':  {'A': 155, 'B' : 515, 'C': 4150, 'D': 4150 },
                                  'L':  {'A': 36,  'B' : 120, 'C': 970, 'D': 970 },
                                  'S':  {'A': 18,  'B' : 58, 'C': 490, 'D': 490 },
                                  'C':  {'A': 8.9, 'B' : 29, 'C': 240, 'D': 240 },
                                  'X':  {'A': 5.3, 'B' : 17, 'C': 145, 'D': 145 },
                                  'Ku': {'A': 3.6, 'B' : 12, 'C': 97, 'D': 97 },
                                  'K':  {'A': 2.4, 'B' : 7.9, 'C': 66, 'D': 66 },
                                  'Ka': {'A': 1.6, 'B' : 5.3, 'C': 44, 'D': 44 },
                                  'Q':  {'A': 1.2, 'B' : 3.9, 'C': 32, 'D': 32 },
                                  }
        
        HI_21cm_freq = (Elevels[1]-Elevels[0])/h #in Hz
        THINGS_BW_freq = 1.56e6 / u.s
        #MHONGOOSE_BW_freq = 1.56e6 / u.s #Leave the same for now for simplicity...
        HI_21cm_freq -= args.z*HI_21cm_freq # account for redshift of the 21 cm line

        #THINGS uses B array to reach it's max resolution, but uses D and C array data to recover extended emissions
        THINGS_min_spatial_freq = 1. / (VLA_max_angular_scales['L']['D'] * arcsec_to_rad) #max angular scale obtained while in D configuration
        MHONGOOSE_min_spatial_freq = THINGS_min_spatial_freq * 29./35.
        FAST_min_spatial_freq = 0.
        VLA_EffectiveArea = 27. * np.pi * (25.*u.m)**2
        MEERKAT_EffectiveArea = 64. * np.pi * (13.5*u.m)**2
        FAST_Area = 500.


        if self.name =="THINGS":
            SpecRes_kms = 5.2#2.6
            nChannels = 64#128
        elif self.name=="MHONGOOSE_HR" or self.name=="MHONGOOSE_LR":
            SpecRes_kms = 1.4
            nChannels = int(np.round(128 * 2.6/1.4)) # limited bandwidth for now as MHONGOOSE has a very wide band, pad as necesarry for the galaxy
        elif self.name=="FAST":
            SpecRes_kms = 1.67 # smoothed to 3.4?
            nChannels = int(np.round(128 * 2.6/1.67)) # limited bandwidth for now, pad as needed
 
        if args.max_projected_velocity is not None:
            # Pads the bandwidth to account for the maximum projected velocity of the galaxy.
            # This is to account for the high central rotational velocity in the FOGGIE galaxies.
            dopplerBeta = args.max_projected_velocity  / c
            nu_ul = (Elevels[1]-Elevels[0])/h #in Hz
            doppler_freq_shift = nu_ul * (np.sqrt( np.divide(-dopplerBeta+1 , dopplerBeta+1) ) - 1) # in Hz
            if 2 * np.abs(doppler_freq_shift) > THINGS_BW_freq:
                args.freq_range_mult = 2 * np.abs(doppler_freq_shift) / THINGS_BW_freq
                nChannels = int(np.round(nChannels*args.freq_range_mult))
                print("Multiplying bandwidth by a factor of", args.freq_range_mult,"nChannels set to:",nChannels)



        self.instrument_dict = {'THINGS': {'obs_freq_range': (HI_21cm_freq - THINGS_BW_freq*args.freq_range_mult/2., HI_21cm_freq + THINGS_BW_freq*args.freq_range_mult/2.), 'obs_spec_res': SpecRes_kms, 'obs_spatial_res': 6.,  'obs_channels':nChannels, 'min_spatial_freq':THINGS_min_spatial_freq, 'min_baseline':0.035,'primary_beam_FWHM_deg':0.53,'spec_res_kms':SpecRes_kms,'integrated_channels_for_noise':3,} \
                            ,'MHONGOOSE_HR': {'obs_freq_range': (HI_21cm_freq - THINGS_BW_freq*args.freq_range_mult/2., HI_21cm_freq + THINGS_BW_freq*args.freq_range_mult/2.), 'obs_spec_res': SpecRes_kms, 'obs_spatial_res': 22., 'obs_channels':nChannels, 'min_spatial_freq':MHONGOOSE_min_spatial_freq, 'min_baseline':0.029, 'primary_beam_FWHM_deg':1,'spec_res_kms':SpecRes_kms,'integrated_channels_for_noise':3,} \
                            ,'MHONGOOSE_LR': {'obs_freq_range': (HI_21cm_freq - THINGS_BW_freq*args.freq_range_mult/2., HI_21cm_freq + THINGS_BW_freq*args.freq_range_mult/2.), 'obs_spec_res': SpecRes_kms, 'obs_spatial_res': 65., 'obs_channels':nChannels, 'min_spatial_freq':MHONGOOSE_min_spatial_freq, 'min_baseline':0.029,'primary_beam_FWHM_deg':1,'spec_res_kms':SpecRes_kms,'integrated_channels_for_noise':3,} \
                            ,'FAST': {'obs_freq_range': (HI_21cm_freq - THINGS_BW_freq*args.freq_range_mult/2., HI_21cm_freq + THINGS_BW_freq*args.freq_range_mult/2.), 'obs_spec_res': SpecRes_kms, 'obs_spatial_res': 2.9*60., 'obs_channels':nChannels, 'min_spatial_freq':FAST_min_spatial_freq, 'min_baseline':0, 'primary_beam_FWHM_deg':None,'spec_res_kms':SpecRes_kms,'integrated_channels_for_noise':3,} \
                           }


        if self.name in self.instrument_dict: # assigning known parameters (by overriding user input parameters, if any) for a known instrument
            print('Known instrument: ' + self.name + '; using pre-assigned attributes (over-riding user inputs, if any)', args)
            for key in self.instrument_dict[self.name]:
                setattr(self, key, self.instrument_dict[self.name][key])
            self.obs_freq_range = self.obs_freq_range
        else: # assigning user input parameters for this unknown instrument
            self.name = 'dummy'
            self.obs_freq_range = args.obs_freq_range
            self.obs_spec_res = args.obs_spec_res # in km/s
            self.obs_spatial_res = args.obs_spatial_res # in arcseconds

        if args.min_baseline is not None:
            #Override the minimum baseline for setting the minimum spatial frequency
            print("Overriding minimum baseline to:",args.min_baseline)
            self.min_baseline = args.min_baseline