import foggie.render.shade_maps as sm

#This track is always valid for 8508 regardless of "run" or resolution - right? 
trackfile = '/Users/tumlinson/Dropbox/FOGGIE/foggie/foggie/halo_tracks/008508/nref11n_selfshield_15/halo_track_200kpc_nref10' 

# nref11c_nref9f z = 0.0 RD0042 
dsname = '/Users/tumlinson/Dropbox/FOGGIE/snapshots/halo_008508/nref11c_nref9f/RD0042/RD0042'

sm.simple_plot(dsname, trackfile, 'radius_corrected', 'temperature', 'phase', ( (0,200), (1,8) ), \
                'RD0042_11c_9f_radius_temperature_cgm_phase', region='cgm')

sm.simple_plot(dsname, trackfile, 'radius_corrected', 'cooling_time', 'phase', ( (0,200), (4,13) ), \
                'RD0042_11c_9f_radius_cooling_time_cgm_phase', region='cgm')

sm.simple_plot(dsname, trackfile, 'temperature', 'cooling_time', 'phase', ( (1,8), (4,13) ), \
                'RD0042_11c_9f_temperature_cooling_time_cgm_phase', region='cgm')



sm.simple_plot(dsname, trackfile, 'radius_corrected', 'cooling_time', 'phase', ( (0,200), (4,13) ), \
                'RD0042_11c_9f_radius_cooling_time_cgm_phase_fOVI', region='cgm', screenfield='O_p5_ion_fraction', screenrange=[0.1, 1.])

sm.simple_plot(dsname, trackfile, 'radius_corrected', 'temperature', 'phase', ( (0,200), (1,8) ), \
                'RD0042_11c_9f_radius_temperature_cgm_phase_fOVI', region='cgm', screenfield='O_p5_ion_fraction', screenrange=[0.1, 1.])

sm.simple_plot(dsname, trackfile, 'temperature', 'cooling_time', 'phase', ( (1,8), (4,13) ), \
                'RD0042_11c_9f_temperature_cooling_time_cgm_phase_fOVI', region='cgm', screenfield='O_p5_ion_fraction', screenrange=[0.1, 1.])
