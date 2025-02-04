import foggie.render.shade_maps as sm

#This track is always valid for 8508 regardless of "run" or resolution - right? 
trackfile = '/Users/tumlinson/Dropbox/FOGGIE/foggie/foggie/halo_tracks/008508/nref11n_selfshield_15/halo_track_200kpc_nref10' 

# nref11c_nref9f z = 0.1 RD0040 
dsname = '/Users/tumlinson/Dropbox/FOGGIE/snapshots/halo_008508/nref11c_nref9f/RD0040/RD0040'
sm.simple_plot(dsname, trackfile, 'density', 'temperature', 'phase', ( (-32,-22), (1,8) ), \
                'RD0040_11c_9f_density_temperature_rvir_phase', region='rvir')

sm.simple_plot(dsname, trackfile, 'density', 'temperature', 'metal', ( (-32,-22), (1,8) ), \
                'RD0040_11c_9f_density_temperature_rvir_metal', region='rvir')

sm.simple_plot(dsname, trackfile, 'density', 'temperature', 'phase', ( (-32,-22), (1,8) ), \
                'RD0040_11c_9f_density_temperature_ism_phase', region='ism')

sm.simple_plot(dsname, trackfile, 'density', 'temperature', 'metal', ( (-32,-22), (1,8) ), \
                'RD0040_11c_9f_density_temperature_ism_metal', region='ism')

sm.simple_plot(dsname, trackfile, 'density', 'temperature', 'phase', ( (-32,-22), (1,8) ), \
                'RD0040_11c_9f_density_temperature_cgm_phase', region='cgm')

sm.simple_plot(dsname, trackfile, 'density', 'temperature', 'metal', ( (-32,-22), (1,8) ), \
                'RD0040_11c_9f_density_temperature_cgm_metal', region='cgm')


# nref11n_nref10f z = 0.0 RD0042 
dsname = '/Users/tumlinson/Dropbox/FOGGIE/snapshots/halo_008508/nref11n_nref10f/RD0042/RD0042'
sm.simple_plot(dsname, trackfile, 'density', 'temperature', 'phase', ( (-32,-22), (1,8) ), \
                'RD0042_11n_10f_density_temperature_rvir_phase', region='rvir')

sm.simple_plot(dsname, trackfile, 'density', 'temperature', 'metal', ( (-32,-22), (1,8) ), \
                'RD0042_11n_10f_density_temperature_rvir_metal', region='rvir')

sm.simple_plot(dsname, trackfile, 'density', 'temperature', 'phase', ( (-32,-22), (1,8) ), \
                'RD0042_11n_10f_density_temperature_ism_phase', region='ism')

sm.simple_plot(dsname, trackfile, 'density', 'temperature', 'metal', ( (-32,-22), (1,8) ), \
                'RD0042_11n_10f_density_temperature_ism_metal', region='ism')

sm.simple_plot(dsname, trackfile, 'density', 'temperature', 'phase', ( (-32,-22), (1,8) ), \
                'RD0042_11n_10f_density_temperature_cgm_phase', region='cgm')

sm.simple_plot(dsname, trackfile, 'density', 'temperature', 'metal', ( (-32,-22), (1,8) ), \
                'RD0042_11n_10f_density_temperature_cgm_metal', region='cgm')