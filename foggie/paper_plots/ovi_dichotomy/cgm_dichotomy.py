import foggie.render.shade_maps as sm

#This track is always valid for 8508 regardless of "run" or resolution - right? 
trackfile = '/Users/tumlinson/Dropbox/FOGGIE/foggie/foggie/halo_tracks/008508/nref11n_selfshield_15/halo_track_200kpc_nref10' 

# nref11c_nref9f z = 0.1 RD0040 
dsname = '/Users/tumlinson/Dropbox/FOGGIE/snapshots/halo_008508/nref11c_nref9f/RD0040/RD0040'
sm.simple_plot(dsname, trackfile, 'position_x', 'position_y', 'phase', ( (-200,200), (-200,200) ), \
                'RD0040_11c_9f_x_y_phase_fOVI', region='cgm', screenfield='O_p5_ion_fraction', screenrange=[0.1, 1.], pixspread=2)

sm.simple_plot(dsname, trackfile, 'position_x', 'position_y', 'metal', ( (-200,200), (-200,200) ), \
                'RD0040_11c_9f_x_y_metal_fOVI', region='cgm', screenfield='O_p5_ion_fraction', screenrange=[0.1, 1.], pixspread=2)



# nref11n_nref10f z = 0.0 RD0042 
dsname = '/Users/tumlinson/Dropbox/FOGGIE/snapshots/halo_008508/nref11n_nref10f/RD0042/RD0042'
sm.simple_plot(dsname, trackfile, 'position_x', 'position_y', 'phase', ( (-200,200), (-200,200) ), \
                'RD0042_11n_10f_x_y_phase_fOVI', region='cgm', screenfield='O_p5_ion_fraction', screenrange=[0.1, 1.], pixspread=2)

sm.simple_plot(dsname, trackfile, 'position_x', 'position_y', 'metal', ( (-200,200), (-200,200) ), \
                'RD0042_11n_10f_x_y_metal_fOVI', region='cgm', screenfield='O_p5_ion_fraction', screenrange=[0.1, 1.], pixspread=2)
