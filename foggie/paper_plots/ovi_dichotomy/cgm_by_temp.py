import foggie.render.shade_maps as sm

#This track is always valid for 8508 regardless of "run" or resolution - right? 
trackfile = '/Users/tumlinson/Dropbox/FOGGIE/foggie/foggie/halo_tracks/008508/nref11n_selfshield_15/halo_track_200kpc_nref10' 

# nref11c_nref9f z = 0.0 RD0042 
dsname = '/Users/tumlinson/Dropbox/FOGGIE/snapshots/halo_008508/nref11c_nref9f/RD0042/RD0042'
sm.simple_plot(dsname, trackfile, 'position_x', 'position_y', 'phase', ( (-200,200), (-200,200) ), \
                'RD0042_11c_9f_x_y_phase_hot', region='cgm', screenfield='temperature', screenrange=[6, 9], pixspread=1)

sm.simple_plot(dsname, trackfile, 'position_x', 'position_y', 'phase', ( (-200,200), (-200,200) ), \
                'RD0042_11c_9f_x_y_phase_warm', region='cgm', screenfield='temperature', screenrange=[5, 6], pixspread=1)

sm.simple_plot(dsname, trackfile, 'position_x', 'position_y', 'phase', ( (-200,200), (-200,200) ), \
                'RD0042_11c_9f_x_y_phase_cool', region='cgm', screenfield='temperature', screenrange=[4, 5], pixspread=1)

sm.simple_plot(dsname, trackfile, 'position_x', 'position_y', 'phase', ( (-200,200), (-200,200) ), \
                'RD0042_11c_9f_x_y_phase_cold', region='cgm', screenfield='temperature', screenrange=[0, 4], pixspread=1)


# now color coded by metallicity  
sm.simple_plot(dsname, trackfile, 'position_x', 'position_y', 'metal', ( (-200,200), (-200,200) ), \
                'RD0042_11c_9f_x_y_metal_hot', region='cgm', screenfield='temperature', screenrange=[6, 9], pixspread=1)

sm.simple_plot(dsname, trackfile, 'position_x', 'position_y', 'metal', ( (-200,200), (-200,200) ), \
                'RD0042_11c_9f_x_y_metal_warm', region='cgm', screenfield='temperature', screenrange=[5, 6], pixspread=1)

sm.simple_plot(dsname, trackfile, 'position_x', 'position_y', 'metal', ( (-200,200), (-200,200) ), \
                'RD0042_11c_9f_x_y_metal_cool', region='cgm', screenfield='temperature', screenrange=[4, 5], pixspread=1)

sm.simple_plot(dsname, trackfile, 'position_x', 'position_y', 'metal', ( (-200,200), (-200,200) ), \
                'RD0042_11c_9f_x_y_metal_cold', region='cgm', screenfield='temperature', screenrange=[0, 4], pixspread=1)





# nref11n_nref10f z = 0.0 RD0042 
dsname = '/Users/tumlinson/Dropbox/FOGGIE/snapshots/halo_008508/nref11n_nref10f/RD0042/RD0042'
sm.simple_plot(dsname, trackfile, 'position_x', 'position_y', 'phase', ( (-200,200), (-200,200) ), \
                'RD0042_11n_10f_x_y_phase_hot', region='cgm', screenfield='temperature', screenrange=[6, 9], pixspread=1)

sm.simple_plot(dsname, trackfile, 'position_x', 'position_y', 'phase', ( (-200,200), (-200,200) ), \
                'RD0042_11n_10f_x_y_phase_warm', region='cgm', screenfield='temperature', screenrange=[5, 6], pixspread=1)

sm.simple_plot(dsname, trackfile, 'position_x', 'position_y', 'phase', ( (-200,200), (-200,200) ), \
                'RD0042_11n_10f_x_y_phase_cool', region='cgm', screenfield='temperature', screenrange=[4, 5], pixspread=1)

sm.simple_plot(dsname, trackfile, 'position_x', 'position_y', 'phase', ( (-200,200), (-200,200) ), \
                'RD0042_11n_10f_x_y_phase_cold', region='cgm', screenfield='temperature', screenrange=[0, 4], pixspread=1)



sm.simple_plot(dsname, trackfile, 'position_x', 'position_y', 'metal', ( (-200,200), (-200,200) ), \
                'RD0042_11n_10f_x_y_metal_hot', region='cgm', screenfield='temperature', screenrange=[6, 9], pixspread=1)

sm.simple_plot(dsname, trackfile, 'position_x', 'position_y', 'metal', ( (-200,200), (-200,200) ), \
                'RD0042_11n_10f_x_y_metal_warm', region='cgm', screenfield='temperature', screenrange=[5, 6], pixspread=1)

sm.simple_plot(dsname, trackfile, 'position_x', 'position_y', 'metal', ( (-200,200), (-200,200) ), \
                'RD0042_11n_10f_x_y_metal_cool', region='cgm', screenfield='temperature', screenrange=[4, 5], pixspread=1)

sm.simple_plot(dsname, trackfile, 'position_x', 'position_y', 'metal', ( (-200,200), (-200,200) ), \
                'RD0042_11n_10f_x_y_metal_cold', region='cgm', screenfield='temperature', screenrange=[0, 4], pixspread=1)
