import foggie.render.shade_maps as sm

#This track is always valid for 8508 regardless of "run" or resolution - right? 
trackfile = '/Users/tumlinson/Dropbox/FOGGIE/foggie/foggie/halo_tracks/008508/nref11n_selfshield_15/halo_track_200kpc_nref10' 

# nref11c_nref9f z = 0.1 RD0040 
dsname = '/Users/tumlinson/Dropbox/FOGGIE/snapshots/halo_008508/nref11c_nref9f/RD0040/RD0040'

sm.simple_plot(dsname, trackfile, 'position_y', 'position_z', 'phase', ( (-200,200), (-200,200) ), \
                'cgm_y_z', region='cgm')

sm.simple_plot(dsname, trackfile, 'position_x', 'position_y', 'phase', ( (-200,200), (-200,200) ), \
                'cgm_x_y', region='cgm')

sm.simple_plot(dsname, trackfile, 'position_x', 'position_y', 'phase', ( (-200,200), (-200,200) ), \
                'ism_x_y', region='ism')

sm.simple_plot(dsname, trackfile, 'position_y', 'position_z', 'phase', ( (-200,200), (-200,200) ), \
                'ism_y_z', region='ism')

