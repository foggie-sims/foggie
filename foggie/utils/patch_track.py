import yt, numpy as np, os, matplotlib.pyplot as plt 
from foggie.utils.consistency import density_color_map, metal_color_map
from foggie.utils.foggie_load import *
from foggie.utils.get_halo_center import get_halo_center 
from astropy.table import Table, row

def patch_track(snap_start, snap_end, trackfile): 
    """ This function will iteratively derive a tracked halo 
    center for progenitor halos going back in time. It starts 
    with the existing track file and a given redshift, and uses
    get_halo_center to find and output the patched track. This 
    will help in cases where the original track breaks down or 
    bounces between progenitors.  
    
    """

    new_track = Table(np.array([0., 0., 0., 0.]), names=('redshift', 'x', 'y', 'z')) 
    print(new_track)

    track = Table.read(trackfile, format='ascii')
    track.sort('col1') 

    snaps_to_compute = np.flip(np.arange(snap_start-snap_end+1) + snap_end) 
    print('snaps_to_compute: ', snaps_to_compute)

    for thissnap in snaps_to_compute: 
        
        if (thissnap < 10000): snap_string='DD'+str(thissnap)
        if (thissnap < 1000): snap_string='DD0'+str(thissnap)
        if (thissnap < 100): snap_string='DD00'+str(thissnap)
        if (thissnap < 10): snap_string='DD000'+str(thissnap)
        print('Hello your snap_number is:', snap_start, snap_string)
        
        snapname = snap_string + '/' + snap_string
        ds = yt.load(snapname)

        this_redshift = ds.current_redshift  # <---- only needed the first time 

        try:
            center_guess
        except NameError:
            print('redshift_start ', this_redshift)
            x_start = 0.5 * np.interp(this_redshift, track['col1'], track['col2']) + 0.5 * np.interp(this_redshift, track['col1'], track['col5'])
            print('x_start ', x_start)
            y_start = 0.5 * np.interp(this_redshift, track['col1'], track['col3']) + 0.5 * np.interp(this_redshift, track['col1'], track['col6'])
            print('y_start ', y_start)
            z_start = 0.5 * np.interp(this_redshift, track['col1'], track['col4']) + 0.5 * np.interp(this_redshift, track['col1'], track['col7'])
            print('z_start ', z_start)

            center_guess = [x_start, y_start, z_start]

        new_center, vel_center = get_halo_center(ds, center_guess, radius=10.)

        new_track.add_row([ this_redshift, new_center[0], new_center[1], new_center[2] ])

        center_guess = new_center
    
    print(new_track)
    
    new_track.reverse()

    print() 
    print() 
    for row in new_track: 
        print(row['redshift'], row['x']-0.001, row['y']-0.001, row['z']-0.001, row['x']+0.001, row['y']+0.001, row['z']+0.001, 99)

    return new_track


