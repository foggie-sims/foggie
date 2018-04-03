import yt 
from astropy.table import Table
import foggie 
import numpy as np 

def get_center_track(first_center, latesnap, earlysnap, interval):

    snaplist = np.flipud(np.arange(earlysnap,latesnap+1)) 
    print(snaplist) 

    t = Table([[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0], ['       ', '       ']],
        names=('redshift', 'x0', 'y0', 'z0', 'name'))

    center_guess = first_center 

    for isnap in snaplist: 
        if (isnap <= 999): name = 'RD0'+str(isnap)
        if (isnap <= 99): name = 'RD00'+str(isnap)
        if (isnap <= 9): name = 'RD000'+str(isnap)

        print 
        print 
        print(name) 
        ds = yt.load(name+'/'+name) 
        comoving_box_size = ds.get_parameter('CosmologyComovingBoxSize') 
        print('Comoving Box Size:', comoving_box_size) 

        new_center, vel_center = foggie.get_halo_center(ds, center_guess, radius=500., vel_radius=500.)  
        print(new_center) 

        t.add_row( [ds.get_parameter('CosmologyCurrentRedshift'),  
            new_center[0], new_center[1],new_center[2], name])

        center_guess = new_center 

        p = yt.ProjectionPlot(ds, 'x', 'density', center=new_center, width=(500., 'kpc')) 
        p.annotate_timestamp(corner='upper_left', redshift=True, draw_inset_box=True)
        p.save() 
        p = yt.ProjectionPlot(ds, 'y', 'density', center=new_center, width=(500., 'kpc')) 
        p.annotate_timestamp(corner='upper_left', redshift=True, draw_inset_box=True)
        p.save() 
        p = yt.ProjectionPlot(ds, 'z', 'density', center=new_center, width=(500., 'kpc')) 
        p.annotate_timestamp(corner='upper_left', redshift=True, draw_inset_box=True)
        p.save() 
        print(t) 
        print 
        print 

    t = t[2:] 
    print(t) 

    # now interpolate the track to the interval given as a parameter 
    n_points = int((np.max(t['redshift']) - np.min(t['redshift']))  / interval) 
    newredshift = np.min(t['redshift']) + np.arange(n_points+2) * interval 
    newx = np.interp(newredshift, t['redshift'], t['x0']) 
    newy = np.interp(newredshift, t['redshift'], t['y0']) 
    newz = np.interp(newredshift, t['redshift'], t['z0']) 

    tt = Table([ newredshift, newx, newy, newz], names=('redshift','x','y','z')) 

    t.write('track.fits',overwrite=True) 
    tt.write('track_interpolate.fits',overwrite=True) 

    return t, tt 
