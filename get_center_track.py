import yt
from astropy.table import Table
from astropy.io import ascii

from get_halo_center import get_halo_center
import numpy as np

def get_center_track(first_center, latesnap, earlysnap, interval):

    snaplist = np.flipud(np.arange(earlysnap,latesnap+1))
    print(snaplist)

    t = Table([[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0], ['       ', '       ']],
        names=('redshift', 'x0', 'y0', 'z0', 'name'))

    center_guess = first_center

    for isnap in snaplist:
        if (isnap <= 999): name = 'DD0'+str(isnap)
        if (isnap <= 99): name = 'DD00'+str(isnap)
        if (isnap <= 9): name = 'DD000'+str(isnap)

        print
        print
        print(name)
        ds = yt.load(name+'/'+name)
        comoving_box_size = ds.get_parameter('CosmologyComovingBoxSize')
        print('Comoving Box Size:', comoving_box_size)

        # decreased these from 500 to 100 because outputs so short spaced now
        new_center, vel_center = get_halo_center(ds, center_guess, radius=100., vel_radius=100.)
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

    ascii.write(t, 'track.dat', format='fixed_width_two_line', overwrite=True)
    ascii.write(tt, 'track_interpolate.dat', format='fixed_width_two_line', overwrite=True)

    return t, tt


if __name__ == "__main__":
    ## first_center is the center at the last output; working backwards
    ## DD0493 for nref11n_selfshield_z15
    first_center = [0.49400806427001953, 0.48881053924560547, 0.50222492218017578]
    get_center_track(first_center, 'DD0493', 'DD0040', 0.001)
