import yt 
from astropy.table import Table 
import numpy as np 
import os 
import foggie 

def drive_shader(halo, run, axis, firstsnap, lastsnap, trackname): 

    track_name = '/astro/simulations/FOGGIE/halo_00'+halo+'/'+trackname 
    print("opening track: "+track_name) 
    track = Table.read(track_name, format='ascii') 
    track.sort('col1') 

    print firstsnap, lastsnap 
    outs = [x+firstsnap for x in range(lastsnap+1-firstsnap)]
    print outs 

    os.chdir('/astro/simulations/FOGGIE/halo_00'+halo+'/'+run) 
    prefix = '/Users/tumlinson/Dropbox/foggie/plots/halo_00'+halo+'/'+run+'/'

    for n in outs: 

        # load the snapshot 
        strset = 'DD00'+str(n) 
        if (n > 99): strset = 'DD0'+str(n) 
        snap_to_open = strset+'/'+strset  
        print('opening snapshot '+snap_to_open) 
        ds = yt.load(snap_to_open)
        zsnap = ds.get_parameter('CosmologyCurrentRedshift')

        def _msun_density(field, data):
            return data["density"]*1.0

        ds.add_field(("gas","Msun_density"),function=_msun_density, units="Msun/pc**3")
    
        # interpolate the center from the track
        centerx = np.interp(zsnap, track['col1'], track['col2']) 
        centery = np.interp(zsnap, track['col1'], track['col3']) 
        centerz = 0.5 * ( np.interp(zsnap, track['col1'], track['col4']) + np.interp(zsnap, track['col1'], track['col7'])) 
    
        center = [centerx, centery+20. / 143886., centerz] 
    
        x_left = np.interp(zsnap, track['col1'], track['col2']) 
        y_left = np.interp(zsnap, track['col1'], track['col3']) 
        z_left = np.interp(zsnap, track['col1'], track['col4']) 
        x_right = np.interp(zsnap, track['col1'], track['col5']) 
        y_right = np.interp(zsnap, track['col1'], track['col6']) 
        z_right = np.interp(zsnap, track['col1'], track['col7']) 
    
        refine_box_center = [0.5*(x_left+x_right), 0.5*(y_left+y_right), 0.5*(z_left+z_right)] 
        refine_box = ds.r[x_left:x_right, y_left:y_right, z_left:z_right] 

        foggie.shade_phase_diagram(refine_box, strset) 
        foggie.shade_mass_diagram(refine_box, strset) 
        
    
