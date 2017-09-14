import yt 
import numpy as np 
import os 

def open_script(halo, run, axis, firstsnap, lastsnap): 

    outs = [x+firstsnap for x in range(lastsnap+1-firstsnap)]

    os.chdir('/astro/simulations/FOGGIE/halo_00'+halo+'/'+run) 

    for n in outs: 

        strset = 'DD00'+str(n) 
        if (n > 99): strset = 'DD0'+str(n) 
        snap_to_open = strset+'/'+strset  
        print('opening snapshot '+snap_to_open) 
        help(yt)
        ds = yt.load(snap_to_open)
        zsnap = ds.get_parameter('CosmologyCurrentRedshift')

        print(snap_to_open, zsnap) 

