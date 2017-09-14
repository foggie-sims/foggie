import yt
from astropy.table import Table
import foggie 
import numpy as np 

def get_moving_box(ds, tt, dx, dy, dz, nref): 

    #tt = input table of redshfit centers 

    tt['xleft'] = tt['z'] * 0.0
    tt['yleft'] = tt['z'] * 0.0
    tt['zleft'] = tt['z'] * 0.0

    tt['xright'] = tt['z'] * 0.0
    tt['yright'] = tt['z'] * 0.0
    tt['zright'] = tt['z'] * 0.0
  
    tt['nref'] = tt['x'] * 0 + nref 

    for line in tt: 
        line['xleft'] = line['x'] + dx[0] / (ds.get_parameter('CosmologyComovingBoxSize') * 1000.) 
        line['yleft'] = line['y'] + dy[0] / (ds.get_parameter('CosmologyComovingBoxSize') * 1000.) 
        line['zleft'] = line['z'] + dz[0] / (ds.get_parameter('CosmologyComovingBoxSize') * 1000.) 

        line['xright'] = line['x'] + dx[1] / (ds.get_parameter('CosmologyComovingBoxSize') * 1000.) 
        line['yright'] = line['y'] + dy[1] / (ds.get_parameter('CosmologyComovingBoxSize') * 1000.) 
        line['zright'] = line['z'] + dz[1] / (ds.get_parameter('CosmologyComovingBoxSize') * 1000.) 

    tt.reverse() 
 
    return tt 

    
