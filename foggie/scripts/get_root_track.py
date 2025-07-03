
import yt
from yt_astro_analysis.halo_analysis import HaloCatalog, add_quantity
import numpy as np, os, argparse 
from foggie.utils.foggie_load import *
from foggie.utils.consistency import *
from foggie.utils.halo_quantity_callbacks import *
import matplotlib.pyplot as plt
from astropy.table import Table
import unyt

def get_halo_root_track(halo_id, snap_number, box_size): 

    if (snap_number < 10000): snap_string='DD'+str(snap_number)
    if (snap_number < 1000): snap_string='DD0'+str(snap_number)
    if (snap_number < 100): snap_string='DD00'+str(snap_number)
    if (snap_number < 10): snap_string='DD000'+str(snap_number)
            
    dataset_name = snap_string+'/'+snap_string
        
    path = os.getenv('FOGGIE_REPO')+'/halo_tracks/'+halo_id+'/root_tracks/' 
    print(path) 

    root_particles = Table.read(path + 'halo_'+halo_id+'_root_index.txt', format='ascii')
    halo0 = root_particles['root_index']
        
    ds = yt.load(dataset_name) 
    ad = ds.all_data()
        
    x = ad['particle_position_x']
    y = ad['particle_position_y']
    z = ad['particle_position_z']
        
    root_indices = halo0
    now_indices = ad['particle_index']
    indices = np.where(np.isin(now_indices, root_indices))[0]
        
    center_x  = float(np.mean(x[indices].in_units('code_length'))) 
    center_y  = float(np.mean(y[indices].in_units('code_length'))) 
    center_z  = float(np.mean(z[indices].in_units('code_length'))) 
        
    center1 = [center_x, center_y, center_z]
        
    print('track ', snap_string, ds.current_redshift, center1[0]-box_size/2, center1[1]-box_size/2, 
                                             center1[2]-box_size/2, center1[0]+box_size/2, 
                                             center1[1]+box_size/2, center1[2]+box_size/2, 9)
    

parser = argparse.ArgumentParser()
parser.add_argument('--halo', type=int, required=True)
parser.add_argument('--snap_number', type=int, required=True)
parser.add_argument('--boxsize', type=float, required=True)
args = parser.parse_args()

halo_id = '00'+str(args.halo)

get_halo_root_track(halo_id, args.snap_number, args.boxsize) 

