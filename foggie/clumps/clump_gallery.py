
import yt, os, numpy as np, astropy 
from foggie.utils.foggie_load import *
import seaborn as sns
from astropy import units as u

density_color_map = sns.blend_palette(
    ("black", 'cyan', "#4575b4", "#4daf4a", "#ffe34d", "darkorange"), as_cmap=True)

dataset_name = os.getenv('DROPBOX_DIR') + '/FOGGIE/snapshots/halo_008508/nref11c_nref9f/RD0027/RD0027'
trackname = os.getenv('DROPBOX_DIR') + '/FOGGIE/foggie/foggie/halo_tracks/008508/nref11n_selfshield_15/halo_track_200kpc_nref10'
track_dir = os.getenv('DROPBOX_DIR') + '/FOGGIE/foggie/foggie/halo_tracks/008508/nref11n_selfshield_15/'
ds, refine_box = foggie_load(dataset_name, trackname, halo_c_v_name=track_dir + 'halo_c_v', disk_relative=True, particle_type_for_angmom='young_stars')

files = ['shell_level1_20cells_20.0kpc/halo_008508_nref11c_nref9f_RD0027_RD0027_clumps_tree.h5', 
         'shell_level2_20cells_20.0kpc/halo_008508_nref11c_nref9f_RD0027_RD0027_clumps_tree.h5',
         'shell_level3_20cells_20.0kpc/halo_008508_nref11c_nref9f_RD0027_RD0027_clumps_tree.h5',
         'shell_level4_20cells_20.0kpc/halo_008508_nref11c_nref9f_RD0027_RD0027_clumps_tree.h5']

all_leaves = []

for file in files: 
    tree = yt.load(file)
    print(file, len(tree.leaves))
    all_leaves = all_leaves + tree.leaves

print("There are this many leaves: ", len(all_leaves)) 

for l in all_leaves: 
    for axis in ['x', 'y', 'z']: 
        p = yt.ProjectionPlot(ds, axis, 'density', weight_field='density', center=ds.halo_center_code, width=(80, 'kpc'))  
        p.set_cmap('density', density_color_map)
        p.annotate_clumps([l,l], color='white')
        p.annotate_title('Clump ID = ' + str(l.__dict__['clump_id']) + ", Parent = " + str(l.__dict__['parent_id']) ) 
        p.save(str(l.__dict__['clump_id']) + '_'+str(l.__dict__['parent_id'])) 

