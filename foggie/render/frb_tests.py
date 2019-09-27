import yt
from astropy.table import Table 
import foggie.utils.get_refine_box as grb                                                                                                                                                                                        
import trident 
from foggie.utils.consistency import * 




#get the dataset, add some ion fields 
ds = yt.load(dataset_name)
trident.add_ion_fields(ds, ions=['H I', 'O VI']) 

#get the track and the box 
trackfile = '/Users/tumlinson/Dropbox/FOGGIE/foggie/halo_tracks/008508/nref11n_selfshield_15/halo_track_200kpc_nref10'
track = Table.read(trackfile, format='ascii')                                                                                                                                                                           
track.sort('col1') 
refine_box, refine_box_center, _ = grb.get_refine_box(ds, ds.current_redshift, track)  

ad = ds.all_data() 

allcells = ad.cut_region(["obj['O_p5_number_density'] > 1e-20"])
o6cells = ad.cut_region(["obj['O_p5_number_density'] > 1e-10"])


integral_all = allcells.integrate('O_p5_number_density', weight=None, axis='x')                                                                                 
integral_o6 = o6cells.integrate('O_p5_number_density', weight=None, axis='x')                                                                                 

map_all= integral_all.to_frb( (100., 'kpc') , [1000,1000], center=refine_box_center)                                                                               
map_o6= integral_o6.to_frb( (100., 'kpc') , [1000,1000], center=refine_box_center)                                                                               


plt.imshow(np.log10(np.array(fff['O_p5_number_density'])))                                                                                                        


