import yt
from yt_astro_analysis.halo_analysis import HaloCatalog, add_quantity
import numpy as np, os 
from foggie.utils.foggie_load import *
from foggie.utils.consistency import * 
from foggie.utils.halo_quantity_callbacks import * 
from astropy.table import Table 
import argparse 

def get_root_halo(halo_id, snapname): 

    simulation_dir = './'
    dataset_name = simulation_dir+'/'+snapname+'/'+snapname

    trackname = '/nobackupnfs1/jtumlins/foggie/foggie/halo_tracks/'+halo_id+'/nref11n_selfshield_15/halo_track_200kpc_nref10'
    print(trackname)
    ds, region = foggie_load(dataset_name, trackname) 
    box = ds.r[ds.halo_center_code[0]-0.02:ds.halo_center_code[0]+0.02, 
              ds.halo_center_code[1]-0.02:ds.halo_center_code[1]+0.02, 
              ds.halo_center_code[2]-0.02:ds.halo_center_code[2]+0.02] # halo finder only accepts box cut_regions? 
    ad = ds.all_data() 
    
    p = yt.ProjectionPlot(ds, 'x', 'density', weight_field='density', data_source=box, 
                          center=ds.halo_center_code, width=(400, 'kpc'))
    p.set_cmap('density', density_color_map)
    p.annotate_title(ds._input_filename[-6:])
    p.annotate_timestamp(redshift=True)
    p.set_zlim('density', 1e-28, 1e-21)
    p.save()
    
    hc = HaloCatalog(data_ds=ds, finder_method='hop', finder_kwargs={"subvolume": box, "threshold":160., "ptype":"nbody"}, 
                     output_dir=simulation_dir+'/halo_catalogs') 
    hc.add_filter("quantity_value", "virial_radius", ">", 1, "kpc")
    
    hc.create()
    
    hds = yt.load(simulation_dir+'/halo_catalogs/'+snapname+'/'+snapname+'.0.h5')
    hc = HaloCatalog(data_ds=ds, halos_ds=hds, output_dir=simulation_dir+'/halo_catalogs')
    hc.add_callback("sphere")
    
    hc.add_filter("quantity_value", "virial_radius", ">", 1., "kpc")
    
    add_quantity("average_temperature", halo_average_temperature)
    add_quantity("average_metallicity", halo_average_metallicity)
    add_quantity("total_gas_mass", halo_total_gas_mass)
    add_quantity("total_star_mass", halo_total_star_mass)
    add_quantity("average_fH2", halo_average_fH2)
    add_quantity("total_young_stars8_mass", halo_total_young_stars8_mass)
    add_quantity("total_young_stars7_mass", halo_total_young_stars7_mass)
    add_quantity("sfr8", halo_sfr8)
    add_quantity("sfr7", halo_sfr7)
    
    
    hc.add_quantity("average_temperature")
    hc.add_quantity("average_metallicity")
    hc.add_quantity("total_gas_mass")
    hc.add_quantity("total_star_mass")
    hc.add_quantity("average_fH2")
    hc.add_quantity("total_young_stars8_mass")
    hc.add_quantity("total_young_stars7_mass")
    hc.add_quantity("sfr8")
    hc.add_quantity("sfr7")
    
    hc.create()
    
    new_ds = yt.load(simulation_dir+'/halo_catalogs/'+snapname+'/'+snapname+'.0.h5')
    all_data = new_ds.all_data()

    temp = all_data["halos", "average_temperature"] 
    metals = all_data["halos", "average_metallicity"]
    x = all_data["halos", "particle_position_x"].in_units('kpc')
    y = all_data["halos", "particle_position_y"].in_units('kpc')
    z = all_data["halos", "particle_position_z"].in_units('kpc')
    total_gas_mass = all_data["halos", "total_gas_mass"].in_units('Msun')
    total_halo_mass = all_data["halos", "particle_mass"].in_units('Msun')
    total_star_mass = all_data["halos", "total_star_mass"].in_units('Msun')
    average_fH2 = all_data["halos", "average_fH2"]
    young_stars8_mass = all_data["halos", "total_young_stars8_mass"].in_units('Msun')
    sfr8 = all_data["halos", "sfr8"].in_units('Msun/yr')
    sfr7 = all_data["halos", "sfr7"].in_units('Msun/yr')
    
    rvir = all_data["halos", "virial_radius"].in_units('kpc') 
    
    average_fH2 = all_data["halos", "average_fH2"]
    
    center0 = [float(x.in_units('code_length')[0]), float(y.in_units('code_length')[0]), float(z.in_units('code_length')[0])] 
    center0 
    
    halo0 = ds.sphere(center0, radius = (float(rvir[0]) , 'kpc') ) 
    
    a = Table() 
    a['root_index'] = halo0['particle_index']
    a.write('halo_'+halo_id+'_root_index.txt', format='ascii') 
    
    p = yt.ProjectionPlot(ds, 'y', 'density', weight_field='density', 
                          data_source=box, center=center0,  origin='native', width=(400, 'kpc'))
    p.set_cmap('density', density_color_map)
    p.annotate_title(ds._input_filename[-6:])
    p.annotate_timestamp(redshift=True)
    p.set_zlim('density', 1e-28, 1e-21)
    p.annotate_halos(new_ds) 
    p.save()
    



parser = argparse.ArgumentParser()
parser.add_argument('--halo', type=int, required=True)
parser.add_argument('--snapshot', type=str, required=True)
args = parser.parse_args()

halo_id = '00'+str(args.halo)

get_root_halo(halo_id, args.snapshot) 

