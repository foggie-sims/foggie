import yt
from astropy.table import Table 
import foggie.utils.get_refine_box as grb                                                                                                                                                                                        
import trident 
from foggie.utils.consistency import * 




#box = ds.r[xL:xR:640j,yL:yR:640j,zL:zR:640j]




def filtered_projections(dataset_name): 
    ds = yt.load(dataset_name)
    ad = ds.all_data() 
    trident.add_ion_fields(ds, ions=['H I', 'C IV', 'O VI', 'Mg II', 'Si II', 'C II', 'Si III', 'Si IV', 'Ne VIII'])

    trackfile = '/Users/tumlinson/Dropbox/FOGGIE/foggie/halo_tracks/008508/nref11n_selfshield_15/halo_track_200kpc_nref10'                                                                                                  

    track = Table.read(trackfile, format='ascii')                                                                                                                                                                           
    track.sort('col1') 
    refine_box, refine_box_center, _ = grb.get_refine_box(ds, ds.current_redshift, track)                                                                                                                                   

    # all HI cells 
    p = yt.ProjectionPlot(ds, 'x', "H_p0_number_density", center=refine_box_center, width=(0.5, 'Mpc'))                                                                                
    p.set_cmap(field='H_p0_number_density', cmap = h1_color_map)                                                                                                                                                            
    p.annotate_timestamp(redshift=True)                                                                                                                                                                                     
    p.set_zlim('H_p0_number_density', 1e8, h1_proj_max)                                                                                                                                                             
    p.save('h1_all')                                                                                                                                                                                                                

    # HI 'clouds' only 
    h1clouds = ad.cut_region(["obj['H_p0_number_density'] > 1e-8"])                                                                                                                                                         
    p = yt.ProjectionPlot(ds, 'x', "H_p0_number_density", center=refine_box_center, width=(0.5, 'Mpc'), data_source=h1clouds)                                                                                
    p.set_cmap(field='H_p0_number_density', cmap = h1_color_map)                                                                                                                                                            
    p.annotate_timestamp(redshift=True)                                                                                                                                                                                     
    p.set_zlim('H_p0_number_density', 1e8, h1_proj_max)                                                                                                                                                             
    p.save('h1_clouds')   

    # all OVI cells 
    p = yt.ProjectionPlot(ds, 'x', "O_p5_number_density", center=refine_box_center, width=(0.5, 'Mpc'))                                                                                
    p.set_cmap(field='O_p5_number_density', cmap = o6_color_map)                                                                                                                                                            
    p.annotate_timestamp(redshift=True)                                                                                                                                                                                     
    p.set_zlim('O_p5_number_density', o6_min, o6_max)                                                                                                                                                             
    p.save('o6_all')                                                                                                                                                                                             

    # OVI 'clouds' only  
    o6clouds = ad.cut_region(["obj['O_p5_number_density'] > 1e-10"])                                                                                                                                                         
    p = yt.ProjectionPlot(ds, 'x', "O_p5_number_density", center=refine_box_center, width=(0.5, 'Mpc'), data_source=o6clouds)                                                                                
    p.set_cmap(field='O_p5_number_density', cmap = o6_color_map)                                                                                                                                                            
    p.annotate_timestamp(redshift=True)                                                                                                                                                                                     
    p.set_zlim('O_p5_number_density', o6_min, o6_max)                                                                                                                                                             
    p.save('o6_clouds')                                                                                                                                                                                             

    #crosses 
    p = yt.ProjectionPlot(ds, 'x', "H_p0_number_density", center=refine_box_center, width=(0.5, 'Mpc'), data_source=o6clouds)                                                                                
    p.set_cmap(field='H_p0_number_density', cmap = h1_color_map)                                                                                                                                                            
    p.annotate_timestamp(redshift=True)                                                                                                                                                                                     
    p.set_zlim('H_p0_number_density', 1e8, h1_proj_max)                                                                                                                                                             
    p.save('h1_o6_cross')   
                                                                                                                                                      
    p = yt.ProjectionPlot(ds, 'x', "O_p5_number_density", center=refine_box_center, width=(0.5, 'Mpc'), data_source=h1clouds)                                                                                
    p.set_cmap(field='O_p5_number_density', cmap = o6_color_map)                                                                                                                                                            
    p.annotate_timestamp(redshift=True)                                                                                                                                                                                     
    p.set_zlim('O_p5_number_density', o6_min, o6_max)                                                                                                                                                             
    p.save('o6_h1_cross')    

    # both 
    both = ad.cut_region(["(obj['H_p0_number_density'] > 1e-8) & (obj['O_p5_number_density'] > 1e-10)"])                                                                                                                                                         
    p = yt.ProjectionPlot(ds, 'x', "H_p0_number_density", center=refine_box_center, width=(0.5, 'Mpc'), data_source=both)                                                                                
    p.set_cmap(field='H_p0_number_density', cmap = h1_color_map)                                                                                                                                                            
    p.annotate_timestamp(redshift=True)                                                                                                                                                                                     
    p.set_zlim('H_p0_number_density', 1e8, h1_proj_max)                                                                                                                                                              
    p.save('h1_o6_both')   

    p = yt.ProjectionPlot(ds, 'x', "O_p5_number_density", center=refine_box_center, width=(0.5, 'Mpc'), data_source=both)                                                                                
    p.set_cmap(field='O_p5_number_density', cmap = o6_color_map)                                                                                                                                                            
    p.annotate_timestamp(redshift=True)                                                                                                                                                                                     
    p.set_zlim('O_p5_number_density', o6_min, o6_max)                                                                                                                                                             
    p.save('o6_h1_both')   
