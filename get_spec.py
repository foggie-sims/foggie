import trident
import numpy as np 
import yt
import analysis 

def get_spec(ds, halo_center, rho_ray): 

    proper_box_size = ds.get_parameter('CosmologyComovingBoxSize') / ds.get_parameter('CosmologyHubbleConstantNow') * 1000. # in kpc 
    line_list = ['H', 'O', 'C', 'N', 'Si'] 

    # big rectangle box 
    ray_start = [halo_center[0]-250./proper_box_size, halo_center[1]+71./proper_box_size, halo_center[2]-71./proper_box_size] 
    ray_end =   [halo_center[0]+250./proper_box_size, halo_center[1]+71./proper_box_size, halo_center[2]-71./proper_box_size] 
 
    ray = trident.make_simple_ray(ds, start_position=ray_start,
                                  end_position=ray_end,
                                  data_filename="ray.h5",
                                  lines=line_list,
                                  ftype='gas')

    sg = trident.SpectrumGenerator(lambda_min=1020, lambda_max=1045, dlambda=0.002)
    trident.add_ion_fields(ds, ions=['C IV', 'O VI','H I','Si III'])

    sg.make_spectrum(ray, lines=line_list) 
    sg.plot_spectrum('spec_final.png') 
    sg.save_spectrum('spec_final.txt') 
    
    # this makes for convenient field extraction from the ray 
    ad = ray.all_data() 
    dl = ad["dl"] # line element length 

    out_dict = {'nhi':np.sum(ad['H_number_density']*dl), 'no6':np.sum(ad['O_p5_number_density']*dl), 'nsi3':np.sum(ad['Si_p2_number_density']*dl)} 
    print np.sum(ad['H_number_density']*dl) 
    print np.sum(ad['O_p5_number_density']*dl) 
    print np.sum(ad['Si_p2_number_density']*dl) 

#yt : [INFO     ] 2017-02-16 21:11:32,811 Creating O_number_density from ray's density, temperature, metallicity.
#yt : [INFO     ] 2017-02-16 21:11:32,860 Creating O_p1_number_density from ray's density, temperature, metallicity.
#yt : [INFO     ] 2017-02-16 21:11:32,902 Creating O_p2_number_density from ray's density, temperature, metallicity.
#yt : [INFO     ] 2017-02-16 21:11:32,948 Creating O_p3_number_density from ray's density, temperature, metallicity.
#yt : [INFO     ] 2017-02-16 21:11:32,991 Creating O_p4_number_density from ray's density, temperature, metallicity.
#yt : [INFO     ] 2017-02-16 21:11:33,034 Creating O_p5_number_density from ray's density, temperature, metallicity.
#yt : [INFO     ] 2017-02-16 21:11:33,044 Creating C_number_density from ray's density, temperature, metallicity.
#yt : [INFO     ] 2017-02-16 21:11:33,083 Creating C_p1_number_density from ray's density, temperature, metallicity.
#yt : [INFO     ] 2017-02-16 21:11:33,124 Creating C_p2_number_density from ray's density, temperature, metallicity.
#yt : [INFO     ] 2017-02-16 21:11:33,161 Creating C_p3_number_density from ray's density, temperature, metallicity.
#yt : [INFO     ] 2017-02-16 21:11:33,171 Creating N_number_density from ray's density, temperature, metallicity.
#yt : [INFO     ] 2017-02-16 21:11:33,243 Creating N_p1_number_density from ray's density, temperature, metallicity.
#yt : [INFO     ] 2017-02-16 21:11:33,282 Creating N_p2_number_density from ray's density, temperature, metallicity.
#yt : [INFO     ] 2017-02-16 21:11:33,319 Creating N_p3_number_density from ray's density, temperature, metallicity.
#yt : [INFO     ] 2017-02-16 21:11:33,363 Creating N_p4_number_density from ray's density, temperature, metallicity.
#yt : [INFO     ] 2017-02-16 21:11:33,402 Creating Si_p1_number_density from ray's density, temperature, metallicity.
#yt : [INFO     ] 2017-02-16 21:11:33,479 Creating Si_p2_number_density from ray's density, temperature, metallicity.
#yt : [INFO     ] 2017-02-16 21:11:33,487 Creating Si_p3_number_density from ray's density, temperature, metallicity.
#yt : [INFO     ] 2017-02-16 21:11:33,561 Creating Si_p11_number_density from ray's density, temperature, metallicity.

    return ray, out_dict 


   




