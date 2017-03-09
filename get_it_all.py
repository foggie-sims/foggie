
import analysis 

def get_it_all(run, center_guess, fileroot): 

    # this is a dataset 
    ds = foggie.get_dataset(run) 
    
    # get the halo center, plot refinement vs. radius 
    halo_center, radius, cell_size =  foggie.get_halo_center(ds, center_guess) 
    ad = ds.sphere(halo_center, (500., 'kpc'))

    foggie.plot_cell_vs_radius(ds, ad, halo_center) 

    foggie.shade_render(ds, ad, halo_center, 'rvir') 

    box = ds.box([0.48966962, 0.47133064, 0.5093916], [0.49001712, 0.47202564, 0.509739]) 
    foggie.shade_render(ds, box, halo_center, 'box') 

    # trim out a sphere, plot density profiles 
    #ad = foggie.basic_funcs.trim_halo_sphere(ds, halo_center, 500.)  
    #foggie.basic_funcs.plot_density_profiles(ad)

    ray, out_spec = foggie.get_spec(ds, halo_center, 75.)

    foggie.get_ion_plots(ds, halo_center, ray)


   
    
