import trident
import numpy as np 
import yt
import analysis 

 

def get_ion_plots(ds, halo_center, ray): 

    proper_box_size = ds.get_parameter('CosmologyComovingBoxSize') / ds.get_parameter('CosmologyHubbleConstantNow') * 1000. # in kpc 
    line_list = ['H', 'O'] # 'C', 'N', 'Si', 'O'] 

    for axis in ['z','y','x']: 
    
        p = yt.ProjectionPlot(ds, axis, 'density', center=halo_center, width=(500,'kpc'))
        p.annotate_ray(ray, arrow=True)
        p.annotate_grids() 
        p.save()
    
        p = yt.SlicePlot(ds, axis, 'metallicity', center=halo_center, width=(500,'kpc'))
        p.annotate_ray(ray, arrow=True)
        p.annotate_grids() 
        p.save()

        proj = yt.SlicePlot(ds, axis, "O_p5_number_density", center=halo_center, width=(500,'kpc'))
        p.annotate_grids() 
        proj.save('OVI_Slice')
        
        # O VI map? 
        p = yt.ProjectionPlot(ds, axis, 'O_p5_number_density', center=halo_center, width=(500,'kpc'))
        p.annotate_grids() 
        p.save('OVI_Column') 


        proj = yt.SlicePlot(ds, axis, "H_p0_number_density", center=halo_center, width=(500,'kpc'))
        proj.save('HI')
        p = yt.ProjectionPlot(ds, axis, 'H_p0_number_density', center=halo_center, width=(500,'kpc'))
        p.annotate_grids() 
        p.save('HI_Column') 
   
        proj = yt.SlicePlot(ds, axis, "Si_p2_number_density", center=halo_center, width=(500,'kpc'))
        p.annotate_grids() 
        proj.save('SiIII_Column')
       
        sph = ds.sphere(center=halo_center, radius=(500,'kpc')) 
   
        phase = yt.PhasePlot(sph, "density", "temperature", ["O_p5_mass"], weight_field="O_p5_mass", fractional=True)
        phase.save() 
                     





