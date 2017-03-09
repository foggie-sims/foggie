
import yt
import numpy as np 
import get_background_density 

def get_halo_profile(ds, halo_center): 
 
    BoxDensity = get_background_density.get_background_density(ds) 
    radii = ds.arr((np.arange(100) * 5. + 50.), 'kpc').in_units('Mpc') 

    for radius_to_analyze in radii: 
        sphere = ds.sphere(halo_center, radius_to_analyze) 
        baryon_mass, particle_mass = sphere.quantities.total_quantity(["cell_mass", "particle_mass"])
        TotalMass = (baryon_mass + particle_mass).in_units('Msun')
        GasMass = baryon_mass.in_units('Msun')
        ParticleMass = particle_mass.in_units('Msun')
        HaloDensity = TotalMass / (4. / 3. * np.pi * (radius_to_analyze)**3)
        ptype = sphere["particle_type"]
        mass = sphere['particle_mass'].convert_to_units('Msun') 
        StarMass = np.sum(mass[ptype == 2])
    
        HaloOverDensity = HaloDensity/BoxDensity 
 
        if (HaloOverDensity < 200.): 
	    return radius_to_analyze.in_units('kpc'), TotalMass, ParticleMass, GasMass, StarMass  
