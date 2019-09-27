import trident

%timeit ds = trident.make_onezone_ray(column_densities={'H_number_density': 1e21})
