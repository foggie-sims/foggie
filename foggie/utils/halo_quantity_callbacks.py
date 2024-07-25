

def halo_average_temperature(halo):
    sphere = halo.data_object    # this sphere will have been made for us by the "sphere" callback

    # if the halo was too small, the sphere callback will return a None, then just return 0 (in correct units)
    if sphere is None:
        return halo.halo_catalog.data_ds.quan(0, "K")

    # use the sphere to calculate average temperature weighted by cell mass
    # using the weighted_average_quantity derived quantity
    return sphere.quantities.weighted_average_quantity(
        ("gas", "temperature"), ("gas", "cell_mass"))

def halo_average_metallicity(halo):
    sphere = halo.data_object    # this sphere will have been made for us by the "sphere" callback

    # if the halo was too small, the sphere callback will return a None, then just return 0 (in correct units)
    if sphere is None:
        return halo.halo_catalog.data_ds.quan(0, "dimensionless")

    # use the sphere to calculate average temperature weighted by cell mass
    # using the weighted_average_quantity derived quantity
    return sphere.quantities.weighted_average_quantity(
        ("gas", "metallicity"), ("gas", "cell_mass"))

def halo_total_gas_mass(halo): 
    sphere = halo.data_object    # this sphere will have been made for us by the "sphere" callback

    if sphere is None:
        return halo.halo_catalog.data_ds.quan(0, "Msun")

    return sphere.quantities.total_quantity(("gas", "cell_mass"))

def halo_total_star_mass(halo): 
    sphere = halo.data_object    # this sphere will have been made for us by the "sphere" callback

    if sphere is None:
        return halo.halo_catalog.data_ds.quan(0, "Msun")

    return sphere.quantities.total_quantity(('stars', 'particle_mass'))

def halo_average_fH2(halo):
    sphere = halo.data_object    

    if sphere is None:
        return halo.halo_catalog.data_ds.quan(0, "Msun")

    fH2 = sphere.quantities.weighted_average_quantity(("gas", "H2_fraction"), ("gas", "cell_mass"))

    return fH2