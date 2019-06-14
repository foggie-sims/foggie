
"""
This is a utility function that accepts a dataset and other arguments
and returns a dataframe with the requested fields as columns. 
This is useful to put datasets in the proper dataframe format, 
correct units, log scales, etc. for shading by other code. 

This was forked out of shade_maps into a freestanding function 
on 061419 JT. 
""" 
import pandas as pd


def prep_dataframe(ds, refine_box, refine_width, field1, field2, \
                        halo_center, halo_vcenter):
    """ add fields to the dataset, create dataframe for rendering
        The enzo fields x, y, z, temperature, density, cell_vol, cell_mass,
        and metallicity will always be included, others will be included
        if they are requested as fields. """

    all_data = ds.all_data() 
    
    field_names = field1+'_'+field2
    print('field_names: ', field_names)
    
    temperature = np.log10(all_data['temperature'])
    data_frame = pd.DataFrame({'temperature':temperature}) 

    if ('position' in field_names):
        cell_size = np.array(all_data["cell_volume"].in_units('kpc**3'))**(1./3.)

        if ('position_x' in field_names): 
            x = (all_data['x'].in_units('kpc')).ndarray_view()
            x = x + cell_size * (np.random.rand(np.size(cell_size)) * 2. - 1.)
            x = x - np.min(x)
            data_frame['position_x'] = x

        if ('position_y' in field_names): 
            y = (all_data['y'].in_units('kpc')).ndarray_view()
            y = y + cell_size * (np.random.rand(np.size(cell_size)) * 2. - 1.)
            y = y - np.min(y)
            data_frame['position_y'] = y
   
        if ('position_x' in field_names): 
            z = (all_data['z'].in_units('kpc')).ndarray_view()
            z = z + cell_size * (np.random.rand(np.size(cell_size)) * 2. - 1.)
            z = z - np.min(z)
            data_frame['position_z'] = z


    #if ('rad' in field1 or 'rad' in field2): 
    #    radius = ((x-halo_center[0])**2 + (y-halo_center[1])**2 + (z-halo_center[2])**2 )**0.5
    
    phase = new_categorize_by_temp(temperature)
    data_frame['phase'] = phase
    metal = new_categorize_by_metals(all_data['metallicity'])
    data_frame['metal'] = metal
    frac = categorize_by_fraction(all_data['O_p5_ion_fraction'], all_data['temperature'])
    data_frame['frac'] = frac
    data_frame.phase = data_frame.phase.astype('category')
    data_frame.metal = data_frame.metal.astype('category')
    data_frame.frac  = data_frame.frac.astype('category')

    if ('density' in field1 or 'density' in field2):  
        data_frame['cell_mass'] = np.log10(all_data['density']) 

    if ('cell_mass' in field1 or 'cell_mass' in field2):  
        data_frame['cell_mass'] = np.log10(get_cell_mass(all_data))

    if ('relative_velocity' in field1 or 'relative_velocity' in field2): 
        relative_velocity = ( (all_data['x-velocity'].in_units('km/s')-halo_vcenter[0])**2 \
                            + (all_data['y-velocity'].in_units('km/s')-halo_vcenter[1])**2 \
                            + (all_data['z-velocity'].in_units('km/s')-halo_vcenter[2])**2 )**0.5
        data_frame['relative_velocity'] = relative_velocity

    print("you have requested fields ", field1, field2)

    if field1 not in data_frame.columns:    #  add those two fields
        print("Did not find field 1 = "+field1+" in the dataframe, will add it.")
        if field1 in logfields:
            print("Field 1, "+field1+" is a log field.")
            data_frame[field1] = np.log10(all_data[field1])
        else:
            data_frame[field1] = all_data[field1]
            if ('vel' in field1): data_frame[field1] = all_data[field1].in_units('km/s')
    if field2 not in data_frame.columns:
        print("Did not find field 2 = "+field2+" in the dataframe, will add it.")
        if field2 in logfields:
            print("Field 2, "+field2+" is a log field.")
            data_frame[field2] = np.log10(all_data[field2])
        else:
            data_frame[field2] = all_data[field2]
            if ('vel' in field2): data_frame[field2] = all_data[field2].in_units('km/s')

    return data_frame


