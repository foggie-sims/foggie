"""
This is a utility function that accepts a dataset and other arguments
and returns a dataframe with the requested fields as columns. 
This is useful to put datasets in the proper dataframe format, 
correct units, log scales, etc. for shading by other code. 
""" 
import pandas as pd
import numpy as np 
from foggie.consistency import axes_label_dict, logfields, new_categorize_by_temp, \
        new_categorize_by_metals, categorize_by_fraction

def prep_dataframe(all_data, field1, field2, category, **kwargs):
    """ add fields1 and 2 to the dataframe, and any intermediate 
        fields that are needed to derive those two.  
        
        Currently takes two fields only, field1 and field2. 

        These are checked against the "logfields" dictionary in 
        consistency to take their log before placing into the df. 

        """
    
    field_list = [field1, field2]
    field_names = field1+'_'+field2
    
    print("you have requested fields ", field_list)

    data_frame = pd.DataFrame({}) # create the empty dataframe to which we will add 
                                  # the desired fields. 
        
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


    if ('cell_mass' in field_names):  
        data_frame['cell_mass'] = all_data['cell_volume'].in_units('kpc**3') * \
                                    all_data['density'].in_units('Msun / kpc**3') 

    if ('relative_velocity' in field_names): 
        relative_velocity = ( (all_data['x-velocity'].in_units('km/s')-halo_vcenter[0])**2 \
                            + (all_data['y-velocity'].in_units('km/s')-halo_vcenter[1])**2 \
                            + (all_data['z-velocity'].in_units('km/s')-halo_vcenter[2])**2 )**0.5
        data_frame['relative_velocity'] = relative_velocity

    for thisfield in field_list: 
        if thisfield not in data_frame.columns:    #  add those two fields
            print("Did not find field = "+thisfield+" in the dataframe, will add it.")
            if thisfield in logfields:
                print("Field "+thisfield+" is a log field.")
                print("what the hell is taking so long? 1")
                data_frame[thisfield] = np.log10(all_data[thisfield])
                print("what the hell is taking so long? 2")
            else:
                data_frame[thisfield] = all_data[thisfield]
                if ('vel' in thisfield): data_frame[thisfield] = all_data[thisfield].in_units('km/s')


    if ('phase' in category): 
        if ('temperature' not in data_frame.columns): 
            data_frame['temperature'] = all_data['temperature']
        
        data_frame['phase'] = new_categorize_by_temp(data_frame['temperature'])
        data_frame.phase = data_frame.phase.astype('category')
        print('Added phase category to the dataframe')
    

    if ('metal' in category): 
        if ('metallicity' not in data_frame.columns): 
            data_frame['metallicity'] = all_data['metallicity']
        
        data_frame['metal'] = new_categorize_by_metals(all_data['metallicity'])
        data_frame.metal = data_frame.metal.astype('category')
        print('Added metal category to the dataframe')

    if ('frac' in category): 
        if ('O_p5_ion_fraction' not in data_frame.columns): 
            data_frame['O_p5_ion_fraction'] = all_data['O_p5_ion_fraction']
        
        data_frame['frac'] = categorize_by_fraction(all_data['O_p5_ion_fraction'], all_data['temperature'])
        data_frame.frac = data_frame.frac.astype('category')
        print('Added frac category to the dataframe')

    return data_frame


