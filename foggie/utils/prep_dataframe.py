"""
This is a utility function that accepts a dataset and other arguments
and returns a dataframe with the requested fields as columns.
This is useful to put datasets in the proper dataframe format,
correct units, log scales, etc. for shading by other code.
"""
import pandas as pd
import numpy as np 
import yt 
import glob
import pickle 
import foggie.utils.foggie_utils as futils
from foggie.utils.consistency import axes_label_dict, logfields, categorize_by_temp, \
    categorize_by_metals, categorize_by_fraction


def rays_to_dataframe(halo, run, wildcard): 
    """ This function obtains a list of ray files (h5) and opens then and 
    then concatenates the contents into a single dataframe. 
    
    Try: 
    import foggie.utils.prep_dataframe as pdf
    a = pdf.rays_to_dataframe('8508', 'nref11c_nref9f', '*_x_*')
    
    Then feed the result into render_image 
    """
    
    list_of_trident_rays = futils.get_list_of_trident_rays(halo, run, wildcard)
    list_of_frames = []

    for trident_ray in list_of_trident_rays:   
        ds = yt.load(trident_ray)
        p = ds.all_data()
    
        #need to be able to construct the name of the cloud pkl file from the trident ray name 
        bits_of_names = (trident_ray.split('/')[-1] ).split('_')
        wildcard = '*'+bits_of_names[2]+'_'+bits_of_names[3]+'_'+bits_of_names[4]+'*'
        file = glob.glob(wildcard.replace('.', '')+'pkl')
    
        #now open the pickle 
        pkl = pickle.load(open(file[0], "rb" ))
    
        #sort the dataframe for the ray by the x coordinate 
        #the dataframe coordinates are in code units 
        df = pkl["ray_df"]
        df.sort_values(by='x', axis=0, inplace=True, ascending=True)
    
        list_of_frames.append(df)

    all_sightlines = pd.concat(list_of_frames)

    return all_sightlines

def check_dataframe(frame, field1, field2, count_cat): 
    """ This function checks the input dataframe for properties that 
    it will need to properly shade the fields in the frame. For instance, 
    it checks that certain fields are log fields and have the proper units.
    It only checks the two fields that are going to be shaded so it 
    needs to be run prior to render_image each time.""" 

    field_names = field1+'_'+field2

    if ('pressure' in field_names and 'pressure' in frame.keys()): 
        if (frame['pressure'].max() > 0.):   # if pressure is NOT in log units
            frame['pressure'] = np.log10(frame['pressure'])

    if ('temperature' in field_names and 'temperature' in frame.keys()): 
        if (frame['temperature'].max() > 10.):   # if pressure is NOT in log units
            frame['temperature'] = np.log10(frame['temperature'])

    if ('cell_mass' in field_names and 'cell_mass' in frame.keys()): 
        if (frame['cell_mass'].max() > 1e33):   # if cell_mass is NOT in log units
            frame['cell_mass'] = np.log10(frame['cell_mass'] / 1.989e33)

    if ('number_density' in field1):
        frame[field1] = np.log10(frame[field1])

    if ('number_density' in field2):
        frame[field2] = np.log10(frame[field2])

    if ('cooling_time' in field_names and 'cooling_time' in frame.keys()): 
        if (frame['cooling_time'].max() > 1e12):   # if cooling_time is NOT in log units
            frame['cooling_time'] = np.log10(frame['cooling_time'] / 3.156e7)

    if ('phase' in count_cat): 
        frame['phase'] = categorize_by_temp(frame['temperature'])
        frame.phase = frame.phase.astype('category')

    if ('metal' in count_cat): 
        frame['metal'] = categorize_by_temp(frame['metallicity'])
        frame.metal = frame.metal.astype('category')

    return frame

def prep_dataframe(dataset, all_data, field_list, category, \
                halo_center=[0.5, 0.5, 0.5], halo_vcenter=[0.,0.,0.]):
    """ add the fields specified in the field_list to the dataframe, 
        and any intermediate fields that are needed to derive them.

        The input is a list of fields, and the time this takes 
        will be proportional to the length of this list. 

        These are checked against the "logfields" dictionary in
        consistency to take their log before placing into the df.

        Returns the dataframe with these fields, which can be 
        fed into render_image and other things. 

        The 'category' argument is the label for the method of colorcoding.
        It is usually 'phase' for temperature coding, 'metal' for metallicity 
        coding, etc. 
        """

    print("you have requested fields ", field_list)

    field_names = ''
    for f in field_list: field_names = field_names + '_' + f 

    data_frame = pd.DataFrame({}) # create the empty df for the desired fields.

    if (('position' in field_names) or ('radius' in field_names)):
        cell_size = np.array(all_data["cell_volume"].in_units('kpc**3'))**(1./3.)

        if ('position_x' in field_names):
            x = (all_data['x'].in_units('kpc')).ndarray_view()
            #x = x + cell_size * (np.random.rand(np.size(cell_size)) * 2. - 1.)
            x = x - np.mean(x)
            data_frame['position_x'] = x

        if ('position_y' in field_names):
            y = (all_data['y'].in_units('kpc')).ndarray_view()
            #y = y + cell_size * (np.random.rand(np.size(cell_size)) * 2. - 1.)
            y = y - np.mean(y)
            data_frame['position_y'] = y

        if ('position_z' in field_names):
            z = (all_data['z'].in_units('kpc')).ndarray_view()
            #z = z + cell_size * (np.random.rand(np.size(cell_size)) * 2. - 1.)
            z = z - np.mean(z)
            data_frame['position_z'] = z

    if ('cell_mass' in field_names):
        data_frame['cell_mass'] = np.log10(all_data['cell_volume'].in_units('kpc**3') * \
                                    all_data['density'].in_units('Msun / kpc**3') )

    if ('cell_size' in field_names):
        data_frame['cell_size'] = (all_data['cell_volume'].in_units('pc**3') )**(1./3.)

    if ("entropy" in field_names): data_frame["entropy"] = np.log10(all_data["entropy"].in_units('cm**2*erg')) 
    
    #unfortunately we will have to recompute the radius since it is bogus for some of our input regions 
    if ("radius" in field_names): 
        print("will have to recompute the radius! halo_center = ", halo_center)
        dx = ( all_data['x'].in_units('code_length') - dataset.arr(halo_center[0], 'code_length') ).in_units('kpc') 
        dy = ( all_data['y'].in_units('code_length') - dataset.arr(halo_center[1], 'code_length') ).in_units('kpc') 
        dz = ( all_data['z'].in_units('code_length') - dataset.arr(halo_center[2], 'code_length') ).in_units('kpc') 
        r2 = dx**2 + dy**2 + dz**2 
        data_frame["radius"] = r2**0.5                 

    if ("cooling_time" in field_names): data_frame["cooling_time"] = np.log10(all_data["cooling_time"].in_units('yr'))                 

    #this for loop handles all the fields that come directly from the ds with the same names     
    for thisfield in field_list:
        if thisfield not in data_frame.columns:    #  add those two fields
            print("Did not find field = "+thisfield+" in the dataframe, will add it.")
            if thisfield in logfields:
                print("Field "+thisfield+" is a log field.")
                data_frame[thisfield] = np.log10(all_data[thisfield])
            else:
                data_frame[thisfield] = all_data[thisfield]
                if ('vel' in thisfield): data_frame[thisfield] = all_data[thisfield].in_units('km/s')


    if ('phase' in category):
        if ('temperature' not in data_frame.columns):
            data_frame['temperature'] = np.log10(all_data['temperature'])

        data_frame['phase'] = categorize_by_temp(data_frame['temperature'])
        data_frame.phase = data_frame.phase.astype('category')
        print('Added phase category to the dataframe')

    if ('metal' in category):
        if ('metallicity' not in data_frame.columns):
            data_frame['metallicity'] = all_data['metallicity']

        data_frame['metal'] = categorize_by_metals(all_data['metallicity'])
        data_frame.metal = data_frame.metal.astype('category')
        print('Added metal category to the dataframe')

    if ('ion_fraction' in category):
        if (category not in data_frame.columns):
            data_frame[category] = all_data[category]
        print('Added frac = '+category+' category to the dataframe')

    return data_frame
