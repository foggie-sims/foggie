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

def prep_dataframe(cut_region, field_list, categories):
    """ input is a cut_region, like "cgm", or "cool_outflows" 
        field_list is the fields specified that will be added 
        to the dataframe.

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

    if (('gas','temperature') not in field_list): field_list.append(('gas','temperature')) 
    
    data_frame = cut_region.to_dataframe(field_list) #most of the work is done here. 

    if ( ("gas", "x") in field_list):
        x = (cut_region[("gas", "x")].in_units('kpc')).ndarray_view()
        x = x - np.mean(x)
        data_frame["x"] = x

    if (("gas", "y") in field_list):
        y = (cut_region[("gas", "y")].in_units('kpc')).ndarray_view()
        y = y - np.mean(y)
        data_frame["y"] = y

    if (("gas", "z") in field_list):
        z = (cut_region[("gas", "z")].in_units('kpc')).ndarray_view()
        z = z - np.mean(z)
        data_frame["z"] = z

    if (("gas","cooling_time")in field_list): data_frame["cooling_time"] = cut_region["cooling_time"].in_units('yr')               

    for key in data_frame.keys():
        if (key in logfields): 
            data_frame[key] = np.log10(data_frame[key])

    if ('phase' in categories):
        data_frame['phase'] = categorize_by_temp(data_frame['temperature'])
        data_frame.phase = data_frame.phase.astype('category')
        print('Added phase category to the dataframe')

    if (('gas','cell_mass') in field_list) or ('cell_mass' in categories):
        data_frame['cell_mass'] = np.log10(cut_region[('gas','cell_mass')].in_units('Msun')) 

    if ( ('gas','entropy') in field_list): data_frame["entropy"] = np.log10(cut_region["entropy"].in_units('cm**2*erg'))              

    if ('metal' in categories):
        if ('metallicity' not in data_frame.columns):
            data_frame['metallicity'] = cut_region['metallicity']

        print('df in prep_df', data_frame)
        data_frame['metal'] = categorize_by_metals(cut_region['metallicity'])
        data_frame.metal = data_frame.metal.astype('category')
        print('Added metal category to the dataframe')

    return data_frame

    """
    if ('ion_fraction' in category):
        if (category not in data_frame.columns):
            data_frame[category] = all_data[category]
        print('Added frac = '+category+' category to the dataframe') 
    """