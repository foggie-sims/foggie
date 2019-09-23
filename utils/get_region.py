
from foggie.consistency import cgm_inner_radius, cgm_outer_radius, cgm_field_filter, ism_field_filter
import numpy as np 

def get_region(data_set, region, filter='None'): 
    """ this function takes in a dataset and returns a CutRegion 
    that corresponds to particular FOGGIE regions- JT 091619"""

    refine_box_center = data_set['MustRefineRegionLeftEdge'] + \
                            0.5*( data_set['MustRefineRegionRightEdge'] - data_set['MustRefineRegionLeftEdge'] )
    refine_box = data_set.r[ data_set['MustRefineRegionLeftEdge'][0]:data_set['MustRefineRegionRightEdge'][0], 
                         data_set['MustRefineRegionLeftEdge'][1]:data_set['MustRefineRegionRightEdge'][1], 
                         data_set['MustRefineRegionLeftEdge'][2]:data_set['MustRefineRegionRightEdge'][2] ]

    if region == 'trackbox':
        print("prep_dataset: your region is the refine box as determined from dataset (NOT track)")
        print("prep_dataset: the filter will be: ", filter)
        all_data = refine_box
    elif region == 'rvir':
        print("prep_dataset: your region is Rvir = 200 kpc sphere centered on the box")
        print("prep_dataset: the filter will be: ", filter)
        all_data = data_set.sphere(center=refine_box_center, radius=(200, 'kpc'))
    elif region == 'domain': 
        print("prep_dataset: your region is the entire domain, prepare to wait")
        print("prep_dataset: on second thought maybe you don't want to do this")
        print("prep_dataset: the filter will be: ", filter)
        all_data = data_set.all_data() 
    elif region == 'cgm': 
        print("prep_dataset: your region is the CGM as determined by consistency")
        print("prep_dataset: the filter will be: ", filter)
        cen_sphere = data_set.sphere(refine_box_center, (cgm_inner_radius, "kpc"))     #<---- STILL using the box center from the trackfile above 
        rvir_sphere = data_set.sphere(refine_box_center, (cgm_outer_radius, 'kpc')) 
        cgm = rvir_sphere - cen_sphere
        #print("prep_dataset diagnosis: cgm cen_sphere max radius: ", cen_sphere['radius'].max() / 3.086e21)
        #print("prep_dataset diagnosis: cgm rvir_sphere max radius: ", rvir_sphere['radius'].max() / 3.086e21)
        #print("prep_dataset diagnosis: cgm region cgm max radius: ", cgm['radius'].max() / 3.086e21)
        if (filter == 'None'): 
            all_data = cgm.cut_region(cgm_field_filter)   #<---- cgm_field_filter is from consistency.py 
        else: 
            all_data = cgm.cut_region(filter) 
        #print("prep_dataset diagnosis: cgm region all_data max radius: ", all_data['radius'].max() / 3.086e21)
    elif region == 'ism': 
        print("prep_dataset: your region is the ISM as determined by consistency")
        print("prep_dataset: the filter will be: ", filter)
        cen_sphere = data_set.sphere(refine_box_center, (cgm_inner_radius, "kpc"))     #<---- STILL using the box center from the trackfile above 
        rvir_sphere = data_set.sphere(refine_box_center, (cgm_outer_radius, 'kpc')) 
        #print("prep_dataset diagnosis: ism cen_sphere max radius: ", cen_sphere['radius'].max() / 3.086e21)
        #print("prep_dataset diagnosis: ism rvir_sphere max radius: ", rvir_sphere['radius'].max() / 3.086e21)
        if (filter == 'None'): 
            cold_inside_rvir = rvir_sphere.cut_region(ism_field_filter)   #<---- cgm_field_filter is from consistency.py 
        else: 
            cold_inside_rvir = rvir_sphere.cut_region(filter) 
        all_data = cen_sphere + cold_inside_rvir
        #print("prep_dataset diagnosis: ism region all_data max radius: ", all_data['radius'].max() / 3.086e21)
    else:
        print("prep_dataset: your region is invalid!")

    return all_data