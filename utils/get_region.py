
from foggie.consistency import cgm_inner_radius, cgm_outer_radius, \
                    cgm_field_filter, ism_field_filter


def get_region(data_set, region): 
    """ this function takes in a dataset and returns a CutRegion 
    that corresponds to particular FOGGIE regions- JT 091619"""

    refine_box_center = data_set['MustRefineRegionLeftEdge'] + \
                            0.5*( data_set['MustRefineRegionRightEdge'] - data_set['MustRefineRegionLeftEdge'] )
    refine_box = data_set.r[ data_set['MustRefineRegionLeftEdge'][0]:data_set['MustRefineRegionRightEdge'][0], 
                         data_set['MustRefineRegionLeftEdge'][1]:data_set['MustRefineRegionRightEdge'][1], 
                         data_set['MustRefineRegionLeftEdge'][2]:data_set['MustRefineRegionRightEdge'][2] ]

    if region == 'trackbox':
        print("prep_dataset: your region is the refine box as determined from dataset (NOT track)")
        all_data = refine_box
    elif region == 'rvir':
        print("prep_dataset: your region is Rvir = 300 kpc sphere centered on the box")
        sph = data_set.sphere(center=refine_box_center, radius=(300, 'kpc'))
        all_data = sph
    elif region == 'domain': 
        print("prep_dataset: your region is the entire domain, prepare to wait")
        print("prep_dataset: on second thought maybe you don't want to do this")
        all_data = data_set.all_data() 
    elif region == 'cgm': 
        print("prep_dataset: your region is the CGM as determined by consistency")
        cen_sphere = data_set.sphere(refine_box_center, (cgm_inner_radius, "kpc"))     #<---- STILL using the box center from the trackfile above 
        rvir_sphere = data_set.sphere(refine_box_center, (cgm_outer_radius, 'kpc')) 
        cgm = rvir_sphere - cen_sphere
        cgm_cut = cgm.cut_region([cgm_field_filter])    #<---- cgm_field_filter is from consistency.py 
        all_data = cgm_cut 
        print(np.min(all_data['radius'])) 
    elif region == 'ism': 
        print("prep_dataset: your region is the ISM as determined by consistency")
        cen_sphere = data_set.sphere(refine_box_center, (cgm_inner_radius, "kpc"))     #<---- STILL using the box center from the trackfile above 
        rvir_sphere = data_set.sphere(refine_box_center, (cgm_outer_radius, 'kpc')) 
        cold_inside_rvir =  rvir_sphere.cut_region([ism_field_filter])
        ism = cen_sphere + cold_inside_rvir
        all_data = ism   
    else:
        print("prep_dataset: your region is invalid!")

    return all_data