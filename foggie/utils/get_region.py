
from foggie.utils.consistency import cgm_inner_radius, cgm_outer_radius, cgm_field_filter, ism_field_filter
import numpy as np 
from unyt import unyt_array

def get_region(data_set, region, filter='None', sphere_size=25., left_corner=[0,0,0], right_corner=[80,80,80]): 
    """ this function takes in a dataset and returns a CutRegion 
    that corresponds to particular FOGGIE regions- JT 091619"""

    refine_box_center = data_set['MustRefineRegionLeftEdge'] + \
                            0.5*( data_set['MustRefineRegionRightEdge'] - data_set['MustRefineRegionLeftEdge'] )
    refine_box = data_set.r[ data_set['MustRefineRegionLeftEdge'][0]:data_set['MustRefineRegionRightEdge'][0], 
                         data_set['MustRefineRegionLeftEdge'][1]:data_set['MustRefineRegionRightEdge'][1], 
                         data_set['MustRefineRegionLeftEdge'][2]:data_set['MustRefineRegionRightEdge'][2] ]

    if region == 'trackbox':
        print("get_region: your region is the refine box as determined from dataset (NOT track)")
        print("get_region: the filter will be: ", filter)
        if (filter == 'None'): 
            all_data = refine_box   #<---- cgm_field_filter is from consistency.py 
        else: 
            all_data = refine_box.cut_region(filter) 
    elif region == 'rvir':
        print("get_region: your region is Rvir = 200 kpc sphere centered on the box")
        print("get_region: the filter will be: ", filter)
        rvir  = data_set.sphere(center=data_set.halo_center_code, radius=(200, 'kpc'))   #<---- cgm_field_filter is from consistency.py 
        if (filter == 'None'): 
            all_data = rvir
        else: 
            all_data = rvir.cut_region(filter) 
    elif region == 'domain': 
        print("get_region: your region is the entire domain, prepare to wait")
        print("get_region: on second thought maybe you don't want to do this")
        print("get_region: the filter will be: ", filter)
        all_data = data_set.all_data() 
    elif region == 'cgm': 
        print("get_region: your region is the CGM as determined by consistency, center = ", data_set.halo_center_code)
        print("get_region: the filter will be: ", filter)
        cen_sphere = data_set.sphere(data_set.halo_center_code, (cgm_inner_radius, "kpc"))  #<--using box center from the trackfile above 
        rvir_sphere = data_set.sphere(data_set.halo_center_code, (cgm_outer_radius, 'kpc')) 
        cgm = rvir_sphere - cen_sphere
        if (filter == 'None'): 
            all_data = cgm.cut_region(cgm_field_filter)   #<---- cgm_field_filter is from consistency.py 
        else: 
            all_data = cgm.cut_region(filter) 
    elif region == 'ism': 
        print("get_region: your region is the ISM as determined by consistency, center = ", data_set.halo_center_code)
        print("get_region: the filter will be: ", filter)
        cen_sphere = data_set.sphere(data_set.halo_center_code, (cgm_inner_radius, "kpc"))     #<--using the box center from the trackfile above 
        rvir_sphere = data_set.sphere(data_set.halo_center_code, (cgm_outer_radius, 'kpc')) 
        if (filter == 'None'): 
            cold_inside_rvir = rvir_sphere.cut_region(ism_field_filter)   #<---- cgm_field_filter is from consistency.py 
        else: 
            cold_inside_rvir = rvir_sphere.cut_region(filter) 
        all_data = cen_sphere + cold_inside_rvir
    elif region == 'xyslice': 
        print("get_region: your region is a slice along x-y axes, short along z")
        refine_box = data_set.r[0:1, 0:1, data_set['MustRefineRegionLeftEdge'][2]:data_set['MustRefineRegionRightEdge'][2] ]
        if (filter == 'None'): 
            all_data = refine_box
        else: 
            all_data = refine_box.cut_region(filter) 
    elif region == 'yzslice': 
        print("get_region: your region is a slice along y-z axes, short along x")
        refine_box = data_set.r[data_set['MustRefineRegionLeftEdge'][0]:data_set['MustRefineRegionRightEdge'][0], 0:1, 0:1 ]
        if (filter == 'None'): 
            all_data = refine_box
        else: 
            all_data = refine_box.cut_region(filter) 
    elif region == 'xzslice': 
        print("get_region: your region is a slice along z-z axes, short along y")
        refine_box = data_set.r[0:1, data_set['MustRefineRegionLeftEdge'][1]:data_set['MustRefineRegionRightEdge'][1], 0:1]
        if (filter == 'None'): 
            all_data = refine_box
        else: 
            all_data = refine_box.cut_region(filter) 
    elif region == 'cube-sphere': 
        # draw a cube of given size/shape relative to center, cut by a sphere of given size
        # originally developed for efficient clump finding 
        cube =  data_set.r[data_set.halo_center_kpc[0]+unyt_array(left_corner[0],'kpc'):data_set.halo_center_kpc[0]+unyt_array(right_corner[0], 'kpc'), \
                data_set.halo_center_kpc[1]+unyt_array(left_corner[1],'kpc'):data_set.halo_center_kpc[1]+unyt_array(right_corner[1], 'kpc'), \
                data_set.halo_center_kpc[2]+unyt_array(left_corner[2],'kpc'):data_set.halo_center_kpc[2]+unyt_array(right_corner[2], 'kpc')]
        
        sph = data_set.sphere(data_set.halo_center_kpc, radius=(sphere_size, 'kpc'))
        cut_region = cube-sph
        all_data = cut_region
    else:
        print("get_region: your region is invalid!")

    return all_data
