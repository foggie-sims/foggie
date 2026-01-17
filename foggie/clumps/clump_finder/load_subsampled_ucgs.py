import numpy as np
from foggie.clumps.clump_finder.clump_load import create_simple_ucg
from foggie.utils.consistency import *

import h5py

from foggie.clumps.clump_finder.utils_clump_finder import add_cell_id_field

import matplotlib.pyplot as plt
#from matplotlib.colors import LogNorm

from scipy.ndimage import label   
from scipy.ndimage import find_objects
from scipy.ndimage import binary_dilation 
from scipy.ndimage import binary_closing


def load_field_into_subsampled_ucg(ds, refine_box, fields, target_refinement_level=9, max_refinement_level = 11, split_methods=["copy"], merge_methods=["max"]):
    '''
    Load a field into a subsampled uniform covering grid.

    Arguments are:
        ds: The yt dataset
        refine_box: The refine box or cut region to load the field from
        field: The field to load
        target_refinement_level: The target refinement level to subsample to
        max_refinement_level: The maximum refinement level of the refine box
        split_method: The method to use when splitting cells. Options are 'copy', 'average', 'sum'
        merge_method: The method to use when merging cells. Options are 'max', 'min', 'mean', 'sum'
    '''


    if not isinstance(fields,list): fields = [fields]
    if not isinstance(split_methods,list): split_methods = [split_methods]
    if not isinstance(merge_methods,list): merge_methods = [merge_methods]
    
    field_ucgs = create_simple_ucg(ds,refine_box, fields, max_refinement_level,split_methods,merge_methods) #parallelize? Double check overlaps and which edges are being set to nans

    new_ucgs = []
    for field_ucg in field_ucgs:
          new_ucgs.append( subsample_ucg(field_ucg, max_refinement_level-target_refinement_level,merge_methods) )#subsample from 11 to 9

    return new_ucgs

def load_field_into_subsampled_ucg_2(ds, cut_region, fields, target_refinement_level=9, max_refinement_level = 11, split_methods=["copy"], merge_methods=["max"],weight_field=None,weight_merge_method="sum"):
    '''
    Load a field into a subsampled uniform covering grid in a less memory intensive way

    Arguments are:
        ds: The yt dataset
        cut_region: The refine box or cut region to load the field from
        field: The field to load
        target_refinement_level: The target refinement level to subsample to
        max_refinement_level: The maximum refinement level of the refine box
        split_method: The method to use when splitting cells. Options are 'copy' or 'halve'
        merge_method: The method to use when merging cells. Options are 'max', 'min', 'mean', 'sum'
    '''


    if not isinstance(fields,list): fields = [fields]
    if not isinstance(split_methods,list): split_methods = [split_methods]
    if not isinstance(merge_methods,list): merge_methods = [merge_methods]
    
    if target_refinement_level >= max_refinement_level:
        raise ValueError("Target refinement level must be less than max refinement level")

    print('Fields=',fields)
    print("max_refinement_level=",max_refinement_level)
    field_ucgs = create_simple_ucg(ds,cut_region, fields, max_refinement_level-1,split_methods,merge_methods,weight_field=weight_field) #parallelize? Double check overlaps and which edges are being set to nans

    weight_ucg = None
    if weight_field is not None:
        if not isinstance(weight_field,list): weight_field = [weight_field]
        weight_ucg = create_simple_ucg(ds,cut_region, weight_field, max_refinement_level-1,['copy'],[weight_merge_method])[0]


    new_ucgs = []
    for field_ucg in field_ucgs:
        if target_refinement_level < max_refinement_level-1:
            new_ucgs.append( subsample_ucg(field_ucg, max_refinement_level-1-target_refinement_level,merge_methods,weight_ucg=weight_ucg) )#e.g., subsample from 10 to 9
        else:
            new_ucgs.append(field_ucg)

    if len(new_ucgs)==1:
        return new_ucgs[0]
    return new_ucgs

def subsample_ucg(ucg_list, n_subsample,merge_methods,weight_ucg = None):
    from scipy import stats
    '''
    Subsample a uniform covering grid by a factor of 2^n_subsample in each dimension.

    Arguments are:
        ucg: The uniform covering grid to be subsampled
        n_subsample: The number of times to subsample by a factor of 2
        merge_method: The method to use when merging cells. Options are 'max', 'min', 'mean', 'sum'
    '''

    if not isinstance(ucg_list,list): ucg_list = [ucg_list]
    if not isinstance(merge_methods,list): merge_methods = [merge_methods]


    new_ucg_list=[]
    for i in range(0,len(ucg_list)):
        nx,ny,nz = np.shape(ucg_list[i])
        new_nx = np.round(nx/(2**n_subsample)).astype(int)
        new_ny = np.round(ny/(2**n_subsample)).astype(int)
        new_nz = np.round(nz/(2**n_subsample)).astype(int)

        sample = np.indices((nx,ny,nz)).reshape(3,-1).T

        if merge_methods[i] == "weighted_mean" and weight_ucg is not None:
            new_ucg,tmp,tmp =     stats.binned_statistic_dd(sample, np.multiply(weight_ucg,ucg_list[i]).flatten(), statistic="sum", bins=[new_nx,new_ny,new_nz])
            new_weights,tmp,tmp = stats.binned_statistic_dd(sample, weight_ucg.flatten(), statistic="sum", bins=[new_nx,new_ny,new_nz])

            new_ucg = np.divide(new_ucg,new_weights, out=np.zeros_like(new_ucg), where=new_weights!=0)
        else:
            new_ucg,tmp,tmp = stats.binned_statistic_dd(sample,ucg_list[i].flatten(),statistic=merge_methods[i],bins=[new_nx,new_ny,new_nz])

        new_ucg_list.append(new_ucg.astype(ucg_list[i].dtype))

    return new_ucg_list




def load_subsampled_disk_mask(ds, refine_box, disk_clump_dir, target_refinement_level=9, cut_region = None, max_refinement_level = 11):
    '''
    Load a subsampled disk mask for the foggie halos.
    Cut region must completely include the refine box
    '''


    hf = h5py.File(disk_clump_dir,'r')
    disk_cell_ids = hf['cell_ids'][...]
    hf.close()
    

    add_cell_id_field(ds)
    cell_id_field = ('index','cell_id_2')

    fields = [cell_id_field]
    split_methods = ["copy"]
    merge_methods = ["max"]
    ucg_list = create_simple_ucg(ds,refine_box, fields, max_refinement_level,split_methods,merge_methods) #parallelize? Double check overlaps and which edges are being set to nans
    cell_id_ucg = ucg_list[0]
    print("cell_id_ucg shape is",np.shape(cell_id_ucg))
    disk_mask_ucg = np.zeros_like(cell_id_ucg).astype(bool)
    disk_mask_ucg[np.isin(cell_id_ucg,disk_cell_ids)] = True

    disk_mask_ucg = subsample_ucg(disk_mask_ucg, max_refinement_level-target_refinement_level,merge_methods) #subsample from 11 to 9
    disk_mask_ucg=disk_mask_ucg[0]

    if cut_region is not None:
        rb_ledge  = [(refine_box['index','x']-refine_box['index','dx']/2.).min().in_units('kpc').v,
                              (refine_box['index','y']-refine_box['index','dx']/2.).min().in_units('kpc').v,
                              (refine_box['index','z']-refine_box['index','dx']/2.).min().in_units('kpc').v]
        rb_redge = [(refine_box['index','x']+refine_box['index','dx']/2.).max().in_units('kpc').v,
                              (refine_box['index','y']+refine_box['index','dx']/2.).max().in_units('kpc').v,
                              (refine_box['index','z']+refine_box['index','dx']/2.).max().in_units('kpc').v]
        cr_ledge  = [(cut_region['index','x']-cut_region['index','dx']/2.).min().in_units('kpc').v,
                              (cut_region['index','y']-cut_region['index','dx']/2.).min().in_units('kpc').v,
                              (cut_region['index','z']-cut_region['index','dx']/2.).min().in_units('kpc').v]
        cr_redge = [(cut_region['index','x']+cut_region['index','dx']/2.).max().in_units('kpc').v,
                              (cut_region['index','y']+cut_region['index','dx']/2.).max().in_units('kpc').v,
                              (cut_region['index','z']+cut_region['index','dx']/2.).max().in_units('kpc').v]

        target_dx = np.min(refine_box['index','dx'].in_units('kpc').v) * 2**(max_refinement_level - target_refinement_level)


        target_cut_region_shape = np.round( np.array([cr_redge[0] - cr_ledge[0], cr_redge[1] - cr_ledge[1], cr_redge[2] - cr_ledge[2]]) / target_dx).astype(int)

        new_disk_mask_ucg = np.zeros(target_cut_region_shape).astype(bool)

        x0 = int( (rb_ledge[0] - cr_ledge[0]) / target_dx )
        x1 = int(x0 + disk_mask_ucg.shape[0])
        y0 = int( (rb_ledge[1] - cr_ledge[1]) / target_dx )
        y1 = int(y0 + disk_mask_ucg.shape[1])
        z0 = int( (rb_ledge[2] - cr_ledge[2]) / target_dx )
        z1 = int(z0 + disk_mask_ucg.shape[2])

        new_disk_mask_ucg[x0:x1,y0:y1,z0:z1] = disk_mask_ucg
        
        return new_disk_mask_ucg

    return disk_mask_ucg