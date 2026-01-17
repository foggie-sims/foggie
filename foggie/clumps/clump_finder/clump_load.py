import numpy as np
from foggie.utils.foggie_load import foggie_load
from foggie.utils.consistency import *
from yt.data_objects.level_sets.api import * #TODO - Not super slow here, but replace with custom clump finding to find disk?

from scipy import stats
import unyt

def get_clump_bbox(clump):
        '''
        Get the edges of a cut region
        '''
        min_x = clump[('gas','x')].min().in_units('kpc')
        max_x = clump[('gas','x')].max().in_units('kpc')

        min_y = clump[('gas','y')].min().in_units('kpc')
        max_y = clump[('gas','y')].max().in_units('kpc')

        min_z = clump[('gas','z')].min().in_units('kpc')
        max_z = clump[('gas','z')].max().in_units('kpc')

        left_edge = [min_x,min_y,min_z]
        right_edge = [max_x,max_y,max_z]

        return left_edge,right_edge
    


def create_simple_ucg(ds, data_source, fields, target_nref, split_method=["copy"], merge_method=["max"], weight_field=None):
    '''
    Function to convert a list of fields into a uniform covering grid at a refinement level given by target_nref
    Arguments are:
        ds: the yt dataset
        data_source: the cut_region of interest
        fields: A list of the target fields you want in a ucg. e.g. [('gas','density'),('gas','cell_ids_2')]
        target_nref: The target refinement level
        split_method: list of how you want coarser cells for each field to be split ('copy' or 'halve')
        merge_method: list of how you want finer cells to be merged. Any method that works with scipy.stats.binned_statistic will work
                       e.g. 'max','min','mean' etc.

    '''
    data_source_left_edge  = [(data_source['index','x']-data_source['index','dx']/2.).min(),
                              (data_source['index','y']-data_source['index','dx']/2.).min(),
                              (data_source['index','z']-data_source['index','dx']/2.).min()]
    data_source_right_edge = [(data_source['index','x']+data_source['index','dx']/2.).max(),
                              (data_source['index','y']+data_source['index','dx']/2.).max(),
                              (data_source['index','z']+data_source['index','dx']/2.).max()]

    dx_uniform = ds.all_data()['index','dx'].max().in_units('code_length') / float(2**target_nref)

    print(data_source)
    print("dx_uniform=",dx_uniform)
    print("data_source_left_edge=",data_source_left_edge)
    print("data_source_right_edge=",data_source_right_edge)

    ucg_nx = int(np.round((data_source_right_edge[0] - data_source_left_edge[0]) / dx_uniform))
    ucg_ny = int(np.round((data_source_right_edge[1] - data_source_left_edge[1]) / dx_uniform))
    ucg_nz = int(np.round((data_source_right_edge[2] - data_source_left_edge[2]) / dx_uniform))


    
    
    ucg_list = []
    nFields = len(fields)
    for i in range(0,nFields):
        if fields[i] != ('index','cell_id_2'):
            ucg_list.append(np.zeros((ucg_nx,ucg_ny,ucg_nz)))
            ucg_list[i][:] = np.nan
        else:
            ucg_list.append(np.zeros((ucg_nx,ucg_ny,ucg_nz)).astype('uint64'))


    for itr_nref in [1,2,3,4,5,6,7,8,9,10,11,12,13,14]: 
      for g,m in data_source.blocks: #grid and mask in the refine box for each grid amr refinement grid
        #print("Mask=",m)
       # print("g.id",g.id,"...",g.keys())
        g_nref = g['index','grid_level'][0,0,0]
        if (g_nref != itr_nref): continue #do in order from lowest to highest refinement, such that the highest refinement is always available
        g_dx = g['index','dx'][0,0,0]
        new_grid_list = []
        weight_grid_list = []
        if weight_field is not None:
            weight_grid_list.append(g[weight_field])
        for field in fields:
            new_grid_list.append(g[field])
            try:
                #Gas fields will be in grids
                nx,ny,nz = np.shape(g[field])
            except:
                #Particle fields will be in flat arrays
                nx = int(g['index','x'].max()/g_dx - g['index','x'].min()/g_dx +1)
                ny = int(g['index','y'].max()/g_dx - g['index','y'].min()/g_dx +1)
                nz = int(g['index','z'].max()/g_dx - g['index','z'].min()/g_dx +1)
                temp_grid = np.zeros((nx,ny,nz)) * unyt.g / unyt.kpc**3
                #temp_grid[:]=np.nan
                field_position_x =  field[0],'particle_position_x'
                field_position_y =  field[0],'particle_position_y'
                field_position_z =  field[0],'particle_position_z'
                x_indices = ((g[field_position_x] - g['index','x'].min()) / g_dx)#.astype('int')
                y_indices = ((g[field_position_y] - g['index','y'].min()) / g_dx)#.astype('int')
                z_indices = ((g[field_position_z] - g['index','z'].min()) / g_dx)#.astype('int')

                ix_min = np.floor(x_indices).astype(int)
                iy_min = np.floor(y_indices).astype(int)
                iz_min = np.floor(z_indices).astype(int)

                ix_max = np.ceil(x_indices).astype(int)
                iy_max = np.ceil(y_indices).astype(int)
                iz_max = np.ceil(z_indices).astype(int)

                cic_kernel=np.ones((2,2,2))
                kernel_volume = g_dx*g_dx*g_dx*np.size(cic_kernel)
                for j in range(0,np.size(g[field_position_x])):
                    ###REPLACE THIS SO IT DOESN'T OVERRIDE PREVIOUS PARTICLES (i.e. add)
                    cic_kernel=np.ones((2,2,2))

                    cic_kernel[0,:,:] *= (1 - (x_indices[j] - ix_min[j]))
                    cic_kernel[1,:,:] *= (x_indices[j] - ix_min[j])
                    cic_kernel[:,0,:] *= (1 - (y_indices[j] - iy_min[j]))
                    cic_kernel[:,1,:] *= (y_indices[j] - iy_min[j])
                    cic_kernel[:,:,0] *= (1 - (z_indices[j] - iz_min[j]))
                    cic_kernel[:,:,1] *= (z_indices[j] - iz_min[j])

                    cx_min = 0;cy_min=0;cz_min=0
                    cx_max = 2;cy_max=2;cz_max=2

                    if ix_min[j]<0:
                        ix_min[j]=0
                        cx_min = 1
                    if ix_max[j]>=nx:
                        ix_max[j]=nx-1
                        cx_max = 1
                    if iy_min[j]<0:
                        iy_min[j]=0
                        cy_min = 1
                    if iy_max[j]>=ny:
                        iy_max[j]=ny-1
                        cy_max = 1
                    if iz_min[j]<0:
                        iz_min[j]=0
                        cz_min = 1
                    if iz_max[j]>=nz:
                        iz_max[j]=nz-1
                        cz_max = 1

                    cic_kernel = cic_kernel[cx_min:cx_max,cy_min:cy_max,cz_min:cz_max]
                    cic_kernel = cic_kernel / np.sum(cic_kernel) #Normalize kernel
                    kernel_volume = g_dx*g_dx*g_dx*np.size(cic_kernel)
                    to_add = g[field][j] * cic_kernel / kernel_volume 
                    to_add[np.isnan(to_add)] = 0.0
                    temp_grid[ix_min[j]:ix_max[j]+1, iy_min[j]:iy_max[j]+1, iz_min[j]:iz_max[j]+1] += to_add  #Divide by cell volume to get density-like quantity
                    #temp_grid[x_indices[j],y_indices[j],z_indices[j]] = g[field][j] / g_dx / g_dx / g_dx #Divide by cell volume to get density-like quantity
                if np.isnan(temp_grid).any(): print("Warning, NaN in temp grid?")
                new_grid_list[-1] = temp_grid

        nx,ny,nz = np.shape(new_grid_list[0])


        
        if g_nref > target_nref: #Need to merge cells -> Not good generally
            nrep = int(np.round(2**(g_nref - target_nref)))
            new_nx = nx/nrep
            new_ny = ny/nrep
            new_nz = nz/nrep
            sample = np.zeros((np.size(g['index','x']) , 3))
            sample[:,0] = g['index','x'].flatten()
            sample[:,1] = g['index','y'].flatten()
            sample[:,2] = g['index','z'].flatten()
            for i in range(0,nFields):
                old_shape = np.shape(new_grid_list[i])
                if merge_method[i] == "weighted_mean":
                    new_grid_list[i],tmp,tmp = stats.binned_statistic_dd(sample,np.multiply(new_grid_list[i],weight_grid_list[i]).flatten(),'sum',bins=[new_nx,new_ny,new_nz])
                    summed_weight,tmp,tmp = stats.binned_statistic_dd(sample,weight_grid_list[i].flatten(),'sum',bins=[new_nx,new_ny,new_nz])
                    new_grid_list[i] = np.divide(new_grid_list[i],summed_weight, out=np.zeros_like(new_grid_list[i]), where=summed_weight!=0)
                else:
                    new_grid_list[i],tmp,tmp = stats.binned_statistic_dd(sample,new_grid_list[i].flatten(),statistic=merge_method[i],bins=[new_nx,new_ny,new_nz])
            #print("Shape of new grid set to be",np.shape(new_grid))
        
        if g_nref < target_nref: #Need to split cells
            nrep = int(np.round(2**(target_nref - g_nref)))
            
            for i in range(0,nFields):
                new_grid_list[i] = np.repeat(new_grid_list[i] , nrep, axis=0)
                new_grid_list[i] = np.repeat(new_grid_list[i] , nrep, axis=1)
                new_grid_list[i] = np.repeat(new_grid_list[i] , nrep, axis=2)

                if split_method[i] == "halve": new_grid_list[i] = new_grid_list[i] / float(nrep)
            
        
        xmin = int(np.round( ( g['index','x'].min() - g_dx/2 - data_source_left_edge[0] ) / dx_uniform))
        xmax = int(np.round( ( g['index','x'].max() + g_dx/2 - data_source_left_edge[0] ) / dx_uniform))

        ymin = int(np.round( ( g['index','y'].min() - g_dx/2 - data_source_left_edge[1] ) / dx_uniform))
        ymax = int(np.round( ( g['index','y'].max() + g_dx/2 - data_source_left_edge[1] ) / dx_uniform))
            
        zmin = int(np.round( ( g['index','z'].min() - g_dx/2 - data_source_left_edge[2] ) / dx_uniform))
        zmax = int(np.round( ( g['index','z'].max() + g_dx/2 - data_source_left_edge[2] ) / dx_uniform))
        
        


        #print("Grid",g.id,"spans [",xmin,xmax,",",ymin,ymax,",",zmin,zmax,"]")
        ng_xmin=0;ng_ymin=0;ng_zmin=0;
        ng_xmax,ng_ymax,ng_zmax = np.shape(new_grid_list[0])
        #ng_xmax-=1;ng_ymax-=1;ng_zmax-=1
        #print(ng_xmax,ng_ymax,ng_zmax)
        
        if xmin>ucg_nx or xmax<0:print("SKIPPING BLOCK"); continue
        if ymin>ucg_ny or ymax<0:print("SKIPPING BLOCK"); continue
        if zmin>ucg_nz or zmax<0:print("SKIPPING BLOCK"); continue

        
        if xmin<0:
            ng_xmin=-xmin
           # print("xmin=",xmin,"so ng_xmin",ng_xmin+xmin,"->",ng_xmin)
            xmin=0
        if xmax>ucg_nx:
            ng_xmax=ng_xmax - (xmax - (ucg_nx))
            #print("xmax=",xmax,"so ng_xmax->",ng_xmax)
            xmax=ucg_nx
        if ymin<0:
            ng_ymin=-ymin
            ymin=0
        if ymax>ucg_ny:
            ng_ymax=ng_ymax - (ymax - (ucg_ny))
            ymax=ucg_ny
        if zmin<0:
            ng_zmin=-zmin
            zmin=0
        if zmax>ucg_nz:
            ng_zmax=ng_zmax - (zmax - (ucg_nz))
            zmax=ucg_nz

        for i in range(0,nFields):
            ucg_list[i][xmin:xmax, ymin:ymax, zmin:zmax] = new_grid_list[i][ng_xmin:ng_xmax , ng_ymin:ng_ymax, ng_zmin:ng_zmax]

    print("ucg_list:",np.min(ucg_list[0]),np.max(ucg_list[0]))
    for i in range(0,nFields):
        ucg_list[i][np.isnan(ucg_list[i])] = 0.0
        if fields[i][0] == 'dm':
            import scipy
            ucg_list[i] = scipy.ndimage.gaussian_filter(ucg_list[i], sigma=4) #smooth from 11 to 9
    return ucg_list
        



