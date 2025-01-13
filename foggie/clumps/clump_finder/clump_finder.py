import numpy as np
from clump_finder_argparser import parse_args
from clump_load import create_simple_ucg

from foggie.utils.foggie_load import foggie_load

from foggie.utils.consistency import *

from joblib import Parallel, delayed
import multiprocessing
from functools import partial
import h5py

import time

from collections import defaultdict

from utils_diskproject import get_cgm_density_cut
from fill_topology import fill_voids
from fill_topology import fill_holes

import matplotlib.pyplot as plt
#from matplotlib.colors import LogNorm

from scipy.ndimage import label   
from scipy.ndimage import find_objects     

import merge_clumps

'''
    Finds 3-d clumps in simulation data along an arbitrary clumping field. This code was designed to work with the FOGGIE simulations
    through yt.
    
    The algorithm loads in the specified clumping field (typically density) in the forced refinement box (refine_box) of the given snapshot, and
    converts it into a uniform covering grid at the given refinement level. From here, a marching cubes alogrithm is run on the UCG to label
    contiguous regions above the minimum clumping threshold. If --run_mc_parallel is set to true, the UCG is first subdivided into a number of
    subarrays specified by --nSubarrays, and each thread runs the algorithm on a subarray and stitches them back together (this induces an
    overhead that is likely not worth it in most cases though.)
        
    The algorithm then generates a list of unique cell-ids for each clump, such that they can be re-loaded as cut regions for future analysis.
    This step is parallelized along the number of subarrays defined by -nSubarrays.
    
    After this is done at a the first clumping threshold (--clump_min), the threshold is multiplied by --step and the algorithm is re-ran.
    
    During these sucessive iterations, the hierarchy of the clumps is calculated and stored in the main_clump class object. Parents of clumps
    are identified as clumps 1 level up that have an overlap in cell_ids.


    As an alternative mode of use, setting the --identifty_disk flag will run the clump finder as a disk finder instead. The disk is identified
    as the largest clump above a certain density threshold. Depending on the values assigned to --max_disk_void_size and --max_disk_hole_size, 
    3-D topologically enclosed voids are filled in this disk mask, as well as 2-D topologically enclosed holes along the disk axis.
    
    The args are parsed as follows:
    
    IO Arguments:
    --code_dir: Where is the foggie analysis directory?
    --data_dir: Where are the simulation outputs?
    --refinement_level: To which refinement_level should the uniform covering grid be made. Defaults to the maximum refinement level in the box.
    --halo: Which halo should be analyzed. Default is 008508 (Tempest)
    --snapshot: Which snapshot should be analyzed? Default is RD0042
    --run: What refinement run should be analyzed? Default is nref11c_nref9f  

    --output: Where should the clump data be written? Default is ./output/clump_test
    --only_save_leaves: Set to True to only save the leaf clumps. Default is False.

    
    Algorithm Arguments:
    --clumping_field: What field are you clumping on? Default is 'density'
    --clumping_field_type: What field type are you clumping on? Default is 'gas' (i.e. this and the previous argument give you ('gas','density')).

    --clump_min: What should the starting density cutoff be? Default is defined as cgm_density_cutoff in foggie.utils.consistency
    --clump_max: What should the final density cutoff be? Default is the maximum density in the simulation.
    --step: By what factor should the density cutoff be incremented during each step? Default is 2

    --min_cells: What is the minimum cell count (on the uniform covering grid) to define as a "clump"

    --include_diagonal_neighbors: Include cells that neighbor on the diagonal during marching cubes. Default is False.
    --mask_disk: Should the disk be masked out? Default is False. Not needed any more, but may offer performance upgrades
    --max_void_size: What is the maximum size of voids (in number of cells) to fill. Set to above 0 to fill voids in clump. Default is 0.


    Parallelization Arguments:
    --nthreads: How many threads to run on? Defaults to number of cores - 1
    --Nsubarrays: How many subarrays should the UCG be split into during parallelization. Default is 64. Should be set to the smallest perfect cube that is larger than nthreads.

    --run_mc_parallel: Do you want to run the marching cubes algorithm in parallel? Incurs additional computational overhead, but may be faster with a large (>30) number of cores or for large datacubes. Default is False.
    --run_mapping_linearly: Do you want to run the clump to cell id mapping without parallelization? Should generally be slower, but can be done while filling voids and holes. Default is False.

    
    Disk Identification Arguments:
    --identify_disk: Run the clump finder as a disk finder instead.
    --cgm_density_cut_type: When identifying the disk how do you want to define the CGM density cut? Options are ["comoving_density,"relative_density","cassis_cut"]. Default is "relative_density".')
    --cgm_density_factor: When identifying the disk, what factor should the cgm_density_cut use. Default is 200 for relative density, 0.2 for comoving density, and 1 for cassis_cut.
    --max_disk_void_size: What is the maximum size of 3D voids (in number of cells) to fill in the disk. Set to above 0 to fill voids. Default is 2000.
    --mask_disk_hole_size: What is the maximum size of 2D holes (in number of cells) to fill in the disk. Set to above 0 to fill holes. Default is 2000.
    
    --make_disk_mask_figures: Do you want to make additional figures illustrating the void/hole filling process when defining the disk? Default is False.

    
'''
    

time_marching_cubes = 0
t0 = time.time()

class TqdmProgressBar:
    '''
    Basic display bar for progress
    '''
    def __init__(self, title, maxval,position):
        from tqdm import tqdm

        self._pbar = tqdm(leave=True, total=maxval, desc=title, position=position)
        self.i = 0

    def update(self, i=None):
        if i is None:
            i = self.i + 1
        n = i - self.i
        self.i = i
        self._pbar.update(n)

    def finish(self):
        self._pbar.close()
        
        


class ClumpCriteria:
    '''
    Criteria for what is classified as a clump. Currently just based on number of cells.
    '''
    def __init__(self,args):
        self.min_cells = args.min_cells

####Define clump class
class Clump:
    '''
    The clump class. The root clump (tree level 0) contains additional information. All sub-clumps contain information about their parents and children.
    '''
    def __init__(self,ds,cut_region,args,tree_level,is_disk=False):
        self.parent=None
        self.Nsubarrays=None
        self.Nslices=None
        self.clump_ids=None
        self.clump_merge_map=None
        self.max_refinement_level=None
        self.ucg_shape = None
        self.ucg = None
        self.nChildren = 0

        self.self_id = -1
        self.self_index = -1
        self.parent_id = -1
        self.parent_index = -1
        self.child_ids = []
        self.child_indices = []


        self.cell_ids=None #1d list of unique cell ids for each clump. Should not be defined on the master clump object, only children
        


        self.tree_level=tree_level
        
        if self.tree_level==0:
            self.n_levels = np.ceil( np.log(args.clump_max / args.clump_min) / np.log(args.step) + 1 ).astype(int)
            print("n_levels=",self.n_levels)
            self.clump_tree=[]
            for i in range(0,self.n_levels):
                self.clump_tree.append([])
        
        
            self.Set_Nsubarrays(args)
            self.clump_criteria = ClumpCriteria(args)
            self.ds = ds
            self.cut_region = cut_region
            self.clump_min = args.clump_min
            self.clump_max = args.clump_max
            self.step = args.step

        self.is_disk = is_disk
        if is_disk:
            self.x_disk_ucg=None
            self.y_disk_ucg=None
            self.z_disk_ucg=None
            self.ucg_dx = ds.all_data()['index','dx'].max().in_units('kpc') / float(2**args.refinement_level)

            


    def Set_Nsubarrays(self,args):
        '''
        Get the requested number of subarrays and calculate the correspoding number of slices.
        '''
        if args.Nsubarrays is None:
            self.Nsubarrays = args.nthreads
            args.Nsubarrays = args.nthreads
        else:
            self.Nsubarrays = args.Nsubarrays
            
        if args.Nsubarrays < args.nthreads:
            print("Warning: There are less subarrays (",args.Nsubarrays,") then threads (",args.nthreads,")...")
        
        
        Nsub = self.Nsubarrays
        if Nsub==1:
            self.Nslices = 0
        elif (int(np.round(Nsub**(1./3.)))**3 == int(Nsub)): #ideal case, perfect cube, if Nsub in [8,27,64,125,...]:
            self.Nslices = 3*(int(np.round(Nsub**(1./3.)))-1)
        elif (int(Nsub**(1./2.))**2 == int(Nsub)): #perfect square, if Nsub in [4,9,16,25,36,49,64,81,100,...]:
            self.Nslices = 2*(int(Nsub**(1./2.))-1)
        elif (int((Nsub/2.)**(1./2.)**2 == int(Nsub/2.))): #twice a perfect square, in [18,32,50,72,98,128,162,200...]:
            self.Nslices = 2*(int(Nsub**(1./2.))-1) + 1
        else:
            self.Nslices = Nsub - 1 #bad
        
    def ReshapeSubarrays(self):
        '''
        Transform the subarrays back to the full UCG datacube.
        '''
        #map from nSubarrays*nSubBox*nSubBox*nSubBox to nBox*nBox*nBox
        new_ucg = np.zeros((self.ucg_shape))
        nx,ny,nz = self.ucg_shape
        for i in range(0,self.Nsubarrays):
            xrange,yrange,zrange=get_indices(i,self.Nsubarrays,nx,ny,nz)
            new_ucg[xrange[0]:xrange[1], yrange[0]:yrange[1], zrange[0]:zrange[1]] = self.clump_ids[i][:,:,:]
        self.clump_ids = new_ucg.astype(int)



    def UpdateClumpCatalog_linear(self,args):
        '''
        Linear implementation of mapping from the UCG back to the native simulation suite. Gets the unique cell_ids associated for each clump.
        '''
        unique_clumps = np.unique(self.clump_ids[self.clump_ids>0])
 
        itr=-1
       
        nClumpsAdded=0

        slices = find_objects(self.clump_ids)

        if self.is_disk:
            current_max = 0
            disk_label = 0
            for label, slice_obj in enumerate(slices,start=1):
                if slice_obj is not None:
                    clump_id = self.clump_ids[slice_obj]
                    mask = (clump_id == label)
                    if np.size(np.where(mask))>current_max:
                        current_max = np.size(self.clump_ids[slice_obj])
                        disk_label = label
            print("disk_label set to:",disk_label)

        if not self.is_disk: pbar = TqdmProgressBar("Adding Children",np.size(unique_clumps),position=0)
        else: print("Cataloging and filling in disk...")

        for label, slice_obj in enumerate(slices,start=1):
          if slice_obj is not None:
            clump_id = self.clump_ids[slice_obj]
            id_region = self.cell_id_ucg[slice_obj]
            mask = (clump_id == label)


            if self.is_disk:
                #try:
                if label==disk_label:
                    n0 = np.size(np.where(mask))
                    if args.max_disk_void_size>0:
                        if args.make_disk_mask_figures:
                            plt.figure()
                            plt.imshow(np.sum(mask,axis=2).astype(bool).astype(int))
                            plt.xticks([])
                            plt.yticks([])
                            plt.savefig(args.output + "_disk_mask_faceon_0.png")
                        mask = fill_voids(mask,args.max_disk_void_size,structure=None)
                        n1 = np.size(np.where(mask))
                        print("Void filling filled",n1-n0,"cells in 3d cavities.")
                    if args.max_disk_hole_size>0:
                        if args.make_disk_mask_figures:
                            plt.figure()
                            plt.imshow(np.sum(mask,axis=2).astype(bool).astype(int))
                            plt.xticks([])
                            plt.yticks([])
                            plt.savefig(args.output + "_disk_mask_faceon_1.png")
                        nx = int(np.round((np.max(self.x_disk_ucg[slice_obj]) - np.min(self.x_disk_ucg[slice_obj])) / self.ucg_dx))
                        mask = fill_holes(ds.x_unit_disk, mask, args.max_disk_hole_size, nx, structure=None)
                        n2 = np.size(np.where(mask))
                        print("Hole filling along x-hat filled",n2-n1,"cells in 2d holes.")
                    if args.max_disk_hole_size>0:
                        if args.make_disk_mask_figures:
                            plt.figure()
                            plt.imshow(np.sum(mask,axis=2).astype(bool).astype(int))
                            plt.xticks([])
                            plt.yticks([])
                            plt.savefig(args.output + "_disk_mask_faceon_2.png")
                        ny = int(np.round((np.max(self.y_disk_ucg[slice_obj]) - np.min(self.y_disk_ucg[slice_obj])) / self.ucg_dx))
                        mask = fill_holes(ds.y_unit_disk, mask, args.max_disk_hole_size, ny, structure=None)
                        n3 = np.size(np.where(mask))
                        print("Hole filling along y-hat filled",n3-n2,"cells in 2d holes.")
                        if args.make_disk_mask_figures:
                            plt.figure()
                            plt.imshow(np.sum(mask,axis=2).astype(bool).astype(int))
                            plt.xticks([])
                            plt.yticks([])
                            plt.savefig(args.output + "_disk_mask_faceon_3.png")
                        nz = int(np.round((np.max(self.z_disk_ucg[slice_obj]) - np.min(self.z_disk_ucg[slice_obj])) / self.ucg_dx))
                        mask = fill_holes(ds.z_unit_disk, mask, args.max_disk_hole_size, nz, structure=None)
                        n4 = np.size(np.where(mask))
                        print("Hole filling along z-hat filled",n4-n3,"cells in 2d holes.")
                        if args.make_disk_mask_figures:
                            plt.figure()
                            plt.imshow(np.sum(mask,axis=2).astype(bool).astype(int))
                            plt.xticks([])
                            plt.yticks([])
                            plt.savefig(args.output + "_disk_mask_faceon_4.png")

            elif args.max_void_size>0:
                mask = fill_voids(mask,args.max_void_size,structure=None)

            clump_cell_ids = id_region[mask]

            itr+=1
            if itr%1==0 and not self.is_disk: pbar.update(itr)
   
            if self.CheckClumpCriteria(clump_cell_ids):
                nClumpsAdded+=1
                clump_cell_ids = np.unique(clump_cell_ids)


                nClumpsAdded+=1   

                self.clump_tree[self.tree_level].append(Clump(self.ds,None,args,self.tree_level+1))

                clump_index = len(self.clump_tree[self.tree_level])-1
                self.clump_tree[self.tree_level][-1].cell_ids = clump_cell_ids
                self.clump_tree[self.tree_level][-1].self_id = clump_id
                self.clump_tree[self.tree_level][-1].self_index = clump_index

                if self.tree_level>0:
                    parent_clump = self.FindParent(clump_cell_ids)
                    parent_clump.nChildren+=1
                    parent_clump.child_ids.append(clump_id)
                    self.clump_tree[self.tree_level][-1].parent_id = parent_clump.self_id
                    
                    parent_clump.child_indices.append(clump_index)
                    self.clump_tree[self.tree_level][-1].parent_index = parent_clump.self_index


        if not self.is_disk:
            pbar.update(np.size(unique_clumps))
            pbar.finish()
        return nClumpsAdded

    def UpdateClumpCatalog_parallel(self,args):
        '''
        Parallel implementation of mapping from the UCG back to the native simulation suite. Gets the associated cell_ids for each clump.
        '''
        t0=time.time()

        unique_clumps = np.unique(self.clump_ids[self.clump_ids>0])
 
        itr=-1
        nClumpsAdded=0

        print("Finding parallel clump_id_sublists...")
        clump_id_subarrays = self.split_ucg(self.clump_ids) #list of subarrays
        cell_id_subarrays = self.split_ucg(self.cell_id_ucg)
        _iterate_get_clump_cell_ids = partial(iterate_get_clump_cell_ids, args=args)
        if args.nthreads > self.Nsubarrays:
            n_jobs = self.Nsubarrays
        else:
            n_jobs = args.nthreads

        print("Working with",n_jobs,"threads...")
        clump_cell_ids_sublists = Parallel(n_jobs=n_jobs)(delayed(_iterate_get_clump_cell_ids)(clump_id_subarray=clump_id_subarrays[i],cell_id_subarray=cell_id_subarrays[i],thread_id=i) for i in range(0,self.Nsubarrays)) #Nsubarray * Nbox * Nbox * Nbox array
        
        print("Combining clump id sublists...")
        clump_cell_ids = defaultdict(list)
        for sublist in clump_cell_ids_sublists:
            for key in sublist:
                clump_cell_ids[key].extend(sublist[key])


        print("Time for parallel clump mapping=",time.time()-t0)
        
        pbar = TqdmProgressBar("Adding Children",np.size(unique_clumps),position=0)
        itr=-1
        for clump_id in unique_clumps: 
            itr+=1
            if itr%1==0: pbar.update(itr)
            clump_cell_ids[int(clump_id)] = np.unique(clump_cell_ids[int(clump_id)])
            if self.CheckClumpCriteria(clump_cell_ids[int(clump_id)]):
                nClumpsAdded+=1

                self.clump_tree[self.tree_level].append(Clump(self.ds,None,args,self.tree_level+1))

                clump_index = len(self.clump_tree[self.tree_level])-1
                self.clump_tree[self.tree_level][-1].cell_ids = clump_cell_ids[int(clump_id)]
                self.clump_tree[self.tree_level][-1].self_id = clump_id
                self.clump_tree[self.tree_level][-1].self_index = clump_index

                if self.tree_level>0:
                    parent_clump = self.FindParent(clump_cell_ids[int(clump_id)])
                    parent_clump.nChildren+=1
                    parent_clump.child_ids.append(clump_id)
                    self.clump_tree[self.tree_level][-1].parent_id = parent_clump.self_id
                    
                    parent_clump.child_indices.append(clump_index)
                    self.clump_tree[self.tree_level][-1].parent_index = parent_clump.self_index



        pbar.update(np.size(unique_clumps))
        pbar.finish()


        return nClumpsAdded
        
    def FindParent(self,clump_cell_ids):
        '''
        Search all clumps at the higher tree level for one that shares the first cell_id to identify a clumps parent.
        '''
        for uncle in self.clump_tree[self.tree_level-1]:
            if np.isin(clump_cell_ids[0] , uncle.cell_ids):
                return uncle#,np.size(clump_cell_ids)>np.size(uncle.cell_ids)

        print("\n\nWARNING: NO PARENT FOUND! SETTING RANDOM PARENT!")
        print(clump_cell_ids[0],"\n\n")

 
        return self.clump_tree[self.tree_level-1][0]
            
    
        
    def CheckClumpCriteria(self,clump_cell_ids,ncells=None):
        '''
        Implement clump criteria to cull clumps. (i.e. get rid of clumps that are too small).
        '''
        if self.clump_criteria.min_cells>0:
            if ncells is not None:
                 if ncells < self.clump_criteria.min_cells: return False
            elif np.size(clump_cell_ids) < self.clump_criteria.min_cells: return False
             
        #To do add more criteria
        return True
            
    def split_ucg(self,ucg):
        '''
        Split the full UCG datacube into subarrays for parallelization
        '''
        nx,ny,nz = self.ucg_shape
        new_ucg = [] #list of arrays
        for i in range(0,self.Nsubarrays):
            xrange,yrange,zrange=get_indices(i,self.Nsubarrays,nx,ny,nz)

            new_ucg.append(ucg[xrange[0]:xrange[1] , yrange[0]:yrange[1] , zrange[0]:zrange[1]])
        return new_ucg
            
    def DefineUniqueClumpIds(self):
        '''
        Make sure the clump ids from the parallel marching cubes algorithm are unique. Only used if args.run_mc_parallel=True.
        '''
        #determine unique integer values for each clump id accross the various subarrays
        max_id = 0
        for i in range(0,len(self.clump_ids)):
            tmp_max = np.max(self.clump_ids[i])
            if tmp_max > max_id: max_id = tmp_max

        for i in range(0,len(self.clump_ids)):
            self.clump_ids[i][self.clump_ids[i]>0] = self.clump_ids[i][self.clump_ids[i]>0] + i*(max_id+1)

    def get_boundary_slices(self):
        '''
        Get the clump ids that are on the boundary between subarrays to stitch them back together. Only used if args.run_mc_parallel=True.
        '''
        nthreads = self.Nsubarrays
        
        nx,ny,nz = self.ucg_shape
        
        boundary_slices = []

        if nthreads==1:
            return None
        elif (int(np.round(nthreads**(1./3.)))**3 == int(nthreads)): #ideal case, perfect cube, if nthreads in [8,27,64,125,...]:
            print("Nslices=",self.Nslices)
            NslicePerDim = int(round(self.Nslices / 3.))
            print("NslicesPerDim=",NslicePerDim)
            for i in range(0,NslicePerDim):
                nc = int(round(nx / (NslicePerDim + 1) * (i+1)))-1
                boundary_slices.append(self.clump_ids[nc:nc+2,:,:])
            for i in range(0,NslicePerDim):
                nc =  int(round(ny / (NslicePerDim + 1) * (i+1)))-1
                boundary_slices.append(self.clump_ids[:,nc:nc+2,:])
            for i in range(0,NslicePerDim):
                nc =  int(round(nz / (NslicePerDim + 1) * (i+1)))-1
                boundary_slices.append(self.clump_ids[:,:,nc:nc+2])                
        elif (int(nthreads**(1./2.))**2 == int(nthreads)): #perfect square, if nthreads in [4,9,16,25,36,49,64,81,100,...]:
            NslicePerDim = int(round(self.Nslices / 2))
            print("NslicePerDim=",NslicePerDim)
            for i in range(0,NslicePerDim):
                nc =  int(round(nx / (NslicePerDim + 1) * (i+1)))-1
                print("Placing slice at nc=",nc,"-",nc+1,"for nx=",np.shape(self.clump_ids)[0])
                boundary_slices.append(self.clump_ids[nc:nc+2,:,:])
            for i in range(0,NslicePerDim):
                nc =  int(round(ny / (NslicePerDim + 1) * (i+1)))-1
                print("Placing slice at nc=",nc,"-",nc+1,"for ny=",np.shape(self.clump_ids)[1])
                boundary_slices.append(self.clump_ids[:,nc:nc+2,:])
        elif (int((nthreads/2.)**(1./2.)**2 == int(nthreads/2.))): #twice a perfect square, in [18,32,50,72,98,128,162,200...]:
            NslicePerDim = int(round(self.Nslices / 2))
            for i in range(0,NslicePerDim):
                nc =  int(round(nx / (NslicePerDim + 1) * (i+1)))-1
                boundary_slices.append(self.clump_ids[nc:nc+2,:,:])
            for i in range(0,NslicePerDim):
                nc =  int(round(ny / (NslicePerDim + 1) * (i+1)))-1
                boundary_slices.append(self.clump_ids[:,nc:nc+2,:])
            #bonus slice
            nc = int(round(nz/2))-1
            boundary_slices.append(self.clump_ids[:,:,nc:nc+2])
        else:
            for i in range(0,self.Nslices):
                nc = int(round(nx / (self.Nslices + 1) * (i+1)))-1
                boundary_slices.append(self.clump_ids[nc:nc+2,:,:])     

        return boundary_slices   
            
    def FindClumps(self,clump_threshold,args, ids_to_ignore=None):
        '''
        Main function for running the clump finder. Iterates from clump_threshold to the maximum clump density by a factor of args.step. 
        Each iteration identifies contiguous region above the current_threshold and defines a separate clump object. The root clump
        stores the hierarchy information.
        '''
        if self.is_disk:
            print("Defining disk ucgs...")
            fields = [args.clumping_field,cell_id_field,('gas','z_disk'),('gas','y_disk'),('gas','x_disk')]
            split_methods = ["copy","copy","copy","copy","copy"]
            merge_methods = ["max" ,"max","mean","mean","mean"]
            self.ucg, self.cell_id_ucg, self.z_disk_ucg, self.x_disk_ucg, self.y_disk_ucg = create_simple_ucg(self.ds, self.cut_region, fields, args.refinement_level,split_methods,merge_methods) #parallelize? Double check overlaps and which edges are being set to nans
            self.ucg_shape = np.shape(self.ucg)
        else:
            print("Defining ucgs...")
            fields = [args.clumping_field,cell_id_field]
            split_methods = ["copy","copy"]
            merge_methods = ["max" ,"max" ]
            self.ucg, self.cell_id_ucg= create_simple_ucg(self.ds, self.cut_region, fields, args.refinement_level,split_methods,merge_methods) #parallelize? Double check overlaps and which edges are being set to nans
            self.ucg_shape = np.shape(self.ucg)


        if ids_to_ignore is not None:
             self.ucg[ np.isin(self.cell_id_ucg , ids_to_ignore) ] = 0 #mask out these ids

        self.ucg[np.isnan(self.ucg)]=0

        if (np.max(self.ucg) < clump_threshold):
            print("ALL CELLS ARE BELOW CURRENT DENSITY THRESHOLD...")
            return
            
        parallelize_marching_cubes=False
        if args.nthreads>1 and args.run_mc_parallel:
            parallelize_marching_cubes = True
            print("Splitting ucg...")
            split_clumping_ucg = self.split_ucg(self.ucg) #list of subarrays


        
        current_threshold = clump_threshold
        self.tree_level=0
        while current_threshold < args.clump_max:
            t1=time.time()
            print("Iterating for clump threshold=",current_threshold)

            if parallelize_marching_cubes:
                _iterate_clump_marching_cubes = partial(iterate_clump_marching_cubes,args=args,minval=current_threshold,maxval=None)

                print("Marching cubes...")
    
                if args.nthreads > self.Nsubarrays:
                    n_jobs = self.Nsubarrays
                else:
                    n_jobs = args.nthreads
                self.clump_ids = Parallel(n_jobs=n_jobs)(delayed(_iterate_clump_marching_cubes)(ucg_subarray=split_clumping_ucg[i],thread_id=i) for i in range(0,self.Nsubarrays)) #Nsubarray * Nbox * Nbox * Nbox array

                t2=time.time()
                print("Time to run marching cubes in parallel=",t2-t1)
                print("Defining unique clump ids...")
                self.DefineUniqueClumpIds()

                print("Reshaping subarrays...")
                self.ReshapeSubarrays()
              
                if args.nthreads > self.Nslices:
                    n_jobs = self.Nslices
                else:
                    n_jobs = args.nthreads
                
                print("Iterating boundaries...")
                _iterate_boundary_slice = partial(iterate_boundary_slice,args=args)
                boundary_slices = self.get_boundary_slices()

                self.clump_merge_map = Parallel(n_jobs=n_jobs)(delayed(_iterate_boundary_slice)(boundary_slice=boundary_slices[i],thread_id=i) for i in range(0,self.Nslices)) #Nsubarray * Nbox * Nbox * Nbox array
                print("Merging clumps...")
                if args.nthreads > self.Nsubarrays:
                    n_jobs = self.Nsubarrays
                else:
                    n_jobs = args.nthreads
                
                #print("Merge map is:",self.clump_merge_map)

                merging_ucg = self.split_ucg(self.clump_ids) #list of subarrays
                _iterate_merge_clumps = partial(iterate_merge_clumps,args=args,clump_merge_map=self.clump_merge_map)

                #print("Unique clumps before merging=",np.unique(self.clump_ids))
                self.clump_ids = Parallel(n_jobs=n_jobs)(delayed(_iterate_merge_clumps)(clump_id_subarray=merging_ucg[i],thread_id=i) for i in range(0,self.Nsubarrays)) #Nsubarray * Nbox * Nbox * Nbox array
                print("Reshaping merged clumps...")
                self.ReshapeSubarrays()
                print("Time to merge clumps=",time.time()-t2)
                #print("Unique clumps after merging=",np.unique(self.clump_ids))


            else:
                print("Marching cubes...")
                self.clump_ids, num_features = label((self.ucg>current_threshold))
                print("Time to march cubes linearly=",time.time()-t1)

            global time_marching_cubes
            time_marching_cubes += time.time()-t1
        
            print("Updating clump catalog...")
            parallelize_mapping = True
            if self.is_disk: parallelize_mapping = False
            if args.run_mapping_linearly: parallelize_mapping = False
            if args.max_void_size>0: parallelize_mapping = False
            if args.nthreads<=1: parallelize_mapping = False

            if parallelize_mapping:#np.max(self.clump_ids) > 300 or current_threshold > 1e-29:
                nClumpsAdded = self.UpdateClumpCatalog_parallel(args)
            else:
                nClumpsAdded = self.UpdateClumpCatalog_linear(args)

            if nClumpsAdded==0:
                print("No clumps found at this threshold...terminating")
                return
              
                    
            current_threshold *= args.step
            self.tree_level+=1
  
        
    def SaveClumps(self,args):
        '''
        Call to save all clumps in the hiearchy. If args.only_save_leaves is set to True will only save leaf clumps.
        '''
        print("Saving clumps...")
        for tree_level in range(0,len(self.clump_tree)):
            for clump in self.clump_tree[tree_level]:
                if clump.nChildren==0: #leaf clump
                    self.SaveClump(clump,args,leaf=True)
                elif not args.only_save_leaves:
                    self.SaveClump(clump,args)
                   
    def SaveClump(self,clump,args,leaf=False,output=None):
        '''
        Save an individual clump object
        '''
        if output is None:
            if leaf: output = args.output + "_Leaf_TL"+str(clump.tree_level)+"_C"+str(clump.self_index)+".h5"
            else: output = args.output + "_TL"+str(clump.tree_level)+"_C"+str(clump.self_index)+".h5"

        print("Saving clump at",output)

        hf = h5py.File(output,'w')
        hf.create_dataset("cell_ids", data = clump.cell_ids)

        if not args.only_save_leaves:
            parent_output=None
            if clump.parent_index>=0:
                parent_output = args.output+"_TL"+str(clump.tree_level-1)+"_C"+str(clump.parent_index)+".h5"
                hf.create_dataset("parent_output",data=parent_output)            
        
        hf.close()
        
        
    def SaveClumpLeaf(self,args):
        '''
        Save an individual clump leaf.
        '''
        hf = hdf5.File(args.output+"_leaf"+self.self_id[0]+"_"+self.self_id[1]+"_"+self.self_id[2]+".hdf5",'w')
        hf.create_dataset("cell_ids",data = self.cell_ids)
        hf.close()
        
    
    
#######Push the ucg of the clumping variable into a clump class
def get_indices(thread_id,nthreads,nx,ny,nz):
    '''
    Function to get indices on the UCG associated with a subarray
    '''
    if nthreads==1:
        xrange=[0,nx]
        yrange=[0,ny]
        zrange=[0,nz]
    elif (int(np.round(nthreads**(1./3.)))**3 == int(nthreads)): #ideal case, perfect cube, if nthreads in [8,27,64,125,...]:
        ncut = int(np.round(nthreads**(1./3.)))
        thread_x = thread_id % ncut
        thread_y = int( (thread_id%(ncut**2))/ncut )
        thread_z = int( thread_id / (ncut**2) )
        
        xrange=[thread_x*nx/ncut,(thread_x+1)*nx/ncut]
        yrange=[thread_y*ny/ncut,(thread_y+1)*ny/ncut]
        zrange=[thread_z*nz/ncut,(thread_z+1)*nz/ncut]
    elif (int(nthreads**(1./2.))**2 == int(nthreads)): #perfect square, if nthreads in [4,9,16,25,36,49,64,81,100,...]:
        ncut = int(nthreads**(1./2.))
        thread_x = thread_id % ncut
        thread_y = int(np.floor(thread_id/ncut))
        xrange=[thread_x*nx/ncut,(thread_x+1)*nx/ncut]
        yrange=[thread_y*ny/ncut,(thread_y+1)*ny/ncut]
        zrange=[0,nz]  
    else:
        xrange=[thread_id*nx/nthreads,(thread_id+1)*nx/nthreads] #This should be avoided at all costs
        yrange=[0,ny]
        zrange=[0,nz]
        
    xrange=[int(round(xrange[0])),int(round(xrange[1]))]
    yrange=[int(round(yrange[0])),int(round(yrange[1]))]
    zrange=[int(round(zrange[0])),int(round(zrange[1]))]

    
    return xrange,yrange,zrange




def iterate_clump_marching_cubes(args,minval,maxval,ucg_subarray,thread_id):
    '''
    Function for running marching cubes in parallel
    '''
    #t_mc = time.time()
    clump_id, num_features = label((ucg_subarray>minval))
    #print("Subarray "+str(thread_id)+": Finished Marching Cubes in",time.time()-t_mc)
    return clump_id 
            
def iterate_boundary_slice(args,boundary_slice,thread_id):
    '''
    Function for generating a merge map to stitch subarrays back together in parallel
    '''
    nx,ny,nz = np.shape(boundary_slice)
    merge_map = np.zeros((int(np.ceil(nx*ny*nz/2)),2)).astype(np.int32)
    merge_map = merge_clumps.gen_merge_map(boundary_slice.astype(np.int32), merge_map)
    return np.unique(merge_map,axis=0).astype(np.int32)



def iterate_merge_clumps(args,clump_id_subarray,clump_merge_map,thread_id):
        '''
        Call cython code defined by merge_clumps.pyx to apply the merge map to the subarray
        '''
        return merge_clumps.merge_clumps(clump_id_subarray.astype(np.int32), clump_merge_map)



def iterate_get_clump_cell_ids(args,thread_id, clump_id_subarray, cell_id_subarray):
    '''
    Function to map from the UCG of clump ids to cell ids in parallel
    '''
    slices = find_objects(clump_id_subarray)
    clump_cell_ids = defaultdict(list)

    for label, slice_obj in enumerate(slices,start=1):
        if slice_obj is not None:
            clump_id = clump_id_subarray[slice_obj]
            id_region = cell_id_subarray[slice_obj]

            cell_ids_to_add = np.unique(id_region[clump_id == label])
            clump_cell_ids[label] = cell_ids_to_add #add the cell id to the list of clumps

    return clump_cell_ids



def identify_clump_hierarchy(ds,refine_box,args):
    '''
    Call this function to run the clump finder. This will load the dataset, push the needed data onto a uniform covering grid,
    define the root clump, and finally call master_clump.FindClumps() to perform the clump finding.

    Arguments are:
    ds: the yt dataset
    refine_box: The cut region you want to run the clump finder on. Could technically be something beyond just the refine_box
    args: the system arguments parsed by clump_finder_argparser.py
    '''
    #define ds, make sure no data is loaded
    tmp_step = args.step
    tmp_clump_min = args.clump_min
    
    print("Clump min was set to",args.clump_min)
    
    disk_ids=None
    if args.mask_disk or args.identify_disk:
        if args.cgm_density_factor is None:
            if args.cgm_density_cut_type=="relative_density": args.cgm_density_factor=200.
            elif args.cgm_density_cut_type=="comoving_density": args.cgm_density_factor=0.2
            else: args.cgm_density_factor = 1.
        cgm_density_cut = get_cgm_density_cut(ds, args.cgm_density_cut_type,additional_factor=args.cgm_density_factor,code_dir=code_dir)




        sphere = ds.sphere(center=ds.halo_center_kpc, radius=(60, 'kpc'))
        sph_ism = sphere.cut_region("obj['density'] > %.3e" % (cgm_density_cut))
        args.clump_min = sph_ism["gas", "density"].min()
        args.clump_max = sph_ism["gas", "density"].max()
        args.step = args.clump_max / args.clump_min
        
        disk = Clump(ds,refine_box,args,tree_level=0,is_disk=True)
        disk.FindClumps(args.clump_min,args,ids_to_ignore=None)

        current_max = 0
        disk_index = 0
        i=0
        for child in disk.clump_tree[0]:
            if np.size(child.cell_ids)>current_max:
                current_max = np.size(child.cell_ids)
                disk_index = i
            i+=1
        

        args.step=tmp_step
        args.clump_max = None
        args.clump_min = tmp_clump_min

        if args.mask_disk: disk_ids = disk.clump_tree[0][disk_index].cell_ids
        if args.identify_disk:
            disk_output = args.output + "_Disk.h5"
            disk.SaveClump(disk.clump_tree[0][disk_index],args,leaf=False,output=disk_output)
            return None

    print("Clump_min is set to",args.clump_min)
   
    
    if args.clump_min is None or args.clump_max is None:
        current_time = ds.current_time.in_units('Myr').v
        if (current_time<=8656.88):
            density_cut_factor = 1.
        elif (current_time<=10787.12):
            density_cut_factor = 1. - 0.9*(current_time-8656.88)/2130.24
        else:
            density_cut_factor = 0.1



        sphere = ds.sphere(center=ds.halo_center_kpc, radius=(60, 'kpc'))
        sph_ism = sphere.cut_region("obj['density'] > %.3e" % (density_cut_factor * cgm_density_max))
        if args.clump_min is None: args.clump_min = sph_ism["gas", "density"].min()
        if args.clump_max is None: args.clump_max = sph_ism["gas", "density"].max()


    master_clump = Clump(ds,refine_box,args,tree_level=0)
    master_clump.FindClumps(args.clump_min,args,ids_to_ignore=disk_ids)

    
    return master_clump



####This is the example implementation for this code####

args = parse_args()

num_cores = multiprocessing.cpu_count()-1
if args.nthreads is None or args.nthreads>num_cores:
    args.nthreads = num_cores


if args.clumping_field_type is not None:
    args.clumping_field = (args.clumping_field_type , args.clumping_field)
    
if args.code_dir is None:
    code_dir = '/Users/ctrapp/Documents/GitHub/foggie/'
else:
    code_dir = args.code_dir
    
if args.data_dir is None:
    data_dir = '/Volumes/FoggieCam/foggie_halos/'
else:
    data_dir = args.data_dir

halo_id = args.halo #008508
snapshot = args.snapshot #RD0042
nref = args.run #nref11c_nref9f



snap_name = data_dir + "halo_"+halo_id+"/"+nref+"/"+snapshot+"/"+snapshot
trackname = code_dir+"/foggie/halo_tracks/"+halo_id+"/nref11n_selfshield_15/halo_track_200kpc_nref9"
halo_c_v_name = code_dir+"/foggie/halo_infos/"+halo_id+"/"+nref+"/halo_c_v"

#particle_type_for_angmom = 'young_stars' ##Currently the default
particle_type_for_angmom = 'gas' #Should be defined by gas with Temps below 1e4 K

catalog_dir = code_dir + 'foggie/halo_infos/' + halo_id + '/'+nref+'/'
#smooth_AM_name = catalog_dir + 'AM_direction_smoothed'
smooth_AM_name = None

ds, refine_box = foggie_load(snap_name, trackname, halo_c_v_name=halo_c_v_name, do_filter_particles=True,disk_relative=True,particle_type_for_angmom=particle_type_for_angmom,smooth_AM_name = smooth_AM_name)


max_gid=-1
for g,m in ds.all_data().blocks:
    if g.id>max_gid: max_gid=g.id
    
gx_min = np.zeros((max_gid))
gy_min = np.zeros((max_gid))
gz_min = np.zeros((max_gid))
gx_max = np.zeros((max_gid))
gy_max = np.zeros((max_gid))
gz_max = np.zeros((max_gid))
for g,m in ds.all_data().blocks:
    g_dx = g['index','dx'].max()

    gx_min[g.id-1] = (g['index','x'].min() - g_dx/2.)  / g_dx
    gy_min[g.id-1] = (g['index','y'].min() - g_dx/2.)  / g_dx
    gz_min[g.id-1] = (g['index','z'].min() - g_dx/2.)  / g_dx

    gx_max[g.id-1] = (g['index','x'].max() - g_dx/2.)  / g_dx
    gy_max[g.id-1] = (g['index','y'].max() - g_dx/2.)  / g_dx
    gz_max[g.id-1] = (g['index','z'].max() - g_dx/2.)  / g_dx

     
    
def get_cell_grid_ids(field, data):
    gids = data['index','grid_indices'] + 1 #These are different in yt and enzo...
    u_id = np.copy(gids)
    
    idx_dx = data['index','dx']

    x_id = np.divide(data['index','x'] - idx_dx/2. , idx_dx)
    y_id = np.divide(data['index','y'] - idx_dx/2. , idx_dx)
    z_id = np.divide(data['index','z'] - idx_dx/2. , idx_dx)
    
    
    for gid in np.round(np.unique(gids)).astype(int): 
        if gid<=0: continue
        grid_mask = (gids==gid)
        if np.size(np.where(grid_mask)[0])<=0: continue

        gx = x_id[grid_mask]
        gy = y_id[grid_mask]
        gz = z_id[grid_mask]

        gx = gx - gx_min[gid-1]
        gy = gy - gy_min[gid-1]
        gz = gz - gz_min[gid-1]


        max_x = gx_max[gid-1]-gx_min[gid-1]
        max_y = gy_max[gid-1]-gy_min[gid-1]

        c_id =  gx+gy*(max_x+1) +gz*(max_x+1)*(max_y+1)

        u_id[grid_mask] = np.round(gid + c_id * (max_gid+1)).astype(np.uint64)
    return u_id    
    
    
    
ds.add_field(('index', 'cell_id_2'), function=get_cell_grid_ids, sampling_type='cell', force_override=True)
cell_id_field = ('index','cell_id_2')


master_clump = identify_clump_hierarchy(ds,refine_box,args)
if master_clump is None:
    print("Done!")
else:
    master_clump.SaveClumps(args)

    print("Done!")

    ''' Report on some timing info '''
    print("For",args.nthreads,"threads total time=",time.time()-t0)
    print("Time for algorithm=",time_marching_cubes)
    print("Time for io=",time.time()-t0-time_marching_cubes)