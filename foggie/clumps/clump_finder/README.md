DIRECTORY: `clump_finder`
AUTHOR: Cameron Trapp
DATE STARTED: 01/13/2025
LAST UPDATED: 03/20/2025

This directory contains a set of python and cython scripts to run a clump finder on a FOGGIE halo.

To use for the first time you may need to run 'python setup.py build_ext --inplace' to compile the cython code.

To run the clump finder directly, use clump_finder.py (see below).
You can also load the clump_finder(args, ds, cut_region) function from clump_finder.py to run this modularly. (See ModularUseExample.ipynb for example)
To load a clump in as a cut region, use the function load_disk(ds,clump_file) in utils_clump_finder.py. (See ModularUseExample.ipynb for example)

The algorithm loads in the specified clumping field (typically density) in the forced refinement box (refine_box) of the given snapshot, and
converts it into a uniform covering grid at the given refinement level. From here, a marching cubes alogrithm is run on the UCG to label
contiguous regions above the minimum clumping threshold. If --run_mc_parallel is set to true, the UCG is first subdivided into a number of
subarrays specified by --nSubarrays, and each thread runs the algorithm on a subarray and stitches them back together (this induces an
overhead that is likely not worth it in most cases though.)
        
The algorithm then generates a list of unique cell-ids for each clump, such that they can be re-loaded as cut regions for future analysis.
This step is parallelized along the number of subarrays defined by -nSubarrays.
    
After this is done at a the first clumping threshold (--clump_min), the threshold is multiplied by --step and the algorithm is re-ran.
    
During these sucessive iterations, the hierarchy of the clumps is calculated and stored in the main_clump class object. Parents of clumps are identified as clumps 1 level up that have an overlap in cell_ids.


As an alternative mode of use, setting the --identifty_disk flag will run the clump finder as a disk finder instead. The disk is identified
as the largest clump above a certain density threshold. Depending on the values assigned to --max_disk_void_size and --max_disk_hole_size, 
3-D topologically enclosed voids are filled in this disk mask, as well as 2-D topologically enclosed holes along the disk axis.

By default, the clumps are saved in a single hdf5 file, where each group corresponds to a single clump object with a list of cell ids, parent clump ids, child clump ids, and tree level. There are various functions in utils_clumpfinder.py to help loading and navigating clumps within these files. Disks clumps and shells are saved as individual clump objects.
    
Basic Example usage:
For full clump finding:
python clump_finder.py --refinement_level 11 --clump_min 1.3e-30 --system cameron_local
For disk finding:
python clump_finder.py --refinement_level 11 --identify_disk 1 --system cameron_local


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

    --system: Set the system to get data paths from get_run_loc_etc if not None Overrides --code_dir and --data_dir. Default is None. 
    --pwd: Use pwd arguments in get_run_loc_etc. Default is False.
    --forcepath: Use forcepath in get_run_loc_etc. Default is False.

    --save_clumps_individually: Save each clump as an individual hdf5 file instead of single hdf5 hierarchy. Default is False.

    
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

    --n_dilation_iterations: If greater than 0, the binary mask for each clump will be dilated by n_cells_per_dilation cells this many times. Default is 0. Recommended for disk dilation only.
    --n_cells_per_dilation: How many cells each dilation iteration dilates the clump binary mask by. Default is 1 cell. Total dilation in units of cells is n_dilation_iterations*n_cells_per_dilation.


Parallelization Arguments:

    --nthreads: How many threads to run on? Defaults to number of cores - 1
    --Nsubarrays: How many subarrays should the UCG be split into during parallelization. Default is 64. Should be set to the smallest perfect cube that is larger than nthreads.

    --run_mc_parallel: Do you want to run the marching cubes algorithm in parallel? Incurs additional computational overhead, but may be faster with a large (>30) number of cores or for large datacubes. Default is False.
    --run_mapping_linearly: Do you want to run the clump to cell id mapping without parallelization? Should generally be slower, but can be done while filling voids and holes. Default is False.

    
Disk Identification Arguments:

    --identify_disk: Run the clump finder as a disk finder instead.
    --cgm_density_cut_type: When identifying the disk how do you want to define the CGM density cut? Options are comoving_density, relative_density, or cassis_cut. Default is "relative_density".
    --cgm_density_factor: When identifying the disk, what factor should the cgm_density_cut use. Default is 200 for relative density, 0.2 for comoving density, and 1 for cassis_cut.
    --max_disk_void_size: What is the maximum size of 3D voids (in number of cells) to fill in the disk. Set to above 0 to fill voids. Default is 2000.
    --mask_disk_hole_size: The diameter of the binary closing structure function in number of cells. Will roughly fill holes of this size or smaller. Set to 0 to not fill holes Default is 25 (~7 kpc)

    --closing_iterations: How many iterations of the binary closing algorithm to perform (to fill holes). Default is 1.
    
    --make_disk_mask_figures: Do you want to make additional figures illustrating the void/hole filling process when defining the disk? Default is False.

    --cut_radius: Define a spherical cut region of this radius instead of using the full refine box. Default is None.


| Folder/Module        | Description |
|----------------------|-------------|
| `clump_finder.py` | Contains the main functions and classes for running the clump finder. Can be run from this file or loaded in to another script. |
| `clump_load.py` | Contains functions to convert yt datasets into a uniform covering grid. |
| `merge_clumps.pyx` | Cython code for merging clumps at boundary slices. |
| `setup.py` | Code used to compile the cython code into merge_clumps.c. Run as 'python setup.py build_ext --inplace'. |
| `fill_topology.py` | Contains functions to fill holes in the datacube and in 2-D slices. Used mostly for disk finding. |
| `clump_finder_argparser.py` | Handles the input arguments for running the clump finder. |
| `utils_clump_finder.py` | Contains some basic utility functions, including functions to load in clumps as cut regions in yt. |
| `ModularUseExample.ipynb` | Jupyter notebook with example usage for modular use of the clump_finder and loading in the clump objects as a cut region. |
| `README.md` | Me. |