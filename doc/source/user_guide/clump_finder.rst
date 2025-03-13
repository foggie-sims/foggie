Clump Finder
===============

This document describes the clump finding algorithm in the ``foggie/clumps/clump_finder`` directory.
It also describes how to run the clump finder both from command line and as an importable module, as
well as how to load in the clump files it outputs as a YT cut region for analysis.


Clump Finding Algorithm
-----------------------

**Description:**

The algorithm loads in the specified clumping field (typically density) in a given cut region (default is the refine box)
of the given snapshot, and converts it into a uniform covering grid (UCG) at the given refinement level.
From here, a marching cubes alogrithm is run on the UCG to label contiguous regions above the minimum clumping threshold.
If ``--run_mc_parallel`` is set to true, the UCG is first subdivided into a number of subarrays specified by ``--nSubarrays``,
and each thread runs the algorithm on a subarray and stitches them back together, but this introduces an overhead
that is ends up being computationally slower in most cases.

The algorithm additionally generates a unique cell-id for each cell, and maps these to the same UCG.
These cell-ids are then mapped in parallel to their corresponding clumps. Each clump is then saved as a list of unique cell-ids.
    
After this is done at a the first clumping threshold (``--clump_min``), the threshold is multiplied by ``--step`` and the algorithm is ran again.
    
During these sucessive iterations, the hierarchy of the clumps is calculated and stored in the main_clump class object.
Parents of the successive clumps are identified as clumps 1 level up that have an overlap in cell_ids.

For basic usage examples in a jupyter notebook, see ``ModularUseExample.ipynb``.

**List of files:**

| Folder/Module        | Description |

* ``clump_finder.py``: Contains the main functions and classes for running the clump finder. Can be run from this file or loaded in to another script.
* ``clump_load.py``: Contains functions to convert yt datasets into a uniform covering grid.
* ``merge_clumps.pyx``: Cython code for merging clumps at boundary slices.
* ``setup.py``: Code used to compile the cython code into merge_clumps.c. Run as 'python setup.py build_ext --inplace'.
* ``fill_topology.py``: Contains functions to fill holes in the datacube and in 2-D slices. Used mostly for disk finding.
* ``clump_finder_argparser.py``: Handles the input arguments for running the clump finder.
* ``utils_clump_finder.py``: Contains some basic utility functions, including functions to load in a clump as a cut region in yt.
* ``ModularUseExample.ipynb``: Jupyter notebook with example usage for modular use of the clump_finder and loading in the clump objects as a cut region.
* ``README.md``:  Similar information as is in this user guide.

**Author:**

Cameron

Running the Clump Finder
------------------------

**Running from command line:**

To run the clump finder directly, run ``clump_finder.py`` with by your needed arguments. For example:
::

    python clump_finder.py --refinement_level 11 --clump_min 1.3e-30 --system cameron_local

For a full list of arguments, see the end of this section.

**Running as an importable module:**

The clump finder can also be imported as a module using:
::

    from foggie.clumps.clump_finder import *

or as:
::

    from foggie.clumps.clump_finder.clump_finder import clump_finder
    from foggie.clumps.clump_finder.clump_finder_argparser import get_default_args

In order to run the clump finder, you must first load in the default argument structure using get_default_args()
and modify them accordingly as follows:

::

    args = get_default_args()
    args.refinement_level = 11
    args.clump_min = 1.3e-30
    args.system = "cameron_local"

The clump finder can then be run on any cut region defined in YT. An example for running on the full refine box is given below:
::

    ds, refine_box = foggie_load(snap_name, trackname, halo_c_v_name=halo_c_v_name,
                                 do_filter_particles=True, disk_relative=True)
    cut_region = refine_box
    clump_finder(args, ds, cut_region)

**Special Uses:**

This algorithm can also be used to identify the main galaxy in a system. To use this functionality, you must set the 
``--identify_disk`` input argument to True (default is False). The disk is identified as the largest clump above a single density
threshold. This density threshold can be calculated in a few different ways and is controlled by the following arguments:

* ``--cgm_density_cut_type``: There are a three different options, "relative_density", "comoving_density", and "cassis_cut".
    * "relative_density" (default) ties the cutoff to the mean density + the standard deviation of the smooth component of the CGM multiplied by ``--cgm_density_factor``.
    * "comoving density" evolves the density defined by ``--cgm_density_factor`` times ``foggie.utils.consistency.cgm_density_max`` as a function of scale factor.
    * "cassis_cut" is a by eye cut that evolves with redshift.

* ``--cgm_density_factor`` will make a given cut more stringent if it is a higher value, or less stringent if it is a lower value. The defaults are 200 for "relative_density", 0.2 for "comoving_density", and 1 for "cassis_cut".

There are a few additional considerations that go into identifying the disk,
such as if/how you want to fill voids and holes within the identified disk clump, as well as if you want to save out
shells around the disk to either expand your disk definition or investigate the disk-halo interface.

To fill voids (3d topologically enclosed regions), set ``--max_disk_void_size`` to the size of the largest void you want to fill (in terms of number of cells in the void).

To fill holes (2d topologically enclosed regions that pierce the disk), set ``--max_disk_hole_size`` to the diameter of largest
hole you want to fill in units of cells. For instance, if you want to fill holes with a diameter of 7 kpc on a covering grid with a resolution of 0.274
kpc, you would set this parameter to 26.

To save out shells surround the disk, set the ``--n_dilation_iterations`` and ``--n_cells_per_dilation parameters``. If these are
larger than 0, the algorithm will apply a series of binary dilation operations to the disk mask to identify the shells surrounding
the disk object. ``--n_dilation_iterations`` sets how many shells you will save out, and ``--n_cells_per_dilation`` sets how thick each
shell is.

An example for running the disk finder with void filling, hole filling, and saving out dilation shells from the command line:
::

    python clump_finder.py --refinement_level 11 --identify_disk 1 --max_disk_void_size 2000
                           --max_disk_hole_size 26 --n_dilation_iterations 10 --system cameron-local

or as an importable module:
::

    args = get_default_args()
    args.refinement_level = 11
    args.identify_disk = True
    args.max_disk_void_size = 2000
    args.max_disk_hole_size = 26
    args.n_dilation_iterations = 10
    args.system = "cameron_local"
    ds, refine_box = foggie_load(snap_name, trackname, halo_c_v_name=halo_c_v_name,
                                 do_filter_particles=True, disk_relative=True)
    clump_finder(args, ds, refine_box)


**Full List of Arguments:**


* ``--code_dir``: Where is the foggie analysis directory?
* ``--data_dir``: Where are the simulation outputs?
* ``--refinement_level``: To which refinement_level should the uniform covering grid be made. Defaults to the maximum refinement level in the box.
* ``--halo``: Which halo should be analyzed. Default is 008508 (Tempest)
* ``--snapshot``: Which snapshot should be analyzed? Default is RD0042
* ``--run``: What refinement run should be analyzed? Default is nref11c_nref9f

* ``--output``: Where should the clump data be written? Default is ./output/clump_test
* ``--only_save_leaves``: Set to True to only save the leaf clumps. Default is False.

* ``--system``: Set the system to get data paths from get_run_loc_etc if not None Overrides ``--code_dir`` and ``--data_dir``. Default is None.
* ``--pwd``: Use pwd arguments in get_run_loc_etc. Default is False.
* ``--forcepath``: Use forcepath in get_run_loc_etc. Default is False.


    
Algorithm Arguments:

* ``--clumping_field``: What field are you clumping on? Default is 'density'
* ``--clumping_field_type``: What field type are you clumping on? Default is 'gas' (i.e. this and the previous argument give you ('gas','density')).

* ``--clump_min``: What should the starting density cutoff be? Default is defined as cgm_density_cutoff in foggie.utils.consistency
* ``--clump_max``: What should the final density cutoff be? Default is the maximum density in the simulation.
* ``--step``: By what factor should the density cutoff be incremented during each step? Default is 2

* ``--min_cells``: What is the minimum cell count (on the uniform covering grid) to define as a "clump"

* ``--include_diagonal_neighbors``: Include cells that neighbor on the diagonal during marching cubes. Default is False.
* ``--mask_disk``: Should the disk be masked out? Default is False. Not needed any more, but may offer performance upgrades
* ``--max_void_size``: What is the maximum size of voids (in number of cells) to fill. Set to above 0 to fill voids in clump. Default is 0.

* ``--n_dilation_iterations``: If greater than 0, the binary mask for each clump will be dilated by ``n_cells_per_dilation`` cells this many times. Default is 0. Recommended for disk dilation only.
* ``--n_cells_per_dilation``: How many cells each dilation iteration dilates the clump binary mask by. Default is 1 cell. Total dilation in units of cells is ``n_dilation_iterations`` times ``n_cells_per_dilation``.


Parallelization Arguments:

* ``--nthreads``: How many threads to run on? Defaults to number of cores - 1
* ``--Nsubarrays``: How many subarrays should the UCG be split into during parallelization. Default is 64. Should be set to the smallest perfect cube that is larger than nthreads.

* ``--run_mc_parallel``: Do you want to run the marching cubes algorithm in parallel? Incurs additional computational overhead, but may be faster with a large (>30) number of cores or for large datacubes. Default is False.
* ``--run_mapping_linearly``: Do you want to run the clump to cell id mapping without parallelization? Should generally be slower, but can be done while filling voids and holes. Default is False.

    
Disk Identification Arguments:

* ``--identify_disk``: Run the clump finder as a disk finder instead.
* ``--cgm_density_cut_type``: When identifying the disk how do you want to define the CGM density cut? Options are comoving_density, relative_density, or cassis_cut. Default is "relative_density".
* ``--cgm_density_factor``: When identifying the disk, what factor should the cgm_density_cut use. Default is 200 for relative density, 0.2 for comoving density, and 1 for cassis_cut.
* ``--max_disk_void_size``: What is the maximum size of 3D voids (in number of cells) to fill in the disk. Set to above 0 to fill voids. Default is 2000.
* ``--mask_disk_hole_size``: The diameter of the binary closing structure function in number of cells. Will roughly fill holes of this size or smaller. Set to 0 to not fill holes Default is 25 (~7 kpc)

* ``--closing_iterations``: How many iterations of the binary closing algorithm to perform (to fill holes). Default is 1.
    
* ``--make_disk_mask_figures``: Do you want to make additional figures illustrating the void/hole filling process when defining the disk? Default is False.

* ``--cut_radius``: Define a spherical cut region of this radius instead of using the full refine box. Default is None.


**Author:**

Cameron


Analyzing Clumps
-----------------



Clump objects are saved out as ``hdf5`` files that contain a list of unique cell_ids that belong to the clump. In order to analyze the clumps, you
must load the dataset into YT, add the cell_id field, and then add a masking field that isolates the given clump of interest. This is all done
in the ``load_clumps()`` function in ``utils_clump_finder.py`` and can be used as follows:

First you must import the load_clumps function either as:
::

    from foggie.clumps.clump_finder import *

or:
::

    from foggie.clumps.clump_finder.utils_clump_finder import load_clumps

You must then load the dataset and specify the clump file you wish to read in:
::

    ds, refine_box = foggie_load(snap_name, trackname, halo_c_v_name=halo_c_v_name,
                                 do_filter_particles=True, disk_relative=True)
    clump_file = "/path/to/your/clump.h5"
    clump_cut_region = load_clump(ds, clump_file)

From here the clump can be used as any other cut region in YT. Note, the disks and shells that can be output with the clump finder can
be analyzed in the exact same way. Two clumps can be analzying together by adding the cut regions together. A clump can removed from a cut region
either by subtracting the cut regions, or using the ``mask_clump()`` function in ``utils_clump_finder.py``.


**Author:**

Cameron

