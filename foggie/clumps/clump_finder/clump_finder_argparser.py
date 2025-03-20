import argparse
from argparse import Namespace

def parse_args():
    '''Parse command line arguments. Returns args object.
    NOTE: Need to move command-line argument parsing to separate file.'''

    parser = argparse.ArgumentParser()
    # Optional arguments:
    parser.add_argument('--refinement_level', metavar='refinement_level', type=int, action='store', \
                        help='Which refinement level should the uniform grid be? Default is the max within the region considered.')
    parser.set_defaults(refinement_level=None)
    
    parser.add_argument('--halo', metavar='halo', type=str, action='store', \
                        help='Which halo id? Default is 008508 (Tempest).')
    parser.set_defaults(halo="008508")
    
    parser.add_argument('--snapshot', metavar='snapshot', type=str, action='store', \
                        help='Which snapshot? Default is RD0042 (redshift 0).')
    parser.set_defaults(snapshot="RD0042")
    
    parser.add_argument('--run', metavar='run', type=str, action='store', \
                        help='Which run? Default is nref11c_nref9f.')
    parser.set_defaults(run="nref11c_nref9f")
    
    parser.add_argument('--output', metavar='output', type=str, action='store', \
                        help='Where do you want to write clump data? Default is ./output/clump_test')
    parser.set_defaults(output="./output/clump_test")
    
    parser.add_argument('--clump_min', metavar='clump_min', type=float, action='store', \
                        help='What should starting density cutoff be? Default is cgm density cutoff.')
    parser.set_defaults(clump_min=None)
    
    
    parser.add_argument('--clump_max', metavar='clump_max', type=float, action='store', \
                        help='What should ending density cutoff be? Default is the maximum in the region.')
    parser.set_defaults(clump_max=None)
    
    parser.add_argument('--step', metavar='step', type=float, action='store', \
                        help='By what factor should clump_min be incremented? Default is 2.')
    parser.set_defaults(step=2)
    
    parser.add_argument('--mask_disk', metavar='mask_disk', type=bool, action='store', \
                        help='Should the disk be masked out? Default is False')
    parser.set_defaults(mask_disk=False)
    
    parser.add_argument('--min_cells', metavar='min_cells', type=int, action='store', \
                        help='Minimum cells required to define a clump. Default is 20')
    parser.set_defaults(min_cells=20)


    parser.add_argument('--nthreads', metavar='nthreads', type=int, action='store', \
                        help='How many threads to run on? Defaults to num_cores-1.')
    parser.set_defaults(nthreads=None)
    
    parser.add_argument('--Nsubarrays', metavar='Nsubarrays', type=int, action='store', \
                        help='How many arrays do you want to split the ucg into? Defaults to 64.')
    parser.set_defaults(Nsubarrays=64)
    
    parser.add_argument('--clumping_field', metavar='clumping_field', type=str, action='store', \
                        help='What field do you want to define clumps on? Default is density.')
    parser.set_defaults(clumping_field="density")
    
    parser.add_argument('--clumping_field_type', metavar='clumping_field_type', type=str, action='store', \
                        help='What data type (gas, dm, stars) do you want to read for clumping_field? Default is gas.')
    parser.set_defaults(clumping_field_type="gas")      
    
    parser.add_argument('--only_save_leaves', metavar='only_save_leaves', type=bool, action='store', \
                        help='Set to True to only save leaf clumps. Default saves full parent hierarchy.')
    parser.set_defaults(only_save_leaves=False)  
    
    parser.add_argument('--code_dir', metavar='code_dir', type=str, action='store', \
                        help='Where is the foggie analysis directory?')
    parser.set_defaults(code_dir=None)  
    
    parser.add_argument('--data_dir', metavar='data_dir', type=str, action='store', \
                        help='Where are the simulation outputs?')
    parser.set_defaults(data_dir=None)  

    parser.add_argument('--include_diagonal_neighbors', metavar='include_diagonal_neighbors', type=bool, action='store', \
                        help='Include neighbors on the diagonal as well. Default is False.')
    parser.set_defaults(include_diagonal_neighbors=False) 
    
    parser.add_argument('--identify_disk', metavar='identify_disk', type=bool, action='store', \
                        help='Save a clump defining the disk with holes filled. Default is False.')
    parser.set_defaults(identify_disk=False) 
    
    parser.add_argument('--cgm_density_cut_type', metavar='cgm_density_cut_type', type=str, action='store', \
                        help='How do you want to define the CGM density cut? Options are ["comoving_density,"relative_density","cassis_cut"]. Default is "relative_density".')
    parser.set_defaults(cgm_density_cut_type="relative_density") 
    
    parser.add_argument('--cgm_density_factor', metavar='cgm_density_factor', type=float, action='store', \
                        help='By what additional factor should the CGM density cut be multiplied by? Default is 200 for relative_density, 0.2 for comoving_density, and 1 for cassis_cut.')
    parser.set_defaults(cgm_density_factor=None) 
    
    parser.add_argument('--max_void_size', metavar='max_void_size', type=int, action='store', \
                        help='If filling voids, what is the maximum cell size on the ucg you wish to fill? Set to 0 to not fill. Default is 2000 cells.')
    parser.set_defaults(max_void_size=0) 

    parser.add_argument('--max_disk_void_size', metavar='max_disk_void_size', type=int, action='store', \
                        help='If filling voids in the disk, what is the maximum cell size on the ucg you wish to fill? Set to 0 to not fill. Default is 2000 cells.')
    parser.set_defaults(max_disk_void_size=2000) 
    
    parser.add_argument('--max_disk_hole_size', metavar='max_disk_hole_size', type=int, action='store', \
                        help='The diameter of the dilation structure function in number of cells. Will roughly fill holes of this size or smaller. Set to 0 to not fill holes. Default is 25')
    parser.set_defaults(max_disk_hole_size=25)

    parser.add_argument('--closing_iterations', metavar='closing_iterations', type=int, action='store', \
                        help='How many closing iterations for filling. Default is 1.')
    parser.set_defaults(closing_iterations=5)
    
    parser.add_argument('--run_mc_parallel', metavar='run_mc_parallel', type=bool, action='store', \
                        help='Do you want to run the marching cubes algorithm in parallel? Incurs additional computational overhead, but may be faster with a large (>30) number of cores or for large datacubes. Default is False.')
    parser.set_defaults(run_mc_parallel=False) 

    parser.add_argument('--run_mapping_linearly', metavar='run_mapping_linearly', type=bool, action='store', \
                        help='Do you want to run the clump to cell id mapping without parallelization? Should generally be slower, but can be done while filling voids and holes. Default is False.')
    parser.set_defaults(run_mapping_linearly=False) 

    parser.add_argument('--system', metavar='system', type=str, action='store', \
                        help='Set the system to get data paths in get_run_loc_etc. Overrides --code_dir and --data_dir. Default is None.')
    parser.set_defaults(system=None) 

    parser.add_argument('--pwd', metavar='pwd', type=bool, action='store', \
                        help='Use pwd arguments in get_run_loc_etc. Default is False.')
    parser.set_defaults(pwd=False) 

    parser.add_argument('--forcepath', metavar='forcepath', type=bool, action='store', \
                        help='Use forcepath in get_run_loc_etc. Default is False.')
    parser.set_defaults(forcepath=False) 

    parser.add_argument('--cut_radius', metavar='cut_radius', type=float, action='store', \
                        help='Define a spherical cut region of this radius instead of using the full refine box. Default is None.')
    parser.set_defaults(cut_radius=None) 

    parser.add_argument('--skip_saving_clumps', metavar='skip_saving_clumps', type=bool, action='store', \
                        help='Set to True to not save the clumps to hdf5 files. Default is False.')
    parser.set_defaults(skip_saving_clumps=False) 

    parser.add_argument('--n_dilation_iterations', metavar='n_dilation_iterations', type=int, action='store', \
                        help='If greater than 0, the mask of each clump will be dilated this many times. Default is 0.')
    parser.set_defaults(n_dilation_iterations=0) 

    parser.add_argument('--n_cells_per_dilation', metavar='n_cells_per_dilation', type=int, action='store', \
                        help='If n_dilation_iterations>0, each iteration will dilate the clump by this many cells. Default is 1.')
    parser.set_defaults(n_cells_per_dilation=1)


    parser.add_argument('--use_cylindrical_connectivity_matrix', metavar='use_cylindrical_connectivity_matrix', type=bool, action='store', \
                        help='Use a cylindrical connectivy matrix (instead of spherical/square) for hole filling. Default is False.')
    parser.set_defaults(use_cylindrical_connectivity_matrix=False)


        
    parser.add_argument('--save_clumps_individually', metavar='save_clumps_individually', type=bool, action='store', \
                        help='If True will save each clump as an individual hdf5 file. Default is False.')
    parser.set_defaults(save_clumps_individually=False)


    args = parser.parse_args()

    return args



def get_default_args():
    """
    Return a Namespace object with the default values for the arguments
    defined in the parse_args function.
    """
    return Namespace(
        refinement_level=None,
        halo="008508",
        snapshot="RD0042",
        run="nref11c_nref9f",
        output="./output/clump_test",
        clump_min=None,
        clump_max=None,
        step=2,
        mask_disk=False,
        min_cells=20,
        nthreads=None,
        Nsubarrays=64,
        clumping_field="density",
        clumping_field_type="gas",
        only_save_leaves=False,
        code_dir=None,
        data_dir=None,
        include_diagonal_neighbors=False,
        identify_disk=False,
        cgm_density_cut_type="relative_density",
        cgm_density_factor=None,
        max_void_size=0,
        max_disk_void_size=2000,
        max_disk_hole_size=2000,
        run_mc_parallel=False,
        run_mapping_linearly=False,
        system=None,
        pwd=False,
        forcepath=False,
        cut_radius=None,
        skip_saving_clumps=False,
        n_dilation_iterations=0,
        n_cells_per_dilation=1,
        closing_iterations=1,
        use_cylindrical_connectivity_matrix=False,
        save_clumps_individually=False,
    )