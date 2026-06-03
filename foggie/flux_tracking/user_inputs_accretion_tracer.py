'''
This file contains user inputs, which are stored in a dictionary for the sake of flexibility.
Definitions of specific parameters are right above the dictionary entries.
'''

user_inputs = {

    # this is the directory where the dataset that is going to be modified lives.
    # Do not leave a trailing backslash at the end of that directory, and do not
    # add any file names here.
    # NOTE: Python seems to have problems with directories that have spaces in their names, even if
    #       you put backslashes in them.
    "dataset_directory":"/Users/clochhaas/Documents/Research/FOGGIE/Simulation_Data/halo_005036/Accretion_tracer/DD1967",

    # This is the name of the restart parameter file in the dataset directory
    # The code knows how to figure out the names of other files from that.
    "filename_stem": "DD1967",

    # Number of tracer fluid fields.  Must be at least 1 and at most 8
    # (the "at most 8" comes from the Enzo tracer fluid code).
    "NumberOfTracerFluidFields": 5,

    # This is the number of baryon fields that are in the ORIGINAL dataset.
    # If you don't know offhand, look in the dataset's .hierarchy file - each
    # grid entry has a line that says 'NumberOfBaryonFields', and it should be
    # the same for every grid entry.  Use that number.
    "NumberOfOriginalBaryonFields": 14,

    # This controls the level of verbosity of the outputs.  If you set it to True
    # you will get a lot of output, but it will also tell you what the code is doing.
    "DEBUG_OUTPUTS": True,  # True of False

    # if True, this will actually write the tracer fields.
    # if False, it does everything BUT write the tracer fields (dataset is unmodified)
    # It seems useful to have this feature because adding the fields is a bit tricky with
    # the various unit conversions, so you might want to do a dry run first.
    "MODIFY_FILES": True,

    # This sets the default values of the tracer fluid density. "tiny_number" is an
    # Enzo internal value that is typically set to 1e-20.  You probably don't need
    # to modify this.
    "tiny_number": 1.0e-20
}

import yt
import h5py
import numpy as np
from foggie.utils.consistency import *
from foggie.utils.get_run_loc_etc import get_run_loc_etc
from foggie.utils.yt_fields import *
from foggie.utils.foggie_load import *
from foggie.utils.analysis_utils import *
from foggie.clumps.clump_finder.utils_clump_finder import add_cell_id_field
from foggie.clumps.clump_finder.clump_load import create_simple_ucg
from foggie.clumps.clump_finder.load_subsampled_ucgs import *

def modify_grid_files(user_inputs):
    '''
    This is the routine that actually modifies the grid files.  This is much more annoying than any other part of this
    particular code, because what it does is entirely problem-dependent, so this function will need to be modified.  The
    example given here is meant to show users how to work with grids based on cell positions.

    The general flow is:

    1. Use yt to get the basic grid information (it could be done with the hierarchy file, but no need to reinvent the wheel).
    2. Loop over all grids in the dataset.  For each grid:
         * Get the grid and cell positions and calculate some useful quantities
         * Open the HDF5 file the grid lives in
         * Open up the density dataset (which always must exist)
         * Loop over the number of tracer fields that the user has specified.  For each tracer field:
            * Create an array that's the same size/shape/precision as the density array
            * Set it to tiny_number first (to have starting values for the dataset)
            * Modify it to whatever values the user wants based on each cell's position. This will automatically be
              saved in the file upon closing.
         * Close the HDF5 file, which will save the datasets.

    VERY IMPORTANT NOTES FOR USERS:
      * The tracer fluid fields must be added to ALL grids, not just grids where you want to trace something.  This
        is because Enzo requires that all grids have the same set of baryon fields.  Just set values in grids that
        you aren't interested in to some small value.
      * This routine currently does everything in Enzo's internal coordinate system (which is 0-1 in all three spatial
        dimensions for cosmology simulations).
      * Enzo uses column-major array ordering in memory (z-dimension goes first: k, j, i) due to its solvers being
        in Fortran. Python (and numpy) use row-major array ordering in memory (x-dimension goes first: i, j, k).  So,
        any Enzo array that is read from a .cpu file into a numpy array needs to be transposed so that it is in the order
        that numpy expects. It then needs to be transposed BACK before being written to disk so that Enzo gets the arrays
        in the ordering that it expects. The code below does all of this.
    '''

    print("******** Modifying the grid files. ********")

    # Brian's example injection region:
    # sphere center (user sets this)
    #sph_cen_x = 0.52587891
    #sph_cen_y = 0.51708984
    #sph_cen_z = 0.48486328
    # sphere radius - will be multiplied by tracer field number as a test (user sets this)
    #sph_dr = 0.015625

    # load up the Enzo dataset we're interested in (from user inputs)
    enzo_param_file = user_inputs['dataset_directory'] + "/" + user_inputs['filename_stem']
    code_dir = '/Users/clochhaas/Documents/Research/FOGGIE/Analysis_Code/foggie/foggie/'
    ds, refine_box = foggie_load(enzo_param_file, trackfile_name = code_dir + 'halo_tracks/005036/nref11n_selfshield_15/halo_track_200kpc_nref9')
    add_cell_id_field(ds)

    from scipy.ndimage import gaussian_filter
    import scipy.ndimage as ndimage

    # Create covering grid at level 9 to use for identifying streams
    kpctocm = 3.086e+21
    left_edge  = ds.halo_center_kpc - ds.arr([200., 200., 200.], 'kpc')
    right_edge =  ds.halo_center_kpc + ds.arr([200., 200., 200.], 'kpc')
    x_right = right_edge[0]
    y_right = right_edge[1]
    z_right = right_edge[2]
    x_left = left_edge[0]
    y_left = left_edge[1]
    z_left = left_edge[2]
    virial_box = ds.r[x_left:x_right, y_left:y_right, z_left:z_right]
    fields = [('gas','density'),('gas','radius_corrected'),('gas', 'radial_velocity_corrected'),('index','cell_id_2')]
    box = load_field_into_subsampled_ucg_2(ds, virial_box, fields, target_refinement_level=9, max_refinement_level = 11, split_methods=["copy",'copy','copy','copy'], merge_methods=["mean","mean","mean","max"])
    box_den = box[0][0]
    box_radius = box[1][0]
    box_rv = box[2][0]
    box_flux_den = -box_den*box_rv*(box_radius*kpctocm)**2.

    # Step through radii and identify connected accretion streams
    flux_ratio_array = np.zeros(np.shape(box_den)) + 0.01    # array same size as grid initialized with small values everywhere
    radii = np.arange(10., 195., 5.)
    for r in range(len(radii)-1):
        avg_flux_to = np.mean(box_flux_den[(box_radius >= radii[r]) & (box_radius < radii[r+1]) & (box_rv < -50.)])       # average of the flux density of everything moving into radii[r]
        flux_ratio_array[(box_radius >= radii[r]) & (box_radius < radii[r+1])] = box_flux_den[(box_radius >= radii[r]) & (box_radius < radii[r+1])]/avg_flux_to          # set cells that are moving into shape to the ratio with the average flux density value of all cells moving into radii[r]
    flux_ratio_array_smoothed = gaussian_filter(flux_ratio_array, 2.)
    filament_cores = (flux_ratio_array_smoothed > 0.9)

    import napari
    viewer = napari.view_image(np.log10(box_den), name='density', colormap='viridis', contrast_limits=[-30,-20])

    fils_labeled, n_fils = ndimage.label(filament_cores)
    # Ignore filaments that are too small
    unique, counts = np.unique(fils_labeled, return_counts=True)
    for f in range(1,len(unique)):
        if (counts[f]<300):
            fils_labeled[fils_labeled==unique[f]] = 0
    fils_labeled, n_fils = ndimage.label(fils_labeled)
    # Ignore filaments that don't come from maximum radius
    unique = np.unique(fils_labeled)
    for f in range(1,len(unique)):
        if (np.max(box_radius[fils_labeled==unique[f]]) < 0.9*radii[-1]):
            fils_labeled[fils_labeled==unique[f]] = 0
    fils_labeled, n_fils = ndimage.label(fils_labeled)
    print(n_fils)

    fil_layer = viewer.add_image(fils_labeled, name='filaments')

    # If there are more filaments than number of tracer fields, pick the largest ones
    if (n_fils > user_inputs['NumberOfTracerFluidFields']):
        label_counts = np.transpose(np.unique(fils_labeled, return_counts=True))
        sorted_counts = label_counts[label_counts[:, 1].argsort()[::-1]]
        big_fils = []
        for i in range(user_inputs['NumberOfTracerFluidFields']):
            big_fils.append(sorted_counts[1+i][0])
        Nfields = user_inputs['NumberOfTracerFluidFields']
    else:
        Nfields = n_fils
        big_fils = list(range(1,n_fils+1))


    # Loop over the identified streams to get the cell ids of the streams
    masks = []
    box_cell_ids = box[3][0]
    for i in range(n_fils):
        if (i+1 in big_fils):
            print(i+1)
            fil = np.copy(fils_labeled)
            fil[fils_labeled!=i+1] = 0
            fil[fils_labeled==i+1] = 1
            # Restrict the filament to a radial shell
            fil[(box_radius < 180.)] = 0
            fil[(box_radius > 190.)] = 0
            fil_ids = box_cell_ids[fil==1]
            masks.append(fil_ids)
            fil_layer = viewer.add_image(fil, name='filament' + str(i+1))

    napari.run()
    

    # Loop over all of the grids and do things.
    # Note that we have to add the tracer fluid fields to all of the grids, even if you only want to
    # trace fluids in some subvolume of the simulations.  This is because Enzo expects that all grids
    # will have the same baryon fields. Just set values of the tracer field to a very small value in
    # uninteresting regions.

    max_gid=-1
    for g,m in ds.all_data().blocks:
        if g.id>max_gid: max_gid=g.id        

    print("max_gid is ",max_gid,"...len(ds.index.grids)=",len(ds.index.grids))
    for i in range(len(ds.index.grids)):

        if user_inputs['DEBUG_OUTPUTS']:
            print("working on grid", i, "in file", ds.index.grids[i].filename)

        # grid numbers (in the grid names) are 1-indexed, not zero-indexed
        # also the name is zero-padded to have 8 digits total.  If this
        # is ever changed in an Enzo dataset (to have more padding, for example)
        # this is immediately going to crash.
        grid_name = 'Grid' + '{:08d}'.format(i+1)

        # print out some useful information about this grid
        if user_inputs['DEBUG_OUTPUTS']:
            print("grid name is     ", grid_name)  # the actual grid name (that we created)
            print("Grid left edge:  ", ds.index.grids[i].LeftEdge)   # yt-provided grid left edge
            print("Grid right edge: ", ds.index.grids[i].RightEdge)  # yt-provided grid right edge
            print("Grid dimensions: ", ds.index.grids[i].ActiveDimensions)  # yt-provided grid active dimensions (no ghost zones)
            print("Grid level:      ", ds.index.grids[i].Level)      # yt-provided grid level

        # The following is a lot of Brian's original code that Cameron commented out
        # figure out grid edge size along each dimension - they should ALWAYS be identical.
        ##dx_each_dim = (ds.index.grids[i].RightEdge.d - ds.index.grids[i].LeftEdge.d)/ds.index.grids[i].ActiveDimensions

        ##if user_inputs['DEBUG_OUTPUTS']:
        ##    print("Grid dx (per dim):", dx_each_dim)

        # we are going to use the numpy meshgrid functionality to create a 3D grid of x,y,z cell centers so that we can later
        # modify cell values based on their spatial positions (which seems more intuitive than array indices).  First we need
        # the cell centers along each dimension so we can fill in the meshgrid.
       ## xcenters_1D = ds.index.grids[i].LeftEdge.d[0] + (0.5+np.arange(ds.index.grids[i].ActiveDimensions[0]))*dx_each_dim[0]
        ##ycenters_1D = ds.index.grids[i].LeftEdge.d[1] + (0.5+np.arange(ds.index.grids[i].ActiveDimensions[1]))*dx_each_dim[1]
        ##zcenters_1D = ds.index.grids[i].LeftEdge.d[2] + (0.5+np.arange(ds.index.grids[i].ActiveDimensions[2]))*dx_each_dim[2]

        ##if user_inputs['DEBUG_OUTPUTS']:
         ##   print("x,y,z centers (for mesh grid):")
         ##   print(xcenters_1D)
         ##   print(ycenters_1D)
         ##   print(zcenters_1D)

        # Now we actually create the 3D mesh grid, which annoyingly can have two different indexing schemes
        # and also is returned as a list.  The 'ij' indexing scheme does things in the way that is aligned
        # with how Enzo works (after the data arrays are transposed, at least), so we use that.
        ##mesh_3D = np.meshgrid(xcenters_1D,ycenters_1D,zcenters_1D, indexing='ij')

        # split this out into three 3D arrays, one for each dimension.  Each of these arrays
        # now has the x, y, or z cell center for the indexed cell.
        ##xcenters_3D = mesh_3D[0]
        ##ycenters_3D = mesh_3D[1]
        ##zcenters_3D = mesh_3D[2]

        # calculate a grid of radius arrays using the sphere center provided by the user.
        # This will have the same dimensions as the various *centers_3D arrays.
        # This is not required in general, but is an example of something you could do!
        ##radius = ((xcenters_3D-sph_cen_x)**2 + (ycenters_3D-sph_cen_y)**2 + (zcenters_3D-sph_cen_z)**2 )**0.5

        #if user_inputs['DEBUG_OUTPUTS']:
            #print("radius min, max:", radius.min(), radius.max(), "Grid, level:", i, ds.index.grids[i].Level)

        # Cameron's way of doing this:
        xindices=np.arange(ds.index.grids[i].ActiveDimensions[0])
        yindices=np.arange(ds.index.grids[i].ActiveDimensions[1])
        zindices=np.arange(ds.index.grids[i].ActiveDimensions[2])

        indices_3d = np.meshgrid(xindices,yindices,zindices,indexing='ij')
        xindices=indices_3d[0]
        yindices=indices_3d[1]
        zindices=indices_3d[2]
        gid  = ds.index.grids[i].id#i#not +/- 1

        max_x = np.max(xindices)+1
        max_y = np.max(yindices)+1
        c_ids = xindices + max_x*yindices + max_x*max_y*zindices
        grid_cell_ids = np.round(gid + np.multiply(c_ids, max_gid+1)).astype(np.uint64) 


        # open up HDF5 file
        # The 'r+' option allows both reading and writing to the file.
        f = h5py.File(ds.index.grids[i].filename,'r+')

        # read density field (which should always be there) to get dataset dimensions
        # (the tracer fluid fields must be the same size as the other baryon fields, so we're
        # just going to be creatively lazy here)
        dens_name = grid_name + "/Density"

        if user_inputs['DEBUG_OUTPUTS']:
            print("density dataset name:", dens_name)

        # actually read the dataset here
        dens_dset = f[dens_name]

        # Enzo uses column-major ordering in the internal datasets, so we have to transpose datasets
        # to work with them in matplotlib.  This means we have to transpose our tracer fields
        # back to the correct ordering when we write them to the files!
        dens_dset = np.transpose(dens_dset)

        # Add tracer fluids, up to the number the user has specified
        # TracerFluid fields in Enzo are 1-indexed, which is why the range starts with 1
        # Remember that tracer fluids need to be added to ALL grids or else it will break Enzo!
        for tfnum in range(1,Nfields+1):

            # This will create a tracer fluid dataset name that is aligned with what Enzo expects
            tracer_fluid_name = 'TracerFluid' + '{:02d}'.format(tfnum)
            tf_dset_name = grid_name + '/' + tracer_fluid_name

            if user_inputs['DEBUG_OUTPUTS']:
                print("tracer fluid number, field name:", tfnum, tracer_fluid_name)
                print("tracer fluid dataset name:", tf_dset_name)

            # first create a tracer field of zeros
            this_tracer_field = np.zeros_like(dens_dset)

            # then set it to tiny_number (not necessary, but it's consistent with how Enzo
            # generates initial uniform grids)
            this_tracer_field[...] = user_inputs['tiny_number']

            # ***** And now we actually modify the tracer fluid in some spatially-aware way! *****

            # radius now depends on the tracer fluid number (variable tfnum), so the
            # spatial extent of each tracer fluid field is different
            #myrad = sph_dr*tfnum

            # if the tracer fluid is within myrad, give it the same value as
            # the density field (this is arbitrary but convenient, you can do whatever
            # you want)
            #this_tracer_field[radius<=myrad] = dens_dset[radius<=myrad]

            this_mask = np.isin(grid_cell_ids, masks[tfnum-1])#CT 03122025
            this_tracer_field[this_mask] = dens_dset[this_mask] #CT 03122025

            # We now take the tracer fluid field and transpose it back into the
            # column-major order that Enzo expects so that we can write it to disk.
            this_tracer_field = np.transpose(this_tracer_field)

            # Then actually write the dataset, if the user wants you to!
            if user_inputs['MODIFY_FILES']:
                if user_inputs['DEBUG_OUTPUTS']:
                    print("writing dataset", tf_dset_name, "for field", tfnum, "in grid", grid_name)
                f.create_dataset(tf_dset_name,data=this_tracer_field)

            # do a bit of housekeeping in case Python is sloppy with memory management
            # this is not always necessary, but when you have a lot of grids/arrays being created
            # sometimes weird and annoying things happen
            del this_tracer_field

        # memory housekeeping, as described immediately above.
        #del xcenters_1D, ycenters_1D, zcenters_1D, mesh_3D, xcenters_3D, ycenters_3D, zcenters_3D, radius, dens_dset
        del xindices, yindices, zindices, c_ids, grid_cell_ids
        # close HDF5 file, ensuring everything gets written to disk.
        f.close()

    return
