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
    #"dataset_directory":"/Users/acharyya/models/simulation_output/foggie/halo_008508/nref11c_nref9f/RD0027",
    "dataset_directory":"/nobackupp19/aachary2/tracer_fluid_runs/RD0014",

    # This is the name of the restart parameter file in the dataset directory
    # The code knows how to figure out the names of other files from that.
    #"filename_stem":"RD0027",
    "filename_stem":"RD0014",

    # Number of tracer fluid fields.  Must be at least 1 and at most 8
    # (the "at most 8" comes from the Enzo tracer fluid code).
    "NumberOfTracerFluidFields": 4,

    # This is the number of baryon fields that are in the ORIGINAL dataset.
    # If you don't know offhand, look in the dataset's .hierarchy file - each
    # grid entry has a line that says 'NumberOfBaryonFields', and it should be
    # the same for every grid entry.  Use that number.
    "NumberOfOriginalBaryonFields": 14,

    # This controls the level of verbosity of the outputs.  If you set it to True
    # you will get a lot of output, but it will also tell you what the code is doing.
    "DEBUG_OUTPUTS": True,  # True of False

    # If True, this will actually write the tracer fields.
    # If False, it does everything BUT write the tracer fields (dataset is unmodified)
    # It seems useful to have this feature because adding the fields is a bit tricky with
    # the various unit conversions, so you might want to do a dry run first.
    "MODIFY_FILES": True,

    # This has been added to fix a problem with the file system on the NASA Pleiades
    # supercomputer's file system where, if a single file is closed and then opened too
    # quickly, it will throw an HDF5 error.  If set to True, if two grids in a row are
    # stored in the same file then it will wait PLEIADES_SLEEP_TIME_SECONDS seconds
    # after it closes the file to try opening it again.
    "PLEIADES_SLEEP": False,

    # Number of seconds to wait before trying to open a file after it has been closed,
    # if PLEAIDES_SLEEP is set to True.
    # Note that this is set to what seems like a reasonable value, but it's possible it
    # could be smaller and still be fine, or may need to be larger if the file system is
    # very laggy.
    "PLEIADES_SLEEP_TIME_SECONDS": 1,

    # This sets the default values of the tracer fluid density. "tiny_number" is an
    # Enzo internal value that is typically set to 1e-20.  You probably don't need
    # to modify this.
    "tiny_number": 1.0e-20
}

import yt
import h5py
import numpy as np
import time

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

    This specific example creates a set of nested spheres in the tracer fluids.  The user
    specifies the number of tracer fluids to be created in the user_inputs dictionary and
    then sets the sphere center (sph_cen_x/y/z below) as well as the radius of the smallest
    sphere (for tracer fluid 1), which is sph_dr (below).  The second tracer fluid will occupy a
    sphere that is 2x the size of sph_dr, the third will occupy a sphere of radius 3x sph_dr,
    etc.  The tracer fluid is set to the value of the Density field.

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

    # sphere center (user sets this)
    # this is for Tempest RD0027
    # sph_cen_x = 0.49248219
    # sph_cen_y = 0.48288059
    # sph_cen_z = 0.50463009

    # this is for Tempest RD0014
    sph_cen_x = 0.49541579
    sph_cen_y = 0.49414359
    sph_cen_z = 0.49955474

    # sphere radius - will be multiplied by tracer field number as a test (user sets this)
    #sph_dr = 2.77999994e-05 # this corresponds to 2 kpc for RD0027
    sph_dr = 6.95e-05 # this corresponds to 2 kpc for RD0014

    # load up the Enzo dataset we're interested in (from user inputs)
    enzo_param_file = user_inputs['dataset_directory'] + "/" + user_inputs['filename_stem']
    ds = yt.load(enzo_param_file)

    # keeps track of the last file that was opened so that we can hold off
    # on re-opening it if necessary.
    last_file_opened = None

    # Loop over all of the grids and do things.
    # Note that we have to add the tracer fluid fields to all of the grids, even if you only want to
    # trace fluids in some subvolume of the simulations.  This is because Enzo expects that all grids
    # will have the same baryon fields. Just set values of the tracer field to a very small value in
    # uninteresting regions.
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

        # figure out grid edge size along each dimension - they should ALWAYS be identical.
        dx_each_dim = (ds.index.grids[i].RightEdge.d - ds.index.grids[i].LeftEdge.d)/ds.index.grids[i].ActiveDimensions

        if user_inputs['DEBUG_OUTPUTS']:
            print("Grid dx (per dim):", dx_each_dim)

        # we are going to use the numpy meshgrid functionality to create a 3D grid of x,y,z cell centers so that we can later
        # modify cell values based on their spatial positions (which seems more intuitive than array indices).  First we need
        # the cell centers along each dimension so we can fill in the meshgrid.
        xcenters_1D = ds.index.grids[i].LeftEdge.d[0] + (0.5+np.arange(ds.index.grids[i].ActiveDimensions[0]))*dx_each_dim[0]
        ycenters_1D = ds.index.grids[i].LeftEdge.d[1] + (0.5+np.arange(ds.index.grids[i].ActiveDimensions[1]))*dx_each_dim[1]
        zcenters_1D = ds.index.grids[i].LeftEdge.d[2] + (0.5+np.arange(ds.index.grids[i].ActiveDimensions[2]))*dx_each_dim[2]

        if user_inputs['DEBUG_OUTPUTS']:
            print("x,y,z centers (for mesh grid):")
            print(xcenters_1D)
            print(ycenters_1D)
            print(zcenters_1D)

        # Now we actually create the 3D mesh grid, which annoyingly can have two different indexing schemes
        # and also is returned as a list.  The 'ij' indexing scheme does things in the way that is aligned
        # with how Enzo works (after the data arrays are transposed, at least), so we use that.
        mesh_3D = np.meshgrid(xcenters_1D,ycenters_1D,zcenters_1D, indexing='ij')

        # split this out into three 3D arrays, one for each dimension.  Each of these arrays
        # now has the x, y, or z cell center for the indexed cell. (i.e., the value given is the
        # spatial location of that specific cell's center).  These arrays are 3D arrays and
        # can either be looped through or their values can be looped over.
        xcenters_3D = mesh_3D[0]
        ycenters_3D = mesh_3D[1]
        zcenters_3D = mesh_3D[2]

        # calculate a grid of radius arrays using the sphere center provided by the user.
        # This will have the same dimensions as the various *centers_3D arrays (which should
        # be a 3-dimensional array with the same shape as the baryon fields in that grid).
        # This is not required in general, but is an example of something you could do!
        radius = ((xcenters_3D-sph_cen_x)**2 + (ycenters_3D-sph_cen_y)**2 + (zcenters_3D-sph_cen_z)**2 )**0.5

        if user_inputs['DEBUG_OUTPUTS']:
            print("radius min, max:", radius.min(), radius.max(), "Grid, level:", i, ds.index.grids[i].Level)

        # If the last grid was in the same file as the current grid we're working on (i.e., if the last
        # file we worked on is the same as this file) then some file systems need a bit of time to
        # realized that the file was recently closed so that HDF5/h5py doesn't throw an error.  So,
        # if this feature is turned on and the last file is the same as this file, then wait for
        # a user-specified number of seconds.
        if last_file_opened == ds.index.grids[i].filename and user_inputs['PLEIADES_SLEEP'] == True:
            if user_inputs['DEBUG_OUTPUTS']:
                print("************  Last file is the same as this file.  Waiting for this many seconds:", user_inputs['PLEIADES_SLEEP_TIME_SECONDS'])
            time.sleep(user_inputs['PLEIADES_SLEEP_TIME_SECONDS'])

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
        for tfnum in range(1,user_inputs['NumberOfTracerFluidFields']+1):

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
            inner_rad = 0 if tfnum == 1 else sph_dr * (tfnum - 1)
            outer_rad = sph_dr*tfnum

            # if the tracer fluid is within inner_rad and outer_rad, give it the same value as
            # the density field (this is arbitrary but convenient, you can do whatever
            # you want and don't have to be constrained to a sphere!)
            this_tracer_field[(radius > inner_rad) & (radius <= outer_rad)] = dens_dset[(radius > inner_rad) & (radius <= outer_rad)]

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
        del xcenters_1D, ycenters_1D, zcenters_1D, mesh_3D, xcenters_3D, ycenters_3D, zcenters_3D, radius, dens_dset

        # close HDF5 file, ensuring everything gets written to disk.
        f.close()

        last_file_opened = ds.index.grids[i].filename

    return
