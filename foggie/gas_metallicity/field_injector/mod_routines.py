'''
These are the routines that modify the Enzo parameter file, hierarchy file, and both
the text-based and HDF5 boundary conditions files.  In general, these should not be
modified unless you are very confident that you're doing the right thing, because Enzo
will behave in unpredictable ways if this is not all internally consistent.
'''


import os
import sys
import h5py
import numpy as np

################################################################################
def edit_param_file(user_inputs):
    '''
    edit_param_file

    Edits the parameter file to add tracer fluid information. This routine may feel
    like it's doing an awful lot of error-checking, but we're trying to do something
    very invasive so we need to be a bit cautious.

    What this routine actually DOES is go through the parameter file and look for the
    line that starts with "UseTracerFluid" and, if it exists, set it to '1' (i.e., on).
    It also looks for the line that starts with "NumberOfTracerFluidFields" and sets
    that to the user-specified values.  It will also look for a line that starts with
    "SetTracerFluidFieldsOnStart" and set that to 0. If those lines do NOT exist then 
    they are created and added to the file.  Then, we add DataLabel lines for each of 
    the new tracer fluids.

    Note that we do the check for UseTracerField and NumberOfTracerFluidFields because
    older simulation datasets (pre implementation of the tracer fluid methods in Enzo) will
    not have those lines at all.
    '''

    print("******** Editing parameter file. ********")

    # create some parameter files
    orig_param_file = user_inputs['dataset_directory'] + "/" + user_inputs['filename_stem']
    new_param_file = orig_param_file + ".new"
    backup_param_file = orig_param_file + ".orig"

    # print out file names
    if(user_inputs['DEBUG_OUTPUTS']):
        print(orig_param_file)
        print(new_param_file)
        print(backup_param_file)


    did_UTF_exist = False   # we will check for "UseTracerFluid" in the parameter file.
    did_NOTFF_exist = False  # we will check for "NumberOfTracerFluidFields" in the parameter file.

    # does original parameter file exist?  It should.
    if(os.path.exists(orig_param_file)==True):
        print("Original parameter file exists, continuing")

    # does new parameter file exist?  It should NOT at this point.
    if(os.path.exists(new_param_file)==True):
        print("*** New parameter file exists at this point and shouldn't. You may need to delete something. Exiting.")
        sys.exit(1)
    else:
        print("No new parameter file exists, continuing.")

    # Does backup parameter file exist?  It should NOT at this point.
    if(os.path.exists(backup_param_file)==True):
        print("*** Backup parameter file exists at this point and shouldn't. You may need to delete something. Exiting.")
        sys.exit(1)
    else:
        print("No backup parameter file exists, continuing.")

    # Now we copy the original parameter file into a backup
    mycommand = "cp " + orig_param_file + " " + backup_param_file

    if(user_inputs['DEBUG_OUTPUTS']):
        print(mycommand)

    # do the actual copying here
    errcode = os.system(mycommand)

    # error check copying
    if errcode != 0:
        print("*** System did something fishy (parameter file), quitting.")
        sys.exit(1)

    # Now check to make sure the backup file really exists!
    if(os.path.exists(backup_param_file)==True):
        print("Backup parameter file exists at this point and SHOULD. YAY!")
    else:
        print("*** No new backup parameter file exists, and it should. Something funny is happening.  Quitting.")
        sys.exit(1)

    # And now the magic happens.  We open the original parameter file and our new one,
    # read through the original file and modify the TracerFluid-related lines as we
    # encounter them, and then finally add the extra DataLabel entries at the end of
    # the file.

    # open files
    inputfile = open(orig_param_file,"r")
    outputfile = open(new_param_file,"w")

    # loop over every line in the new file
    for thisline in inputfile:

        # split the string.
        split_line = thisline.split()

        # DO SOME ERROR-CHECKING.
        # This code strongly assumes that the simulation is 3D and that
        # the top grid is a cube, so check to make sure that's true.

        if len(split_line) > 0 and split_line[0] == 'TopGridRank':
            if int(split_line[2]) != 3:
                print("*** I expect this simulation to be 3D! Your simulation has dimensionality ", split_line[2])
                print("*** Exiting.")
                sys.exit(1)

        if len(split_line) > 0 and split_line[0] == 'TopGridDimensions':
            if (int(split_line[2]) != int(split_line[3])) or (int(split_line[2]) != int(split_line[4])):
                print("*** I expect this simulation to be a cube! Your simulation has root grid dimensions ", split_line[2], split_line[3], split_line[4])
                print("*** Exiting.")
                sys.exit(1)


        # Look for UseTracerFluid line.  Note that some lines have
        # length of zero, so check for that too.
        if len(split_line) > 0 and split_line[0] == 'UseTracerFluid':
            did_UTF_exist = True
            if(user_inputs['DEBUG_OUTPUTS']):
                print(split_line)
            if int(split_line[2]) == 1: # do some error checking - make sure that UseTracerFluid is not turned on!
                print("*** Wait, this parameter file already has tracer fluids (UseTracerFluid = 1). Quitting.")
                sys.exit(1)
            else: # assuming we pass error checking, make our modified line.
                split_line[2] = '1'
                thisline = '  '.join(split_line) + "\n"

                if(user_inputs['DEBUG_OUTPUTS']):
                    print("****new UseTracerFluid line:", thisline)

        # As immediately above, but for the NumberOfTracerFluidFields line
        if len(split_line) > 0 and split_line[0] == 'NumberOfTracerFluidFields':
            did_NOTFF_exist = True
            if(user_inputs['DEBUG_OUTPUTS']):
                print(split_line)
            if int(split_line[2]) > 0:  # do error checking - there should not be any tracer fluids in the original file.
                print("*** Wait, this parameter file already has tracer fluids (NumberOfTracerFluidFields > 0). Quitting.")
                sys.exit(1)
            else: # assuming we pass error checking, make our modified line
                split_line[2] = str(user_inputs['NumberOfTracerFluidFields'])
                thisline = '  '.join(split_line) + "\n"
                if(user_inputs['DEBUG_OUTPUTS']):
                    print("****new NumberOfTracerFluidFields line:", thisline)

        # now we write either the original line or the modified line to the output file
        print(thisline, end = "", file=outputfile)


    print("\n", file=outputfile) # add a newline

    # add a 'UserTracerFluid' line if it didn't already exist (i.e., if the dataset is older
    # than this functionality)
    if(did_UTF_exist == False):
        newline = "UseTracerFluid = 1"
        if(user_inputs['DEBUG_OUTPUTS']):
            print("UserTracerFluid line did not exist, creating it. NEW LINE:")
            print(newline)
        print(newline, file=outputfile)

    # as above, but for the NumberOfTracerFluidFields lines
    if(did_NOTFF_exist == False):
        newline = "NumberOfTracerFluidFields = " + str(user_inputs['NumberOfTracerFluidFields'])
        if(user_inputs['DEBUG_OUTPUTS']):
            print("NumberOfTracerFluidFields line did not exist, creating it. NEW LINE:")
            print(newline)
        print(newline, file=outputfile)

    # If we didn't have either of the two lines listed above, the SetTracerFluidFieldsOnStart
    # also definitely doesn't exist, so add it.
    if(did_UTF_exist == False and did_NOTFF_exist == False):
        newline = "SetTracerFluidFieldsOnStart = 0"
        if(user_inputs['DEBUG_OUTPUTS']):
            print("SetTracerFluidFieldsOnStart line almost certainly did not exist, creating it. NEW LINE:")
            print(newline)
        print(newline, file=outputfile)

    # now we add DataLabel entries for the tracer fluids.  This is not
    # necessary for Enzo, but yt (and other analysis codes) need it.
    for i in range(user_inputs['NumberOfTracerFluidFields']):
        print(i, i+user_inputs['NumberOfOriginalBaryonFields'])

        # we're assuming that the new fields are the last fields in that grid entry (which is currently true for tracer fields)
        newline = 'DataLabel[{:d}]             = '.format(i+user_inputs['NumberOfOriginalBaryonFields']) + 'TracerFluid' + '{:02d}'.format(i+1) + "\n"
        print(newline)
        print(newline, end = "", file=outputfile)

    print("\n", file=outputfile)

    # close the output files.
    inputfile.close()
    outputfile.close()

    # Remove the starting parameter file (no extra extension) and move the .new one to the original name.
    # Note that we still have a backup file with the ".orig" extension!

    mycommand = "mv " + new_param_file + " " + orig_param_file

    if(user_inputs['DEBUG_OUTPUTS']):
        print(mycommand)

    # remove our original parameter file
    os.remove(orig_param_file)

    # do the actual moving here
    errcode = os.system(mycommand)

    # error check file move
    if errcode != 0:
        print("*** system did something fishy (parameter file moving), quitting.")
        sys.exit(1)

    return


################################################################################
def edit_hierarchy_file(user_inputs):
    '''

    edit_hierarchy_file

    This file edits the hierarchy file to add tracer fluid information.  As with the
    parameter file editing routine, it does a lot of error-checking, but that's probably
    a good thing here.

    What this actually does is modify each grid entry in two ways.  First, it updates the
    NumberOfBaryonField lines to increment it by the number of tracer fluid fields that has
    been added.  Then, it updates the FieldType line to include the typedefs (from enzo's typedefs.h file)
    for the tracer fluids that have been added.
    '''

    print("******** Editing hierarchy file. ********")

    orig_hierarchy_file = user_inputs['dataset_directory'] + "/" + user_inputs['filename_stem'] + ".hierarchy"

    new_hierarchy_file = orig_hierarchy_file + ".new"

    backup_hierarchy_file = orig_hierarchy_file + ".orig"

    if(user_inputs['DEBUG_OUTPUTS']):
        print(orig_hierarchy_file)
        print(new_hierarchy_file)
        print(backup_hierarchy_file)

    # does original hierarchy file exist?  It should.
    if(os.path.exists(orig_hierarchy_file)==True):
        print("Original hierarchy file exists, continuing")

    # does new hierarchy file exist?  It should NOT at this point.
    if(os.path.exists(new_hierarchy_file)==True):
        print("*** New hierarchy file exists at this point and shouldn't. You may need to delete something. Exiting.")
        sys.exit(1)
    else:
        print("No new hierarchy file exists, continuing.")

    # Does backup hierarchy file exist?  It should NOT at this point.
    if(os.path.exists(backup_hierarchy_file)==True):
        print("*** Backup hierarchy file exists at this point and shouldn't. You may need to delete something. Exiting.")
        sys.exit(1)
    else:
        print("No backup hierarchy file exists, continuing.")

    # Now we copy the original hierarchy file into a backup
    mycommand = "cp " + orig_hierarchy_file + " " + backup_hierarchy_file

    if(user_inputs['DEBUG_OUTPUTS']):
        print(mycommand)

    # do the actual copying here
    errcode = os.system(mycommand)

    # error check copying
    if errcode != 0:
        print("*** system did something fishy (hierarchy file), quitting.")
        sys.exit(1)

    # Now check to make sure the backup file really exists!
    if(os.path.exists(backup_hierarchy_file)==True):
        print("Backup hierarchy file exists at this point and SHOULD. YAY!")
    else:
        print("*** No new backup hierarchy file exists, and it should.  Something funny is happening.  Quitting.")
        sys.exit(1)

    # And now the magic happens.  We open the original hierarchy file and our new one,
    # read through the original file and modify the TracerFluid-related lines as we
    # encounter them.
    #
    # The lines that we have to modify in each grid entry are the "NumberOfBaryonFields" lines,
    # which needs to be incremented by NumberOfTracerFluidFields, and then the FieldType line needs
    # to have an additional NumberOfBaryonFields with the typedefs for each of the tracer fluid fields
    # (as defined in Enzo's typedefs.h file).

    # open files
    inputfile = open(orig_hierarchy_file,"r")
    outputfile = open(new_hierarchy_file,"w")

    # loop over every line in the new file
    for thisline in inputfile:

        # split the string.
        split_line = thisline.split()

        # Look for NumberOfBaryonFields line.  Note that some lines have
        # length of zero, so check for that too.
        if len(split_line) > 0 and split_line[0] == 'NumberOfBaryonFields':
            if(user_inputs['DEBUG_OUTPUTS']):
                print(split_line)

            orig_field_num = int(split_line[2])

            if orig_field_num != user_inputs['NumberOfOriginalBaryonFields']:
                print("*** The number of baryon fields you THINK are in the original dataset are not")
                print("*** what the .hierarchy file thinks they are.  Check NumberOfOriginalBaryonFields")
                print("*** in user_inputs.py.  Exiting!")
                sys.exit(1)

            new_field_num = orig_field_num + user_inputs['NumberOfTracerFluidFields']
            split_line[2] = str(new_field_num)
            thisline = ' '.join(split_line) + "\n"

            if(user_inputs['DEBUG_OUTPUTS']):
                print("****new NumberOfBaryonFields line:", thisline)


        # Look for FieldType line.  Note that some lines have
        # length of zero, so check for that too.
        if len(split_line) > 0 and split_line[0] == 'FieldType':
            if(user_inputs['DEBUG_OUTPUTS']):
                print(split_line)

            for i in range(user_inputs['NumberOfTracerFluidFields']):
                split_line.append( str(106 + i)  )

            thisline = ' '.join(split_line) + "\n"

            if(user_inputs['DEBUG_OUTPUTS']):
                print("****new FieldType line:", thisline)

        # now we write either the original line or the modified line to the output file
        print(thisline, end = "", file=outputfile)


    print("\n", file=outputfile) # add a newline

    # close the output files.
    inputfile.close()
    outputfile.close()

    # Remove the starting hierarchy file (no extra extension) and move the .new one to the original name.
    # Note that we still have a backup file with the ".orig" extension!

    mycommand = "mv " + new_hierarchy_file + " " + orig_hierarchy_file

    if(user_inputs['DEBUG_OUTPUTS']):
        print(mycommand)

    # remove our original parameter file
    os.remove(orig_hierarchy_file)

    # do the actual moving here
    errcode = os.system(mycommand)

    # error check file move
    if errcode != 0:
        print("*** system did something fishy (hierarchy file moving), quitting.")
        sys.exit(1)

    return


################################################################################
def edit_boundary_files(user_inputs):
    '''

    edit_boundary_file

    This file edits both of the boundary files to add tracer fluid information.  As with
    the other file editing routines, it does a lot of error-checking, but that's probably
    a good thing here.


    '''

    print("******** Editing boundary conditions files. ********")

    orig_boundary_file = user_inputs['dataset_directory'] + "/" + user_inputs['filename_stem'] + ".boundary"
    orig_HDF_boundary_file = user_inputs['dataset_directory'] + "/" + user_inputs['filename_stem'] + ".boundary.hdf"

    new_boundary_file = orig_boundary_file + ".new"
    new_HDF_boundary_file = orig_HDF_boundary_file + ".new"

    backup_boundary_file = orig_boundary_file + ".orig"
    backup_HDF_boundary_file = orig_HDF_boundary_file + ".orig"

    if(user_inputs['DEBUG_OUTPUTS']):
        print(orig_boundary_file)
        print(new_boundary_file)
        print(backup_boundary_file)
        print(orig_HDF_boundary_file)
        print(new_HDF_boundary_file)

    # does original boundary file exist?  It should.
    if(os.path.exists(orig_boundary_file)==True):
        print("Original boundary file exists, continuing")

    # does original HDF boundary file exist?  It should.
    if(os.path.exists(orig_HDF_boundary_file)==True):
        print("Original HDF boundary file exists, continuing")

    # does new boundary file exist?  It should NOT at this point.
    if(os.path.exists(new_boundary_file)==True):
        print("*** New boundary file exists at this point and shouldn't. You may need to delete something. Exiting.")
        sys.exit(1)
    else:
        print("No new boundary file exists, continuing.")

    # does new HDF boundary file exist?  It should NOT at this point.
    if(os.path.exists(new_HDF_boundary_file)==True):
        print("*** New HDF boundary file exists at this point and shouldn't. You may need to delete something. Exiting.")
        sys.exit(1)
    else:
        print("No new HDF boundary file exists, continuing.")

    # Does backup boundary file exist?  It should NOT at this point.
    if(os.path.exists(backup_boundary_file)==True):
        print("*** Backup boundary file exists at this point and shouldn't. You may need to delete something. Exiting.")
        sys.exit(1)
    else:
        print("No backup boundary file exists, continuing.")

    # Does backup HDF boundary file exist?  It should NOT at this point.
    if(os.path.exists(backup_HDF_boundary_file)==True):
        print("*** Backup HDF boundary file exists at this point and shouldn't. You may need to delete something. Exiting.")
        sys.exit(1)
    else:
        print("No backup HDF boundary file exists, continuing.")

    # Now we copy the original boundary file into a backup.
    mycommand = "cp " + orig_boundary_file + " " + backup_boundary_file

    if(user_inputs['DEBUG_OUTPUTS']):
        print(mycommand)

    # do the actual copying here
    errcode = os.system(mycommand)

    # error check copying
    if errcode != 0:
        print("*** system did something fishy (boundary file), quitting.")
        sys.exit(1)

    # Now check to make sure the backup file really exists!
    if(os.path.exists(backup_boundary_file)==True):
        print("Backup boundary file exists at this point and SHOULD. YAY!")
    else:
        print("*** No new backup boundary file exists, and it should.  Something funny is happening.  Quitting.")
        sys.exit(1)

    # Now we copy the original HDF boundary file into a backup.
    mycommand = "cp " + orig_HDF_boundary_file + " " + backup_HDF_boundary_file

    if(user_inputs['DEBUG_OUTPUTS']):
        print(mycommand)

    # do the actual copying here
    errcode = os.system(mycommand)

    # error check copying
    if errcode != 0:
        print("*** system did something fishy (HDF boundary file), quitting.")
        sys.exit(1)

    # Now check to make sure the backup HDF file really exists!
    if(os.path.exists(backup_HDF_boundary_file)==True):
        print("Backup HDF boundary file exists at this point and SHOULD. YAY!")
    else:
        print("*** No new backup HDF boundary file exists, and it should.  Something funny is happening.  Quitting.")
        sys.exit(1)

    # we're going to need the size of the boundary conditions for when we create our new HDF5
    # boundary files.
    bcx = bcy = bcz = -1


    # And now the magic happens.  We open the original boundary file and our new one,
    # read through the original file and modify the TracerFluid-related lines as we
    # encounter them.
    #
    # The lines that we have to modify in each grid entry are the "NumberOfBaryonFields" lines,
    # which needs to be incremented by NumberOfTracerFluidFields, and then the FieldType line needs
    # to have an additional NumberOfBaryonFields with the typedefs for each of the tracer fluid fields
    # (as defined in Enzo's typedefs.h file).

    # open files
    inputfile = open(orig_boundary_file,"r")
    outputfile = open(new_boundary_file,"w")

    # loop over every line in the new file
    for thisline in inputfile:

        # split the string.
        split_line = thisline.split()

        # Look for NumberOfBaryonFields line.  Note that some lines have
        # length of zero, so check for that too.
        if len(split_line) > 0 and split_line[0] == 'NumberOfBaryonFields':
            if(user_inputs['DEBUG_OUTPUTS']):
                print(split_line)

            orig_field_num = int(split_line[2])
            new_field_num = orig_field_num + user_inputs['NumberOfTracerFluidFields']
            split_line[2] = str(new_field_num)
            thisline = ' '.join(split_line) + "\n"

            if(user_inputs['DEBUG_OUTPUTS']):
                print("****new NumberOfBaryonFields line:", thisline)


        # Look for FieldType line.  Note that some lines have
        # length of zero, so check for that too.
        if len(split_line) > 0 and split_line[0] == 'BoundaryFieldType':
            if(user_inputs['DEBUG_OUTPUTS']):
                print(split_line)

            for i in range(user_inputs['NumberOfTracerFluidFields']):
                split_line.append( str(106 + i)  )

            thisline = ' '.join(split_line) + "\n"

            if(user_inputs['DEBUG_OUTPUTS']):
                print("****new FieldType line:", thisline)


        # Look for BoundaryDimension line.  Note that some lines have
        # length of zero, so check for that too.
        if len(split_line) > 0 and split_line[0] == 'BoundaryDimension':
            if(user_inputs['DEBUG_OUTPUTS']):
                print(split_line)

            bcx = int(split_line[2])
            bcy = int(split_line[3])
            bcz = int(split_line[4])

            if bcx != bcy or bcx != bcz:
                print("*** this box is not a cube, lots of assumptions break.  Exiting.")
                sys.exit(1)

            if(user_inputs['DEBUG_OUTPUTS']):
                print("boundary dimensions: ", bcx, bcy, bcz)


        # now we write either the original line or the modified line to the output file
        print(thisline, end = "", file=outputfile)


    print("\n", file=outputfile) # add a newline

    # close the output files.
    inputfile.close()
    outputfile.close()

    # Remove the starting boundary file (no extra extension) and move the .new one to the original name.
    # Note that we still have a backup file with the ".orig" extension!

    mycommand = "mv " + new_boundary_file + " " + orig_boundary_file

    if(user_inputs['DEBUG_OUTPUTS']):
        print(mycommand)

    # remove our original boundary file
    os.remove(orig_boundary_file)

    # do the actual moving here
    errcode = os.system(mycommand)

    # error check file move
    if errcode != 0:
        print("*** system did something fishy (boundary file moving), quitting.")
        sys.exit(1)

    # Now on to the HDF5 file! There's no point in copying this in a streaming way like we did the text files.
    # We're just going to go ahead and create a whole new file with h5py.

    # need total number of baryon fields
    total_num_baryon_fields = user_inputs['NumberOfOriginalBaryonFields'] + user_inputs['NumberOfTracerFluidFields']

    # The arrays are for the 2D faces of the root grid, including ghost zones, and for each dimension there are
    # two faces and total_num_baryon_fields entries.
    array_size = bcx**2 * 2 * total_num_baryon_fields

    if(user_inputs['DEBUG_OUTPUTS']):
        print("boundary array sizes are: ", array_size)
        print("estimated file size: ", array_size * 4 * 6)  # 4 comes from 4 bytes/float; 6 comes from 6 arrays total.

    # the boundary dimension type arrays are ones
    ones_array = np.ones(array_size, dtype='float32')

    # the boundary VALUE type is zero.
    zeros_array = np.zeros(array_size, dtype='float32')

    # open file
    f = h5py.File(new_HDF_boundary_file,'w')

    # open first boundary dimension dataset and fill it with ones (and the specific type of floats that Enzo wants here)
    # Then add a bunch of attributes that Enzo expects to see in this file, based on the values calculated above.
    # All of the '>f4' and '>i4' stuff is making sure that the floats and ints are 32-bit big-endian, which is what Enzo needs.
    dset = f.create_dataset("BoundaryDimensionType.0",data=ones_array.astype('>f4'))
    dset.attrs.create(name='BoundaryDimension', data=np.array((bcx, bcx, 0), dtype='int32'), shape=(3,), dtype='>i4')
    dset.attrs.create(name='BoundaryRank', data=np.array((3), dtype='int32'), shape=(1,), dtype='>i4')
    dset.attrs.create(name='Index', data=np.array((2), dtype='int32'), shape=(1,), dtype='>i4')
    dset.attrs.create(name='NumberOfBaryonFields', data=np.array((total_num_baryon_fields), dtype='int32'), shape=(1,), dtype='>i4')
    dset.attrs.create(name='size', data=np.array((bcx*bcx), dtype='int32'), shape=(1,), dtype='>i4')

    # do the same for the second boundary dimension
    dset = f.create_dataset("BoundaryDimensionType.1",data=ones_array.astype('>f4'))
    dset.attrs.create(name='BoundaryDimension', data=np.array((bcx, bcx, 0), dtype='int32'), shape=(3,), dtype='>i4')
    dset.attrs.create(name='BoundaryRank', data=np.array((3), dtype='int32'), shape=(1,), dtype='>i4')
    dset.attrs.create(name='Index', data=np.array((2), dtype='int32'), shape=(1,), dtype='>i4')
    dset.attrs.create(name='NumberOfBaryonFields', data=np.array((total_num_baryon_fields), dtype='int32'), shape=(1,), dtype='>i4')
    dset.attrs.create(name='size', data=np.array((bcx*bcx), dtype='int32'), shape=(1,), dtype='>i4')

    # do the same for the third boundary dimension
    dset = f.create_dataset("BoundaryDimensionType.2",data=ones_array.astype('>f4'))
    dset.attrs.create(name='BoundaryDimension', data=np.array((bcx, bcx, 0), dtype='int32'), shape=(3,), dtype='>i4')
    dset.attrs.create(name='BoundaryRank', data=np.array((3), dtype='int32'), shape=(1,), dtype='>i4')
    dset.attrs.create(name='Index', data=np.array((2), dtype='int32'), shape=(1,), dtype='>i4')
    dset.attrs.create(name='NumberOfBaryonFields', data=np.array((total_num_baryon_fields), dtype='int32'), shape=(1,), dtype='>i4')
    dset.attrs.create(name='size', data=np.array((bcx*bcx), dtype='int32'), shape=(1,), dtype='>i4')

    # now set the actual values for the boundary dimension.  This is all zeros, and Enzo doesn't want any attributes.
    f.create_dataset("BoundaryDimensionValue.0",data=zeros_array.astype('>f4'))
    f.create_dataset("BoundaryDimensionValue.1",data=zeros_array.astype('>f4'))
    f.create_dataset("BoundaryDimensionValue.2",data=zeros_array.astype('>f4'))

    f.close()

    # Remove the starting HDF5 boundary file (no extra extension) and move the .new one to the original name.
    # Note that we still have a backup file with the ".orig" extension!

    mycommand = "mv " + new_HDF_boundary_file + " " + orig_HDF_boundary_file

    if(user_inputs['DEBUG_OUTPUTS']):
        print(mycommand)

    # remove our original HDF5 boundary condition file
    os.remove(orig_HDF_boundary_file)

    # do the actual moving here
    errcode = os.system(mycommand)

    # error check file move
    if errcode != 0:
        print("*** system did something fishy (boundary HDF5 file moving), quitting.")
        sys.exit(1)

    return
