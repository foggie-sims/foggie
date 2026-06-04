'''
This is the driver routine to inject tracer fluid fields into an existing Enzo dataset.

WARNING: This program will modify a dataset in-place, so you should absolutely make a
         backup copy of your original Enzo dataset before you start and then verify
         that the tracer fluids were correctly edited before you delete said backup copy.

All of the user inputs, including the routine that actually edits the .cpu files, can be
found in the file user_inputs.py.  You should not have to modify any other files.

Author:  Brian O'Shea (oshea@msu.edu), Feb. 2025
'''

from user_inputs import *
from mod_routines import *

if user_inputs['DEBUG_OUTPUTS']:
    print("*"*40)
    print("input user dictionary:\n")
    print(user_inputs)
    print("*"*40,"\n")

# Creates the new parameter file (with tracer fluid contents in it)
if user_inputs['MODIFY_FILES'] == True:
    edit_param_file(user_inputs)
else:
    print("Skipping edit_param_file because this is a dry run (MODIFY_FILES = False)\n")

# Creates the new hierarchy file (with tracer fluid contents in it)
if user_inputs['MODIFY_FILES'] == True:
    edit_hierarchy_file(user_inputs)
else:
    print("Skipping edit_hierarchy_file because this is a dry run (MODIFY_FILES = False)\n")

# Creates the two new boundary conditions files (with tracer fluid contents in them)
if user_inputs['MODIFY_FILES'] == True:
    edit_boundary_files(user_inputs)
else:
    print("Skipping edit_boundary_files because this is a dry run (MODIFY_FILES = False)\n")

# Modifies the existing grid files to add tracer fluid fields. Note that we don't use the
# same logic as the prior function calls (re: MODIFY_FILES) because we often will want to
# actually go through the logic of modifying the grid files without doing so, but the other
# files are much more straightforward so that's not as big of a concern.
modify_grid_files(user_inputs)
