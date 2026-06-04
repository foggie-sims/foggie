# Enzo field injector and field modifier codes

Written by Brian O'Shea (oshea@msu.edu), Feb. 2025

---

## Enzo field injector

### Purpose of the field_injector code

The goal of this code is to add tracer fluids to a three-dimensional Enzo dataset that **does not currently have said tracer fluid fields in it** so that the simulation can continue to be run WITH the tracer fluid fields (so it requires a version of Enzo that can advect tracer fluid fields as well). It does so by modifying basically all of the files in a standard Enzo data output (parameter file, hierarchy file, both boundary conditions files, and the actual .cpuNNNN files where the simulation data is kept).

**WARNING: Make sure to create a backup copy of your simulation dataset before you modify it - this type of modification will irrevocably change parts of the data output being modified!**

### The code

**Running the code:** Type `python field_injector.py` and then step back and watch the fireworks.

The files included in this code are:

`field_injector.py` - the driver code, which calls routines that edit the various Enzo data output files.

`user_inputs.py` - This includes both a dictionary of user inputs that are used throughout the code as well as a routine called `modify_grid_files` that will inevitably need to be edited by users so that it sets the tracer fluid fields to the values they want.

`mod_routines.py` - This includes the routines that modify the parameter file, hierarchy file, and both the ASCII and binary boundary conditions files.  This should not need to be edited by the user.

---

## Enzo field modifier

### Purpose of the field_modifier code

The goal of this code is to **modify** tracer fluids to a three-dimensional Enzo dataset that **already has tracer fluid fields in it**. It does so by modifying only the .cpuNNNN files where the simulation data is kept, since all of the other machinery is in place.

**WARNING: Make sure to create a backup copy of your simulation dataset before you modify it - this type of modification will irrevocably change parts of the data output being modified!**

### The code

**Running the code:** Type `python field_modifier.py` and then step back and watch the fireworks.

The only file required for this program is `field_modifier.py`, which includes the driver, a dictionary of user inputs, and a routine called `modify_tracer_fields`.  The user input dictionary and `modify_tracer_fields` will need to be edited by users so that it sets the tracer fluid fields to the values they want.

---

## Notes/Caveats/known issues (these apply to both codes)

* When adding the tracer fluid fields these codes go through grids in numerical order, which means that the code is accessing the binary (.cpuNNNN) files in effectively random order.  This could cause a performance issue; if so, the routine that modifies grids can be modified so that it goes through the grids in a different order so that a single .cpuNNNN file is being edited at a time.
* These codes should be easy to parallelize if need be - the .cpuNNNN files can be edited completely independently of each other.
* These codes only run on 3D, cubic datasets at present. This limitation is due to assumptions baked into the code in various places, and is relatively easy to fix if necessary.  This is NOT a problem with cosmological simulations, since they are required to be cubes. If you are doing something else, however, it could end up being an issue.
* Enzo stores data in column-major order (i.e., 3D arrays are stored in (z,y,x) order) because it uses Fortran solvers, and Fortran orders things in memory using column-major order.  Numpy uses row-major order (i.e., 3D arrays are stored in (x,y,z) order), just like C and C++ (and Python in general). SO, in order to correctly modify the tracer fluid fields one has to read in the arrays from the Enzo .cpu files (which are in column-major order), transpose them into row-major order, create tracer fluid fields and modified them using this transposed (row-major) order, transpose the tracer fluid fields to get them back into column-major order, and then write the tracer fluid fields into the .cpu files. This is confusing, but the `modify_grid_files` routine in `user_inputs.py` in the field injection code and the `modify_tracer_fields` routine in `field_modifier.py` in the field modification code) is heavily commented to explain what is going on.
* When restarting an Enzo simulation with tracer fluids, you MUST use an Enzo binary that includes the tracer fluid advection code!  If you do not do so, the code will (probably) crash.
