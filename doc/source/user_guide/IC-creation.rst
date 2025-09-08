How to Make New Enzo Zoom Initial Conditions 
============================================

This documentation covers the procedures for creating new zoom initial conditions for FOGGIE simulations. 
We use a framework based on the MUSIC multi-scale IC generation tool and the enzo-mrp-music code from 
John Wise that works up through the levels of refinment. These components are combined using custom 
python that is part of the FOGGIE repo. 

This document assumes that you have access to the FOGGIE repo and to the big-box simulation outputs 
in which we create zooms. The bulk of the code needed to coordinate ICs is in the repo under 
foggie/foggie/initial_conditions. 

We are given the big box simulation from whicb we will zoom into a single halo or structure. 
We also have a halo catalog on disk that lists halos from whicn we can choose. 

IC generation follows a basic procedure like so (adapted from the docs for enzo-mrp-music). 
In practice, all these steps are scripted to remove almost all human intervention 

Procedure
=========

1. Collect parameters from the original set of initial conditions
   (ICs) to obtain the shift of the box.

2. Given a halo position and mass, it will calculate the Lagrangian
   volume (LV) at the initial redshift.

3. With this LV, it will modify the appropriate parameters and supply
   a particle position list to MUSIC.  Afterwards, MUSIC is
   automatically executed.

   * Four options for the LV are given: box, ellipsoid, convex hull,
     or exact.

4. The initial conditions directory is then moved to some
   user-specified location.

Example workflow
================

1. Create a set of initial conditions for a unigrid simulation with
   MUSIC. Its parameter file needs to be references in this script's
   configuration file (e.g. sample.conf) as the variable
   template_config.  See template.conf for an example.
   
2. Run unigrid simulation (If an iteration after step #7, run zoom-in
   simulation.)

3. Find halo of interest in the last output

4. Modify parameters in the [region] section in the configuration file
   (e.g. sample.conf). The parameters are documented above their
   declaration.
   
5. Run "python enzo-mrp-music.py sample.conf 1"

6. Run simulation with 1 nested grid with the appropriate Enzo
   parameter file.  MUSIC will only give a skeleton parameter file,
   but the must-refine and nested grid parameters are given in it.
   
7. If another nested grid is needed, goto step #2 and increase the
   level argument by 1.  If not, you're finished!




**Author:**

JT, August 2025 
