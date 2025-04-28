Analysis Scripts
================

**Author: Cassi**

The analysis scripts exist to produce a large number of user-requested plots
of FOGGIE simulations, generally focused on determining how "realistic" the 
new physics of FOGGIE 2.0 makes the simulated galaxies.

There are a significant number of user options and functionality provided by this 
script, each of which may be more relevant in different scenarios.

All files associated with the analysis script are located in the foggie 
repo under ``foggie/fogghorn/``. The main driver is ``fogghorn_analysis.py``,
and there are also ``util.py`` and ``header.py`` in this same directory that provide 
some supporting pieces. All plot-making files are separated by group (see below) 
and also located in this directory.

There are two general modes in which the script operates:
1. Make one plot per simulation output, per run. Examples of plots made in 
this way include gas density projections and the KS relation (for all gas cells in 
the disk of the central galaxy). Plots that
require halo finding and that deposit information from all halos in the snapshot on one plot 
fall into this category as well, such as a stellar-mass-halo-mass relation for 
all halos in a high-z snapshot.
2. Make one plot on which information from *many* simulation outputs is collected.
These plots can be used to show the time evolution of the central galaxy, such as 
how the central galaxy evolves along the star-forming main sequence. These plots 
require the calculation of the desired information before the plot can be made, 
and this is done by creating and then adding rows to a table. The plots then 
read from the table. The plots can be re-made quickly if the information in 
the table does not need updating.

The script is designed to be run from the command line with many optional 
arguments. The general structure for calling the script is given in this example:
::

    python fogghorn_analysis.py --directory /path/to/simulation/outputs --trackfile /path/to/trackfile --all_plots

The mandatory arguments are ``--directory``, which gives the path to where 
the DD or RD simulation outputs live, and ``--trackfile``, which gives the path 
to the trackfile for the corresponding run. The trackfile is currently mandatory 
because it is needed in ``foggie_load`` as a first guess for finding the center of 
the zoom region (or the halo center at low redshift), but this may be updated 
in the future and no longer be mandatory to run the analysis script. For now, it is still mandatory.

The ``--all_plots`` option specifies that the script should make all available 
plots. There are other options the user can give instead if only a subset 
of plots are desired (see below).

Other optional arguments:

* ``--save_directory /path/to/saved/plots``: By default, the script will put 
  all the plots it makes into the same directory where the simulation outputs 
  are located, within a subdirectory called ``plots``. The user can use the 
  ``--save_directory`` option to specify a different location to save the plots.
* ``--output DD2000-DD2520``: By default, the script will make plots for every 
  simulation output located in the provided simulation output directory. If instead 
  a subset of outputs is desired, the user can use the ``--output`` option to 
  specify specific outputs separated by commas (no spaces), like ``DD1200,DD1300,DD1400``, 
  or can specify a range that includes all snapshots within the range (endpoints inclusive),
  like ``DD2000-DD2520``.
* ``--clobber``: By default, the script will not re-make any plots that already 
  exist in the plot directory. If instead the user wishes those plots to be remade, 
  specify this with the ``--clobber`` option.
* ``--silent``: By default, the script makes a lot of output as it runs. It will 
  tell you which plots you have asked for, which simulation outputs those plots will 
  be made from, and if any of them already exist, it will tell you if it is skipping 
  over those. If the user would rather not see this output, specify that with the 
  option ``--silent``.
* ``--use_cen_smoothed``: If the plots you are making require the center of the halo, 
  they will by default use the halo center as calculated and stored in the ``halo_c_v``
  file in the repo. If you would rather use the time-smoothed center (such as to make
  less jittery movies), then use this option to use the ``halo_cen_smoothed`` file 
  in the repo instead.

There are more optional arguments, but the rest relate to specific plots and 
will be discussed in the sections below.

All the plots that the script makes are broken into several categories as follows.
The user can request a grouping of plots with the listed option, or can request 
individual plots using ``--make_plots list,of,plot,names`` where the names of 
each individual plot are given in their groupings below. When requesting individual plots,
separate plot names with commas and do not use any spaces.

Jump:

:ref:`all-vis-plots`

:ref:`all-sf-plots`

:ref:`all-edge-plots`

:ref:`all-fb-plots`

:ref:`all-highz-halos-plots`

:ref:`all-time-evol-plots`

:ref:`den-temp-phase`

:ref:`new-plot`

.. raw:: html

   <div style="margin-top: 4em;"></div>

.. _all-vis-plots:

``--all_vis_plots``
-------------------

Use this option to make the following visualization plots:

* **gas_density_projection**:
  This makes a plot of the projected gas density, centered on the center of the 
  halo (as found by ``foggie_load``), zoomed in so as to show mostly just the gas disk.
* **gas_metallicity_projection**:
  Same as above, but for the projected metallicity of the gas disk.

By default, projections will be made in the x direction. If the user wants a different, 
or multiple, projection direction, specify it with the optional argument 
``--projection x,z,x-disk,y-disk`` where the options are x, y, z, x-disk, y-disk, and z-disk.
Specify multiple projection directions with a comma-separated list without spaces.

The code for all these plots is located in ``visualization_plots.py``.

.. _all-sf-plots:

``--all_sf_plots``
------------------

Use this option to make the following star formation related plots:

* **gas_density_projection**:
  This makes the same plot of projected gas density as listed above in ``--all_vis_plots``.
* **young_stars_density_projection**:
  This makes a projection of the density of young stars, defined as stars with 
  ages < 3 Myr. This projection is set up the same way as the gas density 
  projection (same centering and width) so the two can be easily compared.
* **KS_relation**:
  This makes a plot of the Kennicutt-Schmidt relation, which is the surface density 
  of the star formation rate (in Msun/yr/kpc^2) vs. the H I gas surface density. Both of these 
  are computed in projection, so may be somewhat different when projected in different directions.
  The star formation rate is calculated as the mass of stars with ages < 3 Myr divided by 3 Myr.
  This plot also includes a relation from
  `Krumholz, McKee, & Tumlinson (2009) <https://ui.adsabs.harvard.edu/abs/2009ApJ...699..850K/abstract>`_,
  which is taken from the log cZ' = 0.2 curve in Fig. 2.

By default, projections will be made in the x direction. If the user wants a different, 
or multiple, projection direction, specify it with the optional argument 
``--projection x,z,x-disk,y-disk`` where the options are x, y, z, x-disk, y-disk, and z-disk.
Specify multiple projection directions with a comma-separated list without spaces.

The code for all these plots is located in ``star_formation_plots.py``.

.. _all-edge-plots:

``--all_edge_plots``
--------------------

Use this option to make the following plots that orient the disk edge-on:

* **edge_projection**:
  Use this to make a thin-slice density-weighted projection of the gas temperature, 
  with the disk oriented edge-on.
* **edge_slice**:
  Use this to make a slice through the center of the galaxy of the gas temperature, 
  with the disk oriented edge-on.

These projections and slices can only have x-disk and y-disk as the options for the projection 
direction given by the option ``--projection``, since they require the disk to be oriented edge-on.

The code for all these plots is located in ``visualization_plots.py``.

.. _all-fb-plots:

``--all_fb_plots``
------------------

Use this option to make the following feedback-related plots:

* **outflow_rates**:
  Use this to make plots of the mass and energy outflow rates as functions of 
  both radius from center of galaxy and height above/below the galaxy disk.
* **rad_vel_temp_colored**:
  Use this to make a datashader plot of the radial velocity as a function of radius from 
  the center of the galaxy, where the color-coding is by gas temperature.

These plots have no additional user options. The code for all these plots is
located in ``feedback_plots.py``.

.. _all-highz-halos-plots:

``--all_highz_halos_plots``
---------------------------

Use this option to first make halo catalogs using yt's built-in halo finder, then plots 
where every halo in the refine region of the snapshot is placed as one point on each plot. If 
the halo catalog already exists for a given snapshot, it will not re-make the catalog, 
just the plots (if the plots do not already exist). Note that yt's halo finder is very
approximate and should not be used at low redshift. These plots are designed to be used 
for testing how well the star formation and feedback physics produces galaxies on various 
scaling relations in the entire refine region. Recommended not to use these on any snapshots
below a redshift of 2 or 3.

* **halos_density_projection**:
  Use this to make a projection of gas density with all the halos in the catalog 
  overplotted as circles.
* **halos_SMHM**:
  Use this to put all the halos in the snapshot on a stellar-mass-halo-mass relation 
  that also includes observed relations at a few different redshifts and extrapolation 
  of those relations down to lower masses. The relations come from Fig. 7 of 
  `Behroozi et al. (2013) <https://ui.adsabs.harvard.edu/abs/2013ApJ...770...57B/abstract>`_.
* **halos_SFMS**:
  Use this to put all the halos in the snapshot on a star-forming main sequence relation 
  that also includes observed relations at a few different redshifts and extrapolation 
  of those relations down to lower masses. The relations come from Table 9 of 
  `Speagle et al. (2014) <https://ui.adsabs.harvard.edu/abs/2014ApJS..214...15S/abstract>`_.
* **halos_MZR**:
  Use this to put all the halos in the snapshot on a mass-metallicity relation that also 
  includes observed relations at a few different redshifts and an extrapolation of those 
  relations down to lower masses. The relations come from Fig. 10 of 
  `Nakajima et al. (2023) <https://ui.adsabs.harvard.edu/abs/2023ApJS..269...33N/abstract>`_.
* **halos_gasMHM**:
  Use this to put all the halos in the snapshot on a gas-mass-halo-mass plot. This one 
  does not include any observed relations.
* **halos_h2_frac**:
  Use this to put all the halos in the snapshot on a plot of H2 fraction vs. halo mass.
  This one does not include any observed relations.

These plots have no additional user options. The code for all these plots is located
in ``highz_halos_plots.py``.

.. _all-time-evol-plots:

``--all_time_evol_plots``
-------------------------

Use this option to first make a catalog of various properties of the central galaxy 
for every (or user-specified) simulation output, then plot all those outputs on each plot.
This is useful for seeing how the central galaxy evolves over time, and should only be 
used at lower redshifts below ~2. If the user requests any of the plots in this 
category, the catalog will first need to be made. If the catalog already exists, 
making additional plots from this category is very fast because it will read in the 
catalog rather than loading each snapshot. The catalog is a table in a text file 
called ``central_galaxy_info.txt`` that is saved in the same directory as the plots.

* **plot_SMHM**:
  Use this to plot the time evolution of the central galaxy across the stellar-mass-halo-mass
  relation. Each point will be one simulation output, color-coded by redshift.
  This plot also includes some observational relations, also color-coded by redshift in the
  same way. These relations come from Fig. 7 of 
  `Behroozi et al. (2013) <https://ui.adsabs.harvard.edu/abs/2013ApJ...770...57B/abstract>`_.
* **plot_SFMS**:
  Use this to plot the time evolution of the central galaxy across the star-forming main 
  sequence. Each point will be one simulation output, color-coded by redshift. 
  This plot also includes some observational relations, also color-coded by redshift
  in the same way. These relations come from Table 9 of 
  `Speagle et al. (2014) <https://ui.adsabs.harvard.edu/abs/2014ApJS..214...15S/abstract>`_.

These plots have no additional user options. The code for all these plots is located in 
``time_evol_plots.py`` and the code to update the catalog is located in ``central_info_table.py``.
These are the only plots that operate on multiple snapshots at once. Every other 
plot in this script makes a single plot per simulation output.

.. _den-temp-phase:

``--den_temp_phase``
--------------------

Use this to make both mass- and volume-weighted gas density-temperature phase plots for all gas in a sphere
of a radius given by ``--up_to_kpc R`` where R is the desired radius in kpc. 
If no radius is specified, the entire refine box will be used instead.

This plot is not part of any grouping, but can be requested with the option 
``--make_plots den_temp_phase`` or by including it in a comma-separated list (no spaces)
passed to ``--make_plots``.

The code to make this plot is located in ``phase_plots.py``.

.. raw:: html

   <div style="margin-top: 4em;"></div>

.. _new-plot:

To Make a New Analysis Plot
---------------------------

If you want to add a plot that makes one plot per simulation output, follow these steps:

1. Decide if your new plot fits into one of the existing plot categories 
   or should be a new one entirely. If it fits into an existing category, 
   follow the steps below labeled A. If it will be part of a new category, 
   follow the steps below labeled B.

2. Write the function that creates your plot. This function **must** take
   these arguments: ``(ds, region, args, output_filename)``. If you need 
   additional arguments for your function, put them into ``args``. The function 
   that makes this plot **must** produce only a **single** .png file, or else 
   all the machinery that checks if a plot already exists will not work! If you 
   want to make multiple plots, make multiple functions.

   A\. If your plot fits into an existing category, put the function that creates it into one of the existing python scripts.

   B\. If your plot will be part of a new category, create a new python script
   and give it a name that matches how the category will be called, for 
   ease of understanding. Edit ``header.py`` to include this line:
   ``from new_plot_file import *``.

3. Add the name of the function that creates the plot to a grouping list.
   Search in ``fogghorn_analysis.py`` for a comment "IF YOU ADD A PLOT STEP 3".
   This comment indicates the section where the grouping lists are defined.

   A\. If your plot fits into an existing category, add the name of your function as a string 
   to the group you want it to be part of.

   B\. If you're making a new category, define a new ``args.new_category`` (but called
   something that makes sense) as a list and put the name of your plot function 
   as a string into that list. Then, search for "IF YOU ADD A PLOT STEP 3B".
   This comment indicates the section where the user arguments are defined. Add an
   argument for your new plot category that starts with "all", like ``--all_new_category``.
   Then, search for "IF YOU ADD A PLOT STEP 3C". This section is where the list of plots is made from the 
   user-requested arguments. Add your ``args.new_category`` to the ``plots_asked_for`` list
   here so it is called both if the user specifies ``--all_plots`` or if they 
   specify ``--all_new_category``.

4. Add the output filename to the dictionary of file names. Search in ``fogghorn_analysis.py``
   for "IF YOU ADD A PLOT STEP 4". This indicates where this massive dictionary is.
   Add your new plot to this dictionary in the form ``'function_name':snap + '_output_filename.png'``.
   Name your plot something reasonable and don't forget to include the snap number.

5. (Optional): If your new plot needs a projection direction from the user, 
   search ``fogghorn_analysis.py`` for "IF YOU ADD A PLOT STEP 5". This is the section 
   where the plots that need projection directions are defined. Add the function 
   name of your plot to the list ``plots_needing_projection``. You will also need to 
   define multiple functions, one for each projection direction, and multiple output
   filenames (step 4), one for each projection direction. Take a look at how the 
   ``gas_density_projection`` plots are set up, both in ``fogghorn_analysis.py`` and in
   ``visualization_plots.py`` where the functions live, and copy that format.

6. (Optional): If your new plot needs additional options from the user, search 
   ``fogghorn_analysis.py`` for "IF YOU ADD A PLOT STEP 6". This indicates the section 
   where more optional user arguments are defined. Add whatever arguments you need here.

If you want to add a plot that puts multiple simulation outputs on one plot, follow these steps:

1. Add the function that makes your plot to ``time_evol_plots.py``. The function
   **must** take the arguments ``(args, output_filename)`` and **must** create 
   only one .png file. Read in the information from the table using:
   :: 
      
      data = Table.read(args.save_directory + '/central_galaxy_info.txt', format='ascii.ecsv')

2. Follow all the above steps for creating new plots. However, DO NOT create
   a new category for your plot. Add it to the ``args.time_evol_plots`` list.

3. Add whatever you need to calculate from each snapshot to the ``central_info_table.txt``.
   In ``central_info_table.py``, add the name, type, and units to the table in the ``make_table()``
   function. In the ``get_halo_info(ds, snap, args)`` function, add whatever calculation 
   you need, and append it onto the row.

If you had previously created ``central_info_table.txt``, and you are adding new information
to the table, you will need to delete the old ``central_info_table.txt``. The code only checks 
if a snapshot is already in the table, it does not check if the information has been 
changed, and adding additional columns to the table without completely re-making it 
will probably result in some weird errors.
