Getting Started with FOGGIE
===========================

If you would prefer to use this guide as a Jupyter notebook, you can find it in the FOGGIE GitHub repository
in ``foggie/foggie/documentation/my_first_foggie_script.py``. It contains the same information as listed on
this page, plus a few quick examples for using yt to make projection and slice plots of FOGGIE data.

If you're looking for some quick intro material for a summer undergrad research student, check out
``foggie/foggie/documentation/undergrad_foggie_intro.ipynb``, which boils down most of the below
information into a much more user-friendly form with more description and some example plots.

Python Packages Needed
----------------------

* numpy
* scipy
* astropy
* matplotlib
* yt
* datashader
* seaborn

Installing the GitHub FOGGIE Repo
---------------------------------

1. Make a directory where you'd like to store the FOGGIE code
2. Go into that directory in a terminal and type
    ::

        git clone https://github.com/foggie-sims/foggie

3. Open your .bash_profile or .cshrc or .tcshrc or .zshrc, whichever file is your terminal profile. Add this line:
    ::

        export PYTHONPATH="/path/to/foggie/directory:${PYTHONPATH}"

    and change the path to wherever you're storing the FOGGIE code.


FOGGIE Simulation Data
----------------------

There are six directories for the six halos: ``halo_00XXXX`` where XXXX is the halo ID.

Within each halo directory, there is another directory called ``nref11c_nref9f`` -- this is the refinement scheme
for the run and means level 11 cooling refinement and level 9 forced refinement.

Finally, within the ``nref11c_nref9f`` directories are all the time snapshots for all the halos, which are called
DD1234 or RD0012, for example. The DD's are spaced every 5.38 Myr and the RD's are spaced at certain redshifts
(z = 2.5, 2.25, 2.0, 1.75, 1.5, 1.4, 1.3, etc).

Halo ID and name matching:

* Tempest: 8508
* Squall: 5016
* Maelstrom: 5036
* Blizzard: 4123
* Hurricane: 2392
* Cyclone: 2878

There is some basic information about the five galaxies that have reached z = 0 in Table 1 of `Wright et al. (2023) <https://ui.adsabs.harvard.edu/abs/2023arXiv230910039W/abstract>`_

Using Code in the FOGGIE Repo
-----------------------------

Include at least these imports:
::

    import yt  # Simulation plotting package
    import matplotlib.pyplot as plt  # General purpose plotting
    import matplotlib as mpl

    import numpy as np   # General purpose math and array manipulation

    # These are FOGGIE-specific functions that make loading in the dataset easier
    from foggie.utils.consistency import *
    from foggie.utils.yt_fields import *
    from foggie.utils.foggie_load import *
    from foggie.utils.analysis_utils import *

Now you need to specify the directories where the data and analysis scripts are stored:

::

    # Path to where the simulation data itself is stored:
    # CHANGE THIS TO YOUR OWN DIRECTORY PATH
    foggie_dir = "/path/to/data/storage/FOGGIE_data/halo_005036/nref11c_nref9f/"

    # Path to where the FOGGIE code you downloaded from GitHub is stored:
    # CHANGE THIS TO YOUR OWN DIRECTORY PATH
    code_path = "/path/to/code/storage/foggie/foggie/"

    # These next two are needed for finding the galaxy halo we're interested in from the whole dataset.
    # If you're using a different halo, change only the halo ID number (ex. here is 5036)
    halo_c_v_name = code_path + 'halo_infos/005036/nref11c_nref9f/halo_c_v'
    track_name = code_path + 'halo_tracks/005036/nref11n_selfshield_15/halo_track_200kpc_nref9'

    # Finally, specify which snapshot you want to load in.
    snap = 'DD2520'
    snap_name = foggie_dir + snap + '/' + snap

Use yt for loading and examining simulation data: https://yt-project.org/doc/

We have our own function that replaces ``yt.load()`` called ``foggie_load()``. It returns a data structure, ``ds``, just like ``yt.load()``,
and also a yt data object ``refine_box`` that is just the high-resolution region. It sets up finding the galaxy halo and a few other important
pieces, but otherwise everything is still within the framework of yt.

Use ``foggie_load`` by putting the following in your python script:
::

    ds, refine_box = foggie_load(snap_name, trackfile_name=track_name, halo_c_v_name=halo_c_v_name)

Note that the optional argument ``trackfile_name`` points towards the file that provides the location of the halo track box. You should 
always use this when analyzing a run that had a track box, which is all of the production runs. This used to be a mandatory argument but 
was changed 9/5/25, so if you're using old code that has this as mandatory you will probably need to fix it.

There are several necessary variables and fields that foggie_load sets up for you:

* ``ds.halo_center_kpc``: This is an array of [x, y, z] position of the center of the halo (defined as dark matter density peak)
* ``ds.halo_velocity_kms``: This is an array of [v_x, v_y, v_z] velocity vector of the center of the halo (defined as bulk velocity of the
  stars and dark matter particles within 3 kpc of the halo center)
* ``ds.refine_width``: This is the size of the "refine box" - the high-resolution halo track box
* If you give ``foggie_load`` the optional argument ``disk_relative = True``, then ``ds.z_unit_disk`` is an array of [n_x, n_y, n_z] normal
  vector of the galaxy's angular momentum (defined using stars with ages less than 10 Myr within 15 kpc of halo center). ``ds.x_unit_disk`` and 
  ``ds.y_unit_disk`` are also defined and they are the orthogonal vectors within the plane of the galaxy disk.
* If you don't care about stars or dark matter particles and want ``foggie_load`` to run faster, give it the optional argument ``filter_particles = False``.
* ``foggie_load()`` adds some new fields that are corrected for the location and motion of the halo through the cosmological domain:
    - ``('gas', 'vx_corrected')``, ``('gas', 'vy_corrected')``, and ``('gas', 'vz_corrected')`` should be used instead of
      ``('gas', 'velocity_x')``, ``('gas', 'velocity_y')``, and ``('gas', 'velocity_z')``. They are corrected for the motion of the halo,
      so the halo center is defined at zero velocity in all three directions.
    - Likewise, ``('gas', 'vel_mag_corrected')`` and ``('gas', 'radial_velocity_corrected')`` should be used for velocity magnitude and
      radial velocity (negative is toward halo center, positive is away from halo center).
* ``foggie_load()`` calculates some spherical coordinates too. NOTE these are relative to the (arbitrary) code coordinates, NOT relative to the galaxy disk:
    - ``('gas', 'radius_corrected')``, ``('gas', 'theta_pos')``, and ``('gas', 'phi_pos')`` give the r, theta, phi coordinates relative to the halo
      center (theta is azimuth and phi is altitude)
    - ``('gas', 'theta_velocity_corrected')`` and ``('gas', 'phi_velocity_corrected')`` give the two directions of the velocity tangential to the radial
      velocity, the azimuthal velocity (theta) and the altitudinal velocity (phi)
    - If you've passed the optional argument 'disk_relative = True', then there are some additional fields:
        - ``('gas', 'x_disk')``, ``('gas', 'y_disk')``, and ``('gas', 'z_disk')`` are the x, y, z positions of each gas cell relative to the galaxy disk
          (z is minor axis, x and y are in disk plane)
        - ``('gas', 'vx_disk')``, ``('gas', 'vy_disk')``, and ``('gas', 'vz_disk')`` are the disk-relative x, y, z velocities
        - ``('gas', 'phi_pos_disk')`` and ``('gas', 'theta_pos_disk')`` are the altitude and azimuth positions of each gas cell relative to the galaxy disk
          (phi = 0 and phi = pi are the north and south poles of the galaxy disk, theta goes from 0 to 2pi around in the plane of the disk)
        - ``('gas', 'vphi_disk')`` and ``('gas', 'vtheta_disk')`` are the two directions of the tangential velocity relative to the galaxy disk
* Particle filtering: ``foggie_load()`` defines particles as either ``'stars'`` or ``'dm'`` (dark matter), and further splits up stars into
  ``'young_stars'`` (ages less than 10 Myr), ``'old_stars'`` (ages greater than 10 Myr), and ``'young_stars8'`` (ages less than 100 Myr)
* ``foggie_load()`` also returns ``refine_box`` in addition to ``ds``. ``refine_box`` is a yt data object that contains only the data inside the high-resolution halo track box.

