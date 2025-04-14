Halo Info Files
===============

This document describes the different file types that can be found in the ``halo_infos`` folder
and how to use them. Scroll down to the bottom for the process to create all halo info files.

The ``halo_infos`` folder is first broken into subfolders by halo ID, then by run type. For example, the
halo ID for Tempest is ``008508``, and the run type could be ``nref11c_nref9f``.

halo_c_v
--------

**Description:**

These files are ASCII data files where the columns are:
::

    redshift        snapshot name       xc      yc      zc      xv      yv      zv

where ``redshift`` gives the redshift of the snapshot, ``snapshot name`` is the file name of the
snapshot (e.g. RD0036, DD1534), ``xc, yc, zc`` gives the x, y, z coordinates of the center of the halo
as the peak of the DM density distribution, as returned by ``get_halo_center.py``, and ``vx, vy, vz`` are
the x, y, z velocities of the center of the halo, as determined by taking a mass-weighted average
of the gas velocity in a sphere of radius 10 kpc centered on the halo center.

**Units:**

``xc, yc, zc`` are given in physical kpc. ``xv, yv, zv`` are given in km/s.

**Exists for:**

* 8508/nref11c_nref9f snapshots DD0044-DD2427, RD0012-RD0042
* 8508/nref11n_nref10f snapshots DD0044-DD1630
* 5016/nref11c_nref9f snapshots DD0150-DD2520, RD0020-RD0042
* 5036/nref11c_nref9f snapshots DD0139-DD2520, RD0015-RD0042
* 4123/nref11c_nref9f snapshots DD0169-DD2520, RD0016-RD0042
* 2392/nref11c_nref9f snapshots DD0138-DD2520, RD0019-RD0042
* 2878/nref11c_nref9f snapshots DD0140-DD2199, RD0016-RD0040

* 8508/high_feedback_restart snapshots DD1728-DD1882, RD0036-RD0037
* 8508/feedback_return snapshots DD1747-DD2427, RD0036-RD0040
* 8508/low_feedback_06 snapshots DD1621-DD2427, RD0034-RD0042
* 8508/low_feedback_07 snapshots DD1621-DD2427, RD0034-RD0042

* 8508/feedback-10-track snapshots DD0139-DD2520, RD0012-RD0042

**Created by:**

utils/get_halo_c_v_parallel.py

**Author:**

Cassi


halo_cen_smoothed
-----------------

**Description:**

These files are ASCII data files where the columns are:
::

    snap       redshift       time      xc      yc      zc

where ``snap`` is the file name of the snapshot (e.g., RD0036, DD1534), ``redshift`` gives the redshift
of the snapshot, ``time`` is the time of the snapshot, and ``xc, yc, zc`` gives the x, y, z
coordinates of the center of a smoothed path of the halo. This smoothed path is created by removing
sudden jumps from the path that occur when there is a merger and the halo center finder jumps between
the two galaxies back and forth, and then smoothing the result.

**Units:**

time is given in Myr, xc, yc, zc are given in physical kpc

**Exists for:**

* 8508/nref11c_nref9f snapshots DD0044-DD2427, RD0012-RD0042
* 5016/nref11c_nref9f snapshots DD0150-DD2520, RD0020-RD0042
* 5036/nref11c_nref9f snapshots DD0139-DD2520, RD0015-RD0042
* 4123/nref11c_nref9f snapshots DD0169-DD2520, RD0016-RD0042
* 2392/nref11c_nref9f snapshots DD0138-DD2199, RD0019-RD0042
* 2878/nref11c_nref9f snapshots DD0140-DD2199, RD0016-RD0040

* 8508/feedback-10-track snapshots DD0139-DD2520, RD0012-RD0042

**Created by:**

utils/smooth_halo_center.py

**Author:**

Cassi


angmom_table.hdf5
-----------------

**Description:**

These files are data files where the columns are:
::

    snap       redshift       time      Lx      Ly      Lz

where ``snap`` is the file name of the snapshot (e.g., RD0036, DD1534), ``redshift`` gives the redshift
of the snapshot, ``time`` is the time of the snapshot, and ``Lx, Ly, Lz`` gives the (normalized) x, y, z
components of the disk's angular momentum. The angular momentum vector direction is
calculated from stars with ages less than 100 Myr within 15 kpc of the (true, not smoothed) center of the halo.

**Units:**

time is given in Myr, everything else is dimensionless

**Exists for:**

* 8508/nref11c_nref9f snapshots DD0044-DD2427, RD0012-RD0042
* 8508/low_feedback_06 snapshots DD1621-DD2427
* 8508/feedback-10-track snapshots DD0139-DD2520, RD0012-RD0042
* 5016/nref11c_nref9f snapshots DD0150-DD2520, RD0020-RD0042
* 5036/nref11c_nref9f snapshots DD0139-DD2520, RD0015-RD0042
* 4123/nref11c_nref9f snapshots DD0169-DD2520, RD0016-RD0042
* 2392/nref11c_nref9f snapshots DD0138-DD2520, RD0019-RD0042
* 2878/nref11c_nref9f snapshots DD0140-DD2199, RD0016-RD0040

**Created by:**

utils/save_ang_mom.py

**Author:**

Cassi


AM_direction_smoothed
---------------------

**Description:**

These files are ASCII data files where the columns are:
::

    snap       redshift       time      Lx      Ly      Lz

where ``snap`` is the file name of the snapshot (e.g., RD0036, DD1534), ``redshift`` gives the redshift
of the snapshot, ``time`` is the time of the snapshot, and ``Lx, Ly, Lz`` gives the (normalized) x, y, z
components of the disk's angular momentum smoothed over time. This smoothed path is created by removing
sudden jumps from the path (saved in ``angmom_table.hdf5``), and then smoothing the result. The angular momentum vector direction is
calculated from stars with ages less than 100 Myr within 15 kpc of the (true, not smoothed) center of the halo.

**Units:**

time is given in Myr, everything else is dimensionless

**Exists for:**

* 8508/nref11c_nref9f snapshots DD0044-DD2427, RD0012-RD0042
* 8508/low_feedback_06 snapshots DD1621-DD2427
* 8508/feedback-10-track snapshots DD0139-DD2520, RD0012-RD0042
* 5016/nref11c_nref9f snapshots DD0150-DD2520, RD0020-RD0042
* 5036/nref11c_nref9f snapshots DD0139-DD2520, RD0015-RD0042
* 4123/nref11c_nref9f snapshots DD0169-DD2520, RD0016-RD0042
* 2392/nref11c_nref9f snapshots DD0138-DD2520, RD0019-RD0042
* 2878/nref11c_nref9f snapshots DD0140-DD2199, RD0016-RD0040

**Created by:**

utils/smooth_halo_center.py

**Author:**

Cassi


masses_z-gtr-2.hdf5 and masses_z-less-2.hdf5
--------------------------------------------

**Description:**

These files are data files (that can be read with ``astropy.table``) that give profiles of mass enclosed
versus radius for a number of different snapshots for a given run, all saved in the same file.
The columns are:
::

    redshift   snapshot   radius   total_mass   dm_mass   stars_mass   young_stars_mass   old_stars_mass   gas_mass
    gas_metal_mass   gas_H_mass   gas_HI_mass   gas_HII_mass   gas_CII_mass  gas_CIII_mass   gas_CIV_mass   gas_OVI_mass
    gas_OVII_mass   gas_MgII_mass   gas_SiII_mass   gas_SiIII_mass   gas_SiIV_mass   gas_NeVIII_mass

where ``redshift`` gives the redshift of the snapshot, ``snapshot`` gives the name of the snapshot
(e.g. RD0036, DD1534), ``radius`` gives the radius at which the mass enclosed within that radius is
calculated, ``total_mass`` gives the mass of dark matter, stars, and gas enclosed within the
corresponding radius, ``dm_mass`` gives the mass of just dark matter enclosed within the corresponding
radius, ``stars_mass`` gives the mass of just stars enclosed within the corresponding radius,
``young_stars_mass`` is the mass of star particles with ages less than 1e7 yrs, ``old_stars_mass`` is the mass
of star particles with ages greater than 1e7 yrs,
``gas_mass`` gives the mass of just gas, and ``gas_metal_mass`` gives the mass of metals in the gas phase
enclosed within the corresponding radius. The rest of the gas masses
after that give the mass of several ions in the gas. There are 250 radii
at which the enclosed mass is calculated for each snapshot, from ``0.01*refine_width`` out to
``5*refine_width``, log-spaced.

The files are split into snapshots with redshift greater than and less than 2, to avoid github's
file size limit. This division happens between DD0486 and DD0487 and between RD0020 and RD0021.

**Units:**

``radius`` is given in physical kpc. All masses are given in Msun.

**Exists for:**

* 8508/nref11c_nref9f snapshots DD0044-DD2427, RD0012-RD0042
* 5016/nref11c_nref9f snapshots DD0150-DD2520, RD0020-RD0042
* 5036/nref11c_nref9f snapshots DD0139-DD2520, RD0015-RD0042
* 4123/nref11c_nref9f snapshots DD0169-DD2520, RD0016-RD0042
* 2392/nref11c_nref9f snapshots DD0138-DD2520, RD0019-RD0042
* 2878/nref11c_nref9f snapshots DD0140-DD2199, RD0016-RD0040

* 8508/high_feedback_restart snapshots DD1728-DD1882, RD0036-RD0037
* 8508/feedback_return snapshots DD1747-DD2427, RD0036-RD0040
* 8508/low_feedback_06 snapshots DD1621-DD2427, RD0034-RD0042
* 8508/low_feedback_07 snapshots DD1621-DD2427, RD0034-RD0042
* 8508/feedback-10-track snapshots DD0139-DD2520, RD0012-RD0042

**Created by:**

``utils/get_mass_profile.py``, which will output one ``snapshot_masses.hdf5`` per snapshot that can
later be combined. The ``masses_z-gtr-2.hdf5`` and ``masses_z-less-2.hdf5`` files are the combined versions.

**How to use:**

::

    from astropy.table import Table
    masses = Table.read('/path/to/table/masses_z-less-2.hdf5', path='all_data')

    # To plot the mass enclosed profiles for a specific snapshot, e.g. RD0042:
    import matplotlib.pyplot as plt
    plt.plot(masses['radius'][masses['snapshot']=='RD0042'], masses['total_mass'][masses['snapshot']=='RD0042'])


**Author:**

Cassi


sfr
---

**Description:**

This file gives the star formation rate within a 20 kpc sphere centered on the halo center for
each snapshot. It is an ascii file, where the columns are:
::

    snapshot    redshift    SFR (Msun/yr)


**Units:**

SFR is given in Msun/yr

**Exists for:**

* 8508/nref11c_nref9f snapshots DD0044-DD2427, RD0012-RD0042
* 5016/nref11c_nref9f snapshots DD0150-DD2520, RD0020-RD0042
* 5036/nref11c_nref9f snapshots DD0139-DD2520, RD0015-RD0042
* 4123/nref11c_nref9f snapshots DD0169-DD2520, RD0016-RD0042
* 2392/nref11c_nref9f snapshots DD0138-DD2520, RD0019-RD0042
* 2878/nref11c_nref9f snapshots DD0140-DD2199, RD0016-RD0040

* 8508/high_feedback_restart snapshots DD1728-DD1882, RD0036-RD0037
* 8508/feedback_return snapshots DD1747-DD2427, RD0036-RD0040
* 8508/low_feedback_06 snapshots DD1621-DD2427, RD0034-RD0042
* 8508/low_feedback_07 snapshots DD1621-DD2427, RD0034-RD0042
* 8508/feedback-10-track snapshots DD0139-DD2520, RD0012-RD0042

**Created by:**

``utils/get_mass_profile.py``, which will output one ``snapshot_masses.hdf5`` per snapshot that can later
be combined and extract the SFR column. The ``sfr`` file is a combined file with only the ``snapshot``, ``redshift``,
and ``sfr`` columns for all snapshots, using only the sfr within a 20 kpc sphere.

**How to use:**

::
    import numpy as np
    snapshots, redshifts, SFRs = np.loadtxt('/path/to/table/sfr', unpack=True, usecols=[0,1,2], skiprows=1)

    # To plot the SFR vs redshift:
    import matplotlib.pyplot as plt
    plt.plot(redshifts, SFRs)


**Author:**

Cassi


rvir_masses.hdf5
----------------

**Description:**

An hdf5 catalog listing the virial radius and masses for each snapshot. Can be read with ``astropy.table``, in the same way as the ``masses_z-less-2.hdf5`` and ``masses_z-gtr-2.hdf5`` catalogs above.

The columns are:
::

    redshift   snapshot   radius   total_mass   dm_mass   stars_mass   young_stars_mass   old_stars_mass   gas_mass
    gas_metal_mass   gas_H_mass   gas_HI_mass   gas_HII_mass   gas_CII_mass  gas_CIII_mass   gas_CIV_mass   gas_OVI_mass
    gas_OVII_mass   gas_MgII_mass   gas_SiII_mass   gas_SiIII_mass   gas_SiIV_mass   gas_NeVIII_mass


where ``radius`` is the virial radius at the given snapshot and the ``masses`` are the masses are of the halo inside rvir. The star particle types are defined as outlined in ``masses_z-gtr-2.hdf5`` and ``masses_z-less-2.hdf5`` above.

**Units:**

``radius`` is given in physical kpc. All masses are given in Msun.

**Exists for:**

* 8508/nref11c_nref9f snapshots DD0044-DD2427, RD0012-RD0042
* 5016/nref11c_nref9f snapshots DD0150-DD2520, RD0020-RD0042
* 5036/nref11c_nref9f snapshots DD0139-DD2520, RD0015-RD0042
* 4123/nref11c_nref9f snapshots DD0169-DD2520, RD0016-RD0042
* 2392/nref11c_nref9f snapshots DD0138-DD2520, RD0019-RD0042
* 2878/nref11c_nref9f snapshots DD0140-DD2199, RD0016-RD0040

* 8508/high_feedback_restart snapshots DD1728-DD1882, RD0036-RD0037
* 8508/feedback_return snapshots DD1747-DD2427, RD0036-RD0040
* 8508/low_feedback_06 snapshots DD1621-DD2427, RD0034-RD0042
* 8508/low_feedback_07 snapshots DD1621-DD2427, RD0034-RD0042
* 8508/feedback-10-track snapshots DD0139-DD2520, RD0012-RD0042

**Created by:**

``utils/get_rvir.py``

Use as e.g., ``python get_rvir.py --halo 8508 --use_catalog_profile``

**Author:**

Raymond (04/23/20), using Cassi's mass profile catalogs ``masses_z-gtr-2.hdf5`` and ``masses_z-less-2.hdf5``


Process to make halo info files for a fresh run where they do not yet exist:
----------------------------------------------------------------------------

1.  Run ``get_halo_c_v_parallel.py`` on all the snapshots of the run. Example:
    ::

        python foggie/utils/get_halo_c_v_parallel.py --halo 4123 --run nref11c_nref9f --output DD1800-DD1900,RD0030-RD0040 --nproc 20

    This is pretty quick and can be run 20 at a time on any node. If there was already a ``halo_c_v file`` for this run and you're just
    updating it, you will need to copy-paste into ``halo_c_v`` the new rows from the ``halo_c_v_DD1800_RD0040`` (in the above example) file.

2.  Run ``get_mass_profile.py`` on all the snapshots of the run. Make sure that there are entries in ``halo_c_v`` for the outputs you want
    to run on (i.e., don't skip step #1!). Example:
    ::

        python foggie/utils/get_mass_profile.py --halo 4123 --run nref11c_nref9f --output DD1800-DD1900,RD0030-RD0040 --nproc 2
    
    This is very slow and requires a lot of RAM. Run on either an LDAN node or endeavour with at least 400GB of memory per snapshot.
    So if running 2 at a time in parallel, need 800GB of RAM. This will make one new ``DDXXXX_masses.hdf5`` file per snapshot.

3.  Run ``save_ang_mom.py`` on all the snapshots of the run. Make sure there are entries in ``halo_c_v`` for the outputs you want
    (i.e., don't skip step #1!). This can be run concurrently with step #2. Example:
    ::

        python foggie/utils/save_ang_mom.py --halo 4123 --run nref11c_nref9f --output DD1800-DD1900,RD0030-RD0040 --nproc 3

    This is quicker and less memory-intensive than the mass profiles. Requires ~50GB of memory per snapshot, so you can run 2-3
    in parallel per haswell node. This produces a single file called ``angmom_table.hdf5`` that has the angular momentum direction
    of each snapshot requested. To combine this with an existing ``angmom_table.hdf5``, do:
    ::

        from astropy.table import Table, vstack
        angmom1 = Table.read('angmom_table_old.hdf5', path='all_data')
        angmom2 = Table.read('angmom_table_new.hdf5', path='all_data')
        combined_angmom = vstack([angmom1, angmom2])
        combined_angmom.write('angmom_table.hdf5', path='all_data', serialize_meta=True, overwrite=True)

4.  Once steps 1 and 2 have been completed, combine the individual snapshot mass profile files using ``combine_halo_infos.py``. Example:
    ::

        python foggie/utils/combine_halo_infos.py --halo 4123 --run nref11c_nref9f

    You don't need to specify the outputs here because it will use everything in a directory named ``masses_halo_004123/nref11c_nref9f``
    (which is the directory ``get_mass_profile.py`` puts all the files in in step 2). This creates two files: ``masses_z-gtr-2.hdf5`` and
    ``masses_z-less-2.hdf5``. Note that if you already have these files and you simply want to update them, THE CODE WILL OVERWRITE THE
    FILES with only the snapshots in your ``masses_halo_004123/nref11c_nref9f`` directory. So don't delete the individual snapshot files
    until you're sure you have all the outputs you're ever going to want for that run!

    This step will also produce a file called sfr, which gives the mass of all stars younger than 10 Myr within 20 kpc of the center of the
    galaxy, divided by 10 Myr, for each snapshot.

5.  Once step 4 has been completed, calculate the virial mass and radius with get_rvir.py. Example:
    ::

        python foggie/utils/get_rvir.py --halo 4123 --run nref11c_nref9f --use_catalog_profile

    This will calculate the virial mass and radius of all mass profiles in the combined ``masses_z-gtr-2.hdf5`` and ``masses_z-less-2.hdf5`` files,
    and save them into a file called ``rvir_masses.hdf5``.

6.  Once steps 1 and 3 have been completed, you can optionally smooth the halo center locations and the angular momentum directions using
    ``smooth_halo_catalogs.py``. This is useful for making videos that are less jittery and don't go haywire during mergers. Example:
    ::

        python foggie/utils/smooth_halo_catalogs.py --halo 4123 --run nref11c_nref9f

    You don't need to specify the outputs because it will use everything in the ``halo_c_v`` and ``angmom_table.hdf5`` files. This
    produces two files: ``halo_cen_smoothed`` and ``AM_direction_smoothed``.


After following all these steps, you should have these files:
* ``halo_c_v``
* ``halo_cen_smoothed``
* ``angmom_table.hdf5``
* ``AM_direction_smoothed``
* ``masses_z-gtr-2.hdf5``
* ``masses_z-less-2.hdf5``
* ``rvir_masses.hdf5``
* ``sfr``

