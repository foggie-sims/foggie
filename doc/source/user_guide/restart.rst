Restarting an Existing FOGGIE Run
=================================

To restart a FOGGIE halo with whatever changes you would like:

1.  Make sure to have the ``enzo-foggie`` branch of enzo installed. See the page `Compiling enzo-foggie on Pleiades <enzo-foggie.html>`_ for the steps.

2.  Make a new directory in your /nobackup/<username> directory for your new run.

3.  In your new run directory, copy over these files:
	-   The snapshot directory that you want to restart from.
	    e.g., if I'm restarting Tempest from DD1970, I would copy
	    ``/nobackup/mpeeples/halo_008508/nref11c_nref9f/DD1970``
	    into my new directory
	-   ``OutputLog`` from the original run directory
	    e.g., for the above example,
	    ``/nobackup/mpeeples/halo_008508/nref11c_nref9f/OutputLog``
	-   ``halo_track`` from the original run directory
	    e.g., for the above example,
	    ``/nobackup/mpeeples/halo_008508/nref11c_nref9f/halo_track``
	-   ``np`` from the directory where you installed ``enzo_foggie``
	    e.g., ``/path/to/enzo_foggie/bin/np``
	-   ``simrun.pl`` from ``/nobackup/clochhaa/reruns/simrun.pl``
	-   ``RunScript.sh`` from ``/nobackup/clochhaa/reruns/RunScript.sh``

4.  If you want backups on lou, then from a pfe, ssh to lou and create a new directory where you want to backup new simulation outputs e.g.:
    ::

	    ssh lou
	    mkdir test_restart
	    logout

    The final step will bring you back to the pfe.

5.  Modify these files:
	-   ``simrun.pl``:
	    Put in your email address on the line that starts with ``$email_address``.
	    Put in the path to your enzo install location on the line that starts with ``$enzo_executable``.
	    Note that the actual executable is in the ``bin`` directory of the enzo install.
	    Look for the line that starts ``$command_line = "shiftc --create-tar $directory``.
	    On this line, change the path after ``lou:`` to the path to the new directory on lou you just
	    made for backup. ``simrun.pl`` will automatically back up new simulation outputs to this
	    directory as they're made while the code runs.
	-   ``OutputLog``:
	    Delete all lines after whichever output you're restarting from. In the above example, I
	    would delete everything after DD1970, so that the last line starts with
	    ``DATASET WRITTEN ./DD1970/DD1970``.
	-   ``RunScript.sh``:
	    Here is an example RunScript.sh file:

		:: 

			#!/bin/bash

			#PBS -N <JOB_NAME>
			#PBS -W group_list=s2358 
			#PBS -l select=1:ncpus=64:mpiprocs=64:model=mil_ait
			#PBS -l walltime=120:00:00
			#PBS -q long
			#PBS -j oe
			#PBS -m abe
			#set output and error directories
			#PBS -e pbs_error.txt
			#PBS -o pbs_output.txt
			#PBS -koed

			module purge
			module use /nasa/modulefiles/testing
			module load comp-intel/2020.4.304
			module load hdf5/1.8.18_serial
			module load mpi-hpe/mpt

			export HDF5_DISABLE_VERSION_CHECK=1

			export PATH="/u/scicon/tools/bin/:$PATH"
			export LD_LIBRARY_PATH="/PATH/TO/grackle/build/lib64":$LD_LIBRARY_PATH

			cd $PBS_O_WORKDIR

			/u/<USERNAME>/memory_gauge.sh $PBS_JOBID > memory.$PBS_JOBID 2>&1 &


			./simrun.pl -mpi "mpiexec -np 64 /u/scicon/tools/bin/mbind.x -cs " -wall 432000 -jf "RunScript.sh"

			mv pbs_output.txt pbs_output_$PBS_JOBID.txt



6.  Modify the parameter file of the last output you're restarting from to include whatever it is you want to change. In the example above, this would be DD1970/DD1970
	-   If you want the new star formation and/or feedback stuff, check the page `Starting a Fresh FOGGIE Run from Initial Conditions <clean_from_ICs.html>`_.

	-   path to grackle file:
	    Look for a parameter called ``grackle_data_file``. Set it to the path to your particular
	    grackle install so that it points toward your ``CloudyData_UVB=HM2012_shielded.h5`` file
	    (should be in the ``input`` directory of your grackle install). See the notes on compiling ``enzo-foggie`` for installing grackle.


7.  If this is your first time doing a re-run, make sure this all works before you start changing things.
    If you have enzo running fine, then the last step is to change whichever parameter you want to explore!

	-   e.g., if I want to explore the impact of different strengths of mechanical feedback, I would want to
	    change the parameter ``MomentumMultiplier``.

8.  Submit your re-run to PBS by cd'ing into the directory and typing

    ::

	    qsub RunScript.sh

9.  Monitoring the run:
	-   You can cd into the directory of the run and type

            ::

                ./np


	    to get information about what the simulation is currently doing, like the timestep

	-   When a run stops, there will be a file called ``.message`` in the directory telling you why
	    it stopped. Could be because the run finished (yay!), could be because ``simrun`` calculated
	    there wasn't enough time to produce another output in your remaining walltime (in which
	    case it will automatically resubmit for you), could be because something went wrong.
	-   All output from running enzo gets piped into ``estd.out``. If there's a problem, look there
	    first. ``simrun`` automatically moves this file to ``estd_1.out``, ``estd_2.out``, etc., each time a
	    run stops so that you have a record of past runs.
	-   ``OutputLog`` will update each time the simulation makes new data.
	-   ``run.log`` will also give you some information. This is written by ``simrun``, and can tell you
	    if the simulation didn't make more data for a bad reason or because it was just short on
	    walltime.
	-   ``memory.JOBID.pbspl1.nas.nasa.gov`` file has information about how much memory is available
	    on each node that the simulation is running on, for load-balancing information. You
	    probably won't need to worry about this now that Jason has implemented a load-balancing
	    fix, but may be something to check if you're really stuck trying to figure out an error.

