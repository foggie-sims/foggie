#!/bin/bash
#
# Enzo run script, rendered from templates_512/RunScript.sh by the IC pipeline.
# Do not edit the copy inside a run directory -- edit this template, or the box
# config in foggie/initial_conditions/pipeline/config.py, and re-render.
#
#PBS -N __JOBNAME__
#PBS -W group_list=__GROUP__
#PBS -l select=__SELECT__
#PBS -l walltime=__WALLTIME__
__QUEUE_LINE__
#PBS -j oe
#PBS -m abe
#PBS -V
#set output and error directories
#PBS -e pbs_error.txt
#PBS -o pbs_output.txt

module load comp-intel/2020.4.304
module load hdf5/1.8.18_serial

export HDF5_DISABLE_VERSION_CHECK=1

export LD_LIBRARY_PATH="/u/jtumlins/installs/mpich-4.0.3/usr/local/lib":"/u/jtumlins/installs/mpich-4.0.3/usr/lib":"/u/jtumlins/grackle/grackle-3.3.1-dev/build/lib64":$LD_LIBRARY_PATH

export PATH="/nobackup/jtumlins/anaconda3/bin:/u/scicon/tools/bin/:/u/jtumlins/installs/mpich-4.0.3/usr/local/bin:$PATH"

cd $PBS_O_WORKDIR

/u/jtumlins/installs/memory_gauge.sh $PBS_JOBID > memory.$PBS_JOBID 2>&1 &

# Walltime this job was given, in seconds.  simrun.pl needs it to decide when to
# stop and resubmit; the gas transition below needs it to work out how much is
# left after the first leg.
SIMRUN_WALL=__SIMRUN_WALL__

# simrun.pl restarts Enzo from the last output, and re-qsubs this script when it
# runs out of walltime.  It returns here in three cases: the run finished, it
# resubmitted itself, or it died.  Wrapped in a function because a gas run calls
# it a second time, for the leg below the transition redshift, and the argument
# list must not drift between the two.
run_simrun () {
    ./simrun.pl -mpi "mpiexec -np __NRANKS__ /u/scicon/tools/bin/mbind.x -cs " \
                -wall "$1" \
                -email "__EMAIL__" \
                -exe "__ENZO_EXE__" \
                -pf "__PARAM_FILE__" \
                -jf "RunScript.sh"
}

run_simrun "$SIMRUN_WALL"

__PHASE_TRANSITION__
# Pipeline hook: advance this halo to the next refinement level.
#
# There is deliberately no "did it finish?" test here.  `advance` re-derives
# stage state from OutputLog and RunFinished, so it is a no-op in the two cases
# above where the run has not actually reached its final redshift dump.  That is
# safer than trying to distinguish simrun.pl's three exit paths in shell.
__PIPELINE_HOOK__

mv pbs_output.txt pbs_output_$PBS_JOBID.txt
