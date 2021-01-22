#!/bin/bash


# sample PBS script for running big enzo jobs on Pleiades; MSP, January 22, 2021
# https://www.nas.nasa.gov/hecc/support/kb/sample-pbs-script-for-pleiades_190.html
# is useful for much of this!

# name the run something useful.
# you can see what runs you have going with %qstat -u username
#PBS -N halo_005036_nref11c_nref9f

# charge to the correct FOGGIE group. s1938 is the main FOGGIE account;
# s1698 is Jason's Roman SIT account (anything Milky Way related, stellar halos)
#PBS -W group_list=s1938

# for memory-intensive runs, we cannot use all of cpus available on each node
# since hyperthreading is allowed, we specify both ncpus and mpiprocs
# in this example, we want 512 total cpus so 47x11 > 512.
# haswell is cheaper than broadwell, but has the same amout of memory per node,
# and so is usually preferable
#PBS -l select=47:ncpus=11:mpiprocs=11:model=has

# this is the max time allowed on the long queue. if using debug, needs to be <2hr
#PBS -l walltime=120:00:00
#PBS -j oe
#PBS -m abe
#set output and error directories
#PBS -e pbs_error.txt
#PBS -o pbs_output.txt


# these modules need to be up to date! and match what enzo was compiled with!
# https://www.nas.nasa.gov/hecc/support/kb/using-software-modules_115.html
module load comp-intel/2018.3.222
module load mpi-hpe/mpt.2.21
module load hdf5/1.8.18_serial


# even so, this helps enzo not whine
export HDF5_DISABLE_VERSION_CHECK=1

cd $PBS_O_WORKDIR

# this is a useful memory tracker JT wrote; update your path :-)
/u/mpeeples/memory_gauge.sh $PBS_JOBID > memory.$PBS_JOBID 2>&1 &

#### the main workhorse
## simrun.pl is Britton Smith's useful code (https://github.com/brittonsmith/simrun)
# for automatically restarting enzo  after each output.
# Need to have "NumberOfOutputsBeforeExit = 1" set in your  enzo parameter file!!
# (here, "./DD2043/DD2043"). this helps with memory leakage
## in the mpi script, "-np 512" specifies the number of processors; check against request!
# -n11 is the number of processors per node; helps with load balancing
# other parts of this I don't really remember where came from, other than suggestions
# from NAS support staff ....
## for simrun.pl, the "-wall 432000" is the magic for automatically resubmitting the
# job if there isn't enough time to get to the next output. Here, 120 hours (the walltime
# requested above) equals 432000 seconds. This doesn't really work if these don't match!
./simrun.pl -mpi "mpiexec -np 512 /u/scicon/tools/bin/mbind.x -cs -n11 -gm" -exe "./enzo" -pf "./DD2043/DD2043" -wall 432000 -jf "RunScript.sh"
mv pbs_output.txt pbs_output_$PBS_JOBID.txt
