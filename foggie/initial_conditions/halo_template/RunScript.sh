#!/bin/bash

#PBS -N haloXXXX-LX
#PBS -W group_list=s1938
#PBS -l select=1:ncpus=16:mpiprocs=16:model=has
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -m abe
#PBS -V
#set output and error directories
#PBS -e pbs_error.txt
#PBS -o pbs_output.txt

module load comp-intel/2020.4.304
module load mpi-hpe/mpt.2.21
module load hdf5/1.8.18_serial

export HDF5_DISABLE_VERSION_CHECK=1

cd $PBS_O_WORKDIR
/u/jtumlins/installs/memory_gauge.sh $PBS_JOBID > memory.$PBS_JOBID 2>&1 &
/u/jtumlins/simrun.pl -mpi "mpiexec -np 16 /u/scicon/tools/bin/mbind.x -cs " -wall 432000 -email "tumlinson@stsci.edu" -pf "25Mpc_DM_256-LX.enzo" -jf "RunScript.sh"
mv pbs_output.txt pbs_output_$PBS_JOBID.txt



