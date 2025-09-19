#!/bin/bash

#PBS -N haloXXXX-LX-gas
#PBS -W group_list=s3128 
#PBS -l select=4:ncpus=16:mpiprocs=16:model=has 
#PBS -l walltime=120:00:00
#PBS -j oe
#PBS -m abe
#PBS -V
#set output and error directories
#PBS -e pbs_error.txt
#PBS -o pbs_output.txt

module load comp-intel/2020.4.304
module load mpi-hpe/mpt.2.23
module load hdf5/1.8.18_serial

export HDF5_DISABLE_VERSION_CHECK=1

export LD_LIBRARY_PATH="/u/jtumlins/installs/mpich-4.0.3/usr/local/lib":"/u/jtumlins/installs/mpich-4.0.3/usr/lib":"/u/jtumlins/grackle/grackle-3.3.1-dev/build/lib64":$LD_LIBRARY_PATH
export PATH="/nobackup/jtumlins/anaconda3/bin:/u/scicon/tools/bin/:/u/jtumlins/installs/mpich-4.0.3/usr/local/bin:$PATH"


cd $PBS_O_WORKDIR
/u/jtumlins/installs/memory_gauge.sh $PBS_JOBID > memory.$PBS_JOBID 2>&1 &
./simrun.pl -mpi "mpiexec -np 64 /u/scicon/tools/bin/mbind.x -cs " -wall 432000 -email "tumlinson@stsci.edu" -pf "25Mpc_DM_512-LX-gas.enzo" -jf "RunScript.sh"
mv pbs_output.txt pbs_output_$PBS_JOBID.txt



