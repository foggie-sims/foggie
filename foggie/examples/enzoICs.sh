#!/bin/bash
#PBS -N sphere1-L1
#PBS -W group_list=s1698
#PBS -l select=1:ncpus=16:mpiprocs=16:model=has
#PBS -l walltime=2:00:00
#PBS -j oe
#PBS -m abe
#PBS -V
#set output and error directories
#PBS -e pbs_error.txt
#PBS -o pbs_output.txt
module load comp-intel/2018.3.222
module load mpi-hpe/mpt.2.21
module load hdf5/1.8.18_serial
export HDF5_DISABLE_VERSION_CHECK=1 
cd $PBS_O_WORKDIR
/nobackupp2/awright5/installs/memory_gauge.sh $PBS_JOBID > memory.$PBS_JOBID 2>&1 &
./simrun.pl -mpi "mpiexec -np 16 " -pf 25Mpc_DM_256-L1.enzo -wall 720000 -email "acwright@jhu.edu" -jf "RunScript.sh"
mv pbs_output.txt pbs_output_$PBS_JOBID.txt
