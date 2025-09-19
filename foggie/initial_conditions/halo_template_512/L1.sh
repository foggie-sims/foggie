#!/bin/bash

#PBS -N L1-DM-512
#PBS -W group_list=s3128
#PBS -l select=1:ncpus=1:mpiprocs=1:model=has
#PBS -l walltime=2:00:00
#PBS -q devel 
#PBS -j oe
#PBS -m abe
#PBS -V
#set output and error directories
#PBS -e L1-DM-error.txt
#PBS -o L1-DM-output.txt

module load comp-intel/2020.4.304
module load mpi-hpe/mpt.2.23
module load hdf5/1.8.18_serial

export HDF5_DISABLE_VERSION_CHECK=1

export LD_LIBRARY_PATH="/u/jtumlins/installs/mpich-4.0.3/usr/local/lib":"/u/jtumlins/installs/mpich-4.0.3/usr/lib":$LD_LIBRARY_PATH
export PATH="/nobackupnfs1/jtumlins/anaconda3/bin:/u/scicon/tools/bin/:/u/jtumlins/installs/mpich-4.0.3/usr/local/bin:$PATH"

cd $PBS_O_WORKDIR
python /u/jtumlins/nobackup/foggie/foggie/initial_conditions/halo_template_512/script512.py --level=1 --gas='no' --halo_id=$HALO_ID 



