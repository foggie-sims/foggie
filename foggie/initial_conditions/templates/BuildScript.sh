#!/bin/bash
#
# IC generation job, rendered from templates/BuildScript.sh.
#
# This exists because enzo-mrp-music loads the parent box in yt and traces the
# halo's particles back to z = 99, which is far too heavy for a login node.
# The job runs `ic_pipeline build` on a compute node; that call generates the
# ICs and then submits the Enzo run itself.
#
#PBS -N __JOBNAME__
#PBS -W group_list=__GROUP__
#PBS -l select=__BUILD_SELECT__
#PBS -l walltime=__BUILD_WALLTIME__
#PBS -q __BUILD_QUEUE__
#PBS -j oe
#PBS -m abe
#PBS -V
#PBS -e pbs_build_error.txt
#PBS -o pbs_build_output.txt

module load comp-intel/2020.4.304
module load hdf5/1.8.18_serial

export HDF5_DISABLE_VERSION_CHECK=1
export PATH="/nobackup/jtumlins/anaconda3/bin:/u/scicon/tools/bin/:$PATH"

cd __HALO_DIR__

__BUILD_CMD__
