#PBS -N drive 
#PBS -W group_list=s1698
#PBS -l select=1:ncpus=20:mpiprocs=20:model=ldan
#PBS -l walltime=10:00:00
#PBS -j oe
#PBS -m abe
#PBS -V
#PBS -e pbs_error.txt
#PBS -o pbs_output.txt

echo $PBS_O_WORKDIR
cd $PBS_O_WORKDIR

module load mpi-sgi/mpt.2.15r20
module load comp-intel/2016.2.181
module load hdf5/1.8.18_serial

/pleiades/u/jtumlins/anaconda3/bin/python /nobackup/jtumlins/halo_008508/nref11c_nref9f/pleiades_plots/drive.py --card='DD0[456789]??/DD????' 
/pleiades/u/jtumlins/anaconda3/bin/python /nobackup/jtumlins/halo_008508/nref11c_nref9f/pleiades_plots/drive.py --card='DD1[01234]??/DD????' 
