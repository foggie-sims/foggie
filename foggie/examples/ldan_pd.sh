#PBS -S /bin/bash
#PBS -N powderdayrun_T11cdebug
#PBS -l select=1:ncpus=16:mem=750GB
#PBS -l walltime=72:00:00
#PBS -q ldan
#PBS -j oe
#PBS -M acwright@jhu.edu
#PBS -o /nobackupp2/awright5/JHU/outdirs/
#PBS -m abe
#PBS -koed
#PBS -l site=needed=/nobackupp2
#PBS -W group_list=s1698

cd /nobackupp2/awright5

export PATH=/nobackupp2/awright5/miniconda3/bin:$PATH
export PYTHONPATH=/nobackupp2/awright5/miniconda3/bin/python3.8:$PATH
export PYTHONPATH=/nobackupp2/awright5/foggie:$PATH

python /nobackupp2/awright5/writepdmodelfiles.py --dstg hidustim

export SPS_HOME=/nobackupp2/awright5/fsps/
export PATH=/nobackupp2/awright5/bin:$PATH
module load comp-intel/2018.3.222
module load mpi-hpe/mpt.2.21
module load hdf5/1.8.18_serial

source activate pd4env

pd_front_end.py /nobackupp2/awright5/JHU parameters_master_enzo parameters_model_8508_nref11c_nref9f_RD0042

conda deactivate
