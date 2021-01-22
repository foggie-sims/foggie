#PBS -S /bin/sh
#PBS -N sat_Mael_2102-2199
#PBS -l select=1:ncpus=16:model=ldan:mem=750GB
#PBS -l walltime=30:00:00
#PBS -q ldan
#PBS -j oe
#PBS -o /home5/clochhaa/FOGGIE/output_sat_Mael_2102-2199
#PBS -koed
#PBS -m abe
#PBS -V
#PBS -W group_list=s1938
#PBS -l site=needed=/home5+/nobackupp2+/nobackupp13

cd $PBS_O_WORKDIR
export PYTHONPATH="/home5/clochhaa/FOGGIE/foggie:${PYTHONPATH}"

/home5/clochhaa/miniconda3/bin/python3 /home5/clochhaa/FOGGIE/foggie/foggie/utils/get_satellite_positions.py --system pleiades_cassi --halo 5036 --nproc 4 --output DD2102-DD2199
