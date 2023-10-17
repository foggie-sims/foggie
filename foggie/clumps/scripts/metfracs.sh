#PBS -S /bin/sh
#PBS -N endeavour_metfrac_5036
#PBS -l select=1:ncpus=16:mem=750GB:model=cas_end
#PBS -q e_vlong
#PBS -l walltime=600:00:00
#PBS -j oe
#PBS -o /home5/raugust4/WORK/Outputs/clumps/endeavour_metfrac_5036.out
#PBS -m abe
#PBS -V
#PBS -W group_list=s2358
#PBS -l site=needed=/home5+/nobackupp13
#PBS -M raugustin@stsci.edu
#PBS -e /home5/raugust4/WORK/Outputs/endeavour_metfrac_5036.err
#PBS -koed

source /home5/raugust4/.bashrc


/home5/raugust4/anaconda3/bin/python3 /home5/raugust4/offlinescripts/metfrac.py --shape 'shell' --level 1 --width 20 --halo=5036 > /home5/raugust4/endeavour_metfrac_5036_1.log
/home5/raugust4/anaconda3/bin/python3 /home5/raugust4/offlinescripts/metfrac.py --shape 'shell' --level 2 --width 20 --halo=5036 > /home5/raugust4/endeavour_metfrac_5036_2.log
/home5/raugust4/anaconda3/bin/python3 /home5/raugust4/offlinescripts/metfrac.py --shape 'shell' --level 3 --width 20 --halo=5036 > /home5/raugust4/endeavour_metfrac_5036_3.log
/home5/raugust4/anaconda3/bin/python3 /home5/raugust4/offlinescripts/metfrac.py --shape 'shell' --level 4 --width 20 --halo=5036 > /home5/raugust4/endeavour_metfrac_5036_4.log




#PBS -S /bin/sh
#PBS -N z2proj
#PBS -l select=1:ncpus=16:mem=750GB:model=cas_end
#PBS -q e_normal
#PBS -l walltime=5:00:00
#PBS -j oe
#PBS -o /home5/raugust4/WORK/Outputs/clumps/z2proj.out
#PBS -m abe
#PBS -V
#PBS -W group_list=s2358
#PBS -l site=needed=/home5+/nobackupp13
#PBS -M raugustin@stsci.edu
#PBS -e /home5/raugust4/WORK/Outputs/z2proj.err
#PBS -koed


source /home5/raugust4/.bashrc

/home5/raugust4/anaconda3/bin/python3 /home5/raugust4/offlinescripts/plotsforkeerthi.py --halo=2392 > /home5/raugust4/z2proj.log
/home5/raugust4/anaconda3/bin/python3 /home5/raugust4/offlinescripts/plotsforkeerthi.py --halo=2878 > /home5/raugust4/z2proj.log
/home5/raugust4/anaconda3/bin/python3 /home5/raugust4/offlinescripts/plotsforkeerthi.py --halo=4123 > /home5/raugust4/z2proj.log
/home5/raugust4/anaconda3/bin/python3 /home5/raugust4/offlinescripts/plotsforkeerthi.py --halo=5016 > /home5/raugust4/z2proj.log
/home5/raugust4/anaconda3/bin/python3 /home5/raugust4/offlinescripts/plotsforkeerthi.py --halo=5036 > /home5/raugust4/z2proj.log
/home5/raugust4/anaconda3/bin/python3 /home5/raugust4/offlinescripts/plotsforkeerthi.py --halo=8508 > /home5/raugust4/z2proj.log
