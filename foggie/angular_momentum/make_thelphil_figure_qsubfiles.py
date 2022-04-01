import numpy as np
import os
sf_all = open('/nobackupp2/rcsimons/foggie/submit_scripts/angmom/submit_all_figures.sh', 'w+')

cores = 10

halos = ['8508', '5016', '5036', '4123', '2392', '2878']
situations = ['inner', 'outflow', 'inflow', 'full']

for halo in halos[:1]:
    for situation in situations[:1]:

        qsub_fname = '%s_%s_thel-phil_plots.qsub'%(halo, situation)
        tempname = qsub_fname.strip('.qsub')
        qf = open('/nobackupp2/rcsimons/foggie/submit_scripts/angmom/%s'%qsub_fname, 'w+')

        qf.write('#PBS -S /bin/bash\n')
        qf.write('#PBS -l select=1:ncpus=%i:mpiprocs=%i:model=bro\n'%(cores, cores))
        qf.write('#PBS -l walltime=4:00:00\n')
        qf.write('#PBS -q normal\n')
        qf.write('#PBS -N %s\n'%tempname)
        qf.write('#PBS -M rsimons@stsci.edu\n')
        qf.write('#PBS -m abe\n')
        qf.write('#PBS -o ./outfiles/%s_pbs.out\n'%tempname)
        qf.write('#PBS -e ./outfiles/%s_pbs.err\n'%tempname)
        qf.write('#PBS -V\n')
        qf.write('#PBS -W group_list=s2361\n\n\n\n')
        qf.write('python /nobackupp2/rcsimons/git/foggie/foggie/angular_momentum/thel_phil.py --protect --cores %i --halo  %s --situation %s > /nobackupp2/rcsimons/foggie/submit_scripts/angmom/outfiles/%s.err > /nobackupp2/rcsimons/foggie/submit_scripts/angmom/outfiles/%s.out\n'%(cores, halo, situation, tempname, tempname))
        #qf.write('cd /nobackupp2/rcsimons/foggie/angular_momentum/figures/thel_phil\n')
        #qf.write('tar cvf %s_thel_phil.tar %s\n'%(halo, halo))
        qf.close()
        sf_all.write('qsub %s\n'%qsub_fname)

sf_all.close()
