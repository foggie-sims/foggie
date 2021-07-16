import numpy
from numpy import *
import numpy as np

sf_all = open('/nobackupp2/rcsimons/foggie/submit_scripts/angmom/submit_all.sh', 'w+')

for (halo, DDmin, DDmax) in [('8508', 487, 2427),
                             ('2392', 581, 1660),
                             ('2878', 581, 929),
                             ('4123', 581, 2067),
                             ('5016', 581, 2520),
                             ('5036', 581, 2520)]:

    N_split = 50.

    sf = open('/nobackupp2/rcsimons/foggie/submit_scripts/angmom/submit_%s_%i_%i_angmom_qsub.sh'%(halo, DDmin, DDmax), 'w+')
    for DD in arange(DDmin, DDmax + N_split, N_split):
        #snap_name = 'DD%.4i_DD%.4i'%(DD, DD + N_split)
        snap_name = 'DD%.4i'%(DD)
        DDuse = np.min((DD, DDmax))
        sim_snap_name = snap_name + '_' + halo

        qsub_fname = '%s.qsub'%(sim_snap_name)

        qf = open('/nobackupp2/rcsimons/foggie/submit_scripts/angmom/%s'%qsub_fname, 'w+')
        
        qf.write('#PBS -S /bin/bash\n')
        qf.write('#PBS -l select=1:ncpus=16:model=has\n')
        qf.write('#PBS -l walltime=3:00:00\n')
        qf.write('#PBS -q normal\n')
        qf.write('#PBS -N %s\n'%sim_snap_name)
        qf.write('#PBS -M rsimons@stsci.edu\n')
        qf.write('#PBS -m abe\n')
        qf.write('#PBS -o ./outfiles/%s_pbs.out\n'%sim_snap_name)
        qf.write('#PBS -e ./outfiles/%s_pbs.err\n'%sim_snap_name)
        qf.write('#PBS -V\n')
        qf.write('#PBS -W group_list=s2361\n\n\n\n')  
        
        #for DDi in arange(DD, DD + N_split):    qf.write('python /nobackupp2/rcsimons/git/foggie/foggie/angular_momentum/AM_distribution.py --output DD%.4i --halo  %s --system pleiades_raymond > /nobackupp2/rcsimons/foggie/submit_scripts/angmom/outfiles/%s.err > /nobackupp2/rcsimons/foggie/submit_scripts/angmom/outfiles/%s.out\n'%(DDi, halo, sim_snap_name, sim_snap_name))
        qf.write('python /nobackupp2/rcsimons/git/foggie/foggie/angular_momentum/AM_distribution.py --output DD%.4i --halo  %s --system pleiades_raymond > /nobackupp2/rcsimons/foggie/submit_scripts/angmom/outfiles/%s.err > /nobackupp2/rcsimons/foggie/submit_scripts/angmom/outfiles/%s.out\n'%(DDuse, halo, sim_snap_name, sim_snap_name))
        

        qf.close()

        sf.write('qsub %s\n'%qsub_fname)
        sf_all.write('qsub %s\n'%qsub_fname)


    sf.close()

sf_all.close()
