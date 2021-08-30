import numpy
from numpy import *
import numpy as np
import os
sf_all = open('/nobackupp2/rcsimons/foggie/submit_scripts/angmom/submit_all.sh', 'w+')



#for (halo, DDmin, DDmax, N_split) in [('8508', 487, 2427, 1),
#                             ('5016', 581, 2520, 5),
#                             ('5036', 581, 2520, 5),
#                             ('4123', 581, 2099, 5),
#                             ('2392', 581, 1699, 5),
#                             ('2878', 581,  929, 5)][2:4]:



for (halo, DDmin, DDmax, N_split) in [('8508', 44, 2427, 1),
                             ('5016', 251,  2520, 5),
                             ('5036', 276,  2520, 5),
                             ('4123', 356,  2199, 5),
                             ('2392', 511,  1699, 5),
                             ('2878', 401,  929, 5)][5:6]:

    count = 0
    sf_count = 0
    qf_count  = 0
    sf = open('/nobackupp2/rcsimons/foggie/submit_scripts/angmom/submit_%s_%i_%i_angmom_qsub_%i.sh'%(halo, DDmin, DDmax, sf_count), 'w+')    
    for DD in arange(DDmin, DDmax + N_split+1, N_split):
        if count > 200:
            sf.close()
            count = 0
            sf_count+=1
            sf = open('/nobackupp2/rcsimons/foggie/submit_scripts/angmom/submit_%s_%i_%i_angmom_qsub_%i.sh'%(halo, DDmin, DDmax, sf_count), 'w+')    
        #snap_name = 'DD%.4i_DD%.4i'%(DD, DD + N_split)
        DDuse = np.min((DD, DDmax))
        snap_name = 'DD%.4i'%(DDuse)
        sim_snap_name = snap_name + '_' + halo

        qsub_fname = '%s.qsub'%(sim_snap_name)

        qf = open('/nobackupp2/rcsimons/foggie/submit_scripts/angmom/%s'%qsub_fname, 'w+')
        
        qf.write('#PBS -S /bin/bash\n')
        qf.write('#PBS -l select=1:ncpus=28:model=cas_end\n')
        qf.write('#PBS -l walltime=8:00:00\n')
        qf.write('#PBS -q normal@pbspl4\n')
        qf.write('#PBS -N %s\n'%sim_snap_name)
        qf.write('#PBS -M rsimons@stsci.edu\n')
        qf.write('#PBS -m abe\n')
        qf.write('#PBS -o ./outfiles/%s_pbs.out\n'%sim_snap_name)
        qf.write('#PBS -e ./outfiles/%s_pbs.err\n'%sim_snap_name)
        qf.write('#PBS -V\n')
        qf.write('#PBS -W group_list=s2361\n\n\n\n')  
        if True:
            for DDi in arange(DDuse, DDuse + N_split, N_split):    
                if os.path.exists('/nobackupp2/rcsimons/foggie/angular_momentum/profiles/%s/Lprof_%s_DD%.4i.npz'%(halo, halo, DDi)): continue
                qf_count+=1
                qf.write('python /nobackupp2/rcsimons/git/foggie/foggie/angular_momentum/AM_distribution.py --output DD%.4i --halo  %s --system pleiades_raymond > /nobackupp2/rcsimons/foggie/submit_scripts/angmom/outfiles/%s.err > /nobackupp2/rcsimons/foggie/submit_scripts/angmom/outfiles/%s.out\n'%(DDi, halo, sim_snap_name, sim_snap_name))
        else:
            qf.write('python /nobackupp2/rcsimons/git/foggie/foggie/angular_momentum/AM_distribution.py --output DD%.4i --halo  %s --system pleiades_raymond > /nobackupp2/rcsimons/foggie/submit_scripts/angmom/outfiles/%s.err > /nobackupp2/rcsimons/foggie/submit_scripts/angmom/outfiles/%s.out\n'%(DDuse, halo, sim_snap_name, sim_snap_name))

        qf.close()
        if qf_count == 0: 
                 os.remove('/nobackupp2/rcsimons/foggie/submit_scripts/angmom/%s'%qsub_fname)
        else:
                 count+=1 
                 sf.write('qsub %s\n'%qsub_fname)
                 sf_all.write('qsub %s\n'%qsub_fname)

        qf_count = 0


    sf.close()

sf_all.close()
