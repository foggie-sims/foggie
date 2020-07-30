import numpy
from numpy import *
for halo in ['8508', '5016', '2392', '5036']:
    DDmin = 200
    DDmax = 1200
    N_split = 20.

    sf = open('/nobackupp2/rcsimons/foggie/submit_scripts/Z_mom/submit_%s_%i_%i_cenmass_qsub.sh'%(halo, DDmin, DDmax), 'w+')
    for DD in arange(DDmin, DDmax, N_split):
        snap_name = 'DD%.4i_DD%.4i'%(DD, DD + N_split)
        sim_snap_name = snap_name + '_' + halo+'_cenmass'

        qsub_fname = '%s.qsub'%(sim_snap_name)

        qf = open('/nobackupp2/rcsimons/foggie/submit_scripts/Z_mom/%s'%qsub_fname, 'w+')
        
        qf.write('#PBS -S /bin/bash\n')
        qf.write('#PBS -l select=1:ncpus=16:model=san\n')
        qf.write('#PBS -l walltime=8:00:00\n')
        qf.write('#PBS -q normal\n')
        qf.write('#PBS -N %s\n'%sim_snap_name)
        qf.write('#PBS -M rsimons@stsci.edu\n')
        qf.write('#PBS -m abe\n')
        qf.write('#PBS -o ./outfiles/%s_pbs.out\n'%sim_snap_name)
        qf.write('#PBS -e ./outfiles/%s_pbs.err\n'%sim_snap_name)
        qf.write('#PBS -V\n')
        qf.write('#PBS -W group_list=s1938\n\n\n\n')  

        for DDi in arange(DD, DD + N_split):
            qf.write('python /nobackupp2/rcsimons/git/foggie_local/measure_Z_central.py --output DD%.4i --halo  %s  > /nobackupp2/rcsimons/foggie_momentum/submit_scripts/Z_mom/outfiles/%s.err > /nobackupp2/rcsimons/foggie_momentum/submit_scripts/Z_mom/outfiles/%s.out\n'%(DDi, halo, sim_snap_name, sim_snap_name))
        qf.close()

        sf.write('qsub %s\n'%qsub_fname)


    sf.close()