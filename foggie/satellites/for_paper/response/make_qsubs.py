import numpy
from numpy import *
for halo in ['8508']:#, '5016', '2392', '5036']:
    DDmin = 392
    DDmax = 487
    N_split = 1.

    sf = open('/nobackupp2/rcsimons/foggie/submit_scripts/DM_orbit/submit_%s_%i_%i_cenmass_qsub.sh'%(halo, DDmin, DDmax), 'w+')
    for DD in arange(DDmin, DDmax, N_split):
        snap_name = 'DD%.4i_DD%.4i'%(DD, DD + N_split)
        sim_snap_name = snap_name + '_' + halo+'_dm_orbit'

        qsub_fname = '%s.qsub'%(sim_snap_name)

        qf = open('/nobackupp2/rcsimons/foggie/submit_scripts/DM_orbit/%s'%qsub_fname, 'w+')
        
        qf.write('#PBS -S /bin/bash\n')
        qf.write('#PBS -l select=1:ncpus=16:model=san\n')
        qf.write('#PBS -l walltime=1:00:00\n')
        qf.write('#PBS -q normal\n')
        qf.write('#PBS -N %s\n'%sim_snap_name)
        qf.write('#PBS -M rsimons@stsci.edu\n')
        qf.write('#PBS -m abe\n')
        qf.write('#PBS -o ./outfiles/%s_pbs.out\n'%sim_snap_name)
        qf.write('#PBS -e ./outfiles/%s_pbs.err\n'%sim_snap_name)
        qf.write('#PBS -V\n')
        qf.write('#PBS -W group_list=s1938\n\n\n\n')  

        qf.write('python /nobackupp2/rcsimons/git/foggie/foggie/satellites/for_paper/response/track_DM_orbit_particles.py --output DD%.4i --halo  %s  > /nobackupp2/rcsimons/foggie/submit_scripts/DM_orbit/outfiles/%s.err > /nobackupp2/rcsimons/foggie/submit_scripts/DM_orbit/outfiles/%s.out\n'%(DDi, halo, sim_snap_name, sim_snap_name))
        qf.close()

        sf.write('qsub %s\n'%qsub_fname)


    sf.close()