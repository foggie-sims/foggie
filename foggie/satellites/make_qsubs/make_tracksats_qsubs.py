import numpy as np
from numpy import *
import glob
from glob import glob
import os


halos  = ['2392', '2878', '4123', '5016', '5036', '8508']
#halos = ['2878']

for halo in halos:
    print (halo)
    dir_name = '/nobackup/mpeeples/halo_00%s/nref11c_nref9f'%halo
    DD_directories = glob(dir_name + '/DD????')
    DDs = sort(array([int(DD_direct.split('/')[-1].strip('DD')) for DD_direct in DD_directories]))
    DDmin, DDmax = min(DDs), max(DDs)

    if halo == '8508': DDmax = 487
    else: DDmax = 581


    submit_dir = '/nobackupp2/rcsimons/foggie/submit_scripts/tracks'

    if not os.path.isdir(submit_dir): os.system('mkdir %s'%submit_dir)
    if not os.path.isdir(submit_dir+'/outfiles'): os.system('mkdir %s/outfiles'%submit_dir)
    

    sf = open('/nobackupp2/rcsimons/foggie/submit_scripts/tracks/submit_%s_%.4i_%.4i_tracksats.sh'%(halo, DDmin, DDmax), 'w+')
    splitn = 5
    for dmn in arange(DDmin, DDmax+splitn, splitn):
        snapname = '%s_%.4i_%.4i'%(halo, dmn, dmn+splitn)
        qsub_fname = 'track_%s_%.4i_%.4i.qsub'%(halo, dmn, dmn+splitn)        
        qf = open('/nobackupp2/rcsimons/foggie/submit_scripts/tracks/%s'%qsub_fname, 'w+')
        
        qf.write('#PBS -S /bin/bash\n')
        qf.write('#PBS -l select=1:ncpus=20:model=ivy\n')
        qf.write('#PBS -l walltime=6:00:00\n')
        qf.write('#PBS -q normal\n')
        qf.write('#PBS -N %s\n'%snapname)
        qf.write('#PBS -M rsimons@jhu.edu\n')
        qf.write('#PBS -m abe\n')
        qf.write('#PBS -o ./outfiles/%s_pbs.out\n'%snapname)
        qf.write('#PBS -e ./outfiles/%s_pbs.err\n'%snapname)
        qf.write('#PBS -V\n')
        qf.write('#PBS -W group_list=s1938\n\n\n\n')  

        qf.write('source /u/rcsimons/.bashrc\n')
        for DD in arange(dmn, dmn+splitn):
                snapname_temp = '%s_%.4i'%(halo, DD)
                qf.write('python /nobackupp2/rcsimons/git/foggie/foggie/satellites/track_satellites.py \
                         --halo %s --output DD%.4i --system pleiades_raymond > ./outfiles/%s_track_satellites.err > \
                         ./outfiles/%s_track_satellites.out\n'%(halo, DD, snapname_temp, snapname_temp))

       	qf.close()	

        sf.write('qsub %s\n'%qsub_fname)

    sf.close()  
