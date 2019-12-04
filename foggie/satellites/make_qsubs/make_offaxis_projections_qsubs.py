import numpy as np
from numpy import *
import glob
from glob import glob
import os


halos  = ['2392', '2878', '4123', '5016', '5036', '8508']
#halos = ['2878']
inputs = [('2878', 'DD0581'), 
          ('5016', 'DD0581'), 
          ('5036', 'DD0581'),
          ('2392', 'DD0581'),
          ('4123', 'DD0581'),
          ('8508', 'DD0487')]
for (halo, output) in inputs:

    submit_dir = '/nobackupp2/rcsimons/foggie/submit_scripts/off_axis_projections'

    if not os.path.isdir(submit_dir): os.system('mkdir %s'%submit_dir)
    if not os.path.isdir(submit_dir+'/outfiles'): os.system('mkdir %s/outfiles'%submit_dir)
    

    sf = open('/nobackupp2/rcsimons/foggie/submit_scripts/off_axis_projections/submit_%s_tracksats.sh'%(halo), 'w+')
    splitn = 2
    for dmn in arange(0, 200+splitn, splitn):
        snapname = '%s_%.4i_%.4i'%(halo, dmn, dmn+splitn)
        qsub_fname = 'track_%s_%.4i_%.4i.qsub'%(halo, dmn, dmn+splitn)        
        qf = open('/nobackupp2/rcsimons/foggie/submit_scripts/off_axis_projections/%s'%qsub_fname, 'w+')
        
        qf.write('#PBS -S /bin/bash\n')
        qf.write('#PBS -l select=1:ncpus=20:model=ivy\n')
        qf.write('#PBS -l walltime=1:00:00\n')
        qf.write('#PBS -q normal\n')
        qf.write('#PBS -N %s\n'%snapname)
        qf.write('#PBS -M rsimons@jhu.edu\n')
        qf.write('#PBS -m abe\n')
        qf.write('#PBS -o ./outfiles/%s_pbs.out\n'%snapname)
        qf.write('#PBS -e ./outfiles/%s_pbs.err\n'%snapname)
        qf.write('#PBS -V\n')
        qf.write('#PBS -W group_list=s1938\n\n\n\n')  

        qf.write('source /u/rcsimons/.bashrc\n')
        for rot_n in arange(dmn, dmn+splitn):
            snapname_temp = '%s_%.4i'%(halo, rot_n)
            qf.write('python /nobackupp2/rcsimons/git/foggie/foggie/satellites/off_axis_projection_plots.py \
                     --halo %s --rot_n %i --output %s --system pleiades_raymond --do_central > ./outfiles/%s_off_axis.err > \
                     ./outfiles/%s_off_axis.out\n'%(halo, rot_n, output, snapname_temp, snapname_temp))

        qf.close()  

        sf.write('qsub %s\n'%qsub_fname)

    sf.close()  










