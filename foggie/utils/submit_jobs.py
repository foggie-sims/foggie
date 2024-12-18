##!/usr/bin/env python3

"""

    Title :      submit_jobs
    Notes :      Python wrapper to create and submit one or more jobs on pleiades
    Author :     Ayan Acharyya
    Started :    July 2021
    Example :    run submit_jobs.py --call filter_star_properties --nnodes 50 --ncores 4 --prefix fsp --halo 8508 --dryrun --opt_args "--do_sll_sims"
    OR :         run submit_jobs.py --call filter_star_properties --nnodes 50 --ncores 4 --prefix fsp --do_all_halos --nhours 24 --dryrun  --opt_args "--do_sll_sims --do_all_halos"
    OR :         run /nobackupp19/aachary2/ayan_codes/foggie/foggie/ayan_acharyya_codes/submit_jobs.py --call datashader_movie --prefix rsv_dsm_Zprofs --halo 8508 --queue s1938_mpe1 --aoe sles12 --proj s1938 --nnodes 5 --ncores 4 --proc has --opt_args "--galrad 20 --xcol rad --ycol metal --colorcol density,vrad,temp --overplot_stars --makemovie --delay 0.1 "
    OR :         run /nobackupp19/aachary2/ayan_codes/foggie/foggie/ayan_acharyya_codes/submit_jobs.py --call datashader_movie --prefix dsm_Zprofs --halo 2392 --queue ldan --opt_args "--galrad 20 --xcol rad --ycol metal --colorcol density,vrad,temp --overplot_stars --makemovie --delay 0.1"
    OR :         run submit_jobs.py --call compute_MZgrad --system ayan_pleiades --halo 8508 --nnodes 50 --ncores 4 --queue normal --prefix cmzg --opt_args "--do_all_sims --upto_re 3 --xcol rad_re --weight mass --write_file --noplot"
    OR :         run /nobackupp19/aachary2/ayan_codes/foggie/foggie/ayan_acharyya_codes/submit_jobs.py --call compute_MZgrad --system ayan_pleiades --halo 8508 --jobarray --do_all_sims --use_onlyDD --nhours 0 --nmins 15 --prefix cmzg --opt_args "--upto_re 3 --xcol rad_re --weight mass --write_file --noplot"
    OR :         run /nobackupp19/aachary2/ayan_codes/foggie/foggie/ayan_acharyya_codes/submit_jobs.py --call compute_MZgrad --system ayan_pleiades --do_all_halos --start 2 --queue ldan --mem 1500GB --prefix cmzg --opt_args "--do_all_sims --use_onlyDD --upto_kpc 10 --xcol rad --weight mass --write_file --noplot"
    OR :         run /nobackupp19/aachary2/ayan_codes/foggie/foggie/ayan_acharyya_codes/submit_jobs.py --call compute_MZgrad --system ayan_pleiades --do_all_halos --queue ldan --mem 1500GB --prefix cmzg --opt_args "--do_all_sims --use_onlyDD --upto_kpc 10 --xcol rad --weight mass --write_file --forpaper"
    OR :         run /nobackupp19/aachary2/ayan_codes/foggie/foggie/ayan_acharyya_codes/submit_jobs.py --call compute_MZscatter --system ayan_pleiades --do_all_halos --queue ldan --mem 1500GB --prefix cmzs --opt_args "--do_all_sims --use_onlyDD --upto_kpc 10 --nbins 100 --weight mass --write_file --forpaper"
"""
import subprocess, argparse, datetime, os
from collections import defaultdict
from util import get_all_sims_for_this_halo
import numpy as np

# ------------------------------------------------------
def execute_command(command, is_dry_run=False):
    '''
    Function to decide whether to execute a command or simply print it out (for dry run)
    '''
    if is_dry_run:
        print('Not executing command:', command, '\n')
        return -99
    else:
        print('Executing command:', command, '\n')
        job = subprocess.check_output([command], shell=True)[:-1]
        return job

# ---------------------------------------------------------
def parse_args():
    '''
    Function to parse keyword arguments
    '''
    parser = argparse.ArgumentParser(description="calling plotobservables for full parameter space")
    parser.add_argument('--system', metavar='system', type=str, action='store', default='ayan_pleiades')
    parser.add_argument('--queue', metavar='queue', type=str, action='store', default='long')
    parser.add_argument('--nnodes', metavar='nnodes', type=int, action='store', default=1)
    parser.add_argument('--ncores', metavar='ncores', type=int, action='store', default=None)
    parser.add_argument('--nhours', metavar='nhours', type=int, action='store', default=None)
    parser.add_argument('--ncpus', metavar='ncpus', type=int, action='store', default=None)
    parser.add_argument('--nmins', metavar='nmins', type=int, action='store', default=0)
    parser.add_argument('--proc', metavar='proc', type=str, action='store', default='has')
    parser.add_argument('--memory', metavar='memory', type=str, action='store', default=None)
    parser.add_argument('--aoe', metavar='aoe', type=str, action='store', default=None)
    parser.add_argument('--start', metavar='start', type=int, action='store', default=1)
    parser.add_argument('--stop', metavar='stop', type=int, action='store', default=1)
    parser.add_argument('--mergeHII', metavar='mergeHII', type=float, action='store', default=None)
    parser.add_argument('--proj', metavar='proj', type=str, action='store', default='s2358')
    parser.add_argument('--halo', metavar='halo', type=str, action='store', default=None)
    parser.add_argument('--run', metavar='run', type=str, action='store', default='nref11c_nref9f')
    parser.add_argument('--prefix', metavar='prefix', type=str, action='store', default=None)
    parser.add_argument('--callfunc', metavar='callfunc', type=str, action='store', default='filter_star_properties')
    parser.add_argument('--dryrun', dest='dryrun', action='store_true', default=False)
    parser.add_argument('--do_all_sims', dest='do_all_sims', action='store_true', default=False)
    parser.add_argument('--nevery', metavar='nevery', type=int, action='store', default=1)
    parser.add_argument('--do_all_halos', dest='do_all_halos', action='store_true', default=False)
    parser.add_argument('--galrad', metavar='galrad', type=str, action='store', default=None)
    parser.add_argument('--xcol', metavar='xcol', type=str, action='store', default=None)
    parser.add_argument('--ycol', metavar='ycol', type=str, action='store', default=None)
    parser.add_argument('--colorcol', metavar='colorcol', type=str, action='store', default=None)
    parser.add_argument('--makemovie', dest='makemovie', action='store_true', default=False)
    parser.add_argument('--delay', metavar='delay', type=str, action='store', default=None)
    parser.add_argument('--fullbox', dest='fullbox', action='store_true', default=False)
    parser.add_argument('--overplot_stars', dest='overplot_stars', action='store_true', default=False)
    parser.add_argument('--overplot_source_sink', dest='overplot_source_sink', action='store_true', default=False)
    parser.add_argument('--nchunks', metavar='nchunks', type=str, action='store', default=None)
    parser.add_argument('--clobber_plot', dest='clobber_plot', action='store_true', default=False)
    parser.add_argument('--units_kpc', dest='units_kpc', action='store_true', default=False)
    parser.add_argument('--units_rvir', dest='units_rvir', action='store_true', default=False)
    parser.add_argument('--temp_cut', dest='temp_cut', action='store_true', default=False)
    parser.add_argument('--opt_args', metavar='opt_args', type=str, action='store', default='')
    parser.add_argument('--jobarray', dest='jobarray', action='store_true', default=False)
    parser.add_argument('--use_onlyRD', dest='use_onlyRD', action='store_true', default=False)
    parser.add_argument('--use_onlyDD', dest='use_onlyDD', action='store_true', default=False)
    parser.add_argument('--snapstart', metavar='snapstart', type=int, action='store', default=30)
    parser.add_argument('--snapstop', metavar='snapstop', type=int, action='store', default=30)
    args, leftovers = parser.parse_known_args()

    return args

# ---------------------------------------------------------
if __name__ == '__main__':
    time_of_begin = datetime.datetime.now()
    args = parse_args()

    # ----------special settings for ldan queue--------
    if args.queue == 'ldan':
        args.proc = 'ldan'
        args.nnodes = 1
        args.ncores = None
    # ----------special settings for endeavour queue--------
    elif args.queue[:2] == 'e_':
        args.proc = 'cas_end'

    # ----------special setting for jobarrays------------
    if args.jobarray:
        args.nnodes = 1
        args.ncores = 1

    #----------setting different variables based on args--------
    systemflag = ' --system ' + args.system
    runsimflag = ' --run ' + args.run
    nchunks_flag = ' --nchunks ' + args.nchunks if args.nchunks is not None else ''
    dryrunflag = ' --dryrun ' if args.dryrun else ''
    mergeHIIflag = ' --mergeHII ' + str(args.mergeHII) if args.mergeHII is not None else ''
    prefixtext = args.prefix + '_' if args.prefix is not None else ''
    makemovie_flag = ' --makemovie ' if args.makemovie else ''
    galrad_flag = ' --galrad ' + args.galrad if args.galrad is not None else ''
    xcol_flag = ' --xcol ' + args.xcol if args.xcol is not None else ''
    ycol_flag = ' --ycol ' + args.ycol if args.ycol is not None else ''
    colorcol_flag = ' --colorcol ' + args.colorcol if args.colorcol is not None else ''
    delay_flag = ' --delay ' + args.delay if args.delay is not None else ''
    fullbox_flag = ' --fullbox ' if args.fullbox else ''
    overplot_stars_flag = ' --overplot_stars ' if args.overplot_stars else ''
    overplot_source_sink_flag = ' --overplot_source_sink ' if args.overplot_source_sink else ''
    clobber_plot_flag = ' --clobber_plot ' if args.clobber_plot else ''
    nevery_flag = ' --nevery ' + str(args.nevery) if args.nevery > 1 else ''
    tempcut_flag = ' --temp_cut ' if args.temp_cut else ''
    units_flag = ' --units_kpc ' if args.units_kpc else ' --units_rvir ' if args.units_rvir else ''

    if 'pleiades' in args.system: jobscript_path = '/nobackupp19/aachary2/ayan_codes/foggie/foggie/ayan_acharyya_codes/'
    elif args.system == 'ayan_local': jobscript_path = os.getenv('HOME') + '/Work/astro/ayan_codes/foggie/foggie/ayan_acharyya_codes/'

    jobarray_or_jobscript = 'jobarray' if args.jobarray else 'jobscript'
    if args.system == 'ayan_local': jobscript_template = jobarray_or_jobscript + '_template_ayan_pleiades.txt'
    else: jobscript_template = jobarray_or_jobscript + '_template_' + args.system + '.txt'

    callfile = jobscript_path + args.callfunc + '.py'

    if args.system == 'avatar':
        workdir = '/avatar/acharyya/enzo_models' # for avatar
        nnodes = '3' if not args.dryrun else '01'
        ncores = '28' if not args.dryrun else '01'
        memory = 'not-needed-for-avatar'
        if args.queue == 'small': qname = 'smallmem'
        elif args.queue == 'large': qname = 'largemem'
    
    elif args.system == 'raijin':
        workdir = '/short/ek9/axa100/enzo_models' # for raijin
        nnodes = 'not-needed-for-raijin'
        ncores = '84' #'128' if not args.dryrun else '01' # 16 cores * 8 nodes
        sizepercore = 8 if args.queue == 'large' else 4
        if args.dryrun: sizepercore = 2
        memory = '768GB' if args.queue == 'large' else '768GB' #str(int(ncores)*int(sizepercore))+'GB'
        qname = 'normalbw' #'normal' if not args.dryrun else 'express'
    
    elif args.system == 'gadi':
        workdir = '/scratch/ek9/axa100/enzo_models' # for gadi
        nnodes = 2
        ncores = 1 if args.nonewcube else str(nnodes * 48)
        sizepercore = 4
        if args.dryrun: sizepercore = 2
        if args.nonewcube: memory = '30GB' if args.queue == 'large' else '8GB'
        else: memory = str(int(ncores)*int(sizepercore))+'GB'
        qname = 'hugemem' if args.queue == 'large' and args.nonewcube else 'normal'

    elif 'pleiades' in args.system or args.system == 'ayan_local':
        procs_dir = {'san':(16, 32), 'ivy':(20, 64), 'has':(24, 128), 'bro':(28, 128), 'bro_ele':(28, 128), 'sky_ele':(40, 192), 'cas_ait':(40, 192), 'ldan':(16, 750), 'cas_end':(28, 185)} # (nnodes, mem) for each proc, from https://www.nas.nasa.gov/hecc/support/kb/pbs-resource-request-examples_188.html
        max_hours_dict = defaultdict(lambda: 120, low=4, normal=8, long=120, e_long=72, e_normal=8, e_vlong=600, e_debug=2, debug=2, devel=2, ldan=72) # from https://www.nas.nasa.gov/hecc/support/kb/pbs-job-queue-structure_187.html
        if 'pleiades' in args.system: workdir = '/nobackupp19/aachary2/foggie_outputs/pleiades_workdir' # for pleiades
        elif args.system == 'ayan_local': workdir = '.'
        nnodes = args.nnodes
        ncores = args.ncores if args.ncores is not None else procs_dir[args.proc][0]
        memory = args.memory if args.memory is not None else str(procs_dir[args.proc][1]) + 'GB' # minimum memory per node; by default the entire node me is allocated, therefore it is redundant to specify mem as the highest available memory per node
        qname = args.queue
        if args.queue[:2] == 'e_':  qname += '@pbspl4' # add this for endeavour nodes

    # ----------determining what resource request goes into the job script, based on queues, procs, etc.---------
    nhours = args.nhours if args.nhours is not None else '01' if args.dryrun or args.queue == 'devel' else '%02d' % (max_hours_dict[args.queue])
    ncpus = nnodes * ncores if args.ncpus is None else args.ncpus

    resources = 'select=' + str(nnodes) + ':ncpus=' + str(ncores)

    if args.queue[:2] == 'e_': resources += ':mem=' + memory # for submitting to endeavour
    else: resources += ':mpiprocs=' + str(ncores)

    if args.queue == 'ldan': resources += ':mem=' + memory # may specify mem per node for jobs on LDAN (for other procs it is by default the max available node mem)
    else: resources += ':model=' + args.proc # need to specify the proc (but not necessarily the mem if I'm using the full node memory) if not an LDAN job

    if args.aoe is not None: resources += ':aoe=' + args.aoe

    #----------looping over and creating + submitting job files--------
    halos = ['8508', '5036', '5016', '4123', '2392', '2878']
    if args.do_all_halos: args.stop = len(halos) # do all halos by submitting ..

    for jobid in range(args.start, args.stop+1):
        thishalo = halos[jobid - 1] if args.halo is None else args.halo
        haloflag = ' --halo ' + thishalo
        jobname = prefixtext + thishalo
        if jobname[:3] != args.proc: jobname = args.proc + '_' + jobname
        if args.nevery > 1: jobname += '_ne' + str(args.nevery)

        # ----------replacing keywords in jobscript template to make the actual jobscript---------
        out_jobscript = workdir + '/' + jobarray_or_jobscript + '_' + jobname + '.sh'

        replacements = {'PROJ_CODE': args.proj, 'RUN_NAME': jobname, 'NHOURS': nhours, 'NMINS': args.nmins, 'CALLFILE': callfile, 'WORKDIR': workdir, \
                        'JOBSCRIPT_PATH': jobscript_path, 'DRYRUNFLAG': dryrunflag, 'QNAME': qname, 'RESOURCES': resources, 'RUNSIMFLAG': runsimflag,\
                        'MERGEHIIFLAG': mergeHIIflag, 'SYSTEMFLAG': systemflag, 'NCPUS': str(ncpus),\
                        'HALOFLAG': haloflag, 'GALRAD_FLAG':galrad_flag, 'XCOL_FLAG': xcol_flag, 'YCOL_FLAG': ycol_flag, \
                        'COLORCOL_FLAG': colorcol_flag, 'MAKEMOVIE_FLAG': makemovie_flag, 'DELAY_FLAG': delay_flag, 'FULLBOX_FLAG': fullbox_flag, \
                        'OVERPLOT_STARS_FLAG': overplot_stars_flag, 'OVERPLOT_SOURCE_SINK_FLAG': overplot_source_sink_flag, 'CLOBBER_PLOT_FLAG': clobber_plot_flag, 'NSECONDS':str(int(nhours) * 3600), \
                        'NEVERY_FLAG': nevery_flag, 'UNITS_FLAG': units_flag, 'TEMPCUT_FLAG': tempcut_flag, 'NCHUNKS_FLAG': nchunks_flag, 'OPT_ARGS': args.opt_args} # keywords to be replaced in template jobscript

        with open(jobscript_path + jobscript_template) as infile, open(out_jobscript, 'w') as outfile:
            for line in infile:
                for src, target in replacements.items():
                    line = line.replace(str(src), str(target))
                outfile.write(line) # replacing and creating new jobscript file

        print('Going to submit job ' + jobname+' at {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
        if args.jobarray:
            if args.do_all_sims:
                if args.system == 'ayan_local': simpath = '/Users/acharyya/models/simulation_output/foggie/'
                else: simpath = '/nobackupp19/mpeeples/'
                list_of_sims = get_all_sims_for_this_halo(args, simpath + 'halo_00' + thishalo + '/' + args.run + '/')
                args.snapstart = int(list_of_sims[0][1][2:])
                args.snapstop = int(list_of_sims[-1][1][2:])

            nbatches = int(np.ceil((args.snapstop - args.snapstart + 1)/500))
            for interval in range(nbatches):
                jobid = execute_command('qsub -J ' + str(args.snapstart + interval) + '-' + str(args.snapstop + interval) + ':' + str(nbatches) + ' ' + out_jobscript, is_dry_run=args.dryrun)
        else:
            jobid = execute_command('qsub ' + out_jobscript, is_dry_run=args.dryrun)

    print('Submitted all '+str(args.stop - args.start + 1)+' job/s from index '+str(args.start)+' to '+str(args.stop) + '\n')