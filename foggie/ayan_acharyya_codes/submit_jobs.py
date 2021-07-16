##!/usr/bin/env python3

"""

    Title :      submit_jobs
    Notes :      Python wrapper to create and submit one or more jobs on pleiades
    Author :     Ayan Acharyya
    Started :    July 2021
    Example :    run submit_jobs.py --call filter_star_properties --do_all_sims --nnodes 50 --ncores 4 --prefix fsp_allsims --halo 8508 --dryrun
    OR :         run submit_jobs.py --call filter_star_properties --do_all_sims --nnodes 50 --ncores 4 --prefix fsp_allsims --do_all_halos --nhours 24 --dryrun

"""
import os, subprocess, argparse, datetime
HOME = os.getenv('HOME')

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
    parser.add_argument('--proc', metavar='proc', type=str, action='store', default='has')
    parser.add_argument('--start', metavar='start', type=int, action='store', default=1)
    parser.add_argument('--stop', metavar='stop', type=int, action='store', default=1)
    parser.add_argument('--mergeHII', metavar='mergeHII', type=float, action='store', default=None)
    parser.add_argument('--proj', metavar='proj', type=str, action='store', default='s1698')
    parser.add_argument('--halo', metavar='halo', type=str, action='store', default=None)
    parser.add_argument('--prefix', metavar='prefix', type=str, action='store', default=None)
    parser.add_argument('--callfunc', metavar='callfunc', type=str, action='store', default='filter_star_properties')
    parser.add_argument('--dryrun', dest='dryrun', action='store_true', default=False)
    parser.add_argument('--do_all_sims', dest='do_all_sims', action='store_true', default=False)
    parser.add_argument('--do_all_halos', dest='do_all_halos', action='store_true', default=False)
    parser.add_argument('--galrad', metavar='galrad', type=str, action='store', default=None)
    parser.add_argument('--xcol', metavar='xcol', type=str, action='store', default=None)
    parser.add_argument('--ycol', metavar='ycol', type=str, action='store', default=None)
    parser.add_argument('--colorcol', metavar='colorcol', type=str, action='store', default=None)
    parser.add_argument('--makemovie', dest='makemovie', action='store_true', default=False)
    parser.add_argument('--delay', metavar='delay', type=str, action='store', default=None)
    parser.add_argument('--fullbox', dest='fullbox', action='store_true', default=False)
    parser.add_argument('--overplot_stars', dest='overplot_stars', action='store_true', default=False)
    parser.add_argument('--clobber_plot', dest='clobber_plot', action='store_true', default=False)
    args, leftovers = parser.parse_known_args()

    return args

# ---------------------------------------------------------
if __name__ == '__main__':
    time_of_begin = datetime.datetime.now()
    args = parse_args()

    #----------setting different variables based on args--------

    systemflag = ' --system ' + args.system
    dryrunflag = ' --dryrun ' if args.dryrun else ''
    do_all_simsflag = ' --do_all_sims ' if args.do_all_sims else ''
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
    clobber_plot_flag = ' --clobber_plot ' if args.clobber_plot else ''


    jobscript_path = HOME+'/Work/astro/ayan_codes/foggie/foggie/ayan_acharyya_codes/'
    jobscript_template = 'jobscript_template_' + args.system + '.txt'
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

    elif 'pleiades' in args.system:
        procs_dir = {'san':(16, 32), 'ivy':(20, 64), 'has':(24, 128), 'bro':(28, 128), 'bro_ele':(28, 128), 'sky_ele':(40, 192), 'cas_ait':(40, 192)} # (nnodes, mem) for each proc, from https://www.nas.nasa.gov/hecc/support/kb/pbs-resource-request-examples_188.html
        max_hours_dict = {'low':4, 'normal':8, 'long':120, 'debug':2, 'devel':2} # from https://www.nas.nasa.gov/hecc/support/kb/pbs-job-queue-structure_187.html
        workdir = '/nobackup/aachary2/foggie_outputs/pleiades_workdir' # for pleiades
        nnodes = args.nnodes
        ncores = args.ncores if args.ncores is not None else procs_dir[args.proc][0]
        if args.dryrun: memory = str(int(ncores) * 2) + 'GB'
        else: memory = str(int(nnodes) * procs_dir[args.proc][1]) + 'GB'
        qname = args.queue

    #----------looping over and creating + submitting job files--------
    halos = ['8508', '5036', '2878', '2392', '5016', '4123']
    if args.do_all_halos:
        args.stop = len(halos) # do all halos by submitting ..
        do_all_halosflag = '' # ..multiple jobs one job for each halo
        #args.stop = args.start # do all halos by submitting..
        #do_all_halosflag = ' --do_all_halos ' # ..ONE massive job that will loop over all halos
    else:
        do_all_halosflag = ''

    for jobid in range(args.start, args.stop+1):
        thishalo = halos[jobid - 1] if args.halo is None else args.halo
        haloflag = ' --halo ' + thishalo
        jobname = prefixtext + thishalo + '_job' + str(jobid)

        nhours = args.nhours if args.nhours is not None else '01' if args.dryrun or args.queue == 'devel' else '%02d'%(max_hours_dict[args.queue])

        out_jobscript = workdir + '/jobscript_' + jobname + '.sh'

        replacements = {'PROJ_CODE': args.proj, 'RUN_NAME': jobname, 'NNODES': nnodes, 'NHOURS': nhours, 'CALLFILE': callfile, 'WORKDIR': workdir, \
                        'JOBSCRIPT_PATH': jobscript_path, 'NCORES': ncores, 'MEMORY': memory, 'DRYRUNFLAG': dryrunflag, 'QNAME': qname, 'PROC': args.proc, \
                        'MERGEHIIFLAG': mergeHIIflag, 'DO_ALL_SIMSFLAG': do_all_simsflag, 'DO_ALL_HALOSFLAG': do_all_halosflag, 'SYSTEMFLAG': systemflag, \
                        'HALOFLAG': haloflag, 'NCPUS': nnodes * ncores, 'GALRAD_FLAG':galrad_flag, 'XCOL_FLAG': xcol_flag, 'YCOL_FLAG': ycol_flag, \
                        'COLORCOL_FLAG': colorcol_flag, 'MAKEMOVIE_FLAG': makemovie_flag, 'DELAY_FLAG': delay_flag, 'FULLBOX_FLAG': fullbox_flag, \
                        'OVERPLOT_STARS_FLAG': overplot_stars_flag, 'CLOBBER_PLOT_FLAG': clobber_plot_flag} # keywords to be replaced in template jobscript

        with open(jobscript_path + jobscript_template) as infile, open(out_jobscript, 'w') as outfile:
            for line in infile:
                for src, target in replacements.items():
                    line = line.replace(str(src), str(target))
                outfile.write(line) # replacing and creating new jobscript file

        print('Going to submit job ' + jobname+' at {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
        jobid = subprocess.check_output(['qsub ' + out_jobscript], shell=True)[:-1]

    print('Submitted all '+str(args.stop - args.start + 1)+' job/s from index '+str(args.start)+' to '+str(args.stop) + '\n')