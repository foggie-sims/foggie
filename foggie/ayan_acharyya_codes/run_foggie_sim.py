##!/usr/bin/env python3

"""

    Title :      run_foggie_sim.py
    Notes :      automated script to run zoom-in FOGGIE simulations (generate ICs with MUSIC, and then submit Enzo jobs); this is based on JT's /nobackup/jtumlins/CGM_bigbox/halo_template/plots.py
    Output :     submitted PBS jobs
    Author :     JT; modified by Ayan Acharyya
    Started :    Jan 2022
    Example :    run run_foggie_sim.py --halo 2430 --level 1 --queue devel --dryrun
                 run run_foggie_sim.py --halo 2139 --level 1 --queue devel --automate --final_level 3 --dryrun

"""
from header import *
from util import *
from astropy.table import Table
start_time = time.time()

# ------------------------------------------------------
def setup_conf_file(args):
    '''
    Function to set up the .conf file: populate it with corresponding simulation directory, halo center, etc.
    '''
    # --------get conf file names-----------
    gas_or_dm = 'gas' if args.gas else 'DM'
    template_conf_file = args.template_dir + '/halo_H_' + gas_or_dm + '_XtoY.conf'
    target_conf_file = args.halo_dir + '/halo_' + args.halo + '_' + gas_or_dm + '_' + str(args.level - 1) + 'to' + str(args.level) + '.conf'

    # ---------getting correct halo center based on the refinement level-----------------
    print('Before starting L-' + str(args.level) + ',',)
    if args.level > 1:
        conf_log_file = args.halo_dir + '/' + args.sim_name + '-L' + str(args.level - 1) + '.conf_log.txt'
        patterns_to_search = ['shift_x = ', 'shift_y = ', 'shift_z = ']
        shifts = []
        for pattern in patterns_to_search:
            with open(conf_log_file, 'r') as infile:
                for line in infile:
                    if re.search(pattern, line):
                        this_shift = int(line[line.find(pattern)+len(pattern) : line.find('\n')])
            shifts.append(this_shift)

        args.center += np.array(shifts) / 255. # to convert shifts into code units
        print('center offsets = ', shifts, ' ',)

    print('halo center = ', args.center)
    halo_cen = ', '.join([str(item) for item in args.center])

    # ---------make substitutions in conf file-----------------
    if args.level > 1: sim_dir = args.halo_dir
    else: sim_dir = args.sim_dir

    replacements = {'SIM_NAME': args.sim_name, 'SIM_DIR': sim_dir, 'FINAL_Z': args.final_z, 'HALO_CEN': halo_cen}  # keywords to be replaced in template jobscript

    with open(template_conf_file) as infile, open(target_conf_file, 'w') as outfile:
        for line in infile:
            for src, target in replacements.items():
                line = line.replace(str(src), str(target))
            outfile.write(line)  # replacing and creating new file

    print('Written', target_conf_file)
    return target_conf_file

# ------------------------------------------------------
def write_gas_conf_file(args):
    '''
    Function to write .conf file for the gas run, based on that of DM only runs
    '''
    # --------get conf file names-----------
    source_conf_file = args.halo_dir + '/' + args.sim_name + '-L' + str(args.level) + '.conf'
    target_conf_file = os.path.splitext(source_conf_file)[0] + '-gas.conf'

    # ---------make substitutions in conf file-----------------
    replacements = {'baryons': 'no', 'Omega_b': 0.0461, 'filename': args.sim_name + '-L' + str(args.level) + '-gas'}  # keywords to be replaced in template jobscript

    with open(source_conf_file) as infile, open(target_conf_file, 'w') as outfile:
        for line in infile:
            for pattern, target in replacements.items():
                if re.search(pattern, line):
                    line = pattern + ' = ' + target
            outfile.write(line)  # replacing and creating new file

    Path(args.halo_dir + args.sim_name + '-L' + str(args.level) + '-gas').mkdir(parents=True, exist_ok=True) # creating the directory structure, if doesn't exist already
    print('Written', target_conf_file)
    return target_conf_file

# ------------------------------------------------------
def run_music(target_conf_file, args):
    '''
    Function to run MUSIC, to generate the initial conditions for a given level
    '''
    os.chdir(args.halo_dir)
    print('Starting MUSIC..\n')

    if do_gas(args): # run MUSIC directly for gas runs
        command = '/nobackup/aachary2/ayan_codes/music/MUSIC ' + target_conf_file
        execute_command(command, args.dryrun)
    else: # run using enzo-mrp-music for DM only runs
        command = 'python ' + args.sim_dir + '/enzo-mrp-music/enzo-mrp-music.py ' + target_conf_file + ' ' + str(args.level)
        execute_command(command, args.dryrun)

        # --------need to move the recently made directory, in case of L1 runs---------
        if args.level == 1: execute_command('mv ../25Mpc_DM_256-L1 .', args.dryrun)

    print('Finished running MUSIC')

# ------------------------------------------------------
def setup_enzoparam_file(args):
    '''
    Function to set up the .enzo parameter file: populate it with corresponding refinement level, grid properties, etc.
    '''
    # --------get conf file names-----------
    gas_or_dm = '-gas' if args.gas else ''
    template_param_file = args.template_dir + '/25Mpc_DM_256-LX' + gas_or_dm + '.enzo'
    target_param_file = args.halo_dir + '/' + args.sim_name + '-L' + str(args.level) + '/' + args.sim_name + '-L' + str(args.level) + gas_or_dm + '.enzo'

    # ---------getting correct grid parameters-----------------
    param_file = args.halo_dir + '/' + args.sim_name + '-L' + str(args.level) + '/parameter_file.txt'
    stuff_from_paramfile = ''
    with open(param_file, 'r') as infile:
        for line in infile:
            if re.search('CosmologySimulationGrid', line):
                stuff_from_paramfile += line

    # ---------make substitutions in conf file-----------------
    replacements = {'CFR': args.final_z, 'MODFR': 8/(8 ** args.level), 'CSNOI': args.level + 1, 'MRPRTL': args.level, 'COPYHERE': stuff_from_paramfile}  # keywords to be replaced in template jobscript

    with open(template_param_file) as infile, open(target_param_file, 'w') as outfile:
        for line in infile:
            for src, target in replacements.items():
                line = line.replace(str(src), str(target))
            outfile.write(line)  # replacing and creating new file

    print('Written', target_param_file)

# ------------------------------------------------------
def run_enzo(nnodes, ncores, nhours, args):
    '''
    Function to submit the Enzo simulation job
    '''
    # --------get runscript file names-----------
    template_runscript = args.template_dir + '/RunScript_LX.sh'
    target_runscript = args.halo_dir + '/' + args.sim_name + '-L' + str(args.level) + '/RunScript.sh'

    # ---------make substitutions in conf file-----------------
    if do_gas(args):
        jobname = args.halo_name + '_gas_' + 'L' + str(args.level)
        enzo_param_file = args.sim_name + '-L' + str(args.level) + '-gas.enzo'
    else:
        jobname = args.halo_name + '_DM_' + 'L' + str(args.level)
        enzo_param_file = args.sim_name + '-L' + str(args.level) + '.enzo'

    path_to_simrun = '/nobackup/aachary2/bigbox/halo_template/simrun.pl'
    calls_to_script = path_to_simrun + ' -mpi \"mpiexec -np ' + str(nnodes * ncores) + ' /u/scicon/tools/bin/mbind.x -cs \" -wall 432000 -pf \"' + enzo_param_file + '\" -jf \"' + os.path.split(target_runscript)[1] + '\"'

    replacements = {'halo_H_DM_LX': jobname, 'NNODES': nnodes, 'NCORES': ncores, 'PROC': args.proc, 'NHOURS': nhours, 'QNAME': args.queue, \
                    'CALLS_TO_SCRIPT': calls_to_script}  # keywords to be replaced in template jobscript

    with open(template_runscript) as infile, open(target_runscript, 'w') as outfile:
        for line in infile:
            for src, target in replacements.items():
                line = line.replace(str(src), str(target))
            outfile.write(line)  # replacing and creating new file

    print('Written', target_runscript)

    workdir = args.halo_dir + '/' + args.sim_name + '-L' + str(args.level)
    os.chdir(workdir)
    execute_command('qsub ' + target_runscript, args.dryrun)
    print('Submitted enzo job:', jobname)

    # -----------if asked to automate, figure out if previous job has finished----------------
    if args.automate and args.level < args.final_level:
        args.nomusic, foundfile = False, False
        delay = 5 # minutes
        print('\nThis is automatic mode. Will keep checking for the output of the job just submitted every', delay, 'minutes..' )

        while not foundfile:
            time.sleep(delay * 60) # seconds
            foundfile = os.path.exists(workdir + '/pbs_output.txt')
            if foundfile: print('Found', workdir + '/pbs_output.txt at', datetime.datetime.now().strftime("%H:%M:%S"))
            else: print('Tried and failed to find', workdir + '/pbs_output.txt at', datetime.datetime.now().strftime("%H:%M:%S"), 'will try again after', delay, 'minutes..' )

        args.level += 1
        print('\nStarting halo', args.halo, 'at next refinement level L' + str(args.level))
        wrap_run_enzo(ncores, nhours, args)

    print('Finished submitting all enzo jobs')

# ------------------------------------------------------
def run_multiple_enzo_levels(nnodes, ncores, nhours, args):
    '''
    Function to submit a single PBS job which would include multiple subsequent Enzo refinment levels
    '''
    # --------set file names and job names, etc.-----------
    template_runscript = args.template_dir + '/RunScript_LX.sh'
    target_runscript = args.halo_dir + '/RunScript-L' + str(args.level) + '-to-L' + str(args.final_level) + '.sh'
    path_to_simrun = '/nobackup/aachary2/bigbox/halo_template/simrun.pl'
    path_to_script = '/nobackup/aachary2/ayan_codes/foggie/foggie/ayan_acharyya_codes/'

    if do_gas(args): jobname = args.halo_name + '_gas_' + 'L' + str(args.level) + '-to-L' + str(args.final_level)
    else: jobname = args.halo_name + '_DM_' + 'L' + str(args.level) + '-to-L' + str(args.final_level)
    dm_or_gas = '-gas' if do_gas(args) else ''

    # ---------loop over subsequent refinement levels to create separate lines in the PBS job script-----------------
    calls_to_script = '\ncd ' + args.halo_dir + '\n\n'
    for thislevel in range(args.level, args.final_level + 1):
        dummy_args = copy.copy(args)
        dummy_args.automate, dummy_args.dryrun = False, False  # such that only individual levels are addressed inside the loop
        dummy_args.level = thislevel
        enzo_param_file = dummy_args.sim_name + '-L' + str(dummy_args.level) + dm_or_gas + '.enzo'

        argslist = {key: val for key, val in vars(dummy_args).items() if val is not None}
        call_to_pyscript = 'python ' + path_to_script + 'run_foggie_sim.py ' + ' '.join(['--' + key + ' ' + str(val) for key,val in argslist.items()]) # this runs the setup required BEFORE the enzo job (including running MUSIC) for a given refinement level
        call_to_cd = 'cd ' + dummy_args.halo_dir + '/' + dummy_args.sim_name + '-L' + str(dummy_args.level)
        call_to_simrun = path_to_simrun + ' -mpi \"mpiexec -np ' + str(nnodes * ncores) + ' /u/scicon/tools/bin/mbind.x -cs \" -wall 432000 -pf \"' + enzo_param_file + '\" -jf \"' + os.path.split(target_runscript)[1] + '\"' # this runs the enzo job for a given refinement level
        calls_to_script += call_to_pyscript + '\n\n' + call_to_cd + '\n\n' + call_to_simrun + '\n\n'

    # ---------make substitutions in conf file-----------------
    replacements = {'halo_H_DM_LX': jobname, 'NNODES': nnodes, 'NCORES': ncores, 'PROC': args.proc, 'NHOURS': nhours,
                    'QNAME': args.queue, 'CALLS_TO_SCRIPT': calls_to_script}  # keywords to be replaced in template jobscript

    with open(template_runscript) as infile, open(target_runscript, 'w') as outfile:
        for line in infile:
            for src, target in replacements.items():
                line = line.replace(str(src), str(target))
            outfile.write(line)  # replacing and creating new file

    print('Written', target_runscript)
    execute_command('qsub ' + target_runscript, args.dryrun)

    print('Submitted enzo job:', jobname)

# ------------------------------------------------------
def execute_command(command, is_dry_run):
    '''
    Function to decide whether to execute a command or simply print it out (for dry run)
    '''
    if is_dry_run:
        print('Not executing command:', command, '\n')
    else:
        print('Executing command:', command, '\n')
        os.system(command)

# ------------------------------------------------------
def wrap_run_enzo(ncores, nhours, args):
    '''
    Wrapper function to execute other functions i.e. setup conf files, run MUSIC, run enzo, etc. for a given refinement level
    '''
    if do_gas(args):conf_file_name = write_gas_conf_file(args)
    else: conf_file_name = setup_conf_file(args)

    if not args.nomusic: run_music(conf_file_name, args)
    setup_enzoparam_file(args)
    run_enzo(args.nnodes, ncores, nhours, args)

    print('Completed submission for L' + str(args.level))

# ------------------------------------------------------
def do_gas(args):
    '''
    Function to determine whether to include baryons, depending on user argument AND refinement level
    '''
    return (args.gas and args.level > 3) # gas is not included up to L3

# -----main code-----------------
if __name__ == '__main__':
    # ------------- arg parse -------------------------------
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='''identify satellites in FOGGIE simulations''')

    parser.add_argument('--system', metavar='system', type=str, action='store', default='ayan_pleiades')
    parser.add_argument('--queue', metavar='queue', type=str, action='store', default='long')
    parser.add_argument('--nnodes', metavar='nnodes', type=int, action='store', default=4)
    parser.add_argument('--ncores', metavar='ncores', type=int, action='store', default=16)
    parser.add_argument('--nhours', metavar='nhours', type=int, action='store', default=None)
    parser.add_argument('--proc', metavar='proc', type=str, action='store', default='has')
    parser.add_argument('--memory', metavar='memory', type=str, action='store', default=None)
    parser.add_argument('--sim_dir', metavar='sim_dir', type=str, action='store', default='bigbox', help='Specify which directory the dataset lies in')
    parser.add_argument('--sim_name', metavar='sim_name', type=str, action='store', default='25Mpc_DM_256', help='Specify simulation name')
    parser.add_argument('--halo', metavar='halo', type=str, action='store', default='test', help='which halo?')
    parser.add_argument('--level', metavar='level', type=int, action='store', default=1, help='which refinement level? default 1')
    parser.add_argument('--gas', dest='gas', action='store_true', default=False, help='run DM+gas?, default is no, only DM')
    parser.add_argument('--final_z', metavar='final_z', type=float, action='store', default=2, help='final redshift till which the simulation should run; default is 2')
    parser.add_argument('--nomusic', dest='nomusic', action='store_true', default=False, help='skip MUSIC (if ICs already exist)?, default is no')
    parser.add_argument('--dryrun', dest='dryrun', action='store_true', default=False, help='just do a dry run i.e. print commands instead of actually executing them?, default is no')
    parser.add_argument('--automate', dest='automate', action='store_true', default=False, help='automatically progress to the next refinement level?, default is no')
    parser.add_argument('--final_level', metavar='final_level', type=int, action='store', default=3, help='final level of refinement till which the simulation should run; default is 3')

    args = parser.parse_args()

    # ------------- paths, dict, etc. set up -------------------------------
    root_dir = '/nobackup/aachary2/' # '/Users/acharyya/Work/astro/'
    args.halo_name = 'halo_' + args.halo
    args.sim_dir = root_dir + args.sim_dir
    args.halo_dir = args.sim_dir + '/' + args.halo_name
    args.template_dir = args.sim_dir + '/' + 'halo_template'

    Path(args.halo_dir).mkdir(parents=True, exist_ok=True) # creating the directory structure, if doesn't exist already

    #----------special settings for ldan queue--------
    if args.queue == 'ldan':
        args.proc = 'ldan'
        args.nnodes = 1
        args.ncores = None
    # ----------special settings for endeavour queue--------
    elif args.queue[:2] == 'e_':
        args.proc = 'cas_end'

    # ----------settings for pleiades--------
    procs_dir = {'san': (16, 32), 'ivy': (20, 64), 'has': (24, 128), 'bro': (28, 128), 'bro_ele': (28, 128),
                 'sky_ele': (40, 192), 'cas_ait': (40, 192), 'ldan': (16, 750), 'cas_end': (28, 185)}  # (nnodes, mem) for each proc, from https://www.nas.nasa.gov/hecc/support/kb/pbs-resource-request-examples_188.html
    max_hours_dict = defaultdict(lambda: 120, low=4, normal=8, long=120, e_long=72, e_normal=8, e_vlong=600, e_debug=2, debug=2, devel=2, ldan=72)  # from https://www.nas.nasa.gov/hecc/support/kb/pbs-job-queue-structure_187.html
    ncores = args.ncores if args.ncores is not None else procs_dir[args.proc][0]
    memory = args.memory if args.memory is not None else str(procs_dir[args.proc][1]) + 'GB'  # minimum memory per node; by default the entire node me is allocated, therefore it is redundant to specify mem as the highest available memory per node
    nhours = args.nhours if args.nhours is not None else '02' if args.dryrun or args.queue == 'devel' else '%02d' % (max_hours_dict[args.queue])

    # ----------- read halo catalogue, to get center -------------------
    halos = Table.read('/nobackup/jtumlins/CGM_bigbox/25Mpc_256_shielded-L0/BigBox_z2_rockstar/out_0.list', format='ascii', header_start=0)
    halos.sort('Mvir')
    halos.reverse()

    index = [halos['ID'] == int(args.halo[:4])]
    thishalo = halos[index]
    args.center = np.array([thishalo['X'][0]/25., thishalo['Y'][0]/25., thishalo['Z'][0]/25.]) # divided by 25 to convert Mpc units to code units
    rvir = np.max([thishalo['Rvir'][0], 200.])
    print('Starting halo', thishalo['ID'][0], 'centered at =', args.center, 'with Rvir =', rvir, 'kpc', 'at refinement level', args.level)

    # -----------run MUSIC and Enzo -------------------
    #if args.automate: run_multiple_enzo_levels(args.nnodes, ncores, nhours, args)
    #else:
    wrap_run_enzo(ncores, nhours, args)

    print('Completed in %s minutes' % datetime.timedelta(seconds=(time.time() - start_time)))
