'''
Filename: get_mass_profile.py
Author: Cassi
Created: 10-21-19
Last modified: 10-21-19
Modified by Ayan Feb 2022

This file calculates and outputs to file radial profiles of enclosed stellar mass, dark matter mass,
gas mass, and total mass. The files it creates are located in halo_infos.
'''
from header import *
from util import *
import multiprocessing as multi

def parse_args():
    '''Parse command line arguments. Returns args object.
    NOTE: Need to move command-line argument parsing to separate file.'''

    parser = argparse.ArgumentParser(description='Calculates and saves to file a bunch of fluxes.')

    # Optional arguments:
    parser.add_argument('--halo', metavar='halo', type=str, action='store', \
                        help='Which halo? Default is 8508 (Tempest)')
    parser.set_defaults(halo='8508')

    parser.add_argument('--run', metavar='run', type=str, action='store', \
                        help='Which run? Default is nref11c_nref9f')
    parser.set_defaults(run='nref11c_nref9f')

    parser.add_argument('--output', metavar='output', type=str, action='store', \
                        help='Which output(s)? Options: Specify a single output (this is default' \
                        + ' and the default output is RD0036) or specify a range of outputs ' + \
                        'using commas to list individual outputs and dashes for ranges of outputs ' + \
                        '(e.g. "RD0020-RD0025" or "DD1341,DD1353,DD1600-DD1700", no spaces!)')
    parser.set_defaults(output='RD0036')

    parser.add_argument('--output_step', metavar='output_step', type=int, action='store', \
                        help='If you want to do every Nth output, this specifies N. Default: 1 (every output in specified range)')
    parser.set_defaults(output_step=1)

    parser.add_argument('--system', metavar='system', type=str, action='store', \
                        help='Which system are you on? Default is ayan_local')
    parser.set_defaults(system='ayan_local')

    parser.add_argument('--pwd', dest='pwd', action='store_true',
                        help='Just use the working directory?, Default is no')
    parser.set_defaults(pwd=False)

    parser.add_argument('--local', dest='local', action='store_true',
                        help='Are the simulation files stored locally? Default is no')
    parser.set_defaults(local=False)

    parser.add_argument('--nproc', metavar='nproc', type=int, action='store', \
                        help='How many processes do you want? Default is 1 ' + \
                        '(no parallelization), if multiple outputs and multiple processors are' + \
                        ' specified, code will run one output per processor')
    parser.set_defaults(nproc=1)

    args = parser.parse_args()
    return args

def set_table_units(table):
    '''Sets the units for the table. Note this needs to be updated whenever something is added to
    the table. Returns the table.'''

    table_units = {'redshift':None,'snapshot':None,'radius':'kpc','total_mass':'Msun', \
             'dm_mass':'Msun', 'stars_mass':'Msun', 'young_stars_mass':'Msun', 'old_stars_mass':'Msun', \
             'sfr':'Msun/yr', 'gas_mass':'Msun', 'gas_metal_mass':'Msun', 'gas_H_mass':'Msun', 'gas_HI_mass':'Msun', \
             'gas_HII_mass':'Msun', 'gas_CII_mass':'Msun', 'gas_CIII_mass':'Msun', 'gas_CIV_mass':'Msun', \
             'gas_OVI_mass':'Msun', 'gas_OVII_mass':'Msun', 'gas_MgII_mass':'Msun', 'gas_SiII_mass':'Msun', \
             'gas_SiIII_mass':'Msun', 'gas_SiIV_mass':'Msun', 'gas_NeVIII_mass':'Msun'}
    for key in table.keys():
        table[key].unit = table_units[key]
    return table

def calc_masses(ds, snap, zsnap, refine_width_kpc, tablename):
    """Computes the mass enclosed in spheres centered on the halo center.
    Takes the dataset for the snapshot 'ds', the name of the snapshot 'snap', the redshfit of the
    snapshot 'zsnap', and the width of the refine box in kpc 'refine_width_kpc'
    and does the calculation, then writes a hdf5 table out to 'tablename'. If 'ions' is True then it
    computes the enclosed mass for various gas-phase ions.
    """

    halo_center_kpc = ds.halo_center_kpc

    # Set up table of everything we want
    # NOTE: Make sure table units are updated when things are added to this table!
    data = Table(names=('redshift', 'snapshot', 'radius', 'total_mass', 'dm_mass', \
                        'stars_mass', 'young_stars_mass', 'old_stars_mass', 'sfr', 'gas_mass', \
                        'gas_metal_mass'), \
                 dtype=('f8', 'S6', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8'))

    # Define the radii of the spheres where we want to calculate mass enclosed
    radii = refine_width_kpc * np.logspace(-4, 0, 250)

    # Initialize first sphere
    print('Beginning calculation for snapshot', snap)
    print('Loading field arrays')
    sphere = ds.sphere(halo_center_kpc, radii[-1])

    gas_mass = sphere['gas','cell_mass'].in_units('Msun').v
    gas_metal_mass = sphere['gas','metal_mass'].in_units('Msun').v

    dm_mass = sphere['dm','particle_mass'].in_units('Msun').v
    stars_mass = sphere['stars','particle_mass'].in_units('Msun').v
    young_stars_mass = sphere['young_stars','particle_mass'].in_units('Msun').v
    old_stars_mass = sphere['old_stars','particle_mass'].in_units('Msun').v
    gas_radius = sphere['gas','radius_corrected'].in_units('kpc').v
    dm_radius = sphere['dm','radius_corrected'].in_units('kpc').v
    stars_radius = sphere['stars','radius_corrected'].in_units('kpc').v
    young_stars_radius = sphere['young_stars','radius_corrected'].in_units('kpc').v
    old_stars_radius = sphere['old_stars','radius_corrected'].in_units('kpc').v

    # Loop over radii
    for i in range(len(radii)):

        if (i%10==0): print("Computing radius " + str(i) + "/" + str(len(radii)-1) + \
                            " for snapshot " + snap)

        # Cut the data interior to this radius
        gas_mass_enc = np.sum(gas_mass[gas_radius <= radii[i]])
        gas_metal_mass_enc = np.sum(gas_metal_mass[gas_radius <= radii[i]])

        dm_mass_enc = np.sum(dm_mass[dm_radius <= radii[i]])
        stars_mass_enc = np.sum(stars_mass[stars_radius <= radii[i]])
        young_stars_mass_enc = np.sum(young_stars_mass[young_stars_radius <= radii[i]])
        old_stars_mass_enc = np.sum(old_stars_mass[old_stars_radius <= radii[i]])
        sfr_enc = young_stars_mass_enc/1.e7
        total_mass_enc = gas_mass_enc + dm_mass_enc + stars_mass_enc

        # Add everything to the table
        data.add_row([zsnap, snap, radii[i], total_mass_enc, dm_mass_enc, stars_mass_enc, \
                        young_stars_mass_enc, old_stars_mass_enc, sfr_enc, gas_mass_enc, gas_metal_mass_enc])

    # Save to file
    data = set_table_units(data)
    data.write(tablename + '.hdf5', path='all_data', serialize_meta=True, overwrite=True)

    return "Masses have been calculated for snapshot" + snap + "!"

def load_and_calculate(system, foggie_dir, run_dir, track, halo_c_v_name, snap, tablename):
    '''This function loads a specified snapshot 'snap' located in the 'run_dir' within the
    'foggie_dir', the halo track 'track', the halo center file 'halo_c_v_name', and the name
    of the table to output 'tablename', then does the calculation on the loaded snapshot.
    If 'ions' is True then it computes the enclosed mass of various gas-phase ions.'''

    snap_name = foggie_dir + run_dir + snap + '/' + snap
    if (system=='pleiades_cassi'):
        print('Copying directory to /tmp')
        snap_dir = '/tmp/' + snap
        shutil.copytree(foggie_dir + run_dir + snap, snap_dir)
        snap_name = snap_dir + '/' + snap
    ds, refine_box = foggie_load(snap_name, track, halo_c_v_name=halo_c_v_name)
    refine_width_kpc = ds.quan(ds.refine_width, 'kpc')
    zsnap = ds.get_parameter('CosmologyCurrentRedshift')

    # Do the actual calculation
    message = calc_masses(ds, snap, zsnap, refine_width_kpc, tablename)
    if (system=='pleiades_cassi'):
        print('Deleting directory from /tmp')
        shutil.rmtree(snap_dir)
    print(message)
    print(str(datetime.datetime.now()))

def get_outputs_in_between(range, step):
    first, last = [item for item in range.split('-')]
    first, last = int(first[2:]), int(last[2:])
    output_numbers = np.arange(first, last, step)
    outputs = [range[:2] + '%04d' % item for item in output_numbers]
    return outputs

if __name__ == "__main__":
    args = parse_args()
    foggie_dir, output_dir, run_dir, code_path, trackname, haloname, spectra_dir, infofile = get_run_loc_etc(args)

    # Set directory for output location, making it if necessary
    prefix = output_dir + 'masses_halo_00' + args.halo + '/' + args.run + '/'
    if not (os.path.exists(prefix)): os.system('mkdir -p ' + prefix)

    halo_c_v_name = code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/halo_c_v'

    # Build output list
    if (',' in args.output):
        outs = args.output.split(',')
        for i in range(len(outs)):
            if ('-' in outs[i]):
                insert = get_outputs_in_between(outs[i], args.output_step)
                outs = outs[:i] + insert + outs[i + 1:]
    elif ('-' in args.output): outs = get_outputs_in_between(args.output, args.output_step)
    else: outs = [args.output]

    # Loop over outputs, for either single-processor or parallel processor computing
    if (args.nproc==1):
        for i in range(len(outs)):
            snap = outs[i]
            # Make the output table name for this snapshot
            tablename = prefix + snap + '_masses'
            # Do the actual calculation
            load_and_calculate(args.system, foggie_dir, run_dir, trackname, halo_c_v_name, snap, tablename)
    else:
        # Split into a number of groupings equal to the number of processors
        # and run one process per processor
        for i in range(len(outs)//args.nproc):
            threads = []
            for j in range(args.nproc):
                snap = outs[args.nproc*i+j]
                tablename = prefix + snap + '_masses'
                threads.append(multi.Process(target=load_and_calculate, \
			       args=(args.system, foggie_dir, run_dir, trackname, halo_c_v_name, snap, tablename)))
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        # For any leftover snapshots, run one per processor
        threads = []
        for j in range(len(outs)%args.nproc):
            snap = outs[-(j+1)]
            tablename = prefix + snap + '_masses'
            threads.append(multi.Process(target=load_and_calculate, \
			   args=(args.system, foggie_dir, run_dir, trackname, halo_c_v_name, snap, tablename)))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    print(str(datetime.datetime.now()))
    print("All snapshots finished!")
