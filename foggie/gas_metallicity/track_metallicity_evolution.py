##!/usr/bin/env python3

"""

    Title :      track_metallicity_evolution
    Notes :      To TRACK ambient gas metallicity around young (< 10Myr) stars as function of redshift, write to file, and plot
    Input :      txt file with all young star properties for each snapshot of each halo (generated by filter_star_properties.py)
    Output :     One pandas dataframe per halo as a txt file (Or fits file?)
    Author :     Ayan Acharyya
    Started :    July 2021
    Example :    run track_metallicity_evolution.py --system ayan_hd --halo 8508 --nocallback
    Example :    run track_metallicity_evolution.py --system ayan_pleiades --halo 8508 --nocallback

"""
from header import *
from util import *
from foggie.galaxy_mocks.mock_ifu.filter_star_properties import get_star_properties

# ----------------------------------------------------------------------------------
def write_list_file(Z_arr, m_arr, z_arr, filename, args):
    '''
    Function to write fits file with given arrays
    '''
    master_arr = []
    for index in range(len(Z_arr)): master_arr.append([z_arr[index], Z_arr[index], m_arr[index]]) # master_arr = [[redshift1, [Zstar1, Zstar2, ....], [mstar1, mstar2, ....]], [redshift2, [Zstar1, Zstar2, Zstar3,..], [mstar1, mstar2, mstar3...]]]
    np.savetxt(filename, master_arr, fmt='%s')
    myprint('Written file ' + filename, args)
    return

# ----------------------------------------------------------------------------------
def read_list_file(filename, args):
    '''
    Function to read in file with given arrays
    '''
    z_arr, Z_arr, m_arr =[], [], []
    lines = np.atleast_1d(np.genfromtxt(filename, delimiter='\n', dtype=str))
    for index, line in enumerate(lines):
        myprint('Reading line (i.e. snapshot) ' + str(index + 1) + ' of ' + str(len(lines)) + '..', args)
        arr = line.split('] [')
        thisZ = arr[0].split(', ')
        thism = arr[1].split(', ')

        thisz = float(thisZ[0].split(' [')[0])
        thisZ[0] = thisZ[0].split(' [')[1]
        thism[-1] = thism[-1].split(']')[0]

        thisZ_arr = np.array([float(i) for i in thisZ])
        thism_arr = np.array([float(i) for i in thism])

        z_arr.append(thisz)
        Z_arr.append(thisZ_arr)
        m_arr.append(thism_arr)

    return np.array(z_arr), np.array(Z_arr), np.array(m_arr)

# ----------------------------------------------------------------------------------
def read_metallicity(args):
    '''
    Function to read in the properties of young stars and select the metallicity column
    '''
    infilename = args.output_dir + 'txtfiles/' + args.output + '_young_star_properties.txt'

    # ----------------------Reading in simulation data-------------------------------------------
    if not os.path.exists(infilename):
        myprint(infilename + 'does not exist. Calling get_star_properties() first..', args)
        dummy = get_star_properties(args) # this creates the infilename

    paramlist = pd.read_table(infilename, delim_whitespace=True, comment='#')
    Z_arr = paramlist['gas_metal'].values
    m_arr = paramlist['mass'].values
    return Z_arr.tolist(), m_arr.tolist()

# ----------------------------------------------------------------------------------
def assimilate_this_halo(args):
    '''
    Function to assimilate metallicities of all young stars of all snapshots for a given halo and write to fits file
    :return: a fits file
    '''
    foggie_dir, output_dir, run_loc, code_path, trackname, haloname, spectra_dir, infofile = get_run_loc_etc(args)
    outfilename = output_dir + 'txtfiles/' + args.halo + '_Z_vs_z_allsnaps.txt'

    if args.nocallback: # only consider those snapshots which ALREADY have young star properties filtered out by filter_star_properties.py
        list_of_sims = []
        snashot_paths = glob.glob(args.output_dir + 'txtfiles/*_young_star_properties.txt')
        snapshots = [item.split('/')[-1][:6] for item in snashot_paths]
        for thissnap in snapshots: list_of_sims.append([args.halo, thissnap])
        myprint('Only using the ' + str(len(snapshots)) + ' found that already exist, due to --nocallback', args)
    else:
        list_of_sims = get_all_sims_for_this_halo(args) # consider ALL snapshots of this halo available in storage and call filter_star_properties.py if a snapshot has not already been filtered

    if not os.path.exists(outfilename) or args.clobber:
        Z_arr, z_arr, m_arr = [], [], []
        # ----------looping over snapshots----------------------
        for index, this_sim in enumerate(list_of_sims):
            myprint('Doing snapshot ' + this_sim[1] + '; ' + str(index+1) + ' of ' + str(len(list_of_sims)), args)
            args = parse_args(this_sim[0], this_sim[1], fast=True) # use fast=True if you do not need to compute args.halo_center
            if type(args) is tuple:
                args, ds, refine_box = args  # if the sim has already been loaded in, in order to compute the box center (via utils.pull_halo_center()), then no need to do it again
                myprint('ds ' + str(ds) + ' for halo ' + str(this_sim[0]) + ' was already loaded at some point by utils; using that loaded ds henceforth', args)

            this_Z_arr, this_m_arr = read_metallicity(args)

            this_z = pull_halo_redshift(args)
            if this_z == -99: this_z = ds.current_redshift  # if this snapshot is not yet in the halo catalog

            Z_arr.append(this_Z_arr)
            z_arr.append(this_z)
            m_arr.append(this_m_arr)

        write_list_file(Z_arr, m_arr, z_arr, outfilename, args)
    else:
        myprint(outfilename + ' already exists; use --clobber to overwrite', args)
        z_arr, Z_arr, m_arr = read_list_file(outfilename, args)

    return z_arr, Z_arr, m_arr

# -----------------------------------------------------------------------------------
if __name__ == '__main__':
    start_time = time.time()

    args = parse_args('8508', 'RD0042', fast=True)
    if type(args) is tuple: args = args[0] # if the sim has already been loaded in, in order to compute the box center (via utils.pull_halo_center()), then no need to do it again

    if args.do_all_halos: list_of_halos = get_all_halos(args)
    else: list_of_halos = [args.halo]

    # ----------looping over halos----------------------
    for index, this_halo in enumerate(list_of_halos):
        myprint('Doing halo ' + this_halo + '; ' + str(index+1) + ' of ' + str(len(list_of_halos)), args)
        args.halo = this_halo
        z_arr, Z_arr, m_arr = assimilate_this_halo(args)

    myprint('All halos done in %s minutes' % ((time.time() - start_time) / 60), args)