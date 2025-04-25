##!/usr/bin/env python3

"""

    Filename :   util.py
    Notes :      Contains various generic utility functions and classes used by the other scripts in FOGGHORN, including a function to parse args
    Authors :    Ayan, Cassi
    Created: 06-12-24
    Last modified: 07-22-24 by Cassi

"""

from foggie.fogghorn.header import *

# --------------------------------------------------------------------------------------------------------------------
def need_to_make_this_plot(output_filename, args):
    '''
    Determines whether a figure with this name already exists, and if so, should it be over-written
    :return boolean
    '''
    if os.path.exists(output_filename):
        if not args.silent: print(output_filename + ' already exists.')
        if args.clobber:
            if not args.silent: print('But we will re-make it...')
            return True
        else:
            if not args.silent: print('So we will skip it.')
            return False
    else:
        if not args.silent: print('About to make ' + output_filename + '...')
        return True

# -------------------------------------------------------------------------------------------
def print_mpi(string, args):
    '''
    Function to print corresponding to each mpi thread
    '''
    comm = MPI.COMM_WORLD
    myprint('[' + str(comm.rank) + '] {' + subprocess.check_output(['uname -n'],shell=True)[:-1].decode("utf-8") + '} ' + string + '\n', args)

# -------------------------------------------------------------------------------------------
def print_master(string, args):
    '''
    Function to print only if on the head node/thread
    '''
    comm = MPI.COMM_WORLD
    if comm.rank == 0: myprint('[' + str(comm.rank) + '] ' + string + '\n', args)

# --------------------------------------------------------------------------------------------
def fix_time_format(text, keyword):
    '''
     Function to modify the way time is formatted in print statements
    '''
    arr = text.split(' ' + keyword)
    pre_time = ' '.join(arr[0].split(' ')[:-1])
    this_time = float(arr[0].split(' ')[-1])
    post_time = ' '.join(arr[1].split(' '))
    text = pre_time + ' %s' % (datetime.timedelta(minutes=this_time)) + post_time

    return text
# -------------------------------------------------------------------------------------------
def myprint(text, args):
    '''
    Function to direct the print output to stdout or a file, depending upon user args
    '''
    if not isinstance(text, list) and not text[-1] == '\n': text += '\n'
    if 'minutes' in text: text = fix_time_format(text, 'minutes')
    elif 'mins' in text: text = fix_time_format(text, 'mins')

    if not args.silent: print(text)

# --------------------------------------------------------------------
def get_density_cut(t):
    '''
    Function to get density cut based on Cassi's paper. The cut is a function of ime.
    if z > 0.5: rho_cut = 2e-26 g/cm**3
    elif z < 0.25: rho_cut = 2e-27 g/cm**3
    else: linearly from 2e-26 to 2e-27 from z = 0.5 to z = 0.25
    Takes time in Gyr as input
    '''
    t1, t2 = 8.628, 10.754 # Gyr; corresponds to z1 = 0.5 and z2 = 0.25
    rho1, rho2 = 2e-26, 2e-27 # g/cm**3
    t = np.float64(t)
    rho_cut = np.piecewise(t, [t < t1, (t >= t1) & (t <= t2), t > t2], [rho1, lambda t: rho1 + (t - t1) * (rho2 - rho1) / (t2 - t1), rho2])
    return rho_cut

# -------------------------------------------------------------------------------
def get_df_from_ds(box, args, outfilename=None):
    '''
    Function to make a pandas dataframe from the yt dataset, including only the metallicity profile (for now),
    then writes dataframe to file for faster access in future
    This function is somewhat based on foggie.utils.prep_dataframe.prep_dataframe()
    :return: pandas dataframe
    '''
    # ------------- Set up paths and dicts -------------------
    Path(args.save_directory + 'txtfiles/').mkdir(parents=True, exist_ok=True)  # creating the directory structure, if doesn't exist already
    if outfilename is None: outfilename = args.save_directory + 'txtfiles/' + args.snap + '_df_metallicity_vs_rad_%s%s.txt' % (args.upto_text, args.density_cut_text)

    field_dict = {'rad': ('gas', 'radius_corrected'), 'density': ('gas', 'density'), 'mass': ('gas', 'mass'), 'metal': ('gas', 'metallicity'), 'temp': ('gas', 'temperature'), \
                  'vrad': ('gas', 'radial_velocity_corrected'), 'vdisp': ('gas', 'velocity_dispersion_3d'), 'phi_L': ('gas', 'angular_momentum_phi'), 'theta_L': ('gas', 'angular_momentum_theta'), \
                  'volume': ('gas', 'volume'), 'phi_disk': ('gas', 'phi_pos_disk'), 'theta_disk': ('gas', 'theta_pos_disk')} # this is a superset of many quantities, only a few of these quantities will be extracted from the dataset to build the dataframe

    unit_dict = {'rad': 'kpc', 'rad_re': '', 'density': 'g/cm**3', 'metal': r'Zsun', 'temp': 'K', 'vrad': 'km/s',
                 'phi_L': 'deg', 'theta_L': 'deg', 'PDF': '', 'mass': 'Msun', 'stars_mass': 'Msun',
                 'ystars_mass': 'Msun', 'ystars_age': 'Gyr', 'gas_frac': '', 'gas_time': 'Gyr', 'volume': 'pc**3',
                 'phi_disk': 'deg', 'theta_disk': 'deg', 'vdisp': 'km/s'}

    # ------------- Write new pandas df file -------------------
    if not os.path.exists(outfilename) or args.clobber:
        print(outfilename + ' does not exist. Creating afresh..')

        if args.use_density_cut:
            rho_cut = get_density_cut(args.current_time)  # based on Cassi's CGM-ISM density cut-off
            box = box.cut_region(['obj["gas", "density"] > %.1E' % rho_cut])
            print('Imposing a density criteria to get ISM above density', rho_cut, 'g/cm^3')

        df = pd.DataFrame()
        fields = ['rad', 'metal'] # only the relevant properties
        if args.weight is not None: fields += [args.weight]

        for index, field in enumerate(fields):
            print('Loading property: ' + field + ', which is ' + str(index + 1) + ' of the ' + str(len(fields)) + ' fields..')
            df[field] = box[field_dict[field]].in_units(unit_dict[field]).ndarray_view()

        df.to_csv(outfilename, sep='\t', index=None)
    else:
        # ------------- Read from existing pandas df file -------------------
        print('Reading from existing file ' + outfilename)
        try:
            df = pd.read_table(outfilename, delim_whitespace=True, comment='#')
        except pd.errors.EmptyDataError:
            print('File existed, but it was empty, so making new file afresh..')
            dummy_args = copy.deepcopy(args)
            dummy_args.clobber = True
            df = get_df_from_ds(box, dummy_args, outfilename=outfilename)

    df['log_metal'] = np.log10(df['metal'])

    return df

# ----------------------------------------------------------------------------
def weighted_quantile(values, quantiles, weight=None):
    '''
    Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of initial array
    :param old_style: if True, will correct output to be consistent with numpy.percentile.
    :return: numpy.array with computed quantiles.
    This function was adapted from https://stackoverflow.com/questions/21844024/weighted-percentile-using-numpy
    '''
    if weight is None: weight = np.ones(len(values))
    values = np.array(values)
    quantiles = np.array(quantiles)
    weight = np.array(weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), 'quantiles should be in [0, 1]'

    sorter = np.argsort(values)
    values = values[sorter]
    weight = weight[sorter]

    weighted_quantiles = np.cumsum(weight) - 0.5 * weight
    weighted_quantiles /= np.sum(weight)
    return np.interp(quantiles, weighted_quantiles, values)









