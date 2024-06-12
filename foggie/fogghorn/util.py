##!/usr/bin/env python3

"""

    Filename :   util.py
    Notes :      Contains various generic utility functions and classes used by the other scripts in FOGGHORN, including a function to parse args
    Authors :    Ayan, 
    Created: 06-12-24
    Last modified: 06-12-24 by Ayan

"""

from foggie.fogghorn.header import *

# --------------------------------------------------------------------------------------------------------------------
def parse_args():
    '''Parse command line arguments. Returns args object.'''

    parser = argparse.ArgumentParser(description='Produces analysis plots for FOGGHORN runs.')

    # Optional arguments:
    parser.add_argument('--directory', metavar='directory', type=str, action='store', default='', help='What is the directory of the enzo outputs you want to make plots of?')
    parser.add_argument('--save_directory', metavar='save_directory', type=str, action='store', default=None, help='Where do you want to store the plots, if different from where the outputs are stored?')
    parser.add_argument('--output', metavar='output', type=str, action='store', default=None, help='If you want to make the plots for specific output/s then specify those here separated by comma (e.g., DD0030,DD0040). Otherwise (default) it will make plots for ALL outputs in that directory')
    parser.add_argument('--trackfile', metavar='trackfile', type=str, action='store', default=None, help='What is the directory of the track file for this halo?\n' + 'This is needed to find the center of the galaxy of interest.')
    parser.add_argument('--pwd', dest='pwd', action='store_true', default=False, help='Just use the working directory?, Default is no')
    parser.add_argument('--nproc', metavar='nproc', type=int, action='store', default=1, help='How many processes do you want? Default is 1 (no parallelization), if multiple processors are specified, code will run one output per processor')
    parser.add_argument('--rockstar_directory', metavar='rockstar_directory', type=str, action='store', default=None, help='What is the directory where your rockstar outputs are located?')

    parser.add_argument('--clobber', dest='clobber', action='store_true', default=False, help='Over-write existing plots? Default is no.')
    parser.add_argument('--silent', dest='silent', action='store_true', default=False, help='Suppress some generic pritn statements? Default is no.')
    parser.add_argument('--upto_kpc', metavar='upto_kpc', type=float, action='store', default=None, help='Limit analysis out to a certain physical kpc. By default it does the entire refine box.')
    parser.add_argument('--docomoving', dest='docomoving', action='store_true', default=False, help='Consider the input upto_kpc as a comoving quantity? Default is No.')
    parser.add_argument('--weight', metavar='weight', type=str, action='store', default=None, help='Name of quantity to weight the metallicity by. Default is None i.e., no weighting.')
    parser.add_argument('--projection', metavar='projection', type=str, action='store', default='z', help='Which projection do you want to plot, i.e., which axis is your line of sight? Default is z; but user can input multiple comma-separated values')
    parser.add_argument('--disk_rel', dest='disk_rel', action='store_true', default=False, help='Consider projection plots w.r.t the disk rather than the box edges? Be aware that this will turn on disk_relative=True while reading in each snapshot whic might slow down the loading of data. Default is No.')
    parser.add_argument('--use_density_cut', dest='use_density_cut', action='store_true', default=False, help='Impose a density cut to get just the disk? Default is no.')
    parser.add_argument('--nbins', metavar='nbins', type=int, action='store', default=100, help='Number of bins to use for the metallicity histogram plot. Default is 100')
    parser.add_argument('--use_cen_smoothed', dest='use_cen_smoothed', action='store_true', default=False, help='use Cassis new smoothed center file?, default is no')

    # The following is for the user to choose which plots they want
    parser.add_argument('--all_plots', dest='all_plots', action='store_true', default=False, help='Make all the plots? Default is no.')
    parser.add_argument('--all_sf_plots', dest='all_sf_plots', action='store_true', default=False, help='Make all star formation plots? Default is no.')
    parser.add_argument('--all_fb_plots', dest='all_fb_plots', action='store_true', default=False, help='Make all feedback plots? Default is no.')
    parser.add_argument('--all_vis_plots', dest='all_vis_plots', action='store_true', default=False, help='Make all visualisation plots? Default is no.')
    parser.add_argument('--all_metal_plots', dest='all_metal_plots', action='store_true', default=False, help='Make all resolved metallicity plots? Default is no.')
    parser.add_argument('--all_pop_plots', dest='all_pop_plots', action='store_true', default=False, help='Make all population plots? Default is no.')
    parser.add_argument('--make_plots', metavar='make_plots', type=str, action='store', default='', help='Which plots to make? Comma-separated names of the plotting routines to call. Default is none.')

    # The following three args are used for backward compatibility, to find the trackfile for production runs, if a trackfile has not been explicitly specified
    parser.add_argument('--system', metavar='system', type=str, action='store', default='ayan_local', help='Which system are you on? This is used only when trackfile is not specified. Default is ayan_local')
    parser.add_argument('--halo', metavar='halo', type=str, action='store', default='8508', help='Which halo? Default is Tempesxt. This is used only when trackfile is not specified.')
    parser.add_argument('--run', metavar='run', type=str, action='store', default='nref11c_nref9f', help='Which run? Default is nref11c_nref9f. This is used only when trackfile is not specified.')

    # ------- wrap up and processing args ------------------------------
    args = parser.parse_args()
    args.projection_arr = [item for item in args.projection.split(',')]
    args.plots_asked_for = [item for item in args.make_plots.split(',')]

    return args

# -------------------------------------------------------------------------------------------
def print_mpi(string, args):
    '''
    Function to print corresponding to each mpi thread
    '''
    comm = MPI.COMM_WORLD
    myprint_orig('[' + str(comm.rank) + '] {' + subprocess.check_output(['uname -n'],shell=True)[:-1].decode("utf-8") + '} ' + string + '\n', args)

# -------------------------------------------------------------------------------------------
def print_master(string, args):
    '''
    Function to print only if on the head node/thread
    '''
    comm = MPI.COMM_WORLD
    if comm.rank == 0: myprint_orig('[' + str(comm.rank) + '] ' + string + '\n', args)

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
def myprint_orig(text, args):
    '''
    Function to direct the print output to stdout or a file, depending upon user args
    '''
    if not isinstance(text, list) and not text[-1] == '\n': text += '\n'
    if 'minutes' in text: text = fix_time_format(text, 'minutes')
    elif 'mins' in text: text = fix_time_format(text, 'mins')

    if not args.silent: print(text)

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

# ----------------------------------------------------------------------------
def generate_plot_filename(quantity, args):
    '''
    Generates filename for a plot that is about to be made
    This way the nomenclature is consistent
    '''
    output_filename_dict = {'young_stars_density_projection':args.snap + '_Projection_' + args.projection + '_young_stars3_cic.png', \
                            'KS_relation': args.snap + '_KS-relation.png', \
                            'outflow_rates': args.snap + '_outflows.png', \
                            'gas_density_projection': args.snap + '_Projection_' + args.projection + '_density.png', \
                            'gas_metallicity_projection': args.snap + '_gas_metallicity_projection_' + args.projection_text + args.upto_text + args.density_cut_text + '.png', \
                            'edge_visualizations': args.snap + '_Projection_disk-x_temperature_density.png', \
                            'gas_metallicity_resolved_MZR': args.snap + '_resolved_gas_MZR' + args.upto_text + args.density_cut_text + '.png', \
                            'gas_metallicity_histogram': args.snap + '_gas_metallicity_histogram' + args.upto_text + args.density_cut_text + '.png', \
                            'gas_metallicity_radial_profile': args.snap + '_gas_metallicity_radial_profile' + args.upto_text + args.density_cut_text + '.png', \
                            'plot_SFMS': 'SFMS.png', \
                            'plot_SMHM': 'SMHM.png', \
                            'plot_MZR': 'MZR.png'}

    output_filename = args.save_directory + '/' + output_filename_dict[quantity]
    return output_filename









