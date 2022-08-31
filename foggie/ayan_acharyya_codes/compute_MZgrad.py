#!/usr/bin/env python3

"""

    Title :      compute_MZgrad
    Notes :      Compute mass - metallicity gradient relation for a given FOGGIE galaxy
    Output :     txt file storing all the gradients & mass plus, optionally, Z profile plots
    Author :     Ayan Acharyya
    Started :    Feb 2022
    Examples :   run compute_MZgrad.py --system ayan_local --halo 8508 --output RD0030 --upto_re 3 --xcol rad_re --keep --weight mass
                 run compute_MZgrad.py --system ayan_local --halo 8508 --output RD0030 --upto_kpc 10 --xcol rad --keep --weight mass
                 run compute_MZgrad.py --system ayan_pleiades --halo 8508 --upto_re 3 --xcol rad_re --do_all_sims --weight mass --write_file --noplot
                 run compute_MZgrad.py --system ayan_pleiades --halo 8508 --upto_kpc 10 --xcol rad --do_all_sims --weight mass --write_file --noplot

"""
from header import *
from util import *
from datashader_movie import *
from uncertainties import ufloat, unumpy
from yt.utilities.physical_ratios import metallicity_sun
start_time = time.time()

# -------------------------------------------------------------------------------
def calc_masses(ds, snap, refine_width_kpc, tablename, get_gas_profile=False):
    """Computes the mass enclosed in spheres centered on the halo center.
    Takes the dataset for the snapshot 'ds', the name of the snapshot 'snap', the redshfit of the
    snapshot 'zsnap', and the width of the refine box in kpc 'refine_width_kpc'
    and does the calculation, then writes a hdf5 table out to 'tablename'. If 'ions' is True then it
    computes the enclosed mass for various gas-phase ions.
    This is mostly copied from Cassi's get_mass_profile.calc_mass(), but for a shortened set of parameters, to save runtime
    """

    halo_center_kpc = ds.halo_center_kpc

    # Set up table of everything we want
    # NOTE: Make sure table units are updated when things are added to this table!
    if get_gas_profile: data = Table(names=('radius', 'stars_mass', 'gas_mass', 'gas_metal_mass'), dtype=('f8', 'f8', 'f8', 'f8'))
    else: data = Table(names=('radius', 'stars_mass'), dtype=('f8', 'f8'))

    # Define the radii of the spheres where we want to calculate mass enclosed
    radii = refine_width_kpc * np.logspace(-4, 0, 250)

    # Initialize first sphere
    print('Loading field arrays for snapshot', snap)
    sphere = ds.sphere(halo_center_kpc, radii[-1])

    if get_gas_profile:
        gas_mass = sphere['gas','cell_mass'].in_units('Msun').v
        gas_metal_mass = sphere['gas','metal_mass'].in_units('Msun').v
        gas_radius = sphere['gas', 'radius_corrected'].in_units('kpc').v

    stars_mass = sphere['stars','particle_mass'].in_units('Msun').v
    stars_radius = sphere['stars','radius_corrected'].in_units('kpc').v

    # Loop over radii
    for i in range(len(radii)):
        if (i%10==0): print('Computing radius ' + str(i) + '/' + str(len(radii)-1) + ' for snapshot ' + snap)

        # Cut the data interior to this radius
        if get_gas_profile:
            gas_mass_enc = np.sum(gas_mass[gas_radius <= radii[i]])
            gas_metal_mass_enc = np.sum(gas_metal_mass[gas_radius <= radii[i]])
        stars_mass_enc = np.sum(stars_mass[stars_radius <= radii[i]])

        # Add everything to the table
        if get_gas_profile: data.add_row([radii[i], stars_mass_enc, gas_mass_enc, gas_metal_mass_enc])
        else: data.add_row([radii[i], stars_mass_enc])

    # Save to file
    table_units = {'radius':'kpc', 'stars_mass':'Msun', 'gas_mass':'Msun', 'gas_metal_mass':'Msun'}
    for key in data.keys(): data[key].unit = table_units[key]

    data.write(tablename + '.hdf5', path='all_data', serialize_meta=True, overwrite=True)
    print('Masses have been calculated for snapshot' + snap)

# --------------------------------------------------------------------------------
def get_re_from_stars(ds, args):
    '''
    Function to determine the effective radius of stellar disk, based on the stellar mass profile, given a dataset
    Returns the effective radius in kpc
    '''
    re_hmr_factor = 2.0 # from the Illustris group (?)
    foggie_dir, output_dir, run_dir, code_path, trackname, haloname, spectra_dir, infofile = get_run_loc_etc(args)
    prefix = '/'.join(output_dir.split('/')[:-2]) + '/' + 'mass_profiles/' + args.run + '/'
    tablename = prefix + args.output + '_masses.hdf5'

    if os.path.exists(tablename):
        print('Reading mass profile file', tablename)
    else:
        print('File not found:', tablename, '\n', 'Therefore computing mass profile now..')
        Path(prefix).mkdir(parents=True, exist_ok=True)  # creating the directory structure, if doesn't exist already
        refine_width_kpc = ds.quan(ds.refine_width, 'kpc')
        calc_masses(ds, args.output, refine_width_kpc, os.path.splitext(tablename)[0], get_gas_profile=args.get_gasmass)

    mass_profile = pd.read_hdf(tablename, key='all_data')
    mass_profile = mass_profile.sort_values('radius')
    total_mass = mass_profile['stars_mass'].iloc[-1]
    half_mass_radius = mass_profile[mass_profile['stars_mass'] <= total_mass/2]['radius'].iloc[-1]
    re = re_hmr_factor * half_mass_radius

    print('\nStellar-profile: Half mass radius for halo ' + args.halo + ' output ' + args.output + ' (z=%.1F' %(args.current_redshift) + ') is %.2F kpc' %(re))
    return re

# --------------------------------------------------------------------------------
def get_re_from_coldgas(gasprofile, args):
    '''
    Function to determine the effective radius of stellar disk, based on the cold gas profile, given a dataset
    Returns the effective radius in kpc
    '''
    re_hmr_factor = 1.0

    if args.output[:2] == 'DD' and args.output[2:] in gasprofile.keys(): # because cold gas profile is only present for all the DD outputs
        this_gasprofile = gasprofile[args.output[2:]]
        this_coldgas = this_gasprofile['cold']
        mass_profile = pd.DataFrame({'radius': this_coldgas['r'], 'coldgas':np.cumsum(this_coldgas['mass'])})
        mass_profile = mass_profile.sort_values('radius')
        total_mass = mass_profile['coldgas'].iloc[-1]
        half_mass_radius = mass_profile[mass_profile['coldgas'] <= total_mass/2]['radius'].iloc[-1]
        re = re_hmr_factor * half_mass_radius
        print('\nCold gas profile: Half mass radius for halo ' + args.halo + ' output ' + args.output + ' (z=%.1F' % (args.current_redshift) + ') is %.2F kpc' % (re))
    else:
        re = -99
        print('\nCold gas profile not found for halo ' + args.halo + ' output ' + args.output + '; therefore returning dummy re %d' % (re))

    return re

# -----------------------------------------------------------------------------
def get_disk_stellar_mass(args):
    '''
    Function to get the disk stellar mass for a given output, which is defined as the stellar mass contained within args.galrad, which can either be a fixed absolute size in kpc OR = args.upto_re*Re
    '''
    mass_filename = args.code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/masses_z-less-2.hdf5'
    upto_radius = args.galrad # should this be something else? 2*Re may be?

    if os.path.exists(mass_filename):
        print('Reading in', mass_filename)
        alldata = pd.read_hdf(mass_filename, key='all_data')
        thisdata = alldata[alldata['snapshot'] == args.output]

        if len(thisdata) == 0: # snapshot not found in masses less than z=2 file, so try greater than z=2 file
            mass_filename = mass_filename.replace('less', 'gtr')
            print('Could not find spanshot in previous file, now reading in', mass_filename)
            alldata = pd.read_hdf(mass_filename, key='all_data')
            thisdata = alldata[alldata['snapshot'] == args.output]

            if len(thisdata) == 0: # snapshot still not found in file
                print('Snapshot not found in either file. Returning bogus mass')
                return -999

        thisshell = thisdata[thisdata['radius'] <= upto_radius]
        if len(thisshell) == 0: # the smallest shell available in the mass profile is larger than the necessary radius within which we need the stellar mass
            if thisdata['radius'].iloc[0] <= 1.5*args.galrad:
                print('Smallest shell available in the mass profile is larger than args.galrad, taking the mass in the smallest shell as galaxy stellar mass')
                mstar = thisdata['stars_mass'].iloc[0] # assigning the mass of the smallest shell as the stellar mass
            else:
                print('Smallest shell avialable in mass profile is too small compared to args.galrad. Returning bogus mass')
                return -999
        else:
            mstar = thisshell['stars_mass'].values[-1] # radius is in kpc, mass in Msun
    else:
        print('File not found:', mass_filename)
        mstar = -999

    print('Stellar mass for halo ' + args.halo + ' output ' + args.output + ' (z=%.1F' %(args.current_redshift) + ') within ' + str(upto_radius) + ' kpc is %.2E' %(mstar))
    return mstar

# --------------------------------------------------------------------------------------
def bin_data(array, data, bins):
    '''
    Function to bin data based on another array
    '''
    bins_cen = bins[:-1] + np.diff(bins) / 2.
    indices = np.digitize(array, bins)
    binned_data, binned_err = [], []
    for i in range(1, len(bins)):  # assuming all values in array are within limits of bin, hence last cell of binned_data=nan
        thisdata = data[indices == i]
        mean_data, mean_err = np.mean(thisdata), np.std(thisdata)
        binned_data.append(mean_data)
        binned_err.append(mean_err)
    binned_data = np.array(binned_data)
    binned_err = np.array(binned_err)
    return bins_cen, binned_data, binned_err

# ---------------------------------------------------------------------------------
def fit_binned(df, xcol, ycol, x_bins, ax=None, is_logscale=False, color='maroon', weightcol=None):
    '''
    Function to overplot binned data on existing plot
    '''
    df['binned_cat'] = pd.cut(df[xcol], x_bins)

    if weightcol is not None:
        agg_func = lambda x: np.sum(x * df.loc[x.index, weightcol]) / np.sum(df.loc[x.index, weightcol]) # function to get weighted mean
        #agg_u_func = lambda x: np.sqrt(((np.sum(df.loc[x.index, weightcol] * x**2) / np.sum(df.loc[x.index, weightcol])) - (np.sum(x * df.loc[x.index, weightcol]) / np.sum(df.loc[x.index, weightcol]))**2) * (np.sum(df.loc[x.index, weightcol]**2)) / (np.sum(df.loc[x.index, weightcol])**2 - np.sum(df.loc[x.index, weightcol]**2))) # eq 6 of http://seismo.berkeley.edu/~kirchner/Toolkits/Toolkit_12.pdf
        agg_u_func = np.std
    else:
        agg_func, agg_u_func = np.mean, np.std

    y_binned = df.groupby('binned_cat', as_index=False).agg([(ycol, agg_func)])[ycol]
    y_u_binned = df.groupby('binned_cat', as_index=False).agg([(ycol, agg_u_func)])[ycol]
    if is_logscale: y_binned, y_u_binned = np.log10(y_binned.values), np.log10(y_u_binned.values)

    # ----------to plot mean binned y vs x profile--------------
    x_bin_centers = x_bins[:-1] + np.diff(x_bins) / 2
    linefit, linecov = np.polyfit(x_bin_centers, y_binned.flatten(), 1, cov=True, w=1/(y_u_binned.flatten())**2)

    Zgrad = ufloat(linefit[0], np.sqrt(linecov[0][0]))
    Zcen = ufloat(linefit[1], np.sqrt(linecov[1][1]))
    print('Upon radially binning: Inferred slope for halo ' + args.halo + ' output ' + args.output + ' is', Zgrad, 'dex/re' if 're' in args.xcol else 'dex/kpc')

    if ax is not None:
        ax.errorbar(x_bin_centers, y_binned, c=color, yerr=y_u_binned, lw=1)
        ax.scatter(x_bin_centers, y_binned, c=color, s=60)
        ax.plot(x_bin_centers, np.poly1d(linefit)(x_bin_centers), color='maroon', lw=1, ls='dashed')
        units = 'dex/re' if 're' in xcol else 'dex/kpc'
        ax.text(0.033, 0.25, 'Slope = %.2F ' % linefit[0] + units, color='maroon', transform=ax.transAxes, fontsize=args.fontsize)
        return ax
    else:
        return Zcen, Zgrad

# -----------------------------------------------------------
def plot_gradient(df, args, linefit=None):
    '''
    Function to plot the metallicity profile, along with the fitted gradient if provided
    Saves plot as .png
    '''
    weightby_text = '' if args.weight is None else '_wtby_' + args.weight
    upto_text = '_upto%.1Fkpc' % dummy_args.upto_kpc if dummy_args.upto_kpc is not None else '_upto%.1FRe' % dummy_args.upto_re
    outfile_rootname = 'datashader_log_metal_vs_%s%s%s.png' % (args.xcol,upto_text, weightby_text)
    if args.do_all_sims: outfile_rootname = 'z=*_' + outfile_rootname
    filename = args.fig_dir + outfile_rootname.replace('*', '%.5F' % (args.current_redshift))

    # ---------first, plot both cell-by-cell profile first, using datashader---------
    fig, ax = plt.subplots(figsize=(8,8))
    fig.subplots_adjust(hspace=0.05, wspace=0.05, right=0.95, top=0.95, bottom=0.1, left=0.1)
    artist = dsshow(df, dsh.Point(args.xcol, 'log_metal'), dsh.count(), norm='linear', x_range=(0, args.galrad / args.re if 're' in args.xcol else args.galrad), y_range=(args.ylim[0], args.ylim[1]), aspect = 'auto', ax=ax, cmap='Blues_r')#, shade_hook=partial(dstf.spread, px=1, shape='square')) # the 40 in alpha_range and `square` in shade_hook are to reproduce original-looking plots as if made with make_datashader_plot()

    # --------bin the metallicity profile and plot the binned profile-----------
    ax = fit_binned(df, args.xcol, 'metal', args.bin_edges, ax=ax, is_logscale=True, weightcol=args.weight)

    # ----------plot the fitted metallicity profile---------------
    if linefit is not None:
        fitted_y = np.poly1d(linefit)(args.bin_edges)
        ax.plot(args.bin_edges, fitted_y, color='darkblue', lw=2, ls='dashed')
        units = 'dex/re' if 're' in args.xcol else 'dex/kpc'
        plt.text(0.033, 0.2, 'Slope = %.2F ' % linefit[0] + units, color='darkblue', transform=ax.transAxes, fontsize=args.fontsize)

    # ----------tidy up figure-------------
    ax.xaxis = make_coordinate_axis(args.xcol, 0, args.galrad / args.re if 're' in args.xcol else args.galrad, ax.xaxis, args.fontsize, dsh=False, log_scale=False)
    ax.yaxis = make_coordinate_axis('metal', args.ylim[0], args.ylim[1], ax.yaxis, args.fontsize, dsh=False, log_scale=False)

    # ---------annotate and save the figure----------------------
    plt.text(0.033, 0.05, 'z = %.4F' % args.current_redshift, transform=ax.transAxes, fontsize=args.fontsize)
    plt.text(0.033, 0.1, 't = %.3F Gyr' % args.current_time, transform=ax.transAxes, fontsize=args.fontsize)
    plt.savefig(filename, transparent=False)
    myprint('Saved figure ' + filename, args)
    if not args.makemovie: plt.show(block=False)

    return fig

# -------------------------------
def fit_gradient(df, args):
    '''
    Function to linearly fit the (log) metallicity profile out to certain Re, given a dataframe containing metallicity profile
    Returns the fitted gradient with uncertainty
    '''

    if args.weight is None: linefit, linecov = np.polyfit(df[args.xcol], df['log_metal'], 1, cov=True)
    else: linefit, linecov = np.polyfit(df[args.xcol], df['log_metal'], 1, cov=True, w=df[args.weight])
    Zgrad = ufloat(linefit[0], np.sqrt(linecov[0][0]))
    Zcen = ufloat(linefit[1], np.sqrt(linecov[1][1]))
    print('Inferred slope for halo ' + args.halo + ' output ' + args.output + ' is', Zgrad, 'dex/re' if 're' in args.xcol else 'dex/kpc')

    return Zcen, Zgrad

# -------------------------------------------------------------------------------
def get_df_from_ds(ds, args):
    '''
    Function to make a pandas dataframe from the yt dataset, including only the metallicity profile,
    then writes dataframe to file for faster access in future
    This function is somewhat based on foggie.utils.prep_dataframe.prep_dataframe()
    :return: dataframe
    '''
    # -------------read/write pandas df file with ALL fields-------------------
    Path(args.output_dir + 'txtfiles/').mkdir(parents=True, exist_ok=True)  # creating the directory structure, if doesn't exist already
    outfilename = get_correct_tablename(args)

    if not os.path.exists(outfilename) or args.clobber:
        myprint(outfilename + ' does not exist. Creating afresh..', args)

        df = pd.DataFrame()
        fields = ['rad', 'metal'] # only the relevant properties
        if args.weight is not None: fields += [args.weight]

        for index, field in enumerate(fields):
            myprint('Doing property: ' + field + ', which is ' + str(index + 1) + ' of the ' + str(len(fields)) + ' fields..', args)
            df[field] = ds[field_dict[field]].in_units(unit_dict[field]).ndarray_view()

        df.to_csv(outfilename, sep='\t', index=None)
    else:
        myprint('Reading from existing file ' + outfilename, args)
        df = pd.read_table(outfilename, delim_whitespace=True, comment='#')
        df = df[df['rad'].between(0, args.galrad)] # in case this dataframe has been read in from a file corresponding to a larger chunk of the box

    df['log_metal'] = np.log10(df['metal'])
    df['rad_re'] = df['rad'] / args.re  # normalise by Re
    cols_to_extract = [args.xcol, 'log_metal']
    if args.weight is not None: cols_to_extract += [args.weight]
    df = df[cols_to_extract] # only keeping the columns that are needed to get Z gradient
    if 'metal' not in df: df['metal'] = 10 ** (df['log_metal'])

    return df

# -----main code-----------------
if __name__ == '__main__':
    dummy_args_tuple = parse_args('8508', 'RD0042')  # default simulation to work upon when comand line args not provided
    if type(dummy_args_tuple) is tuple: dummy_args = dummy_args_tuple[0] # if the sim has already been loaded in, in order to compute the box center (via utils.pull_halo_center()), then no need to do it again
    else: dummy_args = dummy_args_tuple
    if not dummy_args.keep: plt.close('all')

    if dummy_args.do_all_sims: list_of_sims = get_all_sims_for_this_halo(dummy_args) # all snapshots of this particular halo
    else: list_of_sims = list(itertools.product([dummy_args.halo], dummy_args.output_arr))
    total_snaps = len(list_of_sims)

    # -------set up dataframe and filename to store/write gradients in to--------
    cols_in_df = ['output', 'redshift', 'time']
    cols_to_add = ['mass', 'Zcen', 'Zcen_u', 'Zgrad', 'Zgrad_u', 'Zcen_binned', 'Zcen_u_binned', 'Zgrad_binned', 'Zgrad_u_binned', 'Ztotal']
    if dummy_args.upto_kpc is not None: cols_in_df = np.hstack([cols_in_df, ['re_stars', 're_coldgas'], [item + '_fixedr' for item in cols_to_add]])
    else: cols_in_df = np.hstack([cols_in_df, ['re_stars', 're_coldgas'], [item + '_re_stars' for item in cols_to_add], [item + '_re_coldgas' for item in cols_to_add]])

    df_grad = pd.DataFrame(columns=cols_in_df)
    weightby_text = '' if dummy_args.weight is None else '_wtby_' + dummy_args.weight
    if dummy_args.upto_kpc is not None:
        upto_text = '_upto%.1Fckpchinv' % dummy_args.upto_kpc if dummy_args.docomoving else '_upto%.1Fkpc' % dummy_args.upto_kpc
    else:
        upto_text = '_upto%.1FRe' % dummy_args.upto_re
    grad_filename = dummy_args.output_dir + 'txtfiles/' + dummy_args.halo + '_MZR_xcol_%s%s%s.txt' % (dummy_args.xcol, upto_text, weightby_text)
    if dummy_args.write_file and dummy_args.clobber and os.path.isfile(grad_filename): subprocess.call(['rm ' + grad_filename], shell=True)

    if dummy_args.dryrun:
        print('List of the total ' + str(total_snaps) + ' sims =', list_of_sims)
        sys.exit('Exiting dryrun..')
    # parse column names, in case log

    # --------------read in the cold gas profile file ONCE for a given halo-------------
    foggie_dir, output_dir, run_dir, code_path, trackname, haloname, spectra_dir, infofile = get_run_loc_etc(dummy_args)
    if dummy_args.write_file or dummy_args.upto_kpc is None:
        gasfilename = '/'.join(output_dir.split('/')[:-2]) + '/' + 'mass_profiles/' + dummy_args.run + '/all_rprof_' + dummy_args.halo + '.npy'
        print('Reading in cold gas profile from', gasfilename)
        gasprofile = np.load(gasfilename, allow_pickle=True)[()]
    else:
        print('Not reading in cold gas profile because any re calculation is not needed')
        gasprofile = None

    # --------domain decomposition; for mpi parallelisation-------------
    comm = MPI.COMM_WORLD
    ncores = comm.size
    rank = comm.rank
    print_master('Total number of MPI ranks = ' + str(ncores) + '. Starting at: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()), dummy_args)
    comm.Barrier() # wait till all cores reached here and then resume

    split_at_cpu = total_snaps - ncores * int(total_snaps/ncores)
    nper_cpu1 = int(total_snaps / ncores)
    nper_cpu2 = nper_cpu1 + 1
    if rank < split_at_cpu:
        core_start = rank * nper_cpu2
        core_end = (rank+1) * nper_cpu2 - 1
    else:
        core_start = split_at_cpu * nper_cpu2 + (rank - split_at_cpu) * nper_cpu1
        core_end = split_at_cpu * nper_cpu2 + (rank - split_at_cpu + 1) * nper_cpu1 - 1

    # --------------------------------------------------------------
    print_mpi('Operating on snapshots ' + str(core_start + 1) + ' to ' + str(core_end + 1) + ', i.e., ' + str(core_end - core_start + 1) + ' out of ' + str(total_snaps) + ' snapshots', dummy_args)

    for index in range(core_start + dummy_args.start_index, core_end + 1):
        start_time_this_snapshot = time.time()
        this_sim = list_of_sims[index]
        this_df_grad = pd.DataFrame(columns=cols_in_df)
        print_mpi('Doing snapshot ' + this_sim[1] + ' of halo ' + this_sim[0] + ' which is ' + str(index + 1 - core_start) + ' out of the total ' + str(core_end - core_start + 1) + ' snapshots...', dummy_args)
        halos_df_name = code_path + 'halo_infos/00' + this_sim[0] + '/' + dummy_args.run + '/' + 'halo_cen_smoothed'
        try:
            if len(list_of_sims) == 1 and not dummy_args.do_all_sims: args = dummy_args_tuple # since parse_args() has already been called and evaluated once, no need to repeat it
            else: args = parse_args(this_sim[0], this_sim[1])

            if type(args) is tuple:
                args, ds, refine_box = args  # if the sim has already been loaded in, in order to compute the box center (via utils.pull_halo_center()), then no need to do it again
                print_mpi('ds ' + str(ds) + ' for halo ' + str(this_sim[0]) + ' was already loaded at some point by utils; using that loaded ds henceforth', args)
            else:
                isdisk_required = np.array(['disk' in item for item in [args.xcol, args.ycol] + args.colorcol]).any()
                ds, refine_box = load_sim(args, region='refine_box', do_filter_particles=True, disk_relative=False, halo_c_v_name=halos_df_name)
        except Exception as e:
            print_mpi('Skipping ' + this_sim[1] + ' because ' + str(e), dummy_args)
            continue

        # parse paths and filenames
        args.fig_dir = args.output_dir + 'figs/' if args.do_all_sims else args.output_dir + 'figs/' + args.output + '/'
        Path(args.fig_dir).mkdir(parents=True, exist_ok=True)

        args.current_redshift = ds.current_redshift
        args.current_time = ds.current_time.in_units('Gyr').v
        args.ylim = [-2.2, 1.2] # [-3, 1]

        re_from_stars = get_re_from_stars(ds, args) if args.write_file or args.upto_kpc is None else None # kpc
        re_from_coldgas = get_re_from_coldgas(gasprofile, args)  if args.write_file or args.upto_kpc is None else None # kpc
        thisrow = [args.output, args.current_redshift, args.current_time, re_from_stars, re_from_coldgas] # row corresponding to this snapshot to append to df

        if args.upto_kpc is not None:
            method_arr = ['']
            upto_radius_arr = [args.upto_kpc]
        else:
            method_arr = ['stars', 'coldgas'] # it is important that stars and coldgas appear in this sequence
            upto_radius_arr = np.array([re_from_stars, re_from_coldgas])

        for this_upto_radius in upto_radius_arr:
            if this_upto_radius > 0:
                if args.upto_kpc is not None:
                    args.re = np.nan
                    if args.docomoving: args.galrad = this_upto_radius / (1 + args.current_redshift) / 0.695 # fit within a fixed comoving kpc h^-1, 0.695 is Hubble constant
                    else: args.galrad = this_upto_radius # fit within a fixed physical kpc
                else:
                    args.re = this_upto_radius
                    args.galrad = args.re * args.upto_re  # kpc

                # extract the required box
                box_center = ds.arr(args.halo_center, kpc)
                box_width = args.galrad * 2  # in kpc
                box_width_kpc = ds.arr(box_width, 'kpc')
                box = ds.r[box_center[0] - box_width_kpc / 2.: box_center[0] + box_width_kpc / 2., box_center[1] - box_width_kpc / 2.: box_center[1] + box_width_kpc / 2., box_center[2] - box_width_kpc / 2.: box_center[2] + box_width_kpc / 2., ]

                df = get_df_from_ds(box, args) # get dataframe with metallicity profile info

                Zcen, Zgrad = fit_gradient(df, args)
                args.bin_edges = np.linspace(0, args.galrad / args.re if 're' in args.xcol else args.galrad, 10)
                Zcen_binned, Zgrad_binned = fit_binned(df, args.xcol, 'metal', args.bin_edges, ax=None, is_logscale=True, weightcol=args.weight)

                if not args.noplot: fig = plot_gradient(df, args, linefit=[Zgrad.n, Zcen.n]) # plotting the Z profile, with fit

                mstar = get_disk_stellar_mass(args) # Msun

                df['metal_mass'] = df['mass'] * df['metal'] * metallicity_sun
                Ztotal = (df['metal_mass'].sum()/df['mass'].sum())/metallicity_sun # in Zsun

                thisrow += [mstar, Zcen.n, Zcen.s, Zgrad.n, Zgrad.s, Zcen_binned.n, Zcen_binned.s, Zgrad_binned.n, Zgrad_binned.s, Ztotal]
            else:
                thisrow += (np.ones(11)*np.nan).tolist()

        this_df_grad.loc[len(this_df_grad)] = thisrow
        df_grad = pd.concat([df_grad, this_df_grad])
        if args.write_file:
            if not os.path.isfile(grad_filename):
                this_df_grad.to_csv(grad_filename, sep='\t', index=None, header='column_names')
                print('Wrote to gradient file', grad_filename)
            else:
                this_df_grad.to_csv(grad_filename, sep='\t', index=None, mode='a', header=False)
                print('Appended to gradient file', grad_filename)

        print_mpi('This snapshots completed in %s mins' % ((time.time() - start_time_this_snapshot) / 60), dummy_args)


    if ncores > 1: print_master('Parallely: time taken for ' + str(total_snaps) + ' snapshots with ' + str(ncores) + ' cores was %s mins' % ((time.time() - start_time) / 60), dummy_args)
    else: print_master('Serially: time taken for ' + str(total_snaps) + ' snapshots with ' + str(ncores) + ' core was %s mins' % ((time.time() - start_time) / 60), dummy_args)
