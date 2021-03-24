##!/usr/bin/env python3

"""

    Title :      compute_hiir_radii
    Notes :      To compute the instantaneous radii of HII regions around young star particles using subgrid dynamical HII modeling; based on Verdolini+2013
    Output :     One pandas dataframe as a txt file
    Author :     Ayan Acharyya
    Started :    January 2021
    Example :    run <scriptname>.py --system ayan_local --halo 8508 --output RD0042

"""
from header import *
from lookup_flux import *

# ----------------------------------------------------------------------------------------------------
def merge_HIIregions(df, args):
    '''
    Function for merging HII regions within args.mergeHII kpc distance; weighted by weightcol
    Columns of input df are: 'pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z', 'age', 'mass', 'gas_pressure', 'gas_metal', 'Q_H0'
    '''

    myprint('Merging HII regions within '+ str(args.mergeHII * 1e3) + ' pc. May take 10-20 seconds...', args)
    groupbycol = 'cell_index'
    weightcol = 'Q_H0'
    initial_nh2r = len(df)

    g = int(np.ceil(args.galrad * 2 / args.mergeHII))
    gz = int(np.ceil(args.galthick / args.mergeHII))

    xind = ((df['pos_x'] - args.halo_center[0] + args.galrad) / args.mergeHII).astype(np.int) # (df['x(kpc)'][j] - args.halo_center[0]) used to range from (-galrad, galrad) kpc, which is changed here to (0, galrad*2) kpc
    yind = ((df['pos_y'] - args.halo_center[1] + args.galrad) / args.mergeHII).astype(np.int)
    zind = ((df['pos_z'] - args.halo_center[2] + args.galthick / 2) / args.mergeHII).astype(np.int)
    df[groupbycol] = xind + yind * g + zind * g * gz

    if 'Sl.' in df.columns: df.drop(['Sl.'], axis=1, inplace=True)
    weighted_mean = lambda x: np.average(x, weights=df.loc[x.index, weightcol]) # function to weight by weightcol
    cols_to_sum = [weightcol] + ['mass'] # which columns should be summed after merging?
    cols_to_wtmean = df.columns[~df.columns.isin([groupbycol] + cols_to_sum)] # which columns should be weighted mean after merging

    sum_operations = {item:sum for item in cols_to_sum}
    weighted_mean_operations = {item:weighted_mean for item in cols_to_wtmean}
    all_operations = {**sum_operations, **weighted_mean_operations, **{groupbycol:'count'}}

    df = df.groupby(groupbycol, as_index=False).agg(all_operations)
    df.rename(columns={groupbycol:'count'}, inplace=True)

    myprint('Merged ' + str(initial_nh2r) + ' HII regions into ' + str(len(df)) + ' HII regions\n', args)
    return df

# ----------------------------------------------------------------------------------------------
def compute_radii(paramlist):
    '''
    Function to compute final radius of HII regions, and append a few new columns to the dataframe
    '''

    # --------calculating characteristic radius-----------#
    r_ch = (alpha_B * eps ** 2 * f_trap ** 2 * psi ** 2 * paramlist['Q_H0']) / (12 * np.pi * phi * k_B ** 2 * TII ** 2 * c ** 2)

    # --------calculating characteristic time-----------#
    m_SI = paramlist['mass'] * 1.989e30  # converting Msun to kg
    age_SI = paramlist['age'] * 3.1536e13  # converting Myr to sec
    r_0 = (m_SI ** 2 * (3 - k_rho) ** 2 * G / (paramlist['gas_pressure'] * 8 * np.pi)) ** (1 / 4.)
    rho_0 = (2 / np.pi) ** (1 / 4.) * (paramlist['gas_pressure'] / G) ** (3 / 4.) / np.sqrt(m_SI * (3 - k_rho))
    t_ch = np.sqrt(4 * np.pi * rho_0 * r_0 ** k_rho * c * r_ch ** (4 - k_rho) / ((3 - k_rho) * f_trap * psi * eps * paramlist['Q_H0']))
    tau = age_SI / t_ch

    # --------calculating radiation pressure radius-----------#
    xII_rad = ((tau ** 2) * (4 - k_rho) / 2) ** (1 / (4 - k_rho))

    # --------calculating gas pressure radius-----------#
    xII_gas = ((7 - 2 * k_rho) ** 2 * tau ** 2 / (4 * (9 - 2 * k_rho))) ** (2 / (7 - 2 * k_rho))

    # --------calculating approximate instantaneous radius-----------#
    xII_apr = (xII_rad ** ((7 - k_rho) / 2) + xII_gas ** ((7 - k_rho) / 2)) ** (2 / (7 - k_rho))
    paramlist['r_inst'] = xII_apr * r_ch

    # --------calculating stall radius-----------#
    Prad_const = psi * eps * f_trap * paramlist['Q_H0'] / (4 * np.pi * c)
    Pgas_const = np.sqrt(3 * phi * paramlist['Q_H0'] / (4 * np.pi * alpha_B * (1 + Y / (4 * X)))) * (mu_H * m_H * cII ** 2)
    r0 = (Pgas_const /paramlist['gas_pressure']) ** (2/3.)

    for index in range(len(paramlist)):
        paramlist.loc[index, 'r_stall'] = 10 ** (op.newton(Flog, math.log10(r0[index]), args=(Prad_const[index], Pgas_const[index], paramlist.loc[index, 'gas_pressure']), maxiter=100))

    # --------determining minimum of the two-----------#
    paramlist['r'] = paramlist[['r_inst', 'r_stall']].min(axis=1)

    # --------calculating density inside HII region (assumed constant)-----------#
    paramlist['nII'] = np.sqrt(3 * phi * paramlist['Q_H0'] / (4 * (paramlist['r'] ** 3) * np.pi * alpha_B * (1 + Y / (4 * X))))

    # --------calculating volume averaged ionisation parameter inside HII region-----------#
    paramlist['<U>'] = paramlist['Q_H0'] / (4 * np.pi * paramlist['r'] ** 2 * c * paramlist['nII'])

    # --------calculating HII region pressure-----------#
    paramlist['log(P/k)'] = np.log10(paramlist['nII'] * TII)

    # --------converting units-----------#
    paramlist['r'] /= 3.06e16 # to convert distance to pc units
    paramlist['r_inst'] /= 3.06e16 # to convert distance to pc units
    paramlist['r_stall'] /= 3.06e16 # to convert distance to pc units
    return paramlist

# ---------------------------------------------------------------------------------------------------
def Flog(x, a, b, c):
    '''
    Function to solve the equation:
    P^2r^4 -2aPr^2 -br + a^2 = 0
    '''

    # return (c**2)*(10**(4*x)) - 2*a*c*(10**(2*x)) - b*10**x + a**2
    return a / (c * 10 ** (2 * x)) + b / (c * 10 ** (x * 1.5)) - 1

# -------------------------------------------------------------------------------------------------
def get_radii_for_df(paramlist, args):
    '''
    Function to handle input/outut dataframe of list of parameters
    '''

    start_time = time.time()
    # -----------------------------------------------------------------------------------
    outfilename = args.output_dir + 'txtfiles/' + args.output + '_radius_list' + args.mergeHII_text + '.txt'

    # ----------------------Creating new radius list file if one doesn't exist-------------------------------------------
    if not os.path.exists(outfilename) or args.clobber:
        if not os.path.exists(outfilename): myprint(outfilename + ' does not exist. Creating afresh..', args)
        elif args.clobber: myprint(outfilename + ' exists but over-writing..', args)

        # ----------------------Reading starburst99 file-------------------------------------------
        SB_data = pd.read_table(sb99_dir + sb99_model + '/' + sb99_model + '.quanta', delim_whitespace=True,comment='#', skiprows=6, \
                                header=None, names=('age', 'HI/sec', 'HI%ofL', 'HeI/sec', 'HeI%ofL', 'HeII/sec', 'HeII%ofL', 'logQ'))
        SB_data.loc[0, 'age'] = 1e-6 # Myr # force first i.e. minimum age to be 1 yr instead of 0 yr to avoid math error
        interp_func = interp1d(np.log10(SB_data['age']/1e6), SB_data['HI/sec'], kind='cubic')

        nh2r_initial = len(paramlist)
        paramlist = paramlist[['pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z', 'age', 'mass', 'gas_pressure', 'gas_metal']] # only need these columns henceforth
        paramlist['Q_H0'] = (paramlist['mass']/sb99_mass) * 10 ** (interp_func(np.log10(paramlist['age']))) # scaling by starburst99 model mass
        paramlist['gas_pressure'] /= 10 # to convert from dyne/cm^2 to N/m^2
        if args.mergeHII is not None: paramlist = merge_HIIregions(paramlist, args)

        # ------------------solving--------------------------------------------------------------
        paramlist = compute_radii(paramlist)
        myprint('Using ' + str(len(paramlist)) + ' HII regions of ' + str(nh2r_initial), args)

        # ------------------writing dataframe to file--------------------------------------------------------------
        header = 'Units for the following columns: \n\
        pos_x, pos_y, pos_z: kpc \n\
        vel_x, vel_y, vel_z: km/s \n\
        age: Myr \n\
        mass: Msun \n\
        gas_pressure in a cell: g/cm^3 \n\
        Q_H0: ionisation photon flux from star particle, photons/s \n\
        r_stall: stalled HII region radius, pc \n\
        r_inst: instantaneous HII region radius, pc \n\
        r: assigned HII region radius, pc \n\
        <U>: volumne averaged ionsation parameter, dimensionless \n\
        nII: HII region number density per m^3\n\
        log(P/k): HII region pressure, SI units\n\
        Z: metallicity of ambient gas, Zsun units\n'

        np.savetxt(outfilename, [], header=header, comments='#')
        paramlist.to_csv(outfilename, sep='\t', mode='a', index=None)
        myprint('Radii list saved at ' + outfilename, args)
    else:
        myprint('Reading from existing file ' + outfilename, args)
        paramlist = pd.read_table(outfilename, delim_whitespace=True, comment='#')

    myprint(args.output + ' done in %s minutes' % ((time.time() - start_time) / 60), args)
    if args.automate:
        myprint('Will execute lookup_grid() for ' + args.output + '...', args)
        paramlist = lookup_grid(paramlist, args)
    return paramlist

# -------------------defining constants for assumed models---------------------------------------------------
k_rho = 0. #1. corrected to zero, since we assume constant density profile
mu_H = 1.33
m_H = 1.67e-27  # kg
cII = 9.74e3  # m/s
phi = 0.73
Y = 0.23
X = 0.75
psi = 3.2
eps = 2.176e-18  # Joules or 13.6 eV
c = 3e8  # m/s
f_trap = 2  # Verdolini et. al.
#alpha_B = 3.46e-19  # m^3/s OR 3.46e-13 cc/s, Krumholz Matzner (2009) for 7e3 K
alpha_B = 2.59e-19  # m^3/s OR 2.59e-13 cc/s, for Te = 1e4 K, referee quoted this values
k_B = 1.38e-23  # m^2kg/s^2/K
TII = 1e4 # K, because MAPPINGS models are computed at this temperature
G = 6.67e-11  # Nm^2/kg^2

# -------------------variables pertaining to simulation ouputs---------------------------------------------------
ng = 350 # number of grid points to break the simulation in to, in each of x and y dimensions
res = 0.02  # kpc; base resolution of simulation inside refined box
size = 287.76978417  # kpc; size of refined simulation box

# -------------------------------------------------------------------------------
if __name__ == '__main__':
    args = parse_args('8508', 'RD0042')
    infilename = args.output_dir + 'txtfiles/' + args.output + '_young_star_properties.txt'

    # ----------------------Reading star particles-------------------------------------------
    paramlist = pd.read_table(infilename, delim_whitespace=True, comment='#')
    paramlist = get_radii_for_df(paramlist, args)

    # --------------------------------------------------------------------------------
    nstalled = sum(paramlist['r'] == paramlist['r_stall'])
    myprint(str(nstalled) + ' HII regions have stalled expansion which is ' + str(nstalled * 100. / len(paramlist)) + ' % of the total.', args)
