'''
Filename: make_rates_table.py
Author: Cassi
Date created: 9-9-25

This script takes the ionization rate tables output from a CLOUDY run and
combines them into a numpy array that gives the ionization and recombination rates
for O VI as a function of density and temperature.'''

import numpy as np


def combine_rates(densities, temperatures, ionization_state, code_path):
    '''Combines CLOUDY run outputs for ionization rates into numpy arrays and
    returns the arrays.

    Inputs:
    densities        --  list of H density values used by CLOUDY. This corresponds 
                        to different _runX file names, if run with CIAOLoop.
    temperatures     -- list of temperature values used by CLOUDY. These should
                        be the same across all _runX files.
    ionization_state -- the number of the ionization state you want rates
                        for. e.g. if you want O VI, ionization_state = 6.
    code_path        --  path to your foggie repo
                        
    Outputs:
    ion_rates -- an array of shape (len(densities), len(temperatures))
                 that gives the ionization rates per ion from ionization_state up 
                 to the next level in 1/s.
    rec_rates -- an array of shape (len(densities), len(temperatures))
                 that gives the recombination rates per ion from ionization_state 
                 down to the level one below in 1/s.

    To get an estimate of the timescale for this particular ionization state
    to reach equilibrium, use:
        t_eq = 1 / (ion_rate + rec_rate)
    where ion_rate is the ionization rate upwards and rec_rate is the recombination
    rate downwards. This function returns the ion_rates and rec_rates that 
    can be used in this way elsewhere -- it does not actually calculate this 
    timescale here. (Gnat & Sternberg 2007)
    '''

    # CLOUDY rates tables are set up with 3 leading columns then 4 columns for each
    # ionization state of the element
    # 3 leading columns are depth, electron density, and flow rate (unused in our CLOUDY models)
    # Each ion block columns are abundance, ionization rate, recombination rate, and flow rate (unused)
    ion_block = 4*(ionization_state - 1) + 3       # Start of columns for ionization state of interest
    ion_column = ion_block + 1
    rec_column = ion_block + 2

    ion_rates = np.zeros(shape=(len(densities), len(temperatures)))
    rec_rates = np.zeros(shape=(len(densities), len(temperatures)))
    for d in range(len(densities)):
        T_idx = 0
        filename = code_path + '/cgm_emission/cloudy_extended_z0_selfshield/rates/TEST_z0_HM12_sh_run' + str(d+1) + '.rates'
        f = open(filename, 'r')
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('## User output punch for T ='):     # Actual data for this T is 2 rows down
                data_row = np.array(lines[i+2].split(), dtype=float)    # Split row on whitespace
                ion_rates[d,T_idx] = data_row[ion_column]
                rec_rates[d,T_idx] = data_row[rec_column]
                T_idx += 1

    return ion_rates, rec_rates
