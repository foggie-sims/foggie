import numpy as np
import unyt as u

from foggie.galaxy_mocks.mock_hi_imager.HICubeHeader import * #Constants

def load_line_properties(line):
    '''
    Line Properties used to calculate emission profile. Only defined for HI 21cm currently.
    '''
    if line=="HI_21cm":
        #Hyperfine splitting
        species_mass = 1.6735575*np.power(10.,-24) * u.g #grams to unit mass
        g_upper = 3
        g_lower = 1
        E_upper = (-13.6 + 5.87433 * np.power(10.,-6.)) * u.eV #eV
        E_lower = -13.6 * u.eV
        A_10 = 2.85 * np.power(10.,-15.) / u.s #Hz
        gamma_ul = A_10 #Need to consider other transitions??
        A_ul = A_10
        Elevels = [E_lower , E_upper]
        Glevels = [g_lower , g_upper] #Safe to simplify as 2 state system? Gas emitting this should be cold anyway. May need to expand on this if hot gas emitting too much?
        n_u_fraction = 0.75
        n_l_fraction = 0.25

        return species_mass,gamma_ul, A_ul, Elevels, Glevels, n_u_fraction, n_l_fraction

def _Emission_HI_21cm(field,data):
        '''
        Psuedo function to add an HI 21 cm emission field to the dataset.
        In practice, we would normally convert the cell properties to an emission power, and then convert that emission power back to a column density.
        I'm currently just skipping that and keeping it as a column density of each cell. In the optically thin limit this should be the same.
        '''

        return data['gas','H_p0_number_density'] * data['gas','dx']

        '''
        species_mass,gamma_ul, A_ul, Elevels, Glevels, n_u_fraction, n_l_fraction = load_line_properties("HI_21cm")
        Nlevels = len(Elevels)
        partitionFunction = 0


        num = -(Elevels[1]-Elevels[0]) / u.eV
        denom = kb/(u.eV/u.K) * data['gas','temperature'].in_units('K')
        for l in range(0,Nlevels):
            partitionFunction += Glevels[l] * np.exp( np.divide(num,denom) ) #Unitless
        
        n_total = np.multiply( data['gas','H_p0_number_density'] , np.power(data['gas','dx'],3.) )
        n_u = np.multiply(n_total  , np.divide( np.exp( np.divide(num,denom)) * Glevels[1] , partitionFunction)) #unitless
        upperToLowerRate = n_u*A_ul

        return upperToLowerRate * h 
        '''
