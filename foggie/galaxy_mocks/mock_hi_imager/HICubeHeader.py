import unyt as u
import numpy as np

projection_dict = {'x': ('y', 'z', 'x'), 'y':('z', 'x', 'y'), 'z':('x', 'y', 'z')} # which axes are projected for which line of sight args.projection
c = 3e5 * u.km / u.s  # km/s
H0 = 70. * u.km / u.s / u.Mpc  # km/s/Mpc Hubble's constant
h = 4.135667696e-15 * u.eV * u.s #eV * s
Mpc_to_m = 3.08e22
Mpc_to_cm = Mpc_to_m * 100
kpc_to_cm = Mpc_to_cm / 1000
kb = 8.617333262e-5 * u.eV / u.K
m_e = 9.1094*np.power(10.,-28.) * u.g #grams
e = 4.8032*np.power(10.,-10.) * u.statC #cm^(3/2) * g^(1/2) * s^(-1)
amu = 1.6735575*np.power(10.,-24) * u.g

arcsec_to_rad = 1./60./60. * np.pi/180.
rad_to_arcsec = 180. * 3600. / np.pi
