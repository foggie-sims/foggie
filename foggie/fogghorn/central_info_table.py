'''
    Filename: central_info_table.py
    Author: Cassi
    Created: 6-12-24
    Last modified: 3-31-25 by Cassi
    This file works with fogghorn_analysis.py to make a table of various halo info properties that can be used for plotting.
'''

from foggie.fogghorn.header import *
from foggie.fogghorn.util import *
from astropy.cosmology import Planck15 as cosmo

# --------------------------------------------------------------------------------------------------------------------
def make_table():
    '''
    Initialises table to hold properties of each halo at each snapshot
    '''
    data_names = ['snapshot','time','redshift','halo_x','halo_y','halo_z','halo_vx','halo_vy','halo_vz','halo_mass','halo_radius','stellar_mass','SFR']
    data_types = ['S6','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8']
    data_units = ['none','Myr','none','kpc','kpc','kpc','km/s','km/s','km/s','Msun','kpc','Msun','Msun/yr']
    data = Table(names=data_names, dtype=data_types)
    for i in range(len(data.keys())):
        key = data.keys()[i]
        data[key].unit = data_units[i]
    return data

# --------------------------------------------------------------------------------------------------------------------
def get_halo_info(ds, snap, args):
    '''
    Calculates basic information about the halo: snapshot name, time, redshift, halo x,y,z location, halo vx,vy,vz bulk velocity, virial mass, virial radius, stellar mass, star formation rate.
    NOTE: The virial mass and radius as currently written will only work for the central galaxies! Rockstar is not being run to find satellite halos.
    '''

    row = [snap, ds.current_time.in_units('Myr').v, ds.get_parameter('CosmologyCurrentRedshift'), \
            ds.halo_center_kpc[0], ds.halo_center_kpc[1], ds.halo_center_kpc[2], \
            ds.halo_velocity_kms[0], ds.halo_velocity_kms[1], ds.halo_velocity_kms[2]]
    
    sph = ds.sphere(center = ds.halo_center_kpc, radius = (400., 'kpc'))
    filter_particles(sph)
    prof_dm = yt.create_profile(sph, ('index', 'radius'), fields = [('deposit', 'dm_mass')], \
                                n_bins = 500, weight_field = None, accumulation = True)
    prof_stars = yt.create_profile(sph, ('index', 'radius'), fields = [('deposit', 'stars_mass')], \
                                n_bins = 500, weight_field = None, accumulation = True)
    prof_young_stars = yt.create_profile(sph, ('index', 'radius'), fields = [('deposit', 'young_stars_mass')], \
                                n_bins = 500, weight_field = None, accumulation = True)
    prof_gas = yt.create_profile(sph, ('index', 'radius'), fields = [('gas', 'cell_mass')],\
                                n_bins = 500, weight_field = None, accumulation = True)

    internal_density =  (prof_dm[('deposit', 'dm_mass')].to('g') + prof_stars[('deposit', 'stars_mass')].to('g') + \
                            prof_gas[('gas', 'cell_mass')].to('g'))/(4*np.pi*prof_dm.x.to('cm')**3./3.)

    rho_crit = cosmo.critical_density(ds.current_redshift)
    rvir = prof_dm.x[np.argmin(abs(internal_density.value - 200*rho_crit.value))]
    Mdm_rvir    = prof_dm[('deposit', 'dm_mass')][np.argmin(abs(internal_density.value - 200*rho_crit.value))]
    Mstars_rvir = prof_stars[('deposit', 'stars_mass')][np.argmin(abs(internal_density.value - 200*rho_crit.value))]
    Mgas_rvir   = prof_gas[('gas', 'cell_mass')][np.argmin(abs(internal_density.value - 200*rho_crit.value))]
    Mvir = Mdm_rvir + Mstars_rvir + Mgas_rvir
    Myoung_stars = prof_young_stars[('deposit','young_stars_mass')][np.where(prof_young_stars.x.to('kpc') >= 20.)[0][0]]
    SFR = Myoung_stars.to('Msun').v/1e7

    row.append(Mvir.to('Msun').v)
    row.append(rvir.to('kpc').v)
    row.append(Mstars_rvir.to('Msun').v)
    row.append(SFR)

    return row