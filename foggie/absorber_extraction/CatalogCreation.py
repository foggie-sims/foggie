# ### This script implements Brendan's code for creating absorber catalogs
# This has been boiled down as much as possible for simplicity in scripting. 
# The notebook is in "absorber_catalogs".
# for now, just run this in the command line in the dir where you
# have the snapshots and want the outputs, etc. 

import numpy as np 
from foggie.clouds.absorber_catalogs import read_absorber_catalog, plot_ions, plot_absorbers 
from foggie.absorber_extraction import salsa
from foggie.absorber_extraction.salsa.utils.utility_functions import parse_cut_filter
from foggie.utils.consistency import default_spice_fields, min_absorber_dict
from foggie.utils.foggie_load import foggie_load

box_trackfile = '/Users/tumlinson/Dropbox/FOGGIE/foggie/foggie/halo_tracks/008508/nref11n_selfshield_15/halo_track_200kpc_nref10'
hcv_file='/Users/tumlinson/Dropbox/FOGGIE/foggie/foggie/halo_infos/008508/nref11c_nref9f/halo_c_v'

dataset_name = 'RD0020'
rvir = 52.3 
ds, reg_foggie = foggie_load('../'+dataset_name+'/'+dataset_name, box_trackfile, halo_c_v_name=hcv_file)

cgm_cut = [ "(obj[('gas', 'temperature')].in_units('K') > 1.e1)", #<---- temperature 
            "(obj[('gas', 'radial_velocity_corrected')] <= 1e12)",  #<---- radial velocity 
            "(obj[('gas', 'radius_corrected')].in_units('kpc') > 10.0)", #<----- Radius 
            "(obj[('gas', 'radius_corrected')].in_units('kpc') < 200.0)", 
            "(obj['temperature'] > 15000.0 ) | (obj['density'] < 2e-26)"] #<----- cuts out ISM

cgm_outflow_cut = [ "(obj[('gas', 'temperature')].in_units('K') > 1.e1)", #<---- temperature 
            "(obj[('gas', 'radial_velocity_corrected')] > 50e5)",  #<---- radial velocity 
            "(obj[('gas', 'radius_corrected')].in_units('kpc') > 10.0)", #<----- Radius 
            "(obj[('gas', 'radius_corrected')].in_units('kpc') < 200.0)", 
            "(obj['temperature'] > 15000.0 ) | (obj['density'] < 2e-26)"] #<----- cuts out ISM

cgm_inflow_cut = [ "(obj[('gas', 'temperature')].in_units('K') > 1.e1)", #<---- temperature 
            "(obj[('gas', 'radial_velocity_corrected')] < -50e5)",  #<---- radial velocity 
            "(obj[('gas', 'radius_corrected')].in_units('kpc') > 10.0)", #<----- Radius 
            "(obj[('gas', 'radius_corrected')].in_units('kpc') < 200.0)", 
            "(obj['temperature'] > 15000.0 ) | (obj['density'] < 2e-26)"] #<----- cuts out ISM

def get_absorber_table(cut, directory): 
    
    df = salsa.generate_catalog(ds, 1000, directory, 
                                ['H I', 'O VI'],
                                center=ds.halo_center_code, #<----halo center from dataset
                                impact_param_lims=(0, rvir*2.), #<----impact parameter limits 
                                cut_region_filters=cut,
                                ray_length=200, fields=default_spice_fields,
                                extractor_kwargs={'absorber_min':11.0})

    return df

df_cgm = get_absorber_table(cgm_cut, dataset_name+'.absorbers/cgm')
df_cgm.to_csv(dataset_name+'.absorbers/cgm.txt')

df_outflow = get_absorber_table(cgm_outflow_cut, dataset_name+'.absorbers/cgm_outflow')
df_outflow.to_csv(dataset_name+'.absorbers/cgm_outflow.txt')

df_inflow = get_absorber_table(cgm_inflow_cut, dataset_name+'.absorbers/cgm_inflow')
df_inflow.to_csv(dataset_name+'.absorbers/cgm_inflow.txt')
