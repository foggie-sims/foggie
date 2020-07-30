import glob
from glob import glob
import numpy as np
from astropy.io import ascii
from astropy.table import Table 

keys = ['stars_Z', 'stars_M', 'stars_L_x', 'stars_L_y', 'stars_L_z', 'young_stars_Z', 'SFR', 'young_stars_M', \
        'young_stars_L_x', 'young_stars_L_y', 'young_stars_L_z', 'all_gas_Z', 'all_gas_M', 'all_gas_L_x',\
        'all_gas_L_y', 'all_gas_L_z', 'cold_gas_Z', 'cold_gas_M', 'cold_gas_L_x', 'cold_gas_L_y', 'cold_gas_L_z',\
        'hot_gas_Z', 'hot_gas_M', 'hot_gas_L_x', 'hot_gas_L_y', 'hot_gas_L_z']


for halo in ['5036']:#'8508', '5016', '2392', '5036']:
  halo_c_v = ascii.read('/nobackupp2/rcsimons/git/foggie/foggie/halo_infos/00%s/nref11c_nref9f/halo_c_v'%(halo))
  t_dic = {}
  t_dic['halo']       = []
  t_dic['redshift']   = []
  t_dic['sim_output'] = []
  for key in keys: t_dic[key] = []

  fls = glob('/nobackupp2/rcsimons/foggie_momentum/mass_metallicity_momentum/nref11c_nref9f_%s_DD*_mass.npy'%halo)
  fls = np.sort(fls)

  for fl in fls:
    DD = fl.split('_')[-2]
    try: redshift = float(halo_c_v['col2'][halo_c_v['col3'] == DD])
    except: continue
    t_dic['halo'].append(halo)
    t_dic['redshift'].append(redshift)
    t_dic['sim_output'].append(DD)
    #print (halo, redshift, DD)

    output = np.load(fl, allow_pickle = True)[()]
    for key in keys: 
      if key == 'SFR':
         t_dic[key].append(float(output['young_stars_M'].value)/(1.e7))
      else:
        t_dic[key].append(float(output[key].value))


    t = Table(t_dic)


    t.meta['comments'] = ['halo: halo name']
    t.meta['comments'].append('all measurements are within a sphere 20 kpc around central galaxy')
    t.meta['comments'].append('metallicities (Z) in units of Zsun')
    t.meta['comments'].append('masses (M) in in units of Msun')
    t.meta['comments'].append('angular momentum (L) in units of cm**2*kg/s')
    t.meta['comments'].append('star-formation rate (SFR) in units of Msun/yr')
    t.meta['comments'].append('stars = all stars')
    t.meta['comments'].append('young stars = <10 Myr')
    t.meta['comments'].append('all_gas = Temp (K)  < 1.e6 K')
    t.meta['comments'].append('hot gas  = 1.5e4 < Temp (K) < 1.e6')
    t.meta['comments'].append('cold_gas = Temp (K)  < 1.5e4 K')

    ascii.write(t,'/nobackupp2/rcsimons/foggie_momentum/mass_metallicity_momentum/catalogs/FOGGIE_nref11c_%s.cat'%halo, format = 'commented_header', overwrite = True)








































