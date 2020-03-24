import astropy
from astropy.table import Table
from astropy.io import ascii, fits
import glob
from glob import glob
import numpy as np


all_outputs = {}



fields = ['DD', 'z', 'stars_youngmass', 'dark_matter','gas_tot','gas_metals','stars_mass',\
          'gas_H','gas_H0','gas_H1','gas_CII','gas_CIII','gas_CIV','gas_OVI',\
          'gas_OVII','gas_MgII','gas_SII','gas_SIII','gas_SIV','gas_NeVIII']


field_new_names = []


for field_name in fields:
    if field_name == 'dark_matter':
        field_new_name = 'mass_dm'
    elif field_name == 'DD':
        field_new_name = 'DD'
    elif field_name == 'z':
        field_new_name = 'z'
    elif field_name == 'gas_tot':
        field_new_name = 'mass_gas'
    elif field_name == 'gas_metals':
        field_new_name = 'mass_gas_metals'
    elif field_name == 'stars_mass':
        field_new_name = 'mass_stars'
    elif field_name == 'stars_mass':
        field_new_name = 'mass_stars'
    elif field_name == 'stars_youngmass':
        field_new_name = 'sfr'
    else:
        field_new_name = field_name.replace('gas', 'mass')

    field_new_names.append(field_new_name)




    all_outputs[field_new_name] = []






#where the mass.fits files are located
fits_outdir = '.'
fls = glob(fits_outdir + '/*mass.fits')
fls = np.sort(fls)

DD_time_npy_file = '/Users/rsimons/Dropbox/rcs_foggie/outputs/DD_time_new.npy'
DD_time = np.load(DD_time_npy_file, allow_pickle = True)[()]

for fl in fls:
    DD = fl.split('_')[-2].strip('DD')
    gd = np.where(np.array(DD_time['DD']) == DD)[0][0]
    z = DD_time['z'][gd]
    all_outputs['DD'].append(DD)
    all_outputs['z'].append(z)

    print (DD)
    fts = fits.open(fl)
    fts_tbl = fts[1].data
    for field_new_name, field in zip(field_new_names[2:], fields[2:]):
        if field_new_name == 'sfr':
            all_outputs[field_new_name].append('%.4f'%(fts_tbl[field][-1]/(2.e7)))
        else:
            all_outputs[field_new_name].append('%.4f'%(fts_tbl[field][-1]/(1.e6)))




t = Table(all_outputs, names = field_new_names)





t.meta['comments'] = ['as measured in a 20 kpc sphere around the central galaxy']
t.meta['comments'].append('star-formation rate in Msun/yr')
t.meta['comments'].append('masses in 10^7 Msun')

cat_outfile = '/Users/rsimons/Desktop/foggie/8508_nref11n_nref10f_central.cat'
ascii.write(t, cat_outfile, \
            format = 'commented_header', overwrite = True)
















