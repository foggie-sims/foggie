import yt
import numpy as np

from astropy.table import Table
import astropy.units as u

from utils.get_proper_box_size import get_proper_box_size
from utils.get_refine_box import get_refine_box

print('tap tap is this thing on')

### WE ARE ON PLEIADES AND THIS IS THE RD0018 OUTPUT OF THE NREF11F BOX
dsr = yt.load('RD0018/RD0018')
track_name = 'halo_track'

track = Table.read(track_name, format='ascii')
track.sort('col1')
proper_box_size = get_proper_box_size(dsr)
width = 15. #kpc
zsnap = dsr.current_redshift
refine_box, refine_box_center, refine_width = get_refine_box(dsr, zsnap, track)

mH = 1.6737236e-24 * u.g
total_forced_HI_mass = sum(mH * refine_box['H_p0_number_density'] * refine_box['cell_volume'].in_units('cm**3'))
print('HI mass: ', total_forced_HI_mass.to('Msun'), )
print('log HI mass: ', np.log10(total_forced_HI_mass.to('Msun').value))

import trident
trident.add_ion_fields(dsr, ions=['C IV', 'O VI', 'Si II',  'Si IV'])

total_forced_SiII_mass = sum(mH * refine_box['Si_p1_number_density'] * refine_box['cell_volume'].in_units('cm**3'))
total_forced_SiII_mass = total_forced_SiII_mass * 28.0855
print('SiII forced: ', total_forced_SiII_mass.to('Msun'),)

total_forced_OVI_mass = sum(mH * refine_box['O_p5_number_density'] * refine_box['cell_volume'].in_units('cm**3'))
total_forced_OVI_mass = total_forced_OVI_mass * 15.999
print('OVI forced: ', total_forced_OVI_mass.to('Msun'))


total_forced_SiIV_mass = sum(mH * refine_box['Si_p3_number_density'] * refine_box['cell_volume'].in_units('cm**3'))
total_forced_SiIV_mass = total_forced_SiIV_mass * 28.0855
print('SiIV forced: ', total_forced_SiIV_mass.to('Msun'))

total_forced_CIV_mass = sum(mH * refine_box['C_p3_number_density'] * refine_box['cell_volume'].in_units('cm**3'))
total_forced_CIV_mass = total_forced_CIV_mass * 12.0107
print('CIV forced: ', total_forced_CIV_mass.to('Msun'))
