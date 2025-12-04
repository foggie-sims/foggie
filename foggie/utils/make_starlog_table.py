import glob, os, numpy as np 
from astropy.table import QTable, Table, hstack, vstack
import matplotlib.pyplot as plt
import astropy.units as u 
from datashader.mpl_ext import dsshow, alpha_colormap
import datashader as ds

starlogfiles = glob.glob('starlog*0.txt') 
print(starlogfiles) 

def prep_file(file): 
    os.system('grep SMInput ' + file + ' > ' + file[0:17] + 'input') 
    os.system('grep SMMath ' + file + ' > ' + file[0:17] + 'math')
    os.system('grep SMOut ' + file + ' > ' + file[0:17] + 'out')

    input = QTable.read(file[0:17] + 'input', format='ascii') 
    input['col1'].name = 'file'
    input['col2'].name = 'd'
    input['col3'].name = 'fHI'
    input['col4'].name = 'Temp'
    input['col5'].name = 't_code'
    input['col6'].name = 'dt_code'
    input['col7'].name = 'redshift'
    input['col8'].name = 'dx_code'
    input['col9'].name = 'Metallicity'
    input['col10'].name = 'H2Method'

    math = QTable.read(file[0:17] + 'math', format='ascii') 

    math['col1'].name = 'file'
    math['col2'].name = 'nH' 
    math['col3'].name = 'Sigma'
    math['col4'].name = 'phi_CNM'
    math['col5'].name = 'chi'
    math['col6'].name = 'tau_c'
    math['col7'].name = 's'
    math['col8'].name = 'H2_fraction'
    math['col9'].name = 'densthresh'
    math['col10'].name = 'timeconstant'
    math['col11'].name = 'tau_ff'
    math['col12'].name = 'gasfrac'
    math['col13'].name = 'starmass'
    math['col14'].name = 'starmass_in_Msun'
    math['col15'].name = 'maxflag'

    out = QTable.read(file[0:17] + 'out', format='ascii') 
    out['col1'].name = 'file'
    out['col2'].name = 'Density'
    out['col3'].name = 'Temperature'
    out['Temperature'].unit = u.K
    out['col4'].name = 'H2_fraction_out'
    out['col5'].name = 'mp'
    out['col6'].name = 'tcp'
    out['col7'].name = 'tdp'

    t = hstack([input, math, out])
    
    return input, math, out, t


a = QTable()
for file in starlogfiles: 
    print('file: ', file) 
    input, math, out, t = prep_file(file) 
    print('stacked file: ', file) 
    a = vstack([a, t]) 

a.write('starlog.ascii', format='ascii') 
