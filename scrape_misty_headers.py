from __future__ import print_function

import numpy as np

from astropy.io import fits
from astropy.table import Table
from astropy.io import ascii

import matplotlib.pyplot as plt
import matplotlib as mpl

import os
import glob

def scrape_misty_headers():
    DATA_DIR = '.'
    dataset_list = glob.glob(os.path.join(DATA_DIR, 'hlsp*rd00*v6_lsf.fits.gz'))
    print('there are',len(dataset_list),'fits files')

    # line_list = ['H I 1216', 'H I 919', \
    # #                  'Al II 1671', 'Al III 1855', \
    #                  'Si II 1260', 'Si III 1207', 'Si IV 1394', \
    #                  'C II 1335', 'C III 977', 'C IV 1548', \
    #                  'O VI 1032', 'Ne VIII 770']
    line_list = ['H I 1216', 'H I 919', \
                    'Si II 1260', 'Si IV 1394', 'C IV 1548', 'O VI 1032']
    ### 'H I 1026', 'H I 973', 'H I 950' probably also exist but ignore for now

    # all_data = Table(names=('z','impact','HI_col','HI_1216_Nmin','HI_1216_EW','HI_1216_Ncomp','HI_919_Nmin','HI_919_Ncomp',\
    #                 'Si_II_col','Si_II_Nmin','Si_II_Ncomp','Si_II_EW','Si_II_dv90',\
    #                 'Si_III_col','Si_III_Nmin','Si_III_Ncomp','Si_III_EW','Si_III_dv90',\
    #                 'Si_IV_col','Si_IV_Nmin','Si_IV_Ncomp','Si_IV_EW','Si_IV_dv90',\
    #                 'C_II_col','C_II_Nmin','C_II_Ncomp','C_II_EW','C_II_dv90',\
    #                 'C_III_col','C_III_Nmin','C_III_Ncomp','C_III_EW','C_III_dv90',\
    #                 'C_IV_col','C_IV_Nmin','C_IV_Ncomp','C_IV_EW','C_IV_dv90',\
    #                 'O_VI_col','O_VI_Nmin','O_VI_Ncomp','O_VI_EW','O_VI_dv90'), \
    #          dtype=('f8','f8','f8','f8','f8','f8','f8','f8',
    #                 'f8','f8','f8','f8','f8',  # Si II
    #                 'f8','f8','f8','f8','f8',  # Si III
    #                 'f8','f8','f8','f8','f8',  # Si IV
    #                 'f8','f8','f8','f8','f8',  # C II
    #                 'f8','f8','f8','f8','f8',  # C III
    #                 'f8','f8','f8','f8','f8',  # C IV
    #                 'f8','f8','f8','f8','f8')) # O VI

    all_data = Table(names=('z','impact','HI_col','HI_1216_Nmin','HI_1216_EW','HI_1216_Ncomp','HI_919_Nmin','HI_919_Ncomp',\
                    'Si_II_col','Si_II_Nmin','Si_II_Ncomp','Si_II_EW','Si_II_dv90',\
                    'Si_IV_col','Si_IV_Nmin','Si_IV_Ncomp','Si_IV_EW','Si_IV_dv90',\
                    'C_IV_col','C_IV_Nmin','C_IV_Ncomp','C_IV_EW','C_IV_dv90',\
                    'O_VI_col','O_VI_Nmin','O_VI_Ncomp','O_VI_EW','O_VI_dv90'), \
             dtype=('f8','f8','f8','f8','f8','f8','f8','f8',
                    'f8','f8','f8','f8','f8',  # Si II
                    'f8','f8','f8','f8','f8',  # Si IV
                    'f8','f8','f8','f8','f8',  # C IV
                    'f8','f8','f8','f8','f8')) # O VI


    # for now, different tables for different ions
    si2_component_data = Table(names=('z','impact', 'losnum', 'tot_col', 'component', 'comp_col', 'comp_b', 'comp_dv'), \
                               dtype=('f8','f8', 'i8', 'f8', 'i8', 'f8', 'f8', 'f8'))
    si4_component_data = Table(names=('z','impact', 'losnum', 'tot_col', 'component', 'comp_col', 'comp_b', 'comp_dv'), \
                               dtype=('f8','f8', 'i8', 'f8', 'i8', 'f8', 'f8', 'f8'))
    c4_component_data = Table(names=('z','impact', 'losnum', 'tot_col', 'component', 'comp_col', 'comp_b', 'comp_dv'), \
                               dtype=('f8','f8', 'i8', 'f8', 'i8', 'f8', 'f8', 'f8'))
    o6_component_data = Table(names=('z','impact', 'losnum', 'tot_col', 'component', 'comp_col', 'comp_b', 'comp_dv'), \
                               dtype=('f8','f8', 'i8', 'f8', 'i8', 'f8', 'f8', 'f8'))

    ion_table_name_dict = {'Si II 1260' : si2_component_data, \
                           'Si IV 1394' : si4_component_data, \
                           'C IV 1548'  : c4_component_data, \
                           'O VI 1032'  : o6_component_data}


    i_file = 0
    for i_file, filename in enumerate(dataset_list):
        print("trying ",filename)
        with fits.open(filename) as f:
            ### some genius forgot to put the redshift in the header, so, hack:
            if 'rd0018' in filename:
                z = 2.5
            elif 'rd0020' in filename:
                z = 2.0
            else:
                z = -1
            row = [z,f[0].header['impact'], f['H I 1216'].header['tot_column'], f['H I 1216'].header['Nmin']]
            if 'totEW' in f['H I 1216'].header:
                row = np.append(row, [f['H I 1216'].header['totEW']], axis=0)
            else:
                row = np.append(row, [-1], axis=0)
            if 'Ncomp' in f['H I 1216'].header:
                row = np.append(row, [f['H I 1216'].header['Ncomp']], axis=0)
            else:
                row = np.append(row, [-1], axis=0)
            row = np.append(row, [f['H I 919'].header['Nmin']], axis=0)
            if 'Ncomp' in f['H I 919'].header:
                row = np.append(row, [f['H I 919'].header['Ncomp']], axis=0)
            else:
                row = np.append(row, [-1], axis=0)
            # for ion in ['Si II 1260', 'Si III 1207', 'Si IV 1394','C II 1335', 'C III 977', 'C IV 1548', 'O VI 1032']:
            for ion in ['Si II 1260', 'Si IV 1394','C IV 1548', 'O VI 1032']:
                if any([x.name.upper() == ion.upper() for x in f]):
                    row = np.append(row, [f[ion].header['tot_column'], f[ion].header['Nmin']], axis=0)
                    #print(f[ion].header)
                    if 'Ncomp' in f[ion].header:
                        row = np.append(row, [f[ion].header['Ncomp']], axis=0)
                        if ion in ion_table_name_dict.keys():
                            Ncomp = f[ion].header['NCOMP']
                            # if Ncomp > 0:
                            if 'FITVCEN0' in f[ion].header:
                                comp_row_start = [z, f[0].header['impact'], i_file, f[ion].header['tot_column']]
                                # comp_row = comp_row_start
                                for comp in range(Ncomp):
                                    bkey = 'fitb' + str(comp)
                                    colkey = 'fitcol' + str(comp)
                                    dvkey = 'fitvcen' + str(comp)
                                    comp_row = np.append(comp_row_start, [comp, f[ion].header[colkey], f[ion].header[bkey], f[ion].header[dvkey]], axis=0)
                                    # print('comp_row = ', comp_row)
                                    ion_table_name_dict[ion].add_row(comp_row)
                    else:
                        row = np.append(row, [-1], axis=0)
                    if 'totEW' in f[ion].header:
                        row = np.append(row, [f[ion].header['totEW']], axis=0)
                    else:
                        row = np.append(row, [-1], axis=0)
                    if 'totdv90' in f[ion].header:
                        row = np.append(row, [f[ion].header['totdv90']], axis=0)
                    else:
                        row = np.append(row, [-1], axis=0)
                else:
                    row = np.append(row, [-1], axis=0)
                    row = np.append(row, [-1], axis=0)
                    row = np.append(row, [-1], axis=0)
                    row = np.append(row, [-1], axis=0)
                    row = np.append(row, [-1], axis=0)
            all_data.add_row(row)

    # now let's save the table!
    ascii.write(all_data, 'misty_v5_rsp.dat', format='fixed_width', overwrite=True)
    ascii.write(si2_component_data, 'misty_si2_v5_rsp.dat', format='fixed_width', overwrite=True)
    ascii.write(si4_component_data, 'misty_si4_v5_rsp.dat', format='fixed_width', overwrite=True)
    ascii.write(c4_component_data, 'misty_c4_v5_rsp.dat', format='fixed_width', overwrite=True)
    ascii.write(o6_component_data, 'misty_o6_v5_rsp.dat', format='fixed_width', overwrite=True)


if __name__ == '__main__':
    scrape_misty_headers()
