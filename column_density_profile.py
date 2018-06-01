import numpy as np
from astropy.io import fits
from astropy.table import Table

import matplotlib.pyplot as plt

import os
import glob

from manage_path_names import get_path_names

def rerun_spectacle(DATA_DIR):
    dataset_list = glob.glob(os.path.join(DATA_DIR, 'hlsp*v3_los.fits'))
    for filename in dataset_list:
        add_spectacle_to_fits(filename,filename)


def get_column_densities():
    output_dir = get_path_names()
    DATA_DIR = "./spectra"
    dataset_list = glob.glob(os.path.join(DATA_DIR, 'hlsp*v3_los.fits'))

    columns = Table(names=("impact","line","regdv90","totew","totcol"),dtype=("f8","S10","f8","f8","f8"))

    for filename in dataset_list:
        with fits.open(filename) as f:
            line = f[0].header['LINE_1']
            # print f[0].header["impact"], ": ", f[line].header["fitcol0"], f[line].header["tot_column"]
            if 'regdv900' in f[line].header.keys():
                columns.add_row([f[0].header["impact"], line, f[line].header["REGDV900"],f[line].header["TOTEW"], f[line].header["tot_column"]])

    return columns.sort("impact")

def plot_column_densities():

    columns = get_column_densities()
