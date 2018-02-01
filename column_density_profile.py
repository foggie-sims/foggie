import numpy as np
from astropy.io import fits
from astropy.table import Table

import matplotlib.pyplot as plt

import os
import glob

def get_column_densities():
    DATA_DIR = "./spectra"
    dataset_list = glob.glob(os.path.join(DATA_DIR, 'hlsp*.fits'))

    columns = Table(names=("impact","line","fitcol","totcol"),dtype=("f8","S10","f8","f8"))

    for filename in dataset_list:
        with fits.open(filename) as f:
            line = f[0].header['LINE_1']
            # print f[0].header["impact"], ": ", f[line].header["fitcol0"], f[line].header["tot_column"]
            columns.add_row([f[0].header["impact"], line, f[line].header["fitcol0"], f[line].header["tot_column"]])

    return columns.sort("impact")

def plot_column_densities():
    columns = get_column_densities()
