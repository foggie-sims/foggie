##!/usr/bin/env python3

"""

    Title :      fit_mock_spectra
    Notes :      Takes a mock IFU datacube and fits the spectra along every pixel
    Output :     FITS cube with each emission line map as a 2D slice
    Author :     Ayan Acharyya
    Started :    March 2021
    Example :    run fit_mock_spectra.py --system ayan_local --halo 8508 --output RD0042 --mergeHII 0.04 --base_spatial_res 0.4 --z 0.25 --obs_wave_range 0.8,0.85 --obs_spatial_res 1 --obs_spec_res 60 --exptime 1200 --snr 5 --debug

"""
from header import *
from util import *
from make_mock_datacube import wrap_get_mock_datacube

