import numpy as np
import sys
import os
import matplotlib as mpl
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})
mpl.rcParams['font.family'] = 'serif'
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import AxesGrid
import matplotlib.image as mpimg
from yt.units import kpc
from astropy.table import Table
from astropy.io import fits
import pickle
from functools import partial
import datashader as dshader
import datashader.transfer_functions as tf
from datashader import reductions
from datashader.utils import export_image
import pandas as pd
import trident
from foggie.utils.get_halo_center import get_halo_center
from foggie.utils.get_refine_box import get_refine_box
from foggie.utils.get_proper_box_size import get_proper_box_size
from foggie.utils.consistency import *
from foggie.utils.enzoGalaxyProps import find_rvirial
import foggie.utils.get_halo_info as ghi
import yt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator)
import csv
from collections import OrderedDict as odict
mpl.rcParams['axes.linewidth']=1
mpl.rcParams['axes.edgecolor']='k'
from astropy.io import fits

instrument = 'COS-G130M'
startwl = 1132
endwl = 1433
wlres = 0.1
line_list = ['H', 'C', 'N', 'O', 'Mg']
halo = '008508'
sim = 'nref11n_nref10f'
snap = 'RD0040'
fn = '/astro/simulations/FOGGIE/halo_'+halo+'/'+sim+'/'+snap+'/'+snap
track_name = "/Users/raugustin/foggie/foggie/halo_tracks/008508/nref11n_selfshield_15/halo_track_200kpc_nref10"
output_dir = "/Users/raugustin/WORK/mockobservations/"
os.chdir(output_dir)

def prep(output_dir,fn,track_name):
    fullds = yt.load(fn)
    track = Table.read(track_name, format='ascii')
    track.sort('col1')
    zsnap = fullds.get_parameter('CosmologyCurrentRedshift')
    proper_box_size = get_proper_box_size(fullds)
    ds, boxcenter, width = get_refine_box(fullds, zsnap, track)
    properwidth = width * proper_box_size
    smallestcell = 2.744e-04 * 1000 # in kpc for RD 00 42
    center, halo_velocity = get_halo_center(fullds, boxcenter)
    xc, yc, zc = center
    leftedge = ds.left_edge
    rightedge = ds.right_edge
    xl, yl, zl = leftedge
    xr, yr, zr = rightedge
    stepsize = smallestcell / proper_box_size
    steps= int(width/stepsize)
    stepsize = stepsize *50
    steps = 3
    return fullds,steps,stepsize,center,leftedge,rightedge,zsnap,width



def make_spectra(steps,stepsize,center,leftedge,rightedge,instrument,line_list,width,px,py,pz):
    xc, yc, zc = center
    xl, yl, zl = leftedge
    xr, yr, zr = rightedge
    countdown = steps*steps
    grid = np.zeros((steps,steps,2))
    for i in range(1):
        print(i)
        for j in range(steps):
            countdown -= 1
            print(countdown)
            startx = xc - 1. * stepsize
            starty = yc - 1. * stepsize
            startz = zc - 1. * stepsize
            ray_start = [startx + i * stepsize, yc, startz + j * stepsize]
            ray_end = [startx + i * stepsize, yr, startz + j * stepsize]
            print(ray_start)
            print(ray_end)
            ray = trident.make_simple_ray(fullds,start_position=ray_start,end_position=ray_end,data_filename="ray.h5",lines=line_list,ftype='gas',redshift=0.00)
            px.annotate_ray(ray, arrow=True)
            py.annotate_ray(ray, arrow=True)
            pz.annotate_ray(ray, arrow=True)
            sgi = trident.SpectrumGenerator(instrument)
            sgi.make_spectrum(ray, lines=line_list)
            sgi.save_spectrum(str(i)+'_'+str(j)+'_'+'spec_rawi.txt')
            sgi.apply_lsf()
            sgi.save_spectrum(str(i)+'_'+str(j)+'_'+'spec_finali.txt')
            sg = trident.SpectrumGenerator(instrument)
            sg.make_spectrum(ray, lines=line_list)
            sg.save_spectrum(str(i)+'_'+str(j)+'_'+'spec_raw.txt')
            sg.add_milky_way_foreground()
            sg.apply_lsf()
            sg.add_gaussian_noise(30)
            sg.save_spectrum(str(i)+'_'+str(j)+'_'+'spec_final.txt')
            trident.plot_spectrum([sg.lambda_field, sgi.lambda_field],[sg.flux_field, sgi.flux_field],lambda_limits=[1306,1310], stagger=0, step=[False, True],label=['Observed','Ideal'], filename=str(i)+'_'+str(j)+'_'+'ideal_and_obs_z1p4_moved_to_z0.png')
        i += 1

fullds,steps,stepsize,center,leftedge,rightedge,zsnap,width = prep(output_dir,fn,track_name)
px = yt.ProjectionPlot(fullds, 'x', 'density', center=center, width=width)
py = yt.ProjectionPlot(fullds, 'y', 'density', center=center, width=width)
pz = yt.ProjectionPlot(fullds, 'z', 'density', center=center, width=width)
make_spectra(steps,stepsize,center,leftedge,rightedge,instrument,line_list,width,px,py,pz)
px.save(halo+'_'+sim+'_'+snap+'_'+'projection_x_annotated.png')
py.save(halo+'_'+sim+'_'+snap+'_'+'projection_y_annotated.png')
pz.save(halo+'_'+sim+'_'+snap+'_'+'projection_z_annotated.png')
