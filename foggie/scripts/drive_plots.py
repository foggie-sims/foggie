""" attempt to multiprocess FOGGIE data on Pleaides""" 
from concurrent.futures import ProcessPoolExecutor
import numpy as np 
import yt
import argparse 
import trident
import matplotlib.pyplot as plt
import foggie.utils.get_region as gr 
from foggie.utils.consistency import colormap_dict, background_color_dict, proj_min_dict, proj_max_dict
import foggie.render.shade_maps as sm
import argparse
plt.rcParams["image.origin"] = 'lower'

import frame 

"""
This is a generic plot script driver.
Import the plotting routine you need above and run it within 'gs' 
"""
snapshot_dir='/nobackup/mpeeples/halo_008508/nref11c_nref9f/'

def gs(ds_name, axis, width, prefix, functions, region):
    print("gs ", ds_name, width, prefix, functions, region) 

    if ('flows' in functions): 
        print('calling flows from gs') 
        frame.flows(ds_name, width, prefix)
    if ('frame' in functions): 
        print('calling frame from gs') 
        frame.frame(ds_name,axis,width,prefix, region)
    if ('velocities' in functions): 
        print('calling velocities from gs') 
        frame.velocities(ds_name,axis,width,prefix, region)
    if ('disk' in functions): 
        print('calling disk from gs') 
        frame.disk(ds_name, axis, width, prefix) 
    if ('age' in functions): 
        print('calling age from gs') 
        frame.age(ds_name, width, prefix) 
    if ('lum' in functions): 
        print('calling lum from gs') 
        frame.lum(ds_name, axis, width, prefix) 
    if ('zfilter' in functions): 
        print('zfilter lum from gs') 
        frame.zfilter(ds_name,axis,width,prefix)

def script(dataset_list, axis, width, prefix, functions, region): 
    for ds in dataset_list: 
        print("script driving :", ds) 
        gs(ds, axis, width, prefix, functions, region) 

def chunks(l, n): 
   """Yield successive n-sized chunks from l.""" 
   for i in range(0, len(l), n): 
       yield l[i:i + n] 

def parse_args():
    parser = argparse.ArgumentParser(description="   ")

    parser.add_argument('--card', metavar='card', type=str, action='store',help='wildcards')
    parser.set_defaults(card='DD????/DD????')

    parser.add_argument('--axis', metavar='axis', type=str, action='store',help='axis to project on') 
    parser.set_defaults(axis='x') 

    parser.add_argument('--width', metavar='width', type=float, action='store',help='plot width in kpc')
    parser.set_defaults(width=0.2)

    parser.add_argument('--prefix', metavar='prefix', type=str, \
        action='store',help='filename prefix')
    parser.set_defaults(prefix='./') 

    parser.add_argument('--nthreads', metavar='nthreads', type=int, \
        action='store',help='number of multiprocessing threads')
    parser.set_defaults(nthreads=5) 

    parser.add_argument('--functions', metavar='functions', type=str, \
        action='store',help='comma-delimited string of analysis functions to apply')
    parser.set_defaults(functions='frame') 

    parser.add_argument('--region', metavar='region', type=str, \
        action='store',help='FOGGIE-defined region to use, e.g. cgm or ism')
    parser.set_defaults(region='cgm') 
    
    args = parser.parse_args()
    return args


parser = argparse.ArgumentParser()
parser.add_argument('--snap_number', type=int, required=True)
args = parser.parse_args()
print('Hello your snap_number is:', args.snap_number)

