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
plt.rcParams["image.origin"] = 'lower'

import frame 

"""
This is a generic plot script driver.
Import the plotting routine you need above and run it within 'gs' 
"""
snapshot_dir='/nobackup/mpeeples/halo_008508/nref11c_nref9f/'

def gs(ds_name, axis, width, prefix, functions):
    print("gs ", ds_name, width, prefix, functions) 


    if ('flows' in functions): frame.flows(ds_name, width, prefix)
    if ('frame' in functions): frame.frame(ds_name,axis,width,prefix)
    if ('velocities' in functions): frame.velocities(ds_name,axis,width,prefix)
    if ('disk' in functions): frame.disk(ds_name, axis, width, prefix) 
    if ('age' in functions): frame.age(ds_name, width, prefix) 
    if ('lum' in functions): frame.lum(ds_name, axis, width, prefix) 
    if ('zfilter' in functions): frame.zfilter(ds_name,axis,width,prefix)


def script(dataset_list, axis, width, prefix): 
    for ds in dataset_list: 
        print("script driving :", ds) 
        gs(ds, axis, width, prefix) 

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

    parser.add_argument('--prefix', metavar='prefix', type=str, action='store',help='filename prefix')
    parser.set_defaults(prefix='./') 

    parser.add_argument('--nthreads', metavar='nthreads', type=int, action='store',help='number of multiprocessing threads')
    parser.set_defaults(prefix='5') 

    parser.add_argument('--functions', metavar='functions', type=str, action='store',help='comma-delimited string of analysis functions to apply')
    parser.set_defaults(functions='frame') 

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    print('  wildcard = ', args.card)
    print('      axis = ', args.axis)
    print('     width = ', args.width)
    print('    prefix = ', args.prefix)
    print('  nthreads = ', args.nthreads)
    print(' functions = ', args.functions)

    ts = yt.load(snapshot_dir+args.card) 
    ts.outputs.reverse() # work backwards in time
    print("the snapshots are: ", ts.outputs)
    print("there are N = ", len(ts.outputs), " outputs ") 

    print("We will be using Nthreads = ", args.nthreads, " processing threads") 
    print("We will apply the plotting functions: ", args.functions)

    chunksize = int(np.ceil(len(ts.outputs) / args.nthreads)) 
    dslist = list(chunks(ts.outputs, chunksize))
 
    with ProcessPoolExecutor(args.nthreads) as executor:
        # these return immediately and are executed in parallel on separate processes
        for index in np.arange(args.nthreads): 
            print(index) 
            _ = executor.submit(script, dslist[index], args.axis, args.width, args.prefix, args.functions)      

