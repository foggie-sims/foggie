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

def gs(ds_name):
    print("gs ", ds_name) 
    #frame.flows(ds_name) 
    frame.frame(ds_name) 
    #frame.velocities(ds_name) 
    #frame.disk(ds_name) 
    #frame.age(ds_name) 

def script(dataset_list): 
    for ds in dataset_list: 
        print("script driving :", ds) 
        gs(ds) 

def chunks(l, n): 
   """Yield successive n-sized chunks from l.""" 
   for i in range(0, len(l), n): 
       yield l[i:i + n] 

def parse_args():
    parser = argparse.ArgumentParser(description="   ")
    parser.add_argument('--card', metavar='card', type=str, action='store',
                        help='wildcards')
    parser.set_defaults(card='DD????/DD????')
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    print('wildcard = ', args.card)

    ts = yt.load(snapshot_dir+args.card) 
    ts.outputs.reverse() # work backwards in time
    print("the snapshots are: ", ts.outputs)
    print("there are N = ", len(ts.outputs), " outputs ") 

    chunksize = int(np.ceil(len(ts.outputs) / 20.)) 
    dslist = list(chunks(ts.outputs, chunksize))

    with ProcessPoolExecutor(20) as executor:
        # these return immediately and are executed in parallel on separate processes
        future_1 =  executor.submit(script, dslist[0])      
        future_2 =  executor.submit(script, dslist[1])      
        future_3 =  executor.submit(script, dslist[2])      
        future_4 =  executor.submit(script, dslist[3])      
        future_5 =  executor.submit(script, dslist[4])      
        future_6 =  executor.submit(script, dslist[5])      
        future_7 =  executor.submit(script, dslist[6])      
        future_8 =  executor.submit(script, dslist[7])      
        future_9 =  executor.submit(script, dslist[8])      
        future_10 = executor.submit(script, dslist[9])      
        future_11 = executor.submit(script, dslist[10])      
        future_12 = executor.submit(script, dslist[11])      
        future_13 = executor.submit(script, dslist[12])      
        future_14 = executor.submit(script, dslist[13])      
        future_15 = executor.submit(script, dslist[14])      
        future_16 = executor.submit(script, dslist[15])      
        future_17 = executor.submit(script, dslist[16])      
        future_18 = executor.submit(script, dslist[17])      
        future_19 = executor.submit(script, dslist[18])      
        future_20 = executor.submit(script, dslist[19])      

