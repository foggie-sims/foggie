"""
Filename: get_root_track_parallel.py
Authors: JT, Cassi
Made: 11/6/25
Last modified: 111/6/25
This program produces an ascii file that can be used as a track file for running forced refinement
box simulations.

NOTE that this script requires halo_00XXXX_root_index.txt files to have been created first!

Process:
1. Run a "natural" refinement simulation.
2. Run this script on all outputs of the natural simulation.
3. Copy the resulting text file into the run directory of the forced refinement
   simulation, and run it with this text file as the track box.
"""

import yt
from yt.units import *
from yt import YTArray
from astropy.table import Table
from astropy.io import ascii
import multiprocessing as multi
import argparse

from foggie.utils.analysis_utils import *
import numpy as np
import os

import warnings

def parse_args():
    '''Parse command line arguments. Returns args object.
    NOTE: Need to move command-line argument parsing to separate file.'''

    parser = argparse.ArgumentParser(description='Calculates and saves to file a bunch of fluxes.')

    # Optional arguments:
    parser.add_argument('--halo', metavar='halo', type=str, action='store', \
                        help='Which halo? Default is 8508 (Tempest)')
    parser.set_defaults(halo='8508')

    parser.add_argument('--run_dir', metavar='run_dir', type=str, action='store', \
                        help='Give the full path to the directory of the sim outputs you want to run on.')
    parser.set_defaults(run_dir=None)

    parser.add_argument('--code_path', metavar='code_path', type=str, action='store', \
                        help='Give the full path to the directory where the foggie repo is stored.')
    parser.set_defaults(code_path=None)

    parser.add_argument('--output', metavar='output', type=str, action='store', \
                        help='Which output(s)? Options: Specify a single output or specify a range of outputs ' + \
                        'using commas to list individual outputs and dashes for ranges of outputs ' + \
                        '(e.g. "RD0020-RD0025" or "DD1341,DD1353,DD1600-DD1700", no spaces!)\n' + \
                        'If you do not specify outputs, it will run on every output in run_dir.')
    parser.set_defaults(output=None)

    parser.add_argument('--ref_level', metavar='ref_level', type=int, action='store', \
                        help='What forced refinement level do you want the track box to be?\n' + \
                        'Default is 9.')
    parser.set_defaults(ref_level=9)

    parser.add_argument('--box_size', metavar='box_size', type=float, action='store', \
                        help='What box size do you want for the track box? Specify in CODE UNITS.\n' + \
                        'Default is 0.002, which is the same size as FOGGIE 1.0 (288 comoving kpc).')
    parser.set_defaults(box_size=0.002)

    parser.add_argument('--nproc', metavar='nproc', type=int, action='store', \
                        help='How many processes do you want? Default is 1, ' + \
                        'code will run one output per processor')
    parser.set_defaults(nproc=1)


    args = parser.parse_args()
    return args

def get_halo_root_track(run_path, halo_id, snap_number, box_size, ref_level, t):
    '''This function calculates the center of mass of the particles listed
    in halo_[halo_id]_root_index.txt in the snapshot given by [snap_number] and
    adds a row to the track file with the corners of the box given by [box_size]
    centered on this center.'''

    print(snap_number)
        
    #path = code_path + '/halo_tracks/' + halo_id + '/root_tracks/'
    #root_particles = Table.read(path + 'halo_' + halo_id + '_root_index.txt', format='ascii')
    root_particles = Table.read(run_path + '/halo_catalogs/root_index.txt', format='ascii')
    halo0 = root_particles['root_index']
        
    ds = yt.load(snap_number) 
    ad = ds.all_data()
        
    x = ad['particle_position_x']
    y = ad['particle_position_y']
    z = ad['particle_position_z']
        
    root_indices = halo0
    now_indices = ad['particle_index']
    indices = np.where(np.isin(now_indices, root_indices))[0]
        
    center_x  = float(np.mean(x[indices].in_units('code_length'))) 
    center_y  = float(np.mean(y[indices].in_units('code_length'))) 
    center_z  = float(np.mean(z[indices].in_units('code_length'))) 
        
    center1 = [center_x, center_y, center_z]
        
    row = [ds.current_redshift, center1[0]-box_size/2., center1[1]-box_size/2., 
                                center1[2]-box_size/2., center1[0]+box_size/2., 
                                center1[1]+box_size/2., center1[2]+box_size/2., ref_level]
    t.put(row)

if __name__ == "__main__":

    args = parse_args()

    if (args.run_dir==None):
        exit('You must specify a run directory.')

    if (args.code_path==None):
        exit('You must specify the path to the foggie repo.')

    halo_id = '00' + args.halo
    run_dir = args.run_dir
    code_path = args.code_path
    box_size = args.box_size
    ref_level = args.ref_level

    print('Running in %s' % (args.run_dir))

    if (args.output==None):
        print('All outputs')
        outs = [name for name in os.listdir(args.run_dir)
          if os.path.isdir(os.path.join(args.run_dir, name)) and (name.startswith('DD') or name.startswith('RD'))]
    else:
        print('Outputs: %s' % (args.output))
        outs = make_output_list(args.output)


    trackname = args.run_dir + '/root_track'
    if os.path.isfile(trackname):
        print("Opening track: " + trackname)
        track = Table.read(trackname, format='ascii')
        track.columns['col1'].name = 'redshift'
    else:
        track = Table(dtype=('f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'i4'),
            names=('redshift', 'left_x', 'left_y', 'left_z', 'right_x', 'right_y', 'right_z', 'ref_level'))

    print('Computing root track for ' + str(len(outs)) + ' snaps ' + \
          'from ' + outs[0] + ' to ' + outs[-1])
    # Split into a number of groupings equal to the number of processors
    # and run one process per processor
    rows = []
    for i in range(len(outs)//args.nproc):
        threads = []
        queue = multi.Queue()
        for j in range(args.nproc):
            snap = args.run_dir + '/' + outs[args.nproc*i+j] + '/' + outs[args.nproc*i+j]
            thr = multi.Process(target=get_halo_root_track, \
               args=(run_dir, halo_id, snap, box_size, ref_level, queue))
            threads.append(thr)
            thr.start()
        for thr in threads:
            row = queue.get()
            rows.append(row)
        for thr in threads:
            thr.join()
    # For any leftover snapshots, run one per processor
    threads = []
    queue = multi.Queue()
    for j in range(len(outs)%args.nproc):
        snap = args.run_dir + '/' + outs[-(j+1)] + '/' + outs[-(j+1)]
        thr = multi.Process(target=get_halo_root_track, \
               args=(run_dir, halo_id, snap, box_size, ref_level, queue))
        threads.append(thr)
        thr.start()
    for thr in threads:
        row = queue.get()
        rows.append(row)
    for thr in threads:
        thr.join()

    for row in rows:
        track.add_row(row)

    track.sort('redshift')
    track.reverse()
    track.write(trackname, format='ascii', overwrite=True)

    # Remove header line
    with open(trackname) as f:
        lines = f.readlines()
    with open(trackname, "w") as f:
        f.writelines(lines[1:])
